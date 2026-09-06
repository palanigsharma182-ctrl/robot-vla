"""Qwen VLA 在严格桌面放置环境中的逐控制步闭环评估。"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from typing_extensions import Self

from robot_vla.adapters import (
    FingerForceNormalizer,
    FrankaObservationAdapter,
    ProprioNormalizer,
)
from robot_vla.contracts import PICK_AND_PLACE_SKILLS, RobotSpec
from robot_vla.evaluation.atomic import AtomicSkillEpisodeResult, derive_atomic_sampling_seed
from robot_vla.evaluation.rollout import (
    RolloutEpisodeResult,
    RolloutEpisodeSpec,
    classify_rollout_failure,
)
from robot_vla.execution import (
    ChunkInferenceStrategy,
    ManiSkillFrankaController,
    RecedingHorizonChunkExecutor,
    RTCConfig,
)
from robot_vla.model.policy import QwenVLAObservationV2Policy, QwenVLAPolicy
from robot_vla.model.qwen_processor import QwenVLAProcessorAdapter
from robot_vla.observation import (
    OBSERVATION_MODALITIES,
    ObservationV2Frame,
    ObservationV2History,
    ObservationV2Window,
    invert_se3,
    opengl_camera_to_opencv,
    validate_se3,
)
from robot_vla.runtime import (
    OnlineObservation,
    QwenVLAObservationV2Runtime,
    QwenVLAReplanLoop,
    QwenVLARuntime,
    RuntimeConfig,
)
from robot_vla.sim import PICK_CUBE_TO_REGION_ENV_ID, register_robot_vla_maniskill_envs
from robot_vla.tasks.pick_place import (
    PickPlaceState,
    PickPlaceTaskProgress,
    PickPlaceTaskTracker,
)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _single_bool(value: Any) -> bool:
    array = _numpy(value)
    if array.size != 1:
        raise ValueError(f"首版 Rollout 只支持单环境 bool，实际 shape={array.shape}")
    return bool(array.reshape(-1)[0])


def derive_episode_sampling_seed(base_seed: int, episode: RolloutEpisodeSpec) -> int:
    if base_seed < 0:
        raise ValueError("base_seed 不能为负数")
    identity = f"{base_seed}:{episode.seed_group}:{episode.seed}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % (2**63 - 1)


def _read_predicate_state(base_env: Any) -> PickPlaceState:
    tcp = _numpy(base_env.agent.tcp_pose.p)[0]
    cube = _numpy(base_env.cube.pose.p)[0]
    goal = _numpy(base_env.goal_site.pose.p)[0]
    linear_velocity = _numpy(base_env.cube.linear_velocity)[0]
    angular_velocity = _numpy(base_env.cube.angular_velocity)[0]
    is_grasped = _single_bool(base_env.agent.is_grasping(base_env.cube))
    return PickPlaceState(
        tcp_position=tuple(float(value) for value in tcp),
        object_position=tuple(float(value) for value in cube),
        goal_position=tuple(float(value) for value in goal),
        object_linear_velocity=tuple(float(value) for value in linear_velocity),
        object_angular_velocity=tuple(float(value) for value in angular_velocity),
        support_center_z_m=float(base_env.cube_half_size),
        is_grasped=is_grasped,
    )


def _read_online_observation(
    observation: dict[str, Any],
    base_env: Any,
    observation_adapter: FrankaObservationAdapter,
    instruction: str,
) -> OnlineObservation:
    sensor_data = observation["sensor_data"]
    external = _numpy(sensor_data["base_camera"]["rgb"])[0]
    wrist = _numpy(sensor_data["hand_camera"]["rgb"])[0]
    if external.dtype != np.uint8 or wrist.dtype != np.uint8:
        raise ValueError("ManiSkill 在线 RGB 必须是 uint8")
    robot = base_env.agent.robot
    qpos = _numpy(robot.get_qpos())
    qvel = _numpy(robot.get_qvel())
    joint_names = tuple(joint.name for joint in robot.active_joints)
    if qpos.shape[0] != 1 or qvel.shape != qpos.shape:
        raise ValueError("首版 Rollout 只支持 num_envs=1")
    proprio = observation_adapter.from_maniskill(qpos[0], qvel[0], joint_names)
    return OnlineObservation(
        rgb_external=external.copy(),
        rgb_wrist=wrist.copy(),
        physical_proprio=proprio.copy(),
        instruction=instruction,
    )


def _single_transform_matrix(pose: Any, name: str) -> np.ndarray:
    value = _numpy(pose.to_transformation_matrix())
    if value.shape == (1, 4, 4):
        value = value[0]
    return validate_se3(value, name)


def _read_observation_v2_frame(
    observation: dict[str, Any],
    base_env: Any,
    observation_adapter: FrankaObservationAdapter,
    spec: RobotSpec,
    *,
    control_step: int,
) -> ObservationV2Frame:
    """仿真 adapter：所有 modality 在同一 Simulator Tick 取样。"""

    current = _read_online_observation(
        observation,
        base_env,
        observation_adapter,
        instruction="observation-v2-frame",
    )
    world_from_base = _single_transform_matrix(
        base_env.agent.robot.pose,
        "world_from_robot_base",
    )
    world_from_tcp = _single_transform_matrix(base_env.agent.tcp_pose, "world_from_tcp")
    wrist_gl = _numpy(
        observation["sensor_param"]["hand_camera"]["cam2world_gl"]
    )
    if wrist_gl.shape == (1, 4, 4):
        wrist_gl = wrist_gl[0]
    base_from_world = invert_se3(world_from_base, "world_from_robot_base")
    base_from_tcp = validate_se3(base_from_world @ world_from_tcp, "base_from_tcp")
    base_from_wrist = validate_se3(
        base_from_world @ opengl_camera_to_opencv(wrist_gl),
        "base_from_wrist_camera_cv",
    )
    scene = base_env.scene
    left = _numpy(
        scene.get_pairwise_contact_forces(base_env.agent.finger1_link, base_env.cube)
    )[0]
    right = _numpy(
        scene.get_pairwise_contact_forces(base_env.agent.finger2_link, base_env.cube)
    )[0]
    # ManiSkill 没有真实应变片；这是 target-pairwise force-magnitude sensor approximation。
    finger_force = np.asarray(
        (float(np.linalg.norm(left)), float(np.linalg.norm(right))),
        dtype=np.float32,
    )
    timestamp = control_step / spec.control_hz
    return ObservationV2Frame(
        rgb_external=current.rgb_external,
        rgb_wrist=current.rgb_wrist,
        physical_proprio=current.physical_proprio,
        base_from_tcp=base_from_tcp.astype(np.float32),
        base_from_wrist_camera=base_from_wrist.astype(np.float32),
        finger_force_n=finger_force,
        timestamp_s=timestamp,
        modality_timestamp_s=np.full(
            len(OBSERVATION_MODALITIES),
            timestamp,
            dtype=np.float64,
        ),
        modality_valid=np.ones(len(OBSERVATION_MODALITIES), dtype=np.bool_),
    )


def _reset_atomic_time_limit(env: Any) -> None:
    """只重置 Gymnasium TimeLimit 计数，不修改仿真物理状态。"""

    current = env
    reset = False
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        if "_elapsed_steps" in vars(current):
            elapsed_steps = vars(current)["_elapsed_steps"]
            if elapsed_steps is None or isinstance(elapsed_steps, int):
                current._elapsed_steps = 0
                reset = True
            elif hasattr(elapsed_steps, "zero_"):
                elapsed_steps.zero_()
                reset = True
        if not hasattr(current, "env"):
            break
        current = current.env
    if not reset:
        raise RuntimeError("原子评估未找到 Gymnasium TimeLimit wrapper")


class _TrackingManiSkillController(ManiSkillFrankaController):
    """执行每个控制步后立即更新项目 Predicate，不以 Replan 近似控制步。"""

    def __init__(
        self,
        env: Any,
        spec: RobotSpec,
        observation: dict[str, Any],
        tracker: PickPlaceTaskTracker,
        progress: PickPlaceTaskProgress,
        observation_adapter: FrankaObservationAdapter,
        *,
        observation_v2_enabled: bool = False,
    ) -> None:
        super().__init__(env, spec)
        self.observation = observation
        self.tracker = tracker
        self.progress = progress
        self.observation_adapter = observation_adapter
        self.environment_steps = 0
        self.environment_success = False
        self.terminated = False
        self.truncated = False
        initial_state = _read_predicate_state(env.unwrapped)
        self._last_tcp_position = np.asarray(initial_state.tcp_position, dtype=np.float64)
        self.last_tcp_linear_speed_m_s = 0.0
        self.skill_completion_environment_steps: list[int | None] = [
            0 if index < progress.completed_skill_count else None
            for index in range(len(PICK_AND_PLACE_SKILLS))
        ]
        self.observation_v2_history = (
            ObservationV2History(spec) if observation_v2_enabled else None
        )
        if self.observation_v2_history is not None:
            self.observation_v2_history.append(
                _read_observation_v2_frame(
                    observation,
                    env.unwrapped,
                    observation_adapter,
                    spec,
                    control_step=0,
                )
            )

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated or (
            self.environment_success and self.progress.task_completed
        )

    def send_action(self, controller_action: np.ndarray) -> None:
        # 当前 Chunk 在成功/截断的最后一步后可能还有未消费前缀；不得再次 step 已结束环境。
        if self.done:
            return
        super().send_action(controller_action)
        observation, _, terminated, truncated, info = self.last_step_output
        self.observation = observation
        self.environment_steps += 1
        if self.observation_v2_history is not None:
            self.observation_v2_history.append(
                _read_observation_v2_frame(
                    observation,
                    self.env.unwrapped,
                    self.observation_adapter,
                    self.spec,
                    control_step=self.environment_steps,
                )
            )
        previous_completed = self.progress.completed_skill_count
        predicate_state = _read_predicate_state(self.env.unwrapped)
        tcp_position = np.asarray(predicate_state.tcp_position, dtype=np.float64)
        self.last_tcp_linear_speed_m_s = float(
            np.linalg.norm(tcp_position - self._last_tcp_position) * self.spec.control_hz
        )
        self._last_tcp_position = tcp_position
        self.progress = self.tracker.update(predicate_state)
        for skill_index in range(previous_completed, self.progress.completed_skill_count):
            if self.skill_completion_environment_steps[skill_index] is None:
                self.skill_completion_environment_steps[skill_index] = self.environment_steps
        self.environment_success = self.environment_success or _single_bool(info["success"])
        self.terminated = self.terminated or _single_bool(terminated)
        self.truncated = self.truncated or _single_bool(truncated)


def run_maniskill_episode(
    env: Any,
    runtime: QwenVLARuntime,
    spec: RobotSpec,
    episode: RolloutEpisodeSpec,
    *,
    sampling_seed_base: int,
    inference_strategy: str | ChunkInferenceStrategy | None = None,
    temporal_ensemble_enabled: bool | None = None,
    recency_decay: float = 0.5,
    rtc_config: RTCConfig | None = None,
    max_anomaly_replans: int = 3,
) -> RolloutEpisodeResult:
    started = time.monotonic()
    observation, _ = env.reset(seed=episode.seed)
    tracker = PickPlaceTaskTracker()
    progress = tracker.update(_read_predicate_state(env.unwrapped))
    observation_adapter = FrankaObservationAdapter(spec)
    controller = _TrackingManiSkillController(
        env,
        spec,
        observation,
        tracker,
        progress,
        observation_adapter,
        observation_v2_enabled=isinstance(runtime, QwenVLAObservationV2Runtime),
    )
    loop = QwenVLAReplanLoop(
        runtime,
        RecedingHorizonChunkExecutor(spec),
        inference_strategy=inference_strategy,
        temporal_ensemble_enabled=temporal_ensemble_enabled,
        recency_decay=recency_decay,
        rtc_config=rtc_config,
        max_anomaly_replans=max_anomaly_replans,
    )
    # Anomaly 可能在执行满 4 步前触发安全重规划，最坏按每控制步一次 Replan 预留。
    max_replans = int(env._max_episode_steps) + 1
    replans = 0
    sampling_seeds: list[int] = []
    normalized_action_abs_max: float | None = None
    physical_arm_delta_abs_max_rad: float | None = None
    gripper_target_min: float | None = None
    gripper_target_max: float | None = None
    action_chunks = 0
    failure_stage: str | None = None
    error: str | None = None
    execution_diagnostic: dict[str, Any] | None = None
    tracking_correction_saturation_count = 0
    tracking_correction_requested_abs_max_rad: float | None = None
    tracking_correction_applied_abs_max_rad: float | None = None
    anomaly_replan_count = 0
    temporal_ensemble_max_buffer_size = 0
    temporal_ensemble_max_proposal_spread = 0.0
    temporal_ensemble_min_newest_weight: float | None = None
    replan_traces: list[dict[str, Any]] = []

    while not controller.done and replans < max_replans:
        replans += 1
        try:
            if controller.observation_v2_history is None:
                online_observation = _read_online_observation(
                    controller.observation,
                    env.unwrapped,
                    observation_adapter,
                    episode.instruction,
                )
            else:
                command_reference = loop.executor.previous_command_q
                if command_reference is None:
                    command_reference = controller.read_state().joint_positions
                online_observation = controller.observation_v2_history.snapshot(
                    episode.instruction,
                    previous_command_q=command_reference,
                    previous_action=loop.executor.previous_action,
                )
        except Exception as observation_error:  # noqa: BLE001 - 必须形成可审计 Episode
            failure_stage = "rollout"
            error = f"{type(observation_error).__name__}: {observation_error}"
            try:
                controller.hold_current()
            except Exception as hold_error:  # noqa: BLE001 - 同时保留 hold 失败证据
                error += f"; hold 失败: {type(hold_error).__name__}: {hold_error}"
            break

        replan_control_step = loop.control_step
        completed_before = controller.progress.completed_skill_count
        tcp_distance_before = controller.progress.outcome.tcp_to_object_distance_m
        tcp_speed_before = controller.last_tcp_linear_speed_m_s
        # 诊断使用当前物理状态；V2 的完整四帧历史仍原样传给 runtime。
        current_proprio = (
            online_observation.physical_proprio[-1]
            if isinstance(online_observation, ObservationV2Window)
            else online_observation.physical_proprio
        )
        joint_velocity_abs_max = float(
            np.max(np.abs(current_proprio[spec.arm_dof : spec.arm_dof * 2]))
        )
        result = loop.replan_and_execute(online_observation, controller)
        execution = result.execution
        replan_trace: dict[str, Any] = {
            "replan_index": replans - 1,
            "control_step": replan_control_step,
            "sampling_seed": None if result.sampling is None else result.sampling.seed,
            "inference_strategy": result.inference_strategy.value,
            "rtc_enabled": result.inference_strategy == ChunkInferenceStrategy.RTC,
            "rtc_guidance_weight": None,
            "rtc_execution_horizon": None,
            "rtc_schedule": None,
            "previous_chunk_available": False,
            "overlap_length": 0,
            "completed_skill_count_before": completed_before,
            "completed_skill_count_after": controller.progress.completed_skill_count,
            "tcp_to_object_distance_before_m": tcp_distance_before,
            "tcp_to_object_distance_after_m": controller.progress.outcome.tcp_to_object_distance_m,
            "tcp_linear_speed_before_m_s": tcp_speed_before,
            "tcp_linear_speed_after_m_s": controller.last_tcp_linear_speed_m_s,
            "joint_velocity_abs_max_rad_s": joint_velocity_abs_max,
            "gripper_target_first": None,
            "temporal_proposal_spread": None,
        }
        anomaly_replan_count += int(execution.replan_required)
        if result.ensemble_trace is not None:
            replan_trace["temporal_proposal_spread"] = (
                result.ensemble_trace.max_proposal_spread
            )
            temporal_ensemble_max_buffer_size = max(
                temporal_ensemble_max_buffer_size,
                result.ensemble_trace.buffer_size,
            )
            temporal_ensemble_max_proposal_spread = max(
                temporal_ensemble_max_proposal_spread,
                result.ensemble_trace.max_proposal_spread,
            )
            newest_weight = min(
                result.ensemble_trace.newest_normalized_weights[: spec.execute_steps]
            )
            temporal_ensemble_min_newest_weight = (
                newest_weight
                if temporal_ensemble_min_newest_weight is None
                else min(temporal_ensemble_min_newest_weight, newest_weight)
            )
        tracking_correction_saturation_count += execution.correction_saturation_steps
        if execution.requested_correction_abs_max_rad is not None:
            tracking_correction_requested_abs_max_rad = max(
                tracking_correction_requested_abs_max_rad or 0.0,
                execution.requested_correction_abs_max_rad,
            )
        if execution.applied_correction_abs_max_rad is not None:
            tracking_correction_applied_abs_max_rad = max(
                tracking_correction_applied_abs_max_rad or 0.0,
                execution.applied_correction_abs_max_rad,
            )
        if result.sampling is not None:
            sampling_seeds.append(result.sampling.seed)
        if result.action_chunk is not None:
            replan_trace["gripper_target_first"] = float(
                result.action_chunk.physical_action[0, -1]
            )
            action_chunks += 1
            normalized_max = float(np.max(np.abs(result.action_chunk.normalized_action)))
            physical_arm_max = float(
                np.max(np.abs(result.action_chunk.physical_action[:, : spec.arm_dof]))
            )
            chunk_gripper_min = float(np.min(result.action_chunk.physical_action[:, -1]))
            chunk_gripper_max = float(np.max(result.action_chunk.physical_action[:, -1]))
            normalized_action_abs_max = max(
                normalized_action_abs_max or 0.0,
                normalized_max,
            )
            physical_arm_delta_abs_max_rad = max(
                physical_arm_delta_abs_max_rad or 0.0,
                physical_arm_max,
            )
            gripper_target_min = (
                chunk_gripper_min
                if gripper_target_min is None
                else min(gripper_target_min, chunk_gripper_min)
            )
            gripper_target_max = (
                chunk_gripper_max
                if gripper_target_max is None
                else max(gripper_target_max, chunk_gripper_max)
            )
            if result.action_chunk.rtc_trace is not None:
                rtc_trace = result.action_chunk.rtc_trace.to_dict()
                rtc_trace.update(
                    {
                        "replan_index": replans - 1,
                        "control_step": replan_control_step,
                        "sampling_seed": result.action_chunk.sampling.seed,
                        "completed_skill_count_before": completed_before,
                        "completed_skill_count_after": controller.progress.completed_skill_count,
                        "tcp_to_object_distance_before_m": tcp_distance_before,
                        "tcp_to_object_distance_after_m": (
                            controller.progress.outcome.tcp_to_object_distance_m
                        ),
                        "tcp_linear_speed_before_m_s": tcp_speed_before,
                        "tcp_linear_speed_after_m_s": controller.last_tcp_linear_speed_m_s,
                        "joint_velocity_abs_max_rad_s": joint_velocity_abs_max,
                        "gripper_target_first": float(
                            result.action_chunk.physical_action[0, -1]
                        ),
                        "temporal_proposal_spread": None,
                    }
                )
                replan_trace.update(rtc_trace)
        replan_trace.update(
            {
                "completed_skill_count_after": controller.progress.completed_skill_count,
                "tcp_to_object_distance_after_m": (
                    controller.progress.outcome.tcp_to_object_distance_m
                ),
                "tcp_linear_speed_after_m_s": controller.last_tcp_linear_speed_m_s,
            }
        )
        replan_traces.append(replan_trace)
        if not result.execution.success:
            failure_stage = result.execution.failure_stage
            error = result.execution.error
            execution_diagnostic = result.execution.diagnostic
            break

    progress = controller.progress
    outcome = progress.outcome
    predicate_success = progress.task_completed
    environment_success = controller.environment_success
    success = predicate_success and environment_success
    predicate_config = tracker.config
    failure_category = classify_rollout_failure(
        completed_skill_count=progress.completed_skill_count,
        predicate_success=predicate_success,
        environment_success=environment_success,
        failure_stage=failure_stage,
        final_is_grasped=outcome.grasped,
        final_object_to_goal_distance_m=outcome.object_to_goal_distance_m,
        place_distance_m=predicate_config.place_distance_m,
        final_object_linear_speed_m_s=outcome.object_linear_speed_m_s,
        static_linear_speed_m_s=predicate_config.static_linear_speed_m_s,
        final_object_angular_speed_rad_s=outcome.object_angular_speed_rad_s,
        static_angular_speed_rad_s=predicate_config.static_angular_speed_rad_s,
    )
    completed_skill_count = progress.completed_skill_count
    return RolloutEpisodeResult(
        seed_group=episode.seed_group,
        seed=episode.seed,
        instruction=episode.instruction,
        sampling_seed_base=sampling_seed_base,
        success=success,
        environment_success=environment_success,
        predicate_success=predicate_success,
        failure_category=failure_category,
        failure_stage=failure_stage,
        error=error,
        environment_steps=controller.environment_steps,
        replans=replans,
        sampling_seeds=tuple(sampling_seeds),
        action_chunks=action_chunks,
        normalized_action_abs_max=normalized_action_abs_max,
        physical_arm_delta_abs_max_rad=physical_arm_delta_abs_max_rad,
        gripper_target_min=gripper_target_min,
        gripper_target_max=gripper_target_max,
        completed_skill_count=completed_skill_count,
        skill_completed=tuple(
            index < completed_skill_count for index in range(len(PICK_AND_PLACE_SKILLS))
        ),
        terminated=controller.terminated,
        truncated=controller.truncated,
        final_is_grasped=outcome.grasped,
        stable_grasp_steps=progress.stable_grasp_steps,
        stable_place_steps=progress.stable_place_steps,
        final_tcp_to_object_distance_m=outcome.tcp_to_object_distance_m,
        final_object_height_above_support_m=outcome.object_height_above_support_m,
        final_object_to_goal_xy_distance_m=outcome.object_to_goal_xy_distance_m,
        final_object_to_goal_distance_m=outcome.object_to_goal_distance_m,
        final_object_linear_speed_m_s=outcome.object_linear_speed_m_s,
        final_object_angular_speed_rad_s=outcome.object_angular_speed_rad_s,
        wall_time_s=time.monotonic() - started,
        execution_diagnostic=execution_diagnostic,
        tracking_correction_saturation_count=tracking_correction_saturation_count,
        tracking_correction_requested_abs_max_rad=tracking_correction_requested_abs_max_rad,
        tracking_correction_applied_abs_max_rad=tracking_correction_applied_abs_max_rad,
        anomaly_replan_count=anomaly_replan_count,
        temporal_ensemble_max_buffer_size=temporal_ensemble_max_buffer_size,
        temporal_ensemble_max_proposal_spread=temporal_ensemble_max_proposal_spread,
        temporal_ensemble_min_newest_weight=temporal_ensemble_min_newest_weight,
        inference_strategy=loop.inference_strategy.value,
        skill_completion_environment_steps=tuple(
            controller.skill_completion_environment_steps
        ),
        replan_traces=tuple(replan_traces),
    )


def run_maniskill_atomic_episode(
    env: Any,
    runtime: QwenVLARuntime,
    spec: RobotSpec,
    *,
    seed: int,
    skill_name: str,
    instruction: str,
    sampling_seed_base: int,
    preparation: Any,
    max_policy_steps: int,
    inference_strategy: str | ChunkInferenceStrategy | None = None,
    temporal_ensemble_enabled: bool | None = None,
    recency_decay: float = 0.5,
    rtc_config: RTCConfig | None = None,
    max_anomaly_replans: int = 3,
) -> AtomicSkillEpisodeResult:
    """从专家验证过的前置状态开始，只评估一个目标原子技能。"""

    if max_policy_steps <= 0:
        raise ValueError("max_policy_steps 必须为正数")
    target_skill_id = PICK_AND_PLACE_SKILLS.index(skill_name)
    if preparation.progress.completed_skill_count != target_skill_id:
        raise ValueError("原子评估前置技能数与目标技能不一致")
    _reset_atomic_time_limit(env)
    started = time.monotonic()
    observation_adapter = FrankaObservationAdapter(spec)
    controller = _TrackingManiSkillController(
        env,
        spec,
        preparation.observation,
        preparation.tracker,
        preparation.progress,
        observation_adapter,
        observation_v2_enabled=isinstance(runtime, QwenVLAObservationV2Runtime),
    )
    loop = QwenVLAReplanLoop(
        runtime,
        RecedingHorizonChunkExecutor(spec),
        inference_strategy=inference_strategy,
        temporal_ensemble_enabled=temporal_ensemble_enabled,
        recency_decay=recency_decay,
        rtc_config=rtc_config,
        max_anomaly_replans=max_anomaly_replans,
    )
    replans = 0
    sampling_seeds: list[int] = []
    action_chunks = 0
    failure_stage: str | None = None
    error: str | None = None
    saturation_count = 0
    requested_abs_max: float | None = None
    applied_abs_max: float | None = None
    anomaly_replan_count = 0
    temporal_ensemble_max_buffer_size = 0
    temporal_ensemble_max_proposal_spread = 0.0
    temporal_ensemble_min_newest_weight: float | None = None
    replan_traces: list[dict[str, Any]] = []

    def target_completed() -> bool:
        return controller.progress.completed_skill_count >= target_skill_id + 1

    while (
        not controller.done
        and not target_completed()
        and controller.environment_steps < max_policy_steps
    ):
        replans += 1
        if controller.observation_v2_history is None:
            online_observation = _read_online_observation(
                controller.observation,
                env.unwrapped,
                observation_adapter,
                instruction,
            )
        else:
            command_reference = loop.executor.previous_command_q
            if command_reference is None:
                command_reference = controller.read_state().joint_positions
            online_observation = controller.observation_v2_history.snapshot(
                instruction,
                previous_command_q=command_reference,
                previous_action=loop.executor.previous_action,
            )
        replan_control_step = loop.control_step
        result = loop.replan_and_execute(online_observation, controller)
        execution = result.execution
        replan_trace: dict[str, Any] = {
            "replan_index": replans - 1,
            "control_step": replan_control_step,
            "sampling_seed": None if result.sampling is None else result.sampling.seed,
            "inference_strategy": result.inference_strategy.value,
            "rtc_enabled": result.inference_strategy == ChunkInferenceStrategy.RTC,
            "rtc_guidance_weight": None,
            "rtc_execution_horizon": None,
            "rtc_schedule": None,
            "previous_chunk_available": False,
            "overlap_length": 0,
            "temporal_proposal_spread": None,
        }
        anomaly_replan_count += int(execution.replan_required)
        if result.ensemble_trace is not None:
            replan_trace["temporal_proposal_spread"] = (
                result.ensemble_trace.max_proposal_spread
            )
            temporal_ensemble_max_buffer_size = max(
                temporal_ensemble_max_buffer_size,
                result.ensemble_trace.buffer_size,
            )
            temporal_ensemble_max_proposal_spread = max(
                temporal_ensemble_max_proposal_spread,
                result.ensemble_trace.max_proposal_spread,
            )
            newest_weight = min(
                result.ensemble_trace.newest_normalized_weights[: spec.execute_steps]
            )
            temporal_ensemble_min_newest_weight = (
                newest_weight
                if temporal_ensemble_min_newest_weight is None
                else min(temporal_ensemble_min_newest_weight, newest_weight)
            )
        if result.sampling is not None:
            sampling_seeds.append(result.sampling.seed)
        action_chunks += int(result.action_chunk is not None)
        if result.action_chunk is not None and result.action_chunk.rtc_trace is not None:
            rtc_trace = result.action_chunk.rtc_trace.to_dict()
            rtc_trace.update(
                {
                    "replan_index": replans - 1,
                    "control_step": loop.control_step - execution.executed_steps,
                    "sampling_seed": result.action_chunk.sampling.seed,
                }
            )
            replan_trace.update(rtc_trace)
        replan_traces.append(replan_trace)
        saturation_count += execution.correction_saturation_steps
        if execution.requested_correction_abs_max_rad is not None:
            requested_abs_max = max(
                requested_abs_max or 0.0,
                execution.requested_correction_abs_max_rad,
            )
        if execution.applied_correction_abs_max_rad is not None:
            applied_abs_max = max(
                applied_abs_max or 0.0,
                execution.applied_correction_abs_max_rad,
            )
        if not execution.success:
            failure_stage = execution.failure_stage
            error = execution.error
            break

    progress = controller.progress
    success = progress.completed_skill_count >= target_skill_id + 1
    stage_categories = {
        "inference": "inference_error",
        "initial_observation": "controller_observation_error",
        "step_observation": "controller_observation_error",
        "chunk_safety": "action_safety_rejection",
        "step_safety": "action_safety_rejection",
        "controller_step": "controller_error",
        "replan_anomaly_exhausted": "replan_anomaly_exhausted",
    }
    failure_category = None if success else stage_categories.get(
        failure_stage,
        f"{skill_name}_failed",
    )
    outcome = progress.outcome
    return AtomicSkillEpisodeResult(
        seed=seed,
        skill_name=skill_name,
        instruction=instruction,
        sampling_seed_base=sampling_seed_base,
        success=success,
        failure_category=failure_category,
        failure_stage=failure_stage,
        error=error,
        preparation_steps=preparation.preparation_steps,
        initial_completed_skill_count=target_skill_id,
        final_completed_skill_count=progress.completed_skill_count,
        policy_environment_steps=controller.environment_steps,
        replans=replans,
        sampling_seeds=tuple(sampling_seeds),
        action_chunks=action_chunks,
        tracking_correction_saturation_count=saturation_count,
        tracking_correction_requested_abs_max_rad=requested_abs_max,
        tracking_correction_applied_abs_max_rad=applied_abs_max,
        final_is_grasped=outcome.grasped,
        final_tcp_to_object_distance_m=outcome.tcp_to_object_distance_m,
        final_object_height_above_support_m=outcome.object_height_above_support_m,
        final_object_to_goal_xy_distance_m=outcome.object_to_goal_xy_distance_m,
        final_object_to_goal_distance_m=outcome.object_to_goal_distance_m,
        final_object_linear_speed_m_s=outcome.object_linear_speed_m_s,
        final_object_angular_speed_rad_s=outcome.object_angular_speed_rad_s,
        wall_time_s=time.monotonic() - started,
        anomaly_replan_count=anomaly_replan_count,
        temporal_ensemble_max_buffer_size=temporal_ensemble_max_buffer_size,
        temporal_ensemble_max_proposal_spread=temporal_ensemble_max_proposal_spread,
        temporal_ensemble_min_newest_weight=temporal_ensemble_min_newest_weight,
        inference_strategy=loop.inference_strategy.value,
        replan_traces=tuple(replan_traces),
    )


class ManiSkillPickPlaceEvaluator:
    def __init__(
        self,
        policy: QwenVLAPolicy,
        processor_adapter: QwenVLAProcessorAdapter,
        proprio_normalizer: ProprioNormalizer,
        spec: RobotSpec,
        *,
        finger_force_normalizer: FingerForceNormalizer | None = None,
        device: str | torch.device = "cuda",
        num_flow_steps: int = 10,
        sampling_seed: int = 42,
        inference_strategy: str | ChunkInferenceStrategy | None = None,
        temporal_ensemble_enabled: bool | None = None,
        recency_decay: float = 0.5,
        rtc_config: RTCConfig | None = None,
        max_anomaly_replans: int = 3,
    ) -> None:
        if sampling_seed < 0:
            raise ValueError("sampling_seed 不能为负数")
        self.policy = policy
        self.processor_adapter = processor_adapter
        self.proprio_normalizer = proprio_normalizer
        self.finger_force_normalizer = finger_force_normalizer
        self.spec = spec
        if isinstance(policy, QwenVLAObservationV2Policy) != (
            finger_force_normalizer is not None
        ):
            raise ValueError("Observation V2 policy 与 FingerForceNormalizer 必须成对提供")
        self.device = torch.device(device)
        self.num_flow_steps = num_flow_steps
        self.sampling_seed = sampling_seed
        self.inference_strategy = inference_strategy
        self.temporal_ensemble_enabled = temporal_ensemble_enabled
        self.recency_decay = recency_decay
        self.rtc_config = rtc_config
        self.max_anomaly_replans = max_anomaly_replans
        register_robot_vla_maniskill_envs()
        self.env = gym.make(
            PICK_CUBE_TO_REGION_ENV_ID,
            obs_mode="rgb",
            control_mode="pd_joint_delta_pos",
            num_envs=1,
        )

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def evaluate(self, episode: RolloutEpisodeSpec) -> RolloutEpisodeResult:
        episode_sampling_seed = derive_episode_sampling_seed(self.sampling_seed, episode)
        runtime_config = RuntimeConfig(
            num_flow_steps=self.num_flow_steps,
            use_bf16=self.device.type == "cuda",
            sampling_seed=episode_sampling_seed,
        )
        if isinstance(self.policy, QwenVLAObservationV2Policy):
            assert self.finger_force_normalizer is not None
            runtime = QwenVLAObservationV2Runtime(
                self.policy,
                self.processor_adapter,
                self.proprio_normalizer,
                self.finger_force_normalizer,
                self.spec,
                self.device,
                runtime_config,
            )
        else:
            runtime = QwenVLARuntime(
                self.policy,
                self.processor_adapter,
                self.proprio_normalizer,
                self.spec,
                self.device,
                runtime_config,
            )
        return run_maniskill_episode(
            self.env,
            runtime,
            self.spec,
            episode,
            sampling_seed_base=episode_sampling_seed,
            inference_strategy=self.inference_strategy,
            temporal_ensemble_enabled=self.temporal_ensemble_enabled,
            recency_decay=self.recency_decay,
            rtc_config=self.rtc_config,
            max_anomaly_replans=self.max_anomaly_replans,
        )


class ManiSkillAtomicPickPlaceEvaluator:
    def __init__(
        self,
        policy: QwenVLAPolicy,
        processor_adapter: QwenVLAProcessorAdapter,
        proprio_normalizer: ProprioNormalizer,
        spec: RobotSpec,
        *,
        finger_force_normalizer: FingerForceNormalizer | None = None,
        device: str | torch.device = "cuda",
        num_flow_steps: int = 10,
        sampling_seed: int = 42,
        inference_strategy: str | ChunkInferenceStrategy | None = None,
        temporal_ensemble_enabled: bool | None = None,
        recency_decay: float = 0.5,
        rtc_config: RTCConfig | None = None,
        max_anomaly_replans: int = 3,
    ) -> None:
        from robot_vla.sim.collector import TrustedPickPlaceCollector

        self.policy = policy
        self.processor_adapter = processor_adapter
        self.proprio_normalizer = proprio_normalizer
        self.finger_force_normalizer = finger_force_normalizer
        self.spec = spec
        if isinstance(policy, QwenVLAObservationV2Policy) != (
            finger_force_normalizer is not None
        ):
            raise ValueError("Observation V2 policy 与 FingerForceNormalizer 必须成对提供")
        self.device = torch.device(device)
        self.num_flow_steps = num_flow_steps
        self.sampling_seed = sampling_seed
        self.inference_strategy = inference_strategy
        self.temporal_ensemble_enabled = temporal_ensemble_enabled
        self.recency_decay = recency_decay
        self.rtc_config = rtc_config
        self.max_anomaly_replans = max_anomaly_replans
        self.preparer = TrustedPickPlaceCollector(None, spec)

    def close(self) -> None:
        self.preparer.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def evaluate(
        self,
        *,
        seed: int,
        skill_name: str,
        instruction: str,
        max_policy_steps: int,
    ) -> AtomicSkillEpisodeResult:
        preparation = self.preparer.prepare_atomic(seed=seed, skill_name=skill_name)
        episode_sampling_seed = derive_atomic_sampling_seed(
            self.sampling_seed,
            seed,
            skill_name,
        )
        runtime_config = RuntimeConfig(
            num_flow_steps=self.num_flow_steps,
            use_bf16=self.device.type == "cuda",
            sampling_seed=episode_sampling_seed,
        )
        if isinstance(self.policy, QwenVLAObservationV2Policy):
            assert self.finger_force_normalizer is not None
            runtime = QwenVLAObservationV2Runtime(
                self.policy,
                self.processor_adapter,
                self.proprio_normalizer,
                self.finger_force_normalizer,
                self.spec,
                self.device,
                runtime_config,
            )
        else:
            runtime = QwenVLARuntime(
                self.policy,
                self.processor_adapter,
                self.proprio_normalizer,
                self.spec,
                self.device,
                runtime_config,
            )
        return run_maniskill_atomic_episode(
            self.preparer.env,
            runtime,
            self.spec,
            seed=seed,
            skill_name=skill_name,
            instruction=instruction,
            sampling_seed_base=episode_sampling_seed,
            preparation=preparation,
            max_policy_steps=max_policy_steps,
            inference_strategy=self.inference_strategy,
            temporal_ensemble_enabled=self.temporal_ensemble_enabled,
            recency_decay=self.recency_decay,
            rtc_config=self.rtc_config,
            max_anomaly_replans=self.max_anomaly_replans,
        )


__all__ = [
    "ManiSkillAtomicPickPlaceEvaluator",
    "ManiSkillPickPlaceEvaluator",
    "derive_episode_sampling_seed",
    "run_maniskill_atomic_episode",
    "run_maniskill_episode",
]
