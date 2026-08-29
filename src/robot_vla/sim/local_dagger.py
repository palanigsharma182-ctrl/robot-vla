"""E012：Policy roll-in 后在同一 Session 内由 Expert 完整接管。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from robot_vla.contracts import RobotSpec
from robot_vla.data.events import EVENT_STATE_CONTRACT_VERSION
from robot_vla.data.trajectory import (
    ACTION_SOURCE_POLICY,
    LOCAL_DAGGER_BOUNDARIES,
    LOCAL_DAGGER_WINDOW_STEPS,
    LocalDaggerProvenance,
    OutcomeEvidence,
    TrajectoryMeta,
)
from robot_vla.execution import ManiSkillFrankaController, RecedingHorizonChunkExecutor
from robot_vla.runtime import OnlineObservation, QwenVLAReplanLoop, QwenVLARuntime
from robot_vla.sim import PICK_CUBE_TO_REGION_ENV_ID
from robot_vla.sim.collector import (
    EpisodeRejected,
    TrustedPickPlaceCollector,
    _CollectionSession,
    _numpy,
    _single_bool,
)
from robot_vla.sim.snapshot import (
    CollectionSnapshotBundle,
    CollectionSnapshotRing,
    SnapshotRoundTripReport,
    build_snapshot_evidence,
    capture_collection_snapshot,
    verify_snapshot_round_trip,
)
from robot_vla.tasks.pick_place import PickPlaceState, build_pick_place_task


@dataclass(frozen=True)
class BoundaryDiagnostics:
    boundary_type: str
    control_step: int
    tcp_object_relative_xyz_m: tuple[float, float, float]
    tcp_object_xy_error_m: float
    tcp_object_relative_z_m: float
    tcp_linear_speed_m_s: float
    joint_velocity_rms_rad_s: float
    joint_velocity_abs_max_rad_s: float
    gripper_opening: float
    gripper_target: float
    object_linear_speed_m_s: float
    object_angular_speed_rad_s: float
    robot_object_contact_force_n: float
    support_contact_force_n: float
    is_grasped: bool
    temporal_buffer_size: int
    temporal_proposal_count: int
    temporal_max_proposal_spread: float
    arm_mean_pairwise_disagreement: float
    gripper_mean_pairwise_disagreement: float
    arm_newest_vs_oldest: float
    gripper_newest_vs_oldest: float
    arm_newest_vs_weighted_history: float
    gripper_newest_vs_weighted_history: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalDaggerCollectionResult:
    meta: TrajectoryMeta
    boundary: BoundaryDiagnostics
    policy_replans: int
    policy_sampling_seeds: tuple[int, ...]
    policy_replan_traces: tuple[dict[str, Any], ...]
    snapshot_summaries: tuple[dict[str, Any], ...]
    snapshot_round_trip: SnapshotRoundTripReport | None
    boundary_snapshot: CollectionSnapshotBundle = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory": self.meta.to_dict(),
            "boundary": self.boundary.to_dict(),
            "policy_replans": self.policy_replans,
            "policy_sampling_seeds": list(self.policy_sampling_seeds),
            "policy_replan_traces": list(self.policy_replan_traces),
            "snapshot_summaries": list(self.snapshot_summaries),
            "snapshot_round_trip": (
                None
                if self.snapshot_round_trip is None
                else self.snapshot_round_trip.to_dict()
            ),
        }


@dataclass(frozen=True)
class CleanExpertBoundaryResult:
    boundary: BoundaryDiagnostics
    num_steps: int
    task_completed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary.to_dict(),
            "num_steps": self.num_steps,
            "task_completed": self.task_completed,
        }


class _LocalDaggerPolicyController(ManiSkillFrankaController):
    """记录 Policy 执行动作，并在目标 Predicate 跨越后停止当前 Chunk。"""

    def __init__(
        self,
        collector: LocalDaggerPickPlaceCollector,
        session: _CollectionSession,
        *,
        target_completed_skill_count: int,
    ) -> None:
        super().__init__(collector.env, collector.spec)
        self.collector = collector
        self.session = session
        self.target_completed_skill_count = target_completed_skill_count
        self.chunk_stop_requested = False
        self.terminal_before_boundary = False
        initial = collector._read_predicate_state()
        self._last_tcp_position = np.asarray(initial.tcp_position, dtype=np.float64)
        self.last_tcp_linear_speed_m_s = 0.0

    def send_action(self, controller_action: np.ndarray) -> None:
        if self.chunk_stop_requested:
            raise RuntimeError("Boundary 已发生后仍尝试执行 Policy Action")
        action = np.asarray(controller_action, dtype=np.float32)
        if action.shape != (self.spec.action_dim,) or not np.isfinite(action).all():
            raise ValueError("Policy controller_action shape/dtype 无效")
        actual_q = self.collector._actual_arm_q()
        physical = np.empty(self.spec.action_dim, dtype=np.float32)
        physical[: self.spec.arm_dof] = (
            action[: self.spec.arm_dof] * self.spec.maniskill_arm_delta_range_rad
        )
        physical[-1] = (action[-1] + 1.0) * 0.5
        target_q = actual_q + physical[: self.spec.arm_dof]
        self.collector.action_adapter.normalize(physical, strict=True)

        predicate_before = self.collector._read_predicate_state()
        self.session.recorder.record_before_action(
            self.session.observation,
            physical,
            self.session.progress.active_skill_id,
            predicate_before,
            *self.collector._read_contact_forces(),
            target_q,
            physical[: self.spec.arm_dof],
            action_source=ACTION_SOURCE_POLICY,
        )
        super().send_action(action)
        observation, _, terminated, truncated, info = self.last_step_output
        self.session.recorder.record_after_action(terminated, truncated, info)
        self.session.observation = observation
        previous_completed = self.session.progress.completed_skill_count
        predicate_after = self.collector._read_predicate_state()
        self.session.progress = self.session.tracker.update(predicate_after)
        tcp_position = np.asarray(predicate_after.tcp_position, dtype=np.float64)
        self.last_tcp_linear_speed_m_s = float(
            np.linalg.norm(tcp_position - self._last_tcp_position) * self.spec.control_hz
        )
        self._last_tcp_position = tcp_position
        self.session.previous_command_q = target_q.astype(np.float32, copy=True)

        reached_boundary = (
            previous_completed < self.target_completed_skill_count
            <= self.session.progress.completed_skill_count
        )
        was_terminated = _single_bool(terminated)
        was_truncated = _single_bool(truncated)
        if was_terminated or was_truncated:
            self.session.done = True
            if not reached_boundary:
                self.terminal_before_boundary = True
        if reached_boundary or was_terminated or was_truncated:
            self.chunk_stop_requested = True


class LocalDaggerPickPlaceCollector(TrustedPickPlaceCollector):
    """复用可信 Expert，只新增 frozen Policy roll-in 和一次连续 takeover。"""

    def __init__(
        self,
        dataset_root: str | Path,
        spec: RobotSpec | None = None,
    ) -> None:
        super().__init__(dataset_root, spec)

    def _online_observation(
        self,
        session: _CollectionSession,
        instruction: str,
    ) -> OnlineObservation:
        sensor_data = session.observation["sensor_data"]
        external = _numpy(sensor_data["base_camera"]["rgb"])[0]
        wrist = _numpy(sensor_data["hand_camera"]["rgb"])[0]
        if external.dtype != np.uint8 or wrist.dtype != np.uint8:
            raise EpisodeRejected("Local DAgger 在线 RGB 必须是 uint8")
        robot = self.base_env.agent.robot
        qpos = _numpy(robot.get_qpos())
        qvel = _numpy(robot.get_qvel())
        joint_names = tuple(joint.name for joint in robot.active_joints)
        proprio = self.observation_adapter.from_maniskill(
            qpos[0],
            qvel[0],
            joint_names,
        )
        return OnlineObservation(
            rgb_external=external.copy(),
            rgb_wrist=wrist.copy(),
            physical_proprio=proprio.copy(),
            instruction=instruction,
        )

    def _boundary_diagnostics(
        self,
        *,
        boundary_type: str,
        control_step: int,
        controller: _LocalDaggerPolicyController,
        ensemble_trace: Any,
        executed_steps: int,
    ) -> BoundaryDiagnostics:
        state: PickPlaceState = self._read_predicate_state()
        tcp = np.asarray(state.tcp_position, dtype=np.float64)
        obj = np.asarray(state.object_position, dtype=np.float64)
        relative = tcp - obj
        robot = self.base_env.agent.robot
        qpos = _numpy(robot.get_qpos())
        qvel = _numpy(robot.get_qvel())
        proprio = self.observation_adapter.from_maniskill(
            qpos[0],
            qvel[0],
            tuple(joint.name for joint in robot.active_joints),
        )

        joint_velocity = proprio[self.spec.arm_dof : self.spec.arm_dof * 2]
        robot_force, support_force = self._read_contact_forces()
        proposal_index = max(0, min(executed_steps - 1, self.spec.action_horizon - 1))
        proposal_count = (
            1
            if ensemble_trace is None
            else int(ensemble_trace.proposal_counts[proposal_index])
        )
        return BoundaryDiagnostics(
            boundary_type=boundary_type,
            control_step=control_step,
            tcp_object_relative_xyz_m=tuple(float(value) for value in relative),
            tcp_object_xy_error_m=float(np.linalg.norm(relative[:2])),
            tcp_object_relative_z_m=float(relative[2]),
            tcp_linear_speed_m_s=controller.last_tcp_linear_speed_m_s,
            joint_velocity_rms_rad_s=float(np.sqrt(np.mean(np.square(joint_velocity)))),
            joint_velocity_abs_max_rad_s=float(np.max(np.abs(joint_velocity))),
            gripper_opening=float(proprio[-1]),
            gripper_target=float(controller._last_gripper_opening or 0.0),
            object_linear_speed_m_s=float(
                np.linalg.norm(np.asarray(state.object_linear_velocity, dtype=np.float64))
            ),
            object_angular_speed_rad_s=float(
                np.linalg.norm(np.asarray(state.object_angular_velocity, dtype=np.float64))
            ),
            robot_object_contact_force_n=robot_force,
            support_contact_force_n=support_force,
            is_grasped=state.is_grasped,
            temporal_buffer_size=(
                1 if ensemble_trace is None else int(ensemble_trace.buffer_size)
            ),
            temporal_proposal_count=proposal_count,
            temporal_max_proposal_spread=(
                0.0
                if ensemble_trace is None
                else float(ensemble_trace.max_proposal_spread)
            ),
            arm_mean_pairwise_disagreement=(
                0.0
                if ensemble_trace is None
                else float(
                    ensemble_trace.arm_mean_pairwise_disagreement[proposal_index]
                )
            ),
            gripper_mean_pairwise_disagreement=(
                0.0
                if ensemble_trace is None
                else float(
                    ensemble_trace.gripper_mean_pairwise_disagreement[proposal_index]
                )
            ),
            arm_newest_vs_oldest=(
                0.0
                if ensemble_trace is None
                else float(ensemble_trace.arm_newest_vs_oldest[proposal_index])
            ),
            gripper_newest_vs_oldest=(
                0.0
                if ensemble_trace is None
                else float(ensemble_trace.gripper_newest_vs_oldest[proposal_index])
            ),
            arm_newest_vs_weighted_history=(
                0.0
                if ensemble_trace is None
                else float(
                    ensemble_trace.arm_newest_vs_weighted_history[proposal_index]
                )
            ),
            gripper_newest_vs_weighted_history=(
                0.0
                if ensemble_trace is None
                else float(
                    ensemble_trace.gripper_newest_vs_weighted_history[proposal_index]
                )
            ),
        )

    def _verify_boundary_snapshot_round_trip(
        self,
        snapshot: CollectionSnapshotBundle,
        *,
        instruction: str,
    ) -> SnapshotRoundTripReport:
        """完整 live trajectory 已封存后，在同一 env 做 restore 诊断。"""

        def evidence_builder(observation, controller_targets):
            online = self._online_observation(
                SimpleNamespace(observation=observation),
                instruction,
            )
            return build_snapshot_evidence(
                observation,
                physical_proprio=online.physical_proprio,
                predicate_state=self._read_predicate_state(),
                contact_forces_n=self._read_contact_forces(),
                camera_calibration=self._camera_calibration(observation),
                controller_targets=controller_targets,
            )

        hold_physical = np.zeros(self.spec.action_dim, dtype=np.float32)
        hold_physical[-1] = snapshot.evidence.physical_proprio[-1]
        return verify_snapshot_round_trip(
            snapshot,
            self.env,
            controller_action=self.action_adapter.to_maniskill(hold_physical),
            evidence_builder=evidence_builder,
        )

    def _clean_expert_boundary_diagnostics(
        self,
        *,
        boundary_type: str,
        control_step: int,
        tcp_linear_speed_m_s: float,
        gripper_target: float,
    ) -> BoundaryDiagnostics:
        state: PickPlaceState = self._read_predicate_state()
        relative = np.asarray(state.tcp_position, dtype=np.float64) - np.asarray(
            state.object_position,
            dtype=np.float64,
        )
        robot = self.base_env.agent.robot
        qpos = _numpy(robot.get_qpos())
        qvel = _numpy(robot.get_qvel())
        proprio = self.observation_adapter.from_maniskill(
            qpos[0],
            qvel[0],
            tuple(joint.name for joint in robot.active_joints),
        )
        joint_velocity = proprio[self.spec.arm_dof : self.spec.arm_dof * 2]
        robot_force, support_force = self._read_contact_forces()
        return BoundaryDiagnostics(
            boundary_type=boundary_type,
            control_step=control_step,
            tcp_object_relative_xyz_m=tuple(float(value) for value in relative),
            tcp_object_xy_error_m=float(np.linalg.norm(relative[:2])),
            tcp_object_relative_z_m=float(relative[2]),
            tcp_linear_speed_m_s=tcp_linear_speed_m_s,
            joint_velocity_rms_rad_s=float(np.sqrt(np.mean(np.square(joint_velocity)))),
            joint_velocity_abs_max_rad_s=float(np.max(np.abs(joint_velocity))),
            gripper_opening=float(proprio[-1]),
            gripper_target=float(gripper_target),
            object_linear_speed_m_s=float(
                np.linalg.norm(np.asarray(state.object_linear_velocity, dtype=np.float64))
            ),
            object_angular_speed_rad_s=float(
                np.linalg.norm(np.asarray(state.object_angular_velocity, dtype=np.float64))
            ),
            robot_object_contact_force_n=robot_force,
            support_contact_force_n=support_force,
            is_grasped=state.is_grasped,
            temporal_buffer_size=0,
            temporal_proposal_count=0,
            temporal_max_proposal_spread=0.0,
            arm_mean_pairwise_disagreement=0.0,
            gripper_mean_pairwise_disagreement=0.0,
            arm_newest_vs_oldest=0.0,
            gripper_newest_vs_oldest=0.0,
            arm_newest_vs_weighted_history=0.0,
            gripper_newest_vs_weighted_history=0.0,
        )

    def collect_clean_expert_boundary(
        self,
        *,
        seed: int,
        boundary_type: str,
    ) -> CleanExpertBoundaryResult:
        """同 seed 的 clean Expert 完整 rollout，并捕获第一次目标 crossing。"""

        if boundary_type not in LOCAL_DAGGER_BOUNDARIES:
            raise ValueError(f"未知 boundary_type: {boundary_type}")
        session = self._start_session(seed)
        target_completed = 1 if boundary_type == "reach_grasp" else 2
        previous_completed = session.progress.completed_skill_count
        previous_tcp = np.asarray(
            self._read_predicate_state().tcp_position,
            dtype=np.float64,
        )
        boundary: BoundaryDiagnostics | None = None

        def capture_after_action(
            current_session: _CollectionSession,
            gripper_target: float,
        ) -> None:
            nonlocal boundary, previous_completed, previous_tcp
            state = self._read_predicate_state()
            tcp = np.asarray(state.tcp_position, dtype=np.float64)
            tcp_speed = float(np.linalg.norm(tcp - previous_tcp) * self.spec.control_hz)
            completed = current_session.progress.completed_skill_count
            if (
                boundary is None
                and previous_completed < target_completed <= completed
            ):
                boundary = self._clean_expert_boundary_diagnostics(
                    boundary_type=boundary_type,
                    control_step=len(current_session.recorder.action),
                    tcp_linear_speed_m_s=tcp_speed,
                    gripper_target=gripper_target,
                )
            previous_completed = completed
            previous_tcp = tcp

        session.after_action_hook = capture_after_action
        grasp_pose, reach_pose, lift_pose, transport_pose, lower_pose = self._phase_poses()
        self._move_to_pose(session, reach_pose, gripper_opening=1.0)
        self._move_to_pose(session, grasp_pose, gripper_opening=1.0)
        self._hold(session, gripper_opening=0.0, steps=8)
        self._move_to_pose(session, lift_pose, gripper_opening=0.0)
        self._move_to_pose(session, transport_pose, gripper_opening=0.0)
        self._move_to_pose(session, lower_pose, gripper_opening=0.0)
        self._hold(session, gripper_opening=1.0, steps=30, stop_on_success=True)
        if not session.done or not session.progress.task_completed:
            raise EpisodeRejected("Paired clean Expert 未完成完整 Pick-and-Place")
        if boundary is None:
            raise EpisodeRejected("Paired clean Expert 未捕获目标 boundary")
        arrays = session.recorder.build()
        return CleanExpertBoundaryResult(
            boundary=boundary,
            num_steps=arrays.num_steps,
            task_completed=session.progress.task_completed,
        )

    def collect_local_dagger(
        self,
        runtime: QwenVLARuntime,
        *,
        seed: int,
        boundary_type: str,
        policy_checkpoint_sha256: str,
        trajectory_id: str | None = None,
        instruction_index: int | None = None,
        recency_decay: float = 0.5,
        max_anomaly_replans: int = 3,
        verify_snapshot_round_trip: bool = True,
    ) -> LocalDaggerCollectionResult:
        if boundary_type not in LOCAL_DAGGER_BOUNDARIES:
            raise ValueError(f"未知 boundary_type: {boundary_type}")
        if not math.isfinite(recency_decay) or not 0.0 < recency_decay < 1.0:
            raise ValueError("recency_decay 必须位于 (0,1)")
        if self.writer is None:
            raise RuntimeError("Local DAgger collector 必须配置 dataset_root")

        task = build_pick_place_task(
            seed % 3 if instruction_index is None else instruction_index
        )
        session = self._start_session(seed, record_action_provenance=True)
        calibration = self._camera_calibration(session.observation)
        cube_initial = _numpy(self.base_env.cube.pose.p)[0].copy()
        goal_position = _numpy(self.base_env.goal_site.pose.p)[0].copy()
        target_completed = 1 if boundary_type == "reach_grasp" else 2
        controller = _LocalDaggerPolicyController(
            self,
            session,
            target_completed_skill_count=target_completed,
        )
        loop = QwenVLAReplanLoop(
            runtime,
            RecedingHorizonChunkExecutor(self.spec),
            inference_strategy="temporal-ensemble",
            recency_decay=recency_decay,
            max_anomaly_replans=max_anomaly_replans,
        )
        max_replans = int(self.env._max_episode_steps) + 1
        replans = 0
        sampling_seeds: list[int] = []
        replan_traces: list[dict[str, Any]] = []
        boundary_diagnostics: BoundaryDiagnostics | None = None
        boundary_snapshot: CollectionSnapshotBundle | None = None
        snapshot_ring = CollectionSnapshotRing()
        policy_identity = {
            "environment_id": PICK_CUBE_TO_REGION_ENV_ID,
            "control_mode": "pd_joint_delta_pos",
            "inference_strategy": "temporal-ensemble",
            "policy_checkpoint_sha256": policy_checkpoint_sha256,
            "boundary_type": boundary_type,
            "runtime_sampling_seed": runtime.config.sampling_seed,
            "runtime_starting_sample_index": runtime.config.starting_sample_index,
            "num_flow_steps": runtime.config.num_flow_steps,
            "recency_decay": recency_decay,
            "max_anomaly_replans": max_anomaly_replans,
        }

        while not controller.chunk_stop_requested and replans < max_replans:
            replan_index = replans
            online_observation = self._online_observation(session, task.instruction)
            snapshot_ring.append(
                capture_collection_snapshot(
                    self.env,
                    session,
                    loop,
                    label="replan_start",
                    replan_index=replan_index,
                    environment_seed=seed,
                    physical_proprio=online_observation.physical_proprio,
                    predicate_state=self._read_predicate_state(),
                    contact_forces_n=self._read_contact_forces(),
                    camera_calibration=calibration,
                    collection_policy_identity=policy_identity,
                )
            )
            replans += 1
            replan_control_step = loop.control_step
            completed_before = session.progress.completed_skill_count
            result = loop.replan_and_execute(
                online_observation,
                controller,
            )
            if result.sampling is not None:
                sampling_seeds.append(result.sampling.seed)
            trace = {
                "replan_index": replans - 1,
                "control_step": replan_control_step,
                "sampling_seed": None if result.sampling is None else result.sampling.seed,
                "executed_steps": result.execution.executed_steps,
                "completed_skill_count_before": completed_before,
                "completed_skill_count_after": session.progress.completed_skill_count,
                "temporal_buffer_size": (
                    None if result.ensemble_trace is None else result.ensemble_trace.buffer_size
                ),
                "temporal_max_proposal_spread": (
                    None
                    if result.ensemble_trace is None
                    else result.ensemble_trace.max_proposal_spread
                ),
                "replan_required": result.execution.replan_required,
            }
            replan_traces.append(trace)
            if not result.execution.success:
                raise EpisodeRejected(
                    "Policy roll-in 执行失败："
                    f"{result.execution.failure_stage}: {result.execution.error}"
                )
            if controller.terminal_before_boundary:
                raise EpisodeRejected("Policy 在目标 boundary 前终止或截断")
            if session.progress.completed_skill_count >= target_completed:
                boundary_online_observation = self._online_observation(
                    session,
                    task.instruction,
                )
                boundary_snapshot = capture_collection_snapshot(
                    self.env,
                    session,
                    loop,
                    label="boundary_crossing",
                    replan_index=replan_index,
                    environment_seed=seed,
                    physical_proprio=boundary_online_observation.physical_proprio,
                    predicate_state=self._read_predicate_state(),
                    contact_forces_n=self._read_contact_forces(),
                    camera_calibration=calibration,
                    collection_policy_identity=policy_identity,
                )
                snapshot_ring.append(boundary_snapshot)
                boundary_diagnostics = self._boundary_diagnostics(
                    boundary_type=boundary_type,
                    control_step=len(session.recorder.action),
                    controller=controller,
                    ensemble_trace=result.ensemble_trace,
                    executed_steps=result.execution.executed_steps,
                )
                break

        if boundary_diagnostics is None:
            raise EpisodeRejected("Policy 未在 Episode 预算内到达目标 boundary")
        if boundary_snapshot is None:
            raise RuntimeError("Boundary 已发生但缺少 crossing snapshot")
        if session.done:
            raise EpisodeRejected("目标 boundary 发生时环境已经结束")

        expert_takeover_step = len(session.recorder.action)
        if expert_takeover_step != boundary_diagnostics.control_step:
            raise RuntimeError("Boundary 与 takeover Action 索引不一致")

        grasp_pose, _, lift_pose, transport_pose, lower_pose = self._phase_poses()
        if boundary_type == "reach_grasp":
            self._move_to_pose(session, grasp_pose, gripper_opening=1.0)
            self._hold(session, gripper_opening=0.0, steps=8)
        else:
            self._hold(session, gripper_opening=0.0, steps=8)
        if session.progress.completed_skill_count < 2:
            raise EpisodeRejected("Expert takeover 后未形成稳定 Grasp")
        self._move_to_pose(session, lift_pose, gripper_opening=0.0)
        self._move_to_pose(session, transport_pose, gripper_opening=0.0)
        self._move_to_pose(session, lower_pose, gripper_opening=0.0)
        self._hold(session, gripper_opening=1.0, steps=30, stop_on_success=True)
        if not session.done or not session.progress.task_completed:
            raise EpisodeRejected("Local DAgger Expert 未完成完整 Pick-and-Place")

        arrays = session.recorder.build()
        outcome = session.progress.outcome
        snapshot_round_trip = None
        if verify_snapshot_round_trip:
            snapshot_round_trip = self._verify_boundary_snapshot_round_trip(
                boundary_snapshot,
                instruction=task.instruction,
            )
            if not snapshot_round_trip.passed:
                raise EpisodeRejected(
                    "Boundary snapshot round-trip 未通过："
                    f"{snapshot_round_trip.to_dict()}"
                )
        training_window_end = min(
            expert_takeover_step + LOCAL_DAGGER_WINDOW_STEPS,
            arrays.num_steps,
        )
        if training_window_end - expert_takeover_step < self.spec.action_horizon:
            raise EpisodeRejected("Local DAgger Expert 窗口不足一个完整 Action Chunk")
        source = f"dagger_{boundary_type}"
        resolved_trajectory_id = trajectory_id or (
            f"local-dagger-{boundary_type}-seed-{seed:06d}"
        )
        meta = TrajectoryMeta(
            trajectory_id=resolved_trajectory_id,
            source_episode_id=f"maniskill-local-dagger-{boundary_type}-seed-{seed:06d}",
            file=f"trajectories/{resolved_trajectory_id}.npz",
            split="train",
            scene_id=f"{PICK_CUBE_TO_REGION_ENV_ID}:seed={seed}",
            task=task,
            num_steps=arrays.num_steps,
            camera_calibration=calibration,
            randomization={
                "seed": seed,
                "environment_id": PICK_CUBE_TO_REGION_ENV_ID,
                "control_mode": "pd_joint_delta_pos",
                "event_state_contract_version": EVENT_STATE_CONTRACT_VERSION,
                "cube_initial_position_m": cube_initial.tolist(),
                "goal_position_m": goal_position.tolist(),
                "collection_inference_strategy": "temporal-ensemble",
                "collection_recency_decay": recency_decay,
                "collection_max_anomaly_replans": max_anomaly_replans,
            },
            outcome_evidence=OutcomeEvidence(
                predicate_version=session.tracker.config.version,
                task_completed=session.progress.task_completed,
                final_is_released=not outcome.grasped,
                stable_place_steps=session.progress.stable_place_steps,
                external_goal_visible_steps=session.recorder.external_goal_visible_steps,
                wrist_goal_visible_steps=session.recorder.wrist_goal_visible_steps,
                both_goal_visible_steps=session.recorder.both_goal_visible_steps,
                final_object_to_goal_distance_m=outcome.object_to_goal_distance_m,
                final_object_linear_speed_m_s=outcome.object_linear_speed_m_s,
                final_object_angular_speed_rad_s=outcome.object_angular_speed_rad_s,
            ),
            local_dagger=LocalDaggerProvenance(
                source=source,
                rollin_seed=seed,
                rollin_policy_checkpoint_sha256=policy_checkpoint_sha256,
                boundary_type=boundary_type,
                boundary_detection_step=expert_takeover_step,
                expert_takeover_step=expert_takeover_step,
                training_window_start=expert_takeover_step,
                training_window_end=training_window_end,
                expert_recovery_success=True,
            ),
        )
        self.writer.write(meta, arrays)
        return LocalDaggerCollectionResult(
            meta=meta,
            boundary=boundary_diagnostics,
            policy_replans=replans,
            policy_sampling_seeds=tuple(sampling_seeds),
            policy_replan_traces=tuple(replan_traces),
            snapshot_summaries=snapshot_ring.summaries(),
            snapshot_round_trip=snapshot_round_trip,
            boundary_snapshot=boundary_snapshot,
        )


__all__ = [
    "BoundaryDiagnostics",
    "CleanExpertBoundaryResult",
    "LocalDaggerCollectionResult",
    "LocalDaggerPickPlaceCollector",
]
