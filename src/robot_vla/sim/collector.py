"""MPlib 专家到可信 trajectory/v2 的双相机成功轨迹采集器。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import sapien
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from typing_extensions import Self

from robot_vla.adapters import ActionAdapter, FrankaObservationAdapter
from robot_vla.contracts import (
    FINGER_FORCE_SENSOR_VERSION,
    OBSERVATION_V2_VERSION,
    RobotSpec,
)
from robot_vla.data.events import EVENT_STATE_CONTRACT_VERSION
from robot_vla.data.recovery import RECOVERY_CONTRACT_VERSION, RECOVERY_PROFILES
from robot_vla.data.trajectory import (
    ACTION_SOURCE_EXPERT,
    ACTION_SOURCE_POLICY,
    CameraCalibration,
    OutcomeEvidence,
    TrajectoryArrays,
    TrajectoryMeta,
)
from robot_vla.data.writer import TrajectoryDatasetWriter
from robot_vla.observation import (
    OBSERVATION_MODALITIES,
    ObservationV2Frame,
    invert_se3,
    opengl_camera_to_opencv,
    transform_to_position_rotation_6d,
    validate_se3,
)
from robot_vla.precision.collection import PrecisionLabelRecorder
from robot_vla.precision.data import (
    PrecisionLabelDatasetWriter,
    build_precision_label_meta,
)
from robot_vla.sim import PICK_CUBE_TO_REGION_ENV_ID, register_robot_vla_maniskill_envs
from robot_vla.tasks.pick_place import (
    ATOMIC_PICK_PLACE_SKILLS,
    PickPlaceState,
    PickPlaceTaskProgress,
    PickPlaceTaskTracker,
    build_pick_place_task,
)

CALIBRATION_VERSION = "maniskill-panda-wristcam-cam2world-gl/v1"


class EpisodeRejected(RuntimeError):
    """规划、执行或成功证据不满足可信数据契约。"""


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _single_bool(value: Any) -> bool:
    array = _numpy(value)
    if array.size != 1:
        raise EpisodeRejected(f"首版采集器只支持单环境 bool，实际 shape={array.shape}")
    return bool(array.reshape(-1)[0])


def _single_transform_matrix(pose: Any, name: str) -> np.ndarray:
    value = _numpy(pose.to_transformation_matrix())
    if value.shape == (1, 4, 4):
        value = value[0]
    return validate_se3(value, name)


@dataclass
class _EpisodeRecorder:
    spec: RobotSpec
    observation_adapter: FrankaObservationAdapter
    robot: Any
    goal_actor_id: int
    precision_label_recorder: PrecisionLabelRecorder | None = None
    rgb_external: list[np.ndarray] = field(default_factory=list)
    rgb_wrist: list[np.ndarray] = field(default_factory=list)
    proprio: list[np.ndarray] = field(default_factory=list)
    action: list[np.ndarray] = field(default_factory=list)
    skill_id: list[int] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    success: list[bool] = field(default_factory=list)
    robot_object_contact_force_n: list[float] = field(default_factory=list)
    support_contact_force_n: list[float] = field(default_factory=list)
    is_grasped: list[bool] = field(default_factory=list)
    object_position_m: list[np.ndarray] = field(default_factory=list)
    object_linear_velocity_m_s: list[np.ndarray] = field(default_factory=list)
    object_angular_velocity_rad_s: list[np.ndarray] = field(default_factory=list)
    commanded_joint_target_rad: list[np.ndarray] = field(default_factory=list)
    applied_joint_correction_rad: list[np.ndarray] = field(default_factory=list)
    tcp_position_base_m: list[np.ndarray] = field(default_factory=list)
    tcp_rotation_6d_base: list[np.ndarray] = field(default_factory=list)
    wrist_camera_position_base_m: list[np.ndarray] = field(default_factory=list)
    wrist_camera_rotation_6d_base: list[np.ndarray] = field(default_factory=list)
    left_finger_force_n: list[float] = field(default_factory=list)
    right_finger_force_n: list[float] = field(default_factory=list)
    previous_command_q_rad: list[np.ndarray] = field(default_factory=list)
    previous_action: list[np.ndarray] = field(default_factory=list)
    record_action_provenance: bool = False
    action_source: list[int] = field(default_factory=list)
    expert_supervision_mask: list[bool] = field(default_factory=list)
    external_goal_visible_steps: int = 0
    wrist_goal_visible_steps: int = 0
    both_goal_visible_steps: int = 0
    _pending_transition: bool = False

    def record_before_action(
        self,
        observation: dict[str, Any],
        physical_action: np.ndarray,
        skill_id: int,
        predicate_state: PickPlaceState,
        robot_object_contact_force_n: float,
        support_contact_force_n: float,
        commanded_joint_target_rad: np.ndarray,
        applied_joint_correction_rad: np.ndarray,
        action_source: int = ACTION_SOURCE_EXPERT,
        *,
        base_from_tcp: np.ndarray | None = None,
        base_from_wrist_camera: np.ndarray | None = None,
        finger_force_n: np.ndarray | None = None,
        previous_command_q_rad: np.ndarray | None = None,
        object_position_base_m: np.ndarray | None = None,
        goal_position_base_m: np.ndarray | None = None,
    ) -> None:
        if self._pending_transition:
            raise RuntimeError("上一条 Transition 尚未记录执行结果")
        external = observation["sensor_data"]["base_camera"]
        wrist = observation["sensor_data"]["hand_camera"]
        external_rgb = _numpy(external["rgb"])[0]
        wrist_rgb = _numpy(wrist["rgb"])[0]
        if external_rgb.dtype != np.uint8 or wrist_rgb.dtype != np.uint8:
            raise EpisodeRejected("ManiSkill RGB 必须是 uint8")

        qpos = _numpy(self.robot.get_qpos())
        qvel = _numpy(self.robot.get_qvel())
        joint_names = tuple(joint.name for joint in self.robot.active_joints)
        if qpos.shape[0] != 1 or qvel.shape != qpos.shape:
            raise EpisodeRejected("首版采集器只支持 num_envs=1")
        proprio = self.observation_adapter.from_maniskill(
            qpos[0],
            qvel[0],
            joint_names,
        )
        if (
            base_from_tcp is None
            or base_from_wrist_camera is None
            or finger_force_n is None
            or previous_command_q_rad is None
        ):
            raise EpisodeRejected("可信采集必须提供完整 Observation V2 deployable state")
        tcp_position, tcp_rotation = transform_to_position_rotation_6d(base_from_tcp)
        wrist_position, wrist_rotation = transform_to_position_rotation_6d(
            base_from_wrist_camera
        )
        finger_force = np.asarray(finger_force_n)
        if (
            finger_force.shape != (2,)
            or finger_force.dtype != np.float32
            or not np.isfinite(finger_force).all()
            or np.any(finger_force < 0.0)
        ):
            raise EpisodeRejected("F_L/F_R 必须是有限非负 float32 [2]，单位 N")
        previous_command = np.asarray(previous_command_q_rad)
        if (
            previous_command.shape != (self.spec.arm_dof,)
            or previous_command.dtype != np.float32
            or not np.isfinite(previous_command).all()
        ):
            raise EpisodeRejected("previous_command_q_rad 必须是有限 float32 Franka q")
        previous_action = (
            np.zeros(self.spec.action_dim, dtype=np.float32)
            if not self.action
            else np.asarray(self.action[-1], dtype=np.float32).copy()
        )

        if self.precision_label_recorder is not None:
            if object_position_base_m is None or goal_position_base_m is None:
                raise EpisodeRejected("Precision label 采集必须提供 object/goal robot-base pose")
            try:
                self.precision_label_recorder.record(
                    observation,
                    timestep=len(self.action),
                    timestamp_s=len(self.action) / self.spec.control_hz,
                    base_from_wrist_camera_cv=base_from_wrist_camera,
                    object_position_base_m=object_position_base_m,
                    goal_position_base_m=goal_position_base_m,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise EpisodeRejected(f"Precision privileged label 无效: {error}") from error

        external_visible = self._goal_visible(external)
        wrist_visible = self._goal_visible(wrist)
        self.external_goal_visible_steps += int(external_visible)
        self.wrist_goal_visible_steps += int(wrist_visible)
        self.both_goal_visible_steps += int(external_visible and wrist_visible)
        self.rgb_external.append(external_rgb.copy())
        self.rgb_wrist.append(wrist_rgb.copy())
        self.proprio.append(proprio.copy())
        self.action.append(np.asarray(physical_action, dtype=np.float32).copy())
        self.skill_id.append(skill_id)
        self.robot_object_contact_force_n.append(robot_object_contact_force_n)
        self.support_contact_force_n.append(support_contact_force_n)
        self.is_grasped.append(predicate_state.is_grasped)
        self.object_position_m.append(
            np.asarray(predicate_state.object_position, dtype=np.float32)
        )
        self.object_linear_velocity_m_s.append(
            np.asarray(predicate_state.object_linear_velocity, dtype=np.float32)
        )
        self.object_angular_velocity_rad_s.append(
            np.asarray(predicate_state.object_angular_velocity, dtype=np.float32)
        )
        self.commanded_joint_target_rad.append(
            np.asarray(commanded_joint_target_rad, dtype=np.float32).copy()
        )
        self.applied_joint_correction_rad.append(
            np.asarray(applied_joint_correction_rad, dtype=np.float32).copy()
        )
        self.tcp_position_base_m.append(tcp_position)
        self.tcp_rotation_6d_base.append(tcp_rotation)
        self.wrist_camera_position_base_m.append(wrist_position)
        self.wrist_camera_rotation_6d_base.append(wrist_rotation)
        self.left_finger_force_n.append(float(finger_force[0]))
        self.right_finger_force_n.append(float(finger_force[1]))
        self.previous_command_q_rad.append(previous_command.copy())
        self.previous_action.append(previous_action)
        if self.record_action_provenance:
            if action_source not in {ACTION_SOURCE_POLICY, ACTION_SOURCE_EXPERT}:
                raise ValueError("action_source 必须是 Policy 或 Expert")
            self.action_source.append(int(action_source))
            self.expert_supervision_mask.append(action_source == ACTION_SOURCE_EXPERT)
        self._pending_transition = True

    def _goal_visible(self, sensor_data: dict[str, Any]) -> bool:
        if "segmentation" not in sensor_data:
            raise EpisodeRejected("可信采集必须启用 segmentation 以审计目标可见性")
        segmentation = _numpy(sensor_data["segmentation"])[0, ..., 0]
        return bool(np.any(segmentation == self.goal_actor_id))

    def record_after_action(self, terminated: Any, truncated: Any, info: dict[str, Any]) -> None:
        if not self._pending_transition:
            raise RuntimeError("必须先记录 Action 前观测")
        self.terminated.append(_single_bool(terminated))
        self.truncated.append(_single_bool(truncated))
        self.success.append(_single_bool(info["success"]))
        self._pending_transition = False

    def build(self) -> TrajectoryArrays:
        if self._pending_transition or not self.action:
            raise EpisodeRejected("Episode 为空或存在未完成 Transition")
        steps = len(self.action)
        timestamp = np.arange(steps, dtype=np.float64) / self.spec.control_hz
        valid = np.ones(steps, dtype=np.bool_)
        previous_action_valid = valid.copy()
        previous_action_valid[0] = False
        return TrajectoryArrays(
            rgb_external=np.stack(self.rgb_external).astype(np.uint8, copy=False),
            rgb_wrist=np.stack(self.rgb_wrist).astype(np.uint8, copy=False),
            timestamp_external=timestamp.copy(),
            timestamp_wrist=timestamp.copy(),
            timestamp_proprio=timestamp.copy(),
            timestamp_action=timestamp.copy(),
            proprio=np.stack(self.proprio).astype(np.float32, copy=False),
            action=np.stack(self.action).astype(np.float32, copy=False),
            external_valid=valid.copy(),
            wrist_valid=valid.copy(),
            proprio_valid=valid.copy(),
            terminated=np.asarray(self.terminated, dtype=np.bool_),
            truncated=np.asarray(self.truncated, dtype=np.bool_),
            success=np.asarray(self.success, dtype=np.bool_),
            skill_id=np.asarray(self.skill_id, dtype=np.int16),
            robot_object_contact_force_n=np.asarray(
                self.robot_object_contact_force_n,
                dtype=np.float32,
            ),
            support_contact_force_n=np.asarray(
                self.support_contact_force_n,
                dtype=np.float32,
            ),
            is_grasped=np.asarray(self.is_grasped, dtype=np.bool_),
            object_position_m=np.stack(self.object_position_m).astype(
                np.float32,
                copy=False,
            ),
            object_linear_velocity_m_s=np.stack(
                self.object_linear_velocity_m_s
            ).astype(np.float32, copy=False),
            object_angular_velocity_rad_s=np.stack(
                self.object_angular_velocity_rad_s
            ).astype(np.float32, copy=False),
            commanded_joint_target_rad=np.stack(
                self.commanded_joint_target_rad
            ).astype(np.float32, copy=False),
            applied_joint_correction_rad=np.stack(
                self.applied_joint_correction_rad
            ).astype(np.float32, copy=False),
            action_source=(
                np.asarray(self.action_source, dtype=np.int8)
                if self.record_action_provenance
                else None
            ),
            expert_supervision_mask=(
                np.asarray(self.expert_supervision_mask, dtype=np.bool_)
                if self.record_action_provenance
                else None
            ),
            timestamp_tcp_pose=timestamp.copy(),
            timestamp_camera_pose=timestamp.copy(),
            timestamp_finger_force=timestamp.copy(),
            tcp_position_base_m=np.stack(self.tcp_position_base_m).astype(
                np.float32,
                copy=False,
            ),
            tcp_rotation_6d_base=np.stack(self.tcp_rotation_6d_base).astype(
                np.float32,
                copy=False,
            ),
            wrist_camera_position_base_m=np.stack(
                self.wrist_camera_position_base_m
            ).astype(np.float32, copy=False),
            wrist_camera_rotation_6d_base=np.stack(
                self.wrist_camera_rotation_6d_base
            ).astype(np.float32, copy=False),
            left_finger_force_n=np.asarray(self.left_finger_force_n, dtype=np.float32),
            right_finger_force_n=np.asarray(self.right_finger_force_n, dtype=np.float32),
            tcp_pose_valid=valid.copy(),
            camera_pose_valid=valid.copy(),
            finger_force_valid=valid.copy(),
            previous_command_q_rad=np.stack(self.previous_command_q_rad).astype(
                np.float32,
                copy=False,
            ),
            previous_action=np.stack(self.previous_action).astype(np.float32, copy=False),
            previous_command_valid=valid.copy(),
            previous_action_valid=previous_action_valid,
        )


@dataclass
class _CollectionSession:
    observation: dict[str, Any]
    tracker: PickPlaceTaskTracker
    progress: PickPlaceTaskProgress
    recorder: _EpisodeRecorder
    previous_command_q: np.ndarray
    done: bool = False
    after_action_hook: Callable[[_CollectionSession, float], None] | None = None


@dataclass(frozen=True)
class AtomicPreparation:
    observation: dict[str, Any]
    tracker: PickPlaceTaskTracker
    progress: PickPlaceTaskProgress
    preparation_steps: int


class TrustedPickPlaceCollector:
    """显式 reach→grasp→lift→transport→lower→release→settle 专家。"""

    def __init__(
        self,
        dataset_root: str | Path | None,
        spec: RobotSpec | None = None,
        *,
        max_episode_steps: int | None = None,
        precision_label_root: str | Path | None = None,
        shadow_observer: Any | None = None,
    ) -> None:
        self.spec = spec or RobotSpec()
        self.writer = (
            None if dataset_root is None else TrajectoryDatasetWriter(dataset_root, self.spec)
        )
        if precision_label_root is not None and dataset_root is None:
            raise ValueError("配置 Precision label root 时必须同时配置 deployable Dataset root")
        if precision_label_root is not None:
            deployable_root = Path(dataset_root).resolve()
            privileged_root = Path(precision_label_root).resolve()
            if (
                deployable_root == privileged_root
                or privileged_root.is_relative_to(deployable_root)
                or deployable_root.is_relative_to(privileged_root)
            ):
                raise ValueError(
                    "Precision privileged label root 必须与 deployable Dataset 使用独立 sibling root"
                )
        self.precision_label_writer = (
            None
            if precision_label_root is None
            else PrecisionLabelDatasetWriter(precision_label_root)
        )
        if shadow_observer is not None and (
            not callable(getattr(shadow_observer, "reset", None))
            or not callable(getattr(shadow_observer, "observe", None))
        ):
            raise TypeError("shadow_observer 必须提供 reset()/observe()")
        self.shadow_observer = shadow_observer
        self.shadow_observer_errors: list[dict[str, str]] = []
        self.observation_adapter = FrankaObservationAdapter(self.spec)
        self.action_adapter = ActionAdapter(self.spec)
        register_robot_vla_maniskill_envs()
        if max_episode_steps is None:
            self.env = gym.make(
                PICK_CUBE_TO_REGION_ENV_ID,
                obs_mode="rgb+segmentation",
                control_mode="pd_joint_delta_pos",
                num_envs=1,
            )
        else:
            if max_episode_steps <= 0:
                raise ValueError("max_episode_steps 必须为正整数")
            self.env = gym.make(
                PICK_CUBE_TO_REGION_ENV_ID,
                obs_mode="rgb+segmentation",
                control_mode="pd_joint_delta_pos",
                num_envs=1,
                max_episode_steps=max_episode_steps,
            )
        self.base_env = self.env.unwrapped
        self._planner: PandaArmMotionPlanningSolver | None = None

    def close(self) -> None:
        if self._planner is not None:
            self._planner.close()
            self._planner = None
        self.env.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def collect(
        self,
        *,
        seed: int,
        split: str,
        trajectory_id: str | None = None,
        instruction_index: int | None = None,
        recovery_profile: str | None = None,
    ) -> TrajectoryMeta:
        if seed < 0:
            raise ValueError("seed 不能为负数")
        if split not in {"train", "val", "test"}:
            raise ValueError("split 必须是 train/val/test")
        if recovery_profile is not None and recovery_profile not in RECOVERY_PROFILES:
            raise ValueError(f"未知 recovery_profile: {recovery_profile}")
        if self.writer is None:
            raise RuntimeError("未配置 dataset_root 的专家只能用于原子状态准备")
        self.shadow_observer_errors.clear()
        if self.shadow_observer is not None:
            try:
                self.shadow_observer.reset()
            except Exception as error:  # noqa: BLE001 - shadow 失败不得改变专家 Action
                self.shadow_observer_errors.append(
                    {"type": type(error).__name__, "message": str(error)}
                )
        session = self._start_session(seed)
        observation = session.observation
        calibration = self._camera_calibration(observation)
        cube_initial = _numpy(self.base_env.cube.pose.p)[0].copy()
        goal_position = _numpy(self.base_env.goal_site.pose.p)[0].copy()
        grasp_pose, reach_pose, lift_pose, transport_pose, lower_pose = self._phase_poses()
        recovery_direction = -1.0 if float(goal_position[1]) >= 0.0 else 1.0
        recovery_disturbance_step: int | None = None

        if recovery_profile == "reach":
            reach_detour = sapien.Pose(
                [
                    float(cube_initial[0]),
                    float(cube_initial[1] + recovery_direction * 0.08),
                    float(lift_pose.p[2]),
                ],
                grasp_pose.q,
            )
            self._move_to_pose(session, reach_detour, gripper_opening=1.0)
            if session.progress.completed_skill_count != 0:
                raise EpisodeRejected("reach 绕行没有形成预期的未到达状态")
            recovery_disturbance_step = len(session.recorder.action) - 1
        self._move_to_pose(session, reach_pose, gripper_opening=1.0)
        self._move_to_pose(session, grasp_pose, gripper_opening=1.0)
        if recovery_profile == "grasp":
            failed_grasp_pose = sapien.Pose(
                [
                    float(grasp_pose.p[0]),
                    float(grasp_pose.p[1] + recovery_direction * 0.06),
                    float(grasp_pose.p[2]),
                ],
                grasp_pose.q,
            )
            self._move_to_pose(session, failed_grasp_pose, gripper_opening=1.0)
            self._hold(session, gripper_opening=0.0, steps=4)
            if session.progress.completed_skill_count != 1 or session.progress.outcome.grasped:
                raise EpisodeRejected("偏位抓取没有形成预期的可恢复 grasp 失败状态")
            recovery_disturbance_step = len(session.recorder.action) - 1
            self._hold(session, gripper_opening=1.0, steps=4)
            self._move_to_pose(session, grasp_pose, gripper_opening=1.0)
        self._hold(session, gripper_opening=0.0, steps=8)
        if recovery_profile == "lift":
            self._hold(session, gripper_opening=1.0, steps=6)
            if session.progress.outcome.grasped:
                raise EpisodeRejected("lift 恢复扰动没有释放方块")
            if session.progress.completed_skill_count != 2:
                raise EpisodeRejected("lift 恢复扰动发生在错误的技能阶段")
            recovery_disturbance_step = len(session.recorder.action) - 1
            self._hold(session, gripper_opening=0.0, steps=8)
            if not session.progress.outcome.grasped:
                raise EpisodeRejected("lift 恢复扰动后没有重新抓住方块")
        self._move_to_pose(session, lift_pose, gripper_opening=0.0)
        if recovery_profile == "transport":
            transport_detour = sapien.Pose(
                [
                    float((cube_initial[0] + goal_position[0]) * 0.5),
                    float(
                        (cube_initial[1] + goal_position[1]) * 0.5
                        + recovery_direction * 0.08
                    ),
                    float(lift_pose.p[2]),
                ],
                grasp_pose.q,
            )
            self._move_to_pose(session, transport_detour, gripper_opening=0.0)
            if (
                session.progress.completed_skill_count != 3
                or session.progress.outcome.transported
            ):
                raise EpisodeRejected("transport 绕行没有形成预期的目标外状态")
            recovery_disturbance_step = len(session.recorder.action) - 1
        self._move_to_pose(session, transport_pose, gripper_opening=0.0)
        if recovery_profile == "place":
            failed_lower_pose = sapien.Pose(
                [
                    float(goal_position[0]),
                    float(goal_position[1] + recovery_direction * 0.08),
                    float(goal_position[2] + 0.005),
                ],
                grasp_pose.q,
            )
            self._move_to_pose(session, failed_lower_pose, gripper_opening=0.0)
            self._hold(session, gripper_opening=1.0, steps=8)
            if session.done or session.progress.outcome.grasped:
                raise EpisodeRejected("目标外释放没有形成预期的可恢复 place 失败状态")
            recovery_disturbance_step = len(session.recorder.action) - 1
            self._hold(session, gripper_opening=0.0, steps=8)
            if not session.progress.outcome.grasped:
                raise EpisodeRejected("place 恢复扰动后没有重新抓住方块")
            recovery_lift_pose = sapien.Pose(
                [
                    float(failed_lower_pose.p[0]),
                    float(failed_lower_pose.p[1]),
                    float(lift_pose.p[2]),
                ],
                grasp_pose.q,
            )
            self._move_to_pose(session, recovery_lift_pose, gripper_opening=0.0)
        self._move_to_pose(session, lower_pose, gripper_opening=0.0)
        self._hold(session, gripper_opening=1.0, steps=30, stop_on_success=True)
        if not session.done or not session.progress.task_completed:
            raise EpisodeRejected("释放和 settle 后仍未同时满足环境与项目 Predicate 成功")

        arrays = session.recorder.build()
        if recovery_profile is not None and recovery_disturbance_step is None:
            raise EpisodeRejected("恢复轨迹缺少扰动状态证据")
        outcome = session.progress.outcome
        default_trajectory_id = (
            f"pick-place-seed-{seed:06d}"
            if recovery_profile is None
            else f"pick-place-recovery-{recovery_profile}-seed-{seed:06d}"
        )
        resolved_trajectory_id = trajectory_id or default_trajectory_id
        meta = TrajectoryMeta(
            trajectory_id=resolved_trajectory_id,
            source_episode_id=f"maniskill-seed-{seed:06d}",
            file=f"trajectories/{resolved_trajectory_id}.npz",
            split=split,
            scene_id=f"{PICK_CUBE_TO_REGION_ENV_ID}:seed={seed}",
            task=build_pick_place_task(
                seed % 3 if instruction_index is None else instruction_index
            ),
            num_steps=arrays.num_steps,
            camera_calibration=calibration,
            randomization={
                "seed": seed,
                "environment_id": PICK_CUBE_TO_REGION_ENV_ID,
                "control_mode": "pd_joint_delta_pos",
                "event_state_contract_version": EVENT_STATE_CONTRACT_VERSION,
                "observation_contract_version": OBSERVATION_V2_VERSION,
                "finger_force_sensor_version": FINGER_FORCE_SENSOR_VERSION,
                "cube_initial_position_m": cube_initial.tolist(),
                "goal_position_m": goal_position.tolist(),
                "recovery_profile": recovery_profile,
                "recovery_contract_version": (
                    RECOVERY_CONTRACT_VERSION if recovery_profile is not None else None
                ),
                "recovery_evidence": (
                    None
                    if recovery_disturbance_step is None
                    else {
                        "disturbance_end_step": recovery_disturbance_step,
                        "successful_recovery_end_step": arrays.num_steps - 1,
                    }
                ),
            },
            outcome_evidence=OutcomeEvidence(
                predicate_version=session.tracker.config.version,
                task_completed=session.progress.task_completed,
                final_is_released=not outcome.grasped,
                stable_place_steps=session.progress.stable_place_steps,
                external_goal_visible_steps=(session.recorder.external_goal_visible_steps),
                wrist_goal_visible_steps=session.recorder.wrist_goal_visible_steps,
                both_goal_visible_steps=session.recorder.both_goal_visible_steps,
                final_object_to_goal_distance_m=outcome.object_to_goal_distance_m,
                final_object_linear_speed_m_s=outcome.object_linear_speed_m_s,
                final_object_angular_speed_rad_s=outcome.object_angular_speed_rad_s,
            ),
        )
        trajectory_path = self.writer.write(meta, arrays)
        if self.precision_label_writer is not None:
            label_recorder = session.recorder.precision_label_recorder
            if label_recorder is None:
                raise RuntimeError("Precision label writer 已配置但 recorder 缺失")
            label_arrays = label_recorder.build()
            label_meta = build_precision_label_meta(meta, trajectory_path)
            self.precision_label_writer.write(label_meta, label_arrays)
        return meta

    def prepare_atomic(self, *, seed: int, skill_name: str) -> AtomicPreparation:
        """用可信专家完成目标原子技能之前的全部前置阶段。"""

        if skill_name not in {definition.name for definition in ATOMIC_PICK_PLACE_SKILLS}:
            raise ValueError(f"未知原子技能: {skill_name}")
        session = self._start_session(seed)
        grasp_pose, reach_pose, lift_pose, transport_pose, lower_pose = self._phase_poses()
        target_skill_id = next(
            definition.skill_id
            for definition in ATOMIC_PICK_PLACE_SKILLS
            if definition.name == skill_name
        )
        if target_skill_id >= 1:
            self._move_to_pose(session, reach_pose, gripper_opening=1.0)
            self._move_to_pose(session, grasp_pose, gripper_opening=1.0)
        if target_skill_id >= 2:
            self._hold(session, gripper_opening=0.0, steps=8)
        if target_skill_id >= 3:
            self._move_to_pose(session, lift_pose, gripper_opening=0.0)
        if target_skill_id >= 4:
            self._move_to_pose(session, transport_pose, gripper_opening=0.0)
            self._move_to_pose(session, lower_pose, gripper_opening=0.0)
        if session.progress.completed_skill_count != target_skill_id:
            raise EpisodeRejected(
                "原子技能前置状态不满足精确阶段："
                f"目标={target_skill_id}，实际={session.progress.completed_skill_count}"
            )
        return AtomicPreparation(
            observation=session.observation,
            tracker=session.tracker,
            progress=session.progress,
            preparation_steps=len(session.recorder.action),
        )

    def _start_session(
        self,
        seed: int,
        *,
        record_action_provenance: bool = False,
    ) -> _CollectionSession:
        if seed < 0:
            raise ValueError("seed 不能为负数")
        observation, _ = self.env.reset(seed=seed)
        if self._planner is None:
            self._planner = PandaArmMotionPlanningSolver(
                self.env,
                debug=False,
                vis=False,
                base_pose=self.base_env.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_vel_limits=0.4,
                joint_acc_limits=0.4,
            )

        tracker = PickPlaceTaskTracker()
        progress = tracker.update(self._read_predicate_state())
        robot = self.base_env.agent.robot
        current_q = _numpy(robot.get_qpos())[0, : self.spec.arm_dof].astype(np.float32)
        goal_actor_id = int(_numpy(self.base_env.goal_site.per_scene_id).reshape(-1)[0])
        object_actor_id = int(_numpy(self.base_env.cube.per_scene_id).reshape(-1)[0])
        session = _CollectionSession(
            observation=observation,
            tracker=tracker,
            progress=progress,
            recorder=_EpisodeRecorder(
                spec=self.spec,
                observation_adapter=self.observation_adapter,
                robot=robot,
                goal_actor_id=goal_actor_id,
                precision_label_recorder=(
                    None
                    if self.precision_label_writer is None
                    else PrecisionLabelRecorder(
                        object_actor_id=object_actor_id,
                        goal_actor_id=goal_actor_id,
                    )
                ),
                record_action_provenance=record_action_provenance,
            ),
            previous_command_q=current_q.copy(),
        )
        return session

    def _phase_poses(self) -> tuple[sapien.Pose, ...]:
        cube_initial = _numpy(self.base_env.cube.pose.p)[0].copy()
        goal_position = _numpy(self.base_env.goal_site.pose.p)[0].copy()
        grasp_pose = self._build_grasp_pose()
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        safe_height = max(float(cube_initial[2]), float(goal_position[2])) + 0.12
        lift_pose = sapien.Pose(
            [float(cube_initial[0]), float(cube_initial[1]), safe_height],
            grasp_pose.q,
        )
        transport_pose = sapien.Pose(
            [float(goal_position[0]), float(goal_position[1]), safe_height],
            grasp_pose.q,
        )
        lower_pose = sapien.Pose(
            [
                float(goal_position[0]),
                float(goal_position[1]),
                float(goal_position[2]) + 0.005,
            ],
            grasp_pose.q,
        )
        return grasp_pose, reach_pose, lift_pose, transport_pose, lower_pose

    def _build_grasp_pose(self) -> sapien.Pose:
        obb = get_actor_obb(self.base_env.cube)
        approaching = np.asarray([0.0, 0.0, -1.0])
        target_closing = (
            self.base_env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1]
            .cpu()
            .numpy()
        )
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=0.025,
        )
        return self.base_env.agent.build_grasp_pose(
            approaching,
            grasp_info["closing"],
            self.base_env.cube.pose.sp.p,
        )

    def _move_to_pose(
        self,
        session: _CollectionSession,
        pose: sapien.Pose,
        *,
        gripper_opening: float,
    ) -> None:
        if self._planner is None:
            raise RuntimeError("motion planner 尚未初始化")
        result = self._planner.move_to_pose_with_screw(pose, dry_run=True)
        if isinstance(result, int) or result.get("status") != "Success":
            raise EpisodeRejected("MPlib 无法规划可信 screw 路径")
        positions = np.asarray(result["position"], dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != self.spec.arm_dof:
            raise EpisodeRejected(f"MPlib 关节路径 shape 无效: {positions.shape}")
        for target_q in positions:
            self._execute_target(session, target_q, gripper_opening)

    def _hold(
        self,
        session: _CollectionSession,
        *,
        gripper_opening: float,
        steps: int,
        stop_on_success: bool = False,
    ) -> None:
        for _ in range(steps):
            if session.done:
                return
            self._execute_target(
                session,
                session.previous_command_q.copy(),
                gripper_opening,
            )
            if stop_on_success and session.done:
                return

    def _execute_target(
        self,
        session: _CollectionSession,
        target_q: np.ndarray,
        gripper_opening: float,
    ) -> None:
        if session.done:
            raise EpisodeRejected("环境终止后仍尝试执行 Action")
        target = np.asarray(target_q, dtype=np.float32)
        label = np.empty(self.spec.action_dim, dtype=np.float32)
        label[: self.spec.arm_dof] = target - session.previous_command_q
        label[-1] = gripper_opening
        try:
            self.action_adapter.normalize(label, strict=True)
        except ValueError as exc:
            raise EpisodeRejected("规划路径的相邻 q target 增量超出 D017 限制") from exc

        # pd_joint_delta_pos 可能比规划目标滞后一拍。若新目标 correction 将超限，先明确
        # 执行零 Label 的上一命令 settle；不裁剪 correction，也不篡改训练 Label。
        limits = np.asarray(self.spec.effective_joint_delta_limits_rad, dtype=np.float32)
        for _ in range(3):
            actual_q = self._actual_arm_q()
            if np.all(np.abs(target - actual_q) <= limits + 1e-6):
                break
            previous_correction = session.previous_command_q - actual_q
            if np.any(np.abs(previous_correction) > limits + 1e-6):
                raise EpisodeRejected("实际跟踪误差使上一 q target correction 超出安全限制")
            settle_label = np.zeros(self.spec.action_dim, dtype=np.float32)
            settle_label[-1] = gripper_opening
            self._step_with_target(
                session,
                session.previous_command_q,
                settle_label,
                gripper_opening,
            )
        else:
            raise EpisodeRejected("显式 settle 后 controller correction 仍超出安全限制")

        self._step_with_target(session, target, label, gripper_opening)
        session.previous_command_q = target.copy()

    def _step_with_target(
        self,
        session: _CollectionSession,
        target_q: np.ndarray,
        label: np.ndarray,
        gripper_opening: float,
    ) -> None:
        actual_q = self._actual_arm_q()
        controller_physical = np.empty(self.spec.action_dim, dtype=np.float32)
        controller_physical[: self.spec.arm_dof] = target_q - actual_q
        controller_physical[-1] = gripper_opening
        try:
            controller_action = self.action_adapter.to_maniskill(controller_physical)
        except ValueError as exc:
            raise EpisodeRejected("实际跟踪误差使 controller correction 超出安全限制") from exc

        contact_forces = self._read_contact_forces()
        base_from_tcp, base_from_wrist_camera = self._read_observation_v2_poses(
            session.observation
        )
        object_position_base: np.ndarray | None = None
        goal_position_base: np.ndarray | None = None
        if session.recorder.precision_label_recorder is not None:
            # Simulator GT 只允许进入离线 privileged sidecar 采集分支；
            # baseline/shadow 路径不读取，也不会接触 object/goal GT pose。
            object_position_base, goal_position_base = self._read_precision_positions_base()
        session.recorder.record_before_action(
            session.observation,
            label,
            session.progress.active_skill_id,
            self._read_predicate_state(),
            *contact_forces,
            target_q,
            controller_physical[: self.spec.arm_dof],
            base_from_tcp=base_from_tcp,
            base_from_wrist_camera=base_from_wrist_camera,
            finger_force_n=self._last_finger_force_n.copy(),
            previous_command_q_rad=session.previous_command_q.copy(),
            object_position_base_m=object_position_base,
            goal_position_base_m=goal_position_base,
        )
        self._observe_shadow(session, base_from_tcp, base_from_wrist_camera)
        observation, _, terminated, truncated, info = self.env.step(controller_action)
        session.recorder.record_after_action(terminated, truncated, info)
        session.observation = observation
        session.progress = session.tracker.update(self._read_predicate_state())
        if session.after_action_hook is not None:
            session.after_action_hook(session, gripper_opening)

        was_terminated = _single_bool(terminated)
        was_truncated = _single_bool(truncated)
        success = _single_bool(info["success"])
        if was_truncated:
            raise EpisodeRejected("Episode 在可信成功前达到时间上限")
        if was_terminated:
            if not success or not session.progress.task_completed:
                raise EpisodeRejected("环境终止与项目 Outcome Predicate 不一致")
            session.done = True

    def _observe_shadow(
        self,
        session: _CollectionSession,
        base_from_tcp: np.ndarray,
        base_from_wrist_camera: np.ndarray,
    ) -> None:
        """向无返回值 Observer 发送部署输入；异常只能写诊断，不能阻断专家。"""

        if self.shadow_observer is None:
            return
        recorder = session.recorder
        timestep = len(recorder.action) - 1
        timestamp = timestep / self.spec.control_hz
        previous_action = None if timestep == 0 else recorder.previous_action[-1].copy()
        frame = ObservationV2Frame(
            rgb_external=recorder.rgb_external[-1],
            rgb_wrist=recorder.rgb_wrist[-1],
            physical_proprio=recorder.proprio[-1],
            base_from_tcp=base_from_tcp,
            base_from_wrist_camera=base_from_wrist_camera,
            finger_force_n=np.asarray(
                (
                    recorder.left_finger_force_n[-1],
                    recorder.right_finger_force_n[-1],
                ),
                dtype=np.float32,
            ),
            timestamp_s=timestamp,
            modality_timestamp_s=np.full(
                len(OBSERVATION_MODALITIES),
                timestamp,
                dtype=np.float64,
            ),
            modality_valid=np.ones(len(OBSERVATION_MODALITIES), dtype=np.bool_),
        )
        try:
            self.shadow_observer.observe(
                frame,
                previous_command_q=recorder.previous_command_q_rad[-1].copy(),
                previous_action=previous_action,
            )
        except Exception as error:  # noqa: BLE001 - shadow 失败不得改变专家 Action
            self.shadow_observer_errors.append(
                {"type": type(error).__name__, "message": str(error)}
            )

    def _actual_arm_q(self) -> np.ndarray:
        return _numpy(self.base_env.agent.robot.get_qpos())[
            0, : self.spec.arm_dof
        ].astype(np.float32)

    def _read_contact_forces(self) -> tuple[float, float]:
        scene = self.base_env.scene
        agent = self.base_env.agent
        cube = self.base_env.cube
        left = _numpy(scene.get_pairwise_contact_forces(agent.finger1_link, cube))[0]
        right = _numpy(scene.get_pairwise_contact_forces(agent.finger2_link, cube))[0]
        support = _numpy(
            scene.get_pairwise_contact_forces(cube, self.base_env.table_scene.table)
        )[0]
        self._last_finger_force_n = np.asarray(
            (float(np.linalg.norm(left)), float(np.linalg.norm(right))),
            dtype=np.float32,
        )
        robot_object_force = float(np.max(self._last_finger_force_n))
        return robot_object_force, float(np.linalg.norm(support))

    def _read_observation_v2_poses(
        self,
        observation: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回 base_from_tcp 与 base_from_wrist_cv，均来自同一控制 Tick。"""

        world_from_base = _single_transform_matrix(
            self.base_env.agent.robot.pose,
            "world_from_robot_base",
        )
        world_from_tcp = _single_transform_matrix(
            self.base_env.agent.tcp_pose,
            "world_from_tcp",
        )
        wrist_gl = _numpy(
            observation["sensor_param"]["hand_camera"]["cam2world_gl"]
        )
        if wrist_gl.shape == (1, 4, 4):
            wrist_gl = wrist_gl[0]
        world_from_wrist_cv = opengl_camera_to_opencv(wrist_gl)
        base_from_world = invert_se3(world_from_base, "world_from_robot_base")
        base_from_tcp = validate_se3(
            base_from_world @ world_from_tcp,
            "base_from_tcp",
        )
        base_from_wrist = validate_se3(
            base_from_world @ world_from_wrist_cv,
            "base_from_wrist_camera_cv",
        )
        return base_from_tcp.astype(np.float32), base_from_wrist.astype(np.float32)

    def _read_predicate_state(self) -> PickPlaceState:
        tcp = _numpy(self.base_env.agent.tcp_pose.p)[0]
        cube = _numpy(self.base_env.cube.pose.p)[0]
        goal = _numpy(self.base_env.goal_site.pose.p)[0]
        linear_velocity = _numpy(self.base_env.cube.linear_velocity)[0]
        angular_velocity = _numpy(self.base_env.cube.angular_velocity)[0]
        is_grasped = _single_bool(self.base_env.agent.is_grasping(self.base_env.cube))
        return PickPlaceState(
            tcp_position=tuple(float(value) for value in tcp),
            object_position=tuple(float(value) for value in cube),
            goal_position=tuple(float(value) for value in goal),
            object_linear_velocity=tuple(float(value) for value in linear_velocity),
            object_angular_velocity=tuple(float(value) for value in angular_velocity),
            support_center_z_m=float(self.base_env.cube_half_size),
            is_grasped=is_grasped,
        )

    def _read_precision_positions_base(self) -> tuple[np.ndarray, np.ndarray]:
        """仅供离线标签：把 simulator GT object/goal pose 转为 robot base frame。"""

        world_from_base = _single_transform_matrix(
            self.base_env.agent.robot.pose,
            "world_from_robot_base",
        )
        base_from_world = invert_se3(world_from_base, "world_from_robot_base")
        object_world = _numpy(self.base_env.cube.pose.p)[0]
        goal_world = _numpy(self.base_env.goal_site.pose.p)[0]

        def transform(position: np.ndarray) -> np.ndarray:
            homogeneous = np.concatenate(
                (np.asarray(position, dtype=np.float64), np.ones(1, dtype=np.float64))
            )
            return (base_from_world @ homogeneous)[:3].astype(np.float32)

        return transform(object_world), transform(goal_world)

    def _camera_calibration(self, observation: dict[str, Any]) -> CameraCalibration:
        params = observation["sensor_param"]
        external = params["base_camera"]
        wrist = params["hand_camera"]
        intrinsic_external = _numpy(external["intrinsic_cv"])[0]
        intrinsic_wrist = _numpy(wrist["intrinsic_cv"])[0]
        world_from_external = _numpy(external["cam2world_gl"])[0]
        world_from_wrist = _numpy(wrist["cam2world_gl"])[0]
        world_from_tcp = _numpy(
            self.base_env.agent.tcp_pose.to_transformation_matrix()
        )[0]
        tcp_from_wrist = np.linalg.inv(world_from_tcp) @ world_from_wrist
        return CameraCalibration(
            version=CALIBRATION_VERSION,
            intrinsic_external=tuple(float(value) for value in intrinsic_external.ravel()),
            intrinsic_wrist=tuple(float(value) for value in intrinsic_wrist.ravel()),
            world_from_external=tuple(float(value) for value in world_from_external.ravel()),
            tcp_from_wrist=tuple(float(value) for value in tcp_from_wrist.ravel()),
        )


__all__ = [
    "RECOVERY_CONTRACT_VERSION",
    "RECOVERY_PROFILES",
    "AtomicPreparation",
    "EpisodeRejected",
    "TrustedPickPlaceCollector",
]
