"""从可信 GT Action 和可选物理状态自动检测关键事件。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robot_vla.data.trajectory import TrajectoryArrays


EVENT_STATE_CONTRACT_VERSION = "trusted-pick-place-event-state/v1"
EVENT_DETECTION_VERSION = "pick-place-critical-events/v1"
EVENT_STATE_ARRAYS = (
    "robot_object_contact_force_n",
    "support_contact_force_n",
    "is_grasped",
    "object_position_m",
    "object_linear_velocity_m_s",
    "object_angular_velocity_rad_s",
    "commanded_joint_target_rad",
    "applied_joint_correction_rad",
)
EVENT_TYPES = (
    "grasp_command",
    "release_command",
    "contact",
    "linear_velocity_jump",
    "angular_velocity_jump",
    "pickup",
    "place",
)


@dataclass(frozen=True)
class EventDetectionConfig:
    """版本化物理阈值；正式训练前必须审计事件密度。"""

    gripper_state_threshold: float = 0.5
    contact_force_threshold_n: float = 0.5
    linear_velocity_jump_threshold_m_s: float | None = 0.05
    angular_velocity_jump_threshold_rad_s: float | None = 0.5
    version: str = EVENT_DETECTION_VERSION

    def __post_init__(self) -> None:
        if self.version != EVENT_DETECTION_VERSION:
            raise ValueError("event detection version 不兼容")
        if not 0.0 < self.gripper_state_threshold < 1.0:
            raise ValueError("gripper_state_threshold 必须位于 (0,1)")
        if not np.isfinite(self.contact_force_threshold_n) or self.contact_force_threshold_n <= 0:
            raise ValueError("contact_force_threshold_n 必须是有限正数")
        for name in (
            "linear_velocity_jump_threshold_m_s",
            "angular_velocity_jump_threshold_rad_s",
        ):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} 必须是有限正数或 None")


@dataclass(frozen=True)
class TrajectoryEventMasks:
    """每种事件和合并关键帧的 trajectory 全局 Action 索引 mask。"""

    event_mask: np.ndarray
    by_type: dict[str, np.ndarray]
    event_state_available: bool

    def __post_init__(self) -> None:
        if self.event_mask.ndim != 1 or self.event_mask.dtype != np.bool_:
            raise ValueError("event_mask 必须是一维 bool 数组")
        if set(self.by_type) != set(EVENT_TYPES):
            raise ValueError("by_type 事件类型不完整")
        for name, mask in self.by_type.items():
            if mask.shape != self.event_mask.shape or mask.dtype != np.bool_:
                raise ValueError(f"{name} event mask shape/dtype 无效")

    @property
    def counts(self) -> dict[str, int]:
        return {
            name: int(np.count_nonzero(mask))
            for name, mask in self.by_type.items()
        }


def detect_trajectory_events(
    arrays: TrajectoryArrays,
    config: EventDetectionConfig | None = None,
) -> TrajectoryEventMasks:
    """检测与 Action 时间轴严格对齐的 grasp/release/contact/pick/place 事件。"""

    config = config or EventDetectionConfig()
    steps = arrays.num_steps
    by_type = {
        name: np.zeros(steps, dtype=np.bool_)
        for name in EVENT_TYPES
    }

    gripper_target = arrays.action[:, -1]
    previous_target = np.empty_like(gripper_target)
    previous_target[0] = arrays.proprio[0, -1]
    previous_target[1:] = gripper_target[:-1]
    was_open = previous_target >= config.gripper_state_threshold
    is_open = gripper_target >= config.gripper_state_threshold
    by_type["grasp_command"] = was_open & ~is_open
    by_type["release_command"] = ~was_open & is_open

    event_state_available = arrays.event_state_available
    if event_state_available:
        contact = (
            arrays.robot_object_contact_force_n
            >= config.contact_force_threshold_n
        )
        contact_rise = np.zeros(steps, dtype=np.bool_)
        contact_rise[:-1] = ~contact[:-1] & contact[1:]
        by_type["contact"] = contact_rise

        support = arrays.support_contact_force_n >= config.contact_force_threshold_n
        pickup = np.zeros(steps, dtype=np.bool_)
        pickup[:-1] = support[:-1] & ~support[1:] & arrays.is_grasped[1:]
        by_type["pickup"] = pickup
        place = np.zeros(steps, dtype=np.bool_)
        place[:-1] = ~support[:-1] & support[1:] & ~arrays.is_grasped[1:]
        by_type["place"] = place

        if config.linear_velocity_jump_threshold_m_s is not None:
            linear_jump = np.zeros(steps, dtype=np.bool_)
            linear_jump[:-1] = (
                np.linalg.norm(np.diff(arrays.object_linear_velocity_m_s, axis=0), axis=1)
                >= config.linear_velocity_jump_threshold_m_s
            )
            by_type["linear_velocity_jump"] = linear_jump
        if config.angular_velocity_jump_threshold_rad_s is not None:
            angular_jump = np.zeros(steps, dtype=np.bool_)
            angular_jump[:-1] = (
                np.linalg.norm(np.diff(arrays.object_angular_velocity_rad_s, axis=0), axis=1)
                >= config.angular_velocity_jump_threshold_rad_s
            )
            by_type["angular_velocity_jump"] = angular_jump

    event_mask = np.zeros(steps, dtype=np.bool_)
    for mask in by_type.values():
        event_mask |= mask
    return TrajectoryEventMasks(
        event_mask=event_mask,
        by_type=by_type,
        event_state_available=event_state_available,
    )


__all__ = [
    "EVENT_DETECTION_VERSION",
    "EVENT_STATE_ARRAYS",
    "EVENT_STATE_CONTRACT_VERSION",
    "EVENT_TYPES",
    "EventDetectionConfig",
    "TrajectoryEventMasks",
    "detect_trajectory_events",
]
