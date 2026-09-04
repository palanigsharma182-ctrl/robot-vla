"""只在采集进程中存在的 privileged Precision label recorder。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_vla.observation import validate_se3
from robot_vla.precision.data import PrecisionLabelArrays, project_label_keypoints


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass
class PrecisionLabelRecorder:
    """记录 GT segmentation/pose；该对象禁止进入 Runtime Provider。"""

    object_actor_id: int
    goal_actor_id: int
    source_timestep: list[int] = field(default_factory=list)
    timestamp_s: list[float] = field(default_factory=list)
    object_mask: list[np.ndarray] = field(default_factory=list)
    goal_mask: list[np.ndarray] = field(default_factory=list)
    normalized_uv: list[np.ndarray] = field(default_factory=list)
    keypoint_visible: list[np.ndarray] = field(default_factory=list)
    keypoint_projection_valid: list[np.ndarray] = field(default_factory=list)
    object_position_base_m: list[np.ndarray] = field(default_factory=list)
    goal_position_base_m: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        for value, name in (
            (self.object_actor_id, "object_actor_id"),
            (self.goal_actor_id, "goal_actor_id"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} 必须是非负整数")
        if self.object_actor_id == self.goal_actor_id:
            raise ValueError("object/goal actor id 不能相同")

    def record(
        self,
        observation: dict[str, Any],
        *,
        timestep: int,
        timestamp_s: float,
        base_from_wrist_camera_cv: np.ndarray,
        object_position_base_m: np.ndarray,
        goal_position_base_m: np.ndarray,
    ) -> None:
        if timestep != len(self.source_timestep):
            raise ValueError("Precision label timestep 必须从 0 连续递增")
        if not np.isfinite(timestamp_s) or timestamp_s < 0.0:
            raise ValueError("Precision label timestamp_s 必须有限非负")
        sensor_data = observation["sensor_data"]["hand_camera"]
        sensor_param = observation["sensor_param"]["hand_camera"]
        if "segmentation" not in sensor_data:
            raise ValueError("Precision label 采集必须启用 wrist segmentation")
        segmentation = _numpy(sensor_data["segmentation"])
        if segmentation.ndim != 4 or segmentation.shape[0] != 1:
            raise ValueError("wrist segmentation 必须是单环境 [1,H,W,C]")
        actor_ids = segmentation[0, ..., 0]
        if actor_ids.ndim != 2 or not np.issubdtype(actor_ids.dtype, np.integer):
            raise ValueError("wrist segmentation actor channel 必须是整数 [H,W]")
        object_mask = actor_ids == self.object_actor_id
        goal_mask = actor_ids == self.goal_actor_id

        rgb = _numpy(sensor_data["rgb"])
        if rgb.shape[:1] != (1,) or rgb.shape[1:3] != actor_ids.shape:
            raise ValueError("wrist RGB 与 segmentation spatial shape 不一致")
        intrinsic = _numpy(sensor_param["intrinsic_cv"])
        if intrinsic.shape == (1, 3, 3):
            intrinsic = intrinsic[0]
        if intrinsic.shape != (3, 3):
            raise ValueError("wrist intrinsic_cv 必须是 [3,3]")
        base_from_camera = validate_se3(
            base_from_wrist_camera_cv,
            "base_from_wrist_camera_cv",
        )
        object_position = np.asarray(object_position_base_m, dtype=np.float32)
        goal_position = np.asarray(goal_position_base_m, dtype=np.float32)
        if (
            object_position.shape != (3,)
            or goal_position.shape != (3,)
            or not np.isfinite(object_position).all()
            or not np.isfinite(goal_position).all()
        ):
            raise ValueError("Precision object/goal base position 必须是有限 [3]")
        uv, projection_valid = project_label_keypoints(
            object_position,
            goal_position,
            intrinsic,
            base_from_camera,
            actor_ids.shape,
        )
        visible = np.asarray((object_mask.any(), goal_mask.any()), dtype=np.bool_)

        self.source_timestep.append(timestep)
        self.timestamp_s.append(float(timestamp_s))
        self.object_mask.append(np.ascontiguousarray(object_mask))
        self.goal_mask.append(np.ascontiguousarray(goal_mask))
        self.normalized_uv.append(uv)
        self.keypoint_visible.append(visible)
        self.keypoint_projection_valid.append(projection_valid)
        self.object_position_base_m.append(object_position.copy())
        self.goal_position_base_m.append(goal_position.copy())

    def build(self) -> PrecisionLabelArrays:
        if not self.source_timestep:
            raise ValueError("Precision label recorder 为空")
        return PrecisionLabelArrays(
            source_timestep=np.asarray(self.source_timestep, dtype=np.int64),
            timestamp_s=np.asarray(self.timestamp_s, dtype=np.float64),
            object_mask=np.stack(self.object_mask).astype(np.bool_, copy=False),
            goal_mask=np.stack(self.goal_mask).astype(np.bool_, copy=False),
            normalized_uv=np.stack(self.normalized_uv).astype(np.float32, copy=False),
            keypoint_visible=np.stack(self.keypoint_visible).astype(np.bool_, copy=False),
            keypoint_projection_valid=np.stack(self.keypoint_projection_valid).astype(
                np.bool_, copy=False
            ),
            object_position_base_m=np.stack(self.object_position_base_m).astype(
                np.float32,
                copy=False,
            ),
            goal_position_base_m=np.stack(self.goal_position_base_m).astype(
                np.float32,
                copy=False,
            ),
        )


__all__ = ["PrecisionLabelRecorder"]
