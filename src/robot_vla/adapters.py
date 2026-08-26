"""Franka 原始状态、物理动作与模型表示之间的唯一转换边界。"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot_vla.contracts import (
    PROPRIO_STATS_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
    RobotSpec,
)


def _check_last_dim(value: np.ndarray, expected: int, name: str) -> None:
    if value.ndim == 0 or value.shape[-1] != expected:
        raise ValueError(f"{name} 最后一维应为 {expected}，实际为 {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} 包含 NaN 或 Inf")


class FrankaObservationAdapter:
    """把 ManiSkill Panda 9 个 active joint 状态转换为 15 维物理 proprioception。"""

    def __init__(self, spec: RobotSpec) -> None:
        self.spec = spec

    def from_maniskill(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        joint_names: tuple[str, ...] | list[str],
    ) -> np.ndarray:
        if tuple(joint_names) != self.spec.active_joint_names:
            raise ValueError(
                "ManiSkill active joint 顺序不兼容："
                f"期望 {self.spec.active_joint_names}，实际 {tuple(joint_names)}"
            )
        positions = np.asarray(qpos, dtype=np.float32)
        velocities = np.asarray(qvel, dtype=np.float32)
        _check_last_dim(positions, len(self.spec.active_joint_names), "qpos")
        _check_last_dim(velocities, len(self.spec.active_joint_names), "qvel")
        if positions.shape != velocities.shape:
            raise ValueError("qpos 与 qvel shape 必须相同")

        arm_q = positions[..., : self.spec.arm_dof]
        arm_dq = velocities[..., : self.spec.arm_dof]
        position_limits = np.asarray(self.spec.joint_position_limits_rad, dtype=np.float32)
        velocity_limits = np.asarray(self.spec.joint_velocity_limits_rad_s, dtype=np.float32)
        if np.any(arm_q < position_limits[:, 0] - 1e-5) or np.any(
            arm_q > position_limits[:, 1] + 1e-5
        ):
            raise ValueError("qpos 超出 Franka arm 关节位置限制")
        if np.any(np.abs(arm_dq) > velocity_limits + 1e-5):
            raise ValueError("qvel 超出 Franka arm 关节速度限制")

        finger_q = positions[..., self.spec.arm_dof :]
        lower, upper = self.spec.gripper_joint_position_range_m
        if np.any(finger_q < lower - 1e-6) or np.any(finger_q > upper + 1e-6):
            raise ValueError("Franka finger joint 位置超出标定范围")
        normalized_fingers = (finger_q - lower) / (upper - lower)
        gripper_opening = np.mean(normalized_fingers, axis=-1, keepdims=True)
        gripper_opening = np.clip(gripper_opening, 0.0, 1.0)
        return np.concatenate((arm_q, arm_dq, gripper_opening), axis=-1).astype(
            np.float32,
            copy=False,
        )

    def gripper_joint_positions(self, opening_ratio: np.ndarray) -> np.ndarray:
        opening = np.asarray(opening_ratio, dtype=np.float32)
        if not np.isfinite(opening).all() or np.any(opening < 0.0) or np.any(opening > 1.0):
            raise ValueError("gripper opening ratio 必须是 [0,1] 内的有限值")
        lower, upper = self.spec.gripper_joint_position_range_m
        finger_position = lower + opening * (upper - lower)
        return np.stack((finger_position, finger_position), axis=-1)


@dataclass(frozen=True)
class ProprioStats:
    """只允许由训练 split 拟合的 15 维物理 proprioception 统计量。"""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    count: int
    version: str = PROPRIO_STATS_VERSION
    schema_version: str = TRAJECTORY_SCHEMA_VERSION
    embodiment: str = "maniskill-franka-panda-v1"

    def validate(self, spec: RobotSpec) -> None:
        if self.version != PROPRIO_STATS_VERSION:
            raise ValueError(f"不支持的 ProprioStats version: {self.version}")
        if self.schema_version != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(f"ProprioStats schema 不兼容: {self.schema_version}")
        if self.embodiment != spec.embodiment:
            raise ValueError(
                f"ProprioStats embodiment 应为 {spec.embodiment}，实际为 {self.embodiment}"
            )
        if len(self.mean) != spec.proprio_dim or len(self.std) != spec.proprio_dim:
            raise ValueError(
                f"ProprioStats 维度应为 {spec.proprio_dim}，"
                f"实际 mean={len(self.mean)}, std={len(self.std)}"
            )
        if self.count <= 0:
            raise ValueError("统计样本数必须为正数")
        if not np.isfinite(np.asarray(self.mean, dtype=np.float64)).all():
            raise ValueError("mean 包含 NaN 或 Inf")
        std = np.asarray(self.std, dtype=np.float64)
        if not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("std 必须是有限正数")

    @classmethod
    def fit(
        cls,
        batches: Iterable[np.ndarray],
        spec: RobotSpec,
        *,
        min_std: float = 1e-6,
    ) -> ProprioStats:
        if min_std <= 0:
            raise ValueError("min_std 必须为正数")
        total = np.zeros(spec.proprio_dim, dtype=np.float64)
        total_sq = np.zeros(spec.proprio_dim, dtype=np.float64)
        count = 0
        for batch in batches:
            values = np.asarray(batch, dtype=np.float64)
            _check_last_dim(values, spec.proprio_dim, "proprio")
            flat = values.reshape(-1, spec.proprio_dim)
            total += flat.sum(axis=0)
            total_sq += np.square(flat).sum(axis=0)
            count += flat.shape[0]
        if count == 0:
            raise ValueError("不能从空数据拟合 ProprioStats")
        mean = total / count
        variance = np.maximum(total_sq / count - np.square(mean), 0.0)
        std = np.maximum(np.sqrt(variance), min_std)
        stats = cls(
            mean=tuple(mean.tolist()),
            std=tuple(std.tolist()),
            count=count,
            embodiment=spec.embodiment,
        )
        stats.validate(spec)
        return stats

    def to_json(self, path: str | Path) -> None:
        payload = {
            "version": self.version,
            "schema_version": self.schema_version,
            "embodiment": self.embodiment,
            "mean": list(self.mean),
            "std": list(self.std),
            "count": self.count,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> ProprioStats:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            mean=tuple(float(value) for value in payload["mean"]),
            std=tuple(float(value) for value in payload["std"]),
            count=int(payload["count"]),
            version=str(payload["version"]),
            schema_version=str(payload["schema_version"]),
            embodiment=str(payload["embodiment"]),
        )


class ProprioNormalizer:
    def __init__(
        self,
        stats: ProprioStats,
        spec: RobotSpec,
        *,
        clip: float = 5.0,
    ) -> None:
        stats.validate(spec)
        if clip <= 0:
            raise ValueError("clip 必须为正数")
        self.spec = spec
        self.mean = np.asarray(stats.mean, dtype=np.float32)
        self.std = np.asarray(stats.std, dtype=np.float32)
        self.clip = float(clip)

    def normalize(self, value: np.ndarray) -> np.ndarray:
        physical = np.asarray(value, dtype=np.float32)
        _check_last_dim(physical, self.spec.proprio_dim, "proprio")
        normalized = (physical - self.mean) / self.std
        return np.clip(normalized, -self.clip, self.clip)

    def denormalize(self, value: np.ndarray) -> np.ndarray:
        normalized = np.asarray(value, dtype=np.float32)
        _check_last_dim(normalized, self.spec.proprio_dim, "normalized proprio")
        return normalized * self.std + self.mean


@dataclass(frozen=True)
class FrankaCommandSequence:
    joint_position_targets: np.ndarray
    gripper_opening_targets: np.ndarray


class ActionContractViolation(ValueError):
    """携带可序列化证据的 Action 安全契约异常。"""

    def __init__(self, message: str, *, kind: str, details: dict[str, object]) -> None:
        super().__init__(message)
        self.kind = kind
        self.details = details

    def to_diagnostic(self) -> dict[str, object]:
        return {"kind": self.kind, **self.details}


class ActionAdapter:
    """物理 Action、模型 Action 和 ManiSkill controller Action 的唯一转换点。"""

    def __init__(self, spec: RobotSpec) -> None:
        self.spec = spec
        self.delta_limits = np.asarray(
            spec.effective_joint_delta_limits_rad,
            dtype=np.float32,
        )

    def normalize(self, physical_action: np.ndarray, *, strict: bool = True) -> np.ndarray:
        physical = np.asarray(physical_action, dtype=np.float32)
        _check_last_dim(physical, self.spec.action_dim, "physical_action")
        normalized = np.empty_like(physical)
        normalized[..., : self.spec.arm_dof] = (
            physical[..., : self.spec.arm_dof] / self.delta_limits
        )
        normalized[..., -1] = physical[..., -1] * 2.0 - 1.0
        violation_mask = np.abs(normalized) > 1.0 + 1e-5
        if strict and np.any(violation_mask):
            arm_mask = violation_mask[..., : self.spec.arm_dof]
            gripper_mask = violation_mask[..., -1]
            raise ActionContractViolation(
                "物理 Action 超出 delta_q 或 gripper_target 契约限制",
                kind="physical_action_contract",
                details={
                    "physical_action": physical.tolist(),
                    "normalized_action": normalized.tolist(),
                    "arm_violation_indices": np.argwhere(arm_mask).tolist(),
                    "gripper_violation_indices": np.argwhere(gripper_mask).tolist(),
                    "joint_delta_limits_rad": self.delta_limits.tolist(),
                },
            )
        return np.clip(normalized, -1.0, 1.0)

    def denormalize(self, normalized_action: np.ndarray) -> np.ndarray:
        normalized = np.asarray(normalized_action, dtype=np.float32)
        _check_last_dim(normalized, self.spec.action_dim, "normalized_action")
        clipped = np.clip(normalized, -1.0, 1.0)
        physical = np.empty_like(clipped)
        physical[..., : self.spec.arm_dof] = (
            clipped[..., : self.spec.arm_dof] * self.delta_limits
        )
        physical[..., -1] = (clipped[..., -1] + 1.0) * 0.5
        return physical

    def to_maniskill(self, physical_action: np.ndarray) -> np.ndarray:
        physical = np.asarray(physical_action, dtype=np.float32)
        self.normalize(physical, strict=True)
        controller_action = np.empty_like(physical)
        controller_action[..., : self.spec.arm_dof] = (
            physical[..., : self.spec.arm_dof] / self.spec.maniskill_arm_delta_range_rad
        )
        controller_action[..., -1] = physical[..., -1] * 2.0 - 1.0
        if np.any(np.abs(controller_action) > 1.0 + 1e-5):
            raise ActionContractViolation(
                "转换后的 ManiSkill controller action 超出 [-1,1]",
                kind="maniskill_controller_action_contract",
                details={
                    "physical_action": physical.tolist(),
                    "controller_action": controller_action.tolist(),
                    "violation_indices": np.argwhere(
                        np.abs(controller_action) > 1.0 + 1e-5
                    ).tolist(),
                },
            )
        return np.clip(controller_action, -1.0, 1.0)

    def build_receding_horizon_commands(
        self,
        latest_observed_q: np.ndarray,
        physical_chunk: np.ndarray,
    ) -> FrankaCommandSequence:
        q_base = np.asarray(latest_observed_q, dtype=np.float32)
        if q_base.shape != (self.spec.arm_dof,) or not np.isfinite(q_base).all():
            raise ValueError(f"latest_observed_q 应为 [{self.spec.arm_dof}] 有限向量")
        chunk = np.asarray(physical_chunk, dtype=np.float32)
        expected = (self.spec.action_horizon, self.spec.action_dim)
        if chunk.shape != expected:
            raise ValueError(f"physical_chunk 应为 {expected}，实际 {chunk.shape}")
        self.normalize(chunk, strict=True)

        position_limits = np.asarray(self.spec.joint_position_limits_rad, dtype=np.float32)
        if np.any(q_base < position_limits[:, 0]) or np.any(q_base > position_limits[:, 1]):
            raise ActionContractViolation(
                "latest_observed_q 超出 Franka 关节位置限制",
                kind="observed_joint_position_contract",
                details={
                    "observed_joint_positions_rad": q_base.tolist(),
                    "joint_position_limits_rad": position_limits.tolist(),
                    "violation_indices": np.argwhere(
                        (q_base < position_limits[:, 0])
                        | (q_base > position_limits[:, 1])
                    ).reshape(-1).tolist(),
                },
            )
        prefix = chunk[: self.spec.execute_steps]
        targets = q_base + np.cumsum(prefix[:, : self.spec.arm_dof], axis=0)
        if np.any(targets < position_limits[:, 0]) or np.any(targets > position_limits[:, 1]):
            raise ActionContractViolation(
                "Action Chunk 累加后的关节目标超出位置限制",
                kind="chunk_joint_position_contract",
                details={
                    "observed_joint_positions_rad": q_base.tolist(),
                    "physical_action_prefix": prefix.tolist(),
                    "joint_position_targets_rad": targets.tolist(),
                    "joint_position_limits_rad": position_limits.tolist(),
                    "violation_indices": np.argwhere(
                        (targets < position_limits[:, 0])
                        | (targets > position_limits[:, 1])
                    ).tolist(),
                },
            )
        return FrankaCommandSequence(
            joint_position_targets=targets,
            gripper_opening_targets=prefix[:, -1].copy(),
        )
