"""Franka 原始状态、物理动作与模型表示之间的唯一转换边界。"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot_vla.contracts import (
    FINGER_FORCE_SENSOR_VERSION,
    FINGER_FORCE_STATS_VERSION,
    OBSERVATION_V2_VERSION,
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
class FingerForceStats:
    """只从 train split 有效观测拟合的左右指力稳健尺度。

    零接触是有物理意义的原点，因此不做去中心化。先对牛顿值使用 ``log1p``，
    再分别除以各指正样本的 95% 分位数，避免少量碰撞尖峰支配尺度。
    """

    scale_log1p_p95: tuple[float, float]
    count: int
    positive_count: tuple[int, int]
    clip: float = 2.0
    quantile: float = 0.95
    version: str = FINGER_FORCE_STATS_VERSION
    observation_schema: str = OBSERVATION_V2_VERSION
    sensor_version: str = FINGER_FORCE_SENSOR_VERSION
    embodiment: str = "maniskill-franka-panda-v1"

    def validate(self, spec: RobotSpec) -> None:
        if self.version != FINGER_FORCE_STATS_VERSION:
            raise ValueError(f"不支持的 FingerForceStats version: {self.version}")
        if self.observation_schema != OBSERVATION_V2_VERSION:
            raise ValueError(
                f"FingerForceStats observation schema 不兼容: {self.observation_schema}"
            )
        if self.sensor_version != FINGER_FORCE_SENSOR_VERSION:
            raise ValueError(
                f"FingerForceStats sensor version 不兼容: {self.sensor_version}"
            )
        if self.embodiment != spec.embodiment:
            raise ValueError(
                f"FingerForceStats embodiment 应为 {spec.embodiment}，"
                f"实际为 {self.embodiment}"
            )
        scale = np.asarray(self.scale_log1p_p95, dtype=np.float64)
        if scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise ValueError("scale_log1p_p95 必须是两个有限正数")
        if self.count <= 0:
            raise ValueError("FingerForceStats count 必须为正数")
        positive_count = np.asarray(self.positive_count, dtype=np.int64)
        if (
            positive_count.shape != (2,)
            or np.any(positive_count <= 0)
            or np.any(positive_count > self.count)
        ):
            raise ValueError("positive_count 必须是两个位于 [1,count] 的整数")
        if not np.isfinite(self.clip) or self.clip <= 0.0:
            raise ValueError("FingerForceStats clip 必须是有限正数")
        if not np.isfinite(self.quantile) or not 0.5 < self.quantile < 1.0:
            raise ValueError("FingerForceStats quantile 必须位于 (0.5,1.0)")

    @classmethod
    def fit(
        cls,
        batches: Iterable[np.ndarray],
        spec: RobotSpec,
        *,
        quantile: float = 0.95,
        clip: float = 2.0,
    ) -> FingerForceStats:
        if not np.isfinite(quantile) or not 0.5 < quantile < 1.0:
            raise ValueError("quantile 必须位于 (0.5,1.0)")
        if not np.isfinite(clip) or clip <= 0.0:
            raise ValueError("clip 必须是有限正数")
        rows: list[np.ndarray] = []
        for batch in batches:
            values = np.asarray(batch, dtype=np.float64)
            _check_last_dim(values, 2, "finger_force_n")
            flat = values.reshape(-1, 2)
            if np.any(flat < 0.0):
                raise ValueError("finger_force_n 必须非负")
            rows.append(flat)
        if not rows:
            raise ValueError("不能从空数据拟合 FingerForceStats")
        force = np.concatenate(rows, axis=0)
        scales: list[float] = []
        positive_counts: list[int] = []
        for finger_index in range(2):
            positive = force[:, finger_index][force[:, finger_index] > 0.0]
            if positive.size == 0:
                side = "F_L" if finger_index == 0 else "F_R"
                raise ValueError(f"train split 的 {side} 没有正接触样本")
            transformed = np.log1p(positive)
            scale = float(np.quantile(transformed, quantile))
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError("FingerForceStats 稳健尺度退化")
            scales.append(scale)
            positive_counts.append(int(positive.size))
        stats = cls(
            scale_log1p_p95=(scales[0], scales[1]),
            count=int(force.shape[0]),
            positive_count=(positive_counts[0], positive_counts[1]),
            clip=float(clip),
            quantile=float(quantile),
            embodiment=spec.embodiment,
        )
        stats.validate(spec)
        return stats

    def to_json(self, path: str | Path) -> None:
        payload = {
            "version": self.version,
            "observation_schema": self.observation_schema,
            "sensor_version": self.sensor_version,
            "embodiment": self.embodiment,
            "scale_log1p_p95": list(self.scale_log1p_p95),
            "count": self.count,
            "positive_count": list(self.positive_count),
            "clip": self.clip,
            "quantile": self.quantile,
        }
        Path(path).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> FingerForceStats:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            scale_log1p_p95=tuple(float(value) for value in payload["scale_log1p_p95"]),
            count=int(payload["count"]),
            positive_count=tuple(int(value) for value in payload["positive_count"]),
            clip=float(payload["clip"]),
            quantile=float(payload["quantile"]),
            version=str(payload["version"]),
            observation_schema=str(payload["observation_schema"]),
            sensor_version=str(payload["sensor_version"]),
            embodiment=str(payload["embodiment"]),
        )


class FingerForceNormalizer:
    """训练和在线运行共用的 F_L/F_R 版本化变换。"""

    def __init__(self, stats: FingerForceStats, spec: RobotSpec) -> None:
        stats.validate(spec)
        self.spec = spec
        self.stats = stats
        self.scale = np.asarray(stats.scale_log1p_p95, dtype=np.float32)
        self.clip = float(stats.clip)

    def normalize(self, value: np.ndarray) -> np.ndarray:
        force = np.asarray(value, dtype=np.float32)
        _check_last_dim(force, 2, "finger_force_n")
        if np.any(force < 0.0):
            raise ValueError("finger_force_n 必须非负")
        normalized = np.log1p(force) / self.scale
        return np.clip(normalized, 0.0, self.clip).astype(np.float32, copy=False)

    def denormalize(self, value: np.ndarray) -> np.ndarray:
        normalized = np.asarray(value, dtype=np.float32)
        _check_last_dim(normalized, 2, "normalized_finger_force")
        if np.any(normalized < 0.0) or np.any(normalized > self.clip + 1e-6):
            raise ValueError("normalized_finger_force 超出冻结 clip 范围")
        return np.expm1(normalized * self.scale).astype(np.float32, copy=False)


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
        previous_command_q: np.ndarray,
        physical_chunk: np.ndarray,
    ) -> FrankaCommandSequence:
        q_base = np.asarray(previous_command_q, dtype=np.float32)
        if q_base.shape != (self.spec.arm_dof,) or not np.isfinite(q_base).all():
            raise ValueError(f"previous_command_q 应为 [{self.spec.arm_dof}] 有限向量")
        chunk = np.asarray(physical_chunk, dtype=np.float32)
        expected = (self.spec.action_horizon, self.spec.action_dim)
        if chunk.shape != expected:
            raise ValueError(f"physical_chunk 应为 {expected}，实际 {chunk.shape}")
        self.normalize(chunk, strict=True)

        position_limits = np.asarray(self.spec.joint_position_limits_rad, dtype=np.float32)
        if np.any(q_base < position_limits[:, 0]) or np.any(q_base > position_limits[:, 1]):
            raise ActionContractViolation(
                "previous_command_q 超出 Franka 关节位置限制",
                kind="command_reference_joint_position_contract",
                details={
                    "previous_command_joint_positions_rad": q_base.tolist(),
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
                    "previous_command_joint_positions_rad": q_base.tolist(),
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
