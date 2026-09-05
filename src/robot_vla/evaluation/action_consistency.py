"""零参数 Action Chunk 一致性诊断；不参与动作仲裁。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Any

import numpy as np

from robot_vla.contracts import RobotSpec


ACTION_CONSISTENCY_CRITIC_VERSION = "action-consistency-critic/v0"


class ActionConsistencyStatus(str, Enum):
    """一次诊断是否具有可比较的历史 proposal。"""

    SCORED = "scored"
    WARMUP = "warmup"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class ActionConsistencyResult:
    """ActionConsistencyCritic 的结构化 shadow 诊断结果。"""

    version: str
    status: ActionConsistencyStatus
    reason: str | None
    episode_id: str
    current_origin_control_step: int
    previous_origin_control_step: int | None
    advance_steps: int | None
    overlap_steps: int
    arm_tce_mse_normalized: float | None
    gripper_tce_mse_normalized: float | None
    arm_acm_rms_normalized: float | None
    gripper_transition_rms_opening_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        """返回可由严格 JSON 编码器序列化的 payload。"""

        payload = asdict(self)
        payload["status"] = self.status.value
        for key, value in payload.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{key} 不是有限数值")
        return payload


class ActionConsistencyCritic:
    """比较相邻 raw policy proposal，并报告执行前缀动作幅值。

    该组件不含参数、阈值或安全决策。调用方必须在 temporal ensemble 前传入
    raw normalized Action Chunk，并在 Episode 或执行异常边界显式 reset。
    """

    def __init__(self, spec: RobotSpec) -> None:
        if not isinstance(spec, RobotSpec):
            raise TypeError("spec 必须是 RobotSpec")
        canonical = RobotSpec()
        expected = (
            canonical.arm_dof,
            canonical.action_dim,
            canonical.action_horizon,
            canonical.execute_steps,
        )
        actual = (spec.arm_dof, spec.action_dim, spec.action_horizon, spec.execute_steps)
        if actual != expected:
            raise ValueError(f"V0 critic 合同必须是 {expected}，实际为 {actual}")
        self.spec = spec
        self._episode_id: str | None = None
        self._previous_action: np.ndarray | None = None
        self._previous_origin_control_step: int | None = None
        self._warmup_reason: str | None = "no_previous_proposal"

    @staticmethod
    def _validate_episode_id(episode_id: str) -> str:
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id 必须是非空字符串")
        return episode_id

    @staticmethod
    def _validate_origin_control_step(origin_control_step: int) -> int:
        if isinstance(origin_control_step, bool) or not isinstance(
            origin_control_step, Integral
        ):
            raise ValueError("origin_control_step 必须是非负整数")
        origin = int(origin_control_step)
        if origin < 0:
            raise ValueError("origin_control_step 必须是非负整数")
        return origin

    @staticmethod
    def _validate_reason(reason: str) -> str:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason 必须是非空字符串")
        return reason

    def _validate_action(self, normalized_action: np.ndarray) -> np.ndarray:
        try:
            action = np.asarray(normalized_action, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("normalized_action 必须可转换为 float32 数组") from error
        expected = (self.spec.action_horizon, self.spec.action_dim)
        if action.shape != expected:
            raise ValueError(f"normalized_action shape 应为 {expected}，实际为 {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("normalized_action 必须全部为有限数值")
        if np.any(action < -1.0) or np.any(action > 1.0):
            raise ValueError("normalized_action 必须位于 [-1,1]")
        return action.copy()

    @staticmethod
    def _validate_gripper_opening(observed_gripper_opening: float) -> float:
        if isinstance(observed_gripper_opening, bool) or not isinstance(
            observed_gripper_opening, Real
        ):
            raise ValueError("observed_gripper_opening 必须是 [0,1] 内有限数值")
        opening = float(observed_gripper_opening)
        if not math.isfinite(opening) or not 0.0 <= opening <= 1.0:
            raise ValueError("observed_gripper_opening 必须是 [0,1] 内有限数值")
        return opening

    def _action_magnitude(
        self,
        action: np.ndarray,
        observed_gripper_opening: float,
    ) -> tuple[float, float]:
        prefix = action[: self.spec.execute_steps].astype(np.float64, copy=False)
        arm = prefix[:, : self.spec.arm_dof]
        arm_acm = float(np.sqrt(np.mean(np.square(arm), dtype=np.float64)))

        # Gripper action 是 absolute target；幅值必须衡量相邻 target 的变化。
        observed_normalized = 2.0 * observed_gripper_opening - 1.0
        gripper_targets = prefix[:, self.spec.arm_dof]
        gripper_path = np.concatenate(
            (np.asarray([observed_normalized], dtype=np.float64), gripper_targets)
        )
        gripper_delta = np.diff(gripper_path)
        gripper_acm = float(
            np.sqrt(np.mean(np.square(gripper_delta), dtype=np.float64)) / 2.0
        )
        return arm_acm, gripper_acm

    def _store(self, action: np.ndarray, origin_control_step: int) -> None:
        # 独立复制，避免调用方随后修改数组而改变历史诊断。
        self._previous_action = action.copy()
        self._previous_origin_control_step = origin_control_step

    def reset(self, *, episode_id: str, reason: str = "episode_reset") -> None:
        """清空 proposal 历史，并绑定新的 Episode identity。"""

        validated_episode_id = self._validate_episode_id(episode_id)
        validated_reason = self._validate_reason(reason)
        self._episode_id = validated_episode_id
        self._previous_action = None
        self._previous_origin_control_step = None
        self._warmup_reason = validated_reason

    def evaluate(
        self,
        normalized_action: np.ndarray,
        *,
        episode_id: str,
        origin_control_step: int,
        observed_gripper_opening: float,
    ) -> ActionConsistencyResult:
        """诊断一个 raw normalized Action Chunk，但不改变该 Chunk。"""

        # 所有外部输入先完成校验；校验失败不得污染已有历史。
        validated_episode_id = self._validate_episode_id(episode_id)
        origin = self._validate_origin_control_step(origin_control_step)
        opening = self._validate_gripper_opening(observed_gripper_opening)
        action = self._validate_action(normalized_action)
        arm_acm, gripper_acm = self._action_magnitude(action, opening)

        if self._episode_id is None:
            self._episode_id = validated_episode_id

        previous_origin = self._previous_origin_control_step
        if validated_episode_id != self._episode_id:
            return ActionConsistencyResult(
                version=ACTION_CONSISTENCY_CRITIC_VERSION,
                status=ActionConsistencyStatus.ABSTAIN,
                reason="episode_identity_mismatch",
                episode_id=validated_episode_id,
                current_origin_control_step=origin,
                previous_origin_control_step=previous_origin,
                advance_steps=None,
                overlap_steps=0,
                arm_tce_mse_normalized=None,
                gripper_tce_mse_normalized=None,
                arm_acm_rms_normalized=arm_acm,
                gripper_transition_rms_opening_ratio=gripper_acm,
            )

        if self._previous_action is None or previous_origin is None:
            self._store(action, origin)
            reason = self._warmup_reason or "no_previous_proposal"
            self._warmup_reason = None
            return ActionConsistencyResult(
                version=ACTION_CONSISTENCY_CRITIC_VERSION,
                status=ActionConsistencyStatus.WARMUP,
                reason=reason,
                episode_id=validated_episode_id,
                current_origin_control_step=origin,
                previous_origin_control_step=None,
                advance_steps=None,
                overlap_steps=0,
                arm_tce_mse_normalized=None,
                gripper_tce_mse_normalized=None,
                arm_acm_rms_normalized=arm_acm,
                gripper_transition_rms_opening_ratio=gripper_acm,
            )

        advance = origin - previous_origin
        if advance <= 0:
            return ActionConsistencyResult(
                version=ACTION_CONSISTENCY_CRITIC_VERSION,
                status=ActionConsistencyStatus.ABSTAIN,
                reason="non_monotonic_control_step",
                episode_id=validated_episode_id,
                current_origin_control_step=origin,
                previous_origin_control_step=previous_origin,
                advance_steps=advance,
                overlap_steps=0,
                arm_tce_mse_normalized=None,
                gripper_tce_mse_normalized=None,
                arm_acm_rms_normalized=arm_acm,
                gripper_transition_rms_opening_ratio=gripper_acm,
            )

        if advance >= self.spec.action_horizon:
            # 当前 proposal 合法，可作为恢复后的新 baseline。
            self._store(action, origin)
            return ActionConsistencyResult(
                version=ACTION_CONSISTENCY_CRITIC_VERSION,
                status=ActionConsistencyStatus.ABSTAIN,
                reason="no_temporal_overlap",
                episode_id=validated_episode_id,
                current_origin_control_step=origin,
                previous_origin_control_step=previous_origin,
                advance_steps=advance,
                overlap_steps=0,
                arm_tce_mse_normalized=None,
                gripper_tce_mse_normalized=None,
                arm_acm_rms_normalized=arm_acm,
                gripper_transition_rms_opening_ratio=gripper_acm,
            )

        overlap_steps = self.spec.action_horizon - advance
        previous_overlap = self._previous_action[advance:]
        current_overlap = action[:overlap_steps]
        difference = previous_overlap.astype(np.float64) - current_overlap.astype(
            np.float64
        )
        arm_tce = float(
            np.mean(
                np.square(difference[:, : self.spec.arm_dof]),
                dtype=np.float64,
            )
        )
        gripper_tce = float(
            np.mean(
                np.square(difference[:, self.spec.arm_dof]),
                dtype=np.float64,
            )
        )
        self._store(action, origin)
        return ActionConsistencyResult(
            version=ACTION_CONSISTENCY_CRITIC_VERSION,
            status=ActionConsistencyStatus.SCORED,
            reason=None,
            episode_id=validated_episode_id,
            current_origin_control_step=origin,
            previous_origin_control_step=previous_origin,
            advance_steps=advance,
            overlap_steps=overlap_steps,
            arm_tce_mse_normalized=arm_tce,
            gripper_tce_mse_normalized=gripper_tce,
            arm_acm_rms_normalized=arm_acm,
            gripper_transition_rms_opening_ratio=gripper_acm,
        )

    def mark_unavailable(
        self,
        *,
        episode_id: str,
        origin_control_step: int,
        reason: str,
        clear_history: bool = True,
    ) -> ActionConsistencyResult:
        """记录没有可评分 proposal 的 replan；默认清空比较历史。"""

        validated_episode_id = self._validate_episode_id(episode_id)
        origin = self._validate_origin_control_step(origin_control_step)
        validated_reason = self._validate_reason(reason)
        if not isinstance(clear_history, bool):
            raise TypeError("clear_history 必须为 bool")

        same_episode = self._episode_id in {None, validated_episode_id}
        previous_origin = (
            self._previous_origin_control_step if same_episode else None
        )
        advance = (
            origin - previous_origin if previous_origin is not None else None
        )
        if clear_history:
            self._episode_id = validated_episode_id
            self._previous_action = None
            self._previous_origin_control_step = None
            self._warmup_reason = validated_reason

        return ActionConsistencyResult(
            version=ACTION_CONSISTENCY_CRITIC_VERSION,
            status=ActionConsistencyStatus.ABSTAIN,
            reason=validated_reason,
            episode_id=validated_episode_id,
            current_origin_control_step=origin,
            previous_origin_control_step=previous_origin,
            advance_steps=advance,
            overlap_steps=0,
            arm_tce_mse_normalized=None,
            gripper_tce_mse_normalized=None,
            arm_acm_rms_normalized=None,
            gripper_transition_rms_opening_ratio=None,
        )


__all__ = [
    "ACTION_CONSISTENCY_CRITIC_VERSION",
    "ActionConsistencyCritic",
    "ActionConsistencyResult",
    "ActionConsistencyStatus",
]
