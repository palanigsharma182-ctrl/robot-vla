"""按全局控制时间对齐重叠 Action Chunk，并让最新预测占主导。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robot_vla.contracts import RobotSpec


@dataclass(frozen=True)
class TemporalEnsembleConfig:
    recency_decay: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.recency_decay) or not 0.0 < self.recency_decay < 1.0:
            raise ValueError("recency_decay 必须是位于 (0,1) 的有限数值")


@dataclass(frozen=True)
class TemporalEnsembleTrace:
    buffer_size: int
    proposal_counts: tuple[int, ...]
    newest_normalized_weights: tuple[float, ...]
    max_proposal_spread: float
    arm_mean_pairwise_disagreement: tuple[float, ...]
    gripper_mean_pairwise_disagreement: tuple[float, ...]
    arm_max_pairwise_disagreement: tuple[float, ...]
    gripper_max_pairwise_disagreement: tuple[float, ...]
    arm_newest_vs_oldest: tuple[float, ...]
    gripper_newest_vs_oldest: tuple[float, ...]
    arm_newest_vs_weighted_history: tuple[float, ...]
    gripper_newest_vs_weighted_history: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "buffer_size": self.buffer_size,
            "proposal_counts": list(self.proposal_counts),
            "newest_normalized_weights": list(self.newest_normalized_weights),
            "max_proposal_spread": self.max_proposal_spread,
            "arm_mean_pairwise_disagreement": list(
                self.arm_mean_pairwise_disagreement
            ),
            "gripper_mean_pairwise_disagreement": list(
                self.gripper_mean_pairwise_disagreement
            ),
            "arm_max_pairwise_disagreement": list(
                self.arm_max_pairwise_disagreement
            ),
            "gripper_max_pairwise_disagreement": list(
                self.gripper_max_pairwise_disagreement
            ),
            "arm_newest_vs_oldest": list(self.arm_newest_vs_oldest),
            "gripper_newest_vs_oldest": list(self.gripper_newest_vs_oldest),
            "arm_newest_vs_weighted_history": list(
                self.arm_newest_vs_weighted_history
            ),
            "gripper_newest_vs_weighted_history": list(
                self.gripper_newest_vs_weighted_history
            ),
        }


@dataclass(frozen=True)
class TemporalEnsembleOutput:
    normalized_action: np.ndarray
    trace: TemporalEnsembleTrace


@dataclass(frozen=True)
class _StoredChunk:
    origin_control_step: int
    sequence_id: int
    normalized_action: np.ndarray


class TemporalChunkEnsembler:
    """保存仍覆盖未来时间的 Chunk，并输出从当前时刻开始的新 Action Chunk。"""

    def __init__(
        self,
        spec: RobotSpec,
        config: TemporalEnsembleConfig | None = None,
    ) -> None:
        self.spec = spec
        self.config = config or TemporalEnsembleConfig()
        self._chunks: list[_StoredChunk] = []
        self._next_sequence_id = 0

    @property
    def buffer_size(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()

    def add_and_compose(
        self,
        normalized_action: np.ndarray,
        *,
        origin_control_step: int,
    ) -> TemporalEnsembleOutput:
        if origin_control_step < 0:
            raise ValueError("origin_control_step 不能为负数")
        action = np.asarray(normalized_action, dtype=np.float32)
        expected = (self.spec.action_horizon, self.spec.action_dim)
        if action.shape != expected or not np.isfinite(action).all():
            raise ValueError(f"normalized_action 应为有限 {expected} 数组")
        if np.any(np.abs(action) > 1.0 + 1e-5):
            raise ValueError("normalized_action 超出 [-1,1]")

        self._chunks = [
            chunk
            for chunk in self._chunks
            if chunk.origin_control_step + self.spec.action_horizon > origin_control_step
        ]
        sequence_id = self._next_sequence_id
        self._next_sequence_id += 1
        self._chunks.append(
            _StoredChunk(
                origin_control_step=origin_control_step,
                sequence_id=sequence_id,
                normalized_action=action.copy(),
            )
        )

        composed = np.empty(expected, dtype=np.float32)
        proposal_counts: list[int] = []
        newest_weights: list[float] = []
        max_spread = 0.0
        arm_mean_pairwise: list[float] = []
        gripper_mean_pairwise: list[float] = []
        arm_max_pairwise: list[float] = []
        gripper_max_pairwise: list[float] = []
        arm_newest_oldest: list[float] = []
        gripper_newest_oldest: list[float] = []
        arm_newest_history: list[float] = []
        gripper_newest_history: list[float] = []
        for offset in range(self.spec.action_horizon):
            global_step = origin_control_step + offset
            proposals: list[np.ndarray] = []
            weights: list[float] = []
            proposal_sequence_ids: list[int] = []
            for chunk in self._chunks:
                local_index = global_step - chunk.origin_control_step
                if 0 <= local_index < self.spec.action_horizon:
                    age = sequence_id - chunk.sequence_id
                    proposals.append(chunk.normalized_action[local_index])
                    weights.append(self.config.recency_decay**age)
                    proposal_sequence_ids.append(chunk.sequence_id)
            if not proposals:
                raise RuntimeError("Temporal ensemble 没有覆盖当前未来控制步的 proposal")
            proposal_array = np.stack(proposals).astype(np.float32, copy=False)
            weight_array = np.asarray(weights, dtype=np.float32)
            weight_array /= weight_array.sum()
            composed[offset] = np.sum(
                proposal_array * weight_array[:, None],
                axis=0,
                dtype=np.float32,
            )
            proposal_counts.append(len(proposals))
            newest_index = proposal_sequence_ids.index(sequence_id)
            newest_weights.append(float(weight_array[newest_index]))
            spread = np.max(np.abs(proposal_array - composed[offset]), initial=0.0)
            max_spread = max(max_spread, float(spread))

            if len(proposals) == 1:
                arm_mean_pairwise.append(0.0)
                gripper_mean_pairwise.append(0.0)
                arm_max_pairwise.append(0.0)
                gripper_max_pairwise.append(0.0)
                arm_newest_oldest.append(0.0)
                gripper_newest_oldest.append(0.0)
                arm_newest_history.append(0.0)
                gripper_newest_history.append(0.0)
                continue

            pair_differences = np.abs(
                proposal_array[:, None, :] - proposal_array[None, :, :]
            )
            upper = np.triu_indices(len(proposals), k=1)
            pair_values = pair_differences[upper]
            arm_pairs = pair_values[:, : self.spec.arm_dof]
            gripper_pairs = pair_values[:, -1]
            arm_mean_pairwise.append(float(np.mean(arm_pairs)))
            gripper_mean_pairwise.append(float(np.mean(gripper_pairs)))
            arm_max_pairwise.append(float(np.max(arm_pairs)))
            gripper_max_pairwise.append(float(np.max(gripper_pairs)))

            newest = proposal_array[newest_index]
            oldest_index = int(np.argmin(proposal_sequence_ids))
            oldest = proposal_array[oldest_index]
            arm_newest_oldest.append(
                float(np.mean(np.abs(newest[: self.spec.arm_dof] - oldest[: self.spec.arm_dof])))
            )
            gripper_newest_oldest.append(float(abs(newest[-1] - oldest[-1])))

            history_mask = np.ones(len(proposals), dtype=np.bool_)
            history_mask[newest_index] = False
            history_weights = weight_array[history_mask]
            history_weights /= history_weights.sum()
            weighted_history = np.sum(
                proposal_array[history_mask] * history_weights[:, None],
                axis=0,
                dtype=np.float32,
            )
            arm_newest_history.append(
                float(
                    np.mean(
                        np.abs(
                            newest[: self.spec.arm_dof]
                            - weighted_history[: self.spec.arm_dof]
                        )
                    )
                )
            )
            gripper_newest_history.append(
                float(abs(newest[-1] - weighted_history[-1]))
            )

        return TemporalEnsembleOutput(
            normalized_action=np.clip(composed, -1.0, 1.0),
            trace=TemporalEnsembleTrace(
                buffer_size=len(self._chunks),
                proposal_counts=tuple(proposal_counts),
                newest_normalized_weights=tuple(newest_weights),
                max_proposal_spread=max_spread,
                arm_mean_pairwise_disagreement=tuple(arm_mean_pairwise),
                gripper_mean_pairwise_disagreement=tuple(gripper_mean_pairwise),
                arm_max_pairwise_disagreement=tuple(arm_max_pairwise),
                gripper_max_pairwise_disagreement=tuple(gripper_max_pairwise),
                arm_newest_vs_oldest=tuple(arm_newest_oldest),
                gripper_newest_vs_oldest=tuple(gripper_newest_oldest),
                arm_newest_vs_weighted_history=tuple(arm_newest_history),
                gripper_newest_vs_weighted_history=tuple(gripper_newest_history),
            ),
        )


__all__ = [
    "TemporalChunkEnsembler",
    "TemporalEnsembleConfig",
    "TemporalEnsembleOutput",
    "TemporalEnsembleTrace",
]
