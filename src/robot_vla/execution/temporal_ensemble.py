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

        return TemporalEnsembleOutput(
            normalized_action=np.clip(composed, -1.0, 1.0),
            trace=TemporalEnsembleTrace(
                buffer_size=len(self._chunks),
                proposal_counts=tuple(proposal_counts),
                newest_normalized_weights=tuple(newest_weights),
                max_proposal_spread=max_spread,
            ),
        )


__all__ = [
    "TemporalChunkEnsembler",
    "TemporalEnsembleConfig",
    "TemporalEnsembleOutput",
    "TemporalEnsembleTrace",
]
