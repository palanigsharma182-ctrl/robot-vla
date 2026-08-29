"""D013 固定的 Task → Episode → timestep 分层平衡采样。"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from robot_vla.contracts import PICK_AND_PLACE_SKILLS, UNKNOWN_SKILL_ID


class TaskEpisodeBalancedSampler:
    """有放回采样，避免 Task、长 Episode 或高频阶段主导训练。"""

    def __init__(
        self,
        dataset: Any,
        *,
        num_samples: int | None = None,
        seed: int = 42,
        skill_weights: Sequence[tuple[int, float]] = (),
    ) -> None:
        if len(dataset) <= 0:
            raise ValueError("BalancedSampler 需要非空 Dataset")
        self.dataset = dataset
        self.num_samples = len(dataset) if num_samples is None else int(num_samples)
        if self.num_samples <= 0 or seed < 0:
            raise ValueError("num_samples 必须为正整数，seed 不能为负数")
        self.seed = int(seed)
        self.epoch = 0
        allowed_skill_ids = {UNKNOWN_SKILL_ID, *range(len(PICK_AND_PLACE_SKILLS))}
        weights: dict[int, float] = {}
        for skill_id, weight in skill_weights:
            skill_id = int(skill_id)
            weight = float(weight)
            if skill_id not in allowed_skill_ids:
                raise ValueError(f"skill_weights 包含未知 skill_id: {skill_id}")
            if skill_id in weights:
                raise ValueError(f"skill_weights 重复定义 skill_id: {skill_id}")
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("skill weight 必须是有限正数")
            weights[skill_id] = weight
        self.skill_weights = weights

        grouped: dict[str, dict[int, list[tuple[int, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for sample_index, (entry_index, timestep) in enumerate(dataset.index):
            entry = dataset.entries[entry_index]
            grouped[entry.task.task_id][entry_index].append((sample_index, timestep))
        if not grouped:
            raise ValueError("Dataset 没有可采样索引")

        self.task_ids = tuple(sorted(grouped))
        self.episodes_by_task: dict[str, tuple[int, ...]] = {}
        self.samples_by_episode: dict[int, np.ndarray] = {}
        self.probabilities_by_episode: dict[int, np.ndarray | None] = {}
        for task_id in self.task_ids:
            episodes = grouped[task_id]
            self.episodes_by_task[task_id] = tuple(sorted(episodes))
            for entry_index, samples in episodes.items():
                sample_indices = np.asarray([sample[0] for sample in samples], dtype=np.int64)
                self.samples_by_episode[entry_index] = sample_indices
                if not weights:
                    self.probabilities_by_episode[entry_index] = None
                    continue
                arrays = dataset.store.get(dataset.entries[entry_index])
                sample_weights = np.asarray(
                    [weights.get(int(arrays.skill_id[timestep]), 1.0) for _, timestep in samples],
                    dtype=np.float64,
                )
                self.probabilities_by_episode[entry_index] = sample_weights / sample_weights.sum()

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch 不能为负数")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.num_samples):
            task_id = self.task_ids[int(generator.integers(len(self.task_ids)))]
            episodes = self.episodes_by_task[task_id]
            entry_index = episodes[int(generator.integers(len(episodes)))]
            sample_indices = self.samples_by_episode[entry_index]
            probabilities = self.probabilities_by_episode[entry_index]
            if probabilities is None:
                offset = int(generator.integers(len(sample_indices)))
            else:
                offset = int(generator.choice(len(sample_indices), p=probabilities))
            yield int(sample_indices[offset])
