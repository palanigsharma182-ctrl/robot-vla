"""D013 固定的 Task → Episode → timestep 分层平衡采样。"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
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
        source_weights: Sequence[tuple[str, float]] = (),
    ) -> None:
        if len(dataset) <= 0:
            raise ValueError("BalancedSampler 需要非空 Dataset")
        self.dataset = dataset
        self.num_samples = len(dataset) if num_samples is None else int(num_samples)
        if self.num_samples <= 0 or seed < 0:
            raise ValueError("num_samples 必须为正整数，seed 不能为负数")
        self.seed = int(seed)
        self.epoch = 0
        self._last_exposure: Counter[tuple[str, int, int | None]] = Counter()
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

        resolved_source_weights: dict[str, float] = {}
        for source, weight in source_weights:
            source = str(source)
            weight = float(weight)
            if not source or source.strip() != source:
                raise ValueError("source 名称不能为空或包含首尾空白")
            if source in resolved_source_weights:
                raise ValueError(f"source_weights 重复定义 source: {source}")
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("source weight 必须是有限正数")
            resolved_source_weights[source] = weight
        self.source_weights = tuple(resolved_source_weights.items())

        grouped: dict[str, dict[str, dict[Any, list[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self.sample_identity: dict[int, tuple[str, int, int | None]] = {}
        for sample_index in range(len(dataset)):
            if hasattr(dataset, "sampling_metadata"):
                metadata = dataset.sampling_metadata(sample_index)
                source = str(metadata["source"])
                task_id = str(metadata["task_id"])
                episode_key = metadata["episode_key"]
                skill_id = int(metadata["skill_id"])
                raw_boundary_offset = metadata["boundary_offset"]
                boundary_offset = (
                    None
                    if raw_boundary_offset is None
                    else int(raw_boundary_offset)
                )
            else:
                entry_index, timestep = dataset.index[sample_index]
                entry = dataset.entries[entry_index]
                provenance = getattr(entry, "local_dagger", None)
                source = "base_d0" if provenance is None else str(provenance.source)
                task_id = str(entry.task.task_id)
                episode_key = entry_index
                arrays = dataset.store.get(entry)
                skill_id = int(arrays.skill_id[timestep])
                boundary_offset = (
                    None
                    if provenance is None
                    else int(timestep - provenance.training_window_start)
                )
            grouped[source][task_id][episode_key].append(sample_index)
            self.sample_identity[sample_index] = (
                source,
                skill_id,
                boundary_offset,
            )
        if not grouped:
            raise ValueError("Dataset 没有可采样索引")
        observed_sources = set(grouped)
        configured_sources = set(resolved_source_weights)
        if len(observed_sources) > 1 and not resolved_source_weights:
            raise ValueError("多 source Dataset 必须显式配置 source_weights")
        if resolved_source_weights and configured_sources != observed_sources:
            missing = sorted(observed_sources - configured_sources)
            absent = sorted(configured_sources - observed_sources)
            raise ValueError(
                "source_weights 必须精确覆盖 Dataset source: "
                f"未配置={missing}, Dataset 不存在={absent}"
            )

        self.source_ids = tuple(sorted(grouped))
        self.task_ids = tuple(
            sorted({task_id for by_task in grouped.values() for task_id in by_task})
        )
        self.tasks_by_source: dict[str, tuple[str, ...]] = {}
        self.episodes_by_source_task: dict[tuple[str, str], tuple[Any, ...]] = {}
        self.samples_by_episode: dict[Any, np.ndarray] = {}
        self.probabilities_by_episode: dict[Any, np.ndarray | None] = {}
        for source in self.source_ids:
            by_task = grouped[source]
            self.tasks_by_source[source] = tuple(sorted(by_task))
            for task_id, episodes in by_task.items():
                self.episodes_by_source_task[(source, task_id)] = tuple(
                    sorted(episodes, key=str)
                )
                for episode_key, samples in episodes.items():
                    sample_indices = np.asarray(samples, dtype=np.int64)
                    self.samples_by_episode[episode_key] = sample_indices
                    if not weights:
                        self.probabilities_by_episode[episode_key] = None
                        continue
                    sample_weights = np.asarray(
                        [
                            weights.get(self.sample_identity[index][1], 1.0)
                            for index in samples
                        ],
                        dtype=np.float64,
                    )
                    self.probabilities_by_episode[episode_key] = (
                        sample_weights / sample_weights.sum()
                    )

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch 不能为负数")
        self.epoch = int(epoch)

    def _source_schedule(self, generator: np.random.Generator) -> list[str]:
        if not self.source_weights:
            return []
        total_weight = sum(weight for _, weight in self.source_weights)
        exact = [self.num_samples * weight / total_weight for _, weight in self.source_weights]
        counts = [math.floor(value) for value in exact]
        remainder = self.num_samples - sum(counts)
        ranked = sorted(
            range(len(exact)),
            key=lambda index: (
                -(exact[index] - counts[index]),
                (index + self.epoch) % len(exact),
            ),
        )
        for index in ranked[:remainder]:
            counts[index] += 1
        schedule = [
            source
            for (source, _), count in zip(self.source_weights, counts, strict=True)
            for _ in range(count)
        ]
        generator.shuffle(schedule)
        return schedule

    def exposure_rows(self) -> list[dict[str, int | str | None]]:
        """返回最近一次迭代的实际 source × skill × boundary-offset 计数。"""

        return [
            {
                "source": source,
                "skill_id": skill_id,
                "boundary_offset": boundary_offset,
                "samples": samples,
            }
            for (source, skill_id, boundary_offset), samples in sorted(
                self._last_exposure.items(),
                key=lambda item: (
                    item[0][0],
                    item[0][1],
                    -1 if item[0][2] is None else item[0][2],
                ),
            )
        ]

    def __iter__(self) -> Iterator[int]:
        generator = np.random.default_rng(self.seed + self.epoch)
        self._last_exposure = Counter()
        source_schedule = self._source_schedule(generator)
        for sample_number in range(self.num_samples):
            if source_schedule:
                source = source_schedule[sample_number]
            else:
                source = self.source_ids[int(generator.integers(len(self.source_ids)))]
            tasks = self.tasks_by_source[source]
            task_id = tasks[int(generator.integers(len(tasks)))]
            episodes = self.episodes_by_source_task[(source, task_id)]
            episode_key = episodes[int(generator.integers(len(episodes)))]
            sample_indices = self.samples_by_episode[episode_key]
            probabilities = self.probabilities_by_episode[episode_key]
            if probabilities is None:
                offset = int(generator.integers(len(sample_indices)))
            else:
                offset = int(generator.choice(len(sample_indices), p=probabilities))
            sample_index = int(sample_indices[offset])
            self._last_exposure[self.sample_identity[sample_index]] += 1
            yield sample_index
