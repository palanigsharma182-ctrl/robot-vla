from types import SimpleNamespace

import numpy as np
import pytest

from robot_vla.data.sampler import TaskEpisodeBalancedSampler


class FakeStore:
    def __init__(self, skills_by_entry: dict[int, np.ndarray]) -> None:
        self.skills_by_entry = skills_by_entry

    def get(self, entry):
        return SimpleNamespace(skill_id=self.skills_by_entry[entry.entry_index])


class FakeDataset:
    def __init__(self, episode_specs: list[tuple[str, int, np.ndarray]]) -> None:
        self.entries = []
        self.index = []
        skills_by_entry = {}
        for entry_index, (task_id, sample_count, skills) in enumerate(episode_specs):
            assert sample_count == len(skills)
            self.entries.append(
                SimpleNamespace(
                    entry_index=entry_index,
                    task=SimpleNamespace(task_id=task_id),
                )
            )
            skills_by_entry[entry_index] = skills
            self.index.extend((entry_index, timestep) for timestep in range(sample_count))
        self.store = FakeStore(skills_by_entry)

    def __len__(self) -> int:
        return len(self.index)


def test_sampler_balances_tasks_and_episodes_instead_of_global_timesteps() -> None:
    dataset = FakeDataset(
        [
            ("task-a", 90, np.zeros(90, dtype=np.int16)),
            ("task-a", 10, np.zeros(10, dtype=np.int16)),
            ("task-b", 5, np.zeros(5, dtype=np.int16)),
        ]
    )
    sampler = TaskEpisodeBalancedSampler(dataset, num_samples=8_000, seed=7)
    sampled = list(sampler)
    sampled_entries = [dataset.index[index][0] for index in sampled]

    task_a_ratio = sum(entry in {0, 1} for entry in sampled_entries) / len(sampled_entries)
    episode_0_within_a = sampled_entries.count(0) / sum(
        entry in {0, 1} for entry in sampled_entries
    )

    assert task_a_ratio == pytest.approx(0.5, abs=0.03)
    assert episode_0_within_a == pytest.approx(0.5, abs=0.03)


def test_sampler_applies_explicit_skill_weights_inside_selected_episode() -> None:
    skills = np.asarray([0, 1] * 10, dtype=np.int16)
    dataset = FakeDataset([("task-a", len(skills), skills)])
    sampler = TaskEpisodeBalancedSampler(
        dataset,
        num_samples=4_000,
        seed=11,
        skill_weights=((0, 1.0), (1, 9.0)),
    )

    sampled_skills = [skills[dataset.index[index][1]] for index in sampler]

    assert sampled_skills.count(1) / len(sampled_skills) == pytest.approx(0.9, abs=0.03)


def test_sampler_is_reproducible_per_epoch_and_changes_between_epochs() -> None:
    dataset = FakeDataset([("task-a", 20, np.zeros(20, dtype=np.int16))])
    sampler = TaskEpisodeBalancedSampler(dataset, num_samples=20, seed=13)

    first = list(sampler)
    second = list(sampler)
    sampler.set_epoch(1)
    next_epoch = list(sampler)

    assert first == second
    assert first != next_epoch


def test_sampler_rejects_unknown_or_non_positive_skill_weight() -> None:
    dataset = FakeDataset([("task-a", 2, np.zeros(2, dtype=np.int16))])

    with pytest.raises(ValueError, match="未知 skill_id"):
        TaskEpisodeBalancedSampler(dataset, skill_weights=((99, 1.0),))
    with pytest.raises(ValueError, match="有限正数"):
        TaskEpisodeBalancedSampler(dataset, skill_weights=((0, 0.0),))
