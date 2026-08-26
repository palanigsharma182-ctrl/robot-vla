from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("mani_skill")

from robot_vla.contracts import RobotSpec
from robot_vla.evaluation.maniskill import (
    _reset_atomic_time_limit,
    derive_episode_sampling_seed,
    run_maniskill_episode,
)
from robot_vla.evaluation.rollout import RolloutEpisodeSpec
from robot_vla.runtime import RuntimeActionChunk, SamplingTrace
from robot_vla.sim import PICK_CUBE_TO_REGION_ENV_ID, register_robot_vla_maniskill_envs


class ZeroRuntime:
    def __init__(self, spec: RobotSpec, seed: int) -> None:
        self.spec = spec
        self.seed = seed
        self.index = 0
        self._last_sampling_trace = None

    @property
    def last_sampling_trace(self):
        return self._last_sampling_trace

    def infer_action_chunk(self, _observation):
        trace = SamplingTrace(seed=self.seed + self.index, sample_index=self.index)
        self.index += 1
        self._last_sampling_trace = trace
        normalized = np.zeros(
            (self.spec.action_horizon, self.spec.action_dim),
            dtype=np.float32,
        )
        physical = normalized.copy()
        return RuntimeActionChunk(
            normalized_action=normalized,
            physical_action=physical,
            visual_tokens_per_image=(1, 1),
            context_length=2,
            sampling=trace,
        )


def test_zero_policy_runs_to_time_limit_and_is_classified_at_atomic_skill() -> None:
    register_robot_vla_maniskill_envs()
    spec = RobotSpec()
    episode = RolloutEpisodeSpec("unseen", 10_000, "Pick and place the red cube.")
    sampling_seed = derive_episode_sampling_seed(42_424, episode)
    env = gym.make(
        PICK_CUBE_TO_REGION_ENV_ID,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        num_envs=1,
    )
    try:
        result = run_maniskill_episode(
            env,
            ZeroRuntime(spec, sampling_seed),
            spec,
            episode,
            sampling_seed_base=sampling_seed,
        )
    finally:
        env.close()

    assert result.success is False
    assert result.failure_category == "reach_failed"
    assert result.environment_steps == 300
    assert result.replans == 75
    assert len(result.sampling_seeds) == 75
    assert result.completed_skill_count == 0
    assert result.truncated is True


def test_episode_sampling_seed_depends_on_group_and_environment_seed() -> None:
    first = RolloutEpisodeSpec("test", 27, "instruction")
    repeated = RolloutEpisodeSpec("test", 27, "instruction")
    different_group = RolloutEpisodeSpec("unseen", 27, "instruction")
    assert derive_episode_sampling_seed(7, first) == derive_episode_sampling_seed(7, repeated)
    assert derive_episode_sampling_seed(7, first) != derive_episode_sampling_seed(7, different_group)


def test_atomic_time_limit_reset_traverses_wrapper_chain() -> None:
    class TensorLike:
        def __init__(self, value: int):
            self.value = value

        def zero_(self):
            self.value = 0

    class Base:
        def __init__(self):
            self._elapsed_steps = TensorLike(206)

    class Wrapper:
        def __init__(self, env):
            self.env = env

    class TimeLimit(Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self._elapsed_steps = 206

    base = Base()
    time_limit = TimeLimit(base)
    outer = Wrapper(time_limit)

    _reset_atomic_time_limit(outer)

    assert time_limit._elapsed_steps == 0
    assert base._elapsed_steps.value == 0
