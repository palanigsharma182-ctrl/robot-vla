from __future__ import annotations

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("mani_skill")

from robot_vla.contracts import PICK_AND_PLACE_SKILLS
from robot_vla.evaluation.maniskill import _reset_atomic_time_limit
from robot_vla.sim.collector import TrustedPickPlaceCollector


def test_expert_preparation_reaches_exact_atomic_prerequisite() -> None:
    with TrustedPickPlaceCollector(None) as preparer:
        preparation_steps = []
        for target, skill_name in enumerate(PICK_AND_PLACE_SKILLS):
            prepared = preparer.prepare_atomic(seed=321, skill_name=skill_name)
            assert prepared.progress.completed_skill_count == target
            preparation_steps.append(prepared.preparation_steps)

    assert preparation_steps[0] == 0
    assert preparation_steps == sorted(preparation_steps)


def test_atomic_time_limit_reset_preserves_maniskill_counter_tensor() -> None:
    with TrustedPickPlaceCollector(None) as preparer:
        prepared = preparer.prepare_atomic(seed=321, skill_name="place")
        elapsed_steps = preparer.base_env._elapsed_steps
        counter_type = type(elapsed_steps)
        assert int(elapsed_steps.item()) == prepared.preparation_steps > 0

        _reset_atomic_time_limit(preparer.env)

        assert type(preparer.base_env._elapsed_steps) is counter_type
        assert int(preparer.base_env._elapsed_steps.item()) == 0
