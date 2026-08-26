from __future__ import annotations

import pytest

from robot_vla.evaluation.atomic import (
    AtomicSkillEpisodeResult,
    derive_atomic_sampling_seed,
    summarize_atomic_rollouts,
)


def _result(*, seed: int, skill_name: str, success: bool) -> AtomicSkillEpisodeResult:
    target = ("reach", "grasp", "lift", "transport", "place").index(skill_name)
    return AtomicSkillEpisodeResult(
        seed=seed,
        skill_name=skill_name,
        instruction="Pick and place the cube.",
        sampling_seed_base=123,
        success=success,
        failure_category=None if success else f"{skill_name}_failed",
        failure_stage=None,
        error=None,
        preparation_steps=target * 10,
        initial_completed_skill_count=target,
        final_completed_skill_count=target + int(success),
        policy_environment_steps=20,
        replans=5,
        sampling_seeds=(1, 2, 3, 4, 5),
        action_chunks=5,
        tracking_correction_saturation_count=0,
        tracking_correction_requested_abs_max_rad=0.04,
        tracking_correction_applied_abs_max_rad=0.04,
        final_is_grasped=skill_name in {"lift", "transport"},
        final_tcp_to_object_distance_m=0.01,
        final_object_height_above_support_m=0.05,
        final_object_to_goal_xy_distance_m=0.02,
        final_object_to_goal_distance_m=0.02,
        final_object_linear_speed_m_s=0.0,
        final_object_angular_speed_rad_s=0.0,
        wall_time_s=1.0,
    )


def test_atomic_sampling_seed_depends_on_skill_and_seed() -> None:
    first = derive_atomic_sampling_seed(42_424, 10_000, "reach")
    assert first == derive_atomic_sampling_seed(42_424, 10_000, "reach")
    assert first != derive_atomic_sampling_seed(42_424, 10_000, "grasp")
    assert first != derive_atomic_sampling_seed(42_424, 10_001, "reach")


def test_atomic_summary_keeps_skills_independent() -> None:
    results = [
        _result(seed=10_000, skill_name="reach", success=True),
        _result(seed=10_001, skill_name="reach", success=False),
        _result(seed=10_000, skill_name="grasp", success=True),
    ]

    summary = summarize_atomic_rollouts(results)

    assert summary["groups"]["reach"]["success_rate"] == pytest.approx(0.5)
    assert summary["groups"]["grasp"]["success_rate"] == pytest.approx(1.0)
    assert summary["groups"]["reach"]["failure_counts"] == {"reach_failed": 1}


def test_atomic_result_round_trip() -> None:
    original = _result(seed=10_000, skill_name="place", success=False)
    assert AtomicSkillEpisodeResult.from_dict(original.to_dict()) == original
