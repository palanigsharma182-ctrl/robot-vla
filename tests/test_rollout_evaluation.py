from __future__ import annotations

import pytest

from robot_vla.evaluation.rollout import (
    RolloutEpisodeResult,
    classify_rollout_failure,
    summarize_rollouts,
)


def _result(
    *,
    seed_group: str = "unseen",
    seed: int = 10_000,
    completed_skill_count: int = 5,
    failure_category: str | None = None,
) -> RolloutEpisodeResult:
    success = failure_category is None
    completion_steps = (20, 30, 40, 60, 80)
    return RolloutEpisodeResult(
        seed_group=seed_group,
        seed=seed,
        instruction="Pick up the red cube and place it in the green target region.",
        sampling_seed_base=123,
        success=success,
        environment_success=success,
        predicate_success=success,
        failure_category=failure_category,
        failure_stage=None,
        error=None,
        environment_steps=100,
        replans=25,
        sampling_seeds=tuple(range(25)),
        action_chunks=25,
        normalized_action_abs_max=0.9,
        physical_arm_delta_abs_max_rad=0.04,
        gripper_target_min=0.1,
        gripper_target_max=0.9,
        completed_skill_count=completed_skill_count,
        skill_completed=tuple(index < completed_skill_count for index in range(5)),
        terminated=success,
        truncated=not success,
        final_is_grasped=False,
        stable_grasp_steps=2,
        stable_place_steps=4 if success else 0,
        final_tcp_to_object_distance_m=0.1,
        final_object_height_above_support_m=0.0,
        final_object_to_goal_xy_distance_m=0.01,
        final_object_to_goal_distance_m=0.01,
        final_object_linear_speed_m_s=0.0,
        final_object_angular_speed_rad_s=0.0,
        wall_time_s=1.0,
        skill_completion_environment_steps=tuple(
            step if index < completed_skill_count else None
            for index, step in enumerate(completion_steps)
        ),
    )


@pytest.mark.parametrize(
    ("completed", "grasped", "distance", "linear", "angular", "expected"),
    [
        (0, False, 1.0, 0.0, 0.0, "reach_failed"),
        (1, False, 1.0, 0.0, 0.0, "grasp_failed"),
        (2, True, 1.0, 0.0, 0.0, "lift_failed"),
        (3, True, 1.0, 0.0, 0.0, "transport_failed"),
        (4, True, 0.01, 0.0, 0.0, "release_failed"),
        (4, False, 0.03, 0.0, 0.0, "place_position_failed"),
        (4, False, 0.01, 0.02, 0.0, "place_stability_failed"),
        (4, False, 0.01, 0.0, 0.0, "place_stability_timeout"),
    ],
)
def test_failure_classification_uses_first_unfinished_atomic_skill(
    completed,
    grasped,
    distance,
    linear,
    angular,
    expected,
) -> None:
    assert (
        classify_rollout_failure(
            completed_skill_count=completed,
            predicate_success=False,
            environment_success=False,
            failure_stage=None,
            final_is_grasped=grasped,
            final_object_to_goal_distance_m=distance,
            place_distance_m=0.025,
            final_object_linear_speed_m_s=linear,
            static_linear_speed_m_s=0.01,
            final_object_angular_speed_rad_s=angular,
            static_angular_speed_rad_s=0.5,
        )
        == expected
    )


def test_failure_classification_prioritizes_system_and_predicate_errors() -> None:
    common = {
        "completed_skill_count": 0,
        "predicate_success": False,
        "environment_success": False,
        "final_is_grasped": False,
        "final_object_to_goal_distance_m": 1.0,
        "place_distance_m": 0.025,
        "final_object_linear_speed_m_s": 0.0,
        "static_linear_speed_m_s": 0.01,
        "final_object_angular_speed_rad_s": 0.0,
        "static_angular_speed_rad_s": 0.5,
    }
    assert classify_rollout_failure(**common, failure_stage="inference") == "inference_error"
    assert (
        classify_rollout_failure(
            **{**common, "environment_success": True},
            failure_stage=None,
        )
        == "predicate_mismatch"
    )


def test_rollout_summary_separates_seed_groups_and_reports_skill_rates() -> None:
    results = [
        _result(seed_group="test", seed=27),
        _result(
            seed_group="test",
            seed=28,
            completed_skill_count=2,
            failure_category="lift_failed",
        ),
        _result(seed_group="unseen", seed=10_000),
    ]

    summary = summarize_rollouts(results)

    assert summary["overall"]["success_rate"] == pytest.approx(2 / 3)
    assert summary["groups"]["test"]["success_rate"] == pytest.approx(0.5)
    assert summary["groups"]["unseen"]["success_rate"] == pytest.approx(1.0)
    assert summary["overall"]["skill_success_rates"]["grasp"] == pytest.approx(1.0)
    assert summary["overall"]["skill_success_rates"]["lift"] == pytest.approx(2 / 3)
    assert summary["overall"]["failure_counts"] == {"lift_failed": 1}
    assert summary["overall"]["tracking_correction_saturation_count"] == 0
    assert summary["overall"]["grasp_given_reach"]["rate"] == pytest.approx(1.0)
    assert summary["overall"]["lift_given_grasp"] == {
        "numerator": 2,
        "denominator": 3,
        "rate": pytest.approx(2 / 3),
        "wilson_95": pytest.approx([0.2076596, 0.9385081]),
    }
    assert summary["overall"]["transport_given_lift"]["rate"] == pytest.approx(1.0)
    assert summary["overall"]["mean_steps_to_reach"] == pytest.approx(20.0)
    assert summary["overall"]["mean_steps_reach_to_grasp"] == pytest.approx(10.0)
    assert summary["overall"]["mean_steps_lift_to_transport"] == pytest.approx(20.0)
    lower, upper = summary["overall"]["success_rate_wilson_95"]
    assert 0.0 <= lower < 2 / 3 < upper <= 1.0


def test_rollout_result_json_round_trip_preserves_tuple_contracts() -> None:
    original = _result()
    restored = RolloutEpisodeResult.from_dict(original.to_dict())
    assert restored == original


def test_rollout_result_preserves_action_safety_diagnostic() -> None:
    original = _result(completed_skill_count=0, failure_category="action_safety_rejection")
    original = RolloutEpisodeResult(
        **{
            **original.to_dict(),
            "failure_stage": "step_safety",
            "execution_diagnostic": {
                "kind": "physical_action_contract",
                "chunk_step_index": 2,
            },
        }
    )

    restored = RolloutEpisodeResult.from_dict(original.to_dict())

    assert restored.execution_diagnostic == {
        "kind": "physical_action_contract",
        "chunk_step_index": 2,
    }


def test_rollout_summary_rejects_duplicate_episode_identity() -> None:
    with pytest.raises(ValueError, match="重复"):
        summarize_rollouts([_result(), _result()])
