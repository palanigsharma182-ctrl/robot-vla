import pytest

from robot_vla.precision.evaluation import (
    PrecisionEvaluationMetrics,
    PrecisionEvaluationThresholds,
    PrecisionPerformanceTier,
    assess_precision_evaluation,
)


def _metrics(**overrides: object) -> PrecisionEvaluationMetrics:
    values: dict[str, object] = {
        "episode_count": 100,
        "final_xy_error_p50_m": 0.012,
        "final_xy_error_p90_m": 0.020,
        "within_20mm_rate": 0.90,
        "effective_control_hz": 20.0,
        "p95_latency_s": 0.050,
    }
    values.update(overrides)
    return PrecisionEvaluationMetrics(**values)  # type: ignore[arg-type]


def test_recommended_threshold_is_inclusive_and_above_engineering_floor() -> None:
    assessment = assess_precision_evaluation(_metrics())

    assert assessment.tier == PrecisionPerformanceTier.RECOMMENDED_PORTFOLIO
    assert assessment.engineering_floor_passed is True
    assert assessment.recommended_target_passed is True
    assert assessment.optional_stretch_passed is False
    assert assessment.guardrail_failures == ()
    assert not hasattr(assessment, "p50_reduction_from_baseline")
    assert not hasattr(assessment, "p90_reduction_from_baseline")
    assert not hasattr(assessment, "p50_reduction_from_spatial_probe_reference")
    assert not hasattr(assessment, "p90_reduction_from_spatial_probe_reference")


def test_engineering_floor_is_an_acceptable_but_separate_tier() -> None:
    assessment = assess_precision_evaluation(
        _metrics(
            final_xy_error_p50_m=0.015,
            final_xy_error_p90_m=0.025,
            within_20mm_rate=0.79,
        )
    )

    assert assessment.tier == PrecisionPerformanceTier.ENGINEERING_USABLE
    assert assessment.engineering_floor_passed is True
    assert assessment.recommended_target_passed is False


def test_metric_above_engineering_floor_does_not_pass() -> None:
    assessment = assess_precision_evaluation(
        _metrics(final_xy_error_p90_m=0.025_001)
    )

    assert assessment.tier == PrecisionPerformanceTier.BELOW_ENGINEERING_FLOOR
    assert assessment.engineering_floor_passed is False


def test_optional_stretch_is_reported_without_becoming_the_required_target() -> None:
    assessment = assess_precision_evaluation(
        _metrics(
            final_xy_error_p50_m=0.010,
            final_xy_error_p90_m=0.015,
            within_20mm_rate=0.96,
        )
    )

    assert assessment.tier == PrecisionPerformanceTier.OPTIONAL_STRETCH
    assert assessment.recommended_target_passed is True
    assert assessment.optional_stretch_passed is True


@pytest.mark.parametrize(
    ("overrides", "failure"),
    [
        ({"episode_count": 99}, "insufficient_episode_count"),
        ({"effective_control_hz": 19.99}, "control_rate"),
        ({"p95_latency_s": 0.050_001}, "latency"),
        ({"invalid_projection_count": 1}, "invalid_projection"),
        ({"system_failure_count": 1}, "system_failure"),
        ({"safety_failure_count": 1}, "safety_failure"),
        ({"tracking_failure_count": 1}, "tracking_failure"),
        ({"controller_overlap_count": 1}, "controller_overlap"),
        ({"stale_observation_command_count": 1}, "stale_observation_command"),
    ],
)
def test_guardrail_failure_blocks_all_tiers(
    overrides: dict[str, object],
    failure: str,
) -> None:
    assessment = assess_precision_evaluation(_metrics(**overrides))

    assert assessment.tier == PrecisionPerformanceTier.BELOW_ENGINEERING_FLOOR
    assert assessment.engineering_floor_passed is False
    assert failure in assessment.guardrail_failures


def test_invalid_metric_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="p50 不能大于 p90"):
        _metrics(
            final_xy_error_p50_m=0.021,
            final_xy_error_p90_m=0.020,
        )


def test_episode_count_must_be_an_integer() -> None:
    with pytest.raises(ValueError, match="正整数"):
        _metrics(episode_count=100.0)


def test_threshold_tiers_must_be_monotonic() -> None:
    with pytest.raises(ValueError, match="依次放宽"):
        PrecisionEvaluationThresholds(stretch_p50_xy_error_max_m=0.013)
