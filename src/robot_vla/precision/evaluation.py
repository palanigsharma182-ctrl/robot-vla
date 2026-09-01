"""E013 厘米级闭环精调的预注册评估门槛。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class PrecisionPerformanceTier(str, Enum):
    """按系统级最终放置误差划分的最高通过档位。"""

    BELOW_ENGINEERING_FLOOR = "below_engineering_floor"
    ENGINEERING_USABLE = "engineering_usable"
    RECOMMENDED_PORTFOLIO = "recommended_portfolio"
    OPTIONAL_STRETCH = "optional_stretch"


@dataclass(frozen=True)
class PrecisionEvaluationThresholds:
    """冻结的 E013 指标；全部平移量使用米，延迟使用秒。"""

    baseline_p50_xy_error_m: float = 0.0253
    baseline_p90_xy_error_m: float = 0.0388
    engineering_p50_xy_error_max_m: float = 0.015
    engineering_p90_xy_error_max_m: float = 0.025
    recommended_p50_xy_error_max_m: float = 0.012
    recommended_p90_xy_error_max_m: float = 0.020
    recommended_within_20mm_rate_min: float = 0.90
    stretch_p50_xy_error_max_m: float = 0.010
    stretch_p90_xy_error_max_m: float = 0.015
    formal_episode_count_min: int = 100
    effective_control_hz_min: float = 20.0
    p95_latency_max_s: float = 0.050

    def __post_init__(self) -> None:
        distance_fields = (
            self.baseline_p50_xy_error_m,
            self.baseline_p90_xy_error_m,
            self.engineering_p50_xy_error_max_m,
            self.engineering_p90_xy_error_max_m,
            self.recommended_p50_xy_error_max_m,
            self.recommended_p90_xy_error_max_m,
            self.stretch_p50_xy_error_max_m,
            self.stretch_p90_xy_error_max_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in distance_fields):
            raise ValueError("所有距离阈值必须是有限正数")
        if self.baseline_p50_xy_error_m > self.baseline_p90_xy_error_m:
            raise ValueError("baseline p50 不能大于 baseline p90")
        if self.engineering_p50_xy_error_max_m > self.engineering_p90_xy_error_max_m:
            raise ValueError("engineering p50 阈值不能大于 p90 阈值")
        if self.recommended_p50_xy_error_max_m > self.recommended_p90_xy_error_max_m:
            raise ValueError("recommended p50 阈值不能大于 p90 阈值")
        if self.stretch_p50_xy_error_max_m > self.stretch_p90_xy_error_max_m:
            raise ValueError("stretch p50 阈值不能大于 p90 阈值")
        if not (
            self.stretch_p50_xy_error_max_m
            <= self.recommended_p50_xy_error_max_m
            <= self.engineering_p50_xy_error_max_m
        ):
            raise ValueError("p50 阈值必须按 stretch、recommended、engineering 依次放宽")
        if not (
            self.stretch_p90_xy_error_max_m
            <= self.recommended_p90_xy_error_max_m
            <= self.engineering_p90_xy_error_max_m
        ):
            raise ValueError("p90 阈值必须按 stretch、recommended、engineering 依次放宽")
        if not 0.0 <= self.recommended_within_20mm_rate_min <= 1.0:
            raise ValueError("recommended_within_20mm_rate_min 必须位于 [0,1]")
        if (
            not isinstance(self.formal_episode_count_min, int)
            or isinstance(self.formal_episode_count_min, bool)
            or self.formal_episode_count_min <= 0
        ):
            raise ValueError("formal_episode_count_min 必须为正整数")
        if not math.isfinite(self.effective_control_hz_min) or self.effective_control_hz_min <= 0:
            raise ValueError("effective_control_hz_min 必须是有限正数")
        if not math.isfinite(self.p95_latency_max_s) or self.p95_latency_max_s <= 0:
            raise ValueError("p95_latency_max_s 必须是有限正数")


@dataclass(frozen=True)
class PrecisionEvaluationMetrics:
    """正式闭环汇总；失败 Episode 必须保留在上游误差和成功率统计中。"""

    episode_count: int
    final_xy_error_p50_m: float
    final_xy_error_p90_m: float
    within_20mm_rate: float
    effective_control_hz: float
    p95_latency_s: float
    invalid_projection_count: int = 0
    system_failure_count: int = 0
    safety_failure_count: int = 0
    tracking_failure_count: int = 0
    controller_overlap_count: int = 0
    stale_observation_command_count: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.episode_count, int)
            or isinstance(self.episode_count, bool)
            or self.episode_count <= 0
        ):
            raise ValueError("episode_count 必须为正整数")
        if (
            not math.isfinite(self.final_xy_error_p50_m)
            or self.final_xy_error_p50_m < 0.0
            or not math.isfinite(self.final_xy_error_p90_m)
            or self.final_xy_error_p90_m < 0.0
        ):
            raise ValueError("最终 XY 误差必须是有限非负数")
        if self.final_xy_error_p50_m > self.final_xy_error_p90_m:
            raise ValueError("最终 XY p50 不能大于 p90")
        if not math.isfinite(self.within_20mm_rate) or not 0.0 <= self.within_20mm_rate <= 1.0:
            raise ValueError("within_20mm_rate 必须位于 [0,1]")
        if not math.isfinite(self.effective_control_hz) or self.effective_control_hz <= 0.0:
            raise ValueError("effective_control_hz 必须是有限正数")
        if not math.isfinite(self.p95_latency_s) or self.p95_latency_s < 0.0:
            raise ValueError("p95_latency_s 必须是有限非负数")
        count_fields = (
            self.invalid_projection_count,
            self.system_failure_count,
            self.safety_failure_count,
            self.tracking_failure_count,
            self.controller_overlap_count,
            self.stale_observation_command_count,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in count_fields
        ):
            raise ValueError("所有失败计数必须是非负整数")


@dataclass(frozen=True)
class PrecisionEvaluationAssessment:
    """门槛复算结果。"""

    tier: PrecisionPerformanceTier
    engineering_floor_passed: bool
    recommended_target_passed: bool
    optional_stretch_passed: bool
    guardrail_failures: tuple[str, ...]
    p50_reduction_from_baseline: float
    p90_reduction_from_baseline: float


def assess_precision_evaluation(
    metrics: PrecisionEvaluationMetrics,
    thresholds: PrecisionEvaluationThresholds | None = None,
) -> PrecisionEvaluationAssessment:
    """按冻结顺序给正式结果分档；任何 guardrail 失败都不得 promotion。"""

    target = thresholds or PrecisionEvaluationThresholds()
    guardrail_failures: list[str] = []
    if metrics.episode_count < target.formal_episode_count_min:
        guardrail_failures.append("insufficient_episode_count")
    if metrics.effective_control_hz < target.effective_control_hz_min:
        guardrail_failures.append("control_rate")
    if metrics.p95_latency_s > target.p95_latency_max_s:
        guardrail_failures.append("latency")
    for name, value in (
        ("invalid_projection", metrics.invalid_projection_count),
        ("system_failure", metrics.system_failure_count),
        ("safety_failure", metrics.safety_failure_count),
        ("tracking_failure", metrics.tracking_failure_count),
        ("controller_overlap", metrics.controller_overlap_count),
        ("stale_observation_command", metrics.stale_observation_command_count),
    ):
        if value != 0:
            guardrail_failures.append(name)

    guardrails_passed = not guardrail_failures
    engineering_passed = (
        guardrails_passed
        and metrics.final_xy_error_p50_m <= target.engineering_p50_xy_error_max_m
        and metrics.final_xy_error_p90_m <= target.engineering_p90_xy_error_max_m
    )
    recommended_passed = (
        engineering_passed
        and metrics.final_xy_error_p50_m <= target.recommended_p50_xy_error_max_m
        and metrics.final_xy_error_p90_m <= target.recommended_p90_xy_error_max_m
        and metrics.within_20mm_rate >= target.recommended_within_20mm_rate_min
    )
    stretch_passed = (
        recommended_passed
        and metrics.final_xy_error_p50_m <= target.stretch_p50_xy_error_max_m
        and metrics.final_xy_error_p90_m <= target.stretch_p90_xy_error_max_m
    )

    if stretch_passed:
        tier = PrecisionPerformanceTier.OPTIONAL_STRETCH
    elif recommended_passed:
        tier = PrecisionPerformanceTier.RECOMMENDED_PORTFOLIO
    elif engineering_passed:
        tier = PrecisionPerformanceTier.ENGINEERING_USABLE
    else:
        tier = PrecisionPerformanceTier.BELOW_ENGINEERING_FLOOR

    return PrecisionEvaluationAssessment(
        tier=tier,
        engineering_floor_passed=engineering_passed,
        recommended_target_passed=recommended_passed,
        optional_stretch_passed=stretch_passed,
        guardrail_failures=tuple(guardrail_failures),
        p50_reduction_from_baseline=(
            1.0 - metrics.final_xy_error_p50_m / target.baseline_p50_xy_error_m
        ),
        p90_reduction_from_baseline=(
            1.0 - metrics.final_xy_error_p90_m / target.baseline_p90_xy_error_m
        ),
    )


__all__ = [
    "PrecisionEvaluationAssessment",
    "PrecisionEvaluationMetrics",
    "PrecisionEvaluationThresholds",
    "PrecisionPerformanceTier",
    "assess_precision_evaluation",
]
