"""E012 预注册的 10/20/30 epoch checkpoint 排除与字典序选择。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from robot_vla.evaluation.atomic import AtomicSkillEpisodeResult
from robot_vla.evaluation.e012_paired import analyze_e012_pair
from robot_vla.evaluation.rollout import RolloutEpisodeResult

E012_CHECKPOINT_SELECTION_FORMAT = "robot-vla-e012-checkpoint-selection/v1"
E012_CHECKPOINT_EPOCHS = (10, 20, 30)


@dataclass(frozen=True)
class E012CheckpointCandidate:
    label: str
    epoch: int
    validation_total_loss: float
    full_chain: tuple[RolloutEpisodeResult, ...]
    atomic: tuple[AtomicSkillEpisodeResult, ...]

    def __post_init__(self) -> None:
        if not self.label or self.epoch not in E012_CHECKPOINT_EPOCHS:
            raise ValueError("E012 checkpoint candidate label/epoch 无效")
        if (
            not math.isfinite(self.validation_total_loss)
            or self.validation_total_loss < 0.0
        ):
            raise ValueError("E012 checkpoint validation total loss 必须是有限非负数")
        if not self.full_chain or not self.atomic:
            raise ValueError("E012 checkpoint candidate 缺少 full-chain/atomic 结果")


def _candidate_assessment(
    baseline_full_chain: tuple[RolloutEpisodeResult, ...],
    baseline_atomic: tuple[AtomicSkillEpisodeResult, ...],
    candidate: E012CheckpointCandidate,
) -> dict[str, Any]:
    paired = analyze_e012_pair(
        list(baseline_full_chain),
        list(candidate.full_chain),
        replay_atomic=list(baseline_atomic),
        dagger_atomic=list(candidate.atomic),
        protocol="checkpoint-validation",
    )
    full = paired["full_chain"]
    atomic = paired["atomic_guardrail"]
    if atomic is None:
        raise RuntimeError("checkpoint selection 缺少 atomic guardrail")
    full_issues = full["issue_deltas"]
    atomic_issues = atomic["issue_deltas"]
    atomic_groups = atomic["by_skill"]
    required_atomic = ("grasp", "lift", "place")
    if any(skill not in atomic_groups for skill in required_atomic):
        raise ValueError("checkpoint selection 缺少 Grasp/Lift/Place atomic 结果")
    checks = {
        "no_new_system_failure": full_issues["new_system_episodes"] == 0
        and atomic_issues["new_system_episodes"] == 0,
        "no_new_action_safety_rejection": full_issues["new_safety_episodes"] == 0
        and atomic_issues["new_safety_episodes"] == 0,
        "tracking_not_regressed": full_issues[
            "tracking_correction_saturation_count_delta"
        ]
        <= 0
        and atomic_issues["tracking_correction_saturation_count_delta"] <= 0,
        "anomaly_not_regressed": full_issues["anomaly_replan_count_delta"] <= 0
        and atomic_issues["anomaly_replan_count_delta"] <= 0,
        "atomic_grasp_not_regressed": atomic_groups["grasp"][
            "net_dagger_wins"
        ]
        >= 0,
        "atomic_lift_not_regressed": atomic_groups["lift"]["net_dagger_wins"]
        >= 0,
        "atomic_place_not_regressed": atomic_groups["place"]["net_dagger_wins"]
        >= 0,
        "full_chain_reach_delta_at_least_minus_1": full[
            "unconditional_paired"
        ]["reach"]["net_dagger_wins"]
        >= -1,
    }
    dagger = full["dagger"]
    ranking_metrics = {
        "unconditional_lift_successes": dagger["unconditional"]["lift"][
            "numerator"
        ],
        "unconditional_grasp_successes": dagger["unconditional"]["grasp"][
            "numerator"
        ],
        "mean_completed_skill_count": dagger["mean_completed_skill_count"],
        "validation_total_loss": candidate.validation_total_loss,
        "epoch": candidate.epoch,
    }
    return {
        "label": candidate.label,
        "epoch": candidate.epoch,
        "eligible": all(checks.values()),
        "exclusion_checks": checks,
        "ranking_metrics": ranking_metrics,
        "paired_evidence": paired,
    }


def _ranking_key(assessment: dict[str, Any]) -> tuple[Any, ...]:
    metrics = assessment["ranking_metrics"]
    return (
        -int(metrics["unconditional_lift_successes"]),
        -int(metrics["unconditional_grasp_successes"]),
        -float(metrics["mean_completed_skill_count"]),
        float(metrics["validation_total_loss"]),
        int(metrics["epoch"]),
        str(assessment["label"]),
    )


def select_e012_checkpoint(
    *,
    baseline_full_chain: tuple[RolloutEpisodeResult, ...],
    baseline_atomic: tuple[AtomicSkillEpisodeResult, ...],
    candidates: tuple[E012CheckpointCandidate, ...],
) -> dict[str, Any]:
    if not baseline_full_chain or not baseline_atomic:
        raise ValueError("E012 checkpoint selection 缺少 pi_0 baseline")
    labels = [candidate.label for candidate in candidates]
    epochs = [candidate.epoch for candidate in candidates]
    if len(labels) != len(set(labels)):
        raise ValueError("E012 checkpoint candidate label 重复")
    if sorted(epochs) != list(E012_CHECKPOINT_EPOCHS):
        raise ValueError("E012 checkpoint selection 必须精确包含 epoch 10/20/30")
    assessments = [
        _candidate_assessment(baseline_full_chain, baseline_atomic, candidate)
        for candidate in candidates
    ]
    eligible = sorted(
        (assessment for assessment in assessments if assessment["eligible"]),
        key=_ranking_key,
    )
    return {
        "format": E012_CHECKPOINT_SELECTION_FORMAT,
        "selection_gate_passed": bool(eligible),
        "selected": None
        if not eligible
        else {
            "label": eligible[0]["label"],
            "epoch": eligible[0]["epoch"],
            "ranking_metrics": eligible[0]["ranking_metrics"],
        },
        "eligible_ranking": [assessment["label"] for assessment in eligible],
        "candidates": sorted(assessments, key=lambda row: row["epoch"]),
    }


__all__ = [
    "E012_CHECKPOINT_EPOCHS",
    "E012_CHECKPOINT_SELECTION_FORMAT",
    "E012CheckpointCandidate",
    "select_e012_checkpoint",
]
