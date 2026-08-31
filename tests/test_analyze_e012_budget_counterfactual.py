from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_e012_budget_counterfactual import (
    ANALYSIS_FORMAT,
    _beta_binomial_tail,
    _binomial_tail,
    build_analysis,
)


def test_capacity_probability_reference_values() -> None:
    assert _binomial_tail(120, 20, 0.15) == pytest.approx(0.3413833174506157)
    assert _binomial_tail(200, 20, 0.15) == pytest.approx(0.9851197019379513)
    assert _binomial_tail(220, 20, 0.15) == pytest.approx(0.9965508183196291)
    assert _binomial_tail(240, 20, 0.15) == pytest.approx(0.9993103698930453)
    assert _beta_binomial_tail(200, 20, alpha=16.0, beta=86.0) == pytest.approx(
        0.9218174207338423
    )
    assert _beta_binomial_tail(220, 20, alpha=15.5, beta=85.5) == pytest.approx(
        0.9469609870832042
    )


def test_published_compact_results_recompute_decision_metrics() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_root = project_root / "docs" / "results" / "e012"
    analysis = build_analysis(
        collection_summary_path=results_root / "collection_summary.json",
        smoke_root=results_root / "segmented-budget-smoke",
        counterfactual_root=results_root / "segmented-budget-counterfactual",
    )

    assert analysis["format"] == ANALYSIS_FORMAT
    assert analysis["integrity"] == {
        "smoke_passed": True,
        "counterfactual_complete": True,
        "counterfactual_blocked": False,
        "classification_additivity": {
            "classified_candidates": 16,
            "completed_candidates": 16,
            "holds": True,
        },
        "all_prefix_aligned": True,
        "engineering_error_count": 0,
        "prefix_mismatch_count": 0,
        "successful_npz_may_enter_d1": False,
        "candidate_record_audit": {
            "available": False,
            "reason": "published compact result does not contain candidate record directories",
        },
    }
    observed = analysis["observed_counterfactual"]
    assert observed["classification_counts"] == {
        "engineering_error": 0,
        "expert_completed_but_snapshot_or_paired_gate_failed": 1,
        "expert_recovery_budget_exhausted": 4,
        "other_behavioral_rejection": 6,
        "prefix_mismatch": 0,
        "recovered_full_eligible": 5,
    }
    assert observed["recovered_full_eligible_rate"] == pytest.approx(5 / 16)
    assert observed["behavior_completed_before_gate_rate"] == pytest.approx(6 / 16)
    assert observed["hard_deadline_count"] == 0
    assert observed["recovered_expert_actions"] == {
        "count": 5,
        "values": [117, 121, 122, 136, 161],
        "min": 117,
        "median": 122,
        "mean": 131.4,
        "max": 161,
    }
    planning = analysis["capacity_planning"]
    assert planning["retrospective_planning_eligible"] == 15
    assert planning["planning_point_rate"] == pytest.approx(0.15)
    assert planning["strict_low_risk_option"] == {
        "pool_size": 240,
        "seed_start": 30_200,
        "seed_end_exclusive": 30_440,
        "reason": "rounded above the Jeffreys Beta-binomial 95% threshold of 223",
    }
    assert planning["minimum_pool_sizes_by_model"][-1] == {
        "target_probability": 0.95,
        "fixed_p": 182,
        "beta_binomial_uniform": 216,
        "beta_binomial_jeffreys": 223,
    }
