#!/usr/bin/env python3
"""独立核对 E012 repeat-1 compact summary 与正式 checkpoint selection receipts。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: 顶层必须是 object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_projection(candidate: dict[str, Any]) -> dict[str, Any]:
    full_chain = candidate["paired_evidence"]["full_chain"]
    atomic = candidate["paired_evidence"]["atomic_guardrail"]
    ranking = candidate["ranking_metrics"]
    issues = full_chain["issue_deltas"]
    candidate_metrics = full_chain["dagger"]
    baseline_metrics = full_chain["replay"]
    return {
        "label": candidate["label"],
        "epoch": candidate["epoch"],
        "eligible": candidate["eligible"],
        "failed_checks": [
            key for key, passed in candidate["exclusion_checks"].items() if not passed
        ],
        "full_chain_paired_net_wins": {
            skill: full_chain["unconditional_paired"][skill]["net_dagger_wins"]
            for skill in ("reach", "grasp", "lift")
        },
        "atomic_place_paired_net_wins": atomic["by_skill"]["place"][
            "net_dagger_wins"
        ],
        "mean_completed_skills": ranking["mean_completed_skill_count"],
        "mean_completed_skills_delta": full_chain["mean_completed_skill_count_delta"],
        "unconditional_grasp_successes": ranking["unconditional_grasp_successes"],
        "unconditional_lift_successes": ranking["unconditional_lift_successes"],
        "full_successes": candidate_metrics["unconditional"]["full"]["numerator"],
        "full_chain_episodes": candidate_metrics["episodes"],
        "validation_total_loss": ranking["validation_total_loss"],
        "new_anomaly_episodes": issues["new_anomaly_episodes"],
        "anomaly_replan_count_delta": issues["anomaly_replan_count_delta"],
        "new_tracking_episodes": issues["new_tracking_episodes"],
        "tracking_correction_saturation_count_delta": issues[
            "tracking_correction_saturation_count_delta"
        ],
        "baseline_mean_completed_skills": baseline_metrics["mean_completed_skill_count"],
    }


def _assert_value_equal(observed: Any, expected: Any, path: str) -> None:
    if isinstance(observed, float) or isinstance(expected, float):
        if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"{path}: {observed!r} != {expected!r}")
        return
    if observed != expected:
        raise AssertionError(f"{path}: {observed!r} != {expected!r}")


def main() -> None:
    root = Path(__file__).resolve().parent
    summary = _load_object(root / "summary.json")
    rows_by_label = {row["label"]: row for row in summary["candidate_results"]}

    observed_labels: set[str] = set()
    for arm, file_name in (
        ("pi_replay", "pi-replay-selection.json"),
        ("pi_dagger", "pi-dagger-selection.json"),
    ):
        selection_path = root / file_name
        selection = _load_object(selection_path)
        receipt = summary["checkpoint_validation"]["selection_receipts"][arm]
        assert receipt["path"].endswith(file_name)
        assert _sha256(selection_path) == receipt["sha256"]
        assert selection["selection_gate_passed"] is summary["selection"][arm][
            "selection_gate_passed"
        ]
        assert selection["selected"] == summary["selection"][arm]["selected"]
        assert selection["eligible_ranking"] == summary["selection"][arm][
            "eligible_ranking"
        ]

        for candidate in selection["candidates"]:
            projected = _candidate_projection(candidate)
            label = projected.pop("label")
            observed_labels.add(label)
            expected = rows_by_label[label]
            baseline_mean = projected.pop("baseline_mean_completed_skills")
            assert math.isclose(baseline_mean, 0.8, rel_tol=0.0, abs_tol=1e-12)
            assert expected["arm"] == arm
            for key, value in projected.items():
                _assert_value_equal(value, expected[key], f"{label}.{key}")

    assert observed_labels == set(rows_by_label)
    assert len(rows_by_label) == 6
    assert all(not row["eligible"] for row in rows_by_label.values())
    assert all(row["full_successes"] == 0 for row in rows_by_label.values())
    assert summary["selection"]["pi_replay"]["selected"] is None
    assert summary["selection"]["pi_dagger"]["selected"] is None
    assert summary["checkpoint_validation"]["outputs"] == 14
    assert summary["checkpoint_validation"]["episodes"] == 315
    assert sum(
        summary["paired_training"]["pi_dagger"]["aggregate_source_exposure"].values()
    ) == summary["paired_training"]["pi_dagger"]["samples"]

    print("verified 2 selection receipts, 6 candidates, and compact summary invariants")


if __name__ == "__main__":
    main()
