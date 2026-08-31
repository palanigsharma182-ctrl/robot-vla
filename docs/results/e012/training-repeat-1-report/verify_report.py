#!/usr/bin/env python3
"""核对 E012 repeat-1 report snapshot、SQL projection 与 portable payload。"""

from __future__ import annotations

import base64
import gzip
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: 顶层必须是 object")
    return value


def _query_rows(sql: str) -> list[dict[str, Any]]:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(sql).fetchall()]
    finally:
        connection.close()


def _embedded_artifact(report_html: str) -> dict[str, Any]:
    match = re.search(
        r'<template id="data-analytics-portable-artifact-payload-source" '
        r'data-compression="gzip-base64">\s*(.*?)\s*</template>',
        report_html,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError("portable report 缺少 embedded artifact payload")
    encoded = "".join(match.group(1).split())
    value = json.loads(gzip.decompress(base64.b64decode(encoded)).decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("embedded artifact 顶层必须是 object")
    return value


def main() -> None:
    report_root = Path(__file__).resolve().parent
    summary = _load_object(report_root.parent / "checkpoint-validation" / "summary.json")
    artifact = _load_object(report_root / "artifact.json")
    datasets = artifact["snapshot"]["datasets"]

    arm_labels = {"pi_replay": "replay", "pi_dagger": "Dagger"}
    candidates = sorted(
        summary["candidate_results"],
        key=lambda row: (0 if row["arm"] == "pi_replay" else 1, row["epoch"]),
    )
    expected_metrics: list[dict[str, Any]] = []
    expected_long: list[dict[str, Any]] = []
    outcome_fields = (
        (1, "Full Reach", "full_reach_net"),
        (2, "Full Grasp", "full_grasp_net"),
        (3, "Full Lift", "full_lift_net"),
        (4, "Atomic Place", "atomic_place_net"),
    )
    for candidate_order, row in enumerate(candidates, start=1):
        display = f"{arm_labels[row['arm']]} e{row['epoch']}"
        failed = "; ".join(row["failed_checks"])
        nets = row["full_chain_paired_net_wins"]
        projected = {
            "candidate_order": candidate_order,
            "candidate": display,
            "arm": row["arm"],
            "epoch": row["epoch"],
            "eligible": "true" if row["eligible"] else "false",
            "failed_checks": failed,
            "full_reach_net": nets["reach"],
            "full_grasp_net": nets["grasp"],
            "full_lift_net": nets["lift"],
            "atomic_place_net": row["atomic_place_paired_net_wins"],
            "mean_skills": row["mean_completed_skills"],
            "mean_skills_delta": row["mean_completed_skills_delta"],
            "grasp_successes": row["unconditional_grasp_successes"],
            "lift_successes": row["unconditional_lift_successes"],
            "full_successes": row["full_successes"],
            "validation_total_loss": row["validation_total_loss"],
            "new_anomaly_episodes": row["new_anomaly_episodes"],
            "anomaly_replans_delta": row["anomaly_replan_count_delta"],
            "new_tracking_episodes": row["new_tracking_episodes"],
            "tracking_saturation_delta": row[
                "tracking_correction_saturation_count_delta"
            ],
        }
        expected_metrics.append(projected)
        for outcome_order, outcome, field in outcome_fields:
            expected_long.append(
                {
                    "candidate_order": candidate_order,
                    "candidate": display,
                    "outcome_order": outcome_order,
                    "outcome": outcome,
                    "net_wins": projected[field],
                    "failed_checks": failed,
                }
            )

    assert datasets["candidate_metrics"] == expected_metrics
    assert datasets["candidate_net_wins"] == expected_long

    expected_training = []
    for arm in ("pi_replay", "pi_dagger"):
        row = summary["paired_training"][arm]
        exposure = row["aggregate_source_exposure"]
        expected_training.append(
            {
                "arm": arm,
                "epochs": row["epochs"],
                "examples": row["samples"],
                "optimizer_steps": row["optimizer_steps"],
                "base_d0_exposure": exposure["base_d0"],
                "rg_exposure": exposure.get("dagger_reach_grasp", 0),
                "gl_exposure": exposure.get("dagger_grasp_lift", 0),
                "verifier": "passed",
            }
        )
    assert datasets["training_audit"] == expected_training

    sources = {source["id"]: source for source in artifact["sources"]}
    manifest_sources = {source["id"]: source for source in artifact["manifest"]["sources"]}
    assert sources == manifest_sources
    assert _query_rows(sources["e012_candidate_net_projection"]["query"]["sql"]) == expected_long
    assert _query_rows(sources["e012_candidate_projection"]["query"]["sql"]) == expected_metrics
    assert _query_rows(sources["e012_training_projection"]["query"]["sql"]) == expected_training

    embedded = _embedded_artifact((report_root / "report.html").read_text(encoding="utf-8"))
    for key in ("surface", "manifest", "snapshot", "sources"):
        assert embedded[key] == artifact[key]
    assert embedded["package_info"]["deliveryMode"] == "portable_html"
    assert embedded["package_info"]["readOnly"] is True

    assert summary["selection"]["pi_replay"]["selected"] is None
    assert summary["selection"]["pi_dagger"]["selected"] is None
    assert summary["promotion"]["stage_a"]["status"] == "not_run"
    print(
        "verified compact summary, 3 report datasets, 3 SQL projections, "
        "and embedded portable payload"
    )


if __name__ == "__main__":
    main()
