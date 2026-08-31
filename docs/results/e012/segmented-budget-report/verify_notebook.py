#!/usr/bin/env python3
"""用 Python 标准库执行 notebook code cells，并核对 report snapshot。"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: 顶层必须是 object")
    return value


def main() -> None:
    report_root = Path(__file__).resolve().parent
    notebook = _load_object(report_root / "reproduce.ipynb")
    namespace: dict[str, Any] = {"__name__": "__e012_notebook__"}
    executed = 0
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        exec(compile(source, f"reproduce.ipynb:cell-{index}", "exec"), namespace)
        executed += 1

    analysis = namespace["analysis"]
    artifact = _load_object(report_root / "artifact.json")
    datasets = artifact["snapshot"]["datasets"]
    headline = datasets["headline"][0]
    counts = analysis["observed_counterfactual"]["classification_counts"]
    planning = analysis["capacity_planning"]

    assert headline["counterfactual_seeds"] == analysis["observed_counterfactual"]["count"]
    assert headline["recovered_full_eligible"] == counts["recovered_full_eligible"]
    assert headline["engineering_errors"] == counts["engineering_error"]
    assert headline["hard_deadline_count"] == analysis["observed_counterfactual"]["hard_deadline_count"]
    assert headline["planning_rate"] == planning["planning_point_rate"]
    assert headline["eligible_gate"] == planning["gate_required_eligible"]
    assert headline["jeffreys_95_threshold"] == planning["minimum_pool_sizes_by_model"][-1][
        "beta_binomial_jeffreys"
    ]

    artifact_counts = {
        row["classification"]: row["count"]
        for row in datasets["counterfactual_outcomes"]
    }
    assert artifact_counts == counts

    artifact_capacity = {row["pool_size"]: row for row in datasets["capacity_options"]}
    for row in planning["pool_options"]:
        snapshot = artifact_capacity[row["pool_size"]]
        assert math.isclose(
            snapshot["expected_eligible"],
            row["expected_eligible_at_point_rate"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert math.isclose(
            snapshot["fixed_p_pass"],
            row["probability_at_least_20_fixed_p"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert math.isclose(
            snapshot["jeffreys_pass"],
            row["probability_at_least_20_beta_binomial_jeffreys"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )

    expected_sql_rows = {
        "e012_headline_projection": 1,
        "e012_outcome_projection": 6,
        "e012_capacity_projection": 6,
        "e012_recovered_actions_projection": 5,
    }
    connection = sqlite3.connect(":memory:")
    try:
        observed_sql_rows = {
            source["id"]: len(connection.execute(source["query"]["sql"]).fetchall())
            for source in artifact["sources"]
            if "query" in source
        }
    finally:
        connection.close()
    assert observed_sql_rows == expected_sql_rows

    print(
        f"verified {executed} notebook code cells, report snapshot, "
        f"and {len(observed_sql_rows)} SQL projections"
    )


if __name__ == "__main__":
    main()
