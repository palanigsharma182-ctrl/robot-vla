#!/usr/bin/env python3
"""独立核对 E013 公开 compact summary 与报告 snapshot 的关键停止边界。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "summary.json"
ARTIFACT_PATH = ROOT / "report" / "artifact.json"
REPORT_PATH = ROOT / "report" / "report.html"
DELIVERY_RECEIPT_PATH = ROOT / "report" / "delivery_receipt.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_close(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15), (actual, expected)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    summary = _load(SUMMARY_PATH)

    steps = summary["step_gates"]
    assert isinstance(steps, list) and len(steps) == 9
    assert [step["status"] for step in steps[:8]] == ["passed"] * 8
    assert steps[8] == {
        "step": 9,
        "name": "paired-20hz-no-actuation-shadow",
        "status": "failed",
        "stop_promotion": True,
    }

    dataset = summary["dataset"]
    assert dataset["trajectory_count"] == 40
    assert dataset["sample_count"] == 7987
    assert sum(dataset["split_sample_counts"].values()) == 7987
    assert dataset["model_input_privileged_overlap"] == []
    assert dataset["modality_invalid_count"] == 0
    assert dataset["action_semantic_parity_steps"] == 7987
    assert dataset["oracle_roundtrip_error_m"]["invalid_count"] == 0

    training = summary["formal_training"]
    metrics = summary["training_metrics"]
    selected = min(metrics, key=lambda row: (row["val_normalized_uv_mae"], row["epoch"]))
    assert selected["epoch"] == training["selected_epoch"] == 4
    _assert_close(
        selected["val_normalized_uv_mae"],
        training["selected_val_normalized_uv_mae"],
    )
    assert training["total_examples"] == 95520
    assert training["total_optimizer_steps"] == 3000
    assert training["motion_head_parameter_identity_unchanged"] is True

    held_out = summary["held_out_test"]
    assert held_out["sample_count"] == 2044
    assert held_out["valid_keypoint_count"] == 3307
    assert held_out["perception_gate_passed"] is True
    assert held_out["invalid_backprojection_count"] == 0
    _assert_close(held_out["world_xy_error_m"]["p90"], 0.0015172345098108057)
    assert held_out["world_xy_error_m"]["max"] > 0.2

    latency = summary["full_history_provider_latency"]
    assert latency["full_history_calls"] == latency["measurement_calls"] == 200
    assert latency["predicted_frames"] == 800
    assert latency["failed_calls"] == 0
    assert latency["latency_gate_passed"] is True

    shadow = summary["formal_shadow"]
    assert shadow["requested_paired_episodes"] == 100
    assert shadow["completed_baseline_episodes"] == 95
    assert shadow["completed_shadow_episodes"] == 95
    assert shadow["paired_episode_count"] == 95
    assert shadow["baseline_failure_count"] == shadow["shadow_failure_count"] == 5
    assert shadow["failure_seed_sets_equal"] is True
    assert shadow["action_mismatch_count"] == 0
    assert shadow["commanded_target_mismatch_count"] == 0
    assert shadow["episode_length_mismatch_count"] == 0
    assert shadow["provider_failure_count"] == shadow["observer_error_count"] == 0
    assert shadow["deadline_miss_count"] == 7
    _assert_close(shadow["deadline_miss_rate"], 7 / 19100)
    assert shadow["predicted_frames"] == 4 * shadow["provider_calls"] - 6 * 95
    assert shadow["prefix_padding_frame_equation_holds"] is True
    assert shadow["actuation_allowed"] is False
    assert shadow["gate_passed"] is False
    assert shadow["promotion_stopped"] is True

    hashes = summary["artifact_sha256"]
    assert hashes
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes.values())

    serialized = SUMMARY_PATH.read_text(encoding="utf-8")
    for forbidden in ("/home/", "/mnt/", "C:\\\\", "stdout.log", "stderr.log"):
        assert forbidden not in serialized

    if ARTIFACT_PATH.is_file():
        artifact = _load(ARTIFACT_PATH)
        assert artifact["surface"] == "report"
        assert artifact["manifest"]["title"] == "E013 Precision v1：感知通过，正式 shadow 停止 promotion"
        snapshot = artifact["snapshot"]
        assert snapshot["status"] == "ready"
        shadow_rows = snapshot["datasets"]["shadow_completion"]
        actual = {row["arm"]: row["episodes"] for row in shadow_rows if row["measure"] == "actual"}
        assert actual == {"Baseline": 95, "Shadow": 95}
        assert artifact["manifest"]["sources"] == artifact["sources"]

    if DELIVERY_RECEIPT_PATH.is_file():
        delivery = _load(DELIVERY_RECEIPT_PATH)
        assert delivery["validation"] == delivery["package"] == "passed"
        assert delivery["verification"] == "structural_only"
        assert delivery["artifact_sha256"] == _sha256(ARTIFACT_PATH)
        assert delivery["report_html_sha256"] == _sha256(REPORT_PATH)

    print(
        "E013 public summary verified: steps 1-8 passed; step 9 failed; "
        "promotion stopped; no actuation claim."
    )


if __name__ == "__main__":
    main()
