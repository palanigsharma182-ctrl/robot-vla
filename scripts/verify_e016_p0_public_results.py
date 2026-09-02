"""把 E016-P0 public 聚合逐字段核对到 canonical private artifacts。"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} 必须是 JSON object")
    return value


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise RuntimeError(f"E016-P0 public/private {name} 不一致: {actual!r} != {expected!r}")


def _subset(value: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    return {name: value[name] for name in names}


def verify(public_root: Path, private_root: Path) -> None:
    summary = _json(public_root / "sanitized_summary.json")
    public_receipt = _json(public_root / "receipt.json")
    private_receipt = _json(private_root / "receipt.json")
    audit = _json(private_root / "corrected_sidecar_audit.json")
    contract = _json(private_root / "loss_contract.json")
    overfit = _json(private_root / "stratified_overfit.json")
    preflight = _json(private_root / "full_preflight.json")
    subset = _json(private_root / "stratified_subset.json")
    config = _json(private_root / "config_snapshot.json")

    for name, expected in public_receipt["canonical_private_artifacts"].items():
        _require_equal(_sha256(private_root / name), expected, f"private hash {name}")
    for name, expected in public_receipt["files"].items():
        _require_equal(_sha256(public_root / name), expected, f"public hash {name}")
    if list(private_root.rglob("*.pt")):
        raise RuntimeError("E016-P0 private root 不得包含 checkpoint")
    _require_equal(private_receipt["passed"], True, "private receipt passed")
    _require_equal(private_receipt["test_split_read"], False, "test split read")
    _require_equal(
        private_receipt["e015_test_used_for_tuning"],
        False,
        "E015 test tuning",
    )
    _require_equal(
        private_receipt["formal_checkpoint_written"],
        False,
        "formal checkpoint",
    )

    public_audit = summary["corrected_label_audit"]
    _require_equal(
        public_audit,
        {
            "corrected_data_identity_sha256": audit["corrected_data_identity_sha256"],
            "goal_exists_count": audit["goal_exists_count"],
            "goal_observable_count": audit["goal_observable_count"],
            "goal_projection_valid_count": audit["goal_projection_valid_count"],
            "goal_unobservable_count": audit["goal_unobservable_count"],
            "legacy_goal_visible_count": audit["legacy_goal_visible_count"],
            "legacy_visible_but_unobservable_count": audit[
                "legacy_visible_but_unobservable_count"
            ],
            "occlusion_counts": audit["occlusion_counts"],
            "sample_counts": audit["sample_counts"],
            "schema_version": audit["schema_version"],
            "trajectory_counts": audit["trajectory_counts"],
        },
        "corrected label audit",
    )
    public_contract = summary["loss_contract"]
    _require_equal(public_contract["passed"], contract["passed"], "loss contract passed")
    _require_equal(
        public_contract["all_negative_batch_finite"],
        contract["all_negative_batch_finite"],
        "all-negative finite",
    )
    _require_equal(
        public_contract["localization_losses"],
        contract["localization_losses"],
        "localization losses",
    )
    _require_equal(
        public_contract["localization_output_gradient_abs_max"],
        {
            "heatmap_logits": contract["localization_output_gradient_abs_max"][
                "heatmap_logits_abs_max"
            ],
            "keypoint_log_variance": contract[
                "localization_output_gradient_abs_max"
            ]["keypoint_log_variance_abs_max"],
            "subpixel_offsets": contract["localization_output_gradient_abs_max"][
                "subpixel_offsets_abs_max"
            ],
        },
        "localization gradients",
    )

    public_overfit = summary["stratified_overfit"]
    _require_equal(public_overfit["passed"], overfit["passed"], "overfit passed")
    _require_equal(public_overfit["gates"], overfit["gates"], "overfit gates")
    _require_equal(
        public_overfit["optimizer_steps"], overfit["optimizer_steps"], "overfit steps"
    )
    _require_equal(
        public_overfit["sample_count"], overfit["sample_count"], "overfit samples"
    )
    _require_equal(
        public_overfit["strata_counts"], subset["strata_counts"], "overfit strata"
    )
    _require_equal(
        public_overfit["goal_normalized_uv_improvement"],
        overfit["goal_normalized_uv_improvement"],
        "goal UV improvement",
    )
    overfit_fields = (
        "goal_mask_iou",
        "goal_observable_normalized_uv_mae",
        "goal_unobservable_false_positive_rate",
        "goal_visibility_precision",
        "goal_visibility_recall",
        "projection_accuracy",
    )
    _require_equal(
        _subset(public_overfit["initial"], overfit_fields),
        _subset(overfit["initial"], overfit_fields),
        "overfit initial metrics",
    )
    final_fields = (
        *overfit_fields,
        "goal_observable_pixel_error_p50",
        "goal_observable_pixel_error_p90",
        "goal_visibility_f1",
        "object_mask_iou",
        "object_normalized_uv_mae",
    )
    _require_equal(
        _subset(public_overfit["final"], final_fields),
        _subset(overfit["final"], final_fields),
        "overfit final metrics",
    )

    public_preflight = summary["full_preflight"]
    _require_equal(public_preflight["passed"], preflight["passed"], "preflight passed")
    _require_equal(
        public_preflight["epochs_completed"],
        preflight["epochs_completed"],
        "preflight epochs",
    )
    _require_equal(
        public_preflight["optimizer_steps"],
        preflight["optimizer_steps"],
        "preflight steps",
    )
    preflight_fields = (
        "goal_mask_iou",
        "goal_observable_normalized_uv_mae",
        "goal_unobservable_false_positive_rate",
        "goal_visibility_f1",
        "goal_visibility_precision",
        "goal_visibility_recall",
        "object_mask_iou",
        "object_normalized_uv_mae",
        "projection_accuracy",
    )
    _require_equal(
        public_preflight["final_validation"],
        _subset(preflight["epochs"][-1]["validation"], preflight_fields),
        "preflight final validation",
    )
    _require_equal(
        summary["source"]["training_config_sha256"],
        private_receipt["training_config_sha256"],
        "training config identity",
    )
    _require_equal(
        summary["source"]["source_tree_sha256"],
        private_receipt["source_tree_sha256"],
        "source tree identity",
    )
    _require_equal(config["execution"]["persist_checkpoint"], False, "config checkpoint")
    _require_equal(config["execution"]["actuation_allowed"], False, "config actuation")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_e016_p0_public_results.py PUBLIC_ROOT PRIVATE_ROOT")
    verify(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps({"status": "passed", "version": "e016-p0-public-private-check/v1"}))


if __name__ == "__main__":
    main()
