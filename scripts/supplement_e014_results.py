"""从已完成的 E014 frozen 输出确定性补齐 V1.0 诊断字段与报告。

本脚本不会加载 checkpoint 或执行模型 forward，也不会重新分类细粒度 taxonomy。
它只补充可由既有输出精确恢复的字段、五类 failure family 汇总和图标题。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla.cli.analyze_precision_outliers import (
    _atomic_json,
    _atomic_jsonl,
    _atomic_text,
    _plot_top20,
    _public_maximum,
    _readme,
    _SampleArtifact,
)
from robot_vla.precision.data import (
    PrecisionRGBDataset,
    canonical_sha256,
    file_sha256,
)
from robot_vla.precision.outliers import (
    E014_DIAGNOSTIC_VERSION,
    failure_family,
    failure_family_counts,
    taxonomy_counts,
)

SUPPLEMENT_VERSION = "e014-precision-long-tail-supplement/v1"
SUPPLEMENT_SOURCE_FILES = (
    "scripts/supplement_e014_results.py",
    "scripts/verify_e014_public_results.py",
    "src/robot_vla/cli/analyze_precision_outliers.py",
    "src/robot_vla/precision/outliers.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} 必须是 JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} 必须是 JSON object")
            rows.append(value)
    return rows


def _supplement_source_sha256(repository_root: Path) -> str:
    files: dict[str, str] = {}
    for relative in SUPPLEMENT_SOURCE_FILES:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"E014 supplement source 不存在: {path}")
        files[relative] = file_sha256(path)
    return canonical_sha256(
        {
            "version": SUPPLEMENT_VERSION,
            "files": files,
        }
    )


def _verify_original_private(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _read_json(root / "receipt.json")
    summary = _read_json(root / "summary_private.json")
    if receipt.get("version") != E014_DIAGNOSTIC_VERSION:
        raise RuntimeError("原始 E014 private receipt version 漂移")
    if receipt.get("status") != "complete" or summary.get("status") != "complete":
        raise RuntimeError("原始 E014 private 输出未完成")
    for name, expected in receipt.get("files", {}).items():
        if file_sha256(root / name) != expected:
            raise RuntimeError(f"原始 E014 private 文件 SHA-256 漂移: {name}")
    if int(receipt.get("top20_figure_count", -1)) != 20:
        raise RuntimeError("原始 E014 Top-20 figure count 漂移")
    if int(receipt.get("top20_heatmap_array_count", -1)) != 20:
        raise RuntimeError("原始 E014 Top-20 heatmap count 漂移")
    return receipt, summary


def _enrich_rows(
    rows: list[dict[str, Any]],
    *,
    heatmap_support: int,
) -> None:
    if len(rows) != 4088:
        raise RuntimeError("E014 必须包含 4,088 行预测")
    log_support = math.log(heatmap_support)
    for row in rows:
        if row.get("schema_version") != E014_DIAGNOSTIC_VERSION:
            raise RuntimeError("E014 per-sample schema version 漂移")
        world_point = row.get("predicted_world_point_base_m")
        if not isinstance(world_point, list) or len(world_point) != 3:
            raise RuntimeError("E014 supplemented 需要有效 predicted world point")
        plane_z = float(world_point[2])
        normalized_entropy = float(row["normalized_entropy"])
        if not math.isfinite(plane_z) or not math.isfinite(normalized_entropy):
            raise RuntimeError("E014 supplemented 字段必须有限")
        row["gt_plane_z_m"] = plane_z
        row["entropy_nats"] = normalized_entropy * log_support
        row["failure_family"] = failure_family(str(row["failure_taxonomy"]))


def _load_top20_artifacts(
    *,
    original_private_root: Path,
    top20: list[dict[str, Any]],
) -> dict[int, _SampleArtifact]:
    artifacts: dict[int, _SampleArtifact] = {}
    for rank, row in enumerate(top20, start=1):
        fingerprint = str(row["sample_fingerprint"])
        path = (
            original_private_root
            / "top20_heatmaps"
            / f"rank-{rank:02d}-{fingerprint}.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(f"原始 E014 heatmap artifact 不存在: {path}")
        with np.load(path, allow_pickle=False) as payload:
            artifacts[int(row["dataset_index"])] = _SampleArtifact(
                heatmap_probability=np.asarray(payload["heatmap_probability"]).copy(),
                predicted_masks=np.asarray(payload["predicted_masks"]).copy(),
            )
    return artifacts


def _write_public(
    *,
    root: Path,
    repository_root: Path,
    summary: dict[str, Any],
    distributions: dict[str, Any],
    original_private_receipt_sha256: str,
    original_public_receipt_sha256: str,
    rules_sha256: str,
    supplement_source_tree_sha256: str,
) -> dict[str, Any]:
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    _atomic_json(root / "sanitized_summary.json", summary)
    _atomic_json(root / "distributions.json", distributions)
    _atomic_text(root / "README.md", _readme(summary))
    verifier = repository_root / "scripts" / "verify_e014_public_results.py"
    shutil.copyfile(verifier, root / "verify_summary.py")
    files = {
        name: file_sha256(root / name)
        for name in (
            "README.md",
            "sanitized_summary.json",
            "distributions.json",
            "verify_summary.py",
        )
    }
    receipt = {
        "version": E014_DIAGNOSTIC_VERSION,
        "status": "complete-supplemented",
        "supplement_version": SUPPLEMENT_VERSION,
        "supplement_policy": "deterministic-postprocessing-no-model-forward/v1",
        "supplement_source_tree_sha256": supplement_source_tree_sha256,
        "original_private_receipt_sha256": original_private_receipt_sha256,
        "original_public_receipt_sha256": original_public_receipt_sha256,
        "rules_sha256": rules_sha256,
        "files": files,
        "contains_raw_rgb": False,
        "contains_raw_heatmaps": False,
        "contains_model_weights": False,
        "contains_sensitive_paths": False,
    }
    _atomic_json(root / "receipt.json", receipt)
    subprocess.run(
        [sys.executable, str(root / "verify_summary.py"), str(root)],
        cwd=repository_root,
        check=True,
    )
    return receipt


def supplement(args: argparse.Namespace) -> dict[str, Any]:
    original_private_root = args.original_private_root.resolve()
    original_public_root = args.original_public_root.resolve()
    output_private_root = args.output_private_root.resolve()
    output_public_root = args.output_public_root.resolve()
    repository_root = args.repository_root.resolve()
    if output_private_root.exists() or output_public_root.exists():
        raise FileExistsError("E014 supplemented 输出已存在，拒绝覆盖")

    _, original_private_summary = _verify_original_private(
        original_private_root
    )
    original_public_receipt_path = original_public_root / "receipt.json"
    _read_json(original_public_receipt_path)
    subprocess.run(
        [sys.executable, str(original_public_root / "verify_summary.py"), str(original_public_root)],
        cwd=repository_root,
        check=True,
    )

    dataset = PrecisionRGBDataset(
        args.deployable_root,
        args.label_root,
        "test",
        cache_size=args.cache_size,
    )
    if len(dataset) != 2044:
        raise RuntimeError("E014 supplemented test sample count 必须为 2,044")
    image = np.asarray(dataset[0]["model_inputs"]["rgb_wrist"])
    if image.ndim != 3:
        raise RuntimeError("E014 supplemented RGB 必须是 HWC")
    heatmap_support = int(image.shape[0] * image.shape[1])

    rows = _read_jsonl(original_private_root / "per_sample.jsonl")
    _enrich_rows(rows, heatmap_support=heatmap_support)
    valid_rows = [row for row in rows if row.get("world_xy_error_m") is not None]
    valid_rows.sort(key=lambda row: (-float(row["world_xy_error_m"]), row["sample_fingerprint"]))
    if len(valid_rows) != 3307:
        raise RuntimeError("E014 supplemented valid keypoint count 必须为 3,307")
    top20 = valid_rows[:20]
    top50 = valid_rows[:50]
    accepted_catastrophic = [
        row
        for row in valid_rows
        if bool(row["confidence_accepted"]) and float(row["world_xy_error_m"]) > 0.020
    ]
    top20_taxonomy = taxonomy_counts(top20)
    top50_taxonomy = taxonomy_counts(top50)
    all_rows_taxonomy = taxonomy_counts(rows)
    if top20_taxonomy != original_private_summary["top20_taxonomy"]:
        raise RuntimeError("E014 supplemented 不允许改变 Top-20 taxonomy")
    if top50_taxonomy != original_private_summary["top50_taxonomy"]:
        raise RuntimeError("E014 supplemented 不允许改变 Top-50 taxonomy")
    if all_rows_taxonomy != original_private_summary["all_rows_taxonomy"]:
        raise RuntimeError("E014 supplemented 不允许改变 all-row taxonomy")

    top20_failure_family = failure_family_counts(top20)
    top50_failure_family = failure_family_counts(top50)
    all_rows_failure_family = failure_family_counts(rows)
    maximum = top20[0]
    rules_document = _read_json(original_private_root / "frozen_rules.json")
    rules_sha256 = str(rules_document["rules_sha256"])
    supplement_source = _supplement_source_sha256(repository_root)

    output_private_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    _atomic_jsonl(output_private_root / "per_sample.jsonl", rows)
    _atomic_jsonl(output_private_root / "top50_worst.jsonl", top50)
    _atomic_jsonl(
        output_private_root / "accepted_catastrophic_failures.jsonl",
        accepted_catastrophic,
    )
    shutil.copyfile(
        original_private_root / "frozen_rules.json",
        output_private_root / "frozen_rules.json",
    )
    artifacts = _load_top20_artifacts(
        original_private_root=original_private_root,
        top20=top20,
    )
    _plot_top20(
        private_root=output_private_root,
        dataset=dataset,
        rows=rows,
        top20=top20,
        artifacts=artifacts,
    )

    private_summary = dict(original_private_summary)
    private_summary.update(
        {
            "status": "complete-supplemented",
            "supplement_version": SUPPLEMENT_VERSION,
            "supplement_policy": "deterministic-postprocessing-no-model-forward/v1",
            "supplement_source_tree_sha256": supplement_source,
            "maximum_outlier": maximum,
            "top20_failure_family": top20_failure_family,
            "top50_failure_family": top50_failure_family,
            "all_rows_failure_family": all_rows_failure_family,
        }
    )
    _atomic_json(output_private_root / "summary_private.json", private_summary)

    original_public_summary = _read_json(original_public_root / "sanitized_summary.json")
    public_summary = dict(original_public_summary)
    public_summary.update(
        {
            "status": "complete-supplemented",
            "supplement_version": SUPPLEMENT_VERSION,
            "supplement_policy": "deterministic-postprocessing-no-model-forward/v1",
            "supplement_source_tree_sha256": supplement_source,
            "maximum_outlier": _public_maximum(maximum),
            "top20_failure_family": top20_failure_family,
            "top50_failure_family": top50_failure_family,
            "all_rows_failure_family": all_rows_failure_family,
        }
    )
    original_distributions = _read_json(original_public_root / "distributions.json")
    distributions = dict(original_distributions)
    distributions.update(
        {
            "supplement_version": SUPPLEMENT_VERSION,
            "top20_failure_family": top20_failure_family,
            "top50_failure_family": top50_failure_family,
        }
    )
    public_receipt = _write_public(
        root=output_public_root,
        repository_root=repository_root,
        summary=public_summary,
        distributions=distributions,
        original_private_receipt_sha256=file_sha256(original_private_root / "receipt.json"),
        original_public_receipt_sha256=file_sha256(original_public_receipt_path),
        rules_sha256=rules_sha256,
        supplement_source_tree_sha256=supplement_source,
    )

    private_files = {
        name: file_sha256(output_private_root / name)
        for name in (
            "per_sample.jsonl",
            "top50_worst.jsonl",
            "accepted_catastrophic_failures.jsonl",
            "frozen_rules.json",
            "summary_private.json",
        )
    }
    private_receipt = {
        "version": E014_DIAGNOSTIC_VERSION,
        "status": "complete-supplemented",
        "supplement_version": SUPPLEMENT_VERSION,
        "supplement_policy": "deterministic-postprocessing-no-model-forward/v1",
        "supplement_source_tree_sha256": supplement_source,
        "original_private_receipt_sha256": file_sha256(original_private_root / "receipt.json"),
        "original_public_receipt_sha256": file_sha256(original_public_receipt_path),
        "rules_sha256": rules_sha256,
        "files": private_files,
        "top20_figure_count": len(list((output_private_root / "top20").glob("*.png"))),
        "top20_heatmap_array_count": len(
            list((output_private_root / "top20_heatmaps").glob("*.npz"))
        ),
        "public_receipt_sha256": canonical_sha256(public_receipt),
    }
    _atomic_json(output_private_root / "receipt.json", private_receipt)
    _atomic_json(
        output_private_root / "run_state.json",
        {
            "version": E014_DIAGNOSTIC_VERSION,
            "status": "complete-supplemented",
            "supplement_version": SUPPLEMENT_VERSION,
            "test_split": "frozen-e013-test",
            "test_split_status_after_e014": "consumed-for-diagnostic-postmortem",
        },
    )
    return {
        "status": "complete-supplemented",
        "prediction_rows": len(rows),
        "valid_keypoints": len(valid_rows),
        "maximum_outlier": _public_maximum(maximum),
        "top20_failure_family": top20_failure_family,
        "accepted_catastrophic_count": len(accepted_catastrophic),
        "public_receipt_sha256": canonical_sha256(public_receipt),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-private-root", type=Path, required=True)
    parser.add_argument("--original-public-root", type=Path, required=True)
    parser.add_argument("--output-private-root", type=Path, required=True)
    parser.add_argument("--output-public-root", type=Path, required=True)
    parser.add_argument("--deployable-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--cache-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    result = supplement(_parse_args())
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
