"""验证 E016-P0 GitHub 脱敏结果的 hash、门禁与安全边界。"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

VERSION = "e016-p0-corrected-observability-public/v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} 必须是 JSON object")
    return value


def _scan(value: Any) -> None:
    banned_keys = {
        "trajectory_id",
        "scene_id",
        "dataset_index",
        "timestep",
        "rgb_wrist",
        "goal_mask",
        "object_mask",
    }
    if isinstance(value, Mapping):
        overlap = banned_keys & set(value)
        if overlap:
            raise RuntimeError(f"E016-P0 public JSON 含私有字段: {sorted(overlap)}")
        for child in value.values():
            _scan(child)
    elif isinstance(value, list):
        for child in value:
            _scan(child)
    elif isinstance(value, str) and value.startswith(("/home/", "/mnt/", "C:\\", "D:\\")):
        raise RuntimeError("E016-P0 public JSON 含敏感绝对路径")


def verify(root: Path) -> None:
    summary = _read_json(root / "sanitized_summary.json")
    receipt = _read_json(root / "receipt.json")
    if summary.get("version") != VERSION or receipt.get("version") != VERSION:
        raise RuntimeError("E016-P0 public version 漂移")
    if summary.get("status") != "complete" or receipt.get("status") != "complete":
        raise RuntimeError("E016-P0 public status 漂移")
    _scan(summary)
    _scan(receipt)
    source = summary["source"]
    if COMMIT_PATTERN.fullmatch(source["commit"]) is None:
        raise RuntimeError("E016-P0 source commit 无效")
    for name in ("source_tree_sha256", "training_config_sha256"):
        if SHA256_PATTERN.fullmatch(source[name]) is None:
            raise RuntimeError(f"E016-P0 {name} 无效")
    protocol = summary["protocol"]
    if protocol["included_splits"] != ["train", "val"]:
        raise RuntimeError("E016-P0 只能包含 train/val")
    if protocol["excluded_splits"] != ["test"]:
        raise RuntimeError("E016-P0 必须排除 test")
    if protocol["test_label_array_read_count"] != 0:
        raise RuntimeError("E016-P0 禁止读取 test label arrays")
    if protocol["test_samples_used_for_training_or_selection"] is not False:
        raise RuntimeError("E016-P0 禁止使用 test 调参")
    if protocol["e015_test_used_for_tuning"] is not False:
        raise RuntimeError("E016-P0 禁止复用 E015 test 调参")
    if protocol["formal_checkpoint_written"] is not False:
        raise RuntimeError("E016-P0 禁止写入正式 checkpoint")
    if protocol["actuation_allowed"] is not False:
        raise RuntimeError("E016-P0 禁止 actuation")
    if summary["formal_training_performed"] is not False:
        raise RuntimeError("E016-P0 不是正式训练")
    if summary["safe_for_actuator_promotion"] is not False:
        raise RuntimeError("E016-P0 不得授权 actuator promotion")
    contract = summary["loss_contract"]
    if contract["passed"] is not True or contract["all_negative_batch_finite"] is not True:
        raise RuntimeError("E016-P0 loss contract 未通过")
    if any(value != 0.0 for value in contract["localization_losses"].values()):
        raise RuntimeError("E016-P0 unobservable localization loss 非零")
    if any(
        value != 0.0
        for value in contract["localization_output_gradient_abs_max"].values()
    ):
        raise RuntimeError("E016-P0 unobservable localization gradient 非零")
    overfit = summary["stratified_overfit"]
    if overfit["passed"] is not True or not all(overfit["gates"].values()):
        raise RuntimeError("E016-P0 stratified overfit gate 未通过")
    final = overfit["final"]
    if final["goal_observable_normalized_uv_mae"] > 0.01:
        raise RuntimeError("E016-P0 goal UV MAE gate 漂移")
    if final["goal_mask_iou"] < 0.75:
        raise RuntimeError("E016-P0 goal mask IoU gate 漂移")
    if final["goal_visibility_precision"] < 0.95:
        raise RuntimeError("E016-P0 visibility precision gate 漂移")
    if final["goal_visibility_recall"] < 0.95:
        raise RuntimeError("E016-P0 visibility recall gate 漂移")
    if final["goal_unobservable_false_positive_rate"] > 0.05:
        raise RuntimeError("E016-P0 unobservable FPR gate 漂移")
    preflight = summary["full_preflight"]
    if preflight["passed"] is not True or preflight["epochs_completed"] != 3:
        raise RuntimeError("E016-P0 full preflight 未完成")
    if preflight["checkpoint_persisted"] is not False:
        raise RuntimeError("E016-P0 preflight 不得保存 checkpoint")
    for name, expected in receipt["files"].items():
        if _sha256(root / name) != expected:
            raise RuntimeError(f"E016-P0 public file SHA-256 漂移: {name}")
    readme = (root / "README.md").read_text(encoding="utf-8")
    if any(token in readme for token in ("/home/", "/mnt/", "C:\\", "D:\\")):
        raise RuntimeError("E016-P0 README 含敏感绝对路径")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) == 2 else Path(__file__).resolve().parent
    verify(root)
    print(json.dumps({"status": "passed", "version": VERSION}, sort_keys=True))


if __name__ == "__main__":
    main()
