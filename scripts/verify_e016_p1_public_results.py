"""验证 E016-P1 GitHub 脱敏结果的 schema、hash、test-once 与 no-actuation 边界。"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

VERSION = "e016-p1-formal-precision-public/v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


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
        "gt_goal_position_base_m",
        "predicted_goal_position_base_m",
    }
    if isinstance(value, Mapping):
        overlap = banned_keys & set(value)
        if overlap:
            raise RuntimeError(f"E016-P1 public JSON 含私有字段: {sorted(overlap)}")
        for child in value.values():
            _scan(child)
    elif isinstance(value, list):
        for child in value:
            _scan(child)
    elif isinstance(value, str) and value.startswith(("/home/", "/mnt/", "C:\\", "D:\\")):
        raise RuntimeError("E016-P1 public JSON 含敏感绝对路径")


def _require_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise RuntimeError(f"E016-P1 {name} 缺少有效 SHA-256")


def verify(root: Path) -> None:
    summary = _read_json(root / "sanitized_summary.json")
    receipt = _read_json(root / "receipt.json")
    if summary.get("version") != VERSION or receipt.get("version") != VERSION:
        raise RuntimeError("E016-P1 public version 漂移")
    if summary.get("status") != "complete" or receipt.get("status") != "complete":
        raise RuntimeError("E016-P1 public status 漂移")
    _scan(summary)
    _scan(receipt)
    formal = summary["formal_training"]
    if formal["passed"] is not True or formal["formal_checkpoint_written"] is not True:
        raise RuntimeError("E016-P1 formal training/checkpoint 未通过")
    if formal["initialization"] != "random-from-scratch":
        raise RuntimeError("E016-P1 canonical checkpoint 不是随机初始化训练")
    if formal["test_split_read"] is not False or formal["fresh_held_out_read"] is not False:
        raise RuntimeError("E016-P1 formal training 读取了 test")
    if formal["motion_head_unchanged"] is not True:
        raise RuntimeError("E016-P1 Motion Head 发生漂移")
    if not all(formal["selected_validation_guardrails"].values()):
        raise RuntimeError("E016-P1 selected checkpoint 未通过全部 validation guardrails")
    frozen = summary["frozen_conditions"]
    if frozen["checkpoint_changed_after_training"] is not False:
        raise RuntimeError("E016-P1 checkpoint 在 test 前发生变化")
    if frozen["write_threshold_selected_on"] != "fresh-validation-only":
        raise RuntimeError("E016-P1 write threshold 不是 validation-only")
    if frozen["memory_age_selected_on"] != "fresh-validation-only":
        raise RuntimeError("E016-P1 memory age 不是 validation-only")
    if frozen["test_evaluated_once"] is not True:
        raise RuntimeError("E016-P1 test-once 状态漂移")
    if frozen["actuation_allowed"] is not False:
        raise RuntimeError("E016-P1 不允许 actuation")
    _require_sha256(frozen.get("test_once_claim_sha256"), "test-once claim")
    _require_sha256(frozen.get("calibration_receipt_sha256"), "calibration receipt")
    _require_sha256(summary.get("rules_sha256"), "rules")
    if summary["fresh_dataset"]["test_trajectory_count"] != 100:
        raise RuntimeError("E016-P1 fresh test 必须包含 100 trajectories")
    if summary["write_calibration"]["accepted_unsafe_count"] != 0:
        raise RuntimeError("E016-P1 validation calibration 必须零 unsafe write")
    if summary["memory_replay"]["episode_reset_leakage_count"] != 0:
        raise RuntimeError("E016-P1 Episode reset 泄漏")
    if summary["test_split_status"] != "consumed-once":
        raise RuntimeError("E016-P1 test split 状态未冻结")
    if summary["safe_for_actuator_promotion"] is not False:
        raise RuntimeError("E016-P1 public 不得授权 actuator promotion")
    for name, expected in receipt["files"].items():
        if _sha256(root / name) != expected:
            raise RuntimeError(f"E016-P1 public file SHA-256 漂移: {name}")
    readme = (root / "README.md").read_text(encoding="utf-8")
    if any(token in readme for token in ("/home/", "/mnt/", "C:\\", "D:\\")):
        raise RuntimeError("E016-P1 README 含敏感绝对路径")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) == 2 else Path(__file__).resolve().parent
    verify(root)
    print(json.dumps({"status": "passed", "version": VERSION}, sort_keys=True))


if __name__ == "__main__":
    main()
