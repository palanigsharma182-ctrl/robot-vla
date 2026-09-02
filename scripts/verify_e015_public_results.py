"""验证 E015 GitHub 脱敏结果的 schema、hash 与路径安全。"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

VERSION = "e015-precision-goal-memory-public/v1"
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
            raise RuntimeError(f"E015 public JSON 含私有字段: {sorted(overlap)}")
        for child in value.values():
            _scan(child)
    elif isinstance(value, list):
        for child in value:
            _scan(child)
    elif isinstance(value, str) and value.startswith(("/home/", "/mnt/", "C:\\", "D:\\")):
        raise RuntimeError("E015 public JSON 含敏感绝对路径")


def verify(root: Path) -> None:
    summary = _read_json(root / "sanitized_summary.json")
    receipt = _read_json(root / "receipt.json")
    if summary.get("version") != VERSION or receipt.get("version") != VERSION:
        raise RuntimeError("E015 public version 漂移")
    if summary.get("status") not in {"complete", "complete-supplemented"}:
        raise RuntimeError("E015 public summary 状态漂移")
    if receipt.get("status") != summary.get("status"):
        raise RuntimeError("E015 public summary/receipt 状态不一致")
    _scan(summary)
    _scan(receipt)
    if summary["frozen_conditions"]["training_performed"] is not False:
        raise RuntimeError("E015 不允许训练")
    if summary["frozen_conditions"]["actuation_allowed"] is not False:
        raise RuntimeError("E015 不允许 actuation")
    if summary["safe_for_actuator_promotion"] is not False:
        raise RuntimeError("E015 public 不得授权 actuator promotion")
    if summary["write_calibration"]["accepted_unsafe_count"] != 0:
        raise RuntimeError("E015 validation write calibration 必须零 unsafe")
    if summary["e015_b_memory_replay"]["episode_reset_leakage_count"] != 0:
        raise RuntimeError("E015 Episode reset 泄漏")
    if summary["test_split_status_after_e015"] != "consumed-for-evaluation":
        raise RuntimeError("E015 test split 状态未冻结")
    if summary["fresh_dataset"]["evaluation_trajectory_count"] != 100:
        raise RuntimeError("E015 evaluation 必须包含 100 个 fresh trajectories")
    claim_sha256 = summary["frozen_conditions"].get("test_once_claim_sha256")
    if not isinstance(claim_sha256, str) or SHA256_PATTERN.fullmatch(claim_sha256) is None:
        raise RuntimeError("E015 缺少有效 test-once claim SHA-256")
    if summary["frozen_conditions"].get("test_evaluated_once") is not True:
        raise RuntimeError("E015 test-once 状态漂移")
    if summary.get("status") == "complete-supplemented":
        if summary.get("supplement_policy") != (
            "deterministic-postprocessing-no-model-forward/v1"
        ):
            raise RuntimeError("E015 supplement policy 漂移")
        correction = summary.get("aggregation_correction", {})
        if correction.get("model_forward_repeated") is not False:
            raise RuntimeError("E015 supplement 不得重复 model forward")
        if correction.get("test_rules_changed") is not False:
            raise RuntimeError("E015 supplement 不得修改 frozen rules")
        memory = summary["e015_b_memory_replay"]
        expected_unavailable = int(memory["gt_unobservable_count"]) - int(
            memory["memory_valid_while_gt_unobservable_count"]
        )
        if int(memory["stale_or_uninitialized_occluded_count"]) != expected_unavailable:
            raise RuntimeError("E015 supplemented unavailable-memory 聚合错误")
        if int(memory["memory_unavailable_while_gt_unobservable_count"]) != (
            expected_unavailable
        ):
            raise RuntimeError("E015 supplemented unavailable-memory 字段不一致")
        availability = summary["memory_availability_audit"]
        if int(availability["initialized_episode_count"]) + int(
            availability["never_initialized_episode_count"]
        ) != int(memory["episode_count"]):
            raise RuntimeError("E015 initialized Episode 聚合错误")
        if int(summary["unsafe_write_audit"]["count"]) != int(
            summary["e015_b_write_measurements"]["accepted_unsafe_count"]
        ):
            raise RuntimeError("E015 unsafe-write supplement 聚合错误")
    for name, expected in receipt["files"].items():
        if _sha256(root / name) != expected:
            raise RuntimeError(f"E015 public file SHA-256 漂移: {name}")
    readme = (root / "README.md").read_text(encoding="utf-8")
    if any(token in readme for token in ("/home/", "/mnt/", "C:\\", "D:\\")):
        raise RuntimeError("E015 README 含敏感绝对路径")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) == 2 else Path(__file__).resolve().parent
    verify(root)
    print(json.dumps({"status": "passed", "version": VERSION}, sort_keys=True))


if __name__ == "__main__":
    main()
