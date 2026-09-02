"""验证 E014 GitHub 脱敏结果的计数、schema、hash 与路径安全。"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

VERSION = "e014-precision-long-tail-diagnostic/v1"
TAXONOMY = (
    "label_or_channel_contract_failure",
    "temporal_alignment_failure",
    "semantic_swap_failure",
    "geometry_conditioning_failure",
    "multimodal_softargmax_failure",
    "visibility_or_ood_failure",
    "generic_correspondence_failure",
    "unclear_or_mixed",
)
FAILURE_FAMILY = (
    "correspondence_failure",
    "multimodal_softargmax_failure",
    "visibility_or_ood_failure",
    "geometry_conditioning_failure",
    "unclear_or_mixed",
)
FINGERPRINT = re.compile(r"[0-9a-f]{20}")


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
        "dataset_index",
        "timestep",
        "camera_position_base_m",
        "ray_direction_base",
        "gt_world_xy_m",
        "predicted_world_xy_m",
    }
    if isinstance(value, Mapping):
        overlap = banned_keys & set(value)
        if overlap:
            raise RuntimeError(f"public JSON 含私有字段: {sorted(overlap)}")
        for child in value.values():
            _scan(child)
    elif isinstance(value, list):
        for child in value:
            _scan(child)
    elif isinstance(value, str) and value.startswith(("/home/", "/mnt/", "C:\\", "D:\\")):
        raise RuntimeError("public JSON 含敏感绝对路径")


def verify(root: Path) -> None:
    summary = _read_json(root / "sanitized_summary.json")
    distributions = _read_json(root / "distributions.json")
    receipt = _read_json(root / "receipt.json")
    for value, name in (
        (summary, "sanitized_summary"),
        (distributions, "distributions"),
        (receipt, "receipt"),
    ):
        if value.get("version") != VERSION:
            raise RuntimeError(f"{name} version 漂移")
        _scan(value)
    aggregate = summary["aggregate"]
    if aggregate["prediction_row_count"] != 4088:
        raise RuntimeError("E014 public prediction row count 必须为 4,088")
    if aggregate["valid_keypoint_count"] != 3307:
        raise RuntimeError("E014 public valid keypoint count 必须为 3,307")
    for name, expected in (("top20_taxonomy", 20), ("top50_taxonomy", 50)):
        counts = summary[name]
        if set(counts) != set(TAXONOMY) or sum(int(value) for value in counts.values()) != expected:
            raise RuntimeError(f"{name} 未互斥覆盖 {expected} rows")
    for name, expected in (("top20_failure_family", 20), ("top50_failure_family", 50)):
        counts = summary[name]
        if set(counts) != set(FAILURE_FAMILY) or sum(int(value) for value in counts.values()) != expected:
            raise RuntimeError(f"{name} 未互斥覆盖 {expected} rows")
    maximum = summary["maximum_outlier"]
    if FINGERPRINT.fullmatch(str(maximum["sample_fingerprint"])) is None:
        raise RuntimeError("maximum sample fingerprint 无效")
    if summary["test_split_status_after_e014"] != "consumed-for-diagnostic-postmortem":
        raise RuntimeError("E013 test split 后续身份未冻结")
    for name, expected in receipt["files"].items():
        if _sha256(root / name) != expected:
            raise RuntimeError(f"public file SHA-256 漂移: {name}")
    readme = (root / "README.md").read_text(encoding="utf-8")
    if any(token in readme for token in ("/home/", "/mnt/", "C:\\", "D:\\")):
        raise RuntimeError("README 含敏感绝对路径")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) == 2 else Path(__file__).resolve().parent
    verify(root)
    print(json.dumps({"status": "passed", "version": VERSION}, sort_keys=True))


if __name__ == "__main__":
    main()
