"""验证 E017-P0 脱敏结果、失败边界与发布文件 identity。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERSION = "e017-p0-conservative-observability-public/v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    summary = json.loads((ROOT / "sanitized_summary.json").read_text(encoding="utf-8"))
    receipt = json.loads((ROOT / "receipt.json").read_text(encoding="utf-8"))
    if summary["version"] != VERSION or receipt["version"] != VERSION:
        raise RuntimeError("E017-P0 public version 漂移")
    if summary["status"] != "failed":
        raise RuntimeError("E017-P0 必须保留 failed canonical result")
    training = summary["training"]
    if training["epochs"] != 8 or training["total_optimizer_steps"] != 1200:
        raise RuntimeError("E017-P0 训练预算漂移")
    if training["formal_checkpoint_written"] is not False:
        raise RuntimeError("E017-P0 failed run 不得写 checkpoint")
    if training["test_split_read"] is not False:
        raise RuntimeError("E017-P0 禁止读取 test")
    if training["actuation_allowed"] is not False:
        raise RuntimeError("E017-P0 禁止 actuation")
    if any(row["eligible"] is not False for row in summary["validation_epochs"]):
        raise RuntimeError("E017-P0 不得把失败 epoch 标为 eligible")
    drift = summary["frozen_output_drift"]
    if drift["localization_metric_absolute_drift"] != 0.0:
        raise RuntimeError("E017-P0 localization 发生漂移")
    if drift["goal_mask_iou_absolute_drift"] != 0.0:
        raise RuntimeError("E017-P0 goal mask 发生漂移")
    diagnostic = summary["post_result_validation_diagnostic"]
    if diagnostic["canonical_result"] is not False or diagnostic["test_read"] is not False:
        raise RuntimeError("E017-P0 post-result diagnostic 边界漂移")
    if diagnostic["requires_new_held_out"] is not True:
        raise RuntimeError("E017-P0 必须要求新的 held-out")
    if summary["safe_for_actuator_promotion"] is not False:
        raise RuntimeError("E017-P0 不得授权 actuator promotion")
    for name, expected in receipt["files"].items():
        if _sha256(ROOT / name) != expected:
            raise RuntimeError(f"E017-P0 public file SHA-256 漂移: {name}")
    print(json.dumps({"status": "passed", "version": VERSION}, sort_keys=True))


if __name__ == "__main__":
    main()
