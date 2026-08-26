"""根据验证指标选择实际存在的周期 Checkpoint，避免错误 best 标记。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PeriodicCheckpointSelection:
    checkpoint: Path
    epoch: int
    optimizer_steps: int
    validation_loss: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checkpoint"] = str(self.checkpoint)
        return payload


def select_best_periodic_checkpoint(run_dir: str | Path) -> PeriodicCheckpointSelection:
    directory = Path(run_dir)
    metrics_path = directory / "metrics.jsonl"
    checkpoint_dir = directory / "checkpoints"
    if not metrics_path.is_file() or not checkpoint_dir.is_dir():
        raise FileNotFoundError("训练目录缺少 metrics.jsonl 或 checkpoints")

    candidates: list[PeriodicCheckpointSelection] = []
    cumulative_steps = 0
    expected_epoch = 1
    for line_number, line in enumerate(
        metrics_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"metrics.jsonl 第 {line_number} 行无效") from error
        if row.get("event") != "epoch":
            continue
        epoch = int(row["epoch"])
        if epoch != expected_epoch:
            raise ValueError(f"训练 epoch 不连续：期望 {expected_epoch}，实际 {epoch}")
        expected_epoch += 1
        optimizer_steps = int(row["train"]["optimizer_steps"])
        if optimizer_steps <= 0:
            raise ValueError("每个 epoch 的 optimizer_steps 必须为正数")
        cumulative_steps += optimizer_steps
        checkpoint = checkpoint_dir / f"step-{cumulative_steps:08d}.pt"
        if checkpoint.is_file():
            candidates.append(
                PeriodicCheckpointSelection(
                    checkpoint=checkpoint.resolve(),
                    epoch=epoch,
                    optimizer_steps=cumulative_steps,
                    validation_loss=float(row["validation"]["loss"]),
                )
            )
    if not candidates:
        raise ValueError("没有与训练指标对应的 periodic Checkpoint")
    return min(candidates, key=lambda item: (item.validation_loss, item.epoch))


__all__ = ["PeriodicCheckpointSelection", "select_best_periodic_checkpoint"]
