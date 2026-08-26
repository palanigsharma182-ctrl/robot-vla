from __future__ import annotations

import json

import pytest

from robot_vla.evaluation.checkpoint_selection import select_best_periodic_checkpoint


def _metric(epoch: int, loss: float) -> dict[str, object]:
    return {
        "event": "epoch",
        "epoch": epoch,
        "train": {"optimizer_steps": 64},
        "validation": {"loss": loss},
    }


def test_selects_lowest_validation_loss_that_has_periodic_weights(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "step-00000128.pt").touch()
    (checkpoint_dir / "step-00000256.pt").touch()
    rows = [
        _metric(1, 0.5),
        _metric(2, 0.4),
        _metric(3, 0.1),
        _metric(4, 0.3),
    ]
    (tmp_path / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    selected = select_best_periodic_checkpoint(tmp_path)

    assert selected.epoch == 4
    assert selected.optimizer_steps == 256
    assert selected.validation_loss == pytest.approx(0.3)
    assert selected.checkpoint == (checkpoint_dir / "step-00000256.pt").resolve()


def test_rejects_non_contiguous_metric_history(tmp_path) -> None:
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps(_metric(2, 0.4)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="不连续"):
        select_best_periodic_checkpoint(tmp_path)
