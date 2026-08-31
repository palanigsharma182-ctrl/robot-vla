"""用训练证据和正式 paired validation 选择 E012 的 10/20/30 epoch checkpoint。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from robot_vla.cli.analyze_e012_paired_evaluation import (
    _atomic_write_json,
    _load_atomic,
    _load_full_chain,
    _read_json,
    _sha256,
    _strict_json_loads,
    _validate_paired_directories,
)
from robot_vla.evaluation.atomic import AtomicSkillEpisodeResult
from robot_vla.evaluation.e012_checkpoint_selection import (
    E012_CHECKPOINT_EPOCHS,
    E012CheckpointCandidate,
    select_e012_checkpoint,
)
from robot_vla.evaluation.rollout import RolloutEpisodeResult


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline-full-chain", type=Path, required=True)
    parser.add_argument("--baseline-atomic", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        nargs=4,
        action="append",
        required=True,
        metavar=("LABEL", "EPOCH", "FULL_CHAIN_DIR", "ATOMIC_DIR"),
        help="精确传入 epoch 10/20/30 三组评估目录",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_epoch_metrics(run: Path) -> dict[int, dict[str, Any]]:
    path = run / "metrics.jsonl"
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("E012 训练目录缺少非 symlink metrics.jsonl")
    rows: dict[int, dict[str, Any]] = {}
    expected_epoch = 1
    cumulative_optimizer_steps = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"metrics.jsonl 第 {line_number} 行为空")
            value = _strict_json_loads(
                line,
                label=f"metrics.jsonl 第 {line_number} 行",
            )
            if not isinstance(value, dict):
                raise TypeError(f"metrics.jsonl 第 {line_number} 行顶层必须是 object")
            if value.get("event") != "epoch":
                continue
            epoch = value.get("epoch")
            if not isinstance(epoch, int) or isinstance(epoch, bool):
                raise TypeError("metrics epoch 必须是 int")
            if epoch != expected_epoch:
                raise ValueError(
                    f"metrics epoch 不连续：期望 {expected_epoch}，实际 {epoch}"
                )
            expected_epoch += 1
            optimizer_steps = value.get("train", {}).get("optimizer_steps")
            if (
                not isinstance(optimizer_steps, int)
                or isinstance(optimizer_steps, bool)
                or optimizer_steps <= 0
            ):
                raise ValueError(f"epoch {epoch} optimizer_steps 无效")
            cumulative_optimizer_steps += optimizer_steps
            validation_loss = value.get("validation", {}).get("loss")
            if (
                isinstance(validation_loss, bool)
                or not isinstance(validation_loss, (int, float))
                or not math.isfinite(float(validation_loss))
                or float(validation_loss) < 0.0
            ):
                raise ValueError(f"epoch {epoch} validation total loss 无效")
            rows[epoch] = {
                "validation_total_loss": float(validation_loss),
                "optimizer_steps": cumulative_optimizer_steps,
            }
    if sorted(rows) != list(range(1, 31)):
        raise ValueError("E012 正式训练必须完整包含连续 30 epochs")
    return rows


def _checkpoint_receipt(run: Path, epoch_row: dict[str, Any]) -> dict[str, Any]:
    optimizer_steps = int(epoch_row["optimizer_steps"])
    path = run / "checkpoints" / f"step-{optimizer_steps:08d}.pt"
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"缺少 epoch 对应的非 symlink periodic checkpoint: {path}")
    return {
        "file_name": path.name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "optimizer_steps": optimizer_steps,
    }


def _typed_full_results(value: list[Any]) -> tuple[RolloutEpisodeResult, ...]:
    if not value or not all(isinstance(row, RolloutEpisodeResult) for row in value):
        raise TypeError("checkpoint selection full-chain 结果类型无效")
    return tuple(value)


def _typed_atomic_results(value: list[Any]) -> tuple[AtomicSkillEpisodeResult, ...]:
    if not value or not all(isinstance(row, AtomicSkillEpisodeResult) for row in value):
        raise TypeError("checkpoint selection atomic 结果类型无效")
    return tuple(value)


def _evaluation_pair(
    full_path: Path,
    atomic_path: Path,
    *,
    expected_checkpoint_sha256: str,
    baseline_full: Any,
    baseline_atomic: Any,
    label: str,
) -> tuple[
    tuple[RolloutEpisodeResult, ...],
    tuple[AtomicSkillEpisodeResult, ...],
    dict[str, Any],
]:
    full = _load_full_chain(full_path)
    atomic = _load_atomic(atomic_path)
    if full.experiment["checkpoint"]["sha256"] != expected_checkpoint_sha256:
        raise ValueError(f"{label} full-chain checkpoint 与训练 periodic 权重不一致")
    if atomic.experiment["checkpoint"]["sha256"] != expected_checkpoint_sha256:
        raise ValueError(f"{label} atomic checkpoint 与训练 periodic 权重不一致")
    if full.experiment["dataset"] != atomic.experiment["dataset"]:
        raise ValueError(f"{label} full-chain/atomic dataset identity 不一致")
    if (
        full.experiment["evaluation_code_revision"]
        != atomic.experiment["evaluation_code_revision"]
    ):
        raise ValueError(f"{label} full-chain/atomic evaluation code revision 不一致")
    _validate_paired_directories(baseline_full, full, label=f"{label} full-chain")
    _validate_paired_directories(baseline_atomic, atomic, label=f"{label} atomic")
    return (
        _typed_full_results(full.results),
        _typed_atomic_results(atomic.results),
        {"full_chain": full.receipt, "atomic": atomic.receipt},
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run.resolve(strict=True)
    if not run_root.is_dir():
        raise ValueError("--run 不是训练目录")
    training_experiment = _read_json(
        run_root / "experiment.json",
        label="training experiment.json",
    )
    initialization = training_experiment.get("initialization")
    if not isinstance(initialization, dict) or initialization.get("mode") != "init_checkpoint":
        raise ValueError("E012 checkpoint selection 要求独立 init_checkpoint warm start")
    initial_checkpoint = initialization.get("checkpoint")
    if not isinstance(initial_checkpoint, dict):
        raise TypeError("training experiment 缺少 initialization checkpoint")
    initial_sha256 = initial_checkpoint.get("sha256")
    if not isinstance(initial_sha256, str) or len(initial_sha256) != 64:
        raise ValueError("training initialization checkpoint SHA256 无效")

    baseline_full = _load_full_chain(args.baseline_full_chain)
    baseline_atomic = _load_atomic(args.baseline_atomic)
    if baseline_full.experiment["checkpoint"]["sha256"] != initial_sha256:
        raise ValueError("pi_0 full-chain baseline 不是训练 initialization checkpoint")
    if baseline_atomic.experiment["checkpoint"]["sha256"] != initial_sha256:
        raise ValueError("pi_0 atomic baseline 不是训练 initialization checkpoint")
    if baseline_full.experiment["dataset"] != baseline_atomic.experiment["dataset"]:
        raise ValueError("pi_0 full-chain/atomic dataset identity 不一致")
    if (
        baseline_full.experiment["evaluation_code_revision"]
        != baseline_atomic.experiment["evaluation_code_revision"]
    ):
        raise ValueError("pi_0 full-chain/atomic evaluation code revision 不一致")

    metrics = _read_epoch_metrics(run_root)
    candidate_specs = args.candidate
    if not isinstance(candidate_specs, list) or len(candidate_specs) != 3:
        raise ValueError("--candidate 必须精确传入三次")
    candidates: list[E012CheckpointCandidate] = []
    candidate_receipts: list[dict[str, Any]] = []
    for raw in candidate_specs:
        label, raw_epoch, full_path, atomic_path = raw
        try:
            epoch = int(raw_epoch)
        except ValueError as error:
            raise ValueError(f"candidate {label} epoch 不是整数") from error
        if epoch not in E012_CHECKPOINT_EPOCHS:
            raise ValueError(f"candidate {label} epoch 不属于 10/20/30")
        checkpoint = _checkpoint_receipt(run_root, metrics[epoch])
        full_results, atomic_results, evaluation_receipt = _evaluation_pair(
            Path(full_path),
            Path(atomic_path),
            expected_checkpoint_sha256=checkpoint["sha256"],
            baseline_full=baseline_full,
            baseline_atomic=baseline_atomic,
            label=label,
        )
        candidates.append(
            E012CheckpointCandidate(
                label=label,
                epoch=epoch,
                validation_total_loss=metrics[epoch]["validation_total_loss"],
                full_chain=full_results,
                atomic=atomic_results,
            )
        )
        candidate_receipts.append(
            {
                "label": label,
                "epoch": epoch,
                "checkpoint": checkpoint,
                "evaluation": evaluation_receipt,
            }
        )

    selection = select_e012_checkpoint(
        baseline_full_chain=_typed_full_results(baseline_full.results),
        baseline_atomic=_typed_atomic_results(baseline_atomic.results),
        candidates=tuple(candidates),
    )
    selection["input_receipts"] = {
        "training": {
            "directory_name": run_root.name,
            "experiment_sha256": _sha256(run_root / "experiment.json"),
            "metrics_sha256": _sha256(run_root / "metrics.jsonl"),
            "initial_checkpoint_sha256": initial_sha256,
        },
        "baseline": {
            "full_chain": baseline_full.receipt,
            "atomic": baseline_atomic.receipt,
        },
        "candidates": sorted(candidate_receipts, key=lambda row: row["epoch"]),
    }
    _atomic_write_json(args.output, selection)
    return selection


def main() -> None:
    selection = run(_parse_args())
    print(json.dumps(selection, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
