"""独立核验 E012 replay/DAgger paired training identity、步数和实际 exposure。"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from robot_vla.cli.build_e012_d1 import _read_json, _read_jsonl, _sha256_file

VERIFICATION_FORMAT = "robot-vla-e012-paired-training-verification/v1"
REPLAY_SOURCE_WEIGHTS = (("base_d0", 1.0),)
DAGGER_SOURCE_WEIGHTS = (
    ("base_d0", 0.8),
    ("dagger_reach_grasp", 0.1),
    ("dagger_grasp_lift", 0.1),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--dagger", type=Path, required=True)
    parser.add_argument("--expected-epochs", type=int, required=True)
    parser.add_argument("--expected-samples-per-epoch", type=int, required=True)
    parser.add_argument("--expected-optimizer-steps", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _source_weights(config: dict[str, Any]) -> tuple[tuple[str, float], ...]:
    return tuple(
        (str(source), float(weight))
        for source, weight in config.get("source_sampling_weights", [])
    )


def _paired_config_without_source(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result.pop("source_sampling_weights", None)
    return result


def _expected_source_counts(
    source_weights: tuple[tuple[str, float], ...],
    *,
    samples: int,
    epoch: int,
) -> dict[str, int]:
    total_weight = sum(weight for _, weight in source_weights)
    exact = [samples * weight / total_weight for _, weight in source_weights]
    counts = [math.floor(value) for value in exact]
    remainder = samples - sum(counts)
    ranked = sorted(
        range(len(exact)),
        key=lambda index: (
            -(exact[index] - counts[index]),
            (index + epoch) % len(exact),
        ),
    )
    for index in ranked[:remainder]:
        counts[index] += 1
    return {
        source: count
        for (source, _), count in zip(source_weights, counts, strict=True)
    }


def _verify_run(
    root: Path,
    *,
    label: str,
    source_weights: tuple[tuple[str, float], ...],
    expected_epochs: int,
    expected_samples_per_epoch: int,
    expected_optimizer_steps: int,
) -> dict[str, Any]:
    experiment_path = root / "experiment.json"
    metrics_path = root / "metrics.jsonl"
    exposure_path = root / "sampler_exposure.jsonl"
    experiment = _read_json(experiment_path, root=root)
    metrics = _read_jsonl(metrics_path, root=root)
    exposures = _read_jsonl(exposure_path, root=root)
    config = experiment.get("training_config")
    if not isinstance(config, dict):
        raise TypeError(f"{label}: training_config 缺失")
    if _source_weights(config) != source_weights:
        raise ValueError(f"{label}: source_sampling_weights 漂移")
    if int(config.get("samples_per_epoch", -1)) != expected_samples_per_epoch:
        raise ValueError(f"{label}: samples_per_epoch 漂移")
    initialization = experiment.get("initialization", {})
    checkpoint = initialization.get("checkpoint", {})
    if (
        initialization.get("mode") != "init_checkpoint"
        or initialization.get("restored_state") != "adapter_expert_weights_only"
        or initialization.get("trainer_state_reset") is not True
        or initialization.get("rng_restored") is not False
        or len(str(checkpoint.get("sha256", ""))) != 64
        or len(str(checkpoint.get("policy_state_sha256", ""))) != 64
    ):
        raise ValueError(f"{label}: init-checkpoint receipt 不完整")
    epoch_metrics = [row for row in metrics if row.get("event") == "epoch"]
    if len(epoch_metrics) != expected_epochs or len(exposures) != expected_epochs:
        raise ValueError(f"{label}: epoch/ledger 数量不完整")
    observed_optimizer_steps = 0
    aggregate_sources: Counter[str] = Counter()
    for epoch_index, (metric, exposure) in enumerate(
        zip(epoch_metrics, exposures, strict=True)
    ):
        epoch = epoch_index + 1
        if int(metric.get("epoch", -1)) != epoch or int(exposure.get("epoch", -1)) != epoch:
            raise ValueError(f"{label}: epoch 序列不连续")
        if metric.get("source_exposure") != exposure:
            raise ValueError(f"{label}: metrics 与 exposure ledger 漂移")
        if exposure.get("format") != "robot-vla-stage1-sampler-exposure/v1":
            raise ValueError(f"{label}: exposure format 不兼容")
        if int(exposure.get("samples", -1)) != expected_samples_per_epoch:
            raise ValueError(f"{label}: epoch {epoch} exposure 总数漂移")
        configured = tuple(
            (str(source), float(weight))
            for source, weight in exposure.get("configured_source_weights", [])
        )
        if configured != source_weights:
            raise ValueError(f"{label}: epoch {epoch} configured source 漂移")
        actual_sources: Counter[str] = Counter()
        rows = exposure.get("source_skill_boundary_offset")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{label}: epoch {epoch} exposure rows 缺失")
        for row in rows:
            source = str(row["source"])
            samples = int(row["samples"])
            boundary_offset = row.get("boundary_offset")
            if samples <= 0:
                raise ValueError(f"{label}: exposure count 必须为正数")
            if source == "base_d0":
                if boundary_offset is not None:
                    raise ValueError(f"{label}: base_d0 禁止 boundary_offset")
            elif source in {"dagger_reach_grasp", "dagger_grasp_lift"}:
                if not isinstance(boundary_offset, int) or not 0 <= boundary_offset <= 48:
                    raise ValueError(f"{label}: DAgger boundary_offset 超出 [0,48]")
            else:
                raise ValueError(f"{label}: exposure 包含未知 source: {source}")
            actual_sources[source] += samples
        expected_sources = _expected_source_counts(
            source_weights,
            samples=expected_samples_per_epoch,
            epoch=epoch_index,
        )
        if dict(actual_sources) != expected_sources:
            raise ValueError(
                f"{label}: epoch {epoch} source quota 漂移: "
                f"{dict(actual_sources)} != {expected_sources}"
            )
        aggregate_sources.update(actual_sources)
        train = metric.get("train", {})
        if int(train.get("examples", -1)) != expected_samples_per_epoch:
            raise ValueError(f"{label}: Trainer examples 与 exposure 不一致")
        observed_optimizer_steps += int(train.get("optimizer_steps", -1))
    if observed_optimizer_steps != expected_optimizer_steps:
        raise ValueError(
            f"{label}: optimizer steps {observed_optimizer_steps} "
            f"!= {expected_optimizer_steps}"
        )
    return {
        "experiment": experiment,
        "experiment_sha256": _sha256_file(experiment_path, root=root),
        "metrics_sha256": _sha256_file(metrics_path, root=root),
        "sampler_exposure_sha256": _sha256_file(exposure_path, root=root),
        "epochs": expected_epochs,
        "samples": expected_epochs * expected_samples_per_epoch,
        "optimizer_steps": observed_optimizer_steps,
        "aggregate_source_exposure": dict(sorted(aggregate_sources.items())),
    }


def verify_pair(
    replay_root: Path,
    dagger_root: Path,
    *,
    expected_epochs: int,
    expected_samples_per_epoch: int,
    expected_optimizer_steps: int,
) -> dict[str, Any]:
    replay = _verify_run(
        replay_root,
        label="pi_replay",
        source_weights=REPLAY_SOURCE_WEIGHTS,
        expected_epochs=expected_epochs,
        expected_samples_per_epoch=expected_samples_per_epoch,
        expected_optimizer_steps=expected_optimizer_steps,
    )
    dagger = _verify_run(
        dagger_root,
        label="pi_dagger",
        source_weights=DAGGER_SOURCE_WEIGHTS,
        expected_epochs=expected_epochs,
        expected_samples_per_epoch=expected_samples_per_epoch,
        expected_optimizer_steps=expected_optimizer_steps,
    )
    replay_experiment = replay.pop("experiment")
    dagger_experiment = dagger.pop("experiment")
    replay_initialization = replay_experiment["initialization"]
    dagger_initialization = dagger_experiment["initialization"]
    if replay_initialization != dagger_initialization:
        raise ValueError("paired runs 的 init checkpoint/tensor state receipt 不一致")
    if replay_experiment.get("code_revision") != dagger_experiment.get("code_revision"):
        raise ValueError("paired runs 的 code revision 不一致")
    if replay_experiment.get("proprio_stats") != dagger_experiment.get("proprio_stats"):
        raise ValueError("paired runs 的 frozen D0 ProprioStats identity 不一致")
    replay_config = replay_experiment["training_config"]
    dagger_config = dagger_experiment["training_config"]
    if _paired_config_without_source(replay_config) != _paired_config_without_source(
        dagger_config
    ):
        raise ValueError("paired runs 除 source quota 外的 training config 不一致")
    replay_dataset = replay_experiment.get("dataset", {})
    dagger_dataset = dagger_experiment.get("dataset", {})
    if replay_dataset.get("base_d0") != dagger_dataset.get("base_d0"):
        raise ValueError("paired runs 的 D0 identity 不一致")
    if replay_dataset.get("dagger_additions") is not None:
        raise ValueError("pi_replay 禁止使用 DAgger additions")
    if not isinstance(dagger_dataset.get("dagger_additions"), dict):
        raise TypeError("pi_dagger 缺少 DAgger additions identity")
    return {
        "format": VERIFICATION_FORMAT,
        "passed": True,
        "paired_identity": {
            "code_revision": replay_experiment["code_revision"],
            "initialization": replay_initialization,
            "proprio_stats": replay_experiment["proprio_stats"],
            "training_config_without_source": _paired_config_without_source(
                replay_config
            ),
        },
        "pi_replay": replay,
        "pi_dagger": dagger,
    }


def run(args: argparse.Namespace) -> None:
    if (
        args.expected_epochs <= 0
        or args.expected_samples_per_epoch <= 0
        or args.expected_optimizer_steps <= 0
    ):
        raise ValueError("expected epochs/samples/steps 必须为正数")
    result = verify_pair(
        args.replay,
        args.dagger,
        expected_epochs=args.expected_epochs,
        expected_samples_per_epoch=args.expected_samples_per_epoch,
        expected_optimizer_steps=args.expected_optimizer_steps,
    )
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError("paired verification output 已存在")
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
