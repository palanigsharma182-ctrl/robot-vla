"""在非正式 V2 smoke 数据上测 Layer-12 在线 object/goal 相对几何精度。"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from robot_vla.cli.evaluate_maniskill import _load_audit_identity, _sha256_file
from robot_vla.cli.probe_qwen_spatial import (
    _append_jsonl,
    _atomic_torch_save,
    _atomic_write_json,
    _clone_state,
    _move_to_device,
)
from robot_vla.cli.train_stage1 import compute_source_revision
from robot_vla.contracts import (
    PROMPT_VERSION_OBSERVATION_V2,
    QWEN_MODEL_ID,
    QWEN_REVISION,
    RobotSpec,
)
from robot_vla.data.sampler import TaskEpisodeBalancedSampler
from robot_vla.data.trajectory import TrajectoryMeta, load_manifest
from robot_vla.diagnostics.qwen_layer_reach import QWEN_LAYER12
from robot_vla.diagnostics.v2_online_geometry_probe import (
    CURRENT_EXTERNAL_IMAGE_INDEX,
    ONLINE_GEOMETRY_TARGETS,
    V2_IMAGES_PER_SAMPLE,
    V2_ONLINE_GEOMETRY_PROBE_FORMAT,
    OnlineVisualTargetProbe,
    QwenV2OnlineGeometryProbeCollator,
    QwenV2OnlineGeometryProbeDataset,
    build_selected_visual_token_layout,
    compact_selected_visual_tokens,
    online_geometry_probe_loss,
    summarize_online_geometry_predictions,
)
from robot_vla.model.factory import load_frozen_qwen_v01
from robot_vla.model.qwen_context import FrozenQwenLayerContextEncoder
from robot_vla.model.qwen_processor import (
    QwenProcessorConfig,
    QwenVLAProcessorAdapter,
)


@dataclass(frozen=True)
class CachedOnlineGeometrySplit:
    visual_tokens: torch.Tensor
    visual_token_centers: torch.Tensor
    target_uv: torch.Tensor
    image_sizes_hw: torch.Tensor
    grid_shapes_hw: torch.Tensor
    intrinsics: torch.Tensor
    world_from_cameras: torch.Tensor
    target_positions_world_m: torch.Tensor
    unique_windows: int
    unique_trajectories: int
    extraction_seconds: float

    def __len__(self) -> int:
        return int(self.target_uv.shape[0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--qwen-batch-size", type=int, default=1)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--selector-loss-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=913_013)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--expected-train-trajectories", type=int, default=20)
    parser.add_argument("--expected-validation-trajectories", type=int, default=5)
    parser.add_argument("--expected-test-trajectories", type=int, default=5)
    parser.add_argument("--smoke-seed-start", type=int, default=39_000)
    parser.add_argument("--smoke-seed-end", type=int, default=39_099)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "epochs",
        "qwen_batch_size",
        "probe_batch_size",
        "train_samples",
        "validation_samples",
        "test_samples",
        "expected_train_trajectories",
        "expected_validation_trajectories",
        "expected_test_trajectories",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} 必须为正整数")
    for name in ("learning_rate", "weight_decay", "selector_loss_weight"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} 必须是有限非负数")
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate 必须为正数")
    if args.seed < 0:
        raise ValueError("seed 不能为负数")
    if args.smoke_seed_start < 0 or args.smoke_seed_end < args.smoke_seed_start:
        raise ValueError("smoke seed 范围无效")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V2 online geometry probe 请求 CUDA，但当前不可用")
    if args.device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 CUDA 设备不支持冻结 Qwen 所需 BF16")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _entries_for_smoke_split(
    data: Path,
    split: str,
    *,
    expected_trajectories: int,
    seed_start: int,
    seed_end: int,
) -> list[TrajectoryMeta]:
    entries = load_manifest(data, split=split)
    if len(entries) != expected_trajectories:
        raise ValueError(
            f"smoke {split} 必须恰有 {expected_trajectories} 条轨迹，实际 {len(entries)}"
        )
    for entry in entries:
        try:
            seed = int(entry.randomization["seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"smoke {split} 轨迹缺少有限 seed") from error
        if not seed_start <= seed <= seed_end:
            raise ValueError(f"smoke seed {seed} 不在隔离范围 [{seed_start},{seed_end}]")
    return entries


def _dataset(
    data: Path,
    split: str,
    spec: RobotSpec,
    *,
    expected_trajectories: int,
    seed_start: int,
    seed_end: int,
) -> QwenV2OnlineGeometryProbeDataset:
    entries = _entries_for_smoke_split(
        data,
        split,
        expected_trajectories=expected_trajectories,
        seed_start=seed_start,
        seed_end=seed_end,
    )
    return QwenV2OnlineGeometryProbeDataset(data, entries, spec, cache_size=2)


def _loader(
    dataset: QwenV2OnlineGeometryProbeDataset,
    collator: QwenV2OnlineGeometryProbeCollator,
    *,
    samples: int,
    batch_size: int,
    seed: int,
    replacement: bool,
) -> DataLoader:
    if replacement:
        sampler: Any = TaskEpisodeBalancedSampler(
            dataset,
            num_samples=samples,
            seed=seed,
        )
    else:
        generator = np.random.default_rng(seed)
        by_episode: dict[int, list[int]] = {}
        for sample_index, (entry_index, _timestep) in enumerate(dataset.index):
            by_episode.setdefault(entry_index, []).append(sample_index)
        for indices in by_episode.values():
            generator.shuffle(indices)
        episode_order = sorted(by_episode)
        selected: list[int] = []
        offset = 0
        target_count = min(samples, len(dataset))
        while len(selected) < target_count:
            added = False
            for entry_index in episode_order:
                episode_indices = by_episode[entry_index]
                if offset < len(episode_indices):
                    selected.append(episode_indices[offset])
                    added = True
                    if len(selected) == target_count:
                        break
            if not added:
                break
            offset += 1
        if len(selected) != target_count or len(set(selected)) != target_count:
            raise RuntimeError("validation/test balanced unique sampler 构造失败")
        sampler = selected
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )


def _extract_frozen_features(
    loader: DataLoader,
    encoder: FrozenQwenLayerContextEncoder,
    *,
    processor: QwenVLAProcessorAdapter,
    device: torch.device,
) -> CachedOnlineGeometrySplit:
    encoder.train(False)
    visual_tokens: list[torch.Tensor] = []
    visual_token_centers: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    image_sizes: list[torch.Tensor] = []
    grid_shapes: list[torch.Tensor] = []
    intrinsics: list[torch.Tensor] = []
    transforms: list[torch.Tensor] = []
    target_positions: list[torch.Tensor] = []
    identities: set[str] = set()
    trajectory_ids: set[str] = set()
    image_token_id = processor.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if not isinstance(image_token_id, int) or image_token_id < 0:
        raise RuntimeError("Qwen tokenizer 没有有效 <|image_pad|> token id")
    started = time.perf_counter()
    for raw_batch in loader:
        qwen_inputs = _move_to_device(raw_batch["qwen_inputs"], device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=True,
        ):
            context = encoder(qwen_inputs)
        layout = build_selected_visual_token_layout(
            qwen_inputs["input_ids"],
            qwen_inputs["image_grid_thw"],
            image_token_id=image_token_id,
            merge_size=processor.config.merge_size,
        )
        if torch.any(layout.mask & ~context.mask):
            raise RuntimeError("当前 external visual span 被 Qwen context mask 意外屏蔽")
        compact_tokens, compact_centers = compact_selected_visual_tokens(
            context.tokens,
            layout,
        )
        visual_tokens.append(compact_tokens.detach().to(torch.bfloat16).cpu())
        visual_token_centers.append(compact_centers.detach().cpu())
        targets.append(raw_batch["target_uv_external"].float())
        image_sizes.append(raw_batch["image_size_external"].long())
        grid_shapes.append(layout.grid_shapes.detach().cpu())
        intrinsics.append(raw_batch["intrinsic_external"].float())
        transforms.append(raw_batch["world_from_external"].float())
        target_positions.append(raw_batch["target_position_world_m"].float())
        for trajectory_id, timestep in zip(
            raw_batch["trajectory_id"],
            raw_batch["timestep"].tolist(),
            strict=True,
        ):
            identity = f"{trajectory_id}:{int(timestep)}"
            identities.add(identity)
            trajectory_ids.add(str(trajectory_id))
    elapsed = time.perf_counter() - started
    if not targets:
        raise ValueError("V2 online geometry probe dataloader 不能为空")
    compact_counts = {int(value.shape[1]) for value in visual_tokens}
    if len(compact_counts) != 1:
        raise ValueError("feature cache 中 current external visual token 数不一致")
    return CachedOnlineGeometrySplit(
        visual_tokens=torch.cat(visual_tokens),
        visual_token_centers=torch.cat(visual_token_centers),
        target_uv=torch.cat(targets),
        image_sizes_hw=torch.cat(image_sizes),
        grid_shapes_hw=torch.cat(grid_shapes),
        intrinsics=torch.cat(intrinsics),
        world_from_cameras=torch.cat(transforms),
        target_positions_world_m=torch.cat(target_positions),
        unique_windows=len(identities),
        unique_trajectories=len(trajectory_ids),
        extraction_seconds=float(elapsed),
    )


def _run_probe(
    cached: CachedOnlineGeometrySplit,
    probe: OnlineVisualTargetProbe,
    *,
    batch_size: int,
    device: torch.device,
    selector_loss_weight: float,
    optimizer: torch.optim.Optimizer | None,
    seed: int,
    include_position_metrics: bool,
) -> dict[str, Any]:
    training = optimizer is not None
    probe.train(training)
    sample_count = len(cached)
    generator = torch.Generator().manual_seed(seed)
    order = (
        torch.randperm(sample_count, generator=generator)
        if training
        else torch.arange(sample_count)
    )
    loss_numerator = 0.0
    coordinate_numerator = 0.0
    selector_numerator = 0.0
    selector_correct = 0
    predicted: list[torch.Tensor] = []
    selected: list[torch.Tensor] = []
    nearest: list[torch.Tensor] = []
    for start in range(0, sample_count, batch_size):
        indices = order[start : start + batch_size]
        tokens = cached.visual_tokens[indices].to(device)
        centers = cached.visual_token_centers[indices].to(device)
        target_uv = cached.target_uv[indices].to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        output = probe(tokens)
        loss = online_geometry_probe_loss(
            output,
            target_uv,
            centers,
            selector_loss_weight=selector_loss_weight,
        )
        if optimizer is not None:
            loss.loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=10.0)
            optimizer.step()
        current_batch_size = int(target_uv.shape[0])
        target_elements = current_batch_size * len(ONLINE_GEOMETRY_TARGETS)
        loss_numerator += float(loss.loss.detach().item()) * current_batch_size
        coordinate_numerator += float(loss.coordinate_loss.detach().item()) * current_batch_size
        selector_numerator += float(loss.selector_loss.detach().item()) * target_elements
        selector_correct += int(
            (output.selected_token_indices == loss.target_token_indices).sum().item()
        )
        if include_position_metrics:
            predicted.append(output.predicted_uv.detach().cpu())
            selected.append(output.selected_token_indices.detach().cpu())
            nearest.append(loss.target_token_indices.detach().cpu())
    result: dict[str, Any] = {
        "samples": sample_count,
        "unique_windows": cached.unique_windows,
        "unique_trajectories": cached.unique_trajectories,
        "loss": loss_numerator / sample_count,
        "coordinate_mse": coordinate_numerator / sample_count,
        "selector_cross_entropy": selector_numerator
        / (sample_count * len(ONLINE_GEOMETRY_TARGETS)),
        "selector_exact_nearest_token_accuracy": selector_correct
        / (sample_count * len(ONLINE_GEOMETRY_TARGETS)),
    }
    if include_position_metrics:
        result["position"] = summarize_online_geometry_predictions(
            torch.cat(predicted).numpy(),
            cached.target_uv.numpy(),
            torch.cat(selected).numpy(),
            torch.cat(nearest).numpy(),
            cached.image_sizes_hw.numpy(),
            cached.grid_shapes_hw.numpy(),
            cached.intrinsics.numpy(),
            cached.world_from_cameras.numpy(),
            cached.target_positions_world_m.numpy(),
        )
    return result


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    if args.output.exists():
        if not args.output.is_dir():
            raise FileExistsError("V2 online geometry probe 输出路径不是目录")
        if any(args.output.iterdir()):
            raise FileExistsError("V2 online geometry probe 输出目录非空，拒绝覆盖")
    args.output.mkdir(parents=True, exist_ok=True)
    _seed_all(args.seed)
    device = torch.device(args.device)
    spec = RobotSpec()
    dataset_identity = _load_audit_identity(args.data)
    processor = QwenVLAProcessorAdapter.from_pretrained(
        cache_dir=str(args.model_cache),
        local_files_only=True,
        config=QwenProcessorConfig(
            prompt_version=PROMPT_VERSION_OBSERVATION_V2,
        ),
    )
    collator = QwenV2OnlineGeometryProbeCollator(processor)
    expected_counts = {
        "train": args.expected_train_trajectories,
        "val": args.expected_validation_trajectories,
        "test": args.expected_test_trajectories,
    }
    datasets = {
        split: _dataset(
            args.data,
            split,
            spec,
            expected_trajectories=expected,
            seed_start=args.smoke_seed_start,
            seed_end=args.smoke_seed_end,
        )
        for split, expected in expected_counts.items()
    }
    loaders = {
        "train": _loader(
            datasets["train"],
            collator,
            samples=args.train_samples,
            batch_size=args.qwen_batch_size,
            seed=args.seed,
            replacement=True,
        ),
        "validation": _loader(
            datasets["val"],
            collator,
            samples=args.validation_samples,
            batch_size=args.qwen_batch_size,
            seed=args.seed + 1,
            replacement=False,
        ),
        "test": _loader(
            datasets["test"],
            collator,
            samples=args.test_samples,
            batch_size=args.qwen_batch_size,
            seed=args.seed + 2,
            replacement=False,
        ),
    }

    qwen = load_frozen_qwen_v01(
        cache_dir=str(args.model_cache),
        local_files_only=True,
        device=device,
    )
    encoder = FrozenQwenLayerContextEncoder(qwen, QWEN_LAYER12).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        caches = {
            split: _extract_frozen_features(
                loader,
                encoder,
                processor=processor,
                device=device,
            )
            for split, loader in loaders.items()
        }
    extraction_memory = None
    if device.type == "cuda":
        extraction_memory = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    del encoder, qwen
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    probe = OnlineVisualTargetProbe().to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    project_root = Path(__file__).resolve().parents[3]
    experiment = {
        "format": V2_ONLINE_GEOMETRY_PROBE_FORMAT,
        "status": "non-formal-engineering-smoke",
        "must_not_enter_e013_training_or_effect_statistics": True,
        "code_revision": compute_source_revision(project_root),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "dataset": dataset_identity,
        "qwen": {
            "model_id": QWEN_MODEL_ID,
            "revision": QWEN_REVISION,
            "layer": QWEN_LAYER12,
            "frozen": True,
            "compute_dtype": "bfloat16",
        },
        "processor_config": asdict(processor.config),
        "probe": {
            "targets": list(ONLINE_GEOMETRY_TARGETS),
            "input": (
                "all visual tokens from current external image contextualized "
                "by eight-image V2 prompt"
            ),
            "images_per_sample": V2_IMAGES_PER_SAMPLE,
            "selected_image_index": CURRENT_EXTERNAL_IMAGE_INDEX,
            "architecture": "per-target learned token selector plus weighted-token UV decoder",
            "training_supervision": "GT object/goal UV and nearest-token labels",
            "test_target_selection": "online; no GT token index or GT UV is passed to the selector",
            "selector_loss_weight": args.selector_loss_weight,
            "world_metric": "predicted pixel ray intersected with known Reach table-height plane",
            "future_information": False,
        },
        "windows": {
            split: {
                "available": len(dataset),
                "sampled": len(caches[cache_name]),
                "unique_sampled_windows": caches[cache_name].unique_windows,
                "unique_sampled_trajectories": caches[cache_name].unique_trajectories,
                "sha256": dataset.window_sha256,
                "rejected_missing_v2": dataset.rejected_missing_v2,
                "rejected_missing_geometry": dataset.rejected_missing_geometry,
                "rejected_incomplete_history": dataset.rejected_incomplete_history,
                "rejected_out_of_view": dataset.rejected_out_of_view,
            }
            for split, cache_name, dataset in (
                ("train", "train", datasets["train"]),
                ("validation", "validation", datasets["val"]),
                ("test", "test", datasets["test"]),
            )
        },
        "feature_extraction": {
            split: {
                "seconds": cache.extraction_seconds,
                "samples_per_second": len(cache) / cache.extraction_seconds,
            }
            for split, cache in caches.items()
        },
        "feature_extraction_memory": extraction_memory,
        "trainable_parameters": sum(parameter.numel() for parameter in probe.parameters()),
        "device": str(device),
        "gpu_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
    }
    experiment_path = args.output / "experiment.json"
    _atomic_write_json(experiment_path, experiment)
    metrics_path = args.output / "metrics.jsonl"
    best: dict[str, Any] = {
        "loss": float("inf"),
        "epoch": None,
        "state": None,
        "validation": None,
    }
    for epoch in range(args.epochs):
        train_metrics = _run_probe(
            caches["train"],
            probe,
            batch_size=args.probe_batch_size,
            device=device,
            selector_loss_weight=args.selector_loss_weight,
            optimizer=optimizer,
            seed=args.seed + epoch,
            include_position_metrics=False,
        )
        with torch.no_grad():
            validation_metrics = _run_probe(
                caches["validation"],
                probe,
                batch_size=args.probe_batch_size,
                device=device,
                selector_loss_weight=args.selector_loss_weight,
                optimizer=None,
                seed=args.seed + 1,
                include_position_metrics=True,
            )
        validation_loss = float(validation_metrics["loss"])
        if validation_loss < best["loss"]:
            best = {
                "loss": validation_loss,
                "epoch": epoch + 1,
                "state": _clone_state(probe),
                "validation": validation_metrics,
            }
        payload = {
            "event": "epoch",
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        _append_jsonl(metrics_path, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
    if best["state"] is None:
        raise RuntimeError("V2 online geometry probe 没有产生 best state")
    probe.load_state_dict(best["state"])
    with torch.no_grad():
        test_metrics = _run_probe(
            caches["test"],
            probe,
            batch_size=args.probe_batch_size,
            device=device,
            selector_loss_weight=args.selector_loss_weight,
            optimizer=None,
            seed=args.seed + 2,
            include_position_metrics=True,
        )
    checkpoint_path = args.output / "probe.pt"
    _atomic_torch_save(
        checkpoint_path,
        {
            "format": V2_ONLINE_GEOMETRY_PROBE_FORMAT,
            "experiment_sha256": _sha256_file(experiment_path),
            "best_epoch": best["epoch"],
            "validation_loss": best["loss"],
            "state_dict": best["state"],
        },
    )
    summary = {
        "format": V2_ONLINE_GEOMETRY_PROBE_FORMAT,
        "complete": True,
        "screening_only": True,
        "best_epoch": best["epoch"],
        "validation": best["validation"],
        "test": test_metrics,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
        },
        "interpretation_boundary": (
            "通过只表示 Layer-12 在线目标选择值得进入 E013；"
            "不等价于 Action 策略、抓取、放置或真实部署通过"
        ),
    }
    _atomic_write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
