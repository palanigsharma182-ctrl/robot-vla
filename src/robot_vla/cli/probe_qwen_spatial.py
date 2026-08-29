"""冻结 Qwen，一次前向比较 Layer 12/24 方块 token 的连续位置可解码性。"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from robot_vla.cli.evaluate_maniskill import _load_audit_identity, _sha256_file
from robot_vla.cli.train_stage1 import compute_source_revision
from robot_vla.contracts import QWEN_MODEL_ID, QWEN_REVISION, RobotSpec
from robot_vla.data.sampler import TaskEpisodeBalancedSampler
from robot_vla.data.trajectory import load_manifest
from robot_vla.diagnostics.qwen_layer_reach import (
    QWEN_LAYER12,
    QWEN_LAYER24,
    FrozenQwenLayerPairContextEncoder,
)
from robot_vla.diagnostics.qwen_spatial_probe import (
    QWEN_SPATIAL_PROBE_FORMAT,
    QwenExternalSpatialProbeDataset,
    QwenSpatialProbeCollator,
    build_external_visual_token_layout,
    build_matched_linear_probes,
    interpret_layer12_probe,
    nearest_external_visual_token_indices,
    spatial_probe_loss,
    summarize_spatial_predictions,
)
from robot_vla.model.factory import load_frozen_qwen_v01
from robot_vla.model.qwen_processor import QwenVLAProcessorAdapter

LAYER_NAMES = ("layer12", "layer24")


@dataclass(frozen=True)
class CachedSpatialProbeSplit:
    layer12_features: torch.Tensor
    layer24_features: torch.Tensor
    target_uv: torch.Tensor
    image_sizes_hw: torch.Tensor
    grid_shapes_hw: torch.Tensor
    intrinsics: torch.Tensor
    world_from_cameras: torch.Tensor
    object_positions_m: torch.Tensor
    nearest_token_uv: torch.Tensor
    unique_windows: int

    def __len__(self) -> int:
        return int(self.target_uv.shape[0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--validation-samples", type=int, default=1024)
    parser.add_argument("--test-samples", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "epochs",
        "batch_size",
        "train_samples",
        "validation_samples",
        "test_samples",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} 必须为正整数")
    if args.learning_rate <= 0 or not np.isfinite(args.learning_rate):
        raise ValueError("learning_rate 必须是有限正数")
    if args.weight_decay < 0 or not np.isfinite(args.weight_decay):
        raise ValueError("weight_decay 必须是有限非负数")
    if args.seed < 0:
        raise ValueError("seed 不能为负数")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Spatial Probe 请求 CUDA，但当前 PyTorch 无可用 CUDA")
    if args.device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 CUDA 设备不支持 Spatial Probe 的 BF16")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(file_descriptor)
    try:
        torch.save(value, temporary_name)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _dataset(
    data: Path,
    split: str,
    spec: RobotSpec,
) -> QwenExternalSpatialProbeDataset:
    entries = load_manifest(data, split=split)
    return QwenExternalSpatialProbeDataset(data, entries, spec, cache_size=2)


def _loader(
    dataset: QwenExternalSpatialProbeDataset,
    collator: QwenSpatialProbeCollator,
    *,
    samples: int,
    batch_size: int,
    seed: int,
) -> tuple[TaskEpisodeBalancedSampler, DataLoader]:
    sampler = TaskEpisodeBalancedSampler(
        dataset,
        num_samples=samples,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    return sampler, loader


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _extract_frozen_features(
    loader: DataLoader,
    encoder: FrozenQwenLayerPairContextEncoder,
    *,
    processor: QwenVLAProcessorAdapter,
    device: torch.device,
) -> CachedSpatialProbeSplit:
    """每个样本只跑一次 Qwen，并只保留 GT 粗 token 的两层 hidden state。"""

    encoder.train(False)
    layer_features: dict[str, list[torch.Tensor]] = {
        name: [] for name in LAYER_NAMES
    }
    targets: list[torch.Tensor] = []
    image_sizes: list[torch.Tensor] = []
    grid_shapes: list[torch.Tensor] = []
    intrinsics: list[torch.Tensor] = []
    transforms: list[torch.Tensor] = []
    object_positions: list[torch.Tensor] = []
    nearest_token_uv: list[torch.Tensor] = []
    identities: set[str] = set()
    image_token_id = processor.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if not isinstance(image_token_id, int) or image_token_id < 0:
        raise RuntimeError("Qwen tokenizer 没有有效 <|image_pad|> token id")

    for raw_batch in loader:
        qwen_inputs = _move_to_device(raw_batch["qwen_inputs"], device)
        target_uv = raw_batch["target_uv_external"].to(
            device=device,
            non_blocking=device.type == "cuda",
        )
        batch_size = int(target_uv.shape[0])
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=True,
        ):
            context = encoder(qwen_inputs)
        layout = build_external_visual_token_layout(
            qwen_inputs["input_ids"],
            qwen_inputs["image_grid_thw"],
            image_token_id=image_token_id,
            merge_size=processor.config.merge_size,
        )
        target_token_index = nearest_external_visual_token_indices(
            layout,
            target_uv,
        )
        batch_indices = torch.arange(batch_size, device=device)
        layer_features["layer12"].append(
            context.layer12_tokens[batch_indices, target_token_index]
            .detach()
            .float()
            .cpu()
        )
        layer_features["layer24"].append(
            context.layer24_tokens[batch_indices, target_token_index]
            .detach()
            .float()
            .cpu()
        )
        nearest_token_uv.append(
            layout.normalized_centers[
                batch_indices,
                target_token_index,
            ]
            .detach()
            .cpu()
        )
        targets.append(raw_batch["target_uv_external"].float())
        image_sizes.append(raw_batch["image_size_external"].long())
        intrinsics.append(raw_batch["intrinsic_external"].float())
        transforms.append(raw_batch["world_from_external"].float())
        object_positions.append(raw_batch["object_position_m"].float())
        grid_shapes.append(layout.grid_shapes.detach().cpu())
        identities.update(
            f"{trajectory_id}:{int(timestep)}"
            for trajectory_id, timestep in zip(
                raw_batch["trajectory_id"],
                raw_batch["timestep"].tolist(),
                strict=True,
            )
        )

    if not targets:
        raise ValueError("Spatial Probe dataloader 不能为空")
    return CachedSpatialProbeSplit(
        layer12_features=torch.cat(layer_features["layer12"]),
        layer24_features=torch.cat(layer_features["layer24"]),
        target_uv=torch.cat(targets),
        image_sizes_hw=torch.cat(image_sizes),
        grid_shapes_hw=torch.cat(grid_shapes),
        intrinsics=torch.cat(intrinsics),
        world_from_cameras=torch.cat(transforms),
        object_positions_m=torch.cat(object_positions),
        nearest_token_uv=torch.cat(nearest_token_uv),
        unique_windows=len(identities),
    )


def _summarize_cached_predictions(
    cached: CachedSpatialProbeSplit,
    predicted_uv: torch.Tensor,
) -> dict[str, float | int | None]:
    return summarize_spatial_predictions(
        predicted_uv.float().numpy(),
        cached.target_uv.numpy(),
        cached.image_sizes_hw.numpy(),
        cached.grid_shapes_hw.numpy(),
        cached.intrinsics.numpy(),
        cached.world_from_cameras.numpy(),
        cached.object_positions_m.numpy(),
    )


def _run_cached_probes(
    cached: CachedSpatialProbeSplit,
    probes: nn.ModuleDict,
    *,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    seed: int,
    include_position_metrics: bool,
) -> dict[str, Any]:
    training = optimizer is not None
    probes.train(training)
    sample_count = len(cached)
    generator = torch.Generator().manual_seed(seed)
    order = (
        torch.randperm(sample_count, generator=generator)
        if training
        else torch.arange(sample_count)
    )
    loss_numerators = {name: 0.0 for name in LAYER_NAMES}
    predicted: dict[str, list[torch.Tensor]] = {
        name: [] for name in LAYER_NAMES
    }
    features = {
        "layer12": cached.layer12_features,
        "layer24": cached.layer24_features,
    }
    for start in range(0, sample_count, batch_size):
        indices = order[start : start + batch_size]
        target_uv = cached.target_uv[indices].to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        outputs = {
            name: probes[name].decode_selected_tokens(
                features[name][indices].to(device)
            )
            for name in LAYER_NAMES
        }
        losses = {
            name: spatial_probe_loss(output, target_uv)
            for name, output in outputs.items()
        }
        if optimizer is not None:
            sum(losses.values()).backward()
            for name in LAYER_NAMES:
                torch.nn.utils.clip_grad_norm_(
                    probes[name].parameters(),
                    max_norm=10.0,
                )
            optimizer.step()
        current_batch_size = int(target_uv.shape[0])
        for name in LAYER_NAMES:
            loss_numerators[name] += (
                float(losses[name].detach().item()) * current_batch_size
            )
            if include_position_metrics:
                predicted[name].append(outputs[name].detach().cpu())

    result: dict[str, Any] = {
        "samples": sample_count,
        "unique_windows": cached.unique_windows,
        "by_layer": {
            name: {
                "coordinate_mse": loss_numerators[name] / sample_count,
            }
            for name in LAYER_NAMES
        },
    }
    if include_position_metrics:
        for name in LAYER_NAMES:
            result["by_layer"][name]["position"] = _summarize_cached_predictions(
                cached,
                torch.cat(predicted[name]),
            )
        result["references"] = {
            "uniform_image_center": _summarize_cached_predictions(
                cached,
                torch.full_like(cached.target_uv, 0.5),
            ),
            "nearest_visual_token_center": _summarize_cached_predictions(
                cached,
                cached.nearest_token_uv,
            ),
        }
    return result


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    if args.output.exists():
        if not args.output.is_dir():
            raise FileExistsError("Spatial Probe 输出路径已存在且不是目录")
        if any(args.output.iterdir()):
            raise FileExistsError("Spatial Probe 输出目录非空，拒绝覆盖")
    args.output.mkdir(parents=True, exist_ok=True)
    _seed_all(args.seed)
    device = torch.device(args.device)
    spec = RobotSpec()
    dataset_identity = _load_audit_identity(args.data)
    processor = QwenVLAProcessorAdapter.from_pretrained(
        cache_dir=str(args.model_cache),
        local_files_only=True,
    )
    collator = QwenSpatialProbeCollator(processor)
    train_dataset = _dataset(args.data, "train", spec)
    validation_dataset = _dataset(args.data, "val", spec)
    test_dataset = _dataset(args.data, "test", spec)
    _, train_loader = _loader(
        train_dataset,
        collator,
        samples=args.train_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    _, validation_loader = _loader(
        validation_dataset,
        collator,
        samples=args.validation_samples,
        batch_size=args.batch_size,
        seed=1_009,
    )
    _, test_loader = _loader(
        test_dataset,
        collator,
        samples=args.test_samples,
        batch_size=args.batch_size,
        seed=2_017,
    )

    qwen = load_frozen_qwen_v01(
        cache_dir=str(args.model_cache),
        local_files_only=True,
        device=device,
    )
    encoder = FrozenQwenLayerPairContextEncoder(qwen).to(device)
    with torch.no_grad():
        train_cache = _extract_frozen_features(
            train_loader,
            encoder,
            processor=processor,
            device=device,
        )
        validation_cache = _extract_frozen_features(
            validation_loader,
            encoder,
            processor=processor,
            device=device,
        )
        test_cache = _extract_frozen_features(
            test_loader,
            encoder,
            processor=processor,
            device=device,
        )
    extraction_memory: dict[str, int] | None = None
    if device.type == "cuda":
        extraction_memory = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    del encoder, qwen
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    probes = build_matched_linear_probes(seed=args.seed).to(device)
    optimizer = torch.optim.AdamW(
        probes.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    project_root = Path(__file__).resolve().parents[3]
    experiment = {
        "format": QWEN_SPATIAL_PROBE_FORMAT,
        "code_revision": compute_source_revision(project_root),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "dataset": dataset_identity,
        "qwen": {"model_id": QWEN_MODEL_ID, "revision": QWEN_REVISION},
        "processor_config": asdict(processor.config),
        "probe": {
            "layers": [QWEN_LAYER12, QWEN_LAYER24],
            "view": "external/front",
            "target": "projected current object center in continuous normalized UV",
            "architecture": "one Linear(2048,2) position decoder per layer",
            "loss": "MSE on continuous normalized UV",
            "gt_token_selection": (
                "GT UV selects the nearest external visual token; the linear decoder "
                "receives only that token hidden state, not its index or grid coordinate"
            ),
            "subtoken_control": "nearest external visual-token center",
            "qwen_frozen": True,
            "qwen_compute_dtype": "bfloat16",
            "qwen_forward_passes_per_sample": 1,
            "feature_cache": "in-memory selected-token float32 tensors",
            "linear_probe_compute_dtype": "float32",
            "gradient_clipping": "independent norm 10.0 per layer probe",
            "proprio_input": False,
            "future_information": False,
            "same_initialization": all(
                torch.equal(first, second)
                for first, second in zip(
                    probes["layer12"].state_dict().values(),
                    probes["layer24"].state_dict().values(),
                    strict=True,
                )
            ),
            "position_interpretation": (
                "external pixel ray intersected with the GT object-height plane for metrics only"
            ),
        },
        "windows": {
            split: {
                "count": len(dataset),
                "trajectory_count": len(dataset.entries),
                "sha256": dataset.window_sha256,
                "rejected_missing_geometry": dataset.rejected_missing_geometry,
                "rejected_out_of_view": dataset.rejected_out_of_view,
            }
            for split, dataset in (
                ("train", train_dataset),
                ("validation", validation_dataset),
                ("test", test_dataset),
            )
        },
        "trainable_parameters": sum(
            parameter.numel() for parameter in probes.parameters()
        ),
        "sampled_feature_cache": {
            split: {
                "samples": len(cached),
                "unique_windows": cached.unique_windows,
            }
            for split, cached in (
                ("train", train_cache),
                ("validation", validation_cache),
                ("test", test_cache),
            )
        },
        "feature_extraction_memory": extraction_memory,
        "device": str(device),
    }
    experiment_path = args.output / "experiment.json"
    _atomic_write_json(experiment_path, experiment)
    metrics_path = args.output / "metrics.jsonl"
    best: dict[str, dict[str, Any]] = {
        name: {
            "coordinate_mse": float("inf"),
            "epoch": None,
            "state": None,
            "validation": None,
        }
        for name in LAYER_NAMES
    }

    for epoch in range(args.epochs):
        train_metrics = _run_cached_probes(
            train_cache,
            probes,
            batch_size=args.batch_size,
            device=device,
            optimizer=optimizer,
            seed=args.seed + epoch,
            include_position_metrics=False,
        )
        with torch.no_grad():
            validation_metrics = _run_cached_probes(
                validation_cache,
                probes,
                batch_size=args.batch_size,
                device=device,
                optimizer=None,
                seed=1_009,
                include_position_metrics=True,
            )
        for name in LAYER_NAMES:
            coordinate_mse = float(
                validation_metrics["by_layer"][name]["coordinate_mse"]
            )
            if coordinate_mse < best[name]["coordinate_mse"]:
                best[name] = {
                    "coordinate_mse": coordinate_mse,
                    "epoch": epoch + 1,
                    "state": _clone_state(probes[name]),
                    "validation": validation_metrics["by_layer"][name],
                }
        payload = {
            "event": "epoch",
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        if device.type == "cuda":
            payload["memory"] = {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        _append_jsonl(metrics_path, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)

    for name in LAYER_NAMES:
        state = best[name]["state"]
        if state is None:
            raise RuntimeError(f"{name} 没有产生 best probe")
        probes[name].load_state_dict(state)
    with torch.no_grad():
        test_metrics = _run_cached_probes(
            test_cache,
            probes,
            batch_size=args.batch_size,
            device=device,
            optimizer=None,
            seed=2_017,
            include_position_metrics=True,
        )

    checkpoint_path = args.output / "probes.pt"
    checkpoint = {
        "format": QWEN_SPATIAL_PROBE_FORMAT,
        "experiment_sha256": _sha256_file(experiment_path),
        "layers": {
            name: {
                "qwen_layer": QWEN_LAYER12 if name == "layer12" else QWEN_LAYER24,
                "best_epoch": best[name]["epoch"],
                "validation_coordinate_mse": best[name]["coordinate_mse"],
                "state_dict": best[name]["state"],
            }
            for name in LAYER_NAMES
        },
    }
    _atomic_torch_save(checkpoint_path, checkpoint)
    summary = {
        "format": QWEN_SPATIAL_PROBE_FORMAT,
        "complete": True,
        "best": {
            name: {
                "epoch": best[name]["epoch"],
                "validation": best[name]["validation"],
            }
            for name in LAYER_NAMES
        },
        "test": test_metrics,
        "decision": interpret_layer12_probe(
            test_metrics["by_layer"]["layer12"],
            test_metrics["by_layer"]["layer24"],
            test_metrics["references"]["nearest_visual_token_center"],
        ),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
        },
        "limitations": [
            "只诊断 external/front visual token，不直接证明 wrist token 的空间信息",
            "GT 只用于选择方块所在的粗 visual token，因此本实验不测目标检测能力",
            "visual token 到二维网格的映射依赖固定 Qwen Processor 的双图顺序和合并后行优先顺序",
            "世界 XY 误差使用已知物体高度平面进行反投影，只适用于当前桌面 Reach 诊断",
            "线性 probe 成功表示位置可浅层解码；失败不证明完整 Qwen 表征绝对不含位置信息",
            "该结果不替代闭环 Atomic Reach 评估",
        ],
    }
    _atomic_write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
