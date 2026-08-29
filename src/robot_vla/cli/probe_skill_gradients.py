"""E010：冻结 Qwen，不更新参数，测量五技能与 base/event 的梯度 Gram。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from robot_vla.adapters import ProprioNormalizer, ProprioStats
from robot_vla.cli.evaluate_maniskill import _load_audit_identity, _sha256_file
from robot_vla.cli.train_stage1 import compute_source_revision
from robot_vla.contracts import PICK_AND_PLACE_SKILLS, RobotSpec
from robot_vla.data.collator import QwenVLACollator
from robot_vla.data.dataset import ActionChunkDataset
from robot_vla.data.trajectory import load_manifest
from robot_vla.diagnostics.gradient_conflict import (
    CONFIRMATION_PRIMITIVE_LABELS,
    CONFIRMATION_VECTOR_LABELS,
    GRADIENT_PROBE_FORMAT,
    PRIMITIVE_GRADIENT_GROUPS,
    add_all_trainable_gram,
    assess_conflict,
    build_episode_paired_plan,
    expand_confirmation_grams,
    gradient_group_for_parameter,
    index_measurements,
    measurement_identity,
    summarize_measurements,
)
from robot_vla.model.factory import load_qwen_vla_policy
from robot_vla.model.qwen_processor import QwenVLAProcessorAdapter
from robot_vla.training.checkpoint import load_stage1_policy_checkpoint
from robot_vla.training.stage1 import move_to_device

CONFIRMATION_SKILLS = ("reach", "transport")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="可重复；label 和 checkpoint 路径用等号分隔",
    )
    parser.add_argument(
        "--confirmation-checkpoint",
        action="append",
        required=True,
        dest="confirmation_checkpoints",
        metavar="LABEL",
    )
    parser.add_argument("--qwen-context-layer", type=int, choices=(12, 24), default=12)
    parser.add_argument("--train-repeats", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--validation-repeats", type=int, default=5)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--probe-seed", type=int, default=10_010)
    parser.add_argument("--event-loss-weight", type=float, default=0.25)
    parser.add_argument("--executed-action-steps", type=int, default=4)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, "timestamp": _utc_now(), **values}, sort_keys=True), flush=True)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parse_checkpoints(values: Sequence[str]) -> list[tuple[str, Path]]:
    checkpoints: list[tuple[str, Path]] = []
    for raw in values:
        label, separator, path_text = raw.partition("=")
        label = label.strip()
        path_text = path_text.strip()
        if not separator or not label or not path_text:
            raise ValueError(f"checkpoint 参数必须是 LABEL=PATH，实际为 {raw!r}")
        if any(character.isspace() for character in label):
            raise ValueError(f"checkpoint label 不能包含空白: {label!r}")
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到 checkpoint: {path}")
        checkpoints.append((label, path))
    labels = [label for label, _ in checkpoints]
    if len(labels) != len(set(labels)):
        raise ValueError("checkpoint label 不能重复")
    return checkpoints


def _validate_args(
    args: argparse.Namespace,
    checkpoints: Sequence[tuple[str, Path]],
) -> tuple[str, ...]:
    for name in (
        "train_repeats",
        "train_batch_size",
        "validation_repeats",
        "validation_batch_size",
        "executed_action_steps",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} 必须为正整数")
    if args.probe_seed < 0:
        raise ValueError("probe_seed 不能为负数")
    if not math.isfinite(args.event_loss_weight) or args.event_loss_weight < 0:
        raise ValueError("event_loss_weight 必须是有限非负数")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("E010 需要支持 BF16 的 CUDA GPU")
    labels = {label for label, _ in checkpoints}
    confirmation = tuple(str(label) for label in args.confirmation_checkpoints)
    if not confirmation or len(confirmation) != len(set(confirmation)):
        raise ValueError("confirmation checkpoint label 必须非空且不能重复")
    unknown = sorted(set(confirmation) - labels)
    if unknown:
        raise ValueError(f"confirmation checkpoint 不在 checkpoint 列表中: {unknown}")
    return confirmation


def _samples_by_episode(dataset: ActionChunkDataset) -> dict[str, dict[str, list[int]]]:
    grouped: dict[str, dict[str, list[int]]] = {}
    for sample_index, (entry_index, timestep) in enumerate(dataset.index):
        entry = dataset.entries[entry_index]
        arrays = dataset.store.get(entry)
        skill_id = int(arrays.skill_id[timestep])
        if not 0 <= skill_id < len(PICK_AND_PLACE_SKILLS):
            continue
        skill = PICK_AND_PLACE_SKILLS[skill_id]
        grouped.setdefault(entry.trajectory_id, {}).setdefault(skill, []).append(sample_index)
    return grouped


def _sample_plan(
    dataset: ActionChunkDataset,
    *,
    skills: Sequence[str],
    repeats: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    return build_episode_paired_plan(
        _samples_by_episode(dataset),
        skills=skills,
        repeats=repeats,
        batch_size=batch_size,
        seed=seed,
    )


def _expected_identities(
    checkpoints: Sequence[tuple[str, Path]],
    confirmation: Sequence[str],
    *,
    train_repeats: int,
    validation_repeats: int,
) -> list[dict[str, Any]]:
    identities = [
        {"checkpoint_label": label, "stage": "discovery", "repeat": repeat}
        for label, _ in checkpoints
        for repeat in range(train_repeats)
    ]
    identities.extend(
        {"checkpoint_label": label, "stage": "confirmation", "repeat": repeat}
        for label, _ in checkpoints
        if label in confirmation
        for repeat in range(validation_repeats)
    )
    return identities


def _checkpoint_identities(checkpoints: Sequence[tuple[str, Path]]) -> list[dict[str, Any]]:
    return [
        {
            "label": label,
            "path": str(path),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for label, path in checkpoints
    ]


def _manifest(
    args: argparse.Namespace,
    checkpoints: Sequence[tuple[str, Path]],
    confirmation: Sequence[str],
    *,
    discovery_plan: Sequence[Mapping[str, Any]],
    confirmation_plan: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    return {
        "format": GRADIENT_PROBE_FORMAT,
        "created_at": _utc_now(),
        "probe_code_revision": compute_source_revision(project_root),
        "dataset": _load_audit_identity(args.data),
        "data_path": str(args.data.resolve()),
        "model_cache_path": str(args.model_cache.resolve()),
        "checkpoints": _checkpoint_identities(checkpoints),
        "confirmation_checkpoint_labels": list(confirmation),
        "qwen_context_layer": args.qwen_context_layer,
        "qwen_trainable": False,
        "objective": {
            "total": "base_loss + event_loss_weight * event_loss",
            "event_loss_weight": args.event_loss_weight,
            "executed_action_steps": args.executed_action_steps,
            "autocast_dtype": "bfloat16",
            "optimizer_step": False,
            "gradient_clipping": False,
            "gram_accumulation_dtype": "float32",
        },
        "gradient_groups": list(PRIMITIVE_GRADIENT_GROUPS),
        "discovery": {
            "split": "train",
            "skills": list(PICK_AND_PLACE_SKILLS),
            "repeats": args.train_repeats,
            "batch_size": args.train_batch_size,
            "plan_seed": args.probe_seed + 101,
            "flow_seed_base": args.probe_seed + 10_000,
            "sample_plan": list(discovery_plan),
        },
        "confirmation": {
            "split": "val",
            "skills": list(CONFIRMATION_SKILLS),
            "primitive_vector_labels": list(CONFIRMATION_PRIMITIVE_LABELS),
            "vector_labels": list(CONFIRMATION_VECTOR_LABELS),
            "repeats": args.validation_repeats,
            "batch_size": args.validation_batch_size,
            "plan_seed": args.probe_seed + 202,
            "flow_seed_base": args.probe_seed + 20_000,
            "sample_plan": list(confirmation_plan),
        },
        "expected_measurement_identities": _expected_identities(
            checkpoints,
            confirmation,
            train_repeats=args.train_repeats,
            validation_repeats=args.validation_repeats,
        ),
        "thresholds": {
            "median_cosine_max": -0.10,
            "discovery_negative_repeats": f"6/{args.train_repeats}",
            "confirmation_negative_repeats": f"4/{args.validation_repeats}",
        },
    }


def _write_or_validate_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    comparable = dict(manifest)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable["created_at"] = existing.get("created_at")
        if existing != comparable:
            raise ValueError("已有 E010 manifest 与当前实验身份不一致")
        return
    if path.parent.exists() and any(path.parent.iterdir()):
        raise FileExistsError("E010 输出目录非空但缺少 manifest，拒绝覆盖")
    _atomic_write_json(path, manifest)


def _read_measurements(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"measurements.jsonl 第 {line_number} 行为空")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"measurements.jsonl 第 {line_number} 行不是对象")
        records.append(value)
    index_measurements(records)
    return records


def _trainable_named_parameters(policy: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    if any(parameter.requires_grad for parameter in policy.context_encoder.parameters()):
        raise ValueError("E010 的 Qwen Context Encoder 必须完全冻结")
    selected = tuple(
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    )
    if not selected:
        raise ValueError("E010 没有 Adapter/Expert trainable parameter")
    unexpected = [
        name
        for name, _ in selected
        if not name.startswith("adapter.") and not name.startswith("expert.")
    ]
    if unexpected:
        raise ValueError(f"E010 发现边界外 trainable parameter: {unexpected}")
    for name, _ in selected:
        gradient_group_for_parameter(name)
    observed_groups = {gradient_group_for_parameter(name) for name, _ in selected}
    missing = sorted(set(PRIMITIVE_GRADIENT_GROUPS) - observed_groups)
    if missing:
        raise ValueError(f"E010 模型缺少 gradient group: {missing}")
    return selected


def _parameter_hash(named_parameters: Sequence[tuple[str, nn.Parameter]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        value = parameter.detach().cpu().contiguous()
        metadata = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(4, "big"))
        digest.update(metadata)
        raw = value.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _parameter_versions(
    named_parameters: Sequence[tuple[str, nn.Parameter]],
) -> tuple[int, ...]:
    return tuple(int(parameter._version) for _, parameter in named_parameters)


def _cpu_gradients(
    gradients: Sequence[torch.Tensor | None],
    named_parameters: Sequence[tuple[str, nn.Parameter]],
) -> tuple[torch.Tensor, ...]:
    if len(gradients) != len(named_parameters):
        raise ValueError("E010 gradient 数量与 parameter 数量不一致")
    resolved: list[torch.Tensor] = []
    for gradient, (name, parameter) in zip(gradients, named_parameters, strict=True):
        if gradient is None:
            raise RuntimeError(f"E010 parameter 没有 gradient: {name}")
        if gradient.shape != parameter.shape or not torch.isfinite(gradient).all():
            raise FloatingPointError(f"E010 gradient shape/finite 无效: {name}")
        resolved.append(gradient.detach().to(device="cpu", dtype=torch.float32).contiguous())
    return tuple(resolved)


def _group_grams(
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    vectors: Sequence[Sequence[torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, list[list[float]]]:
    if len(vectors) < 2 or any(len(vector) != len(named_parameters) for vector in vectors):
        raise ValueError("E010 gradient vector 数量或 parameter 对齐无效")
    size = len(vectors)
    accumulators = {
        group: torch.zeros(size, size, dtype=torch.float32)
        for group in PRIMITIVE_GRADIENT_GROUPS
    }
    for parameter_index, (name, _) in enumerate(named_parameters):
        group = gradient_group_for_parameter(name)
        stacked = torch.stack(
            [vector[parameter_index].reshape(-1) for vector in vectors], dim=0
        ).to(device=device)
        local = stacked @ stacked.transpose(0, 1)
        if not torch.isfinite(local).all():
            raise FloatingPointError(f"E010 {name} Gram 包含 NaN/Inf")
        accumulators[group].add_(local.detach().cpu())
        del stacked, local
    return {group: value.tolist() for group, value in accumulators.items()}


def _batch(
    dataset: ActionChunkDataset,
    collator: QwenVLACollator,
    indices: Sequence[int],
    *,
    device: torch.device,
) -> dict[str, Any]:
    return move_to_device(collator([dataset[int(index)] for index in indices]), device)


def _flow_output(
    policy: nn.Module,
    batch: Mapping[str, Any],
    *,
    event_loss_weight: float,
    executed_action_steps: int,
    flow_seed: int,
    device: torch.device,
) -> Any:
    generator = torch.Generator(device=device)
    generator.manual_seed(flow_seed)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        return policy.flow_matching_loss(
            batch["qwen_inputs"],
            batch["proprio"],
            batch["action"],
            batch["action_mask"],
            event_mask=batch["event_mask"],
            event_loss_weight=event_loss_weight,
            executed_action_steps=executed_action_steps,
            generator=generator,
        )


def _loss_stats(output: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "loss": float(output.loss.detach().float().item()),
        "base_loss": float(output.base_loss.detach().float().item()),
        "event_loss": float(output.event_loss.detach().float().item()),
        "valid_action_steps": int(batch["action_mask"].sum().item()),
        "critical_action_steps": int(output.critical_mask.sum().item()),
    }


def _discovery_measurement(
    policy: nn.Module,
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    dataset: ActionChunkDataset,
    collator: QwenVLACollator,
    *,
    checkpoint_label: str,
    repeat_plan: Mapping[str, Any],
    flow_seed: int,
    event_loss_weight: float,
    executed_action_steps: int,
    parameter_sha256: str,
    parameter_versions: tuple[int, ...],
    device: torch.device,
) -> dict[str, Any]:
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    vectors: list[tuple[torch.Tensor, ...]] = []
    losses: dict[str, Any] = {}
    parameters = [parameter for _, parameter in named_parameters]
    for skill in PICK_AND_PLACE_SKILLS:
        batch = _batch(dataset, collator, repeat_plan["batches"][skill], device=device)
        output = _flow_output(
            policy,
            batch,
            event_loss_weight=event_loss_weight,
            executed_action_steps=executed_action_steps,
            flow_seed=flow_seed,
            device=device,
        )
        losses[skill] = _loss_stats(output, batch)
        gradients = torch.autograd.grad(output.loss.float(), parameters)
        vectors.append(_cpu_gradients(gradients, named_parameters))
        del gradients, output, batch
        torch.cuda.empty_cache()
    primitive = _group_grams(named_parameters, vectors, device=device)
    del vectors
    group_grams = add_all_trainable_gram(primitive, vector_count=len(PICK_AND_PLACE_SKILLS))
    if _parameter_versions(named_parameters) != parameter_versions:
        raise RuntimeError("E010 discovery 期间 parameter version 发生变化")
    return {
        "format": GRADIENT_PROBE_FORMAT,
        "checkpoint_label": checkpoint_label,
        "stage": "discovery",
        "repeat": int(repeat_plan["repeat"]),
        "vector_labels": list(PICK_AND_PLACE_SKILLS),
        "episodes": list(repeat_plan["episodes"]),
        "sample_indices_by_skill": dict(repeat_plan["batches"]),
        "flow_seed": flow_seed,
        "losses": losses,
        "group_grams": group_grams,
        "parameter_sha256_before_checkpoint": parameter_sha256,
        "parameter_versions_unchanged": True,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.monotonic() - started,
        "completed_at": _utc_now(),
    }


def _confirmation_measurement(
    policy: nn.Module,
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    dataset: ActionChunkDataset,
    collator: QwenVLACollator,
    *,
    checkpoint_label: str,
    repeat_plan: Mapping[str, Any],
    flow_seed: int,
    event_loss_weight: float,
    executed_action_steps: int,
    parameter_sha256: str,
    parameter_versions: tuple[int, ...],
    device: torch.device,
) -> dict[str, Any]:
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    primitive_vectors: list[tuple[torch.Tensor, ...]] = []
    losses: dict[str, Any] = {}
    parameters = [parameter for _, parameter in named_parameters]
    for skill in CONFIRMATION_SKILLS:
        batch = _batch(dataset, collator, repeat_plan["batches"][skill], device=device)
        output = _flow_output(
            policy,
            batch,
            event_loss_weight=event_loss_weight,
            executed_action_steps=executed_action_steps,
            flow_seed=flow_seed,
            device=device,
        )
        losses[skill] = _loss_stats(output, batch)
        base_gradients = torch.autograd.grad(
            output.base_loss.float(), parameters, retain_graph=True
        )
        primitive_vectors.append(_cpu_gradients(base_gradients, named_parameters))
        del base_gradients
        weighted_event = float(event_loss_weight) * output.event_loss.float()
        event_gradients = torch.autograd.grad(weighted_event, parameters)
        primitive_vectors.append(_cpu_gradients(event_gradients, named_parameters))
        del event_gradients, weighted_event, output, batch
        torch.cuda.empty_cache()
    primitive_grams = _group_grams(named_parameters, primitive_vectors, device=device)
    del primitive_vectors
    group_grams = expand_confirmation_grams(primitive_grams)
    if _parameter_versions(named_parameters) != parameter_versions:
        raise RuntimeError("E010 confirmation 期间 parameter version 发生变化")
    return {
        "format": GRADIENT_PROBE_FORMAT,
        "checkpoint_label": checkpoint_label,
        "stage": "confirmation",
        "repeat": int(repeat_plan["repeat"]),
        "primitive_vector_labels": list(CONFIRMATION_PRIMITIVE_LABELS),
        "vector_labels": list(CONFIRMATION_VECTOR_LABELS),
        "episodes": list(repeat_plan["episodes"]),
        "sample_indices_by_skill": dict(repeat_plan["batches"]),
        "flow_seed": flow_seed,
        "losses": losses,
        "group_grams": group_grams,
        "parameter_sha256_before_checkpoint": parameter_sha256,
        "parameter_versions_unchanged": True,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.monotonic() - started,
        "completed_at": _utc_now(),
    }


def _expected_identity_set(manifest: Mapping[str, Any]) -> set[tuple[str, str, int]]:
    return {
        (
            str(value["checkpoint_label"]),
            str(value["stage"]),
            int(value["repeat"]),
        )
        for value in manifest["expected_measurement_identities"]
    }


def _write_incomplete_summary(
    path: Path,
    *,
    completed: int,
    expected: int,
) -> None:
    _atomic_write_json(
        path,
        {
            "format": GRADIENT_PROBE_FORMAT,
            "complete": False,
            "completed_measurements": completed,
            "expected_measurements": expected,
            "updated_at": _utc_now(),
        },
    )


def run(args: argparse.Namespace) -> None:
    checkpoints = _parse_checkpoints(args.checkpoint)
    confirmation = _validate_args(args, checkpoints)
    random.seed(args.probe_seed)
    np.random.seed(args.probe_seed)
    torch.manual_seed(args.probe_seed)
    torch.cuda.manual_seed_all(args.probe_seed)
    device = torch.device(args.device)

    spec = RobotSpec()
    if args.executed_action_steps > spec.action_horizon:
        raise ValueError("executed_action_steps 不能超过 action horizon")
    stats = ProprioStats.from_json(args.data / "proprio_stats.json")
    stats.validate(spec)
    normalizer = ProprioNormalizer(stats, spec)
    train_entries = load_manifest(args.data, split="train")
    validation_entries = load_manifest(args.data, split="val")
    train_dataset = ActionChunkDataset(
        str(args.data), train_entries, spec, normalizer, cache_size=2
    )
    validation_dataset = ActionChunkDataset(
        str(args.data), validation_entries, spec, normalizer, cache_size=2
    )
    discovery_plan = _sample_plan(
        train_dataset,
        skills=PICK_AND_PLACE_SKILLS,
        repeats=args.train_repeats,
        batch_size=args.train_batch_size,
        seed=args.probe_seed + 101,
    )
    confirmation_plan = _sample_plan(
        validation_dataset,
        skills=CONFIRMATION_SKILLS,
        repeats=args.validation_repeats,
        batch_size=args.validation_batch_size,
        seed=args.probe_seed + 202,
    )
    manifest = _manifest(
        args,
        checkpoints,
        confirmation,
        discovery_plan=discovery_plan,
        confirmation_plan=confirmation_plan,
    )
    manifest_path = args.output / "probe-manifest.json"
    measurements_path = args.output / "measurements.jsonl"
    summary_path = args.output / "probe-summary.json"
    _write_or_validate_manifest(manifest_path, manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_identities = _expected_identity_set(manifest)
    records = _read_measurements(measurements_path)
    completed = index_measurements(records)
    unexpected = sorted(set(completed) - expected_identities)
    if unexpected:
        raise ValueError(f"E010 measurements 包含 manifest 外 identity: {unexpected}")
    if set(completed) == expected_identities and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("complete") is True:
            _progress("probe_skip_complete", completed=len(completed), expected=len(expected_identities))
            return
    _write_incomplete_summary(
        summary_path, completed=len(completed), expected=len(expected_identities)
    )
    _progress("probe_start", completed=len(completed), expected=len(expected_identities))

    processor = QwenVLAProcessorAdapter.from_pretrained(
        cache_dir=str(args.model_cache), local_files_only=True
    )
    collator = QwenVLACollator(processor, spec)
    policy = load_qwen_vla_policy(
        cache_dir=str(args.model_cache),
        local_files_only=True,
        device=device,
        context_layer=args.qwen_context_layer,
    )
    policy.train()
    named_parameters = _trainable_named_parameters(policy)
    integrity: dict[str, Any] = {}
    checkpoint_metadata: dict[str, Any] = {}

    for checkpoint_label, checkpoint_path in checkpoints:
        _progress("checkpoint_load_start", checkpoint_label=checkpoint_label)
        metadata = load_stage1_policy_checkpoint(
            checkpoint_path,
            policy,
            spec,
            processor.config,
            stats,
        )
        checkpoint_metadata[checkpoint_label] = metadata
        gc.collect()
        torch.cuda.empty_cache()
        parameter_sha256 = _parameter_hash(named_parameters)
        parameter_versions = _parameter_versions(named_parameters)
        _progress(
            "checkpoint_load_complete",
            checkpoint_label=checkpoint_label,
            parameter_sha256=parameter_sha256,
        )

        for repeat_plan in discovery_plan:
            identity = (checkpoint_label, "discovery", int(repeat_plan["repeat"]))
            if identity in completed:
                _progress(
                    "measurement_skip_complete",
                    checkpoint_label=checkpoint_label,
                    stage="discovery",
                    repeat=identity[2],
                )
                continue
            _progress(
                "measurement_start",
                checkpoint_label=checkpoint_label,
                stage="discovery",
                repeat=identity[2],
            )
            record = _discovery_measurement(
                policy,
                named_parameters,
                train_dataset,
                collator,
                checkpoint_label=checkpoint_label,
                repeat_plan=repeat_plan,
                flow_seed=args.probe_seed + 10_000 + identity[2],
                event_loss_weight=args.event_loss_weight,
                executed_action_steps=args.executed_action_steps,
                parameter_sha256=parameter_sha256,
                parameter_versions=parameter_versions,
                device=device,
            )
            if measurement_identity(record) != identity:
                raise RuntimeError("E010 discovery measurement identity 漂移")
            _append_jsonl(measurements_path, record)
            completed[identity] = record
            _write_incomplete_summary(
                summary_path,
                completed=len(completed),
                expected=len(expected_identities),
            )
            _progress(
                "measurement_complete",
                checkpoint_label=checkpoint_label,
                stage="discovery",
                repeat=identity[2],
                completed=len(completed),
                expected=len(expected_identities),
                elapsed_seconds=record["elapsed_seconds"],
            )

        if checkpoint_label in confirmation:
            for repeat_plan in confirmation_plan:
                identity = (
                    checkpoint_label,
                    "confirmation",
                    int(repeat_plan["repeat"]),
                )
                if identity in completed:
                    _progress(
                        "measurement_skip_complete",
                        checkpoint_label=checkpoint_label,
                        stage="confirmation",
                        repeat=identity[2],
                    )
                    continue
                _progress(
                    "measurement_start",
                    checkpoint_label=checkpoint_label,
                    stage="confirmation",
                    repeat=identity[2],
                )
                record = _confirmation_measurement(
                    policy,
                    named_parameters,
                    validation_dataset,
                    collator,
                    checkpoint_label=checkpoint_label,
                    repeat_plan=repeat_plan,
                    flow_seed=args.probe_seed + 20_000 + identity[2],
                    event_loss_weight=args.event_loss_weight,
                    executed_action_steps=args.executed_action_steps,
                    parameter_sha256=parameter_sha256,
                    parameter_versions=parameter_versions,
                    device=device,
                )
                if measurement_identity(record) != identity:
                    raise RuntimeError("E010 confirmation measurement identity 漂移")
                _append_jsonl(measurements_path, record)
                completed[identity] = record
                _write_incomplete_summary(
                    summary_path,
                    completed=len(completed),
                    expected=len(expected_identities),
                )
                _progress(
                    "measurement_complete",
                    checkpoint_label=checkpoint_label,
                    stage="confirmation",
                    repeat=identity[2],
                    completed=len(completed),
                    expected=len(expected_identities),
                    elapsed_seconds=record["elapsed_seconds"],
                )

        parameter_sha256_after = _parameter_hash(named_parameters)
        versions_unchanged = _parameter_versions(named_parameters) == parameter_versions
        if parameter_sha256_after != parameter_sha256 or not versions_unchanged:
            raise RuntimeError(f"E010 {checkpoint_label} probe 修改了 Adapter/Expert parameter")
        integrity[checkpoint_label] = {
            "parameter_sha256_before": parameter_sha256,
            "parameter_sha256_after": parameter_sha256_after,
            "parameter_versions_unchanged": versions_unchanged,
        }
        _progress("checkpoint_complete", checkpoint_label=checkpoint_label)

    records = _read_measurements(measurements_path)
    indexed = index_measurements(records)
    if set(indexed) != expected_identities:
        missing = sorted(expected_identities - set(indexed))
        raise RuntimeError(f"E010 measurement 不完整: {missing}")
    rows = summarize_measurements(records)
    assessment = assess_conflict(rows, confirmation_checkpoints=confirmation)
    summary = {
        "format": GRADIENT_PROBE_FORMAT,
        "complete": True,
        "completed_measurements": len(records),
        "expected_measurements": len(expected_identities),
        "completed_at": _utc_now(),
        "parameter_integrity": integrity,
        "checkpoint_metadata": checkpoint_metadata,
        "assessment": assessment,
        "summary_rows": rows,
    }
    _atomic_write_json(summary_path, summary)
    _progress(
        "probe_complete",
        completed=len(records),
        expected=len(expected_identities),
        confirmed_checkpoint_labels=assessment["confirmed_checkpoint_labels"],
    )


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
