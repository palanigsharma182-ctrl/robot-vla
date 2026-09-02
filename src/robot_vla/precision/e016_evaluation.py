"""E016-P1 fresh-validation calibration、test-once perception 与 goal-memory replay。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_vla.contracts import RobotSpec
from robot_vla.data.trajectory import load_manifest, resolve_trajectory_path
from robot_vla.data.writer import plan_scene_splits
from robot_vla.observation import rotation_6d_to_matrix
from robot_vla.precision.checkpoint import PrecisionCheckpointRole, load_precision_checkpoint
from robot_vla.precision.data import (
    PrecisionRGBDataset,
    audit_precision_dataset,
    canonical_sha256,
    file_sha256,
    load_precision_label_manifest,
)
from robot_vla.precision.e016_training import E016P1Config, load_e016_p1_config
from robot_vla.precision.geometry import project_base_point_to_normalized_uv
from robot_vla.precision.memory_evaluation import (
    GoalReplayFrame,
    GoalWriteCalibration,
    calibrate_goal_write_threshold,
    replay_goal_memory,
    select_memory_max_age,
    summarize_goal_memory_replay,
)
from robot_vla.precision.observability import (
    GOAL_OBSERVABILITY_SEMANTICS,
    GoalWriteEvidence,
    derive_goal_observability,
    mask_probability_at_normalized_uv,
)
from robot_vla.precision.outliers import assert_public_payload_safe, geometry_conditioning
from robot_vla.precision.state_memory import GoalMemoryConfig
from robot_vla.precision.training import _build_loader, _to_device, source_tree_sha256
from robot_vla.sim import PICK_CUBE_TO_REGION_ENV_ID

E016_P1_EVALUATION_VERSION = "e016-p1-fresh-held-out/v1"
E016_P1_RULES_VERSION = "e016-p1-frozen-memory-rules/v1"
E016_P1_TEST_CLAIM_VERSION = "e016-p1-test-once-claim/v1"
E016_P1_PUBLIC_VERSION = "e016-p1-formal-precision-public/v1"
_CALIBRATION_RECEIPT_FILES = (
    "observability_sidecar.jsonl",
    "prediction_rows.jsonl",
    "memory_replay.jsonl",
    "fresh_registry_audit.json",
    "frozen_rules.json",
    "summary.json",
    "config_snapshot.json",
)


@dataclass(frozen=True)
class _EvaluationContext:
    config: E016P1Config
    source_tree_sha256: str
    registry_audit: dict[str, Any]
    checkpoint_sha256: str
    checkpoint_parameter_sha256: str
    checkpoint_provenance_sha256: str
    training_source_tree_sha256: str
    training_receipt_sha256: str
    training_receipt: dict[str, Any]
    model: Any


@dataclass(frozen=True)
class _CollectedSplit:
    rows: list[dict[str, Any]]
    replay_frames: list[GoalReplayFrame]
    invalid_geometry_count: int
    perception_summary: dict[str, Any]


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} 不存在: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是 JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def audit_e016_fresh_registry(
    *,
    deployable_root: str | Path,
    label_root: str | Path,
    config: E016P1Config,
) -> dict[str, Any]:
    """只读 manifests/collector audit，test privileged label NPZ 保持未读。"""

    deployable = Path(deployable_root)
    labels = Path(label_root)
    collection = _read_json(deployable / "collection_config.json", "collection config")
    expected_collection = {
        "environment_id": PICK_CUBE_TO_REGION_ENV_ID,
        "train": config.fresh_held_out.collector_train_trajectories,
        "val": config.fresh_held_out.validation_trajectories,
        "test": config.fresh_held_out.test_trajectories,
        "start_seed": config.fresh_held_out.start_seed,
        "max_candidates": config.fresh_held_out.max_candidates,
        "precision_label_sidecar": True,
        "precision_label_audit_deferred": True,
    }
    if collection != expected_collection:
        raise RuntimeError(
            f"E016-P1 fresh collection config 漂移: {collection} != {expected_collection}"
        )
    trajectories = load_manifest(deployable)
    label_entries = load_precision_label_manifest(labels)
    trajectory_by_id = {entry.trajectory_id: entry for entry in trajectories}
    label_by_id = {entry.trajectory_id: entry for entry in label_entries}
    if set(trajectory_by_id) != set(label_by_id):
        raise RuntimeError("E016-P1 fresh source/label manifest 集合不一致")
    expected_counts = {
        "train": config.fresh_held_out.collector_train_trajectories,
        "val": config.fresh_held_out.validation_trajectories,
        "test": config.fresh_held_out.test_trajectories,
    }
    actual_counts = Counter(entry.split for entry in trajectories)
    if dict(actual_counts) != expected_counts:
        raise RuntimeError(
            f"E016-P1 fresh split count 漂移: {dict(actual_counts)} != {expected_counts}"
        )
    seeds = [int(entry.randomization["seed"]) for entry in trajectories]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("E016-P1 fresh seed 重复")
    start = config.fresh_held_out.start_seed
    stop = start + config.fresh_held_out.max_candidates
    if any(not start <= seed < stop for seed in seeds):
        raise RuntimeError("E016-P1 fresh seed 超出预注册范围")
    candidate_seeds = list(range(start, stop))
    candidate_scenes = [f"{PICK_CUBE_TO_REGION_ENV_ID}:seed={seed}" for seed in candidate_seeds]
    total = sum(expected_counts.values())
    split_map = plan_scene_splits(
        candidate_scenes,
        train_fraction=expected_counts["train"] / total,
        val_fraction=expected_counts["val"] / total,
    )
    rows: list[dict[str, Any]] = []
    for trajectory_id in sorted(trajectory_by_id):
        trajectory = trajectory_by_id[trajectory_id]
        label = label_by_id[trajectory_id]
        seed = int(trajectory.randomization["seed"])
        if split_map.get(trajectory.scene_id) != trajectory.split:
            raise RuntimeError("E016-P1 fresh scene split 不符合确定性计划")
        if (
            trajectory.split != label.split
            or trajectory.scene_id != label.scene_id
            or trajectory.num_steps != label.num_steps
        ):
            raise RuntimeError("E016-P1 fresh source/label metadata 不一致")
        # 这里只检查存在性，不打开 test NPZ；完整 hash/array audit 延迟到 claim 之后。
        if not resolve_trajectory_path(deployable, trajectory.file).is_file():
            raise FileNotFoundError("E016-P1 fresh trajectory file 不存在")
        label_path = (labels.resolve() / label.file).resolve()
        if not label_path.is_relative_to(labels.resolve()) or not label_path.is_file():
            raise FileNotFoundError("E016-P1 fresh label file 不存在或越界")
        rows.append(
            {
                "seed": seed,
                "split": trajectory.split,
                "source_meta": trajectory.to_dict(),
                "label_meta": label.to_dict(),
            }
        )
    deployable_audit_path = deployable / "audit_report.json"
    deployable_audit = _read_json(deployable_audit_path, "deployable audit")
    if (
        deployable_audit.get("trajectory_count") != total
        or deployable_audit.get("split_trajectory_counts") != expected_counts
        or deployable_audit.get("success_rate") != 1.0
        or deployable_audit.get("wrist_image_shape") != [128, 128, 3]
        or deployable_audit.get("manifest_sha256") != file_sha256(deployable / "manifest.jsonl")
    ):
        raise RuntimeError("E016-P1 deployable collector audit 漂移或未通过")
    registry_identity = canonical_sha256(
        {
            "version": E016_P1_EVALUATION_VERSION,
            "collection_config": collection,
            "deployable_audit_sha256": file_sha256(deployable_audit_path),
            "deployable_manifest_sha256": file_sha256(deployable / "manifest.jsonl"),
            "label_manifest_sha256": file_sha256(labels / "manifest.jsonl"),
            "rows": rows,
        }
    )
    return {
        "version": E016_P1_EVALUATION_VERSION,
        "passed": True,
        "registry_identity_sha256": registry_identity,
        "deployable_dataset_sha256": deployable_audit["dataset_sha256"],
        "deployable_audit_sha256": file_sha256(deployable_audit_path),
        "deployable_manifest_sha256": file_sha256(deployable / "manifest.jsonl"),
        "label_manifest_sha256": file_sha256(labels / "manifest.jsonl"),
        "split_trajectory_counts": expected_counts,
        "seed_count": len(seeds),
        "seed_min": min(seeds),
        "seed_max": max(seeds),
        "test_privileged_label_file_read_count": 0,
        "test_model_forward_count": 0,
    }


def _validate_non_test_split_files(
    *,
    deployable_root: Path,
    label_root: Path,
    split: str,
) -> str:
    if split != "val":
        raise ValueError("E016-P1 pre-test file audit 只允许 validation")
    sources = {entry.trajectory_id: entry for entry in load_manifest(deployable_root, split=split)}
    labels = {
        entry.trajectory_id: entry
        for entry in load_precision_label_manifest(label_root, split=split)
    }
    if set(sources) != set(labels):
        raise RuntimeError("E016-P1 validation source/label 集合不一致")
    files = []
    for trajectory_id in sorted(sources):
        source = sources[trajectory_id]
        label = labels[trajectory_id]
        source_path = resolve_trajectory_path(deployable_root, source.file)
        label_path = (label_root.resolve() / label.file).resolve()
        if not label_path.is_relative_to(label_root.resolve()):
            raise RuntimeError("E016-P1 validation label path 越界")
        source_sha256 = file_sha256(source_path)
        label_sha256 = file_sha256(label_path)
        if source_sha256 != label.source_trajectory_sha256:
            raise RuntimeError("E016-P1 validation source/label hash 绑定漂移")
        files.append(
            {
                "trajectory_id": trajectory_id,
                "source_sha256": source_sha256,
                "label_sha256": label_sha256,
                "source_meta": source.to_dict(),
                "label_meta": label.to_dict(),
            }
        )
    return canonical_sha256({"split": split, "files": files})


def _load_context(
    *,
    deployable_root: Path,
    label_root: Path,
    config_path: Path,
    training_output: Path,
    repository_root: Path,
) -> _EvaluationContext:
    config = load_e016_p1_config(config_path)
    source_identity = source_tree_sha256(repository_root)
    registry = audit_e016_fresh_registry(
        deployable_root=deployable_root,
        label_root=label_root,
        config=config,
    )
    receipt_path = training_output / "checkpoint_receipt.json"
    receipt = _read_json(receipt_path, "E016-P1 training receipt")
    if (
        receipt.get("version") != config.version
        or receipt.get("passed") is not True
        or receipt.get("training_config_sha256") != config.sha256
        or receipt.get("checkpoint_eligible_for_fresh_held_out") is not True
        or receipt.get("test_split_read") is not False
        or receipt.get("fresh_held_out_read") is not False
    ):
        raise RuntimeError("E016-P1 training receipt 不允许进入 fresh held-out")
    checkpoint = receipt.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise TypeError("E016-P1 training receipt checkpoint 必须是 object")
    checkpoint_sha256 = str(checkpoint["checkpoint_sha256"])
    provenance_sha256 = str(checkpoint["provenance_sha256"])
    loaded = load_precision_checkpoint(
        training_output / "precision-formal.pt",
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_provenance_sha256=provenance_sha256,
        expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
    )
    if (
        loaded.provenance.training_config_sha256 != config.sha256
        or loaded.provenance.source_tree_sha256 != receipt["source_tree_sha256"]
    ):
        raise RuntimeError("E016-P1 checkpoint provenance 与 training receipt 不一致")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("E016-P1 held-out 要求支持 BF16 的 CUDA GPU")
    device_name = torch.cuda.get_device_name(torch.device("cuda"))
    if "RTX 4090" not in device_name:
        raise RuntimeError(f"E016-P1 held-out 要求 RTX 4090，实际为 {device_name}")
    return _EvaluationContext(
        config=config,
        source_tree_sha256=source_identity,
        registry_audit=registry,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_parameter_sha256=loaded.receipt.parameter_state_sha256,
        checkpoint_provenance_sha256=provenance_sha256,
        training_source_tree_sha256=loaded.provenance.source_tree_sha256,
        training_receipt_sha256=file_sha256(receipt_path),
        training_receipt=receipt,
        model=loaded.model,
    )


def _base_from_camera(audit: dict[str, Any]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_6d_to_matrix(audit["wrist_camera_rotation_6d_base"])
    transform[:3, 3] = audit["wrist_camera_position_base_m"]
    return transform


def _measurement_covariance(
    local_jacobian_xy_m_per_px: list[list[float]],
    sigma_xy_px: np.ndarray,
) -> np.ndarray:
    jacobian = np.asarray(local_jacobian_xy_m_per_px, dtype=np.float64)
    sigma = np.asarray(sigma_xy_px, dtype=np.float64)
    if jacobian.shape != (2, 2) or sigma.shape != (2,):
        raise ValueError("E016-P1 covariance 输入 shape 漂移")
    covariance = np.zeros((3, 3), dtype=np.float64)
    covariance[:2, :2] = jacobian @ np.diag(np.square(sigma)) @ jacobian.T
    if not np.isfinite(covariance).all():
        raise ValueError("E016-P1 measurement covariance 非有限")
    return covariance


def _ratio(numerator: int, denominator: int, *, empty: float = 0.0) -> float:
    return float(numerator / denominator) if denominator else float(empty)


def _collect_split(
    *,
    context: _EvaluationContext,
    deployable_root: Path,
    label_root: Path,
    split: str,
) -> _CollectedSplit:
    if split not in {
        context.config.fresh_held_out.validation_split,
        context.config.fresh_held_out.test_split,
    }:
        raise ValueError("E016-P1 evaluation split 必须是冻结的 val/test")
    dataset = PrecisionRGBDataset(deployable_root, label_root, split, cache_size=32)
    device = torch.device("cuda")
    model = context.model.to(device)
    model.eval()
    loader = _build_loader(
        dataset,
        batch_size=context.config.formal_training.batch_size,
        shuffle=False,
        seed=0,
        num_workers=0,
    )
    memory_config = context.config.memory_replay
    selection = context.config.validation_selection
    rows: list[dict[str, Any]] = []
    frames: list[GoalReplayFrame] = []
    invalid_geometry = 0
    dataset_index = 0
    goal_mask_intersection = 0
    goal_mask_union = 0
    projection_correct = 0
    projection_total = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                output = model(
                    batch["image"],
                    batch["structured_state"],
                    batch["geometric_motion"],
                )
            decoded = output.decode_for_control(
                temperature=context.config.loss.keypoint_temperature
            )
            predicted_uv = decoded.keypoints.normalized_uv.detach().float().cpu().numpy()
            visibility = decoded.visibility_probability.detach().float().cpu().numpy()
            projection = decoded.projection_validity_probability.detach().float().cpu().numpy()
            entropy = decoded.keypoints.normalized_entropy.detach().float().cpu().numpy()
            sigma = decoded.keypoint_sigma_px.detach().float().cpu().numpy()
            mask_probability = torch.sigmoid(output.mask_logits.detach().float()).cpu().numpy()
            predicted_goal_mask = output.mask_logits[:, 1].detach() > 0.0
            target_goal_mask = batch["mask_targets"][:, 1] > 0.5
            goal_mask_intersection += int((predicted_goal_mask & target_goal_mask).sum().item())
            goal_mask_union += int((predicted_goal_mask | target_goal_mask).sum().item())
            projection_predicted = (
                decoded.projection_validity_probability >= selection.projection_threshold
            )
            projection_correct += int(
                (projection_predicted == batch["projection_valid"]).sum().item()
            )
            projection_total += int(batch["projection_valid"].numel())
            height, width = batch["image"].shape[-2:]
            image_size_hw = (int(height), int(width))
            if predicted_uv.shape[1:] != (2, 2) or mask_probability.shape[1] != 2:
                raise RuntimeError("E016-P1 U-Net object/goal channel contract 漂移")
            for batch_index, audit in enumerate(raw_batch["audit"]):
                trajectory_id = str(audit["trajectory_id"])
                timestep = int(audit["timestep"])
                label_meta = dataset.label_by_trajectory[trajectory_id]
                labels = dataset.label_store.get(label_meta)
                goal_position = labels.goal_position_base_m[timestep].astype(np.float64)
                transform = _base_from_camera(audit)
                intrinsic = np.asarray(audit["intrinsic_wrist_cv"], dtype=np.float64)
                try:
                    gt_uv = project_base_point_to_normalized_uv(
                        goal_position,
                        intrinsic,
                        transform,
                        image_size_hw,
                    )
                    gt_projection_valid = True
                except ValueError:
                    gt_uv = None
                    gt_projection_valid = False
                observability = derive_goal_observability(
                    goal_exists=True,
                    projection_valid=gt_projection_valid,
                    projected_normalized_uv=gt_uv,
                    goal_mask=labels.goal_mask[timestep],
                    object_mask=labels.object_mask[timestep],
                    legacy_visible=bool(labels.keypoint_visible[timestep, 1]),
                    support_radius_px=context.config.observability.support_radius_px,
                )
                goal_uv = predicted_uv[batch_index, 1]
                goal_mask_probability = mask_probability_at_normalized_uv(
                    mask_probability[batch_index, 1],
                    goal_uv,
                )
                object_mask_probability = mask_probability_at_normalized_uv(
                    mask_probability[batch_index, 0],
                    goal_uv,
                )
                predicted_world: tuple[float, float, float] | None
                covariance: np.ndarray | None
                geometry: dict[str, Any]
                try:
                    geometry = geometry_conditioning(
                        normalized_uv=goal_uv,
                        intrinsic_cv=intrinsic,
                        base_from_camera_cv=transform,
                        image_size_hw=image_size_hw,
                        plane_base_z_m=float(goal_position[2]),
                    )
                    predicted_world = tuple(
                        float(value) for value in geometry["predicted_world_point_base_m"]
                    )
                    covariance = _measurement_covariance(
                        geometry["local_jacobian_xy_m_per_px"],
                        sigma[batch_index, 1],
                    )
                    geometry_valid = True
                except ValueError:
                    geometry = {
                        "abs_n_dot_unit_ray": None,
                        "jacobian_sigma_max_mm_per_px": None,
                    }
                    predicted_world = None
                    covariance = None
                    geometry_valid = False
                    invalid_geometry += 1
                evidence = GoalWriteEvidence(
                    visibility_probability=float(visibility[batch_index, 1]),
                    projection_validity_probability=float(projection[batch_index]),
                    goal_mask_probability=goal_mask_probability,
                    object_mask_probability=object_mask_probability,
                    normalized_entropy=float(entropy[batch_index, 1]),
                    radial_sigma_px=float(np.linalg.norm(sigma[batch_index, 1])),
                    geometry_valid=geometry_valid,
                    min_goal_mask_probability=memory_config.min_goal_mask_probability,
                    max_object_mask_probability=memory_config.max_object_mask_probability,
                )
                world_error = (
                    None
                    if predicted_world is None
                    else float(
                        np.linalg.norm(
                            np.asarray(predicted_world[:2], dtype=np.float64) - goal_position[:2]
                        )
                    )
                )
                oracle_safe = bool(
                    observability.observable
                    and world_error is not None
                    and world_error <= memory_config.safe_world_xy_error_m
                )
                uv_abs_error = (
                    None
                    if not observability.observable or gt_uv is None
                    else np.abs(goal_uv - gt_uv).astype(float).tolist()
                )
                pixel_error = (
                    None
                    if not observability.observable or gt_uv is None
                    else float(
                        np.linalg.norm(
                            (goal_uv - gt_uv) * np.asarray((float(width), float(height)))
                        )
                    )
                )
                row = {
                    "version": E016_P1_EVALUATION_VERSION,
                    "split": split,
                    "dataset_index": dataset_index,
                    "trajectory_id": trajectory_id,
                    "scene_id": str(audit["scene_id"]),
                    "timestep": timestep,
                    "timestamp_s": float(audit["timestamp_s"]),
                    "observability": observability.to_dict(),
                    "gt_projected_normalized_uv": (
                        None if gt_uv is None else gt_uv.astype(float).tolist()
                    ),
                    "gt_goal_position_base_m": goal_position.astype(float).tolist(),
                    "predicted_goal_normalized_uv": goal_uv.astype(float).tolist(),
                    "goal_normalized_uv_abs_error": uv_abs_error,
                    "goal_pixel_error": pixel_error,
                    "goal_visibility_probability": float(visibility[batch_index, 1]),
                    "projection_validity_probability": float(projection[batch_index]),
                    "predicted_goal_position_base_m": (
                        None if predicted_world is None else list(predicted_world)
                    ),
                    "measurement_covariance_base_m2": (
                        None if covariance is None else covariance.astype(float).tolist()
                    ),
                    "world_xy_error_m": world_error,
                    "write_evidence": evidence.to_dict(),
                    "oracle_safe_measurement": oracle_safe,
                    "geometry": {
                        "valid": geometry_valid,
                        "abs_n_dot_unit_ray": geometry["abs_n_dot_unit_ray"],
                        "jacobian_sigma_max_mm_per_px": geometry["jacobian_sigma_max_mm_per_px"],
                    },
                }
                rows.append(row)
                frames.append(
                    GoalReplayFrame(
                        episode_id=trajectory_id,
                        timestep=timestep,
                        timestamp_s=float(audit["timestamp_s"]),
                        predicted_position_base_m=predicted_world,
                        measurement_covariance_base_m2=covariance,
                        write_score=evidence.score,
                        structurally_eligible=evidence.structurally_eligible,
                        predicted_observable=evidence.observable,
                        geometry_valid=geometry_valid,
                        gt_position_base_m=goal_position,
                        gt_observable=observability.observable,
                    )
                )
                dataset_index += 1
    if dataset_index != len(dataset) or len(rows) != len(frames):
        raise RuntimeError("E016-P1 DataLoader 未完整覆盖 split")
    labels = [bool(row["observability"]["observable"]) for row in rows]
    predictions = [
        float(row["goal_visibility_probability"]) >= selection.visibility_threshold for row in rows
    ]
    true_positive = sum(predicted and target for predicted, target in zip(predictions, labels))
    false_positive = sum(predicted and not target for predicted, target in zip(predictions, labels))
    true_negative = sum(
        not predicted and not target for predicted, target in zip(predictions, labels)
    )
    false_negative = sum(not predicted and target for predicted, target in zip(predictions, labels))
    uv_errors = np.asarray(
        [
            value
            for row in rows
            if row["goal_normalized_uv_abs_error"] is not None
            for value in row["goal_normalized_uv_abs_error"]
        ],
        dtype=np.float64,
    )
    pixel_errors = np.asarray(
        [row["goal_pixel_error"] for row in rows if row["goal_pixel_error"] is not None],
        dtype=np.float64,
    )
    world_errors = np.asarray(
        [
            row["world_xy_error_m"]
            for row in rows
            if bool(row["observability"]["observable"]) and row["world_xy_error_m"] is not None
        ],
        dtype=np.float64,
    )
    if uv_errors.size == 0 or pixel_errors.size == 0 or world_errors.size == 0:
        raise RuntimeError("E016-P1 split 缺少 observable goal evaluation 样本")
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    perception = {
        "frame_count": len(rows),
        "goal_observable_count": sum(labels),
        "goal_unobservable_count": len(rows) - sum(labels),
        "goal_observable_normalized_uv_mae": float(np.mean(uv_errors)),
        "goal_observable_pixel_error_p50": float(np.quantile(pixel_errors, 0.50)),
        "goal_observable_pixel_error_p90": float(np.quantile(pixel_errors, 0.90)),
        "goal_observable_pixel_error_max": float(np.max(pixel_errors)),
        "goal_observable_world_xy_error_p50_mm": float(np.quantile(world_errors, 0.50) * 1000.0),
        "goal_observable_world_xy_error_p90_mm": float(np.quantile(world_errors, 0.90) * 1000.0),
        "goal_observable_world_xy_error_max_mm": float(np.max(world_errors) * 1000.0),
        "goal_visibility_true_positive": true_positive,
        "goal_visibility_false_positive": false_positive,
        "goal_visibility_true_negative": true_negative,
        "goal_visibility_false_negative": false_negative,
        "goal_visibility_precision": precision,
        "goal_visibility_recall": recall,
        "goal_visibility_f1": (
            float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        ),
        "goal_unobservable_false_positive_rate": _ratio(
            false_positive,
            false_positive + true_negative,
        ),
        "goal_mask_iou": _ratio(goal_mask_intersection, goal_mask_union, empty=1.0),
        "projection_accuracy_against_training_joint_target": _ratio(
            projection_correct,
            projection_total,
        ),
        "visibility_threshold": selection.visibility_threshold,
        "projection_threshold": selection.projection_threshold,
    }
    return _CollectedSplit(
        rows=rows,
        replay_frames=frames,
        invalid_geometry_count=invalid_geometry,
        perception_summary=perception,
    )


def _observability_sidecar(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "version": E016_P1_EVALUATION_VERSION,
            "split": row["split"],
            "trajectory_id": row["trajectory_id"],
            "scene_id": row["scene_id"],
            "timestep": row["timestep"],
            "timestamp_s": row["timestamp_s"],
            "goal_exists": row["observability"]["goal_exists"],
            "goal_projection_valid": row["observability"]["projection_valid"],
            "goal_in_fov": row["observability"]["in_fov"],
            "goal_observable": row["observability"]["observable"],
            "legacy_goal_visible": row["observability"]["legacy_visible"],
            "legacy_contract_mismatch": row["observability"]["legacy_contract_mismatch"],
            "local_goal_visible_fraction": row["observability"]["local_goal_visible_fraction"],
            "goal_occlusion_type": row["observability"]["occlusion_type"],
        }
        for row in rows
    ]


def _observability_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [row["observability"] for row in rows]
    occlusion = Counter(str(label["occlusion_type"]) for label in labels)
    return {
        "frame_count": len(rows),
        "goal_exists_count": sum(bool(label["goal_exists"]) for label in labels),
        "goal_projection_valid_count": sum(bool(label["projection_valid"]) for label in labels),
        "goal_in_fov_count": sum(bool(label["in_fov"]) for label in labels),
        "goal_observable_count": sum(bool(label["observable"]) for label in labels),
        "legacy_goal_visible_count": sum(bool(label["legacy_visible"]) for label in labels),
        "legacy_contract_mismatch_count": sum(
            bool(label["legacy_contract_mismatch"]) for label in labels
        ),
        "occlusion_type_counts": dict(sorted(occlusion.items())),
        "semantics": GOAL_OBSERVABILITY_SEMANTICS,
    }


def _measurement_summary(
    rows: list[dict[str, Any]],
    calibration: GoalWriteCalibration,
    *,
    catastrophic_error_m: float,
) -> dict[str, Any]:
    accepted = [
        row
        for row in rows
        if calibration.enabled
        and bool(row["write_evidence"]["structurally_eligible"])
        and float(row["write_evidence"]["score"]) >= calibration.threshold
    ]
    errors = np.asarray(
        [float(row["world_xy_error_m"]) for row in accepted if row["world_xy_error_m"] is not None],
        dtype=np.float64,
    )
    return {
        "frame_count": len(rows),
        "structurally_eligible_count": sum(
            bool(row["write_evidence"]["structurally_eligible"]) for row in rows
        ),
        "accepted_count": len(accepted),
        "accepted_unsafe_count": sum(not bool(row["oracle_safe_measurement"]) for row in accepted),
        "accepted_while_gt_unobservable_count": sum(
            not bool(row["observability"]["observable"]) for row in accepted
        ),
        "accepted_catastrophic_count": int(np.sum(errors > catastrophic_error_m)),
        "accepted_error_p50_mm": (
            None if errors.size == 0 else float(np.quantile(errors, 0.50) * 1000.0)
        ),
        "accepted_error_p90_mm": (
            None if errors.size == 0 else float(np.quantile(errors, 0.90) * 1000.0)
        ),
        "accepted_error_max_mm": (None if errors.size == 0 else float(errors.max() * 1000.0)),
    }


def _goal_memory_config(config: E016P1Config, max_age_s: float) -> GoalMemoryConfig:
    memory = config.memory_replay
    return GoalMemoryConfig(
        max_unobserved_age_s=max_age_s,
        max_innovation_m=memory.max_innovation_m,
        max_position_std_m=memory.max_position_std_m,
        require_covariance=memory.require_covariance,
        covariance_growth_m2_per_s=memory.covariance_growth_m2_per_s,
    )


def _rules_payload(
    *,
    context: _EvaluationContext,
    validation_data_identity_sha256: str,
    calibration: GoalWriteCalibration,
    max_age_s: float,
) -> dict[str, Any]:
    memory = context.config.memory_replay
    rules = {
        "version": E016_P1_RULES_VERSION,
        "status": "frozen-before-test",
        "config_sha256": context.config.sha256,
        "evaluation_source_tree_sha256": context.source_tree_sha256,
        "training_source_tree_sha256": context.training_source_tree_sha256,
        "training_receipt_sha256": context.training_receipt_sha256,
        "fresh_registry_identity_sha256": context.registry_audit["registry_identity_sha256"],
        "deployable_manifest_sha256": context.registry_audit["deployable_manifest_sha256"],
        "label_manifest_sha256": context.registry_audit["label_manifest_sha256"],
        "validation_data_identity_sha256": validation_data_identity_sha256,
        "checkpoint_sha256": context.checkpoint_sha256,
        "checkpoint_parameter_sha256": context.checkpoint_parameter_sha256,
        "checkpoint_provenance_sha256": context.checkpoint_provenance_sha256,
        "temperature": context.config.loss.keypoint_temperature,
        "write_calibration": calibration.to_dict(),
        "memory_config": {
            "max_unobserved_age_s": max_age_s,
            "max_innovation_m": memory.max_innovation_m,
            "max_position_std_m": memory.max_position_std_m,
            "require_covariance": memory.require_covariance,
            "covariance_growth_m2_per_s": memory.covariance_growth_m2_per_s,
        },
        "test_split_status": "unread",
        "test_privileged_label_file_read_count": 0,
        "test_model_forward_count": 0,
        "actuation_allowed": False,
    }
    rules["rules_sha256"] = canonical_sha256(rules)
    return rules


def _verify_rules(
    rules: dict[str, Any],
    context: _EvaluationContext,
) -> tuple[GoalWriteCalibration, GoalMemoryConfig]:
    expected_hash = rules.get("rules_sha256")
    unhashed = {key: value for key, value in rules.items() if key != "rules_sha256"}
    if expected_hash != canonical_sha256(unhashed):
        raise RuntimeError("E016-P1 frozen rules SHA-256 漂移")
    expected = {
        "version": E016_P1_RULES_VERSION,
        "status": "frozen-before-test",
        "config_sha256": context.config.sha256,
        "evaluation_source_tree_sha256": context.source_tree_sha256,
        "training_source_tree_sha256": context.training_source_tree_sha256,
        "training_receipt_sha256": context.training_receipt_sha256,
        "fresh_registry_identity_sha256": context.registry_audit["registry_identity_sha256"],
        "deployable_manifest_sha256": context.registry_audit["deployable_manifest_sha256"],
        "label_manifest_sha256": context.registry_audit["label_manifest_sha256"],
        "checkpoint_sha256": context.checkpoint_sha256,
        "checkpoint_parameter_sha256": context.checkpoint_parameter_sha256,
        "checkpoint_provenance_sha256": context.checkpoint_provenance_sha256,
        "temperature": context.config.loss.keypoint_temperature,
        "test_split_status": "unread",
        "test_privileged_label_file_read_count": 0,
        "test_model_forward_count": 0,
        "actuation_allowed": False,
    }
    for name, value in expected.items():
        if rules.get(name) != value:
            raise RuntimeError(f"E016-P1 frozen rules {name} 漂移")
    try:
        calibration = GoalWriteCalibration(**rules["write_calibration"])
        memory_config = GoalMemoryConfig(**rules["memory_config"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("E016-P1 frozen rules 无法恢复 write/memory config") from error
    if (
        calibration.accepted_unsafe_count
        > context.config.success_criteria.validation_unsafe_write_count_max
    ):
        raise RuntimeError("E016-P1 frozen write calibration 违反 validation safety gate")
    if memory_config.max_unobserved_age_s not in tuple(
        context.config.memory_replay.max_unobserved_age_candidates_s
    ):
        raise RuntimeError("E016-P1 frozen memory age 不属于预注册候选")
    if memory_config != _goal_memory_config(
        context.config,
        memory_config.max_unobserved_age_s,
    ):
        raise RuntimeError("E016-P1 frozen memory config 与预注册配置漂移")
    return calibration, memory_config


def _verify_calibration_package(
    *,
    root: Path,
    rules: dict[str, Any],
    context: _EvaluationContext,
) -> str:
    """在 claim 前完整绑定 validation-only calibration 产物。"""

    receipt_path = root / "receipt.json"
    receipt = _read_json(receipt_path, "E016-P1 calibration receipt")
    if (
        receipt.get("version") != E016_P1_EVALUATION_VERSION
        or receipt.get("phase") != "calibrate"
        or receipt.get("status") != "complete"
        or receipt.get("rules_sha256") != rules["rules_sha256"]
        or receipt.get("test_split_status") != "unread"
    ):
        raise RuntimeError("E016-P1 calibration receipt 状态或 identity 漂移")
    files = receipt.get("files")
    if not isinstance(files, dict) or set(files) != set(_CALIBRATION_RECEIPT_FILES):
        raise RuntimeError("E016-P1 calibration receipt 文件集合漂移")
    for name in _CALIBRATION_RECEIPT_FILES:
        expected = files[name]
        if not isinstance(expected, str) or file_sha256(root / name) != expected:
            raise RuntimeError(f"E016-P1 calibration artifact SHA-256 漂移: {name}")

    summary = _read_json(root / "summary.json", "E016-P1 calibration summary")
    if (
        summary.get("version") != E016_P1_EVALUATION_VERSION
        or summary.get("phase") != "calibrate"
        or summary.get("status") != "complete"
        or summary.get("split") != context.config.fresh_held_out.validation_split
        or summary.get("trajectory_count") != context.config.fresh_held_out.validation_trajectories
        or summary.get("registry_identity_sha256")
        != context.registry_audit["registry_identity_sha256"]
        or summary.get("validation_data_identity_sha256")
        != rules["validation_data_identity_sha256"]
        or summary.get("checkpoint_sha256") != context.checkpoint_sha256
        or summary.get("config_sha256") != context.config.sha256
        or summary.get("source_tree_sha256") != context.source_tree_sha256
        or summary.get("write_calibration") != rules["write_calibration"]
        or summary.get("selected_max_unobserved_age_s")
        != rules["memory_config"]["max_unobserved_age_s"]
        or summary.get("test_split_status") != "unread"
        or summary.get("test_privileged_label_file_read_count") != 0
        or summary.get("test_model_forward_count") != 0
        or summary.get("actuation_allowed") is not False
    ):
        raise RuntimeError("E016-P1 calibration summary 与 frozen rules/context 漂移")
    if (
        _read_json(
            root / "fresh_registry_audit.json",
            "E016-P1 fresh registry audit",
        )
        != context.registry_audit
    ):
        raise RuntimeError("E016-P1 calibration registry audit 漂移")
    config_snapshot = _read_json(
        root / "config_snapshot.json",
        "E016-P1 calibration config snapshot",
    )
    if canonical_sha256(config_snapshot) != canonical_sha256(context.config.to_dict()):
        raise RuntimeError("E016-P1 calibration config snapshot 漂移")
    run_state = _read_json(root / "run_state.json", "E016-P1 calibration run state")
    if (
        run_state.get("version") != E016_P1_EVALUATION_VERSION
        or run_state.get("phase") != "calibrate"
        or run_state.get("status") != "complete"
        or run_state.get("test_split_status") != "unread"
    ):
        raise RuntimeError("E016-P1 calibration run state 未完成或 test 已读取")
    return file_sha256(receipt_path)


def _paths_overlap(first: Path, second: Path) -> bool:
    left = first.resolve()
    right = second.resolve()
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _preflight_test_outputs(
    *,
    private: Path,
    public: Path,
    protected_roots: tuple[Path, ...],
) -> None:
    """在 claim 前拒绝覆盖/嵌套，并验证输出父目录确实可写。"""

    if _paths_overlap(private, public):
        raise ValueError("E016-P1 private/public output 不能相同或互相嵌套")
    for target, name in ((private, "private"), (public, "public")):
        if os.path.lexists(target):
            raise FileExistsError(f"E016-P1 test {name} output 已存在: {target}")
        for protected in protected_roots:
            if _paths_overlap(target, protected):
                raise ValueError(f"E016-P1 test {name} output 与只读输入目录重叠")
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".e016-p1-output-preflight.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(b"e016-p1-output-preflight\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def claim_e016_p1_test_once(
    path: str | Path,
    *,
    rules_sha256: str,
    registry_identity_sha256: str,
    checkpoint_sha256: str,
    source_tree_sha256: str,
    calibration_receipt_sha256: str,
) -> str:
    """在任何 test privileged-label read/model forward 前创建不可复用 claim。"""

    target = Path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError(f"E016-P1 test claim parent 不存在: {target.parent}")
    payload = {
        "version": E016_P1_TEST_CLAIM_VERSION,
        "status": "claimed-before-test-label-or-model-read",
        "rules_sha256": rules_sha256,
        "registry_identity_sha256": registry_identity_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "source_tree_sha256": source_tree_sha256,
        "calibration_receipt_sha256": calibration_receipt_sha256,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    except FileExistsError as error:
        raise RuntimeError("E016-P1 fresh test 已 claim，禁止重复读取或 model forward") from error
    # 写入异常也不删除 claim，避免失败后无痕重试 test。
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(target)


def _receipt_files(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: file_sha256(root / name) for name in names}


def run_e016_p1_calibration(
    *,
    deployable_root: str | Path,
    label_root: str | Path,
    config_path: str | Path,
    training_output: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """只在 fresh validation 上冻结 write threshold 与 memory max age。"""

    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"E016-P1 calibration output 已存在: {output}")
    deployable = Path(deployable_root)
    labels = Path(label_root)
    context = _load_context(
        deployable_root=deployable,
        label_root=labels,
        config_path=Path(config_path),
        training_output=Path(training_output),
        repository_root=Path(repository_root),
    )
    validation_identity = _validate_non_test_split_files(
        deployable_root=deployable,
        label_root=labels,
        split=context.config.fresh_held_out.validation_split,
    )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _atomic_json(
        output / "run_state.json",
        {
            "version": E016_P1_EVALUATION_VERSION,
            "phase": "calibrate",
            "status": "in-progress",
            "test_split_status": "unread",
        },
    )
    collected = _collect_split(
        context=context,
        deployable_root=deployable,
        label_root=labels,
        split=context.config.fresh_held_out.validation_split,
    )
    calibration = calibrate_goal_write_threshold(
        scores=[float(row["write_evidence"]["score"]) for row in collected.rows],
        structurally_eligible=np.asarray(
            [bool(row["write_evidence"]["structurally_eligible"]) for row in collected.rows],
            dtype=np.bool_,
        ),
        oracle_safe=np.asarray(
            [bool(row["oracle_safe_measurement"]) for row in collected.rows],
            dtype=np.bool_,
        ),
    )
    memory = context.config.memory_replay
    max_age, age_reports = select_memory_max_age(
        collected.replay_frames,
        calibration=calibration,
        max_age_candidates_s=memory.max_unobserved_age_candidates_s,
        max_innovation_m=memory.max_innovation_m,
        max_position_std_m=memory.max_position_std_m,
        require_covariance=memory.require_covariance,
        covariance_growth_m2_per_s=memory.covariance_growth_m2_per_s,
        catastrophic_world_xy_error_m=memory.catastrophic_world_xy_error_m,
    )
    replay = replay_goal_memory(
        collected.replay_frames,
        calibration=calibration,
        memory_config=_goal_memory_config(context.config, max_age),
    )
    replay_summary = summarize_goal_memory_replay(
        replay,
        catastrophic_world_xy_error_m=memory.catastrophic_world_xy_error_m,
    )
    rules = _rules_payload(
        context=context,
        validation_data_identity_sha256=validation_identity,
        calibration=calibration,
        max_age_s=max_age,
    )
    measurement = _measurement_summary(
        collected.rows,
        calibration,
        catastrophic_error_m=memory.catastrophic_world_xy_error_m,
    )
    if measurement["accepted_unsafe_count"] != 0:
        raise RuntimeError("E016-P1 validation calibration 接受了 unsafe write")
    summary = {
        "version": E016_P1_EVALUATION_VERSION,
        "phase": "calibrate",
        "status": "complete",
        "split": context.config.fresh_held_out.validation_split,
        "trajectory_count": context.config.fresh_held_out.validation_trajectories,
        "registry_identity_sha256": context.registry_audit["registry_identity_sha256"],
        "validation_data_identity_sha256": validation_identity,
        "checkpoint_sha256": context.checkpoint_sha256,
        "config_sha256": context.config.sha256,
        "source_tree_sha256": context.source_tree_sha256,
        "perception": collected.perception_summary,
        "observability_audit": _observability_summary(collected.rows),
        "write_calibration": calibration.to_dict(),
        "write_measurements": measurement,
        "selected_max_unobserved_age_s": max_age,
        "memory_age_candidates": age_reports,
        "memory_replay": replay_summary,
        "invalid_geometry_count": collected.invalid_geometry_count,
        "test_split_status": "unread",
        "test_privileged_label_file_read_count": 0,
        "test_model_forward_count": 0,
        "actuation_allowed": False,
    }
    _atomic_jsonl(
        output / "observability_sidecar.jsonl",
        _observability_sidecar(collected.rows),
    )
    _atomic_jsonl(output / "prediction_rows.jsonl", collected.rows)
    _atomic_jsonl(
        output / "memory_replay.jsonl",
        [record.to_dict() for record in replay],
    )
    _atomic_json(output / "fresh_registry_audit.json", context.registry_audit)
    _atomic_json(output / "frozen_rules.json", rules)
    _atomic_json(output / "summary.json", summary)
    _atomic_json(output / "config_snapshot.json", context.config.to_dict())
    receipt = {
        "version": E016_P1_EVALUATION_VERSION,
        "phase": "calibrate",
        "status": "complete",
        "rules_sha256": rules["rules_sha256"],
        "files": _receipt_files(output, _CALIBRATION_RECEIPT_FILES),
        "test_split_status": "unread",
    }
    _atomic_json(output / "receipt.json", receipt)
    _atomic_json(
        output / "run_state.json",
        {
            "version": E016_P1_EVALUATION_VERSION,
            "phase": "calibrate",
            "status": "complete",
            "test_split_status": "unread",
        },
    )
    return summary


def _public_readme(summary: dict[str, Any]) -> str:
    formal = summary["formal_training"]
    perception = summary["fresh_test_perception"]
    measurement = summary["write_measurements"]
    memory = summary["memory_replay"]
    return f"""# E016-P1 — Corrected-observability formal Precision + goal memory

E016-P1 从随机初始化完成 20-epoch corrected-observability Precision U-Net 训练，并在规则冻结后对
100 条 fresh test trajectory 执行一次 no-actuation perception 与显式 base-frame goal-memory replay。

## Formal training

- selected epoch：`{formal["selected_epoch"]}`
- validation observable-goal normalized-UV MAE：`{formal["selected_metric"]:.6f}`
- validation visibility precision / recall：`{formal["selected_validation"]["goal_visibility_precision"]:.6f}` /
  `{formal["selected_validation"]["goal_visibility_recall"]:.6f}`
- validation unobservable FPR：`{formal["selected_validation"]["goal_unobservable_false_positive_rate"]:.6f}`
- Motion Head unchanged：`{formal["motion_head_unchanged"]}`

## Fresh test-once

- observable goal pixel p50 / p90 / max：`{perception["goal_observable_pixel_error_p50"]:.3f}` /
  `{perception["goal_observable_pixel_error_p90"]:.3f}` / `{perception["goal_observable_pixel_error_max"]:.3f}` px
- visibility precision / recall：`{perception["goal_visibility_precision"]:.6f}` /
  `{perception["goal_visibility_recall"]:.6f}`
- write accepted / unsafe：`{measurement["accepted_count"]}` / `{measurement["accepted_unsafe_count"]}`
- current / memory coverage：`{memory["current_measurement_coverage"]:.6f}` /
  `{memory["memory_coverage"]:.6f}`
- memory valid while GT unobservable：`{memory["memory_valid_while_gt_unobservable_count"]}`
- memory catastrophic / reset leakage：`{memory["memory_catastrophic_count"]}` /
  `{memory["episode_reset_leakage_count"]}`

Engineering gate passed=`{summary["engineering_gate_passed"]}`。本实验始终 no-actuation；即使门禁通过，
`safe_for_actuator_promotion` 仍为 `false`，后续还需要独立 controller/shadow safety 验证。
"""


def _write_public(
    *,
    root: Path,
    repository_root: Path,
    summary: dict[str, Any],
    private_receipt_sha256: str,
) -> dict[str, Any]:
    root.mkdir(mode=0o755, parents=True, exist_ok=False)
    assert_public_payload_safe(summary)
    _atomic_json(root / "sanitized_summary.json", summary)
    readme = _public_readme(summary)
    if any(token in readme for token in ("/home/", "/mnt/", "C:\\", "D:\\")):
        raise RuntimeError("E016-P1 public README 含敏感绝对路径")
    _atomic_text(root / "README.md", readme)
    verifier = repository_root / "scripts" / "verify_e016_p1_public_results.py"
    if not verifier.is_file():
        raise FileNotFoundError("E016-P1 public verifier template 不存在")
    shutil.copyfile(verifier, root / "verify_summary.py")
    files = _receipt_files(
        root,
        ("README.md", "sanitized_summary.json", "verify_summary.py"),
    )
    receipt = {
        "version": E016_P1_PUBLIC_VERSION,
        "status": "complete",
        "private_receipt_sha256": private_receipt_sha256,
        "rules_sha256": summary["rules_sha256"],
        "files": files,
        "contains_raw_rgb": False,
        "contains_raw_heatmaps": False,
        "contains_trajectory_identity": False,
        "contains_model_weights": False,
        "contains_sensitive_paths": False,
    }
    assert_public_payload_safe(receipt)
    _atomic_json(root / "receipt.json", receipt)
    subprocess.run(
        [sys.executable, str(root / "verify_summary.py"), str(root)],
        check=True,
        cwd=repository_root,
    )
    return receipt


def run_e016_p1_test_once(
    *,
    deployable_root: str | Path,
    label_root: str | Path,
    config_path: str | Path,
    training_output: str | Path,
    repository_root: str | Path,
    calibration_root: str | Path,
    private_output_root: str | Path,
    public_output_root: str | Path,
) -> dict[str, Any]:
    """先 claim，再完整审计并且只执行一次 fresh test model forward/replay。"""

    private = Path(private_output_root)
    public = Path(public_output_root)
    deployable = Path(deployable_root)
    labels = Path(label_root)
    calibration_path = Path(calibration_root)
    repository = Path(repository_root)
    training = Path(training_output)
    context = _load_context(
        deployable_root=deployable,
        label_root=labels,
        config_path=Path(config_path),
        training_output=training,
        repository_root=repository,
    )
    rules = _read_json(calibration_path / "frozen_rules.json", "E016-P1 frozen rules")
    calibration, memory_config = _verify_rules(rules, context)
    calibration_receipt_sha256 = _verify_calibration_package(
        root=calibration_path,
        rules=rules,
        context=context,
    )
    verifier = repository / "scripts" / "verify_e016_p1_public_results.py"
    if not verifier.is_file():
        raise FileNotFoundError("E016-P1 public verifier template 不存在")
    _preflight_test_outputs(
        private=private,
        public=public,
        protected_roots=(
            deployable,
            labels,
            calibration_path,
            training,
            repository,
        ),
    )
    claim_sha256 = claim_e016_p1_test_once(
        calibration_path / "test_evaluation_claim.json",
        rules_sha256=str(rules["rules_sha256"]),
        registry_identity_sha256=context.registry_audit["registry_identity_sha256"],
        checkpoint_sha256=context.checkpoint_sha256,
        source_tree_sha256=context.source_tree_sha256,
        calibration_receipt_sha256=calibration_receipt_sha256,
    )
    private.mkdir(mode=0o700, parents=True, exist_ok=False)
    _atomic_json(
        private / "run_state.json",
        {
            "version": E016_P1_EVALUATION_VERSION,
            "phase": "test-once",
            "status": "in-progress",
            "test_split_status": "consuming-once",
        },
    )
    # 这是本流程第一次打开 test privileged labels；claim 已经持久化且不会回滚。
    full_audit = audit_precision_dataset(
        deployable,
        labels,
        RobotSpec(),
        write_artifact=False,
    )
    expected_counts = context.registry_audit["split_trajectory_counts"]
    if (
        not full_audit.passed
        or full_audit.split_trajectory_counts != expected_counts
        or full_audit.deployable_manifest_sha256
        != context.registry_audit["deployable_manifest_sha256"]
        or full_audit.label_manifest_sha256 != context.registry_audit["label_manifest_sha256"]
    ):
        raise RuntimeError("E016-P1 fresh Precision Dataset 完整 audit 未通过")
    collected = _collect_split(
        context=context,
        deployable_root=deployable,
        label_root=labels,
        split=context.config.fresh_held_out.test_split,
    )
    replay = replay_goal_memory(
        collected.replay_frames,
        calibration=calibration,
        memory_config=memory_config,
    )
    catastrophic = context.config.memory_replay.catastrophic_world_xy_error_m
    measurement = _measurement_summary(
        collected.rows,
        calibration,
        catastrophic_error_m=catastrophic,
    )
    memory_summary = summarize_goal_memory_replay(
        replay,
        catastrophic_world_xy_error_m=catastrophic,
    )
    coverage_improved = int(memory_summary["memory_valid_while_gt_unobservable_count"]) > int(
        memory_summary["current_valid_while_gt_unobservable_count"]
    )
    criteria = context.config.success_criteria
    engineering_gate = bool(
        int(measurement["accepted_unsafe_count"]) <= criteria.test_unsafe_write_count_max
        and int(memory_summary["memory_catastrophic_count"])
        <= criteria.test_memory_catastrophic_count_max
        and int(memory_summary["episode_reset_leakage_count"])
        <= criteria.episode_reset_leakage_count_max
        and coverage_improved
    )
    observability = _observability_summary(collected.rows)
    private_summary = {
        "version": E016_P1_EVALUATION_VERSION,
        "phase": "test-once",
        "status": "complete",
        "split": context.config.fresh_held_out.test_split,
        "trajectory_count": context.config.fresh_held_out.test_trajectories,
        "full_dataset_audit": full_audit.to_dict(),
        "registry_identity_sha256": context.registry_audit["registry_identity_sha256"],
        "checkpoint_sha256": context.checkpoint_sha256,
        "checkpoint_parameter_sha256": context.checkpoint_parameter_sha256,
        "checkpoint_provenance_sha256": context.checkpoint_provenance_sha256,
        "config_sha256": context.config.sha256,
        "source_tree_sha256": context.source_tree_sha256,
        "rules_sha256": rules["rules_sha256"],
        "calibration_receipt_sha256": calibration_receipt_sha256,
        "test_once_claim_sha256": claim_sha256,
        "perception": collected.perception_summary,
        "observability_audit": observability,
        "write_measurements": measurement,
        "memory_replay": memory_summary,
        "coverage_improved_while_unobservable": coverage_improved,
        "engineering_gate_passed": engineering_gate,
        "invalid_geometry_count": collected.invalid_geometry_count,
        "test_split_status": "consumed-once",
        "test_privileged_label_evaluation_count": 1,
        "test_model_forward_evaluation_count": 1,
        "actuation_allowed": False,
        "safe_for_actuator_promotion": False,
    }
    _atomic_jsonl(
        private / "observability_sidecar.jsonl",
        _observability_sidecar(collected.rows),
    )
    _atomic_jsonl(private / "prediction_rows.jsonl", collected.rows)
    _atomic_jsonl(
        private / "memory_replay.jsonl",
        [record.to_dict() for record in replay],
    )
    _atomic_json(private / "summary_private.json", private_summary)
    _atomic_json(private / "frozen_rules.json", rules)
    private_files = _receipt_files(
        private,
        (
            "observability_sidecar.jsonl",
            "prediction_rows.jsonl",
            "memory_replay.jsonl",
            "summary_private.json",
            "frozen_rules.json",
        ),
    )
    private_receipt = {
        "version": E016_P1_EVALUATION_VERSION,
        "phase": "test-once",
        "status": "complete",
        "rules_sha256": rules["rules_sha256"],
        "files": private_files,
        "test_split_status": "consumed-once",
    }
    _atomic_json(private / "receipt.json", private_receipt)
    public_summary = {
        "version": E016_P1_PUBLIC_VERSION,
        "status": "complete",
        "experiment": "E016-P1 corrected-observability formal Precision and goal memory",
        "formal_training": context.training_receipt,
        "fresh_dataset": {
            "seed_range_start": context.config.fresh_held_out.start_seed,
            "candidate_count": context.config.fresh_held_out.max_candidates,
            "validation_trajectory_count": context.config.fresh_held_out.validation_trajectories,
            "test_trajectory_count": context.config.fresh_held_out.test_trajectories,
            "registry_identity_sha256": context.registry_audit["registry_identity_sha256"],
            "full_dataset_identity_sha256": full_audit.dataset_identity_sha256,
        },
        "frozen_conditions": {
            "checkpoint_sha256": context.checkpoint_sha256,
            "checkpoint_changed_after_training": False,
            "training_config_sha256": context.config.sha256,
            "rules_sha256": rules["rules_sha256"],
            "temperature": context.config.loss.keypoint_temperature,
            "write_threshold_selected_on": "fresh-validation-only",
            "memory_age_selected_on": "fresh-validation-only",
            "test_evaluated_once": True,
            "test_once_claim_sha256": claim_sha256,
            "calibration_receipt_sha256": calibration_receipt_sha256,
            "actuation_allowed": False,
        },
        "write_calibration": calibration.to_dict(),
        "memory_config": rules["memory_config"],
        "fresh_test_perception": collected.perception_summary,
        "observability_audit": observability,
        "write_measurements": measurement,
        "memory_replay": memory_summary,
        "coverage_improved_while_unobservable": coverage_improved,
        "engineering_gate_passed": engineering_gate,
        "rules_sha256": rules["rules_sha256"],
        "safe_for_actuator_promotion": False,
        "test_split_status": "consumed-once",
    }
    public_receipt = _write_public(
        root=public,
        repository_root=Path(repository_root),
        summary=public_summary,
        private_receipt_sha256=file_sha256(private / "receipt.json"),
    )
    _atomic_json(
        private / "run_state.json",
        {
            "version": E016_P1_EVALUATION_VERSION,
            "phase": "test-once",
            "status": "complete",
            "test_split_status": "consumed-once",
            "public_receipt_sha256": canonical_sha256(public_receipt),
        },
    )
    return public_summary


__all__ = [
    "E016_P1_EVALUATION_VERSION",
    "E016_P1_PUBLIC_VERSION",
    "E016_P1_RULES_VERSION",
    "E016_P1_TEST_CLAIM_VERSION",
    "audit_e016_fresh_registry",
    "claim_e016_p1_test_once",
    "run_e016_p1_calibration",
    "run_e016_p1_test_once",
]
