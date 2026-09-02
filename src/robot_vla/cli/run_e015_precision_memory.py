"""运行 E015-A observability audit 与 E015-B frozen-model goal-memory replay。"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_vla.contracts import RobotSpec
from robot_vla.data.trajectory import load_manifest
from robot_vla.observation import rotation_6d_to_matrix
from robot_vla.precision.checkpoint import PrecisionCheckpointRole, load_precision_checkpoint
from robot_vla.precision.data import (
    PrecisionRGBDataset,
    audit_precision_dataset,
    canonical_sha256,
    file_sha256,
)
from robot_vla.precision.geometry import project_base_point_to_normalized_uv
from robot_vla.precision.memory_evaluation import (
    MEMORY_AGE_POLICY,
    WRITE_THRESHOLD_POLICY,
    GoalReplayFrame,
    GoalWriteCalibration,
    calibrate_goal_write_threshold,
    replay_goal_memory,
    select_memory_max_age,
    summarize_goal_memory_replay,
)
from robot_vla.precision.observability import (
    GOAL_OBSERVABILITY_SEMANTICS,
    GOAL_WRITE_SCORE_SEMANTICS,
    GoalWriteEvidence,
    derive_goal_observability,
    mask_probability_at_normalized_uv,
)
from robot_vla.precision.outliers import (
    assert_public_payload_safe,
    geometry_conditioning,
)
from robot_vla.precision.state_memory import (
    GOAL_MEMORY_UPDATE_POLICY,
    GOAL_POSITION_FRAME_SEMANTICS,
    GoalMemoryConfig,
)
from robot_vla.precision.training import (
    _build_loader,
    _to_device,
    load_precision_experiment_config,
    source_tree_sha256,
)

E015_EXPERIMENT_VERSION = "e015-precision-goal-memory/v1"
E015_FROZEN_RULES_VERSION = "e015-precision-goal-memory-rules/v1"
E015_PUBLIC_VERSION = "e015-precision-goal-memory-public/v1"
E015_TEST_ONCE_CLAIM_VERSION = "e015-test-once-claim/v1"


@dataclass(frozen=True)
class _Context:
    config: dict[str, Any]
    config_sha256: str
    source_tree_sha256: str
    dataset_identity_sha256: str
    dataset_audit: dict[str, Any]
    seed_identity_sha256: str
    checkpoint_sha256: str
    checkpoint_parameter_sha256: str
    checkpoint_provenance_sha256: str
    upstream_training_source_tree_sha256: str
    upstream_held_out_receipt_sha256: str
    temperature: float
    model: Any


@dataclass(frozen=True)
class _CollectedSplit:
    rows: list[dict[str, Any]]
    replay_frames: list[GoalReplayFrame]
    invalid_geometry_count: int


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} 不存在: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是 JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _claim_test_evaluation_once(
    path: Path,
    *,
    rules_sha256: str,
    dataset_identity_sha256: str,
    seed_identity_sha256: str,
    source_tree_sha256: str,
) -> str:
    """在读取 test split 前原子创建不可复用的消费凭证。"""

    if not path.parent.is_dir():
        raise FileNotFoundError(f"E015 test-once claim parent 不存在: {path.parent}")
    payload = {
        "version": E015_TEST_ONCE_CLAIM_VERSION,
        "status": "claimed-before-test-read",
        "rules_sha256": rules_sha256,
        "dataset_identity_sha256": dataset_identity_sha256,
        "seed_identity_sha256": seed_identity_sha256,
        "source_tree_sha256": source_tree_sha256,
    }
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o400)
    except FileExistsError as error:
        raise RuntimeError(
            "E015 fresh test 已被 claim；禁止通过更换输出目录重复评估"
        ) from error
    # 写入异常时也不删除 claim，避免失败后无痕重读 test。
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(path)


def _require_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{name} 字段漂移: missing={missing}, extra={extra}")


def _load_e015_config(path: Path) -> dict[str, Any]:
    config = _read_json(path, "E015 config")
    _require_keys(
        config,
        {
            "version",
            "upstream",
            "fresh_dataset",
            "observability",
            "write_gate",
            "memory",
            "execution",
            "success_criteria",
        },
        "E015 config",
    )
    if config["version"] != E015_EXPERIMENT_VERSION:
        raise ValueError("E015 config version 漂移")
    upstream = config["upstream"]
    _require_keys(
        upstream,
        {"checkpoint_sha256", "temperature", "training_performed"},
        "E015 upstream",
    )
    if upstream["training_performed"] is not False:
        raise ValueError("E015 禁止训练或修改 checkpoint")
    fresh = config["fresh_dataset"]
    _require_keys(
        fresh,
        {
            "start_seed",
            "max_candidates",
            "train_trajectories",
            "calibration_trajectories",
            "evaluation_trajectories",
            "calibration_split",
            "evaluation_split",
        },
        "E015 fresh_dataset",
    )
    if (
        fresh["start_seed"] != 133000
        or fresh["max_candidates"] != 1000
        or fresh["calibration_split"] != "val"
        or fresh["evaluation_split"] != "test"
    ):
        raise ValueError("E015 fresh seed/split 预注册条件漂移")
    for name in (
        "train_trajectories",
        "calibration_trajectories",
        "evaluation_trajectories",
    ):
        if not isinstance(fresh[name], int) or isinstance(fresh[name], bool) or fresh[name] <= 0:
            raise ValueError(f"fresh_dataset.{name} 必须是正整数")
    observability = config["observability"]
    if (
        observability.get("corrected_observable_semantics")
        != GOAL_OBSERVABILITY_SEMANTICS
    ):
        raise ValueError("E015 observability semantics 漂移")
    if observability.get("support_radius_px") != 2:
        raise ValueError("E015 support_radius_px 预注册为 2")
    write_gate = config["write_gate"]
    if write_gate.get("score_semantics") != GOAL_WRITE_SCORE_SEMANTICS:
        raise ValueError("E015 write score semantics 漂移")
    if write_gate.get("threshold_policy") != WRITE_THRESHOLD_POLICY:
        raise ValueError("E015 write threshold policy 漂移")
    memory = config["memory"]
    if memory.get("frame_semantics") != GOAL_POSITION_FRAME_SEMANTICS:
        raise ValueError("E015 memory 必须使用 robot-base frame")
    if memory.get("update_policy") != GOAL_MEMORY_UPDATE_POLICY:
        raise ValueError("E015 memory update policy 漂移")
    if memory.get("age_policy") != MEMORY_AGE_POLICY:
        raise ValueError("E015 memory age policy 漂移")
    execution = config["execution"]
    if (
        execution
        != {
            "device": "cuda",
            "use_bf16": True,
            "actuation_allowed": False,
            "mode": "shadow-replay",
        }
    ):
        raise ValueError("E015 execution 必须保持 CUDA/BF16/no-actuation shadow")
    criteria = config["success_criteria"]
    if (
        criteria.get("validation_unsafe_write_count_max") != 0
        or criteria.get("test_unsafe_write_count_max") != 0
        or criteria.get("test_memory_catastrophic_count_max") != 0
        or criteria.get("episode_reset_leakage_count_max") != 0
        or criteria.get("require_memory_unobservable_coverage_improvement") is not True
        or criteria.get("actuator_promotion_allowed") is not False
    ):
        raise ValueError("E015 success criteria 不允许放宽")
    return config


def _base_from_camera(audit: dict[str, Any]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_6d_to_matrix(
        audit["wrist_camera_rotation_6d_base"]
    )
    transform[:3, 3] = audit["wrist_camera_position_base_m"]
    return transform


def _validate_fresh_dataset(
    *,
    deployable_root: Path,
    label_root: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    audit = audit_precision_dataset(
        deployable_root,
        label_root,
        RobotSpec(),
        write_artifact=False,
    )
    if not audit.passed:
        raise RuntimeError("E015 fresh Dataset audit 未通过")
    fresh = config["fresh_dataset"]
    expected = {
        "train": int(fresh["train_trajectories"]),
        "val": int(fresh["calibration_trajectories"]),
        "test": int(fresh["evaluation_trajectories"]),
    }
    if audit.split_trajectory_counts != expected:
        raise RuntimeError(
            "E015 fresh Dataset split 数量漂移: "
            f"actual={audit.split_trajectory_counts}, expected={expected}"
        )
    entries = load_manifest(deployable_root)
    seeds = [int(entry.randomization["seed"]) for entry in entries]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("E015 fresh Dataset seed 重复")
    start = int(fresh["start_seed"])
    stop = start + int(fresh["max_candidates"])
    if any(not start <= seed < stop for seed in seeds):
        raise RuntimeError("E015 Dataset 使用了预注册范围外 seed")
    split_seed_payload = [
        {
            "seed": int(entry.randomization["seed"]),
            "split": entry.split,
            "scene_id": entry.scene_id,
        }
        for entry in entries
    ]
    return audit.to_dict(), canonical_sha256(split_seed_payload)


def _context(args: argparse.Namespace) -> _Context:
    config = _load_e015_config(args.e015_config)
    source_identity = source_tree_sha256(args.repository_root)
    dataset_audit, seed_identity = _validate_fresh_dataset(
        deployable_root=args.deployable_root,
        label_root=args.label_root,
        config=config,
    )
    e013_config = load_precision_experiment_config(args.e013_config)
    training_root = args.training_output
    training_receipt = _read_json(
        training_root / "checkpoint_receipt.json",
        "E013 training receipt",
    )
    checkpoint_receipt = training_receipt["checkpoint"]
    checkpoint_sha256 = str(checkpoint_receipt["checkpoint_sha256"])
    provenance_sha256 = str(checkpoint_receipt["provenance_sha256"])
    if checkpoint_sha256 != str(config["upstream"]["checkpoint_sha256"]):
        raise RuntimeError("E015 upstream checkpoint SHA-256 漂移")
    if str(training_receipt["training_config_sha256"]) != e013_config.sha256:
        raise RuntimeError("E013 training config identity 漂移")
    loaded = load_precision_checkpoint(
        training_root / "precision-formal.pt",
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_provenance_sha256=provenance_sha256,
        expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
    )
    held_out_receipt_path = args.held_out_output / "receipt.json"
    _read_json(held_out_receipt_path, "E013 held-out receipt")
    calibration = _read_json(
        args.held_out_output / "confidence_calibration.json",
        "E013 confidence calibration",
    )
    if str(calibration["checkpoint_sha256"]) != checkpoint_sha256:
        raise RuntimeError("E013 held-out calibration checkpoint 漂移")
    temperature = float(calibration["temperature"])
    if not math.isclose(
        temperature,
        float(config["upstream"]["temperature"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("E015 frozen temperature 漂移")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("E015 frozen-model replay 要求支持 BF16 的 CUDA GPU")
    return _Context(
        config=config,
        config_sha256=canonical_sha256(config),
        source_tree_sha256=source_identity,
        dataset_identity_sha256=str(dataset_audit["dataset_identity_sha256"]),
        dataset_audit=dataset_audit,
        seed_identity_sha256=seed_identity,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_parameter_sha256=loaded.receipt.parameter_state_sha256,
        checkpoint_provenance_sha256=provenance_sha256,
        upstream_training_source_tree_sha256=loaded.provenance.source_tree_sha256,
        upstream_held_out_receipt_sha256=file_sha256(held_out_receipt_path),
        temperature=temperature,
        model=loaded.model,
    )


def _measurement_covariance(
    local_jacobian_xy_m_per_px: list[list[float]],
    sigma_xy_px: np.ndarray,
) -> np.ndarray:
    jacobian = np.asarray(local_jacobian_xy_m_per_px, dtype=np.float64)
    sigma = np.asarray(sigma_xy_px, dtype=np.float64)
    if jacobian.shape != (2, 2) or sigma.shape != (2,):
        raise ValueError("E015 covariance 输入 shape 漂移")
    covariance = np.zeros((3, 3), dtype=np.float64)
    covariance[:2, :2] = jacobian @ np.diag(np.square(sigma)) @ jacobian.T
    if not np.isfinite(covariance).all():
        raise ValueError("E015 measurement covariance 非有限")
    return covariance


def _collect_split(
    *,
    context: _Context,
    deployable_root: Path,
    label_root: Path,
    split: str,
) -> _CollectedSplit:
    dataset = PrecisionRGBDataset(
        deployable_root,
        label_root,
        split,
        cache_size=32,
    )
    device = torch.device("cuda")
    model = context.model.to(device)
    model.eval()
    e013_config = context.config
    batch_size = 32
    loader = _build_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=0,
    )
    support_radius = int(e013_config["observability"]["support_radius_px"])
    min_goal_mask = float(e013_config["write_gate"]["min_goal_mask_probability"])
    max_object_mask = float(e013_config["write_gate"]["max_object_mask_probability"])
    safe_error = float(e013_config["write_gate"]["safe_world_xy_error_m"])
    rows: list[dict[str, Any]] = []
    frames: list[GoalReplayFrame] = []
    invalid_geometry = 0
    dataset_index = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                output = model(
                    batch["image"],
                    batch["structured_state"],
                    batch["geometric_motion"],
                )
            decoded = output.decode_for_control(temperature=context.temperature)
            predicted_uv = decoded.keypoints.normalized_uv.detach().float().cpu().numpy()
            visibility = decoded.visibility_probability.detach().float().cpu().numpy()
            projection_probability = (
                decoded.projection_validity_probability.detach().float().cpu().numpy()
            )
            entropy = decoded.keypoints.normalized_entropy.detach().float().cpu().numpy()
            sigma = decoded.keypoint_sigma_px.detach().float().cpu().numpy()
            mask_probability = torch.sigmoid(output.mask_logits.detach().float()).cpu().numpy()
            height, width = batch["image"].shape[-2:]
            image_size_hw = (int(height), int(width))
            if predicted_uv.shape[1:] != (2, 2) or mask_probability.shape[1] != 2:
                raise RuntimeError("E015 frozen U-Net object/goal channel contract 漂移")
            for batch_index, audit in enumerate(raw_batch["audit"]):
                trajectory_id = str(audit["trajectory_id"])
                timestep = int(audit["timestep"])
                label_meta = dataset.label_by_trajectory[trajectory_id]
                labels = dataset.label_store.get(label_meta)
                goal_position = labels.goal_position_base_m[timestep].astype(np.float64)
                transform = _base_from_camera(audit)
                intrinsic = np.asarray(audit["intrinsic_wrist_cv"], dtype=np.float64)
                try:
                    gt_projected_uv = project_base_point_to_normalized_uv(
                        goal_position,
                        intrinsic,
                        transform,
                        image_size_hw,
                    )
                    gt_projection_valid = True
                except ValueError:
                    gt_projected_uv = None
                    gt_projection_valid = False
                observability = derive_goal_observability(
                    goal_exists=True,
                    projection_valid=gt_projection_valid,
                    projected_normalized_uv=gt_projected_uv,
                    goal_mask=labels.goal_mask[timestep],
                    object_mask=labels.object_mask[timestep],
                    legacy_visible=bool(labels.keypoint_visible[timestep, 1]),
                    support_radius_px=support_radius,
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
                        float(value)
                        for value in geometry["predicted_world_point_base_m"]
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
                write_evidence = GoalWriteEvidence(
                    visibility_probability=float(visibility[batch_index, 1]),
                    projection_validity_probability=float(
                        projection_probability[batch_index]
                    ),
                    goal_mask_probability=goal_mask_probability,
                    object_mask_probability=object_mask_probability,
                    normalized_entropy=float(entropy[batch_index, 1]),
                    radial_sigma_px=float(np.linalg.norm(sigma[batch_index, 1])),
                    geometry_valid=geometry_valid,
                    min_goal_mask_probability=min_goal_mask,
                    max_object_mask_probability=max_object_mask,
                )
                world_error = (
                    None
                    if predicted_world is None
                    else float(
                        np.linalg.norm(
                            np.asarray(predicted_world[:2], dtype=np.float64)
                            - goal_position[:2]
                        )
                    )
                )
                oracle_safe = bool(
                    observability.observable
                    and world_error is not None
                    and world_error <= safe_error
                )
                row = {
                    "version": E015_EXPERIMENT_VERSION,
                    "split": split,
                    "dataset_index": dataset_index,
                    "trajectory_id": trajectory_id,
                    "scene_id": str(audit["scene_id"]),
                    "timestep": timestep,
                    "timestamp_s": float(audit["timestamp_s"]),
                    "observability": observability.to_dict(),
                    "gt_projected_normalized_uv": (
                        None
                        if gt_projected_uv is None
                        else gt_projected_uv.astype(float).tolist()
                    ),
                    "gt_goal_position_base_m": goal_position.astype(float).tolist(),
                    "predicted_goal_normalized_uv": goal_uv.astype(float).tolist(),
                    "predicted_goal_position_base_m": (
                        None if predicted_world is None else list(predicted_world)
                    ),
                    "measurement_covariance_base_m2": (
                        None if covariance is None else covariance.astype(float).tolist()
                    ),
                    "world_xy_error_m": world_error,
                    "write_evidence": write_evidence.to_dict(),
                    "oracle_safe_measurement": oracle_safe,
                    "geometry": {
                        "valid": geometry_valid,
                        "abs_n_dot_unit_ray": geometry["abs_n_dot_unit_ray"],
                        "jacobian_sigma_max_mm_per_px": geometry[
                            "jacobian_sigma_max_mm_per_px"
                        ],
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
                        write_score=write_evidence.score,
                        structurally_eligible=write_evidence.structurally_eligible,
                        predicted_observable=write_evidence.observable,
                        geometry_valid=geometry_valid,
                        gt_position_base_m=goal_position,
                        gt_observable=observability.observable,
                    )
                )
                dataset_index += 1
    if dataset_index != len(dataset) or len(rows) != len(frames):
        raise RuntimeError("E015 DataLoader 未完整覆盖 split")
    return _CollectedSplit(
        rows=rows,
        replay_frames=frames,
        invalid_geometry_count=invalid_geometry,
    )


def _observability_sidecar(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "version": E015_EXPERIMENT_VERSION,
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
            "legacy_contract_mismatch": row["observability"][
                "legacy_contract_mismatch"
            ],
            "local_goal_visible_fraction": row["observability"][
                "local_goal_visible_fraction"
            ],
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
        "goal_projection_valid_count": sum(
            bool(label["projection_valid"]) for label in labels
        ),
        "goal_in_fov_count": sum(bool(label["in_fov"]) for label in labels),
        "goal_observable_count": sum(bool(label["observable"]) for label in labels),
        "legacy_goal_visible_count": sum(
            bool(label["legacy_visible"]) for label in labels
        ),
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
        [
            float(row["world_xy_error_m"])
            for row in accepted
            if row["world_xy_error_m"] is not None
        ],
        dtype=np.float64,
    )
    return {
        "frame_count": len(rows),
        "structurally_eligible_count": sum(
            bool(row["write_evidence"]["structurally_eligible"]) for row in rows
        ),
        "accepted_count": len(accepted),
        "accepted_unsafe_count": sum(
            not bool(row["oracle_safe_measurement"]) for row in accepted
        ),
        "accepted_while_gt_unobservable_count": sum(
            not bool(row["observability"]["observable"]) for row in accepted
        ),
        "accepted_catastrophic_count": int(
            np.sum(errors > catastrophic_error_m)
        ),
        "accepted_error_p50_mm": (
            None if errors.size == 0 else float(np.quantile(errors, 0.50) * 1000.0)
        ),
        "accepted_error_p90_mm": (
            None if errors.size == 0 else float(np.quantile(errors, 0.90) * 1000.0)
        ),
        "accepted_error_max_mm": (
            None if errors.size == 0 else float(errors.max() * 1000.0)
        ),
    }


def _memory_config(context: _Context, max_age_s: float) -> GoalMemoryConfig:
    memory = context.config["memory"]
    return GoalMemoryConfig(
        max_unobserved_age_s=max_age_s,
        max_innovation_m=float(memory["max_innovation_m"]),
        max_position_std_m=float(memory["max_position_std_m"]),
        require_covariance=bool(memory["require_covariance"]),
        covariance_growth_m2_per_s=float(memory["covariance_growth_m2_per_s"]),
    )


def _rules_payload(
    *,
    context: _Context,
    calibration: GoalWriteCalibration,
    max_age_s: float,
) -> dict[str, Any]:
    rules = {
        "version": E015_FROZEN_RULES_VERSION,
        "status": "frozen-before-test",
        "config_sha256": context.config_sha256,
        "source_tree_sha256": context.source_tree_sha256,
        "fresh_dataset_identity_sha256": context.dataset_identity_sha256,
        "fresh_seed_identity_sha256": context.seed_identity_sha256,
        "checkpoint_sha256": context.checkpoint_sha256,
        "checkpoint_parameter_sha256": context.checkpoint_parameter_sha256,
        "temperature": context.temperature,
        "write_calibration": calibration.to_dict(),
        "memory_config": {
            "max_unobserved_age_s": max_age_s,
            "max_innovation_m": context.config["memory"]["max_innovation_m"],
            "max_position_std_m": context.config["memory"][
                "max_position_std_m"
            ],
            "require_covariance": context.config["memory"]["require_covariance"],
            "covariance_growth_m2_per_s": context.config["memory"][
                "covariance_growth_m2_per_s"
            ],
            "update_policy": GOAL_MEMORY_UPDATE_POLICY,
            "frame_semantics": GOAL_POSITION_FRAME_SEMANTICS,
        },
        "test_split_status": "unread",
    }
    rules["rules_sha256"] = canonical_sha256(rules)
    return rules


def _verify_rules(rules: dict[str, Any], context: _Context) -> None:
    expected_hash = rules.get("rules_sha256")
    unhashed = {key: value for key, value in rules.items() if key != "rules_sha256"}
    if expected_hash != canonical_sha256(unhashed):
        raise RuntimeError("E015 frozen rules SHA-256 漂移")
    for actual, expected, name in (
        (rules.get("version"), E015_FROZEN_RULES_VERSION, "version"),
        (rules.get("status"), "frozen-before-test", "status"),
        (rules.get("config_sha256"), context.config_sha256, "config"),
        (
            rules.get("source_tree_sha256"),
            context.source_tree_sha256,
            "source tree",
        ),
        (
            rules.get("fresh_dataset_identity_sha256"),
            context.dataset_identity_sha256,
            "Dataset",
        ),
        (
            rules.get("fresh_seed_identity_sha256"),
            context.seed_identity_sha256,
            "seed identity",
        ),
        (rules.get("checkpoint_sha256"), context.checkpoint_sha256, "checkpoint"),
        (rules.get("test_split_status"), "unread", "test split status"),
    ):
        if actual != expected:
            raise RuntimeError(f"E015 frozen rules {name} 漂移")


def _receipt_files(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: file_sha256(root / name) for name in names}


def _run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output
    if output is None:
        raise ValueError("calibrate phase 必须提供 --output")
    if output.exists():
        raise FileExistsError(f"E015 calibration output 已存在: {output}")
    context = _context(args)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _atomic_json(
        output / "run_state.json",
        {
            "version": E015_EXPERIMENT_VERSION,
            "phase": "calibrate",
            "status": "in-progress",
            "test_split_status": "unread",
        },
    )
    split = str(context.config["fresh_dataset"]["calibration_split"])
    collected = _collect_split(
        context=context,
        deployable_root=args.deployable_root,
        label_root=args.label_root,
        split=split,
    )
    calibration = calibrate_goal_write_threshold(
        scores=[float(row["write_evidence"]["score"]) for row in collected.rows],
        structurally_eligible=np.asarray(
            [
                bool(row["write_evidence"]["structurally_eligible"])
                for row in collected.rows
            ],
            dtype=np.bool_,
        ),
        oracle_safe=np.asarray(
            [bool(row["oracle_safe_measurement"]) for row in collected.rows],
            dtype=np.bool_,
        ),
    )
    memory = context.config["memory"]
    catastrophic = float(
        context.config["write_gate"]["catastrophic_world_xy_error_m"]
    )
    max_age, age_reports = select_memory_max_age(
        collected.replay_frames,
        calibration=calibration,
        max_age_candidates_s=memory["max_unobserved_age_candidates_s"],
        max_innovation_m=float(memory["max_innovation_m"]),
        max_position_std_m=float(memory["max_position_std_m"]),
        require_covariance=bool(memory["require_covariance"]),
        covariance_growth_m2_per_s=float(memory["covariance_growth_m2_per_s"]),
        catastrophic_world_xy_error_m=catastrophic,
    )
    replay = replay_goal_memory(
        collected.replay_frames,
        calibration=calibration,
        memory_config=_memory_config(context, max_age),
    )
    replay_summary = summarize_goal_memory_replay(
        replay,
        catastrophic_world_xy_error_m=catastrophic,
    )
    rules = _rules_payload(
        context=context,
        calibration=calibration,
        max_age_s=max_age,
    )
    summary = {
        "version": E015_EXPERIMENT_VERSION,
        "phase": "calibrate",
        "status": "complete",
        "split": split,
        "trajectory_count": context.config["fresh_dataset"][
            "calibration_trajectories"
        ],
        "dataset_identity_sha256": context.dataset_identity_sha256,
        "seed_identity_sha256": context.seed_identity_sha256,
        "checkpoint_sha256": context.checkpoint_sha256,
        "config_sha256": context.config_sha256,
        "source_tree_sha256": context.source_tree_sha256,
        "observability_audit": _observability_summary(collected.rows),
        "write_calibration": calibration.to_dict(),
        "write_measurements": _measurement_summary(
            collected.rows,
            calibration,
            catastrophic_error_m=catastrophic,
        ),
        "selected_max_unobserved_age_s": max_age,
        "memory_age_candidates": age_reports,
        "memory_replay": replay_summary,
        "invalid_geometry_count": collected.invalid_geometry_count,
        "test_split_status": "unread",
        "actuation_allowed": False,
    }
    _atomic_jsonl(output / "observability_sidecar.jsonl", _observability_sidecar(collected.rows))
    _atomic_jsonl(output / "prediction_rows.jsonl", collected.rows)
    _atomic_jsonl(
        output / "memory_replay.jsonl",
        [record.to_dict() for record in replay],
    )
    _atomic_json(output / "frozen_rules.json", rules)
    _atomic_json(output / "summary.json", summary)
    _atomic_json(output / "config_snapshot.json", context.config)
    files = _receipt_files(
        output,
        (
            "observability_sidecar.jsonl",
            "prediction_rows.jsonl",
            "memory_replay.jsonl",
            "frozen_rules.json",
            "summary.json",
            "config_snapshot.json",
        ),
    )
    receipt = {
        "version": E015_EXPERIMENT_VERSION,
        "phase": "calibrate",
        "status": "complete",
        "rules_sha256": rules["rules_sha256"],
        "files": files,
        "test_split_status": "unread",
    }
    _atomic_json(output / "receipt.json", receipt)
    _atomic_json(
        output / "run_state.json",
        {
            "version": E015_EXPERIMENT_VERSION,
            "phase": "calibrate",
            "status": "complete",
            "test_split_status": "unread",
        },
    )
    return summary


def _public_readme(summary: dict[str, Any]) -> str:
    audit = summary["e015_a_observability_audit"]
    measurement = summary["e015_b_write_measurements"]
    memory = summary["e015_b_memory_replay"]
    return f"""# E015 — Explicit geometric goal state memory

E015 使用 frozen E013 Precision checkpoint，在与 E013 dataset/shadow 不重叠的 fresh seeds 上完成：

- E015-A：将 goal `exists / projection_valid / in_fov / observable` 显式拆分；
- E015-B：只把通过 fresh-validation write gate 的 base-frame measurement 写入 episode-scoped memory；
- 全程 BF16 CUDA shadow replay，未训练、未修改 checkpoint、未产生或执行机器人 Action。

## E015-A observability audit

- evaluation frames：`{audit["frame_count"]}`
- legacy goal visible：`{audit["legacy_goal_visible_count"]}`
- corrected goal observable：`{audit["goal_observable_count"]}`
- legacy-visible 但 center 不可观察：`{audit["legacy_contract_mismatch_count"]}`

## E015-B memory replay

- write-gate accepted measurements：`{measurement["accepted_count"]}`
- accepted unsafe / catastrophic：`{measurement["accepted_unsafe_count"]}` / `{measurement["accepted_catastrophic_count"]}`
- current-measurement coverage：`{memory["current_measurement_coverage"]:.6f}`
- memory coverage：`{memory["memory_coverage"]:.6f}`
- current valid while GT unobservable：`{memory["current_valid_while_gt_unobservable_count"]}`
- memory valid while GT unobservable：`{memory["memory_valid_while_gt_unobservable_count"]}`
- memory catastrophic states：`{memory["memory_catastrophic_count"]}`
- Episode reset leakage：`{memory["episode_reset_leakage_count"]}`

工程 gate passed=`{summary["engineering_gate_passed"]}`。这只说明该显式 memory 是否值得进入下一轮
no-actuation shadow；E015 明确不授权 actuator promotion。
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
        raise RuntimeError("E015 public README 含敏感绝对路径")
    _atomic_text(root / "README.md", readme)
    verifier = repository_root / "scripts" / "verify_e015_public_results.py"
    if not verifier.is_file():
        raise FileNotFoundError("E015 public verifier template 不存在")
    shutil.copyfile(verifier, root / "verify_summary.py")
    files = _receipt_files(
        root,
        ("README.md", "sanitized_summary.json", "verify_summary.py"),
    )
    receipt = {
        "version": E015_PUBLIC_VERSION,
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


def _run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if args.rules is None or args.private_output is None or args.public_output is None:
        raise ValueError(
            "evaluate phase 必须提供 --rules/--private-output/--public-output"
        )
    if args.private_output.exists() or args.public_output.exists():
        raise FileExistsError("E015 evaluation private/public output 已存在")
    context = _context(args)
    rules = _read_json(args.rules, "E015 frozen rules")
    _verify_rules(rules, context)
    test_once_claim_sha256 = _claim_test_evaluation_once(
        args.rules.parent / "test_evaluation_claim.json",
        rules_sha256=str(rules["rules_sha256"]),
        dataset_identity_sha256=context.dataset_identity_sha256,
        seed_identity_sha256=context.seed_identity_sha256,
        source_tree_sha256=context.source_tree_sha256,
    )
    private = args.private_output
    private.mkdir(mode=0o700, parents=True, exist_ok=False)
    _atomic_json(
        private / "run_state.json",
        {
            "version": E015_EXPERIMENT_VERSION,
            "phase": "evaluate",
            "status": "in-progress",
            "test_split_status": "consuming-once",
        },
    )
    split = str(context.config["fresh_dataset"]["evaluation_split"])
    collected = _collect_split(
        context=context,
        deployable_root=args.deployable_root,
        label_root=args.label_root,
        split=split,
    )
    calibration = GoalWriteCalibration(**rules["write_calibration"])
    memory_config = GoalMemoryConfig(**rules["memory_config"])
    replay = replay_goal_memory(
        collected.replay_frames,
        calibration=calibration,
        memory_config=memory_config,
    )
    catastrophic = float(
        context.config["write_gate"]["catastrophic_world_xy_error_m"]
    )
    observability_summary = _observability_summary(collected.rows)
    measurement_summary = _measurement_summary(
        collected.rows,
        calibration,
        catastrophic_error_m=catastrophic,
    )
    replay_summary = summarize_goal_memory_replay(
        replay,
        catastrophic_world_xy_error_m=catastrophic,
    )
    criteria = context.config["success_criteria"]
    coverage_improved = (
        int(replay_summary["memory_valid_while_gt_unobservable_count"])
        > int(replay_summary["current_valid_while_gt_unobservable_count"])
    )
    engineering_gate = bool(
        int(measurement_summary["accepted_unsafe_count"])
        <= int(criteria["test_unsafe_write_count_max"])
        and int(replay_summary["memory_catastrophic_count"])
        <= int(criteria["test_memory_catastrophic_count_max"])
        and int(replay_summary["episode_reset_leakage_count"])
        <= int(criteria["episode_reset_leakage_count_max"])
        and coverage_improved
    )
    private_summary = {
        "version": E015_EXPERIMENT_VERSION,
        "phase": "evaluate",
        "status": "complete",
        "split": split,
        "trajectory_count": context.config["fresh_dataset"][
            "evaluation_trajectories"
        ],
        "dataset_identity_sha256": context.dataset_identity_sha256,
        "seed_identity_sha256": context.seed_identity_sha256,
        "checkpoint_sha256": context.checkpoint_sha256,
        "checkpoint_parameter_sha256": context.checkpoint_parameter_sha256,
        "checkpoint_provenance_sha256": context.checkpoint_provenance_sha256,
        "config_sha256": context.config_sha256,
        "source_tree_sha256": context.source_tree_sha256,
        "rules_sha256": rules["rules_sha256"],
        "test_once_claim_sha256": test_once_claim_sha256,
        "observability_audit": observability_summary,
        "write_measurements": measurement_summary,
        "memory_replay": replay_summary,
        "coverage_improved_while_unobservable": coverage_improved,
        "engineering_gate_passed": engineering_gate,
        "invalid_geometry_count": collected.invalid_geometry_count,
        "test_split_status_after_e015": "consumed-for-evaluation",
        "actuation_allowed": False,
        "actuator_promotion_allowed": False,
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
        "version": E015_EXPERIMENT_VERSION,
        "phase": "evaluate",
        "status": "complete",
        "rules_sha256": rules["rules_sha256"],
        "files": private_files,
        "test_split_status_after_e015": "consumed-for-evaluation",
    }
    _atomic_json(private / "receipt.json", private_receipt)
    private_receipt_sha256 = file_sha256(private / "receipt.json")
    public_summary = {
        "version": E015_PUBLIC_VERSION,
        "status": "complete",
        "experiment": "E015 — explicit geometric goal state memory",
        "fresh_dataset": {
            "seed_range_start": context.config["fresh_dataset"]["start_seed"],
            "candidate_count": context.config["fresh_dataset"]["max_candidates"],
            "calibration_trajectory_count": context.config["fresh_dataset"][
                "calibration_trajectories"
            ],
            "evaluation_trajectory_count": context.config["fresh_dataset"][
                "evaluation_trajectories"
            ],
            "dataset_identity_sha256": context.dataset_identity_sha256,
            "seed_identity_sha256": context.seed_identity_sha256,
        },
        "frozen_conditions": {
            "checkpoint_sha256": context.checkpoint_sha256,
            "checkpoint_changed": False,
            "temperature": context.temperature,
            "training_performed": False,
            "write_threshold_selected_on": "fresh-validation-only",
            "memory_age_selected_on": "fresh-validation-only",
            "test_evaluated_once": True,
            "test_once_claim_sha256": test_once_claim_sha256,
            "actuation_allowed": False,
        },
        "rules_sha256": rules["rules_sha256"],
        "write_calibration": calibration.to_dict(),
        "memory_config": rules["memory_config"],
        "e015_a_observability_audit": observability_summary,
        "e015_b_write_measurements": measurement_summary,
        "e015_b_memory_replay": replay_summary,
        "coverage_improved_while_unobservable": coverage_improved,
        "engineering_gate_passed": engineering_gate,
        "safe_for_actuator_promotion": False,
        "test_split_status_after_e015": "consumed-for-evaluation",
    }
    public_receipt = _write_public(
        root=args.public_output,
        repository_root=args.repository_root,
        summary=public_summary,
        private_receipt_sha256=private_receipt_sha256,
    )
    _atomic_json(
        private / "run_state.json",
        {
            "version": E015_EXPERIMENT_VERSION,
            "phase": "evaluate",
            "status": "complete",
            "test_split_status_after_e015": "consumed-for-evaluation",
            "public_receipt_sha256": canonical_sha256(public_receipt),
        },
    )
    return public_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibrate", "evaluate"), required=True)
    parser.add_argument("--deployable-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--e013-config", type=Path, required=True)
    parser.add_argument("--e015-config", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--held-out-output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rules", type=Path)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = (
        _run_calibration(args)
        if args.phase == "calibrate"
        else _run_evaluation(args)
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
