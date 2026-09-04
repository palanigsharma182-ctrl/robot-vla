"""E018-P0 recorded-validation Object Memory 开发实验；本模块不提供 test 入口。"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_vla.data.trajectory import load_manifest, resolve_trajectory_path
from robot_vla.observation import rotation_6d_to_matrix
from robot_vla.precision.checkpoint import PrecisionCheckpointRole, load_precision_checkpoint
from robot_vla.precision.data import (
    PrecisionRGBDataset,
    canonical_sha256,
    file_sha256,
    load_precision_label_manifest,
)
from robot_vla.precision.e016_training import load_e016_p1_config
from robot_vla.precision.geometry import project_base_point_to_normalized_uv
from robot_vla.precision.object_memory import ObjectMemoryConfig, ObjectMemorySafetyContext
from robot_vla.precision.object_memory_evaluation import (
    ObjectReplayFrame,
    calibrate_object_write_threshold,
    replay_object_memory,
    summarize_object_memory_replay,
)
from robot_vla.precision.object_observability import (
    OBJECT_OBSERVABILITY_SEMANTICS,
    OBJECT_WRITE_SCORE_SEMANTICS,
    ObjectWriteEvidence,
    derive_object_observability,
)
from robot_vla.precision.observability import mask_probability_at_normalized_uv
from robot_vla.precision.outliers import geometry_conditioning
from robot_vla.precision.training import _build_loader, _to_device

E018_P0_DEVELOPMENT_VERSION = "e018-p0-dual-memory-development/v1"
E018_P0_RESULT_VERSION = "e018-p0-recorded-validation-result/v1"
_SOURCE_FILES = (
    "src/robot_vla/precision/e018_evaluation.py",
    "src/robot_vla/precision/object_memory.py",
    "src/robot_vla/precision/object_memory_evaluation.py",
    "src/robot_vla/precision/object_observability.py",
)


@dataclass(frozen=True)
class _EvaluationContext:
    config: dict[str, Any]
    config_sha256: str
    checkpoint_sha256: str
    checkpoint_parameter_sha256: str
    checkpoint_provenance_sha256: str
    keypoint_temperature: float
    model: Any


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


def _require_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} keys 漂移: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} 必须是有限正数")
    return result


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    value = _read_json(path, "E018-P0 development config")
    _require_keys(
        value,
        {
            "version",
            "status",
            "parent",
            "source",
            "geometry",
            "observability",
            "write_calibration",
            "candidate_search",
            "memory_search",
            "safety",
            "execution",
        },
        "E018-P0 config",
    )
    if value["version"] != E018_P0_DEVELOPMENT_VERSION:
        raise ValueError("E018-P0 development config version 漂移")
    if value["status"] != "development-only-validation-no-test":
        raise ValueError("E018-P0 config 不得伪装成 formal/test")
    source = value["source"]
    if not isinstance(source, dict):
        raise TypeError("source 必须是 object")
    _require_keys(
        source,
        {
            "allowed_split",
            "test_split_allowed",
            "expected_trajectory_count",
            "expected_frame_count",
            "expected_pregrasp_frame_count",
            "pregrasp_skill_id",
            "usage",
        },
        "source",
    )
    if source["allowed_split"] != "val" or source["test_split_allowed"] is not False:
        raise ValueError("E018-P0 development 只允许 val，必须显式禁止 test")
    for name in (
        "expected_trajectory_count",
        "expected_frame_count",
        "expected_pregrasp_frame_count",
    ):
        _positive_int(source[name], f"source.{name}")
    if source["pregrasp_skill_id"] != 0:
        raise ValueError("首版 offline pregrasp alignment 必须绑定 reach skill_id=0")
    parent = value["parent"]
    if not isinstance(parent, dict):
        raise TypeError("parent 必须是 object")
    _require_keys(
        parent,
        {
            "e016_config_sha256",
            "checkpoint_sha256",
            "checkpoint_parameter_sha256",
            "checkpoint_provenance_sha256",
            "goal_memory_rules_sha256",
            "goal_memory_calibration_receipt_sha256",
            "selected_epoch",
            "source_camera",
            "source_model_identity",
        },
        "parent",
    )
    for name in (
        "e016_config_sha256",
        "checkpoint_sha256",
        "checkpoint_parameter_sha256",
        "checkpoint_provenance_sha256",
        "goal_memory_rules_sha256",
        "goal_memory_calibration_receipt_sha256",
    ):
        candidate = parent[name]
        if not isinstance(candidate, str) or len(candidate) != 64:
            raise ValueError(f"parent.{name} 必须是 SHA-256")
    if parent["selected_epoch"] != 12 or parent["source_camera"] != "hand_camera":
        raise ValueError("E018-P0 parent epoch/camera 漂移")

    geometry = value["geometry"]
    if geometry.get("plane_source") != "pick-cube-task-contract-not-runtime-gt/v1":
        raise ValueError("Object plane 不得来自 runtime GT")
    _positive(geometry["pregrasp_object_center_plane_base_z_m"], "object plane z")
    observability = value["observability"]
    if observability.get("support_radius_px") != 2:
        raise ValueError("首版 support_radius_px 必须固定为 2")
    for name in ("min_object_mask_probability", "max_goal_mask_probability"):
        probability = float(observability[name])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"observability.{name} 必须位于 [0,1]")
    write = value["write_calibration"]
    if write.get("threshold_policy") != "maximum-coverage-zero-unsafe-on-validation/v1":
        raise ValueError("Object write threshold policy 漂移")
    _positive(write["safe_world_xyz_error_m"], "safe error")
    _positive(write["catastrophic_world_xyz_error_m"], "catastrophic error")

    candidate = value["candidate_search"]
    frame_candidates = candidate.get("min_candidate_frames")
    spread_candidates = candidate.get("max_candidate_position_spread_candidates_m")
    if not isinstance(frame_candidates, list) or not frame_candidates:
        raise ValueError("candidate frame grid 不能为空")
    if not isinstance(spread_candidates, list) or not spread_candidates:
        raise ValueError("candidate spread grid 不能为空")
    if len(set(frame_candidates)) != len(frame_candidates):
        raise ValueError("candidate frame grid 不能重复")
    if len(set(spread_candidates)) != len(spread_candidates):
        raise ValueError("candidate spread grid 不能重复")
    for item in frame_candidates:
        _positive_int(item, "min_candidate_frames candidate")
    for item in spread_candidates:
        _positive(item, "position spread candidate")
    _positive(candidate["max_candidate_gap_s"], "max_candidate_gap_s")

    memory = value["memory_search"]
    age_candidates = memory.get("max_unobserved_age_candidates_s")
    if not isinstance(age_candidates, list) or not age_candidates:
        raise ValueError("memory age grid 不能为空")
    if len(set(age_candidates)) != len(age_candidates):
        raise ValueError("memory age grid 不能重复")
    for item in age_candidates:
        _positive(item, "memory age candidate")
    _positive(memory["max_innovation_m"], "max_innovation_m")
    _positive(memory["max_position_std_m"], "max_position_std_m")
    if memory["require_covariance"] is not True:
        raise ValueError("E018-P0 必须要求 covariance")
    growth = float(memory["covariance_growth_m2_per_s"])
    if not math.isfinite(growth) or growth < 0.0:
        raise ValueError("covariance growth 必须有限非负")
    if (
        memory.get("selection_policy")
        != "max-navigation-gain-zero-safety-violation-then-stricter-tie-break/v1"
    ):
        raise ValueError("memory selection policy 漂移")

    safety = value["safety"]
    for name in (
        "gripper_opening_min",
        "finger_contact_threshold_n",
        "controller_tracking_error_max_rad",
        "max_sensor_skew_s",
    ):
        _positive(safety[name], f"safety.{name}")
    execution = value["execution"]
    if (
        execution.get("device") != "cuda"
        or execution.get("use_bf16") is not True
        or execution.get("num_workers") != 0
        or execution.get("actuation_allowed") is not False
        or execution.get("camera_motion_allowed") is not False
    ):
        raise ValueError("E018-P0 execution 必须是 CUDA BF16 no-actuation/no-camera-motion")
    _positive_int(execution["batch_size"], "execution.batch_size")
    return value


def _load_context(
    *,
    config_path: Path,
    parent_config_path: Path,
    training_output: Path,
    goal_calibration_root: Path,
) -> _EvaluationContext:
    config = _load_config(config_path)
    parent = config["parent"]
    e016_config = load_e016_p1_config(parent_config_path)
    if e016_config.sha256 != parent["e016_config_sha256"]:
        raise RuntimeError("E016 parent config SHA-256 漂移")
    receipt_path = training_output / "checkpoint_receipt.json"
    receipt = _read_json(receipt_path, "E016-P1 checkpoint receipt")
    checkpoint = receipt.get("checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or receipt.get("passed") is not True
        or receipt.get("selected_epoch") != parent["selected_epoch"]
        or receipt.get("training_config_sha256") != e016_config.sha256
        or checkpoint.get("checkpoint_sha256") != parent["checkpoint_sha256"]
        or checkpoint.get("parameter_state_sha256")
        != parent["checkpoint_parameter_sha256"]
        or checkpoint.get("provenance_sha256")
        != parent["checkpoint_provenance_sha256"]
    ):
        raise RuntimeError("E018-P0 checkpoint parent identity 漂移")
    if file_sha256(goal_calibration_root / "receipt.json") != parent[
        "goal_memory_calibration_receipt_sha256"
    ]:
        raise RuntimeError("E016 Goal Memory calibration receipt SHA-256 漂移")
    goal_rules = _read_json(goal_calibration_root / "frozen_rules.json", "Goal Memory rules")
    if goal_rules.get("rules_sha256") != parent["goal_memory_rules_sha256"]:
        raise RuntimeError("E016 Goal Memory rules identity 漂移")
    loaded = load_precision_checkpoint(
        training_output / "precision-formal.pt",
        expected_checkpoint_sha256=parent["checkpoint_sha256"],
        expected_provenance_sha256=parent["checkpoint_provenance_sha256"],
        expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
    )
    if loaded.receipt.parameter_state_sha256 != parent["checkpoint_parameter_sha256"]:
        raise RuntimeError("E018-P0 loaded parameter identity 漂移")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("E018-P0 development replay 要求支持 BF16 的 CUDA GPU")
    return _EvaluationContext(
        config=config,
        config_sha256=canonical_sha256(config),
        checkpoint_sha256=loaded.receipt.checkpoint_sha256,
        checkpoint_parameter_sha256=loaded.receipt.parameter_state_sha256,
        checkpoint_provenance_sha256=loaded.receipt.provenance_sha256,
        keypoint_temperature=e016_config.loss.keypoint_temperature,
        model=loaded.model,
    )


def _validation_identity(
    *,
    deployable_root: Path,
    label_root: Path,
    expected_trajectories: int,
    expected_frames: int,
) -> str:
    sources = {entry.trajectory_id: entry for entry in load_manifest(deployable_root, split="val")}
    labels = {
        entry.trajectory_id: entry
        for entry in load_precision_label_manifest(label_root, split="val")
    }
    if set(sources) != set(labels) or len(sources) != expected_trajectories:
        raise RuntimeError("E018-P0 validation source/label trajectory identity 漂移")
    if sum(entry.num_steps for entry in labels.values()) != expected_frames:
        raise RuntimeError("E018-P0 validation frame count 漂移")
    files = []
    for trajectory_id in sorted(sources):
        source = sources[trajectory_id]
        label = labels[trajectory_id]
        source_path = resolve_trajectory_path(deployable_root, source.file)
        label_path = (label_root.resolve() / label.file).resolve()
        if not label_path.is_relative_to(label_root.resolve()) or not label_path.is_file():
            raise RuntimeError("E018-P0 validation label path 越界或缺失")
        source_sha256 = file_sha256(source_path)
        if source_sha256 != label.source_trajectory_sha256:
            raise RuntimeError("E018-P0 validation source/label SHA-256 绑定漂移")
        files.append(
            {
                "trajectory_id": trajectory_id,
                "source_sha256": source_sha256,
                "label_sha256": file_sha256(label_path),
            }
        )
    return canonical_sha256({"split": "val", "files": files})


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
        raise ValueError("E018-P0 covariance 输入 shape 漂移")
    covariance = np.zeros((3, 3), dtype=np.float64)
    covariance[:2, :2] = jacobian @ np.diag(np.square(sigma)) @ jacobian.T
    if not np.isfinite(covariance).all():
        raise ValueError("E018-P0 measurement covariance 非有限")
    return covariance


def _safety_context(
    arrays: Any,
    timestep: int,
    config: dict[str, Any],
) -> tuple[ObjectMemorySafetyContext, dict[str, Any]]:
    source = config["source"]
    safety = config["safety"]
    skill_id = int(arrays.skill_id[timestep])
    pregrasp = skill_id == source["pregrasp_skill_id"]
    gripper_opening = float(arrays.proprio[timestep, -1])
    finger_force = float(
        max(arrays.left_finger_force_n[timestep], arrays.right_finger_force_n[timestep])
    )
    tracking_error = float(
        np.max(
            np.abs(
                arrays.previous_command_q_rad[timestep]
                - arrays.proprio[timestep, :7]
            )
        )
    )
    modality_valid = bool(
        arrays.wrist_valid[timestep]
        and arrays.proprio_valid[timestep]
        and arrays.tcp_pose_valid[timestep]
        and arrays.camera_pose_valid[timestep]
        and arrays.finger_force_valid[timestep]
    )
    controller_valid = bool(
        modality_valid
        and arrays.previous_command_valid[timestep]
        and tracking_error <= safety["controller_tracking_error_max_rad"]
    )
    context = ObjectMemorySafetyContext(
        pregrasp_window_open=pregrasp,
        gripper_open=gripper_opening >= safety["gripper_opening_min"],
        controller_tracking_valid=controller_valid,
        object_contact_detected=finger_force > safety["finger_contact_threshold_n"],
        gripper_close_commanded=float(arrays.action[timestep, -1])
        < safety["gripper_opening_min"],
        grasp_candidate=False,
        grasp_verified=False,
        object_maybe_moved=False,
    )
    return context, {
        "skill_id": skill_id,
        "pregrasp_window_open": pregrasp,
        "gripper_opening": gripper_opening,
        "max_finger_force_n": finger_force,
        "max_joint_tracking_error_rad": tracking_error,
        "modality_valid": modality_valid,
        "controller_tracking_valid": controller_valid,
    }


def _collect_validation(
    *,
    context: _EvaluationContext,
    deployable_root: Path,
    label_root: Path,
) -> tuple[list[dict[str, Any]], list[ObjectReplayFrame], dict[str, Any]]:
    config = context.config
    dataset = PrecisionRGBDataset(deployable_root, label_root, "val", cache_size=32)
    device = torch.device("cuda")
    model = context.model.to(device)
    model.eval()
    model.requires_grad_(False)
    loader = _build_loader(
        dataset,
        batch_size=config["execution"]["batch_size"],
        shuffle=False,
        seed=0,
        num_workers=0,
    )
    model_config = model.config
    if tuple(model_config.keypoint_names) != ("object_center", "goal_center"):
        raise RuntimeError("E018-P0 需要 object_center/goal_center 固定 keypoint 顺序")
    if tuple(model_config.mask_names) != ("object", "goal"):
        raise RuntimeError("E018-P0 需要 object_mask/goal_mask 固定 channel 顺序")
    observability_config = config["observability"]
    write_config = config["write_calibration"]
    plane_z = float(config["geometry"]["pregrasp_object_center_plane_base_z_m"])
    source_camera = str(config["parent"]["source_camera"])
    model_identity = str(config["parent"]["source_model_identity"])
    rows: list[dict[str, Any]] = []
    frames: list[ObjectReplayFrame] = []
    invalid_geometry_count = 0
    dataset_index = 0
    object_mask_intersection = 0
    object_mask_union = 0
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
                temperature=context.keypoint_temperature
            )
            predicted_uv = decoded.keypoints.normalized_uv.detach().float().cpu().numpy()
            visibility = decoded.visibility_probability.detach().float().cpu().numpy()
            projection = decoded.projection_validity_probability.detach().float().cpu().numpy()
            entropy = decoded.keypoints.normalized_entropy.detach().float().cpu().numpy()
            sigma = decoded.keypoint_sigma_px.detach().float().cpu().numpy()
            mask_probability = torch.sigmoid(output.mask_logits.detach().float()).cpu().numpy()
            predicted_object_mask = output.mask_logits[:, 0].detach() > 0.0
            target_object_mask = batch["mask_targets"][:, 0] > 0.5
            object_mask_intersection += int(
                (predicted_object_mask & target_object_mask).sum().item()
            )
            object_mask_union += int((predicted_object_mask | target_object_mask).sum().item())
            height, width = batch["image"].shape[-2:]
            image_size_hw = (int(height), int(width))
            if predicted_uv.shape[1:] != (2, 2) or mask_probability.shape[1] != 2:
                raise RuntimeError("E018-P0 U-Net object/goal channel contract 漂移")
            for batch_index, audit in enumerate(raw_batch["audit"]):
                trajectory_id = str(audit["trajectory_id"])
                timestep = int(audit["timestep"])
                label_meta = dataset.label_by_trajectory[trajectory_id]
                labels = dataset.label_store.get(label_meta)
                source_meta = dataset.source_by_trajectory[trajectory_id]
                arrays = dataset.base.store.get(source_meta)
                gt_position = labels.object_position_base_m[timestep].astype(np.float64)
                transform = _base_from_camera(audit)
                intrinsic = np.asarray(audit["intrinsic_wrist_cv"], dtype=np.float64)
                try:
                    gt_uv = project_base_point_to_normalized_uv(
                        gt_position,
                        intrinsic,
                        transform,
                        image_size_hw,
                    )
                    gt_projection_valid = True
                except ValueError:
                    gt_uv = None
                    gt_projection_valid = False
                observability = derive_object_observability(
                    object_exists=True,
                    projection_valid=gt_projection_valid,
                    projected_normalized_uv=gt_uv,
                    object_mask=labels.object_mask[timestep],
                    goal_mask=labels.goal_mask[timestep],
                    legacy_visible=bool(labels.keypoint_visible[timestep, 0]),
                    support_radius_px=observability_config["support_radius_px"],
                )
                object_uv = predicted_uv[batch_index, 0]
                object_mask_probability = mask_probability_at_normalized_uv(
                    mask_probability[batch_index, 0],
                    object_uv,
                )
                goal_mask_probability = mask_probability_at_normalized_uv(
                    mask_probability[batch_index, 1],
                    object_uv,
                )
                try:
                    geometry = geometry_conditioning(
                        normalized_uv=object_uv,
                        intrinsic_cv=intrinsic,
                        base_from_camera_cv=transform,
                        image_size_hw=image_size_hw,
                        plane_base_z_m=plane_z,
                    )
                    predicted_world = tuple(
                        float(value) for value in geometry["predicted_world_point_base_m"]
                    )
                    covariance = _measurement_covariance(
                        geometry["local_jacobian_xy_m_per_px"],
                        sigma[batch_index, 0],
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
                    invalid_geometry_count += 1
                evidence = ObjectWriteEvidence(
                    visibility_probability=float(visibility[batch_index, 0]),
                    projection_validity_probability=float(projection[batch_index]),
                    object_mask_probability=object_mask_probability,
                    goal_mask_probability=goal_mask_probability,
                    normalized_entropy=float(entropy[batch_index, 0]),
                    radial_sigma_px=float(np.linalg.norm(sigma[batch_index, 0])),
                    geometry_valid=geometry_valid,
                    min_object_mask_probability=observability_config[
                        "min_object_mask_probability"
                    ],
                    max_goal_mask_probability=observability_config[
                        "max_goal_mask_probability"
                    ],
                )
                world_error = (
                    None
                    if predicted_world is None
                    else float(
                        np.linalg.norm(
                            np.asarray(predicted_world, dtype=np.float64) - gt_position
                        )
                    )
                )
                safety, safety_audit = _safety_context(arrays, timestep, config)
                oracle_safe = bool(
                    safety.pregrasp_window_open
                    and observability.observable
                    and world_error is not None
                    and world_error <= write_config["safe_world_xyz_error_m"]
                )
                timestamp_s = float(audit["timestamp_s"])
                rgb_timestamp_s = float(arrays.timestamp_wrist[timestep])
                camera_pose_timestamp_s = float(arrays.timestamp_camera_pose[timestep])
                tcp_pose_timestamp_s = float(arrays.timestamp_tcp_pose[timestep])
                uv_error = (
                    None
                    if not observability.observable or gt_uv is None
                    else np.abs(object_uv - gt_uv).astype(float).tolist()
                )
                pixel_error = (
                    None
                    if not observability.observable or gt_uv is None
                    else float(
                        np.linalg.norm(
                            (object_uv - gt_uv)
                            * np.asarray((float(width), float(height)))
                        )
                    )
                )
                row = {
                    "version": E018_P0_RESULT_VERSION,
                    "split": "val",
                    "dataset_index": dataset_index,
                    "trajectory_id": trajectory_id,
                    "scene_id": str(audit["scene_id"]),
                    "timestep": timestep,
                    "timestamp_s": timestamp_s,
                    "rgb_timestamp_s": rgb_timestamp_s,
                    "camera_pose_timestamp_s": camera_pose_timestamp_s,
                    "tcp_pose_timestamp_s": tcp_pose_timestamp_s,
                    "safety": safety_audit,
                    "observability": observability.to_dict(),
                    "gt_projected_normalized_uv": (
                        None if gt_uv is None else gt_uv.astype(float).tolist()
                    ),
                    "gt_object_position_base_m": gt_position.astype(float).tolist(),
                    "predicted_object_normalized_uv": object_uv.astype(float).tolist(),
                    "object_normalized_uv_abs_error": uv_error,
                    "object_pixel_error": pixel_error,
                    "object_visibility_probability": float(visibility[batch_index, 0]),
                    "projection_validity_probability": float(projection[batch_index]),
                    "predicted_object_position_base_m": (
                        None if predicted_world is None else list(predicted_world)
                    ),
                    "measurement_covariance_base_m2": (
                        None if covariance is None else covariance.astype(float).tolist()
                    ),
                    "world_xyz_error_m": world_error,
                    "write_evidence": evidence.to_dict(),
                    "oracle_safe_measurement": oracle_safe,
                    "geometry": {
                        "valid": geometry_valid,
                        "plane_base_z_m": plane_z,
                        "plane_source": config["geometry"]["plane_source"],
                        "abs_n_dot_unit_ray": geometry["abs_n_dot_unit_ray"],
                        "jacobian_sigma_max_mm_per_px": geometry[
                            "jacobian_sigma_max_mm_per_px"
                        ],
                    },
                }
                rows.append(row)
                frames.append(
                    ObjectReplayFrame(
                        episode_id=trajectory_id,
                        timestep=timestep,
                        timestamp_s=timestamp_s,
                        rgb_timestamp_s=rgb_timestamp_s,
                        camera_pose_timestamp_s=camera_pose_timestamp_s,
                        tcp_pose_timestamp_s=tcp_pose_timestamp_s,
                        predicted_position_base_m=predicted_world,
                        measurement_covariance_base_m2=covariance,
                        write_score=evidence.score,
                        structurally_eligible=evidence.structurally_eligible,
                        predicted_observable=evidence.observable,
                        geometry_valid=geometry_valid,
                        gt_position_base_m=gt_position,
                        gt_observable=observability.observable,
                        oracle_safe_measurement=oracle_safe,
                        safety=safety,
                        source_camera=source_camera,
                        source_model_identity=model_identity,
                    )
                )
                dataset_index += 1
    if dataset_index != len(dataset) or len(rows) != len(frames):
        raise RuntimeError("E018-P0 DataLoader 未完整覆盖 validation")
    pregrasp_rows = [row for row in rows if row["safety"]["pregrasp_window_open"]]
    if len(pregrasp_rows) != config["source"]["expected_pregrasp_frame_count"]:
        raise RuntimeError("E018-P0 pregrasp frame count 漂移")
    observable_rows = [row for row in pregrasp_rows if row["observability"]["observable"]]
    pixel_errors = np.asarray(
        [
            row["object_pixel_error"]
            for row in observable_rows
            if row["object_pixel_error"] is not None
        ],
        dtype=np.float64,
    )
    world_errors = np.asarray(
        [
            row["world_xyz_error_m"]
            for row in observable_rows
            if row["world_xyz_error_m"] is not None
        ],
        dtype=np.float64,
    )
    if pixel_errors.size == 0 or world_errors.size == 0:
        raise RuntimeError("E018-P0 validation 缺少可评估的 pregrasp object 样本")
    predicted_visible = [row["object_visibility_probability"] >= 0.5 for row in pregrasp_rows]
    labels = [bool(row["observability"]["observable"]) for row in pregrasp_rows]
    true_positive = sum(predicted and target for predicted, target in zip(predicted_visible, labels))
    false_positive = sum(predicted and not target for predicted, target in zip(predicted_visible, labels))
    false_negative = sum(not predicted and target for predicted, target in zip(predicted_visible, labels))
    perception = {
        "frame_count": len(rows),
        "pregrasp_frame_count": len(pregrasp_rows),
        "pregrasp_object_observable_count": sum(labels),
        "pregrasp_object_unobservable_count": len(labels) - sum(labels),
        "object_observable_pixel_error_p50": float(np.quantile(pixel_errors, 0.50)),
        "object_observable_pixel_error_p90": float(np.quantile(pixel_errors, 0.90)),
        "object_observable_world_xyz_error_p50_mm": float(
            np.quantile(world_errors, 0.50) * 1000.0
        ),
        "object_observable_world_xyz_error_p90_mm": float(
            np.quantile(world_errors, 0.90) * 1000.0
        ),
        "object_observable_world_xyz_error_max_mm": float(world_errors.max() * 1000.0),
        "object_visibility_precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        ),
        "object_visibility_recall": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        ),
        "object_mask_iou_all_validation": (
            object_mask_intersection / object_mask_union if object_mask_union else 1.0
        ),
        "invalid_geometry_count": invalid_geometry_count,
        "observability_semantics": OBJECT_OBSERVABILITY_SEMANTICS,
        "write_score_semantics": OBJECT_WRITE_SCORE_SEMANTICS,
    }
    return rows, frames, perception


def _memory_config(
    context: _EvaluationContext,
    *,
    min_candidate_frames: int,
    max_candidate_position_spread_m: float,
    max_unobserved_age_s: float,
) -> ObjectMemoryConfig:
    config = context.config
    return ObjectMemoryConfig(
        max_unobserved_age_s=max_unobserved_age_s,
        max_innovation_m=config["memory_search"]["max_innovation_m"],
        max_position_std_m=config["memory_search"]["max_position_std_m"],
        min_candidate_frames=min_candidate_frames,
        max_candidate_gap_s=config["candidate_search"]["max_candidate_gap_s"],
        max_candidate_position_spread_m=max_candidate_position_spread_m,
        max_sensor_skew_s=config["safety"]["max_sensor_skew_s"],
        expected_source_camera=config["parent"]["source_camera"],
        expected_source_model_identity=config["parent"]["source_model_identity"],
        require_covariance=config["memory_search"]["require_covariance"],
        covariance_growth_m2_per_s=config["memory_search"][
            "covariance_growth_m2_per_s"
        ],
    )


def _select_memory_rules(
    context: _EvaluationContext,
    frames: list[ObjectReplayFrame],
    calibration: Any,
) -> tuple[ObjectMemoryConfig, list[dict[str, Any]], list[Any]]:
    config = context.config
    catastrophic = config["write_calibration"]["catastrophic_world_xyz_error_m"]
    reports: list[dict[str, Any]] = []
    best: tuple[tuple[float, ...], ObjectMemoryConfig, list[Any]] | None = None
    for min_frames in config["candidate_search"]["min_candidate_frames"]:
        for spread in config["candidate_search"][
            "max_candidate_position_spread_candidates_m"
        ]:
            for max_age in config["memory_search"]["max_unobserved_age_candidates_s"]:
                memory_config = _memory_config(
                    context,
                    min_candidate_frames=min_frames,
                    max_candidate_position_spread_m=spread,
                    max_unobserved_age_s=max_age,
                )
                records, leakage = replay_object_memory(
                    frames,
                    calibration=calibration,
                    memory_config=memory_config,
                )
                summary = summarize_object_memory_replay(
                    records,
                    catastrophic_world_xyz_error_m=catastrophic,
                    reset_leakage_count=leakage,
                )
                safety_passed = all(
                    summary[name] == 0
                    for name in (
                        "accepted_update_unsafe_count",
                        "memory_catastrophic_count",
                        "post_pregrasp_memory_valid_count",
                        "memory_only_contact_authorization_count",
                        "post_pregrasp_contact_authorization_count",
                        "episode_reset_leakage_count",
                    )
                )
                report = {
                    "min_candidate_frames": min_frames,
                    "max_candidate_position_spread_m": spread,
                    "max_unobserved_age_s": max_age,
                    "safety_passed": safety_passed,
                    "current_candidate_valid_count": summary[
                        "current_candidate_valid_count"
                    ],
                    "object_memory_valid_count": summary["object_memory_valid_count"],
                    "paired_navigation_availability_gain_count": summary[
                        "paired_navigation_availability_gain_count"
                    ],
                    "memory_valid_while_gt_unobservable_count": summary[
                        "memory_valid_while_gt_unobservable_count"
                    ],
                    "memory_catastrophic_count": summary["memory_catastrophic_count"],
                    "accepted_update_unsafe_count": summary[
                        "accepted_update_unsafe_count"
                    ],
                }
                reports.append(report)
                if not safety_passed:
                    continue
                score = (
                    float(summary["paired_navigation_availability_gain_count"]),
                    float(summary["memory_valid_while_gt_unobservable_count"]),
                    float(summary["object_memory_valid_count"]),
                    float(min_frames),
                    -float(spread),
                    -float(max_age),
                )
                if best is None or score > best[0]:
                    best = (score, memory_config, records)
    if best is None:
        raise RuntimeError("E018-P0 validation 参数网格没有通过零容忍 safety gate 的候选")
    return best[1], reports, best[2]


def _source_identity(repository_root: Path) -> dict[str, Any]:
    safe_repository = str(repository_root.resolve())
    git = ("git", "-c", f"safe.directory={safe_repository}")
    file_hashes = {
        relative: file_sha256(repository_root / relative) for relative in _SOURCE_FILES
    }
    commit = subprocess.run(
        [*git, "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [*git, "status", "--porcelain", "--untracked-files=all"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "git_commit": commit,
        "worktree_clean": not status,
        "git_status": status,
        "source_file_sha256": file_hashes,
        "identity_sha256": canonical_sha256(
            {"git_commit": commit, "git_status": status, "source_file_sha256": file_hashes}
        ),
    }


def run_e018_p0_development(
    *,
    deployable_root: str | Path,
    label_root: str | Path,
    parent_config_path: str | Path,
    training_output: str | Path,
    goal_calibration_root: str | Path,
    config_path: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """运行 recorded validation；函数签名故意不接受 split/test 参数。"""

    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"E018-P0 development output 已存在: {output}")
    repository = Path(repository_root)
    context = _load_context(
        config_path=Path(config_path),
        parent_config_path=Path(parent_config_path),
        training_output=Path(training_output),
        goal_calibration_root=Path(goal_calibration_root),
    )
    source_identity = _source_identity(repository)
    validation_identity = _validation_identity(
        deployable_root=Path(deployable_root),
        label_root=Path(label_root),
        expected_trajectories=context.config["source"]["expected_trajectory_count"],
        expected_frames=context.config["source"]["expected_frame_count"],
    )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _atomic_json(
        output / "run_state.json",
        {
            "version": E018_P0_RESULT_VERSION,
            "status": "in-progress",
            "split": "val",
            "test_split_status": "prohibited-unread",
        },
    )
    rows, frames, perception = _collect_validation(
        context=context,
        deployable_root=Path(deployable_root),
        label_root=Path(label_root),
    )
    pregrasp_rows = [row for row in rows if row["safety"]["pregrasp_window_open"]]
    calibration = calibrate_object_write_threshold(
        scores=[float(row["write_evidence"]["score"]) for row in pregrasp_rows],
        structurally_eligible=np.asarray(
            [bool(row["write_evidence"]["structurally_eligible"]) for row in pregrasp_rows],
            dtype=np.bool_,
        ),
        oracle_safe=np.asarray(
            [bool(row["oracle_safe_measurement"]) for row in pregrasp_rows],
            dtype=np.bool_,
        ),
    )
    selected_config, candidate_reports, selected_records = _select_memory_rules(
        context,
        frames,
        calibration,
    )
    replay_summary = summarize_object_memory_replay(
        selected_records,
        catastrophic_world_xyz_error_m=context.config["write_calibration"][
            "catastrophic_world_xyz_error_m"
        ],
    )
    rules = {
        "version": E018_P0_DEVELOPMENT_VERSION,
        "status": "development-selected-not-frozen-for-test",
        "config_sha256": context.config_sha256,
        "validation_data_identity_sha256": validation_identity,
        "checkpoint_sha256": context.checkpoint_sha256,
        "checkpoint_parameter_sha256": context.checkpoint_parameter_sha256,
        "checkpoint_provenance_sha256": context.checkpoint_provenance_sha256,
        "goal_memory_rules_sha256": context.config["parent"]["goal_memory_rules_sha256"],
        "object_write_calibration": calibration.to_dict(),
        "object_memory_config": {
            "max_unobserved_age_s": selected_config.max_unobserved_age_s,
            "max_innovation_m": selected_config.max_innovation_m,
            "max_position_std_m": selected_config.max_position_std_m,
            "min_candidate_frames": selected_config.min_candidate_frames,
            "max_candidate_gap_s": selected_config.max_candidate_gap_s,
            "max_candidate_position_spread_m": (
                selected_config.max_candidate_position_spread_m
            ),
            "max_sensor_skew_s": selected_config.max_sensor_skew_s,
            "expected_source_camera": selected_config.expected_source_camera,
            "expected_source_model_identity": (
                selected_config.expected_source_model_identity
            ),
            "require_covariance": selected_config.require_covariance,
            "covariance_growth_m2_per_s": (
                selected_config.covariance_growth_m2_per_s
            ),
        },
        "test_split_status": "prohibited-unread",
        "test_model_forward_count": 0,
        "test_privileged_label_file_read_count": 0,
        "actuation_allowed": False,
        "camera_motion_allowed": False,
    }
    rules["rules_sha256"] = canonical_sha256(rules)
    safety_gate_names = (
        "accepted_update_unsafe_count",
        "memory_catastrophic_count",
        "post_pregrasp_memory_valid_count",
        "memory_only_contact_authorization_count",
        "post_pregrasp_contact_authorization_count",
        "episode_reset_leakage_count",
    )
    all_safety_gates_passed = all(
        replay_summary[name] == 0 for name in safety_gate_names
    )
    selected_age_at_search_upper_bound = math.isclose(
        selected_config.max_unobserved_age_s,
        max(context.config["memory_search"]["max_unobserved_age_candidates_s"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    physical_occlusion_bridge_demonstrated = bool(
        replay_summary["memory_valid_while_gt_unobservable_count"] > 0
    )
    ready_for_formal_preregistration = bool(
        all_safety_gates_passed
        and physical_occlusion_bridge_demonstrated
        and not selected_age_at_search_upper_bound
    )
    summary = {
        "version": E018_P0_RESULT_VERSION,
        "status": "complete-development-only",
        "split": "val",
        "source_usage": context.config["source"]["usage"],
        "trajectory_count": context.config["source"]["expected_trajectory_count"],
        "validation_data_identity_sha256": validation_identity,
        "config_sha256": context.config_sha256,
        "source_identity": source_identity,
        "checkpoint_sha256": context.checkpoint_sha256,
        "goal_memory_rules_sha256": context.config["parent"]["goal_memory_rules_sha256"],
        "gpu": torch.cuda.get_device_name(torch.device("cuda")),
        "perception": perception,
        "object_write_calibration": calibration.to_dict(),
        "candidate_memory_grid": candidate_reports,
        "selected_rules_sha256": rules["rules_sha256"],
        "object_memory_replay": replay_summary,
        "safety_gates": {
            name: {"actual": replay_summary[name], "required": 0, "passed": replay_summary[name] == 0}
            for name in safety_gate_names
        },
        "all_development_safety_gates_passed": all_safety_gates_passed,
        "development_efficacy": {
            "physical_occlusion_bridge_demonstrated": (
                physical_occlusion_bridge_demonstrated
            ),
            "selected_age_at_search_upper_bound": selected_age_at_search_upper_bound,
            "memory_age_identified_by_this_dataset": (
                not selected_age_at_search_upper_bound
            ),
            "ready_for_formal_preregistration": ready_for_formal_preregistration,
            "conclusion": (
                "development-evidence-ready-for-fresh-preregistration"
                if ready_for_formal_preregistration
                else "safety-passed-but-core-efficacy-and-age-not-identified"
            ),
        },
        "test_split_status": "prohibited-unread",
        "test_model_forward_count": 0,
        "test_privileged_label_file_read_count": 0,
        "actuation_allowed": False,
        "camera_motion_allowed": False,
        "formal_claim_allowed": False,
    }
    _atomic_jsonl(output / "object_prediction_rows.jsonl", rows)
    _atomic_jsonl(
        output / "object_memory_replay.jsonl",
        [record.to_dict() for record in selected_records],
    )
    _atomic_json(output / "selected_development_rules.json", rules)
    _atomic_json(output / "config_snapshot.json", context.config)
    _atomic_json(output / "summary.json", summary)
    artifact_names = (
        "object_prediction_rows.jsonl",
        "object_memory_replay.jsonl",
        "selected_development_rules.json",
        "config_snapshot.json",
        "summary.json",
    )
    receipt = {
        "version": E018_P0_RESULT_VERSION,
        "status": "complete-development-only",
        "files": {name: file_sha256(output / name) for name in artifact_names},
        "test_split_status": "prohibited-unread",
        "formal_claim_allowed": False,
    }
    _atomic_json(output / "receipt.json", receipt)
    _atomic_json(
        output / "run_state.json",
        {
            "version": E018_P0_RESULT_VERSION,
            "status": "complete-development-only",
            "split": "val",
            "test_split_status": "prohibited-unread",
        },
    )
    return summary


__all__ = ["E018_P0_DEVELOPMENT_VERSION", "E018_P0_RESULT_VERSION", "run_e018_p0_development"]
