"""运行 E014：冻结 val 诊断规则，并分解 E013 held-out long tail。"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_vla.contracts import RobotSpec
from robot_vla.observation import rotation_6d_to_matrix
from robot_vla.precision.checkpoint import PrecisionCheckpointRole, load_precision_checkpoint
from robot_vla.precision.data import (
    PrecisionRGBDataset,
    audit_precision_dataset,
    canonical_sha256,
    file_sha256,
)
from robot_vla.precision.geometry import (
    normalized_uv_to_pixel,
    project_base_point_to_normalized_uv,
)
from robot_vla.precision.held_out import PrecisionConfidenceCalibration
from robot_vla.precision.outliers import (
    E014_DIAGNOSTIC_VERSION,
    FAILURE_FAMILY_ORDER,
    FAILURE_TAXONOMY_PRIORITY,
    aggregate_prediction_rows,
    assert_public_payload_safe,
    classify_outlier,
    derive_validation_rules,
    failure_family,
    failure_family_counts,
    geometry_conditioning,
    local_peak_nms,
    point_distance,
    point_inside_mask,
    sample_fingerprint,
    semantic_distance_features,
    taxonomy_counts,
    temporal_alignment_features,
)
from robot_vla.precision.training import (
    PrecisionExperimentConfig,
    _build_loader,
    _to_device,
    load_precision_experiment_config,
    source_tree_sha256,
)


@dataclass(frozen=True)
class _FrozenContext:
    config: PrecisionExperimentConfig
    model: Any
    checkpoint_sha256: str
    parameter_state_sha256: str
    provenance_sha256: str
    data_identity_sha256: str
    diagnostic_source_tree_sha256: str
    training_source_tree_sha256: str
    upstream_evaluation_source_tree_sha256: str
    upstream_receipt_sha256: str
    calibration: PrecisionConfidenceCalibration
    canonical_held_out: dict[str, Any]


@dataclass
class _SampleArtifact:
    heatmap_probability: np.ndarray
    predicted_masks: np.ndarray


@dataclass
class _CollectionResult:
    rows: list[dict[str, Any]]
    artifacts: dict[int, _SampleArtifact]
    invalid_backprojection_count: int


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
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


def _base_from_camera(audit: dict[str, Any]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_6d_to_matrix(audit["wrist_camera_rotation_6d_base"])
    transform[:3, 3] = audit["wrist_camera_position_base_m"]
    return transform


def _frozen_context(args: argparse.Namespace) -> _FrozenContext:
    repository_root = Path(args.repository_root).resolve()
    diagnostic_source = source_tree_sha256(repository_root)
    config = load_precision_experiment_config(args.config)
    audit = audit_precision_dataset(
        args.deployable_root,
        args.label_root,
        RobotSpec(),
        write_artifact=False,
    )
    if not audit.passed:
        raise RuntimeError("E014 Dataset audit 未通过")

    training_root = Path(args.training_output)
    training_receipt = _read_json(
        training_root / "checkpoint_receipt.json",
        name="E013 checkpoint receipt",
    )
    checkpoint_receipt = training_receipt.get("checkpoint")
    if not isinstance(checkpoint_receipt, dict):
        raise TypeError("E013 checkpoint receipt 缺少 checkpoint object")
    checkpoint_sha256 = str(checkpoint_receipt["checkpoint_sha256"])
    provenance_sha256 = str(checkpoint_receipt["provenance_sha256"])
    checkpoint_path = training_root / "precision-formal.pt"
    loaded = load_precision_checkpoint(
        checkpoint_path,
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_provenance_sha256=provenance_sha256,
        expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
    )
    if loaded.provenance.data_identity_sha256 != audit.dataset_identity_sha256:
        raise RuntimeError("E014 checkpoint/Dataset identity 漂移")
    if loaded.provenance.training_config_sha256 != config.sha256:
        raise RuntimeError("E014 checkpoint/config identity 漂移")
    if str(training_receipt["data_identity_sha256"]) != audit.dataset_identity_sha256:
        raise RuntimeError("E013 training receipt/Dataset identity 漂移")
    if str(training_receipt["training_config_sha256"]) != config.sha256:
        raise RuntimeError("E013 training receipt/config identity 漂移")

    held_out_root = Path(args.held_out_output)
    calibration_payload = _read_json(
        held_out_root / "confidence_calibration.json",
        name="E013 frozen confidence calibration",
    )
    canonical_held_out = _read_json(
        held_out_root / "held_out_evaluation.json",
        name="E013 canonical held-out metrics",
    )
    upstream_receipt_path = held_out_root / "receipt.json"
    upstream_receipt = _read_json(upstream_receipt_path, name="E013 held-out receipt")
    calibration = PrecisionConfidenceCalibration(**calibration_payload)
    if calibration.sha256 != str(canonical_held_out["calibration_sha256"]):
        raise RuntimeError("E013 calibration identity 漂移")
    for actual, expected, name in (
        (calibration.checkpoint_sha256, checkpoint_sha256, "calibration checkpoint"),
        (calibration.data_identity_sha256, audit.dataset_identity_sha256, "calibration Dataset"),
        (calibration.training_config_sha256, config.sha256, "calibration config"),
        (str(canonical_held_out["checkpoint_sha256"]), checkpoint_sha256, "held-out checkpoint"),
        (
            str(canonical_held_out["data_identity_sha256"]),
            audit.dataset_identity_sha256,
            "held-out Dataset",
        ),
        (
            str(upstream_receipt["training_source_tree_sha256"]),
            loaded.provenance.source_tree_sha256,
            "held-out training source",
        ),
    ):
        if actual != expected:
            raise RuntimeError(f"E014 {name} identity 漂移")
    if calibration.temperature not in config.held_out.temperature_grid:
        raise RuntimeError("E013 frozen temperature 不在预注册 grid")
    if calibration.calibration_split != "val" or canonical_held_out.get("split") != "test":
        raise RuntimeError("E013 calibration/evaluation split 漂移")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("E014 正式诊断要求 BF16 CUDA")
    return _FrozenContext(
        config=config,
        model=loaded.model,
        checkpoint_sha256=checkpoint_sha256,
        parameter_state_sha256=loaded.receipt.parameter_state_sha256,
        provenance_sha256=provenance_sha256,
        data_identity_sha256=audit.dataset_identity_sha256,
        diagnostic_source_tree_sha256=diagnostic_source,
        training_source_tree_sha256=loaded.provenance.source_tree_sha256,
        upstream_evaluation_source_tree_sha256=str(
            upstream_receipt["evaluation_source_tree_sha256"]
        ),
        upstream_receipt_sha256=file_sha256(upstream_receipt_path),
        calibration=calibration,
        canonical_held_out=canonical_held_out,
    )


def _refined_peak(
    pixel_uv: tuple[float, float],
    offsets: np.ndarray,
) -> list[float]:
    x = int(pixel_uv[0])
    y = int(pixel_uv[1])
    return [float(x + offsets[0, y, x]), float(y + offsets[1, y, x])]


def _mask_iou(predicted: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.sum(predicted & target))
    union = int(np.sum(predicted | target))
    return 1.0 if union == 0 else float(intersection / union)


def _gt_pixel(
    labels: Any,
    *,
    timestep: int,
    keypoint_index: int,
    image_size_hw: tuple[int, int],
) -> list[float] | None:
    if not bool(labels.keypoint_projection_valid[timestep, keypoint_index]):
        return None
    return (
        normalized_uv_to_pixel(
            labels.normalized_uv[timestep, keypoint_index],
            image_size_hw,
        )
        .astype(float)
        .tolist()
    )


def _collect_rows(
    *,
    model: Any,
    dataset: PrecisionRGBDataset,
    config: PrecisionExperimentConfig,
    temperature: float,
    confidence_threshold: float,
    nms_radius_px: int,
    keep_artifacts: bool,
) -> _CollectionResult:
    device = torch.device("cuda")
    loader = _build_loader(
        dataset,
        batch_size=config.formal_training.batch_size,
        shuffle=False,
        seed=0,
        num_workers=0,
    )
    model = model.to(device)
    model.eval()
    rows: list[dict[str, Any]] = []
    artifacts: dict[int, _SampleArtifact] = {}
    sample_offset = 0
    invalid_backprojection = 0
    keypoint_names = ("object_center", "goal_center")
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                output = model(
                    batch["image"],
                    batch["structured_state"],
                    batch["geometric_motion"],
                )
            decoded = output.decode_for_control(temperature=temperature)
            height, width = batch["image"].shape[-2:]
            image_size_hw = (int(height), int(width))
            logits = output.heatmap_logits.detach().float().cpu().numpy()
            probability = (
                torch.softmax(
                    output.heatmap_logits.detach().float().reshape(logits.shape[0], 2, -1)
                    / float(temperature),
                    dim=-1,
                )
                .reshape(logits.shape)
                .cpu()
                .numpy()
            )
            offsets = output.subpixel_offsets.detach().float().cpu().numpy()
            predicted_masks = (output.mask_logits.detach() > 0.0).cpu().numpy()
            predicted_uv = decoded.keypoints.normalized_uv.detach().cpu().numpy()
            predicted_pixel = decoded.keypoints.pixel_uv.detach().cpu().numpy()
            visibility_probability = decoded.visibility_probability.detach().cpu().numpy()
            projection_probability = decoded.projection_validity_probability.detach().cpu().numpy()
            sigma = decoded.keypoint_sigma_px.detach().cpu().numpy()
            entropy = decoded.keypoints.normalized_entropy.detach().cpu().numpy()
            for batch_index, audit in enumerate(raw_batch["audit"]):
                dataset_index = sample_offset + batch_index
                trajectory_id = str(audit["trajectory_id"])
                timestep = int(audit["timestep"])
                meta = dataset.label_by_trajectory[trajectory_id]
                labels = dataset.label_store.get(meta)
                if int(labels.source_timestep[timestep]) != timestep:
                    source_timestep_mismatch = True
                else:
                    source_timestep_mismatch = False
                gt_masks = (
                    labels.object_mask[timestep],
                    labels.goal_mask[timestep],
                )
                gt_pixels = (
                    _gt_pixel(
                        labels,
                        timestep=timestep,
                        keypoint_index=0,
                        image_size_hw=image_size_hw,
                    ),
                    _gt_pixel(
                        labels,
                        timestep=timestep,
                        keypoint_index=1,
                        image_size_hw=image_size_hw,
                    ),
                )
                semantic = semantic_distance_features(
                    predicted_object_px=predicted_pixel[batch_index, 0],
                    predicted_goal_px=predicted_pixel[batch_index, 1],
                    gt_object_px=gt_pixels[0],
                    gt_goal_px=gt_pixels[1],
                )
                positions = (
                    labels.object_position_base_m[timestep],
                    labels.goal_position_base_m[timestep],
                )
                object_goal_distance_m = float(np.linalg.norm(positions[0][:2] - positions[1][:2]))
                transform = _base_from_camera(audit)
                intrinsic = np.asarray(audit["intrinsic_wrist_cv"], dtype=np.float64)
                if keep_artifacts:
                    artifacts[dataset_index] = _SampleArtifact(
                        heatmap_probability=probability[batch_index].astype(np.float16),
                        predicted_masks=predicted_masks[batch_index].copy(),
                    )

                for keypoint_index, keypoint_name in enumerate(keypoint_names):
                    current_probability = probability[batch_index, keypoint_index]
                    peak_diagnostics = local_peak_nms(
                        current_probability,
                        radius_px=nms_radius_px,
                        max_reported_peaks=height * width,
                    )
                    if not peak_diagnostics.peaks:
                        raise RuntimeError("E014 heatmap 没有 local peak")
                    refined_peaks = [
                        _refined_peak(
                            peak.pixel_uv,
                            offsets[batch_index, keypoint_index],
                        )
                        for peak in peak_diagnostics.peaks
                    ]
                    top1 = peak_diagnostics.peaks[0]
                    top1_refined = refined_peaks[0]
                    top2 = peak_diagnostics.peaks[1] if len(peak_diagnostics.peaks) > 1 else None
                    top2_refined = refined_peaks[1] if len(refined_peaks) > 1 else None
                    top3 = peak_diagnostics.peaks[2] if len(peak_diagnostics.peaks) > 2 else None
                    top3_refined = refined_peaks[2] if len(refined_peaks) > 2 else None
                    soft_pixel = predicted_pixel[batch_index, keypoint_index].astype(float).tolist()
                    gt_pixel = gt_pixels[keypoint_index]
                    projection_valid = bool(
                        labels.keypoint_projection_valid[timestep, keypoint_index]
                    )
                    visible = bool(labels.keypoint_visible[timestep, keypoint_index])
                    keypoint_valid = bool(projection_valid and visible)
                    pixel_error = None if gt_pixel is None else point_distance(soft_pixel, gt_pixel)
                    top1_error = (
                        None if gt_pixel is None else point_distance(top1_refined, gt_pixel)
                    )
                    soft_minus_top1 = (
                        None if top1_error is None else float(pixel_error - top1_error)
                    )

                    previous_gt = (
                        None
                        if timestep == 0
                        else _gt_pixel(
                            labels,
                            timestep=timestep - 1,
                            keypoint_index=keypoint_index,
                            image_size_hw=image_size_hw,
                        )
                    )
                    next_gt = (
                        None
                        if timestep + 1 >= labels.num_steps
                        else _gt_pixel(
                            labels,
                            timestep=timestep + 1,
                            keypoint_index=keypoint_index,
                            image_size_hw=image_size_hw,
                        )
                    )
                    temporal = temporal_alignment_features(
                        prediction_px=soft_pixel,
                        current_gt_px=gt_pixel,
                        previous_gt_px=previous_gt,
                        next_gt_px=next_gt,
                    )
                    try:
                        geometry = geometry_conditioning(
                            normalized_uv=predicted_uv[batch_index, keypoint_index],
                            intrinsic_cv=intrinsic,
                            base_from_camera_cv=transform,
                            image_size_hw=image_size_hw,
                            plane_base_z_m=float(positions[keypoint_index][2]),
                        )
                        predicted_world = np.asarray(
                            geometry["predicted_world_point_base_m"],
                            dtype=np.float64,
                        )
                    except ValueError:
                        invalid_backprojection += 1
                        geometry = {
                            "camera_position_base_m": transform[:3, 3].astype(float).tolist(),
                            "camera_viewing_direction_base": transform[:3, 2]
                            .astype(float)
                            .tolist(),
                            "ray_direction_base": None,
                            "unit_ray_direction_base": None,
                            "n_dot_ray": None,
                            "abs_n_dot_unit_ray": None,
                            "ray_parameter": None,
                            "physical_ray_distance_m": None,
                            "predicted_world_point_base_m": None,
                            "local_jacobian_xy_m_per_px": None,
                            "jacobian_sigma_max_mm_per_px": None,
                        }
                        predicted_world = None
                    world_error = None
                    if keypoint_valid and predicted_world is not None:
                        world_error = float(
                            np.linalg.norm(
                                predicted_world[:2]
                                - positions[keypoint_index][:2].astype(np.float64)
                            )
                        )

                    oracle_roundtrip_error = None
                    oracle_roundtrip_failed = False
                    if projection_valid and gt_pixel is not None:
                        try:
                            oracle_uv = project_base_point_to_normalized_uv(
                                positions[keypoint_index],
                                intrinsic,
                                transform,
                                image_size_hw,
                            )
                            oracle_pixel = normalized_uv_to_pixel(oracle_uv, image_size_hw)
                            oracle_roundtrip_error = point_distance(oracle_pixel, gt_pixel)
                        except ValueError:
                            oracle_roundtrip_failed = True

                    other_index = 1 - keypoint_index
                    gt_mask = gt_masks[keypoint_index]
                    other_mask = gt_masks[other_index]
                    gt_inside_own = (
                        None
                        if gt_pixel is None or not visible
                        else point_inside_mask(gt_pixel, gt_mask)
                    )
                    gt_inside_other = (
                        False
                        if gt_pixel is None
                        or not bool(labels.keypoint_visible[timestep, other_index])
                        else point_inside_mask(gt_pixel, other_mask)
                    )
                    soft_inside_own = point_inside_mask(soft_pixel, gt_mask)
                    soft_inside_other = bool(
                        labels.keypoint_visible[timestep, other_index]
                        and point_inside_mask(soft_pixel, other_mask)
                    )
                    peak_inside_other = bool(
                        labels.keypoint_visible[timestep, other_index]
                        and point_inside_mask(top1_refined, other_mask)
                    )
                    x_gt, y_gt = gt_pixel if gt_pixel is not None else (None, None)
                    edge_distance = (
                        None
                        if gt_pixel is None
                        else float(
                            min(
                                x_gt + 0.5,
                                y_gt + 0.5,
                                width - 0.5 - x_gt,
                                height - 0.5 - y_gt,
                            )
                        )
                    )
                    score = float(
                        min(
                            visibility_probability[batch_index, keypoint_index],
                            projection_probability[batch_index],
                        )
                    )
                    accepted = bool(score >= confidence_threshold)
                    top2_ratio = (
                        None if top2 is None else float(top2.value / max(top1.value, 1e-30))
                    )
                    top1_top2_logit_gap = None
                    if top2 is not None:
                        x1, y1 = map(int, top1.pixel_uv)
                        x2, y2 = map(int, top2.pixel_uv)
                        top1_top2_logit_gap = float(
                            logits[batch_index, keypoint_index, y1, x1]
                            - logits[batch_index, keypoint_index, y2, x2]
                        )
                    nearest_peak_distance = min(
                        point_distance(soft_pixel, peak) for peak in refined_peaks
                    )
                    radial_sigma = float(np.linalg.norm(sigma[batch_index, keypoint_index]))
                    entropy_nats = float(
                        entropy[batch_index, keypoint_index] * math.log(height * width)
                    )
                    observed_scale = (
                        None
                        if world_error is None or pixel_error is None or pixel_error <= 1e-12
                        else float(world_error * 1000.0 / pixel_error)
                    )
                    semantic_margin = (
                        semantic["object_semantic_margin_px"]
                        if keypoint_index == 0
                        else semantic["goal_semantic_margin_px"]
                    )
                    row: dict[str, Any] = {
                        "schema_version": E014_DIAGNOSTIC_VERSION,
                        "sample_fingerprint": sample_fingerprint(
                            trajectory_id=trajectory_id,
                            dataset_index=dataset_index,
                            timestep=timestep,
                            keypoint_type=keypoint_name,
                        ),
                        "trajectory_id": trajectory_id,
                        "scene_id": str(audit["scene_id"]),
                        "dataset_index": dataset_index,
                        "timestep": timestep,
                        "timestamp_s": float(audit["timestamp_s"]),
                        "keypoint_type": keypoint_name,
                        "gt_visible": visible,
                        "gt_projection_valid": projection_valid,
                        "gt_keypoint_valid": keypoint_valid,
                        "label_source_timestep_mismatch": source_timestep_mismatch,
                        "gt_normalized_uv": (
                            None
                            if not projection_valid
                            else labels.normalized_uv[timestep, keypoint_index]
                            .astype(float)
                            .tolist()
                        ),
                        "predicted_normalized_uv": predicted_uv[batch_index, keypoint_index]
                        .astype(float)
                        .tolist(),
                        "gt_pixel_uv": gt_pixel,
                        "predicted_pixel_uv": soft_pixel,
                        "pixel_error_px": pixel_error,
                        "gt_world_xy_m": positions[keypoint_index][:2].astype(float).tolist(),
                        "predicted_world_xy_m": (
                            None
                            if predicted_world is None
                            else predicted_world[:2].astype(float).tolist()
                        ),
                        "world_xy_error_m": world_error,
                        "object_goal_distance_px": semantic["object_goal_distance_px"],
                        "object_goal_distance_m": object_goal_distance_m,
                        "d_pred_object_to_gt_object_px": semantic["d_pred_object_to_gt_object_px"],
                        "d_pred_object_to_gt_goal_px": semantic["d_pred_object_to_gt_goal_px"],
                        "d_pred_goal_to_gt_goal_px": semantic["d_pred_goal_to_gt_goal_px"],
                        "d_pred_goal_to_gt_object_px": semantic["d_pred_goal_to_gt_object_px"],
                        "object_semantic_margin_px": semantic["object_semantic_margin_px"],
                        "goal_semantic_margin_px": semantic["goal_semantic_margin_px"],
                        "semantic_margin_px": semantic_margin,
                        "paired_object_goal_swap": semantic["paired_object_goal_swap"],
                        "gt_inside_own_mask": gt_inside_own,
                        "gt_inside_other_mask": gt_inside_other,
                        "peak_inside_other_gt_mask": peak_inside_other,
                        "softargmax_inside_own_gt_mask": soft_inside_own,
                        "softargmax_inside_other_gt_mask": soft_inside_other,
                        "gt_mask_area_fraction": float(np.mean(gt_mask)),
                        "gt_edge_distance_px": edge_distance,
                        "predicted_mask_iou": _mask_iou(
                            predicted_masks[batch_index, keypoint_index],
                            gt_mask,
                        ),
                        "top1_peak_pixel_uv": top1_refined,
                        "top1_probability": float(top1.value),
                        "top2_peak_pixel_uv": top2_refined,
                        "top2_probability": None if top2 is None else float(top2.value),
                        "top3_peak_pixel_uv": top3_refined,
                        "top3_probability": None if top3 is None else float(top3.value),
                        "top1_top2_peak_distance_px": (
                            None
                            if top2_refined is None
                            else point_distance(top1_refined, top2_refined)
                        ),
                        "top1_top2_logit_gap": top1_top2_logit_gap,
                        "top2_top1_probability_ratio": top2_ratio,
                        "local_maxima_count": peak_diagnostics.local_maxima_count,
                        "separated_local_maxima_count": (
                            peak_diagnostics.separated_local_maxima_count
                        ),
                        "softargmax_to_top1_distance_px": point_distance(
                            soft_pixel,
                            top1_refined,
                        ),
                        "softargmax_to_nearest_peak_distance_px": nearest_peak_distance,
                        "top1_argmax_pixel_error_px": top1_error,
                        "softargmax_minus_top1_error_px": soft_minus_top1,
                        "normalized_entropy": float(entropy[batch_index, keypoint_index]),
                        "entropy_nats": entropy_nats,
                        "visibility_probability": float(
                            visibility_probability[batch_index, keypoint_index]
                        ),
                        "projection_validity_probability": float(
                            projection_probability[batch_index]
                        ),
                        "confidence_score": score,
                        "confidence_accepted": accepted,
                        "keypoint_sigma_x_px": float(sigma[batch_index, keypoint_index, 0]),
                        "keypoint_sigma_y_px": float(sigma[batch_index, keypoint_index, 1]),
                        "radial_sigma_px": radial_sigma,
                        "oracle_roundtrip_error_px": oracle_roundtrip_error,
                        "oracle_roundtrip_failed": oracle_roundtrip_failed,
                        "observed_world_mm_per_pixel_error": observed_scale,
                        **temporal,
                        **geometry,
                    }
                    rows.append(row)
            sample_offset += int(logits.shape[0])
    if sample_offset != len(dataset):
        raise RuntimeError("E014 DataLoader 未完整覆盖 split")
    if len(rows) != len(dataset) * 2:
        raise RuntimeError("E014 必须为每个 sample 保存两行预测")
    return _CollectionResult(
        rows=rows,
        artifacts=artifacts,
        invalid_backprojection_count=invalid_backprojection,
    )


def _rules_document(
    *,
    context: _FrozenContext,
    rules: dict[str, Any],
    validation_summary: dict[str, Any],
) -> dict[str, Any]:
    document = {
        "version": E014_DIAGNOSTIC_VERSION,
        "rules": rules,
        "identities": {
            "checkpoint_sha256": context.checkpoint_sha256,
            "parameter_state_sha256": context.parameter_state_sha256,
            "checkpoint_provenance_sha256": context.provenance_sha256,
            "data_identity_sha256": context.data_identity_sha256,
            "training_source_tree_sha256": context.training_source_tree_sha256,
            "upstream_evaluation_source_tree_sha256": (
                context.upstream_evaluation_source_tree_sha256
            ),
            "upstream_held_out_receipt_sha256": context.upstream_receipt_sha256,
            "diagnostic_source_tree_sha256": context.diagnostic_source_tree_sha256,
            "training_config_sha256": context.config.sha256,
        },
        "frozen_calibration": {
            "calibration_sha256": context.calibration.sha256,
            "temperature": context.calibration.temperature,
            "confidence_threshold": context.calibration.confidence_threshold,
            "confidence_semantics": context.calibration.confidence_semantics,
        },
        "validation_summary": validation_summary,
        "test_accessed": False,
    }
    document["rules_sha256"] = canonical_sha256(document)
    return document


def _freeze_rules(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.rules_output)
    if output.exists():
        raise FileExistsError(f"E014 rules 已存在，拒绝覆盖: {output}")
    context = _frozen_context(args)
    dataset = PrecisionRGBDataset(
        args.deployable_root,
        args.label_root,
        "val",
        cache_size=context.config.formal_training.cache_size,
    )
    if len(dataset) != context.calibration.sample_count:
        raise RuntimeError("E014 val sample count 与 frozen calibration 不一致")
    nms_radius = math.ceil(3.0 * context.config.formal_training.heatmap_sigma_px)
    collection = _collect_rows(
        model=context.model,
        dataset=dataset,
        config=context.config,
        temperature=context.calibration.temperature,
        confidence_threshold=context.calibration.confidence_threshold,
        nms_radius_px=nms_radius,
        keep_artifacts=False,
    )
    if collection.invalid_backprojection_count != 0:
        raise RuntimeError("E014 val 出现 invalid backprojection")
    valid_count = sum(bool(row["gt_keypoint_valid"]) for row in collection.rows)
    if valid_count != context.calibration.valid_keypoint_count:
        raise RuntimeError("E014 val valid keypoint count 与 frozen calibration 不一致")
    rules = derive_validation_rules(
        collection.rows,
        heatmap_sigma_px=context.config.formal_training.heatmap_sigma_px,
    )
    document = _rules_document(
        context=context,
        rules=rules,
        validation_summary=aggregate_prediction_rows(collection.rows),
    )
    _atomic_json(output, document)
    return {
        "status": "rules-frozen",
        "rules_sha256": document["rules_sha256"],
        "validation_samples": len(dataset),
        "validation_prediction_rows": len(collection.rows),
        "validation_valid_keypoints": valid_count,
    }


def _load_and_verify_rules(path: Path, context: _FrozenContext) -> dict[str, Any]:
    document = _read_json(path, name="E014 frozen rules")
    expected_sha256 = str(document.get("rules_sha256", ""))
    unsigned = {key: value for key, value in document.items() if key != "rules_sha256"}
    if canonical_sha256(unsigned) != expected_sha256:
        raise RuntimeError("E014 rules SHA-256 漂移")
    if document.get("version") != E014_DIAGNOSTIC_VERSION:
        raise RuntimeError("E014 rules document version 漂移")
    if document.get("test_accessed") is not False:
        raise RuntimeError("E014 rules 必须在 test 前封存")
    identities = document.get("identities")
    calibration = document.get("frozen_calibration")
    if not isinstance(identities, dict) or not isinstance(calibration, dict):
        raise TypeError("E014 rules identity/calibration 缺失")
    expected_identities = {
        "checkpoint_sha256": context.checkpoint_sha256,
        "parameter_state_sha256": context.parameter_state_sha256,
        "checkpoint_provenance_sha256": context.provenance_sha256,
        "data_identity_sha256": context.data_identity_sha256,
        "training_source_tree_sha256": context.training_source_tree_sha256,
        "upstream_evaluation_source_tree_sha256": (context.upstream_evaluation_source_tree_sha256),
        "upstream_held_out_receipt_sha256": context.upstream_receipt_sha256,
        "diagnostic_source_tree_sha256": context.diagnostic_source_tree_sha256,
        "training_config_sha256": context.config.sha256,
    }
    if identities != expected_identities:
        raise RuntimeError("E014 rules 上游 identity 与当前正式输入不一致")
    if calibration != {
        "calibration_sha256": context.calibration.sha256,
        "temperature": context.calibration.temperature,
        "confidence_threshold": context.calibration.confidence_threshold,
        "confidence_semantics": context.calibration.confidence_semantics,
    }:
        raise RuntimeError("E014 rules frozen calibration 漂移")
    rules = document.get("rules")
    if not isinstance(rules, dict):
        raise TypeError("E014 rules payload 缺失")
    return document


def _metric_reproduction(
    *,
    aggregate: dict[str, Any],
    context: _FrozenContext,
    invalid_backprojection_count: int,
) -> dict[str, Any]:
    distribution = aggregate["world_error"]["all"]
    actual = {
        "sample_count": aggregate["prediction_row_count"] // 2,
        "valid_keypoint_count": aggregate["valid_keypoint_count"],
        "world_xy_error_p50_m": float(distribution["p50_mm"]) / 1000.0,
        "world_xy_error_p90_m": float(distribution["p90_mm"]) / 1000.0,
        "world_xy_error_max_m": float(distribution["max_mm"]) / 1000.0,
        "accepted_validity_precision": aggregate["confidence"]["accepted_validity_precision"],
        "invalid_backprojection_count": invalid_backprojection_count,
    }
    canonical = context.canonical_held_out
    deltas = {
        name: float(actual[name]) - float(canonical[name])
        for name in (
            "world_xy_error_p50_m",
            "world_xy_error_p90_m",
            "world_xy_error_max_m",
        )
    }
    if actual["sample_count"] != int(canonical["sample_count"]):
        raise RuntimeError("E014 未复现 canonical test sample count")
    if actual["valid_keypoint_count"] != int(canonical["valid_keypoint_count"]):
        raise RuntimeError("E014 未复现 canonical valid keypoint count")
    if invalid_backprojection_count != int(canonical["invalid_backprojection_count"]):
        raise RuntimeError("E014 invalid backprojection count 漂移")
    if abs(deltas["world_xy_error_p50_m"]) > 1e-4:
        raise RuntimeError("E014 p50 与 canonical 结果偏差超过 0.1 mm")
    if abs(deltas["world_xy_error_p90_m"]) > 1e-4:
        raise RuntimeError("E014 p90 与 canonical 结果偏差超过 0.1 mm")
    if abs(deltas["world_xy_error_max_m"]) > 1e-3:
        raise RuntimeError("E014 max 与 canonical 结果偏差超过 1 mm")
    return {
        "passed": True,
        "actual": actual,
        "canonical": {
            "sample_count": int(canonical["sample_count"]),
            "valid_keypoint_count": int(canonical["valid_keypoint_count"]),
            "world_xy_error_p50_m": float(canonical["world_xy_error_p50_m"]),
            "world_xy_error_p90_m": float(canonical["world_xy_error_p90_m"]),
            "world_xy_error_max_m": float(canonical["world_xy_error_max_m"]),
            "legacy_confidence_precision": float(canonical["confidence_precision"]),
            "invalid_backprojection_count": int(canonical["invalid_backprojection_count"]),
        },
        "metric_deltas_m": deltas,
        "hardware_note": "BF16 inference may differ slightly across CUDA GPU architectures.",
    }


def _atomic_text(path: Path, text: str) -> None:
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
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _plot_top20(
    *,
    private_root: Path,
    dataset: PrecisionRGBDataset,
    rows: list[dict[str, Any]],
    top20: list[dict[str, Any]],
    artifacts: dict[int, _SampleArtifact],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = private_root / "top20"
    heatmap_root = private_root / "top20_heatmaps"
    figure_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    heatmap_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    rows_by_sample: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        rows_by_sample.setdefault(int(row["dataset_index"]), {})[str(row["keypoint_type"])] = row
    for rank, worst in enumerate(top20, start=1):
        dataset_index = int(worst["dataset_index"])
        sample = dataset[dataset_index]
        rgb = sample["model_inputs"]["rgb_wrist"]
        gt_masks = sample["supervision"]["mask_targets"] > 0.5
        sample_rows = rows_by_sample[dataset_index]
        artifact = artifacts[dataset_index]
        fingerprint = str(worst["sample_fingerprint"])
        stem = f"rank-{rank:02d}-{fingerprint}"

        with (heatmap_root / f"{stem}.npz").open("wb") as handle:
            np.savez_compressed(
                handle,
                heatmap_probability=artifact.heatmap_probability,
                predicted_masks=artifact.predicted_masks,
            )

        fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title("Wrist RGB: GT × / prediction +")
        colours = {"object_center": "cyan", "goal_center": "magenta"}
        for name, row in sample_rows.items():
            colour = colours[name]
            if row["gt_pixel_uv"] is not None:
                axes[0, 0].scatter(*row["gt_pixel_uv"], marker="x", s=90, linewidths=2.5, c=colour)
            axes[0, 0].scatter(
                *row["predicted_pixel_uv"], marker="+", s=110, linewidths=2.5, c=colour
            )
        axes[0, 0].set_axis_off()

        for keypoint_index, name in enumerate(("object_center", "goal_center")):
            row = sample_rows[name]
            axis = axes[0, keypoint_index + 1]
            axis.imshow(artifact.heatmap_probability[keypoint_index], cmap="magma")
            if row["gt_pixel_uv"] is not None:
                axis.scatter(*row["gt_pixel_uv"], marker="x", s=70, c="lime")
            axis.scatter(*row["predicted_pixel_uv"], marker="+", s=80, c="cyan")
            axis.scatter(
                *row["top1_peak_pixel_uv"], marker="o", s=50, facecolors="none", edgecolors="white"
            )
            if row["top2_peak_pixel_uv"] is not None:
                axis.scatter(
                    *row["top2_peak_pixel_uv"],
                    marker="s",
                    s=45,
                    facecolors="none",
                    edgecolors="yellow",
                )
            axis.set_title(f"{name}: heatmap / soft / top-1 / top-2")
            axis.set_axis_off()

        for keypoint_index, name in enumerate(("object_center", "goal_center")):
            axis = axes[1, keypoint_index]
            axis.imshow(rgb)
            if np.any(gt_masks[keypoint_index]):
                axis.contour(
                    gt_masks[keypoint_index], levels=[0.5], colors=["lime"], linewidths=1.5
                )
            if np.any(artifact.predicted_masks[keypoint_index]):
                axis.contour(
                    artifact.predicted_masks[keypoint_index],
                    levels=[0.5],
                    colors=["red"],
                    linewidths=1.5,
                )
            axis.set_title(f"{name}: GT mask green / predicted red")
            axis.set_axis_off()

        axes[1, 2].set_axis_off()
        axes[1, 2].text(
            0.0,
            1.0,
            "\n".join(
                (
                    f"rank={rank}  keypoint={worst['keypoint_type']}",
                    f"trajectory={worst['trajectory_id']}",
                    f"timestep={worst['timestep']}",
                    f"world={worst['world_xy_error_m'] * 1000.0:.3f} mm",
                    f"pixel={worst['pixel_error_px']:.3f} px",
                    f"taxonomy={worst['failure_taxonomy']}",
                    f"accepted={worst['confidence_accepted']}",
                    f"confidence={worst['confidence_score']:.6f}",
                    f"visibility={worst['visibility_probability']:.6f}",
                    f"entropy={worst['normalized_entropy']:.6f}",
                    f"sigma={worst['radial_sigma_px']:.3f} px",
                    f"|n·unit_ray|={worst['abs_n_dot_unit_ray']:.6f}",
                    f"Jmax={worst['jacobian_sigma_max_mm_per_px']:.3f} mm/px",
                    f"top2/top1={worst['top2_top1_probability_ratio']:.6f}",
                )
            ),
            ha="left",
            va="top",
            family="monospace",
            fontsize=10,
        )
        fig.suptitle(
            "E014 private diagnostic — "
            f"trajectory={worst['trajectory_id']} timestep={worst['timestep']} "
            f"keypoint={worst['keypoint_type']} — {fingerprint}",
            fontsize=13,
        )
        fig.savefig(figure_root / f"{stem}.png", dpi=150)
        plt.close(fig)


def _public_maximum(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_fingerprint": row["sample_fingerprint"],
        "keypoint_type": row["keypoint_type"],
        "world_xy_error_mm": float(row["world_xy_error_m"] * 1000.0),
        "pixel_error_px": row["pixel_error_px"],
        "failure_taxonomy": row["failure_taxonomy"],
        "failure_family": row["failure_family"],
        "heatmap_mode_assessment": (
            "multimodal-softargmax"
            if row["failure_taxonomy"] == "multimodal_softargmax_failure"
            else "no-frozen-multimodal-signature"
        ),
        "confidence_accepted": row["confidence_accepted"],
        "confidence_score": row["confidence_score"],
        "visibility_probability": row["visibility_probability"],
        "projection_validity_probability": row["projection_validity_probability"],
        "normalized_entropy": row["normalized_entropy"],
        "radial_sigma_px": row["radial_sigma_px"],
        "top1_probability": row["top1_probability"],
        "top2_top1_probability_ratio": row["top2_top1_probability_ratio"],
        "softargmax_to_top1_distance_px": row["softargmax_to_top1_distance_px"],
        "top1_argmax_pixel_error_px": row["top1_argmax_pixel_error_px"],
        "abs_n_dot_unit_ray": row["abs_n_dot_unit_ray"],
        "jacobian_sigma_max_mm_per_px": row["jacobian_sigma_max_mm_per_px"],
        "object_goal_distance_mm": float(row["object_goal_distance_m"] * 1000.0),
        "semantic_margin_px": row["semantic_margin_px"],
        "best_adjacent_offset": row["best_adjacent_offset"],
        "adjacent_improvement_px": row["adjacent_improvement_px"],
    }


def _readme(summary: dict[str, Any]) -> str:
    maximum = summary["maximum_outlier"]
    confidence = summary["aggregate"]["confidence"]
    top20 = summary["top20_taxonomy"]
    top20_family = summary["top20_failure_family"]
    taxonomy_lines = "\n".join(f"- `{name}`: {top20[name]}" for name in FAILURE_TAXONOMY_PRIORITY)
    family_lines = "\n".join(f"- `{name}`: {top20_family[name]}" for name in FAILURE_FAMILY_ORDER)
    fail_closed = confidence["accepted_over_20mm_count"] == 0
    bottleneck = (
        "毫米级主体分布之外的离散 catastrophic perception failure"
        if summary["aggregate"]["world_error"]["all"]["p90_mm"] < 2.0
        else "主体定位精度"
    )
    return f"""# E014 — E013 Precision long-tail failure decomposition

本实验只诊断冻结的 E013 checkpoint、val/test split、temperature、confidence threshold、soft-argmax 与 GT z-plane；没有重新训练或按 test 调参。原始 RGB、heatmap、逐样本身份和完整几何证据只保存在私有本地结果中。

## 结论

- 重新计算得到 world-XY p50 `{summary["aggregate"]["world_error"]["all"]["p50_mm"]:.3f} mm`、p90 `{summary["aggregate"]["world_error"]["all"]["p90_mm"]:.3f} mm`、max `{summary["aggregate"]["world_error"]["all"]["max_mm"]:.3f} mm`，与 E013 canonical 指标在预注册容差内一致。
- 最大异常是 `{maximum["keypoint_type"]}`，匿名指纹 `{maximum["sample_fingerprint"]}`，分类为 `{maximum["failure_taxonomy"]}`；像素误差 `{maximum["pixel_error_px"]:.3f} px`，world-XY 误差 `{maximum["world_xy_error_mm"]:.3f} mm`。
- 最大异常的 heatmap 判定为 `{maximum["heatmap_mode_assessment"]}`，confidence accepted=`{maximum["confidence_accepted"]}`，visibility probability=`{maximum["visibility_probability"]:.6f}`。
- 当前主要瓶颈是：**{bottleneck}**。
- confidence gate 对超过 20 mm 的错误{"能够全部 fail-closed" if fail_closed else "不能可靠 fail-closed"}；被接受的 >20 / >50 / >100 mm 数量分别为 `{confidence["accepted_over_20mm_count"]}` / `{confidence["accepted_over_50mm_count"]}` / `{confidence["accepted_over_100mm_count"]}`。
- E013 的旧 `confidence_precision` 实际是 accepted validity precision，不是定位准确率；E014 单独报告 5/10/20 mm accepted accuracy。

## Top-20 五类 failure family（实验 Q2 口径）

{family_lines}

## Top-20 细粒度固定 taxonomy

{taxonomy_lines}

## 最大异常的可证伪检查

- soft-argmax 到 top-1 距离：`{maximum["softargmax_to_top1_distance_px"]:.3f} px`
- top-1 自身误差：`{maximum["top1_argmax_pixel_error_px"]:.3f} px`
- top-2 / top-1 probability ratio：`{maximum["top2_top1_probability_ratio"]:.6f}`
- `|n·unit_ray|`：`{maximum["abs_n_dot_unit_ray"]:.6f}`
- 局部几何 Jacobian 最大奇异值：`{maximum["jacobian_sigma_max_mm_per_px"]:.3f} mm/px`
- object/goal 物理间距：`{maximum["object_goal_distance_mm"]:.3f} mm`

这些量分别用于区分语义换位、相邻帧错位、多峰 soft-argmax、不可见/OOD 未拒绝和 ray-plane 几何放大。分类规则及所有数值阈值在读取 test 前由 frozen validation 封存。

## 安全含义

如果未来把该 perception 结果直接接入 actuator，真正可能被执行的是 `confidence_accepted=true` 的 catastrophic rows；其数量和最大误差见脱敏汇总。后续任何修复必须使用新的 validation/test seeds；本次 E013 test split 已标记为 `consumed-for-diagnostic-postmortem`，不能再声称是最终未见 test。
"""


def _write_public_output(
    *,
    public_root: Path,
    repository_root: Path,
    summary: dict[str, Any],
    distributions: dict[str, Any],
    context: _FrozenContext,
    rules_sha256: str,
) -> dict[str, Any]:
    if public_root.exists():
        raise FileExistsError(f"E014 public output 已存在，拒绝覆盖: {public_root}")
    public_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    assert_public_payload_safe(summary)
    assert_public_payload_safe(distributions)
    _atomic_json(public_root / "sanitized_summary.json", summary)
    _atomic_json(public_root / "distributions.json", distributions)
    readme = _readme(summary)
    if any(token in readme for token in ("/home/", "/mnt/", "C:\\")):
        raise RuntimeError("E014 public README 含敏感绝对路径")
    _atomic_text(public_root / "README.md", readme)
    verifier_template = repository_root / "scripts" / "verify_e014_public_results.py"
    if not verifier_template.is_file():
        raise FileNotFoundError("E014 public verifier template 不存在")
    shutil.copyfile(verifier_template, public_root / "verify_summary.py")
    files = {
        name: file_sha256(public_root / name)
        for name in (
            "README.md",
            "sanitized_summary.json",
            "distributions.json",
            "verify_summary.py",
        )
    }
    receipt = {
        "version": E014_DIAGNOSTIC_VERSION,
        "status": "complete",
        "diagnostic_source_tree_sha256": context.diagnostic_source_tree_sha256,
        "upstream_held_out_receipt_sha256": context.upstream_receipt_sha256,
        "rules_sha256": rules_sha256,
        "files": files,
        "contains_raw_rgb": False,
        "contains_raw_heatmaps": False,
        "contains_model_weights": False,
        "contains_sensitive_paths": False,
    }
    assert_public_payload_safe(receipt)
    _atomic_json(public_root / "receipt.json", receipt)
    subprocess.run(
        [sys.executable, str(public_root / "verify_summary.py"), str(public_root)],
        check=True,
        cwd=repository_root,
    )
    return receipt


def _run_test(args: argparse.Namespace) -> dict[str, Any]:
    private_root = Path(args.private_output)
    public_root = Path(args.public_output)
    if private_root.exists():
        raise FileExistsError(f"E014 private output 已存在，拒绝覆盖: {private_root}")
    if public_root.exists():
        raise FileExistsError(f"E014 public output 已存在，拒绝覆盖: {public_root}")
    context = _frozen_context(args)
    rules_document = _load_and_verify_rules(Path(args.rules), context)
    rules = rules_document["rules"]
    private_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    _atomic_json(
        private_root / "run_state.json",
        {
            "version": E014_DIAGNOSTIC_VERSION,
            "status": "in-progress",
            "test_split": "frozen-e013-test",
        },
    )
    try:
        dataset = PrecisionRGBDataset(
            args.deployable_root,
            args.label_root,
            "test",
            cache_size=context.config.formal_training.cache_size,
        )
        if len(dataset) != int(context.canonical_held_out["sample_count"]):
            raise RuntimeError("E014 test sample count 与 canonical held-out 不一致")
        collection = _collect_rows(
            model=context.model,
            dataset=dataset,
            config=context.config,
            temperature=context.calibration.temperature,
            confidence_threshold=context.calibration.confidence_threshold,
            nms_radius_px=int(rules["thresholds"]["nms_radius_px"]),
            keep_artifacts=True,
        )
        for row in collection.rows:
            row["failure_taxonomy"] = classify_outlier(row, rules)
            row["failure_family"] = failure_family(row["failure_taxonomy"])
            row["high_confidence_catastrophic_perception_failure"] = bool(
                row["confidence_accepted"]
                and row["world_xy_error_m"] is not None
                and float(row["world_xy_error_m"]) > 0.020
            )
        aggregate = aggregate_prediction_rows(collection.rows)
        reproduction = _metric_reproduction(
            aggregate=aggregate,
            context=context,
            invalid_backprojection_count=collection.invalid_backprojection_count,
        )
        valid_rows = [row for row in collection.rows if row["world_xy_error_m"] is not None]
        valid_rows.sort(
            key=lambda row: (-float(row["world_xy_error_m"]), row["sample_fingerprint"])
        )
        if len(valid_rows) < 50:
            raise RuntimeError("E014 test 有效 rows 少于 Top-50")
        top20 = valid_rows[:20]
        top50 = valid_rows[:50]
        maximum = top20[0]
        if len(valid_rows) > 1 and math.isclose(
            float(maximum["world_xy_error_m"]),
            float(valid_rows[1]["world_xy_error_m"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("E014 最大 outlier 不是唯一行")
        accepted_catastrophic = [
            row
            for row in valid_rows
            if row["confidence_accepted"] and float(row["world_xy_error_m"]) > 0.020
        ]
        top20_taxonomy = taxonomy_counts(top20)
        top50_taxonomy = taxonomy_counts(top50)
        all_taxonomy = taxonomy_counts(collection.rows)
        top20_failure_family = failure_family_counts(top20)
        top50_failure_family = failure_family_counts(top50)
        all_rows_failure_family = failure_family_counts(collection.rows)

        _atomic_jsonl(private_root / "per_sample.jsonl", collection.rows)
        _atomic_jsonl(private_root / "top50_worst.jsonl", top50)
        _atomic_jsonl(
            private_root / "accepted_catastrophic_failures.jsonl",
            accepted_catastrophic,
        )
        _atomic_json(private_root / "frozen_rules.json", rules_document)
        _plot_top20(
            private_root=private_root,
            dataset=dataset,
            rows=collection.rows,
            top20=top20,
            artifacts=collection.artifacts,
        )
        private_summary = {
            "version": E014_DIAGNOSTIC_VERSION,
            "status": "complete",
            "aggregate": aggregate,
            "metric_reproduction": reproduction,
            "maximum_outlier": maximum,
            "top20_taxonomy": top20_taxonomy,
            "top50_taxonomy": top50_taxonomy,
            "all_rows_taxonomy": all_taxonomy,
            "top20_failure_family": top20_failure_family,
            "top50_failure_family": top50_failure_family,
            "all_rows_failure_family": all_rows_failure_family,
            "accepted_catastrophic_count": len(accepted_catastrophic),
            "test_split_status_after_e014": "consumed-for-diagnostic-postmortem",
        }
        _atomic_json(private_root / "summary_private.json", private_summary)

        sanitized_summary = {
            "version": E014_DIAGNOSTIC_VERSION,
            "status": "complete",
            "experiment": "E014 — E013 Precision long-tail failure decomposition",
            "frozen_conditions": {
                "checkpoint_sha256": context.checkpoint_sha256,
                "data_identity_sha256": context.data_identity_sha256,
                "calibration_sha256": context.calibration.sha256,
                "temperature": context.calibration.temperature,
                "confidence_threshold": context.calibration.confidence_threshold,
                "softargmax_changed": False,
                "checkpoint_changed": False,
                "training_performed": False,
            },
            "aggregate": aggregate,
            "metric_reproduction": reproduction,
            "maximum_outlier": _public_maximum(maximum),
            "top20_taxonomy": top20_taxonomy,
            "top50_taxonomy": top50_taxonomy,
            "all_rows_taxonomy": all_taxonomy,
            "top20_failure_family": top20_failure_family,
            "top50_failure_family": top50_failure_family,
            "all_rows_failure_family": all_rows_failure_family,
            "taxonomy_priority": list(FAILURE_TAXONOMY_PRIORITY),
            "validation_frozen_thresholds": rules["thresholds"],
            "test_split_status_after_e014": "consumed-for-diagnostic-postmortem",
            "public_evidence_policy": "aggregate-and-pseudonymous-only/v1",
        }
        distributions = {
            "version": E014_DIAGNOSTIC_VERSION,
            "world_error": aggregate["world_error"],
            "confidence": aggregate["confidence"],
            "top20_taxonomy": top20_taxonomy,
            "top50_taxonomy": top50_taxonomy,
            "top20_failure_family": top20_failure_family,
            "top50_failure_family": top50_failure_family,
        }
        public_receipt = _write_public_output(
            public_root=public_root,
            repository_root=Path(args.repository_root).resolve(),
            summary=sanitized_summary,
            distributions=distributions,
            context=context,
            rules_sha256=str(rules_document["rules_sha256"]),
        )
        private_receipt = {
            "version": E014_DIAGNOSTIC_VERSION,
            "status": "complete",
            "rules_sha256": str(rules_document["rules_sha256"]),
            "diagnostic_source_tree_sha256": context.diagnostic_source_tree_sha256,
            "upstream_held_out_receipt_sha256": context.upstream_receipt_sha256,
            "files": {
                name: file_sha256(private_root / name)
                for name in (
                    "per_sample.jsonl",
                    "top50_worst.jsonl",
                    "accepted_catastrophic_failures.jsonl",
                    "frozen_rules.json",
                    "summary_private.json",
                )
            },
            "top20_figure_count": len(list((private_root / "top20").glob("*.png"))),
            "top20_heatmap_array_count": len(list((private_root / "top20_heatmaps").glob("*.npz"))),
            "public_receipt_sha256": canonical_sha256(public_receipt),
        }
        _atomic_json(private_root / "receipt.json", private_receipt)
        _atomic_json(
            private_root / "run_state.json",
            {
                "version": E014_DIAGNOSTIC_VERSION,
                "status": "complete",
                "test_split": "frozen-e013-test",
                "test_split_status_after_e014": "consumed-for-diagnostic-postmortem",
            },
        )
        return {
            "status": "complete",
            "prediction_rows": len(collection.rows),
            "valid_keypoints": len(valid_rows),
            "maximum_outlier": _public_maximum(maximum),
            "top20_taxonomy": top20_taxonomy,
            "top20_failure_family": top20_failure_family,
            "accepted_catastrophic_count": len(accepted_catastrophic),
        }
    except Exception as error:
        _atomic_json(
            private_root / "run_state.json",
            {
                "version": E014_DIAGNOSTIC_VERSION,
                "status": "failed-preserved",
                "test_split": "frozen-e013-test",
                "error_type": type(error).__name__,
            },
        )
        raise


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--deployable-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--held-out-output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser(
        "freeze-rules",
        help="只运行 frozen validation，并在访问 test 前封存诊断规则",
    )
    _add_common_arguments(freeze)
    freeze.add_argument("--rules-output", type=Path, required=True)
    analyze = subparsers.add_parser(
        "analyze-test",
        help="读取已封存规则，只运行一次 frozen test",
    )
    _add_common_arguments(analyze)
    analyze.add_argument("--rules", type=Path, required=True)
    analyze.add_argument("--private-output", type=Path, required=True)
    analyze.add_argument("--public-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "freeze-rules":
        result = _freeze_rules(args)
    elif args.command == "analyze-test":
        result = _run_test(args)
    else:  # pragma: no cover - argparse 已限制 command
        raise AssertionError(f"未知 command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
