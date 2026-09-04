"""E013 held-out perception、confidence calibration 与完整 Provider latency。"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_vla.adapters import (
    FingerForceNormalizer,
    FingerForceStats,
    ProprioNormalizer,
    ProprioStats,
)
from robot_vla.contracts import OBSERVATION_HISTORY_LENGTH, RobotSpec
from robot_vla.executive.contracts import PredicateSource
from robot_vla.observation import (
    OBSERVATION_MODALITIES,
    ObservationV2Frame,
    ObservationV2History,
    rotation_6d_to_matrix,
)
from robot_vla.precision.checkpoint import (
    PrecisionCheckpointRole,
    load_precision_checkpoint,
    load_torch_precision_frame_predictor,
)
from robot_vla.precision.data import (
    PrecisionRGBDataset,
    audit_precision_dataset,
    canonical_sha256,
    file_sha256,
)
from robot_vla.precision.detection import PRECISION_TRACK_CONFIDENCE_SEMANTICS
from robot_vla.precision.geometry import normalized_uv_to_base_z_plane
from robot_vla.precision.provider import (
    PrecisionDetectionProvider,
    PrecisionDetectionProviderConfig,
    PrecisionGeometricMotionInput,
    TorchPrecisionFramePredictorConfig,
)
from robot_vla.precision.training import (
    PrecisionExperimentConfig,
    _build_loader,
    _to_device,
    evaluate_precision_model,
    load_precision_experiment_config,
    source_tree_sha256,
)

PRECISION_CALIBRATION_VERSION = "e013-precision-confidence-calibration/v1"
PRECISION_HELD_OUT_VERSION = "e013-precision-held-out-evaluation/v1"
PRECISION_PROVIDER_BENCHMARK_VERSION = "e013-precision-provider-latency/v1"


@dataclass(frozen=True)
class PrecisionConfidenceCalibration:
    version: str
    checkpoint_sha256: str
    data_identity_sha256: str
    training_config_sha256: str
    calibration_split: str
    sample_count: int
    valid_keypoint_count: int
    temperature: float
    confidence_threshold: float
    target_coverage: float
    sigma_scale: float
    validation_coverage: float
    validation_precision: float
    validation_sigma_coverage: float
    confidence_semantics: str = PRECISION_TRACK_CONFIDENCE_SEMANTICS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class PrecisionHeldOutMetrics:
    version: str
    checkpoint_sha256: str
    data_identity_sha256: str
    calibration_sha256: str
    split: str
    sample_count: int
    valid_keypoint_count: int
    world_xy_error_p50_m: float
    world_xy_error_p90_m: float
    world_xy_error_max_m: float
    object_world_xy_error_p90_m: float
    goal_world_xy_error_p90_m: float
    within_15mm_rate: float
    pixel_error_p50: float
    pixel_error_p90: float
    mask_iou: float
    confidence_coverage: float
    confidence_precision: float
    accepted_world_xy_error_p90_m: float | None
    sigma_coverage: float
    invalid_backprojection_count: int
    perception_gate_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrecisionProviderLatencyMetrics:
    version: str
    checkpoint_sha256: str
    provider_identity_sha256: str
    provider_records_sha256: str
    warmup_calls: int
    measurement_calls: int
    full_history_call_count: int
    predicted_frame_count: int
    failed_call_count: int
    provider_latency_p50_s: float
    provider_latency_p95_s: float
    predictor_frame_latency_p50_s: float
    predictor_frame_latency_p95_s: float
    effective_rate_from_p95_hz: float
    latency_gate_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrecisionHeldOutReceipt:
    calibration: PrecisionConfidenceCalibration
    held_out: PrecisionHeldOutMetrics
    provider_latency: PrecisionProviderLatencyMetrics
    evaluation_source_tree_sha256: str
    training_source_tree_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration": self.calibration.to_dict(),
            "held_out": self.held_out.to_dict(),
            "provider_latency": self.provider_latency.to_dict(),
            "evaluation_source_tree_sha256": self.evaluation_source_tree_sha256,
            "training_source_tree_sha256": self.training_source_tree_sha256,
        }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class _PredictionRows:
    sample_count: int
    valid: np.ndarray
    score: np.ndarray
    sigma_px: np.ndarray
    pixel_error: np.ndarray
    world_error_m: np.ndarray
    keypoint_index: np.ndarray
    invalid_backprojection_count: int


def _collect_prediction_rows(
    model: Any,
    dataset: PrecisionRGBDataset,
    *,
    device: torch.device,
    batch_size: int,
    temperature: float,
    use_bf16: bool,
) -> _PredictionRows:
    loader = _build_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=0,
    )
    valid_rows: list[np.ndarray] = []
    score_rows: list[np.ndarray] = []
    sigma_rows: list[np.ndarray] = []
    pixel_rows: list[np.ndarray] = []
    world_rows: list[np.ndarray] = []
    keypoint_rows: list[np.ndarray] = []
    invalid_backprojection = 0
    sample_count = 0
    model.eval()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                output = model(
                    batch["image"],
                    batch["structured_state"],
                    batch["geometric_motion"],
                )
            decoded = output.decode_for_control(temperature=temperature)
            predicted_uv = decoded.keypoints.normalized_uv.detach().cpu().numpy()
            target_uv = batch["normalized_uv_targets"].detach().cpu().numpy()
            valid = batch["keypoint_valid"].detach().cpu().numpy().astype(np.bool_)
            height, width = batch["image"].shape[-2:]
            difference = predicted_uv - target_uv
            pixel_error = np.linalg.norm(
                difference * np.asarray((width, height), dtype=np.float32),
                axis=-1,
            )
            visibility = decoded.visibility_probability.detach().cpu().numpy()
            projection = decoded.projection_validity_probability.detach().cpu().numpy()
            # 必须与运行时 WristKeypointDetection 的冻结 confidence 语义完全一致；
            # entropy/peak/sigma 仅保留为独立诊断，不能暗中混入部署阈值。
            score = np.minimum(visibility, projection[:, None])
            sigma = decoded.keypoint_sigma_px.detach().cpu().numpy()
            radial_sigma = np.linalg.norm(sigma, axis=-1)
            world_error = np.full(valid.shape, np.nan, dtype=np.float64)
            for batch_index, audit in enumerate(batch["audit"]):
                base_from_camera = np.eye(4, dtype=np.float64)
                base_from_camera[:3, :3] = rotation_6d_to_matrix(
                    audit["wrist_camera_rotation_6d_base"]
                )
                base_from_camera[:3, 3] = audit["wrist_camera_position_base_m"]
                positions = (
                    audit["object_position_base_m"],
                    audit["goal_position_base_m"],
                )
                for keypoint_index, target_position in enumerate(positions):
                    if not valid[batch_index, keypoint_index]:
                        continue
                    try:
                        reconstructed = normalized_uv_to_base_z_plane(
                            predicted_uv[batch_index, keypoint_index],
                            audit["intrinsic_wrist_cv"],
                            base_from_camera,
                            (height, width),
                            plane_base_z_m=float(target_position[2]),
                        ).point_base_m
                    except ValueError:
                        invalid_backprojection += 1
                        continue
                    world_error[batch_index, keypoint_index] = float(
                        np.linalg.norm(reconstructed[:2] - target_position[:2])
                    )
            valid_rows.append(valid)
            score_rows.append(score)
            sigma_rows.append(radial_sigma)
            pixel_rows.append(pixel_error)
            world_rows.append(world_error)
            keypoint_rows.append(np.broadcast_to(np.arange(2, dtype=np.int8), valid.shape).copy())
            sample_count += int(valid.shape[0])
    return _PredictionRows(
        sample_count=sample_count,
        valid=np.concatenate(valid_rows, axis=0).reshape(-1),
        score=np.concatenate(score_rows, axis=0).reshape(-1),
        sigma_px=np.concatenate(sigma_rows, axis=0).reshape(-1),
        pixel_error=np.concatenate(pixel_rows, axis=0).reshape(-1),
        world_error_m=np.concatenate(world_rows, axis=0).reshape(-1),
        keypoint_index=np.concatenate(keypoint_rows, axis=0).reshape(-1),
        invalid_backprojection_count=invalid_backprojection,
    )


def _safe_quantile(values: np.ndarray, quantile: float, name: str) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError(f"{name} 没有有限样本")
    return float(np.quantile(finite, quantile))


def _calibrate(
    *,
    checkpoint_sha256: str,
    data_identity_sha256: str,
    training_config_sha256: str,
    rows: _PredictionRows,
    temperature: float,
    target_coverage: float,
) -> PrecisionConfidenceCalibration:
    valid = rows.valid & np.isfinite(rows.world_error_m)
    valid_scores = rows.score[valid]
    threshold = _safe_quantile(
        valid_scores,
        1.0 - target_coverage,
        "calibration confidence score",
    )
    accepted = rows.score >= threshold
    valid_accepted = accepted & valid
    ratio = rows.pixel_error[valid] / np.maximum(rows.sigma_px[valid], 1e-6)
    sigma_scale = _safe_quantile(ratio, target_coverage, "sigma scale")
    sigma_covered = rows.pixel_error[valid] <= sigma_scale * rows.sigma_px[valid]
    return PrecisionConfidenceCalibration(
        version=PRECISION_CALIBRATION_VERSION,
        checkpoint_sha256=checkpoint_sha256,
        data_identity_sha256=data_identity_sha256,
        training_config_sha256=training_config_sha256,
        calibration_split="val",
        sample_count=rows.sample_count,
        valid_keypoint_count=int(valid.sum()),
        temperature=float(temperature),
        confidence_threshold=threshold,
        target_coverage=target_coverage,
        sigma_scale=sigma_scale,
        validation_coverage=float(valid_accepted.sum() / valid.sum()),
        validation_precision=float(valid_accepted.sum() / max(1, accepted.sum())),
        validation_sigma_coverage=float(sigma_covered.mean()),
    )


def _held_out_metrics(
    *,
    checkpoint_sha256: str,
    data_identity_sha256: str,
    calibration: PrecisionConfidenceCalibration,
    rows: _PredictionRows,
    mask_iou: float,
) -> PrecisionHeldOutMetrics:
    valid = rows.valid & np.isfinite(rows.world_error_m)
    accepted = rows.score >= calibration.confidence_threshold
    valid_accepted = valid & accepted
    world = rows.world_error_m[valid]
    object_world = rows.world_error_m[valid & (rows.keypoint_index == 0)]
    goal_world = rows.world_error_m[valid & (rows.keypoint_index == 1)]
    sigma_covered = rows.pixel_error[valid] <= (calibration.sigma_scale * rows.sigma_px[valid])
    accepted_world = rows.world_error_m[valid_accepted]
    p90 = _safe_quantile(world, 0.90, "held-out world error")
    perception_gate = rows.invalid_backprojection_count == 0 and p90 <= 0.015
    return PrecisionHeldOutMetrics(
        version=PRECISION_HELD_OUT_VERSION,
        checkpoint_sha256=checkpoint_sha256,
        data_identity_sha256=data_identity_sha256,
        calibration_sha256=calibration.sha256,
        split="test",
        sample_count=rows.sample_count,
        valid_keypoint_count=int(valid.sum()),
        world_xy_error_p50_m=_safe_quantile(world, 0.50, "held-out world error"),
        world_xy_error_p90_m=p90,
        world_xy_error_max_m=float(np.max(world)),
        object_world_xy_error_p90_m=_safe_quantile(
            object_world,
            0.90,
            "object world error",
        ),
        goal_world_xy_error_p90_m=_safe_quantile(
            goal_world,
            0.90,
            "goal world error",
        ),
        within_15mm_rate=float(np.mean(world <= 0.015)),
        pixel_error_p50=_safe_quantile(rows.pixel_error[valid], 0.50, "pixel error"),
        pixel_error_p90=_safe_quantile(rows.pixel_error[valid], 0.90, "pixel error"),
        mask_iou=float(mask_iou),
        confidence_coverage=float(valid_accepted.sum() / valid.sum()),
        confidence_precision=float(valid_accepted.sum() / max(1, accepted.sum())),
        accepted_world_xy_error_p90_m=(
            None
            if accepted_world.size == 0
            else _safe_quantile(accepted_world, 0.90, "accepted world error")
        ),
        sigma_coverage=float(sigma_covered.mean()),
        invalid_backprojection_count=rows.invalid_backprojection_count,
        perception_gate_passed=perception_gate,
    )


def _transform(position: np.ndarray, rotation_6d: np.ndarray) -> np.ndarray:
    value = np.eye(4, dtype=np.float32)
    value[:3, :3] = rotation_6d_to_matrix(rotation_6d)
    value[:3, 3] = position
    return value


def _window_for_dataset_index(
    dataset: PrecisionRGBDataset,
    index: int,
) -> Any:
    entry_index, timestep = dataset.base.index[index]
    meta = dataset.base.entries[entry_index]
    arrays = dataset.base.store.get(meta)
    history = ObservationV2History(dataset.spec)
    start = max(0, timestep - 3)
    for source in range(start, timestep + 1):
        timestamps = np.asarray(
            (
                arrays.timestamp_external[source],
                arrays.timestamp_wrist[source],
                arrays.timestamp_proprio[source],
                arrays.timestamp_tcp_pose[source],
                arrays.timestamp_camera_pose[source],
                arrays.timestamp_finger_force[source],
            ),
            dtype=np.float64,
        )
        valid = np.asarray(
            (
                arrays.external_valid[source],
                arrays.wrist_valid[source],
                arrays.proprio_valid[source],
                arrays.tcp_pose_valid[source],
                arrays.camera_pose_valid[source],
                arrays.finger_force_valid[source],
            ),
            dtype=np.bool_,
        )
        history.append(
            ObservationV2Frame(
                rgb_external=arrays.rgb_external[source],
                rgb_wrist=arrays.rgb_wrist[source],
                physical_proprio=arrays.proprio[source],
                base_from_tcp=_transform(
                    arrays.tcp_position_base_m[source],
                    arrays.tcp_rotation_6d_base[source],
                ),
                base_from_wrist_camera=_transform(
                    arrays.wrist_camera_position_base_m[source],
                    arrays.wrist_camera_rotation_6d_base[source],
                ),
                finger_force_n=np.asarray(
                    (
                        arrays.left_finger_force_n[source],
                        arrays.right_finger_force_n[source],
                    ),
                    dtype=np.float32,
                ),
                timestamp_s=float(arrays.timestamp_action[source]),
                modality_timestamp_s=timestamps,
                modality_valid=valid,
            )
        )
    previous_command = (
        arrays.previous_command_q_rad[timestep] if arrays.previous_command_valid[timestep] else None
    )
    previous_action = (
        arrays.previous_action[timestep] if arrays.previous_action_valid[timestep] else None
    )
    return history.snapshot(
        meta.task.instruction,
        previous_command_q=previous_command,
        previous_action=previous_action,
    )


def _safe_hold_geometry(window: Any, frame_index: int) -> PrecisionGeometricMotionInput:
    wrist_index = OBSERVATION_MODALITIES.index("rgb_wrist")
    return PrecisionGeometricMotionInput(
        timestamp_s=float(window.modality_timestamp_s[frame_index, wrist_index]),
        motion=(0.0, 0.0, 0.0, 0.0),
        source=PredicateSource.DEPLOYABLE_ESTIMATOR,
    )


def _full_history_dataset_indices(dataset: PrecisionRGBDataset) -> list[int]:
    minimum_timestep = OBSERVATION_HISTORY_LENGTH - 1
    indices = [
        index
        for index, (_, timestep) in enumerate(dataset.base.index)
        if timestep >= minimum_timestep
    ]
    if not indices:
        raise ValueError("Provider latency Dataset 没有完整四帧 history 样本")
    return indices


def _provider_latency(
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    calibration: PrecisionConfidenceCalibration,
    dataset: PrecisionRGBDataset,
    deployable_root: Path,
    config: PrecisionExperimentConfig,
) -> PrecisionProviderLatencyMetrics:
    predictor = load_torch_precision_frame_predictor(
        checkpoint_path,
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
        predictor_config=TorchPrecisionFramePredictorConfig(
            device="cuda",
            use_bf16=True,
            temperature=calibration.temperature,
            synchronize_cuda_for_latency=True,
        ),
    ).predictor
    spec = RobotSpec()
    proprio_stats_path = deployable_root / "proprio_stats.json"
    force_stats_path = deployable_root / "finger_force_stats.json"
    provider = PrecisionDetectionProvider(
        spec,
        predictor,
        ProprioNormalizer(ProprioStats.from_json(proprio_stats_path), spec),
        FingerForceNormalizer(FingerForceStats.from_json(force_stats_path), spec),
        _safe_hold_geometry,
        geometric_motion_provider_id="deployable-safe-hold/provider-latency-only/v1",
        proprio_stats_sha256=file_sha256(proprio_stats_path),
        finger_force_stats_sha256=file_sha256(force_stats_path),
        config=PrecisionDetectionProviderConfig(enabled=True),
    )
    warmup = config.held_out.provider_latency_warmup_calls
    measured = config.held_out.provider_latency_measurement_calls
    full_history_indices = _full_history_dataset_indices(dataset)
    for call_index in range(warmup):
        index = full_history_indices[call_index % len(full_history_indices)]
        provider(_window_for_dataset_index(dataset, index))
    provider.reset()
    for call_index in range(measured):
        index = full_history_indices[call_index % len(full_history_indices)]
        provider(_window_for_dataset_index(dataset, index))
    records = provider.records
    failed = sum(not record.success for record in records)
    full_history_calls = sum(
        record.success and record.detections_count == OBSERVATION_HISTORY_LENGTH
        for record in records
    )
    provider_latency = np.asarray(
        [record.total_latency_s for record in records],
        dtype=np.float64,
    )
    frame_latency = np.asarray(
        [
            frame.predictor_latency_s
            for record in records
            for frame in record.frame_records
            if frame.evidence is not None
        ],
        dtype=np.float64,
    )
    predicted = int(frame_latency.size)
    p95 = _safe_quantile(provider_latency, 0.95, "provider latency")
    return PrecisionProviderLatencyMetrics(
        version=PRECISION_PROVIDER_BENCHMARK_VERSION,
        checkpoint_sha256=checkpoint_sha256,
        provider_identity_sha256=provider.identity.sha256,
        provider_records_sha256=provider.records_sha256,
        warmup_calls=warmup,
        measurement_calls=measured,
        full_history_call_count=full_history_calls,
        predicted_frame_count=predicted,
        failed_call_count=failed,
        provider_latency_p50_s=_safe_quantile(
            provider_latency,
            0.50,
            "provider latency",
        ),
        provider_latency_p95_s=p95,
        predictor_frame_latency_p50_s=_safe_quantile(
            frame_latency,
            0.50,
            "predictor frame latency",
        ),
        predictor_frame_latency_p95_s=_safe_quantile(
            frame_latency,
            0.95,
            "predictor frame latency",
        ),
        effective_rate_from_p95_hz=float(1.0 / p95),
        latency_gate_passed=(
            failed == 0
            and len(records) == measured
            and full_history_calls == measured
            and predicted == measured * OBSERVATION_HISTORY_LENGTH
            and p95 <= config.shadow_rollout.p95_latency_max_s
            and p95 <= 1.0 / config.shadow_rollout.required_control_hz
        ),
    )


def evaluate_precision_checkpoint(
    *,
    deployable_root: str | Path,
    label_root: str | Path,
    config_path: str | Path,
    training_output: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
) -> PrecisionHeldOutReceipt:
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"Precision held-out output 已存在，拒绝覆盖: {output}")
    evaluation_source_identity = source_tree_sha256(repository_root)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    config = load_precision_experiment_config(config_path)
    audit = audit_precision_dataset(
        deployable_root,
        label_root,
        RobotSpec(),
        write_artifact=False,
    )
    if not audit.passed:
        raise RuntimeError("Precision Dataset audit 未通过，禁止 held-out evaluation")
    training_root = Path(training_output)
    training_receipt = json.loads(
        (training_root / "checkpoint_receipt.json").read_text(encoding="utf-8")
    )
    checkpoint_path = training_root / "precision-formal.pt"
    checkpoint_sha256 = str(training_receipt["checkpoint"]["checkpoint_sha256"])
    loaded = load_precision_checkpoint(
        checkpoint_path,
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_provenance_sha256=str(training_receipt["checkpoint"]["provenance_sha256"]),
        expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
    )
    if loaded.provenance.data_identity_sha256 != audit.dataset_identity_sha256:
        raise RuntimeError("Precision checkpoint Dataset identity 与 held-out Dataset 不一致")
    if loaded.provenance.training_config_sha256 != config.sha256:
        raise RuntimeError("Precision checkpoint training config identity 漂移")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Precision held-out evaluation 要求 BF16 CUDA")
    device = torch.device("cuda")
    model = loaded.model.to(device)
    val_dataset = PrecisionRGBDataset(
        deployable_root,
        label_root,
        "val",
        cache_size=config.formal_training.cache_size,
    )
    test_dataset = PrecisionRGBDataset(
        deployable_root,
        label_root,
        "test",
        cache_size=config.formal_training.cache_size,
    )
    temperature_metrics: list[tuple[float, float]] = []
    for temperature in config.held_out.temperature_grid:
        metrics = evaluate_precision_model(
            model,
            val_dataset,
            device=device,
            batch_size=config.formal_training.batch_size,
            use_bf16=config.formal_training.use_bf16,
            heatmap_sigma_px=config.formal_training.heatmap_sigma_px,
            temperature=temperature,
        )
        temperature_metrics.append((temperature, metrics.normalized_uv_mae))
    temperature = min(temperature_metrics, key=lambda item: (item[1], item[0]))[0]
    val_rows = _collect_prediction_rows(
        model,
        val_dataset,
        device=device,
        batch_size=config.formal_training.batch_size,
        temperature=temperature,
        use_bf16=config.formal_training.use_bf16,
    )
    calibration = _calibrate(
        checkpoint_sha256=checkpoint_sha256,
        data_identity_sha256=audit.dataset_identity_sha256,
        training_config_sha256=config.sha256,
        rows=val_rows,
        temperature=temperature,
        target_coverage=config.held_out.confidence_target_coverage,
    )
    test_split_metrics = evaluate_precision_model(
        model,
        test_dataset,
        device=device,
        batch_size=config.formal_training.batch_size,
        use_bf16=config.formal_training.use_bf16,
        heatmap_sigma_px=config.formal_training.heatmap_sigma_px,
        temperature=temperature,
    )
    test_rows = _collect_prediction_rows(
        model,
        test_dataset,
        device=device,
        batch_size=config.formal_training.batch_size,
        temperature=temperature,
        use_bf16=config.formal_training.use_bf16,
    )
    held_out = _held_out_metrics(
        checkpoint_sha256=checkpoint_sha256,
        data_identity_sha256=audit.dataset_identity_sha256,
        calibration=calibration,
        rows=test_rows,
        mask_iou=test_split_metrics.mask_iou,
    )
    latency = _provider_latency(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        calibration=calibration,
        dataset=test_dataset,
        deployable_root=Path(deployable_root),
        config=config,
    )
    receipt = PrecisionHeldOutReceipt(
        calibration=calibration,
        held_out=held_out,
        provider_latency=latency,
        evaluation_source_tree_sha256=evaluation_source_identity,
        training_source_tree_sha256=loaded.provenance.source_tree_sha256,
    )
    _atomic_json(output / "confidence_calibration.json", calibration.to_dict())
    _atomic_json(output / "held_out_evaluation.json", held_out.to_dict())
    _atomic_json(output / "provider_latency.json", latency.to_dict())
    _atomic_json(
        output / "receipt.json",
        {
            **receipt.to_dict(),
            "temperature_search": [
                {"temperature": temperature_value, "normalized_uv_mae": metric}
                for temperature_value, metric in temperature_metrics
            ],
        },
    )
    return receipt


__all__ = [
    "PRECISION_CALIBRATION_VERSION",
    "PRECISION_HELD_OUT_VERSION",
    "PRECISION_PROVIDER_BENCHMARK_VERSION",
    "PrecisionConfidenceCalibration",
    "PrecisionHeldOutMetrics",
    "PrecisionHeldOutReceipt",
    "PrecisionProviderLatencyMetrics",
    "evaluate_precision_checkpoint",
]
