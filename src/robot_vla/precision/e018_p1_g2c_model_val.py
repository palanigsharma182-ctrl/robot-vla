"""E018-P1 G2C model-validation 的 deployable freeze 与 label-only 评分。"""

from __future__ import annotations

import gc
import hashlib
import io
import math
import platform
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla.precision.e018_p1_g2a import canonical_sha256, file_sha256
from robot_vla.precision.e018_p1_g2c import (
    G2C_CANDIDATE_EPOCHS,
    G2C_CANDIDATE_IDS,
    assert_g2c_prediction_ledger_deployable_only,
    select_g2c_checkpoint,
    summarize_g2c_model_val_view,
)
from robot_vla.precision.e018_p1_g2c_data import (
    _LABEL_ARRAYS,
    G2C_LABEL_SCHEMA_VERSION,
    G2C_MANIFEST_SCHEMA_VERSION,
    G2C_VIEW_ORDER,
    G2CDeployableDataset,
    _atomic_json,
    _atomic_jsonl,
    _atomic_npz,
    _read_jsonl,
    _resolve_artifact_file,
)
from robot_vla.precision.e018_p1_g2c_training import (
    _EXPECTED_SPLIT_SAMPLES,
    _LOSS_COMPONENT_NAMES,
    _LOSS_SHARD_ARRAYS,
    E018_P1_G2C_PREDICTION_FREEZE_VERSION,
    E018_P1_G2C_SELECTION_RESULT_VERSION,
    G2C_DIAGNOSTIC_CONTROL_ID,
    _git_source_identity,
    _read_json,
    _read_json_array,
    _require_exact_keys,
    _require_sha256,
    _verify_exact_regular_file_tree,
    load_g2c_formal_training_config,
    validate_g2c_input_view,
    verify_g2c_formal_training,
)
from robot_vla.precision.object_observability import ObjectWriteEvidence
from robot_vla.precision.observability import mask_probability_at_normalized_uv
from robot_vla.precision.outliers import geometry_conditioning

_PREDICTION_ROW_KEYS = {
    "version",
    "phase",
    "candidate_id",
    "epoch",
    "checkpoint_sha256",
    "checkpoint_parameter_sha256",
    "checkpoint_provenance_sha256",
    "checkpoint_model_config_sha256",
    "row_index",
    "batch_index",
    "batch_offset",
    "seed",
    "split",
    "sample_index",
    "viewpoint_id",
    "input_sha256",
    "predicted_object_normalized_uv",
    "predicted_goal_normalized_uv",
    "object_visibility_probability",
    "goal_visibility_probability",
    "projection_validity_probability",
    "object_normalized_entropy",
    "object_sigma_xy_px",
    "object_mask_probability_at_prediction",
    "goal_mask_probability_at_prediction",
    "predicted_observable",
    "geometry_valid",
    "predicted_object_position_base_m",
    "raw_covariance_base_m2",
    "write_score",
    "memory_write_allowed",
    "actuation_allowed",
}
_SCORING_ROW_KEYS = {
    "version",
    "phase",
    "prediction_freeze_sha256",
    "candidate_id",
    "epoch",
    "checkpoint_sha256",
    "seed",
    "sample_index",
    "viewpoint_id",
    "gt_observable",
    "predicted_observable",
    "geometry_valid",
    "world_xyz_error_m",
    "gt_object_position_base_m",
    "predicted_object_position_base_m",
    "raw_covariance_base_m2",
    "write_score",
    "used_for_formal_selection",
    "test_data_read",
}
_SELECTION_ARTIFACT_NAMES = (
    "config_snapshot.json",
    "source_identity.json",
    "prediction_freeze_verification.json",
    "label_input_verification.json",
    "validation_losses.json",
    "validation_loss_batches.json",
    "model_val_scoring_ledger.jsonl",
    "diagnostic_control_scoring_ledger.jsonl",
    "diagnostic_control_summary.json",
    "selection.json",
    "selection_summary.json",
)
_PREDICTION_FREEZE_COUNT_CONTRACT = {
    "selection_checkpoint_count": 8,
    "candidate_prediction_ledger_count": 8,
    "candidate_prediction_row_count": 8800,
    "candidate_loss_output_shard_count": 280,
    "diagnostic_control_prediction_ledger_count": 1,
    "diagnostic_control_prediction_row_count": 1100,
    "diagnostic_control_loss_output_shard_count": 0,
    "total_prediction_ledger_count": 9,
    "total_prediction_row_count": 9900,
    "model_val_unique_deployable_bundle_count": 100,
    "model_val_deployable_bundle_open_count": 900,
    "model_val_deployable_sample_read_count": 9900,
    "privileged_label_open_count_before_freeze": 0,
}
_SELECTION_COUNT_CONTRACT = {
    "selection_checkpoint_count": 8,
    "candidate_prediction_row_count": 8800,
    "candidate_scoring_row_count": 8800,
    "candidate_loss_output_shard_count": 280,
    "validation_loss_count": 8,
    "diagnostic_control_prediction_row_count": 1100,
    "diagnostic_control_scoring_row_count": 1100,
    "diagnostic_control_validation_loss_count": 0,
    "total_prediction_row_count": 9900,
    "total_scoring_row_count": 9900,
    "model_val_privileged_label_bundle_open_count": 100,
}


def _assert_exact_count_contract(
    value: Mapping[str, Any], *, expected: Mapping[str, int], name: str
) -> None:
    mismatches = {
        field: {"expected": expected_value, "actual": value.get(field)}
        for field, expected_value in expected.items()
        if value.get(field) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"{name} count contract 漂移: {mismatches}")


def _assert_prediction_freeze_count_contract(marker: Mapping[str, Any]) -> None:
    _assert_exact_count_contract(
        marker,
        expected=_PREDICTION_FREEZE_COUNT_CONTRACT,
        name="G2C prediction freeze",
    )


def _assert_selection_count_contract(receipt: Mapping[str, Any]) -> None:
    _assert_exact_count_contract(
        receipt,
        expected=_SELECTION_COUNT_CONTRACT,
        name="G2C selection receipt",
    )


def _expected_checkpoint_pairs() -> list[tuple[str, int]]:
    return [
        (candidate_id, epoch)
        for candidate_id in G2C_CANDIDATE_IDS
        for epoch in G2C_CANDIDATE_EPOCHS
    ]


def _prediction_rows_for_batch(
    *,
    samples: Sequence[Mapping[str, Any]],
    output: Any,
    candidate_id: str,
    epoch: int,
    checkpoint_identity: Mapping[str, Any],
    global_start: int,
    batch_index: int,
) -> list[dict[str, Any]]:
    import torch

    from robot_vla.precision.e018_p1_g2c import _measurement_covariance

    decoded = output.decode_for_control(temperature=1.0)
    predicted_uv = decoded.keypoints.normalized_uv.detach().float().cpu().numpy()
    visibility = decoded.visibility_probability.detach().float().cpu().numpy()
    projection = (
        decoded.projection_validity_probability.detach().float().cpu().numpy()
    )
    entropy = decoded.keypoints.normalized_entropy.detach().float().cpu().numpy()
    sigma = decoded.keypoint_sigma_px.detach().float().cpu().numpy()
    mask = torch.sigmoid(output.mask_logits.detach().float()).cpu().numpy()
    rows: list[dict[str, Any]] = []
    for offset, sample in enumerate(samples):
        capture = sample["capture"]
        object_uv = predicted_uv[offset, 0]
        object_mask_probability = mask_probability_at_normalized_uv(mask[offset, 0], object_uv)
        goal_mask_probability = mask_probability_at_normalized_uv(mask[offset, 1], object_uv)
        try:
            geometry = geometry_conditioning(
                normalized_uv=object_uv,
                intrinsic_cv=capture["external_intrinsic_cv"],
                base_from_camera_cv=capture["base_from_external_camera_cv"],
                image_size_hw=(128, 128),
                plane_base_z_m=0.02,
            )
            predicted_position = np.asarray(
                geometry["predicted_world_point_base_m"], dtype=np.float64
            )
            covariance = _measurement_covariance(
                geometry["local_jacobian_xy_m_per_px"], sigma[offset, 0]
            )
            geometry_valid = True
        except ValueError:
            predicted_position = None
            covariance = None
            geometry_valid = False
        evidence = ObjectWriteEvidence(
            visibility_probability=float(visibility[offset, 0]),
            projection_validity_probability=float(projection[offset]),
            object_mask_probability=float(object_mask_probability),
            goal_mask_probability=float(goal_mask_probability),
            normalized_entropy=float(entropy[offset, 0]),
            radial_sigma_px=float(np.linalg.norm(sigma[offset, 0])),
            geometry_valid=geometry_valid,
        )
        rows.append(
            {
                "version": E018_P1_G2C_PREDICTION_FREEZE_VERSION,
                "phase": "deployable-model-val-before-privileged-label-open/v1",
                "candidate_id": candidate_id,
                "epoch": epoch,
                "checkpoint_sha256": checkpoint_identity["checkpoint_sha256"],
                "checkpoint_parameter_sha256": checkpoint_identity[
                    "parameter_state_sha256"
                ],
                "checkpoint_provenance_sha256": checkpoint_identity[
                    "provenance_sha256"
                ],
                "checkpoint_model_config_sha256": checkpoint_identity[
                    "model_config_sha256"
                ],
                "row_index": global_start + offset,
                "batch_index": batch_index,
                "batch_offset": offset,
                "seed": int(capture["seed"]),
                "split": str(capture["split"]),
                "sample_index": int(capture["sample_index"]),
                "viewpoint_id": str(capture["viewpoint_id"]),
                "input_sha256": str(capture["input_sha256"]),
                "predicted_object_normalized_uv": object_uv.astype(float).tolist(),
                "predicted_goal_normalized_uv": predicted_uv[offset, 1]
                .astype(float)
                .tolist(),
                "object_visibility_probability": float(visibility[offset, 0]),
                "goal_visibility_probability": float(visibility[offset, 1]),
                "projection_validity_probability": float(projection[offset]),
                "object_normalized_entropy": float(entropy[offset, 0]),
                "object_sigma_xy_px": sigma[offset, 0].astype(float).tolist(),
                "object_mask_probability_at_prediction": float(
                    object_mask_probability
                ),
                "goal_mask_probability_at_prediction": float(goal_mask_probability),
                "predicted_observable": bool(visibility[offset, 0] >= 0.5),
                "geometry_valid": geometry_valid,
                "predicted_object_position_base_m": (
                    None if predicted_position is None else predicted_position.tolist()
                ),
                "raw_covariance_base_m2": (
                    None if covariance is None else covariance.tolist()
                ),
                "write_score": evidence.score,
                "memory_write_allowed": False,
                "actuation_allowed": False,
            }
        )
    return rows


def _loss_shard_arrays(
    *, samples: Sequence[Mapping[str, Any]], output: Any
) -> dict[str, np.ndarray]:
    import torch

    decoded_uv = output.decode_keypoints(temperature=1.0).normalized_uv
    arrays = {
        "seed": np.asarray(
            [int(sample["capture"]["seed"]) for sample in samples], dtype=np.int64
        ),
        "sample_index": np.asarray(
            [int(sample["capture"]["sample_index"]) for sample in samples],
            dtype=np.int64,
        ),
        "viewpoint_id": np.asarray(
            [str(sample["capture"]["viewpoint_id"]) for sample in samples]
        ),
        "input_sha256": np.asarray(
            [str(sample["capture"]["input_sha256"]) for sample in samples]
        ),
        "heatmap_logits": output.heatmap_logits.detach().float().cpu().numpy(),
        "mask_logits": output.mask_logits.detach().float().cpu().numpy(),
        "decoded_normalized_uv": decoded_uv.detach().float().cpu().numpy(),
        "motion_residual": output.motion_residual.detach().float().cpu().numpy(),
        "keypoint_log_variance": (
            output.keypoint_log_variance.detach().float().cpu().numpy()
        ),
        "motion_log_variance": (
            output.motion_log_variance.detach().float().cpu().numpy()
        ),
        "visibility_logits": output.visibility_logits.detach().float().cpu().numpy(),
        "projection_validity_logit": (
            output.projection_validity_logit.detach().float().cpu().numpy()
        ),
    }
    if any(
        not np.isfinite(value).all()
        for name, value in arrays.items()
        if name not in {"viewpoint_id", "input_sha256"}
        and np.issubdtype(value.dtype, np.floating)
    ):
        raise RuntimeError("G2C Phase A loss output 含 NaN/Inf")
    if any(
        arrays[name].dtype != np.float32
        for name in _LOSS_SHARD_ARRAYS
        if name
        not in {"seed", "sample_index", "viewpoint_id", "input_sha256"}
    ):
        raise RuntimeError("G2C Phase A 所有模型 loss output 必须是 float32")
    if not isinstance(decoded_uv, torch.Tensor):
        raise TypeError("G2C decoded UV 类型漂移")
    return arrays


def run_g2c_model_val_prediction_freeze(
    *,
    config_path: str | Path,
    training_output_root: str | Path,
    e016_training_output: str | Path,
    model_val_deployable_input_root: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
    decision_exit_go: bool,
) -> dict[str, Any]:
    """Phase A：只接 deployable view 与已冻结 training output，不接任何 label path。"""

    if decision_exit_go is not True:
        raise PermissionError("G2C formal model-val prediction 仍为 HOLD")
    import torch

    from robot_vla.precision.checkpoint import PrecisionCheckpointRole, load_precision_checkpoint

    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"G2C prediction-freeze output 已存在: {output}")
    config = load_g2c_formal_training_config(config_path)
    training_verification = verify_g2c_formal_training(
        config_path=config_path, output_root=training_output_root
    )
    source_identity = _git_source_identity(Path(repository_root))
    if (
        source_identity["git_commit"] != training_verification["source_git_commit"]
        or source_identity["identity_sha256"]
        != training_verification["source_identity_sha256"]
    ):
        raise RuntimeError("G2C Phase A source 必须与 formal TRAIN exact-clean source 一致")
    deployable_verification = validate_g2c_input_view(
        config_path=config_path,
        input_root=model_val_deployable_input_root,
        expected_role="model-val-deployable",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("G2C Phase A 要求 CUDA")
    training_receipt = _read_json(
        Path(training_output_root) / "training_receipt.json", "G2C training receipt"
    )
    checkpoints = training_receipt["checkpoint_inventory"]
    if [(item["candidate_id"], item["epoch"]) for item in checkpoints] != (
        _expected_checkpoint_pairs()
    ):
        raise RuntimeError("G2C Phase A checkpoint pool 漂移")
    dataset = G2CDeployableDataset(model_val_deployable_input_root, "model_val")
    if len(dataset) != _EXPECTED_SPLIT_SAMPLES["model_val"]:
        raise RuntimeError("G2C Phase A model-val sample count 必须是 1100")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _atomic_json(output / "config_snapshot.json", config)
    _atomic_json(output / "training_verification.json", training_verification)
    _atomic_json(output / "source_identity.json", source_identity)
    _atomic_json(output / "deployable_input_verification.json", deployable_verification)
    _atomic_json(output / "checkpoint_inventory.json", checkpoints)
    prediction_inventory: list[dict[str, Any]] = []
    shard_inventory: list[dict[str, Any]] = []
    sample_identity_reference: list[str] | None = None
    started = time.perf_counter()
    for checkpoint in checkpoints:
        candidate_id = str(checkpoint["candidate_id"])
        epoch = int(checkpoint["epoch"])
        checkpoint_path = (
            Path(training_output_root)
            / "candidates"
            / candidate_id
            / str(checkpoint["relative_path"])
        )
        loaded = load_precision_checkpoint(
            checkpoint_path,
            expected_checkpoint_sha256=checkpoint["checkpoint_sha256"],
            expected_provenance_sha256=checkpoint["provenance_sha256"],
            expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
        )
        if loaded.receipt.parameter_state_sha256 != checkpoint["parameter_state_sha256"]:
            raise RuntimeError("G2C Phase A checkpoint parameter identity 漂移")
        model = loaded.model.to(torch.device("cuda"))
        model.eval()
        rows: list[dict[str, Any]] = []
        checkpoint_sample_hashes: list[str] = []
        with torch.inference_mode():
            for batch_index, start in enumerate(range(0, len(dataset), 32)):
                samples = [dataset[index] for index in range(start, min(start + 32, len(dataset)))]
                image = np.stack(
                    [sample["model_inputs"]["rgb_external"].transpose(2, 0, 1) for sample in samples]
                ).astype(np.float32)
                image /= np.float32(255.0)
                state = np.stack(
                    [sample["model_inputs"]["structured_state"] for sample in samples]
                )
                motion = np.stack(
                    [sample["model_inputs"]["geometric_motion"] for sample in samples]
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    model_output = model(
                        torch.from_numpy(image).to(torch.device("cuda")),
                        torch.from_numpy(state).to(torch.device("cuda")),
                        torch.from_numpy(motion).to(torch.device("cuda")),
                    )
                arrays = _loss_shard_arrays(samples=samples, output=model_output)
                relative = (
                    Path("loss_outputs")
                    / candidate_id
                    / f"epoch-{epoch:02d}"
                    / f"batch-{batch_index:03d}.npz"
                )
                shard_path = output / relative
                _atomic_npz(shard_path, arrays)
                batch_identity = [
                    {
                        "seed": int(arrays["seed"][index]),
                        "sample_index": int(arrays["sample_index"][index]),
                        "viewpoint_id": str(arrays["viewpoint_id"][index]),
                        "input_sha256": str(arrays["input_sha256"][index]),
                    }
                    for index in range(len(samples))
                ]
                identity_sha = canonical_sha256(batch_identity)
                checkpoint_sample_hashes.append(identity_sha)
                shard_inventory.append(
                    {
                        "candidate_id": candidate_id,
                        "epoch": epoch,
                        "batch_index": batch_index,
                        "global_start": start,
                        "batch_size": len(samples),
                        "relative_path": relative.as_posix(),
                        "raw_sha256": file_sha256(shard_path),
                        "sample_identity_sha256": identity_sha,
                        "array_shapes": {
                            name: list(value.shape) for name, value in arrays.items()
                        },
                        "array_dtypes": {
                            name: str(value.dtype) for name, value in arrays.items()
                        },
                    }
                )
                rows.extend(
                    _prediction_rows_for_batch(
                        samples=samples,
                        output=model_output,
                        candidate_id=candidate_id,
                        epoch=epoch,
                        checkpoint_identity=checkpoint,
                        global_start=start,
                        batch_index=batch_index,
                    )
                )
        if len(rows) != 1100 or len(checkpoint_sample_hashes) != 35:
            raise RuntimeError("G2C Phase A 每 checkpoint 必须有 1100 rows/35 batches")
        if sample_identity_reference is None:
            sample_identity_reference = checkpoint_sample_hashes
        elif checkpoint_sample_hashes != sample_identity_reference:
            raise RuntimeError("G2C Phase A checkpoint 间 batch/sample identity 漂移")
        assert_g2c_prediction_ledger_deployable_only(rows)
        ledger_relative = Path("prediction_ledgers") / (
            f"{candidate_id}__epoch-{epoch:02d}.jsonl"
        )
        _atomic_jsonl(output / ledger_relative, rows)
        prediction_inventory.append(
            {
                "candidate_id": candidate_id,
                "epoch": epoch,
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "checkpoint_parameter_sha256": checkpoint["parameter_state_sha256"],
                "checkpoint_provenance_sha256": checkpoint["provenance_sha256"],
                "checkpoint_model_config_sha256": checkpoint["model_config_sha256"],
                "row_count": len(rows),
                "batch_count": len(checkpoint_sample_hashes),
                "relative_path": ledger_relative.as_posix(),
                "raw_sha256": file_sha256(output / ledger_relative),
                "batch_sample_identity_sha256": checkpoint_sample_hashes,
            }
        )
        del rows, model, loaded
        gc.collect()
        torch.cuda.empty_cache()
    control_parent = config["model_parent"]
    control_path = Path(e016_training_output) / "precision-formal.pt"
    control_loaded = load_precision_checkpoint(
        control_path,
        expected_checkpoint_sha256=control_parent["e016_checkpoint_sha256"],
        expected_provenance_sha256=control_parent[
            "e016_checkpoint_provenance_sha256"
        ],
        expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
    )
    if (
        control_loaded.receipt.parameter_state_sha256
        != control_parent["e016_checkpoint_parameter_sha256"]
        or control_loaded.receipt.model_config_sha256
        != control_parent["e016_checkpoint_model_config_sha256"]
        or control_loaded.provenance.training_config_sha256
        != control_parent["e016_config_sha256"]
    ):
        raise RuntimeError("G2C diagnostic CONTROL E016 checkpoint identity 漂移")
    control_identity = control_loaded.receipt.to_dict()
    control_model = control_loaded.model.to(torch.device("cuda"))
    control_model.eval()
    control_rows: list[dict[str, Any]] = []
    control_batch_hashes: list[str] = []
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, len(dataset), 32)):
            samples = [
                dataset[index]
                for index in range(start, min(start + 32, len(dataset)))
            ]
            image = np.stack(
                [
                    sample["model_inputs"]["rgb_external"].transpose(2, 0, 1)
                    for sample in samples
                ]
            ).astype(np.float32)
            image /= np.float32(255.0)
            state = np.stack(
                [sample["model_inputs"]["structured_state"] for sample in samples]
            )
            motion = np.stack(
                [sample["model_inputs"]["geometric_motion"] for sample in samples]
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                model_output = control_model(
                    torch.from_numpy(image).to(torch.device("cuda")),
                    torch.from_numpy(state).to(torch.device("cuda")),
                    torch.from_numpy(motion).to(torch.device("cuda")),
                )
            batch_identity = [
                {
                    "seed": int(sample["capture"]["seed"]),
                    "sample_index": int(sample["capture"]["sample_index"]),
                    "viewpoint_id": str(sample["capture"]["viewpoint_id"]),
                    "input_sha256": str(sample["capture"]["input_sha256"]),
                }
                for sample in samples
            ]
            control_batch_hashes.append(canonical_sha256(batch_identity))
            control_rows.extend(
                _prediction_rows_for_batch(
                    samples=samples,
                    output=model_output,
                    candidate_id=G2C_DIAGNOSTIC_CONTROL_ID,
                    epoch=12,
                    checkpoint_identity=control_identity,
                    global_start=start,
                    batch_index=batch_index,
                )
            )
    if (
        len(control_rows) != 1100
        or len(control_batch_hashes) != 35
        or control_batch_hashes != sample_identity_reference
    ):
        raise RuntimeError("G2C diagnostic CONTROL sample/batch identity 漂移")
    assert_g2c_prediction_ledger_deployable_only(control_rows)
    control_ledger_relative = "diagnostic_control_prediction_ledger.jsonl"
    _atomic_jsonl(output / control_ledger_relative, control_rows)
    control_prediction_inventory = [
        {
            "control_id": G2C_DIAGNOSTIC_CONTROL_ID,
            "epoch": 12,
            "source_role": "exact-e016-selected-epoch12-role-substitution/v1",
            "eligible_for_selection": False,
            "checkpoint_sha256": control_loaded.receipt.checkpoint_sha256,
            "checkpoint_parameter_sha256": (
                control_loaded.receipt.parameter_state_sha256
            ),
            "checkpoint_provenance_sha256": (
                control_loaded.receipt.provenance_sha256
            ),
            "checkpoint_model_config_sha256": (
                control_loaded.receipt.model_config_sha256
            ),
            "row_count": len(control_rows),
            "batch_count": len(control_batch_hashes),
            "relative_path": control_ledger_relative,
            "raw_sha256": file_sha256(output / control_ledger_relative),
            "batch_sample_identity_sha256": control_batch_hashes,
        }
    ]
    del control_rows, control_model, control_loaded
    del dataset, samples, arrays, model_output, image, state, motion
    _atomic_json(output / "prediction_inventory.json", prediction_inventory)
    _atomic_json(
        output / "diagnostic_control_prediction_inventory.json",
        control_prediction_inventory,
    )
    _atomic_json(output / "loss_output_inventory.json", shard_inventory)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    artifact_files = [
        "config_snapshot.json",
        "training_verification.json",
        "source_identity.json",
        "deployable_input_verification.json",
        "checkpoint_inventory.json",
        "prediction_inventory.json",
        "diagnostic_control_prediction_inventory.json",
        "loss_output_inventory.json",
        *(item["relative_path"] for item in prediction_inventory),
        control_ledger_relative,
        *(item["relative_path"] for item in shard_inventory),
    ]
    artifact_inventory = [
        {
            "relative_path": name,
            "raw_sha256": file_sha256(output / name),
            "size_bytes": (output / name).stat().st_size,
        }
        for name in artifact_files
    ]
    artifact_bytes_before_freeze_marker = sum(
        item["size_bytes"] for item in artifact_inventory
    )
    if (
        artifact_bytes_before_freeze_marker
        > config["protocol"]["budgets"]["artifact_bytes_max"]
    ):
        raise RuntimeError("G2C Phase A artifact 超过 20 GiB")
    marker = {
        "version": E018_P1_G2C_PREDICTION_FREEZE_VERSION,
        "status": "complete-model-val-prediction-freeze-pass",
        "classification": "deployable-only-before-label-development",
        "config_sha256": config["config_sha256"],
        "data_identity_sha256": config["data_parent"]["data_identity_sha256"],
        "training_receipt_raw_sha256": training_verification["receipt_raw_sha256"],
        "training_receipt_internal_sha256": training_verification[
            "receipt_internal_sha256"
        ],
        "source_git_commit": source_identity["git_commit"],
        "source_identity_sha256": source_identity["identity_sha256"],
        "selection_checkpoint_count": len(checkpoints),
        "candidate_prediction_ledger_count": len(prediction_inventory),
        "candidate_prediction_row_count": sum(
            item["row_count"] for item in prediction_inventory
        ),
        "candidate_loss_output_shard_count": len(shard_inventory),
        "diagnostic_control_prediction_ledger_count": 1,
        "diagnostic_control_prediction_row_count": control_prediction_inventory[0][
            "row_count"
        ],
        "diagnostic_control_loss_output_shard_count": 0,
        "total_prediction_ledger_count": len(prediction_inventory) + 1,
        "total_prediction_row_count": sum(
            item["row_count"] for item in prediction_inventory
        )
        + control_prediction_inventory[0]["row_count"],
        "per_checkpoint_batch_sizes": [*[32] * 34, 12],
        "prediction_inventory_raw_sha256": file_sha256(
            output / "prediction_inventory.json"
        ),
        "diagnostic_control_prediction_inventory_raw_sha256": file_sha256(
            output / "diagnostic_control_prediction_inventory.json"
        ),
        "loss_output_inventory_raw_sha256": file_sha256(
            output / "loss_output_inventory.json"
        ),
        "checkpoint_inventory_raw_sha256": file_sha256(
            output / "checkpoint_inventory.json"
        ),
        "artifact_inventory": artifact_inventory,
        "artifact_inventory_sha256": canonical_sha256(artifact_inventory),
        "artifact_bytes_before_freeze_marker": (
            artifact_bytes_before_freeze_marker
        ),
        "privileged_label_open_count_before_freeze": 0,
        "model_and_inference_context_destroyed_before_freeze": True,
        "model_val_unique_deployable_bundle_count": 100,
        "model_val_deployable_bundle_open_count": 900,
        "model_val_deployable_sample_read_count": 9900,
        "test_array_read_count": 0,
        "memory_read_count": 0,
        "memory_write_count": 0,
        "actuation_count": 0,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(torch.device("cuda")),
        },
        "elapsed_s": time.perf_counter() - started,
        "frozen_at_unix_ns": time.time_ns(),
    }
    marker["freeze_sha256"] = canonical_sha256(marker)
    _atomic_json(output / "prediction_freeze.json", marker)
    total_artifact_bytes = _verify_exact_regular_file_tree(
        output,
        expected_files=set(artifact_files) | {"prediction_freeze.json"},
        name="G2C Phase A prediction freeze",
    )
    if total_artifact_bytes > config["protocol"]["budgets"]["artifact_bytes_max"]:
        raise RuntimeError("G2C Phase A 完整 artifact 超过 20 GiB")
    return {
        **marker,
        "total_artifact_bytes": total_artifact_bytes,
        "verification": verify_g2c_prediction_freeze(
            config_path=config_path, output_root=output
        ),
    }


def _load_and_validate_loss_shard(
    path: Path, *, expected_batch_size: int
) -> dict[str, np.ndarray]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("G2C loss-output shard 不存在或是 symlink")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(_LOSS_SHARD_ARRAYS):
            raise RuntimeError("G2C loss-output shard array keys 漂移")
        arrays = {name: archive[name] for name in archive.files}
    expected_shapes = {
        "seed": (expected_batch_size,),
        "sample_index": (expected_batch_size,),
        "viewpoint_id": (expected_batch_size,),
        "input_sha256": (expected_batch_size,),
        "heatmap_logits": (expected_batch_size, 2, 128, 128),
        "mask_logits": (expected_batch_size, 2, 128, 128),
        "decoded_normalized_uv": (expected_batch_size, 2, 2),
        "motion_residual": (expected_batch_size, 4),
        "keypoint_log_variance": (expected_batch_size, 2, 2),
        "motion_log_variance": (expected_batch_size, 4),
        "visibility_logits": (expected_batch_size, 2),
        "projection_validity_logit": (expected_batch_size,),
    }
    for name, expected_shape in expected_shapes.items():
        value = arrays[name]
        if value.shape != expected_shape:
            raise RuntimeError(f"G2C loss-output {name} shape 漂移")
        if name in {"seed", "sample_index"}:
            if value.dtype != np.int64:
                raise RuntimeError(f"G2C loss-output {name} dtype 必须是 int64")
        elif name in {"viewpoint_id", "input_sha256"}:
            if value.dtype.kind != "U":
                raise RuntimeError(f"G2C loss-output {name} dtype 必须是 unicode")
        elif value.dtype != np.float32:
            raise RuntimeError(f"G2C loss-output {name} dtype 必须是 float32")
    if any(
        not np.isfinite(arrays[name]).all()
        for name in _LOSS_SHARD_ARRAYS
        if arrays[name].dtype == np.float32
    ):
        raise RuntimeError("G2C loss-output shard 含 NaN/Inf")
    if np.any(arrays["decoded_normalized_uv"] < 0.0) or np.any(
        arrays["decoded_normalized_uv"] > 1.0
    ):
        raise RuntimeError("G2C frozen decoded UV 超出 [0,1]")
    if any(_require_sha256(str(value), "G2C frozen input_sha256") != str(value) for value in arrays["input_sha256"]):
        raise AssertionError("unreachable")
    return arrays


def _stable_sigmoid(value: float) -> float:
    candidate = float(value)
    if candidate >= 0.0:
        return float(1.0 / (1.0 + math.exp(-candidate)))
    exponential = math.exp(candidate)
    return float(exponential / (1.0 + exponential))


def _normalized_heatmap_entropy(logits: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    shifted = values - float(np.max(values))
    exponential = np.exp(shifted)
    probability = exponential / float(np.sum(exponential))
    if probability.size == 1:
        return 0.0
    return float(
        -np.sum(probability * np.log(np.maximum(probability, 1e-12)))
        / math.log(probability.size)
    )


def _mechanical_prediction_fields_from_shard(
    arrays: Mapping[str, np.ndarray], index: int
) -> dict[str, Any]:
    object_uv = arrays["decoded_normalized_uv"][index, 0].astype(np.float64)
    goal_uv = arrays["decoded_normalized_uv"][index, 1].astype(np.float64)
    visibility = arrays["visibility_logits"][index]
    projection = arrays["projection_validity_logit"][index]
    sigma = (
        np.exp(
            np.float32(0.5) * arrays["keypoint_log_variance"][index, 0]
        ).astype(np.float32)
        * np.asarray((128.0, 128.0), dtype=np.float32)
    )
    mask = np.empty((2, 128, 128), dtype=np.float64)
    for channel in range(2):
        logits = arrays["mask_logits"][index, channel].astype(np.float64)
        positive = logits >= 0.0
        channel_probability = np.empty_like(logits)
        channel_probability[positive] = 1.0 / (
            1.0 + np.exp(-logits[positive])
        )
        exponential = np.exp(logits[~positive])
        channel_probability[~positive] = exponential / (1.0 + exponential)
        mask[channel] = channel_probability
    object_mask_probability = mask_probability_at_normalized_uv(mask[0], object_uv)
    goal_mask_probability = mask_probability_at_normalized_uv(mask[1], object_uv)
    entropy = _normalized_heatmap_entropy(arrays["heatmap_logits"][index, 0])
    radial_sigma = float(np.linalg.norm(sigma.astype(np.float64)))
    object_visibility = _stable_sigmoid(float(visibility[0]))
    projection_probability = _stable_sigmoid(float(projection))
    evidence = ObjectWriteEvidence(
        visibility_probability=object_visibility,
        projection_validity_probability=projection_probability,
        object_mask_probability=object_mask_probability,
        goal_mask_probability=goal_mask_probability,
        normalized_entropy=entropy,
        radial_sigma_px=radial_sigma,
        # score 不读取 geometry_valid；结构资格不在 Phase A ledger 中放权。
        geometry_valid=False,
    )
    return {
        "object_uv": object_uv.astype(float).tolist(),
        "goal_uv": goal_uv.astype(float).tolist(),
        "object_visibility_probability": object_visibility,
        "goal_visibility_probability": _stable_sigmoid(float(visibility[1])),
        "projection_validity_probability": projection_probability,
        "object_normalized_entropy": entropy,
        "object_sigma_xy_px": sigma.astype(float).tolist(),
        "object_mask_probability_at_prediction": object_mask_probability,
        "goal_mask_probability_at_prediction": goal_mask_probability,
        "predicted_observable": object_visibility >= 0.5,
        "write_score": evidence.score,
    }


def _float_close(actual: Any, expected: float) -> bool:
    try:
        value = float(actual)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isclose(
        value, expected, rel_tol=2e-6, abs_tol=2e-7
    )


def _assert_prediction_row_matches_frozen_output(
    row: Mapping[str, Any], frozen_output: Mapping[str, Any]
) -> None:
    sigma = row.get("object_sigma_xy_px")
    expected_sigma = frozen_output["object_sigma_xy_px"]
    if (
        row.get("predicted_object_normalized_uv") != frozen_output["object_uv"]
        or row.get("predicted_goal_normalized_uv") != frozen_output["goal_uv"]
        or not _float_close(
            row.get("object_visibility_probability"),
            frozen_output["object_visibility_probability"],
        )
        or not _float_close(
            row.get("goal_visibility_probability"),
            frozen_output["goal_visibility_probability"],
        )
        or not _float_close(
            row.get("projection_validity_probability"),
            frozen_output["projection_validity_probability"],
        )
        or not _float_close(
            row.get("object_normalized_entropy"),
            frozen_output["object_normalized_entropy"],
        )
        or not isinstance(sigma, list)
        or len(sigma) != 2
        or any(
            not _float_close(actual, expected)
            for actual, expected in zip(sigma, expected_sigma, strict=True)
        )
        or not _float_close(
            row.get("object_mask_probability_at_prediction"),
            frozen_output["object_mask_probability_at_prediction"],
        )
        or not _float_close(
            row.get("goal_mask_probability_at_prediction"),
            frozen_output["goal_mask_probability_at_prediction"],
        )
        or not _float_close(row.get("write_score"), frozen_output["write_score"])
        or row.get("predicted_observable")
        is not frozen_output["predicted_observable"]
    ):
        raise RuntimeError("G2C frozen prediction row/shard同源 output 漂移")


def verify_g2c_prediction_freeze(
    *, config_path: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """只用 freeze artifact 自证完整性；签名刻意不接 checkpoint/deployable/label。"""

    config = load_g2c_formal_training_config(config_path)
    root = Path(output_root)
    marker_path = root / "prediction_freeze.json"
    marker = _read_json(marker_path, "G2C prediction freeze")
    _require_exact_keys(
        marker,
        {
            "version",
            "status",
            "classification",
            "config_sha256",
            "data_identity_sha256",
            "training_receipt_raw_sha256",
            "training_receipt_internal_sha256",
            "source_git_commit",
            "source_identity_sha256",
            "selection_checkpoint_count",
            "candidate_prediction_ledger_count",
            "candidate_prediction_row_count",
            "candidate_loss_output_shard_count",
            "diagnostic_control_prediction_ledger_count",
            "diagnostic_control_prediction_row_count",
            "diagnostic_control_loss_output_shard_count",
            "total_prediction_ledger_count",
            "total_prediction_row_count",
            "per_checkpoint_batch_sizes",
            "prediction_inventory_raw_sha256",
            "diagnostic_control_prediction_inventory_raw_sha256",
            "loss_output_inventory_raw_sha256",
            "checkpoint_inventory_raw_sha256",
            "artifact_inventory",
            "artifact_inventory_sha256",
            "artifact_bytes_before_freeze_marker",
            "privileged_label_open_count_before_freeze",
            "model_and_inference_context_destroyed_before_freeze",
            "model_val_unique_deployable_bundle_count",
            "model_val_deployable_bundle_open_count",
            "model_val_deployable_sample_read_count",
            "test_array_read_count",
            "memory_read_count",
            "memory_write_count",
            "actuation_count",
            "environment",
            "elapsed_s",
            "frozen_at_unix_ns",
            "freeze_sha256",
        },
        "G2C prediction freeze marker",
    )
    internal = marker.get("freeze_sha256")
    unsigned = dict(marker)
    unsigned.pop("freeze_sha256", None)
    expected_batch_sizes = [32] * 34 + [12]
    _assert_prediction_freeze_count_contract(marker)
    if (
        internal != canonical_sha256(unsigned)
        or marker.get("version") != E018_P1_G2C_PREDICTION_FREEZE_VERSION
        or marker.get("status") != "complete-model-val-prediction-freeze-pass"
        or marker.get("classification")
        != "deployable-only-before-label-development"
        or marker.get("config_sha256") != config["config_sha256"]
        or marker.get("data_identity_sha256")
        != config["data_parent"]["data_identity_sha256"]
        or marker.get("per_checkpoint_batch_sizes") != expected_batch_sizes
        or marker.get("model_and_inference_context_destroyed_before_freeze") is not True
        or marker.get("test_array_read_count") != 0
        or marker.get("memory_read_count") != 0
        or marker.get("memory_write_count") != 0
        or marker.get("actuation_count") != 0
    ):
        raise RuntimeError("G2C prediction freeze status/count/permission 漂移")
    if _read_json(root / "config_snapshot.json", "G2C Phase A config snapshot") != config:
        raise RuntimeError("G2C Phase A config snapshot 漂移")
    checkpoint_inventory = _read_json_array(
        root / "checkpoint_inventory.json", "G2C frozen checkpoint inventory"
    )
    prediction_inventory = _read_json_array(
        root / "prediction_inventory.json", "G2C prediction inventory"
    )
    control_prediction_inventory = _read_json_array(
        root / "diagnostic_control_prediction_inventory.json",
        "G2C diagnostic CONTROL prediction inventory",
    )
    shard_inventory = _read_json_array(
        root / "loss_output_inventory.json", "G2C loss output inventory"
    )
    expected_pairs = _expected_checkpoint_pairs()
    if (
        [(item.get("candidate_id"), item.get("epoch")) for item in checkpoint_inventory]
        != expected_pairs
        or [(item.get("candidate_id"), item.get("epoch")) for item in prediction_inventory]
        != expected_pairs
        or file_sha256(root / "checkpoint_inventory.json")
        != marker["checkpoint_inventory_raw_sha256"]
        or file_sha256(root / "prediction_inventory.json")
        != marker["prediction_inventory_raw_sha256"]
        or file_sha256(root / "diagnostic_control_prediction_inventory.json")
        != marker["diagnostic_control_prediction_inventory_raw_sha256"]
        or file_sha256(root / "loss_output_inventory.json")
        != marker["loss_output_inventory_raw_sha256"]
    ):
        raise RuntimeError("G2C frozen inventory identity/order 漂移")
    for item in checkpoint_inventory:
        for name in (
            "checkpoint_sha256",
            "parameter_state_sha256",
            "provenance_sha256",
            "model_config_sha256",
        ):
            _require_sha256(item.get(name), f"G2C frozen checkpoint {name}")
    shards_by_pair: dict[tuple[str, int], list[dict[str, Any]]] = {
        pair: [] for pair in expected_pairs
    }
    for item in shard_inventory:
        pair = (item.get("candidate_id"), item.get("epoch"))
        if pair not in shards_by_pair:
            raise RuntimeError("G2C frozen loss shard candidate/epoch 漂移")
        shards_by_pair[pair].append(item)
    reference_batch_identity: list[str] | None = None
    reference_row_identity: list[dict[str, Any]] | None = None
    expected_regular_files = {
        "prediction_freeze.json",
        "config_snapshot.json",
        "training_verification.json",
        "source_identity.json",
        "deployable_input_verification.json",
        "checkpoint_inventory.json",
        "prediction_inventory.json",
        "diagnostic_control_prediction_inventory.json",
        "loss_output_inventory.json",
    }
    source_identity = _read_json(root / "source_identity.json", "G2C Phase A source")
    training_verification = _read_json(
        root / "training_verification.json", "G2C Phase A training verification"
    )
    if (
        source_identity.get("identity_sha256")
        != canonical_sha256(
            {
                "git_commit": source_identity.get("git_commit"),
                "source_tree_sha256": source_identity.get("source_tree_sha256"),
            }
        )
        or marker.get("source_git_commit") != source_identity.get("git_commit")
        or marker.get("source_identity_sha256")
        != source_identity.get("identity_sha256")
        or training_verification.get("source_git_commit")
        != source_identity.get("git_commit")
        or training_verification.get("source_identity_sha256")
        != source_identity.get("identity_sha256")
    ):
        raise RuntimeError("G2C Phase A/TRAIN source identity 漂移")
    for pair, prediction_item, checkpoint_item in zip(
        expected_pairs, prediction_inventory, checkpoint_inventory, strict=True
    ):
        candidate_id, epoch = pair
        shards = shards_by_pair[pair]
        if (
            len(shards) != 35
            or [item.get("batch_index") for item in shards] != list(range(35))
            or [item.get("global_start") for item in shards]
            != [index * 32 for index in range(35)]
            or [item.get("batch_size") for item in shards] != expected_batch_sizes
        ):
            raise RuntimeError("G2C 每 checkpoint 必须是 34x32 + 1x12 batches")
        batch_identity: list[str] = []
        row_identity: list[dict[str, Any]] = []
        row_outputs: list[dict[str, Any]] = []
        for item, batch_size in zip(shards, expected_batch_sizes, strict=True):
            _require_exact_keys(
                item,
                {
                    "candidate_id",
                    "epoch",
                    "batch_index",
                    "global_start",
                    "batch_size",
                    "relative_path",
                    "raw_sha256",
                    "sample_identity_sha256",
                    "array_shapes",
                    "array_dtypes",
                },
                "G2C loss-output inventory row",
            )
            relative = str(item["relative_path"])
            expected_relative = (
                f"loss_outputs/{candidate_id}/epoch-{epoch:02d}/"
                f"batch-{int(item['batch_index']):03d}.npz"
            )
            if relative != expected_relative:
                raise RuntimeError("G2C loss-output shard relative path 漂移")
            expected_regular_files.add(relative)
            path = root / relative
            if file_sha256(path) != item.get("raw_sha256"):
                raise RuntimeError("G2C loss-output shard raw SHA 漂移")
            arrays = _load_and_validate_loss_shard(path, expected_batch_size=batch_size)
            actual_shapes = {name: list(value.shape) for name, value in arrays.items()}
            actual_dtypes = {name: str(value.dtype) for name, value in arrays.items()}
            identity = [
                {
                    "seed": int(arrays["seed"][index]),
                    "sample_index": int(arrays["sample_index"][index]),
                    "viewpoint_id": str(arrays["viewpoint_id"][index]),
                    "input_sha256": str(arrays["input_sha256"][index]),
                }
                for index in range(batch_size)
            ]
            identity_sha = canonical_sha256(identity)
            if (
                actual_shapes != item.get("array_shapes")
                or actual_dtypes != item.get("array_dtypes")
                or identity_sha != item.get("sample_identity_sha256")
            ):
                raise RuntimeError("G2C loss-output shape/dtype/sample identity 漂移")
            batch_identity.append(identity_sha)
            row_identity.extend(identity)
            row_outputs.extend(
                _mechanical_prediction_fields_from_shard(arrays, index)
                for index in range(batch_size)
            )
        if reference_batch_identity is None:
            reference_batch_identity = batch_identity
            reference_row_identity = row_identity
        elif batch_identity != reference_batch_identity:
            raise RuntimeError("G2C checkpoint 间 frozen batch identity 漂移")
        ledger_relative = str(prediction_item["relative_path"])
        _require_exact_keys(
            prediction_item,
            {
                "candidate_id",
                "epoch",
                "checkpoint_sha256",
                "checkpoint_parameter_sha256",
                "checkpoint_provenance_sha256",
                "checkpoint_model_config_sha256",
                "row_count",
                "batch_count",
                "relative_path",
                "raw_sha256",
                "batch_sample_identity_sha256",
            },
            "G2C prediction inventory row",
        )
        expected_ledger_relative = (
            f"prediction_ledgers/{candidate_id}__epoch-{epoch:02d}.jsonl"
        )
        if ledger_relative != expected_ledger_relative:
            raise RuntimeError("G2C prediction ledger relative path 漂移")
        expected_regular_files.add(ledger_relative)
        ledger_path = root / ledger_relative
        if (
            file_sha256(ledger_path) != prediction_item.get("raw_sha256")
            or prediction_item.get("row_count") != 1100
            or prediction_item.get("batch_count") != 35
            or prediction_item.get("batch_sample_identity_sha256") != batch_identity
            or prediction_item.get("checkpoint_sha256")
            != checkpoint_item.get("checkpoint_sha256")
            or prediction_item.get("checkpoint_parameter_sha256")
            != checkpoint_item.get("parameter_state_sha256")
            or prediction_item.get("checkpoint_provenance_sha256")
            != checkpoint_item.get("provenance_sha256")
            or prediction_item.get("checkpoint_model_config_sha256")
            != checkpoint_item.get("model_config_sha256")
        ):
            raise RuntimeError("G2C prediction ledger inventory/checkpoint 漂移")
        rows = _read_jsonl(ledger_path, "G2C frozen prediction ledger")
        assert_g2c_prediction_ledger_deployable_only(rows)
        if len(rows) != 1100:
            raise RuntimeError("G2C frozen prediction ledger 必须有 1100 rows")
        for index, (row, identity, frozen_output) in enumerate(
            zip(rows, row_identity, row_outputs, strict=True)
        ):
            _require_exact_keys(row, _PREDICTION_ROW_KEYS, "G2C prediction row")
            _assert_prediction_row_matches_frozen_output(row, frozen_output)
            geometry_valid = row.get("geometry_valid")
            predicted_position = row.get("predicted_object_position_base_m")
            raw_covariance = row.get("raw_covariance_base_m2")
            if not isinstance(geometry_valid, bool):
                raise TypeError("G2C prediction geometry_valid 类型漂移")
            if geometry_valid:
                position_array = np.asarray(predicted_position, dtype=np.float64)
                covariance_array = np.asarray(raw_covariance, dtype=np.float64)
                if (
                    position_array.shape != (3,)
                    or covariance_array.shape != (3, 3)
                    or not np.isfinite(position_array).all()
                    or not np.isfinite(covariance_array).all()
                    or not np.allclose(covariance_array, covariance_array.T)
                    or np.min(np.linalg.eigvalsh(covariance_array)) < -1e-12
                ):
                    raise RuntimeError("G2C prediction geometry/covariance 漂移")
            elif predicted_position is not None or raw_covariance is not None:
                raise RuntimeError("G2C invalid geometry 不得保留 position/covariance")
            if (
                row.get("version") != E018_P1_G2C_PREDICTION_FREEZE_VERSION
                or row.get("phase")
                != "deployable-model-val-before-privileged-label-open/v1"
                or row.get("candidate_id") != candidate_id
                or row.get("epoch") != epoch
                or row.get("row_index") != index
                or row.get("batch_index") != index // 32
                or row.get("batch_offset") != index % 32
                or any(row.get(name) != value for name, value in identity.items())
                or row.get("split") != "model_val"
                or row.get("checkpoint_sha256")
                != checkpoint_item.get("checkpoint_sha256")
                or row.get("checkpoint_parameter_sha256")
                != checkpoint_item.get("parameter_state_sha256")
                or row.get("checkpoint_provenance_sha256")
                != checkpoint_item.get("provenance_sha256")
                or row.get("checkpoint_model_config_sha256")
                != checkpoint_item.get("model_config_sha256")
                or row.get("memory_write_allowed") is not False
                or row.get("actuation_allowed") is not False
            ):
                raise RuntimeError("G2C frozen prediction row/shard同源 identity 漂移")
    if len(control_prediction_inventory) != 1:
        raise RuntimeError("G2C diagnostic CONTROL inventory 必须有且仅有一项")
    control_item = control_prediction_inventory[0]
    _require_exact_keys(
        control_item,
        {
            "control_id",
            "epoch",
            "source_role",
            "eligible_for_selection",
            "checkpoint_sha256",
            "checkpoint_parameter_sha256",
            "checkpoint_provenance_sha256",
            "checkpoint_model_config_sha256",
            "row_count",
            "batch_count",
            "relative_path",
            "raw_sha256",
            "batch_sample_identity_sha256",
        },
        "G2C diagnostic CONTROL inventory row",
    )
    parent = config["model_parent"]
    control_relative = "diagnostic_control_prediction_ledger.jsonl"
    if (
        control_item.get("control_id") != G2C_DIAGNOSTIC_CONTROL_ID
        or control_item.get("epoch") != 12
        or control_item.get("source_role")
        != "exact-e016-selected-epoch12-role-substitution/v1"
        or control_item.get("eligible_for_selection") is not False
        or control_item.get("checkpoint_sha256")
        != parent["e016_checkpoint_sha256"]
        or control_item.get("checkpoint_parameter_sha256")
        != parent["e016_checkpoint_parameter_sha256"]
        or control_item.get("checkpoint_provenance_sha256")
        != parent["e016_checkpoint_provenance_sha256"]
        or control_item.get("checkpoint_model_config_sha256")
        != parent["e016_checkpoint_model_config_sha256"]
        or control_item.get("row_count") != 1100
        or control_item.get("batch_count") != 35
        or control_item.get("relative_path") != control_relative
        or control_item.get("batch_sample_identity_sha256")
        != reference_batch_identity
    ):
        raise RuntimeError("G2C diagnostic CONTROL identity/count/role 漂移")
    control_ledger_path = root / control_relative
    expected_regular_files.add(control_relative)
    if file_sha256(control_ledger_path) != control_item.get("raw_sha256"):
        raise RuntimeError("G2C diagnostic CONTROL ledger raw SHA 漂移")
    control_rows = _read_jsonl(
        control_ledger_path, "G2C diagnostic CONTROL prediction ledger"
    )
    assert_g2c_prediction_ledger_deployable_only(control_rows)
    if len(control_rows) != 1100 or reference_row_identity is None:
        raise RuntimeError("G2C diagnostic CONTROL prediction row count 漂移")
    for index, (row, identity) in enumerate(
        zip(control_rows, reference_row_identity, strict=True)
    ):
        _require_exact_keys(row, _PREDICTION_ROW_KEYS, "G2C CONTROL prediction row")
        probability_fields = (
            "object_visibility_probability",
            "goal_visibility_probability",
            "projection_validity_probability",
            "object_normalized_entropy",
            "object_mask_probability_at_prediction",
            "goal_mask_probability_at_prediction",
            "write_score",
        )
        try:
            probabilities = [float(row[name]) for name in probability_fields]
            object_uv = np.asarray(
                row["predicted_object_normalized_uv"], dtype=np.float64
            )
            goal_uv = np.asarray(
                row["predicted_goal_normalized_uv"], dtype=np.float64
            )
            sigma = np.asarray(row["object_sigma_xy_px"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("G2C CONTROL compact prediction 类型漂移") from error
        if (
            row.get("version") != E018_P1_G2C_PREDICTION_FREEZE_VERSION
            or row.get("phase")
            != "deployable-model-val-before-privileged-label-open/v1"
            or row.get("candidate_id") != G2C_DIAGNOSTIC_CONTROL_ID
            or row.get("epoch") != 12
            or row.get("checkpoint_sha256")
            != parent["e016_checkpoint_sha256"]
            or row.get("checkpoint_parameter_sha256")
            != parent["e016_checkpoint_parameter_sha256"]
            or row.get("checkpoint_provenance_sha256")
            != parent["e016_checkpoint_provenance_sha256"]
            or row.get("checkpoint_model_config_sha256")
            != parent["e016_checkpoint_model_config_sha256"]
            or row.get("row_index") != index
            or row.get("batch_index") != index // 32
            or row.get("batch_offset") != index % 32
            or any(row.get(name) != value for name, value in identity.items())
            or row.get("split") != "model_val"
            or object_uv.shape != (2,)
            or goal_uv.shape != (2,)
            or sigma.shape != (2,)
            or not np.isfinite(object_uv).all()
            or not np.isfinite(goal_uv).all()
            or not np.isfinite(sigma).all()
            or np.any(object_uv < 0.0)
            or np.any(object_uv > 1.0)
            or np.any(goal_uv < 0.0)
            or np.any(goal_uv > 1.0)
            or np.any(sigma < 0.0)
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities)
            or not isinstance(row.get("predicted_observable"), bool)
            or row.get("predicted_observable")
            is not (float(row["object_visibility_probability"]) >= 0.5)
            or row.get("memory_write_allowed") is not False
            or row.get("actuation_allowed") is not False
        ):
            raise RuntimeError("G2C diagnostic CONTROL prediction row 漂移")
    artifact_inventory = marker.get("artifact_inventory")
    if not isinstance(artifact_inventory, list):
        raise TypeError("G2C freeze artifact inventory 缺失")
    expected_artifact_paths = expected_regular_files - {"prediction_freeze.json"}
    if [item.get("relative_path") for item in artifact_inventory] != [
        "config_snapshot.json",
        "training_verification.json",
        "source_identity.json",
        "deployable_input_verification.json",
        "checkpoint_inventory.json",
        "prediction_inventory.json",
        "diagnostic_control_prediction_inventory.json",
        "loss_output_inventory.json",
        *(item["relative_path"] for item in prediction_inventory),
        control_relative,
        *(item["relative_path"] for item in shard_inventory),
    ]:
        raise RuntimeError("G2C freeze artifact inventory order 漂移")
    if {item["relative_path"] for item in artifact_inventory} != expected_artifact_paths:
        raise RuntimeError("G2C freeze artifact inventory coverage 漂移")
    for item in artifact_inventory:
        _require_exact_keys(
            item,
            {"relative_path", "raw_sha256", "size_bytes"},
            "G2C freeze artifact inventory row",
        )
        path = root / item["relative_path"]
        if (
            path.is_symlink()
            or file_sha256(path) != item.get("raw_sha256")
            or path.stat().st_size != item.get("size_bytes")
        ):
            raise RuntimeError("G2C freeze artifact file identity 漂移")
    if (
        marker.get("artifact_inventory_sha256") != canonical_sha256(artifact_inventory)
        or marker.get("artifact_bytes_before_freeze_marker")
        != sum(int(item["size_bytes"]) for item in artifact_inventory)
    ):
        raise RuntimeError("G2C freeze artifact aggregate identity 漂移")
    total_artifact_bytes = _verify_exact_regular_file_tree(
        root,
        expected_files=expected_regular_files,
        name="G2C Phase A prediction freeze",
    )
    if total_artifact_bytes > config["protocol"]["budgets"]["artifact_bytes_max"]:
        raise RuntimeError("G2C freeze 完整 artifact byte budget 漂移")
    result = {
        "version": E018_P1_G2C_PREDICTION_FREEZE_VERSION,
        "status": marker["status"],
        "verified": True,
        "config_sha256": config["config_sha256"],
        "data_identity_sha256": marker["data_identity_sha256"],
        "freeze_raw_sha256": file_sha256(marker_path),
        "freeze_internal_sha256": internal,
        "selection_checkpoint_count": 8,
        "candidate_prediction_ledger_count": 8,
        "candidate_prediction_row_count": 8800,
        "candidate_loss_output_shard_count": 280,
        "diagnostic_control_prediction_ledger_count": 1,
        "diagnostic_control_prediction_row_count": 1100,
        "total_prediction_ledger_count": 9,
        "total_prediction_row_count": 9900,
        "batch_sizes_per_checkpoint": expected_batch_sizes,
        "privileged_label_open_count_before_freeze": 0,
        "checkpoint_and_output_inventory_verified": True,
        "total_artifact_bytes": total_artifact_bytes,
    }
    result["verification_sha256"] = canonical_sha256(result)
    return result


def _load_model_val_labels(label_input_root: Path) -> dict[tuple[int, int, str], dict[str, Any]]:
    rows = _read_jsonl(
        label_input_root / "privileged_labels" / "manifest.jsonl",
        "G2C model-val privileged manifest",
    )
    if len(rows) != 100:
        raise RuntimeError("G2C model-val label view 必须有 100 bundles")
    result: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in rows:
        _require_exact_keys(
            row,
            {
                "manifest_schema_version",
                "split",
                "seed",
                "sample_count",
                "view_order",
                "schema_version",
                "file",
                "sha256",
                "contains_model_input_rgb",
                "source_deployable_file",
                "source_deployable_sha256",
            },
            "G2C model-val label manifest row",
        )
        if (
            row["manifest_schema_version"] != G2C_MANIFEST_SCHEMA_VERSION
            or row["schema_version"] != G2C_LABEL_SCHEMA_VERSION
            or row["split"] != "model_val"
            or row["sample_count"] != 11
            or tuple(row["view_order"]) != G2C_VIEW_ORDER
            or row["contains_model_input_rgb"] is not False
        ):
            raise RuntimeError("G2C model-val label manifest schema/role 漂移")
        path = _resolve_artifact_file(
            label_input_root / "privileged_labels", str(row["file"])
        )
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("G2C model-val label bundle identity 漂移")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise RuntimeError("G2C model-val label bundle SHA-256 漂移")
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != set(_LABEL_ARRAYS):
                raise RuntimeError("G2C model-val label bundle arrays 漂移")
            arrays = {name: archive[name] for name in archive.files}
        seed = int(row["seed"])
        if (
            arrays["seed"].shape != (11,)
            or arrays["seed"].dtype != np.int64
            or not np.all(arrays["seed"] == seed)
            or arrays["source_sample_index"].dtype != np.int64
            or not np.array_equal(
                arrays["source_sample_index"], np.arange(11, dtype=np.int64)
            )
            or tuple(str(value) for value in arrays["viewpoint_id"]) != G2C_VIEW_ORDER
            or arrays["normalized_uv"].shape != (11, 2, 2)
            or arrays["normalized_uv"].dtype != np.float32
            or arrays["keypoint_observable"].shape != (11, 2)
            or arrays["keypoint_observable"].dtype != np.bool_
            or arrays["keypoint_projection_valid"].shape != (11, 2)
            or arrays["keypoint_projection_valid"].dtype != np.bool_
            or arrays["object_mask"].shape != (11, 128, 128)
            or arrays["object_mask"].dtype != np.bool_
            or arrays["goal_mask"].shape != (11, 128, 128)
            or arrays["goal_mask"].dtype != np.bool_
            or arrays["object_position_base_m"].shape != (11, 3)
            or arrays["object_position_base_m"].dtype != np.float32
        ):
            raise RuntimeError("G2C model-val label bundle shape/dtype/order 漂移")
        if (
            not np.isfinite(arrays["normalized_uv"]).all()
            or not np.isfinite(arrays["object_position_base_m"]).all()
            or np.any(
                arrays["keypoint_observable"]
                & ~arrays["keypoint_projection_valid"]
            )
        ):
            raise RuntimeError("G2C model-val label finite/observability 漂移")
        for sample_index, viewpoint_id in enumerate(G2C_VIEW_ORDER):
            key = (seed, sample_index, viewpoint_id)
            if key in result:
                raise RuntimeError("G2C model-val label row identity 重复")
            result[key] = {
                "normalized_uv": arrays["normalized_uv"][sample_index].copy(),
                "keypoint_observable": arrays["keypoint_observable"][
                    sample_index
                ].copy(),
                "keypoint_projection_valid": arrays["keypoint_projection_valid"][
                    sample_index
                ].copy(),
                "object_mask": arrays["object_mask"][sample_index].copy(),
                "goal_mask": arrays["goal_mask"][sample_index].copy(),
                "object_position_base_m": arrays["object_position_base_m"][
                    sample_index
                ].copy(),
            }
    if len(result) != 1100:
        raise RuntimeError("G2C model-val label row count 必须是 1100")
    return result


def _build_frozen_batch_supervision(
    identities: Sequence[tuple[int, int, str]],
    labels: Mapping[tuple[int, int, str], Mapping[str, Any]],
) -> Any:
    import torch

    from robot_vla.precision.losses import PrecisionSupervision, build_gaussian_heatmaps

    selected = [labels[identity] for identity in identities]
    observable = np.stack([item["keypoint_observable"] for item in selected])
    normalized_uv = np.stack([item["normalized_uv"] for item in selected]).copy()
    normalized_uv[~observable] = 0.0
    normalized_uv_tensor = torch.from_numpy(normalized_uv.astype(np.float32, copy=False))
    observable_tensor = torch.from_numpy(observable.astype(np.bool_, copy=False))
    projection = np.stack([item["keypoint_projection_valid"] for item in selected])
    masks = np.stack(
        [
            np.stack((item["object_mask"], item["goal_mask"]))
            for item in selected
        ]
    ).astype(np.float32)
    return PrecisionSupervision(
        heatmap_targets=build_gaussian_heatmaps(
            normalized_uv_tensor,
            observable_tensor,
            (128, 128),
            sigma_px=1.5,
        ),
        mask_targets=torch.from_numpy(masks),
        normalized_uv_targets=normalized_uv_tensor,
        keypoint_valid=observable_tensor,
        keypoint_observable=observable_tensor.clone(),
        motion_residual_targets=torch.zeros(len(selected), 4, dtype=torch.float32),
        motion_valid=torch.zeros(len(selected), 4, dtype=torch.bool),
        projection_valid=torch.from_numpy(
            np.asarray(projection.all(axis=1), dtype=np.bool_)
        ),
    )


def _score_frozen_loss_outputs(
    *,
    freeze_root: Path,
    shard_inventory: Sequence[Mapping[str, Any]],
    labels: Mapping[tuple[int, int, str], Mapping[str, Any]],
    loss_config: Any,
) -> tuple[dict[tuple[str, int], float], list[dict[str, Any]]]:
    import torch

    from robot_vla.precision.losses import precision_unet_loss
    from robot_vla.precision.model import PrecisionUNetOutput

    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {
        pair: [] for pair in _expected_checkpoint_pairs()
    }
    for item in shard_inventory:
        grouped[(str(item["candidate_id"]), int(item["epoch"]))].append(item)
    validation_losses: dict[tuple[str, int], float] = {}
    batch_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for pair in _expected_checkpoint_pairs():
            weighted = {name: 0.0 for name in _LOSS_COMPONENT_NAMES}
            sample_total = 0
            for item in grouped[pair]:
                batch_size = int(item["batch_size"])
                shard_path = freeze_root / str(item["relative_path"])
                if file_sha256(shard_path) != item["raw_sha256"]:
                    raise RuntimeError("G2C loss shard 在 verify→score 间发生漂移")
                arrays = _load_and_validate_loss_shard(
                    shard_path,
                    expected_batch_size=batch_size,
                )
                identities = [
                    (
                        int(arrays["seed"][index]),
                        int(arrays["sample_index"][index]),
                        str(arrays["viewpoint_id"][index]),
                    )
                    for index in range(batch_size)
                ]
                if any(identity not in labels for identity in identities):
                    raise RuntimeError("G2C frozen output 与 label row identity 不一致")
                supervision = _build_frozen_batch_supervision(identities, labels)
                output = PrecisionUNetOutput(
                    heatmap_logits=torch.from_numpy(arrays["heatmap_logits"]),
                    mask_logits=torch.from_numpy(arrays["mask_logits"]),
                    # Frozen UV 已包含真实 dense-offset decode；空 sentinel 不参与任何计算。
                    subpixel_offsets=torch.empty(0, dtype=torch.float32),
                    motion_residual=torch.from_numpy(arrays["motion_residual"]),
                    keypoint_log_variance=torch.from_numpy(
                        arrays["keypoint_log_variance"]
                    ),
                    motion_log_variance=torch.from_numpy(
                        arrays["motion_log_variance"]
                    ),
                    visibility_logits=torch.from_numpy(arrays["visibility_logits"]),
                    projection_validity_logit=torch.from_numpy(
                        arrays["projection_validity_logit"]
                    ),
                )
                loss = precision_unet_loss(
                    output,
                    supervision,
                    loss_config,
                    frozen_decoded_normalized_uv=torch.from_numpy(
                        arrays["decoded_normalized_uv"]
                    ),
                )
                values = {
                    name: float(getattr(loss, name).detach().item())
                    for name in _LOSS_COMPONENT_NAMES
                }
                if any(not math.isfinite(value) for value in values.values()):
                    raise RuntimeError("G2C frozen validation loss 含 NaN/Inf")
                for name, value in values.items():
                    weighted[name] += value * batch_size
                sample_total += batch_size
                batch_rows.append(
                    {
                        "candidate_id": pair[0],
                        "epoch": pair[1],
                        "batch_index": int(item["batch_index"]),
                        "batch_size": batch_size,
                        "sample_identity_sha256": item["sample_identity_sha256"],
                        "loss": values,
                        "frozen_decoded_uv_override_used": True,
                        "dense_subpixel_offsets_loaded": False,
                    }
                )
            if sample_total != 1100:
                raise RuntimeError("G2C validation loss 必须按 1100 样本聚合")
            aggregate = {name: weighted[name] / 1100.0 for name in _LOSS_COMPONENT_NAMES}
            validation_losses[pair] = aggregate["loss"]
            batch_rows.append(
                {
                    "candidate_id": pair[0],
                    "epoch": pair[1],
                    "aggregate_sample_count": 1100,
                    "aggregation": "sum(batch_loss*actual_batch_size)/1100",
                    "aggregate_loss": aggregate,
                    "frozen_decoded_uv_override_used": True,
                    "dense_subpixel_offsets_loaded": False,
                }
            )
    return validation_losses, batch_rows


def _score_prediction_rows(
    *,
    freeze_root: Path,
    prediction_inventory: Sequence[Mapping[str, Any]],
    labels: Mapping[tuple[int, int, str], Mapping[str, Any]],
    freeze_sha256: str,
    used_for_formal_selection: bool,
    expected_row_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in prediction_inventory:
        ledger_path = freeze_root / str(item["relative_path"])
        if file_sha256(ledger_path) != item["raw_sha256"]:
            raise RuntimeError("G2C prediction ledger 在 verify→score 间发生漂移")
        predictions = _read_jsonl(
            ledger_path, "G2C frozen prediction ledger"
        )
        for prediction in predictions:
            identity = (
                int(prediction["seed"]),
                int(prediction["sample_index"]),
                str(prediction["viewpoint_id"]),
            )
            if identity not in labels:
                raise RuntimeError("G2C prediction/label identity 漂移")
            label = labels[identity]
            predicted_value = prediction["predicted_object_position_base_m"]
            predicted_position = (
                None
                if predicted_value is None
                else np.asarray(predicted_value, dtype=np.float64)
            )
            gt_position = np.asarray(label["object_position_base_m"], dtype=np.float64)
            world_error = (
                None
                if predicted_position is None
                else float(np.linalg.norm(predicted_position - gt_position))
            )
            rows.append(
                {
                    "version": E018_P1_G2C_SELECTION_RESULT_VERSION,
                    "phase": "privileged-score-after-complete-prediction-freeze/v1",
                    "prediction_freeze_sha256": freeze_sha256,
                    "candidate_id": prediction["candidate_id"],
                    "epoch": prediction["epoch"],
                    "checkpoint_sha256": prediction["checkpoint_sha256"],
                    "seed": identity[0],
                    "sample_index": identity[1],
                    "viewpoint_id": identity[2],
                    "gt_observable": bool(label["keypoint_observable"][0]),
                    "predicted_observable": bool(prediction["predicted_observable"]),
                    "geometry_valid": bool(prediction["geometry_valid"]),
                    "world_xyz_error_m": world_error,
                    "gt_object_position_base_m": gt_position.tolist(),
                    "predicted_object_position_base_m": predicted_value,
                    "raw_covariance_base_m2": prediction["raw_covariance_base_m2"],
                    "write_score": prediction["write_score"],
                    "used_for_formal_selection": used_for_formal_selection,
                    "test_data_read": False,
                }
            )
    if len(rows) != expected_row_count:
        raise RuntimeError(
            f"G2C privileged scoring ledger 必须有 {expected_row_count} rows"
        )
    return rows


def _score_select_g2c_model_val_consumed(
    *,
    config: Mapping[str, Any],
    freeze_verification: Mapping[str, Any],
    freeze_root: Path,
    freeze_marker: Mapping[str, Any],
    label_verification: Mapping[str, Any],
    labels: Mapping[tuple[int, int, str], Mapping[str, Any]],
    output: Path,
    source_identity: Mapping[str, Any],
    label_open_started_at: int,
) -> dict[str, Any]:
    """首次 label array open 后的不可重试计算。"""
    prediction_inventory = _read_json_array(
        freeze_root / "prediction_inventory.json", "G2C prediction inventory"
    )
    control_prediction_inventory = _read_json_array(
        freeze_root / "diagnostic_control_prediction_inventory.json",
        "G2C diagnostic CONTROL prediction inventory",
    )
    shard_inventory = _read_json_array(
        freeze_root / "loss_output_inventory.json", "G2C loss output inventory"
    )
    from robot_vla.precision.losses import PrecisionLossConfig

    loss_values = dict(config["protocol"]["loss"])
    loss_values.pop("heatmap_sigma_px")
    loss_config = PrecisionLossConfig(**loss_values)
    validation_losses, loss_rows = _score_frozen_loss_outputs(
        freeze_root=freeze_root,
        shard_inventory=shard_inventory,
        labels=labels,
        loss_config=loss_config,
    )
    scoring_rows = _score_prediction_rows(
        freeze_root=freeze_root,
        prediction_inventory=prediction_inventory,
        labels=labels,
        freeze_sha256=freeze_marker["freeze_sha256"],
        used_for_formal_selection=True,
        expected_row_count=8800,
    )
    control_scoring_rows = _score_prediction_rows(
        freeze_root=freeze_root,
        prediction_inventory=control_prediction_inventory,
        labels=labels,
        freeze_sha256=freeze_marker["freeze_sha256"],
        used_for_formal_selection=False,
        expected_row_count=1100,
    )
    selection = select_g2c_checkpoint(scoring_rows, validation_losses=validation_losses)
    selected = selection.get("selected")
    if selected is not None:
        matches = [
            item
            for item in _read_json_array(
                freeze_root / "checkpoint_inventory.json", "G2C checkpoint inventory"
            )
            if item["candidate_id"] == selected["candidate_id"]
            and item["epoch"] == selected["epoch"]
        ]
        if len(matches) != 1:
            raise RuntimeError("G2C selected checkpoint identity 不唯一")
        selection["selected_checkpoint_identity"] = matches[0]
    else:
        selection["selected_checkpoint_identity"] = None
    validation_loss_rows = [
        {
            "candidate_id": candidate_id,
            "epoch": epoch,
            "validation_loss": validation_losses[(candidate_id, epoch)],
            "sample_count": 1100,
            "batch_count": 35,
            "aggregation": "sum(batch_loss*actual_batch_size)/1100",
        }
        for candidate_id, epoch in _expected_checkpoint_pairs()
    ]
    control_view_summaries = [
        summarize_g2c_model_val_view(
            [
                row
                for row in control_scoring_rows
                if row["viewpoint_id"] == viewpoint_id
            ],
            viewpoint_id=viewpoint_id,
        )
        for viewpoint_id in G2C_VIEW_ORDER
    ]
    control_item = control_prediction_inventory[0]
    control_summary = {
        "version": E018_P1_G2C_SELECTION_RESULT_VERSION,
        "status": "complete-diagnostic-control-score-no-selection",
        "control_id": G2C_DIAGNOSTIC_CONTROL_ID,
        "source_role": "exact-e016-selected-epoch12-role-substitution/v1",
        "checkpoint_sha256": control_item["checkpoint_sha256"],
        "checkpoint_parameter_sha256": control_item[
            "checkpoint_parameter_sha256"
        ],
        "checkpoint_provenance_sha256": control_item[
            "checkpoint_provenance_sha256"
        ],
        "checkpoint_model_config_sha256": control_item[
            "checkpoint_model_config_sha256"
        ],
        "scoring_row_count": len(control_scoring_rows),
        "view_summaries": control_view_summaries,
        "eligible_non_home_view_count": sum(
            bool(item["eligible"])
            for item in control_view_summaries
            if item["viewpoint_id"] != G2C_VIEW_ORDER[0]
        ),
        "validation_loss_computed": False,
        "eligible_for_selection": False,
        "used_for_formal_selection": False,
    }
    _atomic_json(output / "label_input_verification.json", label_verification)
    _atomic_json(output / "validation_losses.json", validation_loss_rows)
    _atomic_json(output / "validation_loss_batches.json", loss_rows)
    _atomic_jsonl(output / "model_val_scoring_ledger.jsonl", scoring_rows)
    _atomic_jsonl(
        output / "diagnostic_control_scoring_ledger.jsonl",
        control_scoring_rows,
    )
    _atomic_json(output / "diagnostic_control_summary.json", control_summary)
    _atomic_json(output / "selection.json", selection)
    summary = {
        "version": E018_P1_G2C_SELECTION_RESULT_VERSION,
        "status": selection["status"],
        "classification": "development-only-model-val-no-test-no-actuation",
        "config_sha256": config["config_sha256"],
        "data_identity_sha256": config["data_parent"]["data_identity_sha256"],
        "source_git_commit": source_identity["git_commit"],
        "source_identity_sha256": source_identity["identity_sha256"],
        "prediction_freeze_raw_sha256": freeze_verification["freeze_raw_sha256"],
        "prediction_freeze_internal_sha256": freeze_verification[
            "freeze_internal_sha256"
        ],
        "prediction_freeze_fully_verified_before_label_open": True,
        "label_open_started_after_freeze_marker": (
            label_open_started_at > int(freeze_marker["frozen_at_unix_ns"])
        ),
        "selection_checkpoint_count": 8,
        "candidate_prediction_row_count": 8800,
        "candidate_scoring_row_count": len(scoring_rows),
        "candidate_loss_output_shard_count": 280,
        "validation_loss_count": 8,
        "diagnostic_control_prediction_row_count": 1100,
        "diagnostic_control_scoring_row_count": len(control_scoring_rows),
        "diagnostic_control_validation_loss_count": 0,
        "total_prediction_row_count": 9900,
        "total_scoring_row_count": len(scoring_rows) + len(control_scoring_rows),
        "diagnostic_control_summary": control_summary,
        "model_val_privileged_label_bundle_open_count": 100,
        "frozen_decoded_uv_override_used": True,
        "dense_subpixel_offsets_loaded": False,
        "selected": selection.get("selected"),
        "selected_checkpoint_identity": selection["selected_checkpoint_identity"],
        "test_array_read_count": 0,
        "memory_read_count": 0,
        "memory_write_count": 0,
        "actuation_count": 0,
    }
    if not summary["label_open_started_after_freeze_marker"]:
        raise RuntimeError("G2C model-val label 在 freeze marker 前打开")
    _atomic_json(output / "selection_summary.json", summary)
    artifact_names = _SELECTION_ARTIFACT_NAMES
    receipt = {
        "version": E018_P1_G2C_SELECTION_RESULT_VERSION,
        "status": summary["status"],
        "classification": summary["classification"],
        "config_sha256": config["config_sha256"],
        "data_identity_sha256": summary["data_identity_sha256"],
        "source_git_commit": summary["source_git_commit"],
        "source_identity_sha256": summary["source_identity_sha256"],
        "prediction_freeze_raw_sha256": summary["prediction_freeze_raw_sha256"],
        "prediction_freeze_internal_sha256": summary[
            "prediction_freeze_internal_sha256"
        ],
        "selected": summary["selected"],
        "selected_checkpoint_identity": summary["selected_checkpoint_identity"],
        "selection_checkpoint_count": 8,
        "candidate_prediction_row_count": 8800,
        "candidate_scoring_row_count": 8800,
        "candidate_loss_output_shard_count": 280,
        "validation_loss_count": 8,
        "diagnostic_control_prediction_row_count": 1100,
        "diagnostic_control_scoring_row_count": 1100,
        "diagnostic_control_validation_loss_count": 0,
        "total_prediction_row_count": 9900,
        "total_scoring_row_count": 9900,
        "model_val_privileged_label_bundle_open_count": 100,
        "test_array_read_count": 0,
        "memory_read_count": 0,
        "memory_write_count": 0,
        "actuation_count": 0,
        "artifact_sha256": {
            name: file_sha256(output / name) for name in artifact_names
        },
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_json(output / "selection_receipt.json", receipt)
    return {**summary, "receipt": receipt}


def score_select_g2c_model_val(
    *,
    config_path: str | Path,
    prediction_freeze_root: str | Path,
    model_val_label_input_root: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
    decision_exit_go: bool,
) -> dict[str, Any]:
    """Phase B：签名刻意不接 model、checkpoint 或 deployable 输入。"""

    if decision_exit_go is not True:
        raise PermissionError("G2C model-val label scoring 仍为 HOLD")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"G2C score-select output 已存在: {output}")
    config = load_g2c_formal_training_config(config_path)
    # 在首次 label array open 前完成所有可判定 preflight。
    freeze_verification = verify_g2c_prediction_freeze(
        config_path=config_path, output_root=prediction_freeze_root
    )
    freeze_root = Path(prediction_freeze_root)
    freeze_marker = _read_json(freeze_root / "prediction_freeze.json", "G2C freeze marker")
    source_identity = _git_source_identity(Path(repository_root))
    if (
        source_identity["git_commit"] != freeze_marker.get("source_git_commit")
        or source_identity["identity_sha256"]
        != freeze_marker.get("source_identity_sha256")
    ):
        raise RuntimeError("G2C Phase B source 必须与 Phase A exact-clean source 一致")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 1024**3:
        raise RuntimeError("G2C Phase B 可用磁盘不足 1 GiB，拒绝消费 label")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    _atomic_json(output / "config_snapshot.json", config)
    _atomic_json(output / "source_identity.json", source_identity)
    _atomic_json(output / "prediction_freeze_verification.json", freeze_verification)
    phase_state = {
        "version": E018_P1_G2C_SELECTION_RESULT_VERSION,
        "status": "pre-label-freeze-verified",
        "config_sha256": config["config_sha256"],
        "prediction_freeze_internal_sha256": freeze_verification[
            "freeze_internal_sha256"
        ],
        "source_identity_sha256": source_identity["identity_sha256"],
        "label_array_consumed": False,
        "rerun_under_same_identity_allowed": False,
        "created_at_unix_ns": time.time_ns(),
    }
    _atomic_json(output / "phase_state.json", phase_state)
    label_open_started_at = time.time_ns()
    phase_state.update(
        {
            "status": "label-consumption-started",
            "label_array_consumed": True,
            "label_open_started_at_unix_ns": label_open_started_at,
        }
    )
    _atomic_json(output / "phase_state.json", phase_state)
    try:
        # metadata-only 检查不打开 bundle bytes；下一步每个 label bundle 只开一次。
        label_verification = validate_g2c_input_view(
            config_path=config_path,
            input_root=model_val_label_input_root,
            expected_role="model-val-privileged",
            verify_bundle_bytes=False,
        )
        labels = _load_model_val_labels(Path(model_val_label_input_root))
        result = _score_select_g2c_model_val_consumed(
            config=config,
            freeze_verification=freeze_verification,
            freeze_root=freeze_root,
            freeze_marker=freeze_marker,
            label_verification=label_verification,
            labels=labels,
            output=output,
            source_identity=source_identity,
            label_open_started_at=label_open_started_at,
        )
        phase_state.update(
            {
                "status": "selection-artifacts-written-pending-verification",
                "selection_receipt_internal_sha256": result["receipt"][
                    "receipt_sha256"
                ],
                "precompletion_verification_sha256": None,
            }
        )
        _atomic_json(output / "phase_state.json", phase_state)
        precompletion_verification = _verify_g2c_model_val_selection(
            config_path=config_path,
            prediction_freeze_root=prediction_freeze_root,
            output_root=output,
            expected_phase_status=(
                "selection-artifacts-written-pending-verification"
            ),
        )
        phase_state["status"] = "complete-model-val-score-select"
        phase_state["precompletion_verification_sha256"] = (
            precompletion_verification["verification_sha256"]
        )
        _atomic_json(output / "phase_state.json", phase_state)
        result["verification"] = verify_g2c_model_val_selection(
            config_path=config_path,
            prediction_freeze_root=prediction_freeze_root,
            output_root=output,
        )
    except Exception as error:
        failure = {
            "version": E018_P1_G2C_SELECTION_RESULT_VERSION,
            "status": "consumed-model-val-label-failed-no-rerun/v1",
            "config_sha256": config["config_sha256"],
            "prediction_freeze_internal_sha256": freeze_verification[
                "freeze_internal_sha256"
            ],
            "source_identity_sha256": source_identity["identity_sha256"],
            "label_array_consumed": True,
            "rerun_under_same_identity_allowed": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "failed_at_unix_ns": time.time_ns(),
        }
        failure["failure_sha256"] = canonical_sha256(failure)
        _atomic_json(output / "consumed_failure.json", failure)
        phase_state["status"] = failure["status"]
        phase_state["failure_sha256"] = failure["failure_sha256"]
        _atomic_json(output / "phase_state.json", phase_state)
        raise
    return result


def _verify_scoring_rows_against_frozen_predictions(
    *,
    freeze_root: Path,
    prediction_inventory: Sequence[Mapping[str, Any]],
    scoring_rows: Sequence[Mapping[str, Any]],
    freeze_internal_sha256: str,
    expected_used_for_selection: bool,
) -> None:
    frozen_rows: list[dict[str, Any]] = []
    for item in prediction_inventory:
        ledger_path = freeze_root / str(item["relative_path"])
        if file_sha256(ledger_path) != item.get("raw_sha256"):
            raise RuntimeError("G2C frozen prediction 在 verify→Phase B verify 间漂移")
        ledger_rows = _read_jsonl(ledger_path, "G2C frozen prediction ledger")
        if len(ledger_rows) != item.get("row_count"):
            raise RuntimeError("G2C frozen prediction inventory row_count 漂移")
        frozen_rows.extend(ledger_rows)
    if len(frozen_rows) != len(scoring_rows):
        raise RuntimeError("G2C scoring/frozen prediction row count 漂移")
    for scoring, prediction in zip(scoring_rows, frozen_rows, strict=True):
        _require_exact_keys(scoring, _SCORING_ROW_KEYS, "G2C scoring row")
        if (
            scoring.get("version") != E018_P1_G2C_SELECTION_RESULT_VERSION
            or scoring.get("phase")
            != "privileged-score-after-complete-prediction-freeze/v1"
            or scoring.get("prediction_freeze_sha256") != freeze_internal_sha256
            or scoring.get("candidate_id") != prediction.get("candidate_id")
            or scoring.get("epoch") != prediction.get("epoch")
            or scoring.get("checkpoint_sha256")
            != prediction.get("checkpoint_sha256")
            or scoring.get("seed") != prediction.get("seed")
            or scoring.get("sample_index") != prediction.get("sample_index")
            or scoring.get("viewpoint_id") != prediction.get("viewpoint_id")
            or scoring.get("predicted_observable")
            is not prediction.get("predicted_observable")
            or scoring.get("geometry_valid") is not prediction.get("geometry_valid")
            or scoring.get("predicted_object_position_base_m")
            != prediction.get("predicted_object_position_base_m")
            or scoring.get("raw_covariance_base_m2")
            != prediction.get("raw_covariance_base_m2")
            or scoring.get("write_score") != prediction.get("write_score")
            or scoring.get("used_for_formal_selection")
            is not expected_used_for_selection
            or scoring.get("test_data_read") is not False
            or not isinstance(scoring.get("gt_observable"), bool)
        ):
            raise RuntimeError("G2C scoring row 未逐行绑定 frozen prediction")
        try:
            gt_position = np.asarray(
                scoring["gt_object_position_base_m"], dtype=np.float64
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError("G2C scoring GT position 类型漂移") from error
        if gt_position.shape != (3,) or not np.isfinite(gt_position).all():
            raise RuntimeError("G2C scoring GT position shape/finite 漂移")
        predicted_value = prediction["predicted_object_position_base_m"]
        expected_error: float | None
        if predicted_value is None:
            expected_error = None
        else:
            predicted = np.asarray(predicted_value, dtype=np.float64)
            expected_error = float(np.linalg.norm(predicted - gt_position))
        actual_error = scoring.get("world_xyz_error_m")
        if expected_error is None:
            if actual_error is not None:
                raise RuntimeError("G2C scoring invalid geometry error 必须为 null")
        elif not _float_close(actual_error, expected_error):
            raise RuntimeError("G2C scoring world error 重算漂移")


def _verify_g2c_model_val_selection(
    *,
    config_path: str | Path,
    prediction_freeze_root: str | Path,
    output_root: str | Path,
    expected_phase_status: str,
) -> dict[str, Any]:
    """验证已消费的 Phase B artifact；不会再次接收或读取 label input。"""

    config = load_g2c_formal_training_config(config_path)
    freeze = verify_g2c_prediction_freeze(
        config_path=config_path, output_root=prediction_freeze_root
    )
    root = Path(output_root)
    receipt_path = root / "selection_receipt.json"
    receipt = _read_json(receipt_path, "G2C selection receipt")
    _require_exact_keys(
        receipt,
        {
            "version",
            "status",
            "classification",
            "config_sha256",
            "data_identity_sha256",
            "source_git_commit",
            "source_identity_sha256",
            "prediction_freeze_raw_sha256",
            "prediction_freeze_internal_sha256",
            "selected",
            "selected_checkpoint_identity",
            "selection_checkpoint_count",
            "candidate_prediction_row_count",
            "candidate_scoring_row_count",
            "candidate_loss_output_shard_count",
            "validation_loss_count",
            "diagnostic_control_prediction_row_count",
            "diagnostic_control_scoring_row_count",
            "diagnostic_control_validation_loss_count",
            "total_prediction_row_count",
            "total_scoring_row_count",
            "model_val_privileged_label_bundle_open_count",
            "test_array_read_count",
            "memory_read_count",
            "memory_write_count",
            "actuation_count",
            "artifact_sha256",
            "receipt_sha256",
        },
        "G2C selection receipt",
    )
    internal = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    allowed_status = {
        "complete-model-val-pass",
        "complete-model-val-protocol-valid-negative",
    }
    _assert_selection_count_contract(receipt)
    if (
        internal != canonical_sha256(unsigned)
        or receipt.get("version") != E018_P1_G2C_SELECTION_RESULT_VERSION
        or receipt.get("status") not in allowed_status
        or receipt.get("classification")
        != "development-only-model-val-no-test-no-actuation"
        or receipt.get("config_sha256") != config["config_sha256"]
        or receipt.get("data_identity_sha256")
        != config["data_parent"]["data_identity_sha256"]
        or receipt.get("prediction_freeze_raw_sha256") != freeze["freeze_raw_sha256"]
        or receipt.get("prediction_freeze_internal_sha256")
        != freeze["freeze_internal_sha256"]
        or receipt.get("test_array_read_count") != 0
        or receipt.get("memory_read_count") != 0
        or receipt.get("memory_write_count") != 0
        or receipt.get("actuation_count") != 0
    ):
        raise RuntimeError("G2C selection receipt status/count/permission 漂移")
    artifacts = receipt.get("artifact_sha256")
    if not isinstance(artifacts, dict) or set(artifacts) != set(
        _SELECTION_ARTIFACT_NAMES
    ):
        raise RuntimeError("G2C selection artifact inventory 不等于冻结文件白名单")
    for name, sha in artifacts.items():
        _require_sha256(sha, f"G2C selection artifact {name}")
        path = root / name
        if path.is_symlink() or file_sha256(path) != sha:
            raise RuntimeError(f"G2C selection artifact SHA 漂移: {name}")
    if _read_json(root / "config_snapshot.json", "G2C selection config snapshot") != config:
        raise RuntimeError("G2C selection config snapshot 漂移")
    source = _read_json(root / "source_identity.json", "G2C Phase B source")
    freeze_marker = _read_json(
        Path(prediction_freeze_root) / "prediction_freeze.json", "G2C freeze marker"
    )
    if (
        source.get("identity_sha256")
        != canonical_sha256(
            {
                "git_commit": source.get("git_commit"),
                "source_tree_sha256": source.get("source_tree_sha256"),
            }
        )
        or receipt.get("source_git_commit") != source.get("git_commit")
        or receipt.get("source_identity_sha256") != source.get("identity_sha256")
        or source.get("git_commit") != freeze_marker.get("source_git_commit")
        or source.get("identity_sha256") != freeze_marker.get("source_identity_sha256")
    ):
        raise RuntimeError("G2C Phase B/A source identity 漂移")
    phase_state = _read_json(root / "phase_state.json", "G2C selection phase state")
    _require_exact_keys(
        phase_state,
        {
            "version",
            "status",
            "config_sha256",
            "prediction_freeze_internal_sha256",
            "source_identity_sha256",
            "label_array_consumed",
            "rerun_under_same_identity_allowed",
            "created_at_unix_ns",
            "label_open_started_at_unix_ns",
            "selection_receipt_internal_sha256",
            "precompletion_verification_sha256",
        },
        "G2C selection phase state",
    )
    if (
        phase_state.get("version") != E018_P1_G2C_SELECTION_RESULT_VERSION
        or phase_state.get("status") != expected_phase_status
        or phase_state.get("config_sha256") != config["config_sha256"]
        or phase_state.get("prediction_freeze_internal_sha256")
        != freeze["freeze_internal_sha256"]
        or phase_state.get("source_identity_sha256") != source["identity_sha256"]
        or phase_state.get("label_array_consumed") is not True
        or phase_state.get("rerun_under_same_identity_allowed") is not False
        or phase_state.get("selection_receipt_internal_sha256") != internal
        or not isinstance(phase_state.get("created_at_unix_ns"), int)
        or not isinstance(phase_state.get("label_open_started_at_unix_ns"), int)
        or phase_state["label_open_started_at_unix_ns"]
        < phase_state["created_at_unix_ns"]
    ):
        raise RuntimeError("G2C Phase B consumed/completion state 漂移")
    precompletion_sha = phase_state.get("precompletion_verification_sha256")
    if expected_phase_status == "selection-artifacts-written-pending-verification":
        if precompletion_sha is not None:
            raise RuntimeError("G2C provisional phase state 不得预写 verifier identity")
    elif expected_phase_status == "complete-model-val-score-select":
        _require_sha256(
            precompletion_sha, "G2C Phase B precompletion verification SHA"
        )
    else:
        raise ValueError("G2C verifier expected phase status 未冻结")
    validation_rows = _read_json_array(
        root / "validation_losses.json", "G2C validation losses"
    )
    if (
        [(row.get("candidate_id"), row.get("epoch")) for row in validation_rows]
        != _expected_checkpoint_pairs()
        or any(row.get("sample_count") != 1100 for row in validation_rows)
        or any(row.get("batch_count") != 35 for row in validation_rows)
        or any(
            row.get("aggregation") != "sum(batch_loss*actual_batch_size)/1100"
            for row in validation_rows
        )
        or any(not math.isfinite(float(row.get("validation_loss"))) for row in validation_rows)
    ):
        raise RuntimeError("G2C validation loss rows 漂移")
    loss_batch_rows = _read_json_array(
        root / "validation_loss_batches.json", "G2C validation loss batches"
    )
    if len(loss_batch_rows) != 288:
        raise RuntimeError("G2C validation loss evidence 必须是 280 batch + 8 aggregate")
    validation_losses: dict[tuple[str, int], float] = {}
    cursor = 0
    for pair, validation_row in zip(
        _expected_checkpoint_pairs(), validation_rows, strict=True
    ):
        batches = loss_batch_rows[cursor : cursor + 35]
        aggregate = loss_batch_rows[cursor + 35]
        cursor += 36
        expected_sizes = [32] * 34 + [12]
        if (
            [(row.get("candidate_id"), row.get("epoch")) for row in batches]
            != [pair] * 35
            or [row.get("batch_index") for row in batches] != list(range(35))
            or [row.get("batch_size") for row in batches] != expected_sizes
            or any(row.get("frozen_decoded_uv_override_used") is not True for row in batches)
            or any(row.get("dense_subpixel_offsets_loaded") is not False for row in batches)
            or aggregate.get("candidate_id") != pair[0]
            or aggregate.get("epoch") != pair[1]
            or aggregate.get("aggregate_sample_count") != 1100
            or aggregate.get("aggregation")
            != "sum(batch_loss*actual_batch_size)/1100"
            or aggregate.get("frozen_decoded_uv_override_used") is not True
            or aggregate.get("dense_subpixel_offsets_loaded") is not False
        ):
            raise RuntimeError("G2C validation batch boundary/override evidence 漂移")
        recomputed: dict[str, float] = {}
        for name in _LOSS_COMPONENT_NAMES:
            value = sum(
                float(row["loss"][name]) * int(row["batch_size"]) for row in batches
            ) / 1100.0
            if not math.isclose(
                value,
                float(aggregate["aggregate_loss"][name]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RuntimeError("G2C validation aggregate loss 加权重算漂移")
            recomputed[name] = value
        if not math.isclose(
            recomputed["loss"],
            float(validation_row["validation_loss"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError("G2C validation loss summary 漂移")
        validation_losses[pair] = recomputed["loss"]
    scoring_rows = _read_jsonl(
        root / "model_val_scoring_ledger.jsonl", "G2C model-val scoring ledger"
    )
    if len(scoring_rows) != 8800:
        raise RuntimeError("G2C model-val scoring row count 漂移")
    control_scoring_rows = _read_jsonl(
        root / "diagnostic_control_scoring_ledger.jsonl",
        "G2C diagnostic CONTROL scoring ledger",
    )
    if len(control_scoring_rows) != 1100:
        raise RuntimeError("G2C diagnostic CONTROL scoring row count 漂移")
    freeze_root = Path(prediction_freeze_root)
    prediction_inventory = _read_json_array(
        freeze_root / "prediction_inventory.json", "G2C prediction inventory"
    )
    control_prediction_inventory = _read_json_array(
        freeze_root / "diagnostic_control_prediction_inventory.json",
        "G2C diagnostic CONTROL prediction inventory",
    )
    _verify_scoring_rows_against_frozen_predictions(
        freeze_root=freeze_root,
        prediction_inventory=prediction_inventory,
        scoring_rows=scoring_rows,
        freeze_internal_sha256=freeze["freeze_internal_sha256"],
        expected_used_for_selection=True,
    )
    _verify_scoring_rows_against_frozen_predictions(
        freeze_root=freeze_root,
        prediction_inventory=control_prediction_inventory,
        scoring_rows=control_scoring_rows,
        freeze_internal_sha256=freeze["freeze_internal_sha256"],
        expected_used_for_selection=False,
    )
    control_item = control_prediction_inventory[0]
    control_view_summaries = [
        summarize_g2c_model_val_view(
            [
                row
                for row in control_scoring_rows
                if row["viewpoint_id"] == viewpoint_id
            ],
            viewpoint_id=viewpoint_id,
        )
        for viewpoint_id in G2C_VIEW_ORDER
    ]
    expected_control_summary = {
        "version": E018_P1_G2C_SELECTION_RESULT_VERSION,
        "status": "complete-diagnostic-control-score-no-selection",
        "control_id": G2C_DIAGNOSTIC_CONTROL_ID,
        "source_role": "exact-e016-selected-epoch12-role-substitution/v1",
        "checkpoint_sha256": control_item["checkpoint_sha256"],
        "checkpoint_parameter_sha256": control_item[
            "checkpoint_parameter_sha256"
        ],
        "checkpoint_provenance_sha256": control_item[
            "checkpoint_provenance_sha256"
        ],
        "checkpoint_model_config_sha256": control_item[
            "checkpoint_model_config_sha256"
        ],
        "scoring_row_count": 1100,
        "view_summaries": control_view_summaries,
        "eligible_non_home_view_count": sum(
            bool(item["eligible"])
            for item in control_view_summaries
            if item["viewpoint_id"] != G2C_VIEW_ORDER[0]
        ),
        "validation_loss_computed": False,
        "eligible_for_selection": False,
        "used_for_formal_selection": False,
    }
    if _read_json(
        root / "diagnostic_control_summary.json",
        "G2C diagnostic CONTROL summary",
    ) != expected_control_summary:
        raise RuntimeError("G2C diagnostic CONTROL summary 重算漂移")
    selection = _read_json(root / "selection.json", "G2C model-val selection")
    recomputed_selection = select_g2c_checkpoint(
        scoring_rows, validation_losses=validation_losses
    )
    checkpoint_inventory = _read_json_array(
        Path(prediction_freeze_root) / "checkpoint_inventory.json",
        "G2C frozen checkpoint inventory",
    )
    selected = recomputed_selection.get("selected")
    if selected is None:
        recomputed_selection["selected_checkpoint_identity"] = None
    else:
        matches = [
            item
            for item in checkpoint_inventory
            if item["candidate_id"] == selected["candidate_id"]
            and item["epoch"] == selected["epoch"]
        ]
        if len(matches) != 1:
            raise RuntimeError("G2C recomputed selected checkpoint 不唯一")
        recomputed_selection["selected_checkpoint_identity"] = matches[0]
    if selection != recomputed_selection:
        raise RuntimeError("G2C frozen selection 重算漂移")
    summary = _read_json(root / "selection_summary.json", "G2C selection summary")
    if (
        summary.get("status") != receipt["status"]
        or summary.get("selected") != receipt.get("selected")
        or summary.get("selected_checkpoint_identity")
        != receipt.get("selected_checkpoint_identity")
        or summary.get("prediction_freeze_fully_verified_before_label_open") is not True
        or summary.get("label_open_started_after_freeze_marker") is not True
        or summary.get("frozen_decoded_uv_override_used") is not True
        or summary.get("dense_subpixel_offsets_loaded") is not False
        or summary.get("selection_checkpoint_count") != 8
        or summary.get("candidate_prediction_row_count") != 8800
        or summary.get("candidate_scoring_row_count") != 8800
        or summary.get("candidate_loss_output_shard_count") != 280
        or summary.get("validation_loss_count") != 8
        or summary.get("diagnostic_control_prediction_row_count") != 1100
        or summary.get("diagnostic_control_scoring_row_count") != 1100
        or summary.get("diagnostic_control_validation_loss_count") != 0
        or summary.get("total_prediction_row_count") != 9900
        or summary.get("total_scoring_row_count") != 9900
        or summary.get("diagnostic_control_summary")
        != expected_control_summary
    ):
        raise RuntimeError("G2C selection summary/receipt 漂移")
    expected_files = set(_SELECTION_ARTIFACT_NAMES) | {
        "selection_receipt.json",
        "phase_state.json",
    }
    total_bytes = _verify_exact_regular_file_tree(
        root,
        expected_files=expected_files,
        name="G2C Phase B selection",
    )
    if total_bytes > config["protocol"]["budgets"]["artifact_bytes_max"]:
        raise RuntimeError("G2C selection artifact 超过 20 GiB")
    result = {
        "version": E018_P1_G2C_SELECTION_RESULT_VERSION,
        "status": receipt["status"],
        "verified": True,
        "config_sha256": config["config_sha256"],
        "data_identity_sha256": receipt["data_identity_sha256"],
        "source_git_commit": receipt["source_git_commit"],
        "source_identity_sha256": receipt["source_identity_sha256"],
        "receipt_raw_sha256": file_sha256(receipt_path),
        "receipt_internal_sha256": internal,
        "prediction_freeze_internal_sha256": freeze["freeze_internal_sha256"],
        "selected": receipt.get("selected"),
        "selected_checkpoint_identity": receipt.get("selected_checkpoint_identity"),
        "label_bundle_reopen_count_for_verification": 0,
        "artifact_bytes": total_bytes,
    }
    result["verification_sha256"] = canonical_sha256(result)
    return result


def verify_g2c_model_val_selection(
    *,
    config_path: str | Path,
    prediction_freeze_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """验证 complete Phase B；接口刻意不接 label view，因此不会重开 label。"""

    return _verify_g2c_model_val_selection(
        config_path=config_path,
        prediction_freeze_root=prediction_freeze_root,
        output_root=output_root,
        expected_phase_status="complete-model-val-score-select",
    )
