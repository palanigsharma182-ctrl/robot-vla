"""E018-P1 G2C front-provider adaptation、选择、校准与动态资格协议。

正式 DATA 与 TRAIN identity 有意分离。本模块当前可执行的唯一训练路径是四 seed
engineering smoke；它不会保存 checkpoint，也不会运行正式 model selection、calibration
或 500-route qualification。
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla.precision.e018_p1_g2a import (
    FRONT_ALTERNATE_IDS,
    _mahalanobis_squared_psd,
    _measurement_covariance,
    canonical_sha256,
    file_sha256,
)
from robot_vla.precision.e018_p1_g2c_data import (
    _LABEL_ARRAYS,
    E018_P1_G2C_DATA_RESULT_VERSION,
    G2C_SMOKE_SPLIT,
    G2C_VIEW_ORDER,
    G2CDeployableDataset,
    G2CFrontTrainingDataset,
    _atomic_json,
    _atomic_jsonl,
    _load_npz,
    _read_jsonl,
    _resolve_artifact_file,
    load_e018_p1_g2c_data_config,
    run_e018_p1_g2c_data,
    verify_g2c_data_receipt,
)
from robot_vla.precision.object_observability import ObjectWriteEvidence
from robot_vla.precision.observability import mask_probability_at_normalized_uv
from robot_vla.precision.outliers import geometry_conditioning

E018_P1_G2C_TRAIN_PROTOCOL_VERSION = (
    "e018-p1-g2c-front-provider-training-development/v1"
)
E018_P1_G2C_SMOKE_RESULT_VERSION = (
    "e018-p1-g2c-front-provider-engineering-smoke-result/v1"
)
G2C_CANDIDATE_IDS = ("W-KV0", "S")
G2C_CANDIDATE_INITIALIZATION_SEEDS = {"W-KV0": 18021, "S": 18022}
G2C_SHARED_SAMPLER_SEED = 18020
G2C_CANDIDATE_EPOCHS = (5, 10, 15, 20)


def g2c_training_protocol() -> dict[str, Any]:
    """返回 D036 + 后续 decision 补强冻结的独立 TRAIN/v1 协议。"""

    return {
        "version": E018_P1_G2C_TRAIN_PROTOCOL_VERSION,
        "data_binding": {
            "requires_canonical_data_receipt": True,
            "data_receipt_must_precede_train_config": True,
            "train_config_must_bind_data_receipt_raw_and_internal_sha256": True,
            "smoke_data_cannot_be_promoted": True,
        },
        "candidates": {
            "control": {
                "id": "CONTROL",
                "source": "e016-epoch12-role-substitution",
                "diagnostic_only": True,
                "eligible_for_selection": False,
            },
            "diagnostic_W": {
                "id": "W",
                "architecture": "PrecisionThreeHeadUNet",
                "initialization": "e016-selected-epoch12-warm-start",
                "diagnostic_only": True,
                "eligible_for_selection": False,
                "superseded_by": "W-KV0",
            },
            "W-KV0": {
                "architecture": "PrecisionThreeHeadUNet",
                "initialization": (
                    "e016-selected-epoch12-warm-start-then-zero-"
                    "keypoint-logvariance-rows/v1"
                ),
                "initialization_seed": 18021,
            },
            "S": {
                "architecture": "PrecisionThreeHeadUNet",
                "initialization": "random",
                "initialization_seed": 18022,
            },
        },
        "training": {
            "optimizer": "AdamW",
            "epochs": 20,
            "batch_size": 32,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 1.0,
            "scheduler": "cosine-annealing-eta-min-5-percent/v1",
            "eta_min": 1.5e-5,
            "use_bf16": True,
            "num_workers": 0,
            "spatial_augmentation": False,
            "motion_head_policy": "frozen-zero-shadow-only",
            "shared_sampler_seed": G2C_SHARED_SAMPLER_SEED,
            "candidate_run_seed_is_not_sampler_seed": True,
            "require_identical_per_epoch_shuffle": True,
            "candidate_epochs": list(G2C_CANDIDATE_EPOCHS),
        },
        "loss": {
            "heatmap_weight": 1.0,
            "mask_weight": 0.5,
            "mask_dice_weight": 1.0,
            "coordinate_weight": 2.0,
            "motion_weight": 1.0,
            "uncertainty_weight": 0.1,
            "visibility_weight": 1.0,
            "projection_weight": 1.0,
            "keypoint_temperature": 1.0,
            "heatmap_sigma_px": 1.5,
        },
        "model_validation": {
            "minimum_observable_positive_support_per_viewpoint": 30,
            "support_below_minimum_policy": "viewpoint-ineligible-not-protocol-invalid/v1",
            "minimum_visibility_precision": 0.95,
            "minimum_visibility_recall": 0.90,
            "maximum_observable_world_xyz_p90_m": 0.005,
            "maximum_observable_world_xyz_max_m": 0.020,
            "selection_order": [
                "eligible_non_home_view_count_descending",
                "best_eligible_view_p90_ascending",
                "corresponding_max_ascending",
                "validation_loss_ascending",
                "epoch_ascending",
                "candidate_W_KV0_before_S",
            ],
            "zero_eligible_policy": "selected-null-protocol-valid-negative/v1",
        },
        "calibration": {
            "method": "split-conformal-xy-mahalanobis-scalar/alpha-0.05-chi2-5.991/v1",
            "alpha": 0.05,
            "target_coverage": 0.95,
            "chi_square_threshold": 5.991,
            "order_statistic": "ceil((N+1)*0.95)",
            "minimum_support": 30,
            "scale": "max(1,q/5.991)",
            "maximum_calibrated_position_std_m": 0.020,
            "minimum_accepted_safe_coverage": 0.10,
            "maximum_unsafe_accepted_count": 0,
            "threshold_tie_break": "higher-more-conservative",
        },
        "qualification": g2c_dynamic_qualification_plan(),
        "permissions": {
            "test_array_reads": 0,
            "memory_reads": 0,
            "memory_writes": 0,
            "runtime_camera_actuation": 0,
            "physical_camera_actuation": 0,
            "nonzero_arm_motion_commands": 0,
            "gripper_close_commands": 0,
            "manipulation_progression": 0,
        },
    }


def build_g2c_train_config_payload(
    data_receipt_path: str | Path,
) -> dict[str, Any]:
    """从已通过的 canonical DATA receipt 机械产生尚未执行的 TRAIN config。"""

    receipt_path = Path(data_receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("version") != E018_P1_G2C_DATA_RESULT_VERSION
        or receipt.get("status") != "complete-data-pass"
        or receipt.get("gate_passed") is not True
        or receipt.get("canonical_data_receipt") is not True
        or receipt.get("checkpoint_write_count") != 0
    ):
        raise RuntimeError("G2C TRAIN config 只能绑定通过的 canonical DATA receipt")
    unsigned = dict(receipt)
    internal = unsigned.pop("receipt_sha256", None)
    if internal != canonical_sha256(unsigned):
        raise RuntimeError("G2C DATA receipt internal identity 漂移")
    payload = {
        "version": E018_P1_G2C_TRAIN_PROTOCOL_VERSION,
        "status": "frozen-pre-training-no-execution/v1",
        "data_parent": {
            "receipt_raw_sha256": file_sha256(receipt_path),
            "receipt_internal_sha256": internal,
            "data_identity_sha256": receipt["data_identity_sha256"],
            "data_config_sha256": receipt["config_sha256"],
        },
        "protocol": g2c_training_protocol(),
    }
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def summarize_g2c_model_val_view(
    rows: Sequence[Mapping[str, Any]],
    *,
    viewpoint_id: str,
) -> dict[str, Any]:
    if viewpoint_id not in G2C_VIEW_ORDER:
        raise ValueError("G2C model-val viewpoint_id 未冻结")
    if not rows or any(row.get("viewpoint_id") != viewpoint_id for row in rows):
        raise ValueError("G2C model-val rows 必须是单个非空 viewpoint")
    support = sum(bool(row["gt_observable"]) for row in rows)
    predicted_positive = sum(bool(row["predicted_observable"]) for row in rows)
    true_positive = sum(
        bool(row["gt_observable"] and row["predicted_observable"])
        for row in rows
    )
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / support if support else 0.0
    errors = [
        float(row["world_xyz_error_m"])
        for row in rows
        if row["gt_observable"]
        and row["geometry_valid"]
        and row["world_xyz_error_m"] is not None
    ]
    if any(not math.isfinite(value) or value < 0.0 for value in errors):
        raise ValueError("G2C model-val world error 必须有限非负")
    p90 = float(np.quantile(errors, 0.90)) if errors else None
    maximum = max(errors) if errors else None
    rules = g2c_training_protocol()["model_validation"]
    reasons: list[str] = []
    if support < rules["minimum_observable_positive_support_per_viewpoint"]:
        reasons.append("observable_positive_support_below_30")
    if precision < rules["minimum_visibility_precision"]:
        reasons.append("visibility_precision_below_0.95")
    if recall < rules["minimum_visibility_recall"]:
        reasons.append("visibility_recall_below_0.90")
    if len(errors) != support:
        reasons.append("observable_geometry_not_fully_evaluable")
    if p90 is None or p90 > rules["maximum_observable_world_xyz_p90_m"]:
        reasons.append("observable_world_xyz_p90_above_0.005")
    if maximum is None or maximum > rules["maximum_observable_world_xyz_max_m"]:
        reasons.append("observable_world_xyz_max_above_0.020")
    return {
        "viewpoint_id": viewpoint_id,
        "row_count": len(rows),
        "observable_positive_support": support,
        "predicted_positive_count": predicted_positive,
        "true_positive_count": true_positive,
        "visibility_precision": precision,
        "visibility_recall": recall,
        "observable_geometry_evaluable_count": len(errors),
        "observable_world_xyz_p90_m": p90,
        "observable_world_xyz_max_m": maximum,
        "eligible": not reasons,
        "ineligibility_reasons": reasons,
        "protocol_invalid": False,
    }


def select_g2c_checkpoint(
    rows: Sequence[Mapping[str, Any]],
    *,
    validation_losses: Mapping[tuple[str, int], float],
) -> dict[str, Any]:
    """按冻结 model-val 排序选择 W-KV0/S；support<30 只淘汰 viewpoint。"""

    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        candidate = str(row.get("candidate_id"))
        epoch = int(row.get("epoch"))
        viewpoint = str(row.get("viewpoint_id"))
        if candidate not in G2C_CANDIDATE_IDS or epoch not in G2C_CANDIDATE_EPOCHS:
            raise ValueError("G2C model-val candidate/epoch 不在冻结池")
        grouped[(candidate, epoch, viewpoint)].append(row)
    expected_candidates = {
        (candidate, epoch)
        for candidate in G2C_CANDIDATE_IDS
        for epoch in G2C_CANDIDATE_EPOCHS
    }
    if set(validation_losses) != expected_candidates:
        raise ValueError("G2C validation loss identity 必须完整覆盖 W-KV0/S×4 epochs")
    candidates = []
    for candidate, epoch in sorted(expected_candidates):
        loss = float(validation_losses[(candidate, epoch)])
        if not math.isfinite(loss):
            raise ValueError("G2C validation loss 必须有限")
        summaries = [
            summarize_g2c_model_val_view(
                grouped.get((candidate, epoch, viewpoint), []),
                viewpoint_id=viewpoint,
            )
            for viewpoint in G2C_VIEW_ORDER
            if grouped.get((candidate, epoch, viewpoint))
        ]
        by_view = {value["viewpoint_id"]: value for value in summaries}
        if set(by_view) != set(G2C_VIEW_ORDER):
            raise ValueError("G2C model-val ledger 必须完整覆盖 11 viewpoints")
        eligible = [by_view[name] for name in FRONT_ALTERNATE_IDS if by_view[name]["eligible"]]
        best = min(
            eligible,
            key=lambda value: (
                value["observable_world_xyz_p90_m"],
                value["observable_world_xyz_max_m"],
                G2C_VIEW_ORDER.index(value["viewpoint_id"]),
            ),
            default=None,
        )
        candidates.append(
            {
                "candidate_id": candidate,
                "epoch": epoch,
                "validation_loss": loss,
                "eligible_non_home_view_count": len(eligible),
                "best_eligible_view": best,
                "view_summaries": summaries,
                "eligible": bool(eligible),
            }
        )
    eligible_candidates = [item for item in candidates if item["eligible"]]
    if not eligible_candidates:
        return {
            "status": "complete-model-val-protocol-valid-negative",
            "protocol_valid": True,
            "selected": None,
            "candidates": candidates,
            "reason": "no_non_home_viewpoint_eligible",
        }
    selected = min(
        eligible_candidates,
        key=lambda item: (
            -item["eligible_non_home_view_count"],
            item["best_eligible_view"]["observable_world_xyz_p90_m"],
            item["best_eligible_view"]["observable_world_xyz_max_m"],
            item["validation_loss"],
            item["epoch"],
            G2C_CANDIDATE_IDS.index(item["candidate_id"]),
        ),
    )
    return {
        "status": "complete-model-val-pass",
        "protocol_valid": True,
        "selected": {
            "candidate_id": selected["candidate_id"],
            "epoch": selected["epoch"],
            "eligible_non_home_view_count": selected[
                "eligible_non_home_view_count"
            ],
            "best_eligible_viewpoint_id": selected["best_eligible_view"][
                "viewpoint_id"
            ],
        },
        "candidates": candidates,
    }


def calibrate_g2c_viewpoint(
    rows: Sequence[Mapping[str, Any]],
    *,
    viewpoint_id: str,
) -> dict[str, Any]:
    """按 D044 的双 cohort 口径拟合单视角 calibration。

    covariance cohort 只使用 GT 可观察且几何/误差/协方差有效的行；write
    threshold 则始终审计传入的全部行，并以独立的 ``oracle_safe_count``
    作为 coverage 分母。这样 predicted-observable gate 不能自行缩小安全
    分母，也不会漏掉“GT 不可观察但模型高分接受”的 unsafe 行。
    """

    if viewpoint_id not in G2C_VIEW_ORDER:
        raise ValueError("G2C calibration viewpoint 未冻结")
    if any(row.get("viewpoint_id") != viewpoint_id for row in rows):
        raise ValueError("G2C calibration rows 必须属于单 viewpoint")
    rules = g2c_training_protocol()["calibration"]
    covariance_cohort = []
    threshold_rows: list[tuple[float, bool, bool, bool]] = []
    oracle_safe_count = 0
    catastrophic_count = 0
    for row in rows:
        score = float(row["write_score"])
        structurally_eligible = row.get("structurally_eligible")
        oracle_safe = row.get("oracle_safe_measurement")
        catastrophic = row.get("catastrophic_measurement")
        gt_observable = row.get("gt_observable")
        geometry_valid = row.get("geometry_valid")
        for name, value in (
            ("structurally_eligible", structurally_eligible),
            ("oracle_safe_measurement", oracle_safe),
            ("catastrophic_measurement", catastrophic),
            ("gt_observable", gt_observable),
            ("geometry_valid", geometry_valid),
        ):
            if type(value) is not bool:
                raise TypeError(f"G2C calibration {name} 必须是 bool")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("G2C calibration write_score 必须是有限 [0,1]")
        oracle_safe_count += int(oracle_safe)
        catastrophic_count += int(catastrophic)
        threshold_rows.append(
            (score, structurally_eligible, oracle_safe, catastrophic)
        )

        if not (gt_observable and geometry_valid):
            continue
        error = np.asarray(row["world_xy_error_vector_m"], dtype=np.float64)
        covariance = np.asarray(row["raw_covariance_base_m2"], dtype=np.float64)
        if (
            error.shape != (2,)
            or covariance.shape != (3, 3)
            or not np.isfinite(error).all()
            or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12)
            or float(np.linalg.eigvalsh(covariance).min()) < -1e-12
        ):
            raise ValueError("G2C calibration covariance cohort 非有限/shape/PSD 漂移")
        mahalanobis = _mahalanobis_squared_psd(error, covariance[:2, :2])
        covariance_cohort.append((float(mahalanobis), covariance))

    support = len(covariance_cohort)
    calibration: dict[str, Any] | None = None
    maximum_std: float | None = None
    covariance_passed = False
    if support >= rules["minimum_support"]:
        k = min(
            math.ceil((support + 1) * rules["target_coverage"]),
            support,
        )
        quantile = sorted(item[0] for item in covariance_cohort)[k - 1]
        if math.isfinite(quantile):
            scale = max(1.0, quantile / rules["chi_square_threshold"])
            if math.isfinite(scale):
                maximum_std = max(
                    float(
                        np.sqrt(
                            np.linalg.eigvalsh(item[1][:2, :2] * scale).max()
                        )
                    )
                    for item in covariance_cohort
                )
                covariance_passed = math.isfinite(maximum_std) and (
                    maximum_std
                    <= rules["maximum_calibrated_position_std_m"]
                )
                calibration = {
                    "alpha": rules["alpha"],
                    "chi_square_threshold": rules["chi_square_threshold"],
                    "order_statistic_k": k,
                    "quantile_score": quantile,
                    "scale_factor": scale,
                    "maximum_calibrated_position_std_m": maximum_std,
                }

    candidates = sorted({item[0] for item in threshold_rows}, reverse=True)
    threshold_candidates = []
    for threshold in candidates:
        accepted = [item for item in threshold_rows if item[1] and item[0] >= threshold]
        unsafe = sum(not item[2] for item in accepted)
        accepted_safe = sum(item[2] for item in accepted)
        catastrophic_accepted = sum(item[3] for item in accepted)
        coverage = (
            accepted_safe / oracle_safe_count if oracle_safe_count > 0 else 0.0
        )
        if unsafe == 0 and coverage >= rules["minimum_accepted_safe_coverage"]:
            threshold_candidates.append(
                (
                    coverage,
                    len(accepted),
                    threshold,
                    accepted_safe,
                    catastrophic_accepted,
                )
            )
    chosen = max(
        threshold_candidates,
        key=lambda item: (item[0], item[1], item[2]),
        default=None,
    )
    threshold_passed = chosen is not None
    passed = covariance_passed and threshold_passed
    failure_reasons = []
    if support < rules["minimum_support"]:
        failure_reasons.append("covariance_support_below_30")
    elif calibration is None:
        failure_reasons.append("nonfinite_conformal_quantile_or_scale")
    elif not covariance_passed:
        failure_reasons.append("maximum_calibrated_position_std_above_0.020")
    if oracle_safe_count == 0:
        failure_reasons.append("oracle_safe_count_zero")
    if not threshold_passed:
        failure_reasons.append("no_zero_unsafe_threshold_with_coverage_at_least_0.10")
    return {
        "viewpoint_id": viewpoint_id,
        "status": "calibration-pass" if passed else "calibration-no-go",
        "row_count": len(rows),
        "support_count": support,
        "oracle_safe_count": oracle_safe_count,
        "catastrophic_measurement_count": catastrophic_count,
        "covariance_passed": covariance_passed,
        "threshold_passed": threshold_passed,
        "calibration": calibration,
        "write_threshold": None if chosen is None else chosen[2],
        "accepted_safe_coverage": 0.0 if chosen is None else chosen[0],
        "accepted_count": 0 if chosen is None else chosen[1],
        "accepted_and_oracle_safe_count": 0 if chosen is None else chosen[3],
        "unsafe_accepted_count": 0 if chosen is not None else None,
        "catastrophic_accepted_count": 0 if chosen is None else chosen[4],
        "failure_reasons": failure_reasons,
        "passed": passed,
    }


def g2c_dynamic_qualification_plan() -> dict[str, Any]:
    return {
        "parent_g0c_config_sha256": "c93bbfd48b6d9bc2fc75b5b87e4ded7161efebd7eda50cd81cc2ded47810e965",
        "parent_g0c_receipt_sha256": "bf8232b620cd5ff8de8c0007391252b8829c3ebbac320a7d5a60507beaca258e",
        "seed_count": 50,
        "alternate_count": 10,
        "route_count": 500,
        "per_route": {
            "initial_camera_pose_set": 1,
            "home_warmup_ticks": 5,
            "home_anchor_frames": 1,
            "outbound_ticks": 40,
            "alternate_settle_ticks": 4,
            "collect_ticks": 3,
            "return_ticks": 40,
            "home_verify_ticks": 4,
            "camera_pose_set_count": 97,
            "moving_interpolation_command_count": 80,
            "safe_hold_open_step_count": 96,
            "ledger_frame_count": 92,
        },
        "totals": {
            "camera_pose_set_count": 48_500,
            "moving_interpolation_command_count": 40_000,
            "safe_hold_open_step_count": 48_000,
            "ledger_frame_count": 46_000,
            "provider_scored_home_frame_count": 50,
            "provider_scored_alternate_frame_count": 500,
            "provider_scored_frame_count": 550,
        },
        "one_attempt_per_route": True,
        "retry_allowed": False,
        "seed_or_route_replacement_allowed": False,
        "runtime_camera_actuation_count": 0,
        "physical_camera_actuation_count": 0,
        "nonzero_arm_motion_command_count": 0,
        "gripper_close_command_count": 0,
        "manipulation_progression_count": 0,
        "object_contact_event_count": 0,
        "memory_read_count": 0,
        "memory_write_count": 0,
        "test_trajectory_array_read_count": 0,
        "test_label_array_read_count": 0,
    }


def validate_g2c_dynamic_qualification_counters(actual: Mapping[str, Any]) -> None:
    """供后续真实 route runner 在评分前验证 D036 精确计数。"""

    expected = g2c_dynamic_qualification_plan()
    if set(actual) != set(expected["totals"]):
        raise ValueError("G2C qualification total counter keys 漂移")
    if dict(actual) != expected["totals"]:
        raise RuntimeError("G2C qualification 精确计数未满足，整轮 protocol-invalid")


_PREDICTION_FORBIDDEN_KEYS = {
    "gt",
    "ground_truth",
    "label",
    "target",
    "oracle",
    "error",
    "is_grasped",
    "contact",
    "force",
    "safe_measurement",
}
_PREDICTION_FORBIDDEN_EXACT_KEYS = {
    "object_position_base_m",
    "goal_position_base_m",
    "object_mask",
    "goal_mask",
    "keypoint_observable",
    "keypoint_projection_valid",
}


def assert_g2c_prediction_ledger_deployable_only(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).lower()
                tokens = set(filter(None, re.split(r"[^a-z0-9]+", lowered)))
                if (
                    lowered in _PREDICTION_FORBIDDEN_EXACT_KEYS
                    or tokens.intersection(_PREDICTION_FORBIDDEN_KEYS)
                ):
                    raise ValueError(f"G2C prediction ledger 含 privileged 字段: {path}.{key}")
                walk(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    for index, row in enumerate(rows):
        walk(row, f"rows[{index}]")


def freeze_g2c_prediction_ledger(
    output_root: str | Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    config_sha256: str,
    data_identity_sha256: str,
) -> dict[str, Any]:
    assert_g2c_prediction_ledger_deployable_only(rows)
    root = Path(output_root)
    ledger = root / "model_val_prediction_ledger.jsonl"
    marker = root / "model_val_prediction_freeze.json"
    if ledger.exists() or marker.exists():
        raise FileExistsError("G2C prediction ledger/freeze 已存在")
    _atomic_jsonl(ledger, rows)
    value = {
        "version": E018_P1_G2C_SMOKE_RESULT_VERSION,
        "status": "frozen-before-privileged-label-open",
        "config_sha256": config_sha256,
        "data_identity_sha256": data_identity_sha256,
        "prediction_count": len(rows),
        "prediction_ledger_raw_sha256": file_sha256(ledger),
        "privileged_label_open_count_before_freeze": 0,
        "test_array_read_count": 0,
        "frozen_at_unix_ns": time.time_ns(),
    }
    value["freeze_sha256"] = canonical_sha256(value)
    _atomic_json(marker, value)
    return value


def load_frozen_g2c_prediction_ledger(
    output_root: str | Path,
    *,
    config_sha256: str,
    data_identity_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(output_root)
    marker_path = root / "model_val_prediction_freeze.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker_sha = marker.get("freeze_sha256")
    unsigned = dict(marker)
    unsigned.pop("freeze_sha256", None)
    ledger = root / "model_val_prediction_ledger.jsonl"
    if (
        marker_sha != canonical_sha256(unsigned)
        or marker.get("status") != "frozen-before-privileged-label-open"
        or marker.get("config_sha256") != config_sha256
        or marker.get("data_identity_sha256") != data_identity_sha256
        or marker.get("prediction_ledger_raw_sha256") != file_sha256(ledger)
        or marker.get("privileged_label_open_count_before_freeze") != 0
    ):
        raise RuntimeError("G2C prediction freeze identity 漂移")
    rows = _read_jsonl(ledger, "G2C model-val prediction ledger")
    assert_g2c_prediction_ledger_deployable_only(rows)
    if len(rows) != marker["prediction_count"]:
        raise RuntimeError("G2C frozen prediction count 漂移")
    return rows, marker


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _collate_training(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    model_inputs = [sample["model_inputs"] for sample in samples]
    supervision = [sample["supervision"] for sample in samples]
    image = np.stack([item["rgb_external"] for item in model_inputs])
    return {
        "image": torch.from_numpy(
            np.ascontiguousarray(image.transpose(0, 3, 1, 2), dtype=np.float32)
            / np.float32(255.0)
        ),
        "structured_state": torch.from_numpy(
            np.stack([item["structured_state"] for item in model_inputs])
        ),
        "geometric_motion": torch.from_numpy(
            np.stack([item["geometric_motion"] for item in model_inputs])
        ),
        "mask_targets": torch.from_numpy(
            np.stack([item["mask_targets"] for item in supervision])
        ),
        "normalized_uv_targets": torch.from_numpy(
            np.stack([item["normalized_uv_targets"] for item in supervision])
        ),
        "keypoint_valid": torch.from_numpy(
            np.stack([item["keypoint_valid"] for item in supervision])
        ),
        "keypoint_observable": torch.from_numpy(
            np.stack([item["keypoint_observable"] for item in supervision])
        ),
        "motion_residual_targets": torch.from_numpy(
            np.stack([item["motion_residual_targets"] for item in supervision])
        ),
        "motion_valid": torch.from_numpy(
            np.stack([item["motion_valid"] for item in supervision])
        ),
        "projection_valid": torch.from_numpy(
            np.stack([item["projection_valid"] for item in supervision])
        ),
        "sample_order": [
            {
                "seed": int(sample["audit"]["seed"]),
                "viewpoint_id": str(sample["audit"]["viewpoint_id"]),
            }
            for sample in samples
        ],
    }


def _build_supervision(batch: Mapping[str, Any]) -> Any:
    from robot_vla.precision.losses import PrecisionSupervision, build_gaussian_heatmaps

    return PrecisionSupervision(
        heatmap_targets=build_gaussian_heatmaps(
            batch["normalized_uv_targets"],
            batch["keypoint_valid"],
            tuple(int(value) for value in batch["image"].shape[-2:]),
            sigma_px=1.5,
        ),
        mask_targets=batch["mask_targets"],
        normalized_uv_targets=batch["normalized_uv_targets"],
        keypoint_valid=batch["keypoint_valid"],
        keypoint_observable=batch["keypoint_observable"],
        motion_residual_targets=batch["motion_residual_targets"],
        motion_valid=batch["motion_valid"],
        projection_valid=batch["projection_valid"],
    )


def _zero_warm_start_keypoint_log_variance_rows(model: Any) -> dict[str, Any]:
    """只清零 uncertainty 最终 Linear 的 keypoint log-variance 输出行。"""

    import torch

    from robot_vla.precision.checkpoint import precision_parameter_state_sha256

    final_layer = model.uncertainty_head[-1]
    if not isinstance(final_layer, torch.nn.Linear):
        raise TypeError("G2C W-KV0 要求 uncertainty_head 最后一层是 Linear")
    row_count = int(model.config.keypoint_count) * 2
    if row_count <= 0 or final_layer.out_features <= row_count:
        raise ValueError("G2C W-KV0 keypoint log-variance row layout 漂移")
    module_names = [
        name for name, module in model.named_modules() if module is final_layer
    ]
    if len(module_names) != 1:
        raise RuntimeError("G2C W-KV0 无法唯一定位 uncertainty final Linear")
    weight_name = f"{module_names[0]}.weight"
    bias_name = f"{module_names[0]}.bias"
    before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    if weight_name not in before or bias_name not in before:
        raise RuntimeError("G2C W-KV0 uncertainty final Linear state 缺失")

    def target_slice(state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "keypoint_logvariance_weight": state[weight_name][:row_count],
            "keypoint_logvariance_bias": state[bias_name][:row_count],
        }

    def non_target_state(state: Mapping[str, Any]) -> dict[str, Any]:
        result = {
            name: tensor
            for name, tensor in state.items()
            if name not in {weight_name, bias_name}
        }
        result[f"{weight_name}.non_keypoint_rows"] = state[weight_name][row_count:]
        result[f"{bias_name}.non_keypoint_rows"] = state[bias_name][row_count:]
        return result

    parameter_before = precision_parameter_state_sha256(before)
    target_before = precision_parameter_state_sha256(target_slice(before))
    non_target_before = precision_parameter_state_sha256(non_target_state(before))
    motion_before = precision_parameter_state_sha256(model.motion_head.state_dict())
    with torch.no_grad():
        final_layer.weight[:row_count].zero_()
        final_layer.bias[:row_count].zero_()
    after = model.state_dict()
    target_after = precision_parameter_state_sha256(target_slice(after))
    non_target_after = precision_parameter_state_sha256(non_target_state(after))
    motion_after = precision_parameter_state_sha256(model.motion_head.state_dict())
    rows_zero = bool(
        torch.count_nonzero(final_layer.weight[:row_count]).item() == 0
        and torch.count_nonzero(final_layer.bias[:row_count]).item() == 0
    )
    if not rows_zero or non_target_after != non_target_before:
        raise RuntimeError("G2C W-KV0 修改超出 keypoint log-variance rows")
    if motion_after != motion_before:
        raise RuntimeError("G2C W-KV0 reset 改变了 frozen Motion Head")
    return {
        "policy": "zero-final-uncertainty-linear-keypoint-logvariance-rows/v1",
        "row_indices": list(range(row_count)),
        "row_count": row_count,
        "weight_shape": list(final_layer.weight[:row_count].shape),
        "bias_shape": list(final_layer.bias[:row_count].shape),
        "parameter_sha256_before": parameter_before,
        "parameter_sha256_after": precision_parameter_state_sha256(after),
        "keypoint_logvariance_rows_sha256_before": target_before,
        "keypoint_logvariance_rows_sha256_after": target_after,
        "keypoint_logvariance_rows_all_zero_after": rows_zero,
        "non_target_parameter_sha256_before": non_target_before,
        "non_target_parameter_sha256_after": non_target_after,
        "non_target_parameters_unchanged": non_target_before == non_target_after,
        "motion_head_sha256_before": motion_before,
        "motion_head_sha256_after": motion_after,
        "motion_head_unchanged": motion_before == motion_after,
    }


def _load_candidate_model(
    *,
    candidate_id: str,
    e016_config_path: Path,
    training_output: Path,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import torch

    from robot_vla.precision.checkpoint import (
        PrecisionCheckpointRole,
        load_precision_checkpoint,
        precision_parameter_state_sha256,
    )
    from robot_vla.precision.e016_pretraining import new_e016_frozen_motion_model
    from robot_vla.precision.e016_training import load_e016_p1_config

    if candidate_id not in G2C_CANDIDATE_IDS:
        raise ValueError("G2C candidate 必须是 W-KV0/S")
    protocol = g2c_training_protocol()
    init_seed = G2C_CANDIDATE_INITIALIZATION_SEEDS[candidate_id]
    _seed_everything(init_seed)
    e016 = load_e016_p1_config(e016_config_path)
    if e016.sha256 != config["parents"]["e016_config_sha256"]:
        raise RuntimeError("G2C E016 config identity 漂移")
    if candidate_id == "W-KV0":
        loaded = load_precision_checkpoint(
            training_output / "precision-formal.pt",
            expected_checkpoint_sha256=config["parents"]["e016_checkpoint_sha256"],
            expected_provenance_sha256=config["parents"][
                "e016_checkpoint_provenance_sha256"
            ],
            expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
        )
        if (
            loaded.receipt.parameter_state_sha256
            != config["parents"]["e016_checkpoint_parameter_sha256"]
        ):
            raise RuntimeError("G2C W-KV0 warm-start parameter identity 漂移")
        if (
            loaded.receipt.model_config_sha256
            != config["parents"]["e016_checkpoint_model_config_sha256"]
        ):
            raise RuntimeError("G2C W-KV0 warm-start model config identity 漂移")
        model = loaded.model.to(torch.device("cuda"))
        keypoint_variance_reset = _zero_warm_start_keypoint_log_variance_rows(
            model
        )
        if (
            keypoint_variance_reset["parameter_sha256_before"]
            != loaded.receipt.parameter_state_sha256
        ):
            raise RuntimeError("G2C W-KV0 reset parent parameter identity 漂移")
        initialization = {
            "kind": "e016-selected-epoch12-warm-start-keypoint-variance-zero",
            "source_checkpoint_sha256": loaded.receipt.checkpoint_sha256,
            "source_parameter_sha256": loaded.receipt.parameter_state_sha256,
            "source_provenance_sha256": loaded.receipt.provenance_sha256,
            "model_config_sha256": loaded.receipt.model_config_sha256,
            "keypoint_logvariance_reset": keypoint_variance_reset,
        }
    else:
        model, _ = new_e016_frozen_motion_model(torch.device("cuda"))
        initialization = {
            "kind": "random",
            "source_checkpoint_sha256": None,
            "source_parameter_sha256": None,
            "source_provenance_sha256": None,
            "model_config_sha256": canonical_sha256(model.config),
            "keypoint_logvariance_reset": None,
        }
    model.motion_head.requires_grad_(False)
    motion_hash = precision_parameter_state_sha256(model.motion_head.state_dict())
    initialization.update(
        {
            "candidate_id": candidate_id,
            "initialization_seed": init_seed,
            "shared_sampler_seed": protocol["training"]["shared_sampler_seed"],
            "initial_parameter_sha256": precision_parameter_state_sha256(
                model.state_dict()
            ),
            "initial_motion_head_parameter_sha256": motion_hash,
        }
    )
    return model, initialization


def _train_candidate_smoke(
    *,
    model: Any,
    dataset: G2CFrontTrainingDataset,
    candidate_id: str,
    optimizer_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    from robot_vla.precision.checkpoint import precision_parameter_state_sha256
    from robot_vla.precision.losses import PrecisionLossConfig, precision_unet_loss

    if optimizer_steps != 2:
        raise ValueError("G2C engineering smoke 固定为每候选 2 optimizer steps")
    protocol = g2c_training_protocol()
    training = protocol["training"]
    loss_values = dict(protocol["loss"])
    loss_values.pop("heatmap_sigma_px")
    loss_config = PrecisionLossConfig(**loss_values)
    generator = torch.Generator()
    generator.manual_seed(G2C_SHARED_SAMPLER_SEED)
    loader = DataLoader(
        dataset,
        batch_size=training["batch_size"],
        shuffle=True,
        num_workers=training["num_workers"],
        collate_fn=_collate_training,
        generator=generator,
        drop_last=False,
        pin_memory=False,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training["epochs"],
        eta_min=training["eta_min"],
    )
    device = torch.device("cuda")
    model.train()
    trace = []
    sample_order: list[dict[str, Any]] = []
    iterator = iter(loader)
    for step in range(1, optimizer_steps + 1):
        batch = next(iterator)
        sample_order.extend(batch["sample_order"])
        moved = {
            name: value.to(device)
            if isinstance(value, torch.Tensor)
            else value
            for name, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        supervision = _build_supervision(moved)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            output = model(
                moved["image"],
                moved["structured_state"],
                moved["geometric_motion"],
            )
            loss = precision_unet_loss(output, supervision, loss_config)
        if not bool(torch.isfinite(loss.loss)):
            raise RuntimeError(f"G2C {candidate_id} smoke loss NaN/Inf")
        loss.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, training["gradient_clip_norm"]
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"G2C {candidate_id} smoke gradient NaN/Inf")
        post_clip_squared = sum(
            float(torch.sum(parameter.grad.detach().float().square()).item())
            for parameter in parameters
            if parameter.grad is not None
        )
        post_clip_gradient_norm = math.sqrt(post_clip_squared)
        if (
            not math.isfinite(post_clip_gradient_norm)
            or post_clip_gradient_norm
            > float(training["gradient_clip_norm"]) + 1e-5
        ):
            raise RuntimeError(f"G2C {candidate_id} post-clip gradient 漂移")
        keypoint_log_variance = (
            output.keypoint_log_variance.detach().float().reshape(-1)
        )
        if not bool(torch.isfinite(keypoint_log_variance).all()):
            raise RuntimeError(
                f"G2C {candidate_id} keypoint log-variance NaN/Inf"
            )
        loss_components = {
            name: float(getattr(loss, name).detach().float().item())
            for name in (
                "loss",
                "heatmap_loss",
                "mask_loss",
                "coordinate_loss",
                "motion_loss",
                "uncertainty_loss",
                "visibility_loss",
                "projection_loss",
            )
        }
        loss_components["weighted_keypoint_localization_loss"] = (
            loss_values["heatmap_weight"] * loss_components["heatmap_loss"]
            + loss_values["coordinate_weight"]
            * loss_components["coordinate_loss"]
        )
        loss_components["weighted_uncertainty_loss"] = (
            loss_values["uncertainty_weight"]
            * loss_components["uncertainty_loss"]
        )
        if any(not math.isfinite(value) for value in loss_components.values()):
            raise RuntimeError(f"G2C {candidate_id} loss component NaN/Inf")
        optimizer.step()
        trace.append(
            {
                "candidate_id": candidate_id,
                "optimizer_step": step,
                "batch_size": int(moved["image"].shape[0]),
                "loss": loss_components["loss"],
                "loss_components": loss_components,
                "gradient_norm": float(gradient_norm.detach().float().item()),
                "gradient_norm_pre_clip": float(
                    gradient_norm.detach().float().item()
                ),
                "gradient_norm_post_clip": post_clip_gradient_norm,
                "keypoint_log_variance": {
                    "min": float(keypoint_log_variance.min().item()),
                    "p50": float(
                        torch.quantile(keypoint_log_variance, 0.5).item()
                    ),
                    "max": float(keypoint_log_variance.max().item()),
                },
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
    scheduler.step()
    final_motion_hash = precision_parameter_state_sha256(model.motion_head.state_dict())
    first_log_variance = trace[0]["keypoint_log_variance"]
    initial_front_keypoint_log_variance_all_zero = all(
        float(first_log_variance[name]) == 0.0 for name in ("min", "p50", "max")
    )
    if candidate_id == "W-KV0" and not initial_front_keypoint_log_variance_all_zero:
        raise RuntimeError("G2C W-KV0 初始 front keypoint log-variance 不为零")
    result = {
        "candidate_id": candidate_id,
        "optimizer_step_count": optimizer_steps,
        "examples_seen": len(sample_order),
        "shared_sampler_seed": G2C_SHARED_SAMPLER_SEED,
        "post_smoke_parameter_sha256": precision_parameter_state_sha256(
            model.state_dict()
        ),
        "post_smoke_motion_head_parameter_sha256": final_motion_hash,
        "initial_front_keypoint_log_variance_all_zero": (
            initial_front_keypoint_log_variance_all_zero
        ),
        "scheduler_step_count": 1,
        "checkpoint_persisted": False,
    }
    return trace, sample_order, result


def _predict_candidate_smoke(
    *,
    model: Any,
    dataset: G2CDeployableDataset,
    candidate_id: str,
    parameter_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    model.eval()
    batch_size = 32
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    forward_batches = 0
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            samples = [
                dataset[index]
                for index in range(start, min(start + batch_size, len(dataset)))
            ]
            image_np = np.stack(
                [
                    sample["model_inputs"]["rgb_external"].transpose(2, 0, 1)
                    for sample in samples
                ]
            ).astype(np.float32)
            image_np /= np.float32(255.0)
            state_np = np.stack(
                [sample["model_inputs"]["structured_state"] for sample in samples]
            )
            motion_np = np.stack(
                [sample["model_inputs"]["geometric_motion"] for sample in samples]
            )
            image = torch.from_numpy(image_np).to(torch.device("cuda"))
            state = torch.from_numpy(state_np).to(torch.device("cuda"))
            motion = torch.from_numpy(motion_np).to(torch.device("cuda"))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                output = model(image, state, motion)
            decoded = output.decode_for_control(temperature=1.0)
            predicted_uv = decoded.keypoints.normalized_uv.detach().float().cpu().numpy()
            visibility = decoded.visibility_probability.detach().float().cpu().numpy()
            projection = (
                decoded.projection_validity_probability.detach().float().cpu().numpy()
            )
            entropy = decoded.keypoints.normalized_entropy.detach().float().cpu().numpy()
            sigma = decoded.keypoint_sigma_px.detach().float().cpu().numpy()
            mask = torch.sigmoid(output.mask_logits.detach().float()).cpu().numpy()
            forward_batches += 1
            for batch_index, sample in enumerate(samples):
                capture = sample["capture"]
                object_uv = predicted_uv[batch_index, 0]
                object_mask_probability = mask_probability_at_normalized_uv(
                    mask[batch_index, 0], object_uv
                )
                goal_mask_probability = mask_probability_at_normalized_uv(
                    mask[batch_index, 1], object_uv
                )
                try:
                    geometry = geometry_conditioning(
                        normalized_uv=object_uv,
                        intrinsic_cv=capture["external_intrinsic_cv"],
                        base_from_camera_cv=capture[
                            "base_from_external_camera_cv"
                        ],
                        image_size_hw=(128, 128),
                        plane_base_z_m=0.02,
                    )
                    predicted_position = np.asarray(
                        geometry["predicted_world_point_base_m"], dtype=np.float64
                    )
                    covariance = _measurement_covariance(
                        geometry["local_jacobian_xy_m_per_px"],
                        sigma[batch_index, 0],
                    )
                    geometry_valid = True
                except ValueError:
                    predicted_position = None
                    covariance = None
                    geometry_valid = False
                evidence = ObjectWriteEvidence(
                    visibility_probability=float(visibility[batch_index, 0]),
                    projection_validity_probability=float(projection[batch_index]),
                    object_mask_probability=float(object_mask_probability),
                    goal_mask_probability=float(goal_mask_probability),
                    normalized_entropy=float(entropy[batch_index, 0]),
                    radial_sigma_px=float(np.linalg.norm(sigma[batch_index, 0])),
                    geometry_valid=geometry_valid,
                )
                rows.append(
                    {
                        "version": E018_P1_G2C_SMOKE_RESULT_VERSION,
                        "phase": "prediction-before-privileged-sidecar-open/v1",
                        "candidate_id": candidate_id,
                        "candidate_parameter_sha256": parameter_sha256,
                        "candidate_initialization_seed": (
                            G2C_CANDIDATE_INITIALIZATION_SEEDS[candidate_id]
                        ),
                        "shared_sampler_seed": G2C_SHARED_SAMPLER_SEED,
                        "seed": int(capture["seed"]),
                        "split": str(capture["split"]),
                        "sample_index": int(capture["sample_index"]),
                        "viewpoint_id": str(capture["viewpoint_id"]),
                        "input_sha256": str(capture["input_sha256"]),
                        "actual_base_from_camera_cv": np.asarray(
                            capture["base_from_external_camera_cv"], dtype=float
                        ).tolist(),
                        "intrinsic_cv": np.asarray(
                            capture["external_intrinsic_cv"], dtype=float
                        ).tolist(),
                        "predicted_object_normalized_uv": object_uv.astype(
                            float
                        ).tolist(),
                        "predicted_goal_normalized_uv": predicted_uv[
                            batch_index, 1
                        ].astype(float).tolist(),
                        "object_visibility_probability": float(
                            visibility[batch_index, 0]
                        ),
                        "goal_visibility_probability": float(
                            visibility[batch_index, 1]
                        ),
                        "projection_validity_probability": float(
                            projection[batch_index]
                        ),
                        "object_normalized_entropy": float(
                            entropy[batch_index, 0]
                        ),
                        "object_sigma_xy_px": sigma[batch_index, 0].astype(
                            float
                        ).tolist(),
                        "object_mask_probability_at_prediction": float(
                            object_mask_probability
                        ),
                        "goal_mask_probability_at_prediction": float(
                            goal_mask_probability
                        ),
                        "predicted_observable": bool(
                            visibility[batch_index, 0] >= 0.5
                        ),
                        "geometry_valid": geometry_valid,
                        "predicted_object_position_base_m": (
                            None
                            if predicted_position is None
                            else predicted_position.tolist()
                        ),
                        "raw_covariance_base_m2": (
                            None if covariance is None else covariance.tolist()
                        ),
                        "write_score": evidence.score,
                        "qualification_only": True,
                        "memory_write_allowed": False,
                        "actuation_allowed": False,
                    }
                )
    assert_g2c_prediction_ledger_deployable_only(rows)
    audit = {
        "candidate_id": candidate_id,
        "prediction_count": len(rows),
        "forward_batch_count": forward_batches,
        "elapsed_s": time.perf_counter() - started,
        "privileged_label_open_count": 0,
        "test_array_read_count": 0,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return rows, audit


def _score_smoke_after_prediction_freeze(
    *,
    output_root: Path,
    data_root: Path,
    config_sha256: str,
    data_identity_sha256: str,
    evaluation_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Phase B 入口刻意不接收 model、Dataset 或内存 prediction rows。"""

    predictions, marker = load_frozen_g2c_prediction_ledger(
        output_root,
        config_sha256=config_sha256,
        data_identity_sha256=data_identity_sha256,
    )
    label_opened_at = time.time_ns()
    manifest_rows = _read_jsonl(
        data_root / "privileged_labels" / "manifest.jsonl",
        "G2C smoke privileged manifest",
    )
    matches = [
        row
        for row in manifest_rows
        if row.get("split") == G2C_SMOKE_SPLIT
        and int(row.get("seed", -1)) == evaluation_seed
    ]
    if len(matches) != 1:
        raise RuntimeError("G2C smoke evaluation label bundle identity 不唯一")
    meta = matches[0]
    label_path = _resolve_artifact_file(
        data_root / "privileged_labels", str(meta["file"])
    )
    if file_sha256(label_path) != meta["sha256"]:
        raise RuntimeError("G2C smoke evaluation label SHA-256 漂移")
    labels = _load_npz(label_path, _LABEL_ARRAYS)
    by_view = {
        str(labels["viewpoint_id"][index]): index
        for index in range(len(labels["viewpoint_id"]))
    }
    if tuple(by_view) != G2C_VIEW_ORDER:
        raise RuntimeError("G2C smoke evaluation label viewpoint order 漂移")
    scoring_rows = []
    for prediction in predictions:
        viewpoint = str(prediction["viewpoint_id"])
        index = by_view[viewpoint]
        if (
            int(prediction["seed"]) != evaluation_seed
            or int(prediction["sample_index"])
            != int(labels["source_sample_index"][index])
        ):
            raise RuntimeError("G2C smoke prediction/label row identity 漂移")
        gt_position = labels["object_position_base_m"][index].astype(np.float64)
        predicted_value = prediction["predicted_object_position_base_m"]
        predicted_position = (
            None
            if predicted_value is None
            else np.asarray(predicted_value, dtype=np.float64)
        )
        world_error = (
            None
            if predicted_position is None
            else float(np.linalg.norm(predicted_position - gt_position))
        )
        scoring_rows.append(
            {
                "version": E018_P1_G2C_SMOKE_RESULT_VERSION,
                "phase": "privileged-scoring-after-prediction-freeze/v1",
                "prediction_ledger_sha256": marker[
                    "prediction_ledger_raw_sha256"
                ],
                "candidate_id": prediction["candidate_id"],
                "seed": evaluation_seed,
                "viewpoint_id": viewpoint,
                "gt_observable": bool(labels["keypoint_observable"][index, 0]),
                "predicted_observable": bool(prediction["predicted_observable"]),
                "geometry_valid": bool(prediction["geometry_valid"]),
                "world_xyz_error_m": world_error,
                "gt_object_position_base_m": gt_position.tolist(),
                "predicted_object_position_base_m": predicted_value,
                "raw_covariance_base_m2": prediction["raw_covariance_base_m2"],
                "write_score": prediction["write_score"],
                "used_for_formal_selection": False,
                "test_data_read": False,
            }
        )
    summaries = []
    for candidate_id in G2C_CANDIDATE_IDS:
        for viewpoint in G2C_VIEW_ORDER:
            subset = [
                row
                for row in scoring_rows
                if row["candidate_id"] == candidate_id
                and row["viewpoint_id"] == viewpoint
            ]
            summary = summarize_g2c_model_val_view(
                subset, viewpoint_id=viewpoint
            )
            summary["candidate_id"] = candidate_id
            summaries.append(summary)
    audit = {
        "version": E018_P1_G2C_SMOKE_RESULT_VERSION,
        "prediction_freeze_sha256": marker["freeze_sha256"],
        "prediction_ledger_sha256": marker["prediction_ledger_raw_sha256"],
        "prediction_count": len(predictions),
        "scoring_row_count": len(scoring_rows),
        "privileged_label_bundle_open_count": 1,
        "privileged_label_opened_after_freeze": label_opened_at
        > marker["frozen_at_unix_ns"],
        "evaluation_seed": evaluation_seed,
        "expected_low_support_viewpoint_count": len(summaries),
        "ineligible_low_support_viewpoint_count": sum(
            "observable_positive_support_below_30"
            in summary["ineligibility_reasons"]
            for summary in summaries
        ),
        "formal_selection_evaluated": False,
        "test_array_read_count": 0,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return scoring_rows, audit, summaries


def verify_g2c_smoke_receipt(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    path = root / "smoke_receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    internal = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if internal != canonical_sha256(unsigned):
        raise RuntimeError("G2C smoke receipt internal SHA-256 漂移")
    if (
        receipt.get("version") != E018_P1_G2C_SMOKE_RESULT_VERSION
        or receipt.get("status") != "complete-engineering-smoke-pass"
        or receipt.get("checkpoint_write_count") != 0
        or receipt.get("test_trajectory_array_read_count") != 0
        or receipt.get("test_label_array_read_count") != 0
        or receipt.get("memory_read_count") != 0
        or receipt.get("memory_write_count") != 0
        or receipt.get("runtime_camera_actuation_count") != 0
        or receipt.get("physical_camera_actuation_count") != 0
        or receipt.get("arm_motion_command_count") != 0
        or receipt.get("gripper_close_command_count") != 0
        or receipt.get("manipulation_progression_count") != 0
    ):
        raise RuntimeError("G2C smoke receipt status/permission 漂移")
    artifacts = receipt.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("G2C smoke artifact hashes 缺失")
    for name, expected in artifacts.items():
        if file_sha256(root / name) != expected:
            raise RuntimeError(f"G2C smoke artifact SHA-256 漂移: {name}")
    data_verification = verify_g2c_data_receipt(root / "data")
    summary = json.loads((root / "smoke_summary.json").read_text(encoding="utf-8"))
    diagnostic_count_names = (
        "raw_reset_diagnostic_count",
        "post_warmup_diagnostic_count",
        "reset_diagnostic_count",
    )
    timestamp_identity_names = (
        "static_capture_timestamp_source",
        "static_capture_simulation_control_time_s",
        "static_capture_sequence_field",
        "static_views_share_timestamp_without_environment_step",
    )
    provider_root = root / "provider_smoke"
    protocol = json.loads(
        (provider_root / "training_protocol_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    initializations = json.loads(
        (provider_root / "candidate_initializations.json").read_text(
            encoding="utf-8"
        )
    )
    traces = json.loads(
        (provider_root / "training_trace.json").read_text(encoding="utf-8")
    )
    candidate_results = json.loads(
        (provider_root / "candidate_results.json").read_text(encoding="utf-8")
    )
    sampler_orders = json.loads(
        (provider_root / "sampler_orders.json").read_text(encoding="utf-8")
    )
    expected_candidates = set(G2C_CANDIDATE_IDS)
    if (
        canonical_sha256(protocol) != canonical_sha256(g2c_training_protocol())
        or set(initializations) != expected_candidates
        or set(traces) != expected_candidates
        or set(candidate_results) != expected_candidates
        or set(sampler_orders) != expected_candidates
        or summary.get("candidate_initializations") != initializations
        or summary.get("candidate_results") != candidate_results
    ):
        raise RuntimeError("G2C smoke protocol/candidate artifact identity 漂移")

    config = json.loads(
        (root / "data" / "config_snapshot.json").read_text(encoding="utf-8")
    )
    warm = initializations["W-KV0"]
    scratch = initializations["S"]
    reset = warm.get("keypoint_logvariance_reset")
    sha_fields = (
        "parameter_sha256_before",
        "parameter_sha256_after",
        "keypoint_logvariance_rows_sha256_before",
        "keypoint_logvariance_rows_sha256_after",
        "non_target_parameter_sha256_before",
        "non_target_parameter_sha256_after",
        "motion_head_sha256_before",
        "motion_head_sha256_after",
    )
    if (
        warm.get("candidate_id") != "W-KV0"
        or warm.get("initialization_seed") != 18021
        or warm.get("shared_sampler_seed") != G2C_SHARED_SAMPLER_SEED
        or warm.get("kind")
        != "e016-selected-epoch12-warm-start-keypoint-variance-zero"
        or warm.get("source_checkpoint_sha256")
        != config["parents"]["e016_checkpoint_sha256"]
        or warm.get("source_parameter_sha256")
        != config["parents"]["e016_checkpoint_parameter_sha256"]
        or warm.get("source_provenance_sha256")
        != config["parents"]["e016_checkpoint_provenance_sha256"]
        or warm.get("model_config_sha256")
        != config["parents"]["e016_checkpoint_model_config_sha256"]
        or not isinstance(reset, dict)
        or reset.get("policy")
        != "zero-final-uncertainty-linear-keypoint-logvariance-rows/v1"
        or reset.get("row_count") != 4
        or reset.get("row_indices") != [0, 1, 2, 3]
        or reset.get("keypoint_logvariance_rows_all_zero_after") is not True
        or reset.get("non_target_parameters_unchanged") is not True
        or reset.get("motion_head_unchanged") is not True
        or reset.get("parameter_sha256_before")
        != warm.get("source_parameter_sha256")
        or reset.get("parameter_sha256_after")
        != warm.get("initial_parameter_sha256")
        or reset.get("parameter_sha256_before")
        == reset.get("parameter_sha256_after")
        or reset.get("keypoint_logvariance_rows_sha256_before")
        == reset.get("keypoint_logvariance_rows_sha256_after")
        or reset.get("non_target_parameter_sha256_before")
        != reset.get("non_target_parameter_sha256_after")
        or reset.get("motion_head_sha256_before")
        != reset.get("motion_head_sha256_after")
        or reset.get("motion_head_sha256_after")
        != warm.get("initial_motion_head_parameter_sha256")
        or any(
            not isinstance(reset.get(name), str)
            or re.fullmatch(r"[0-9a-f]{64}", reset[name]) is None
            for name in sha_fields
        )
        or scratch.get("candidate_id") != "S"
        or scratch.get("initialization_seed") != 18022
        or scratch.get("shared_sampler_seed") != G2C_SHARED_SAMPLER_SEED
        or scratch.get("kind") != "random"
        or scratch.get("keypoint_logvariance_reset") is not None
    ):
        raise RuntimeError("G2C smoke W-KV0/S initialization audit 漂移")

    if (
        sampler_orders["W-KV0"] != sampler_orders["S"]
        or len(sampler_orders["W-KV0"]) != 33
    ):
        raise RuntimeError("G2C smoke shared sampler order 漂移")
    loss_component_names = {
        "loss",
        "heatmap_loss",
        "mask_loss",
        "coordinate_loss",
        "motion_loss",
        "uncertainty_loss",
        "visibility_loss",
        "projection_loss",
        "weighted_keypoint_localization_loss",
        "weighted_uncertainty_loss",
    }
    for candidate_id in G2C_CANDIDATE_IDS:
        initialization = initializations[candidate_id]
        candidate_result = candidate_results[candidate_id]
        candidate_trace = traces[candidate_id]
        if (
            candidate_result.get("candidate_id") != candidate_id
            or candidate_result.get("optimizer_step_count") != 2
            or candidate_result.get("examples_seen") != 33
            or candidate_result.get("shared_sampler_seed")
            != G2C_SHARED_SAMPLER_SEED
            or candidate_result.get("scheduler_step_count") != 1
            or candidate_result.get("checkpoint_persisted") is not False
            or candidate_result.get("post_smoke_motion_head_parameter_sha256")
            != initialization.get("initial_motion_head_parameter_sha256")
            or len(candidate_trace) != 2
            or [row.get("optimizer_step") for row in candidate_trace] != [1, 2]
            or sum(int(row.get("batch_size", -1)) for row in candidate_trace)
            != 33
        ):
            raise RuntimeError(f"G2C smoke {candidate_id} result/trace 漂移")
        for row in candidate_trace:
            components = row.get("loss_components")
            log_variance = row.get("keypoint_log_variance")
            numeric_values = (
                row.get("loss"),
                row.get("gradient_norm"),
                row.get("gradient_norm_pre_clip"),
                row.get("gradient_norm_post_clip"),
                row.get("learning_rate"),
            )
            if (
                row.get("candidate_id") != candidate_id
                or not isinstance(components, dict)
                or set(components) != loss_component_names
                or not isinstance(log_variance, dict)
                or set(log_variance) != {"min", "p50", "max"}
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in (
                        *numeric_values,
                        *components.values(),
                        *log_variance.values(),
                    )
                )
                or row.get("loss") != components.get("loss")
                or row.get("gradient_norm")
                != row.get("gradient_norm_pre_clip")
                or float(row["gradient_norm_post_clip"]) > 1.0 + 1e-5
            ):
                raise RuntimeError(
                    f"G2C smoke {candidate_id} finite loss/gradient audit 漂移"
                )
    if (
        candidate_results["W-KV0"].get(
            "initial_front_keypoint_log_variance_all_zero"
        )
        is not True
        or traces["W-KV0"][0].get("keypoint_log_variance")
        != {"min": 0.0, "p50": 0.0, "max": 0.0}
    ):
        raise RuntimeError("G2C smoke W-KV0 首批 keypoint log-variance 漂移")
    if (
        summary.get("status") != receipt["status"]
        or summary.get("prediction_before_label", {}).get("passed") is not True
        or summary.get("sampler_order_identical") is not True
        or summary.get("checkpoint_persisted") is not False
        or list(root.rglob("*.pt"))
        or list(root.rglob("*.pth"))
        or list(root.rglob("*.ckpt"))
        or any(
            summary.get(name) != data_verification[name]
            or receipt.get(name) != data_verification[name]
            for name in diagnostic_count_names
        )
        or any(
            summary.get(name) != data_verification[name]
            or receipt.get(name) != data_verification[name]
            for name in timestamp_identity_names
        )
    ):
        raise RuntimeError("G2C smoke summary/no-checkpoint invariant 漂移")
    result = {
        "version": E018_P1_G2C_SMOKE_RESULT_VERSION,
        "status": receipt["status"],
        "verified": True,
        "receipt_raw_sha256": file_sha256(path),
        "receipt_sha256": internal,
        "data_receipt_sha256": data_verification["receipt_sha256"],
        "prediction_ledger_sha256": summary["prediction_before_label"][
            "prediction_ledger_sha256"
        ],
        **{name: data_verification[name] for name in diagnostic_count_names},
        **{name: data_verification[name] for name in timestamp_identity_names},
        "checkpoint_file_count": 0,
    }
    result["verification_sha256"] = canonical_sha256(result)
    return result


def run_e018_p1_g2c_smoke(
    *,
    config_path: str | Path,
    parent_g0c_config_path: str | Path,
    parent_g0c_receipt_path: str | Path,
    e016_config_path: str | Path,
    e013_deployable_root: str | Path,
    e016_fresh_deployable_root: str | Path,
    training_output: str | Path,
    stats_root: str | Path,
    inventory_path: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """运行唯一获批的 4-seed / W-KV0+S / no-checkpoint engineering smoke。"""

    import gc

    import torch

    root = Path(output_root)
    if root.exists():
        raise FileExistsError(f"G2C smoke output 已存在: {root}")
    config = load_e018_p1_g2c_data_config(
        config_path, parent_g0c_config_path=parent_g0c_config_path
    )
    data_result = run_e018_p1_g2c_data(
        config_path=config_path,
        parent_g0c_config_path=parent_g0c_config_path,
        parent_g0c_receipt_path=parent_g0c_receipt_path,
        e013_deployable_root=e013_deployable_root,
        e016_fresh_deployable_root=e016_fresh_deployable_root,
        stats_root=stats_root,
        inventory_path=inventory_path,
        repository_root=repository_root,
        output_root=root / "data",
        mode="smoke",
        decision_exit_go=False,
    )
    if data_result["gate_passed"] is not True:
        raise RuntimeError("G2C DATA smoke lifecycle 未通过，禁止训练 smoke")
    data_receipt_path = root / "data" / "data_receipt.json"
    data_receipt = json.loads(data_receipt_path.read_text(encoding="utf-8"))
    provider_root = root / "provider_smoke"
    provider_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    protocol = g2c_training_protocol()
    _atomic_json(provider_root / "training_protocol_snapshot.json", protocol)
    training_seeds = list(config["engineering_smoke"]["training_seeds"])
    evaluation_seed = int(config["engineering_smoke"]["prediction_freeze_seed"])
    traces: dict[str, list[dict[str, Any]]] = {}
    sample_orders: dict[str, list[dict[str, Any]]] = {}
    candidate_results: dict[str, dict[str, Any]] = {}
    candidate_initializations: dict[str, dict[str, Any]] = {}
    prediction_rows: list[dict[str, Any]] = []
    prediction_audits: list[dict[str, Any]] = []
    for candidate_id in G2C_CANDIDATE_IDS:
        training_dataset = G2CFrontTrainingDataset(
            root / "data", G2C_SMOKE_SPLIT, seeds=training_seeds
        )
        model, initialization = _load_candidate_model(
            candidate_id=candidate_id,
            e016_config_path=Path(e016_config_path),
            training_output=Path(training_output),
            config=config,
        )
        trace, order, candidate_result = _train_candidate_smoke(
            model=model,
            dataset=training_dataset,
            candidate_id=candidate_id,
            optimizer_steps=int(
                config["engineering_smoke"][
                    "training_optimizer_steps_per_candidate"
                ]
            ),
        )
        if (
            candidate_result["post_smoke_motion_head_parameter_sha256"]
            != initialization["initial_motion_head_parameter_sha256"]
        ):
            raise RuntimeError(f"G2C {candidate_id} frozen motion head 发生漂移")
        eval_dataset = G2CDeployableDataset(
            root / "data", G2C_SMOKE_SPLIT, seeds=[evaluation_seed]
        )
        rows, prediction_audit = _predict_candidate_smoke(
            model=model,
            dataset=eval_dataset,
            candidate_id=candidate_id,
            parameter_sha256=candidate_result["post_smoke_parameter_sha256"],
        )
        traces[candidate_id] = trace
        sample_orders[candidate_id] = order
        candidate_results[candidate_id] = candidate_result
        candidate_initializations[candidate_id] = initialization
        prediction_rows.extend(rows)
        prediction_audits.append(prediction_audit)
        del eval_dataset, training_dataset, model
        gc.collect()
        torch.cuda.empty_cache()
    sampler_order_identical = sample_orders["W-KV0"] == sample_orders["S"]
    if not sampler_order_identical:
        raise RuntimeError("G2C W-KV0/S 未使用完全相同 sampler order")
    warm_start_initialization = candidate_initializations["W-KV0"]
    warm_start_reset = warm_start_initialization.get(
        "keypoint_logvariance_reset"
    )
    if (
        not isinstance(warm_start_reset, dict)
        or warm_start_reset.get("keypoint_logvariance_rows_all_zero_after")
        is not True
        or warm_start_reset.get("non_target_parameters_unchanged") is not True
        or warm_start_reset.get("motion_head_unchanged") is not True
        or candidate_results["W-KV0"].get(
            "initial_front_keypoint_log_variance_all_zero"
        )
        is not True
    ):
        raise RuntimeError("G2C W-KV0 初始化审计未通过")
    _atomic_json(provider_root / "candidate_initializations.json", candidate_initializations)
    _atomic_json(provider_root / "training_trace.json", traces)
    _atomic_json(provider_root / "candidate_results.json", candidate_results)
    _atomic_json(provider_root / "sampler_orders.json", sample_orders)
    _atomic_json(provider_root / "prediction_audits.json", prediction_audits)
    freeze = freeze_g2c_prediction_ledger(
        provider_root,
        rows=prediction_rows,
        config_sha256=data_receipt["config_sha256"],
        data_identity_sha256=data_receipt["data_identity_sha256"],
    )
    del prediction_rows
    gc.collect()
    torch.cuda.empty_cache()
    scoring_rows, scoring_audit, view_summaries = (
        _score_smoke_after_prediction_freeze(
            output_root=provider_root,
            data_root=root / "data",
            config_sha256=data_receipt["config_sha256"],
            data_identity_sha256=data_receipt["data_identity_sha256"],
            evaluation_seed=evaluation_seed,
        )
    )
    _atomic_jsonl(provider_root / "model_val_scoring_ledger.jsonl", scoring_rows)
    _atomic_json(provider_root / "model_val_scoring_audit.json", scoring_audit)
    _atomic_json(provider_root / "model_val_view_summaries.json", view_summaries)
    no_checkpoint = not any(
        path.suffix in {".pt", ".pth", ".ckpt"} for path in root.rglob("*")
    )
    if not no_checkpoint:
        raise RuntimeError("G2C smoke 禁止持久化 checkpoint")
    source_identity = json.loads(
        (root / "data" / "source_identity.json").read_text(encoding="utf-8")
    )
    permissions = {
        "simulator_camera_pose_set_count": data_result["permissions"][
            "simulator_camera_pose_set_count"
        ],
        "simulator_safe_hold_open_step_count": data_result["permissions"][
            "simulator_safe_hold_open_step_count"
        ],
        "provider_training_optimizer_step_count": sum(
            item["optimizer_step_count"] for item in candidate_results.values()
        ),
        "training_privileged_sample_read_count": sum(
            item["examples_seen"] for item in candidate_results.values()
        ),
        "model_val_privileged_label_bundle_open_count": scoring_audit[
            "privileged_label_bundle_open_count"
        ],
        "checkpoint_write_count": 0,
        "runtime_camera_actuation_count": 0,
        "physical_camera_actuation_count": 0,
        "arm_motion_command_count": 0,
        "gripper_close_command_count": 0,
        "manipulation_progression_count": 0,
        "memory_read_count": 0,
        "memory_write_count": 0,
        "test_trajectory_array_read_count": 0,
        "test_label_array_read_count": 0,
    }
    summary = {
        "version": E018_P1_G2C_SMOKE_RESULT_VERSION,
        "status": "complete-engineering-smoke-pass",
        "classification": "implementation-smoke-only-not-provider-result",
        "full_data_r2_barrier_reached": True,
        "formal_training_started": False,
        "formal_selection_evaluated": False,
        "formal_calibration_evaluated": False,
        "dynamic_qualification_executed": False,
        "config_sha256": data_receipt["config_sha256"],
        "source_git_commit": source_identity["git_commit"],
        "source_identity_sha256": source_identity["identity_sha256"],
        "data_receipt_raw_sha256": file_sha256(data_receipt_path),
        "data_receipt_internal_sha256": data_receipt["receipt_sha256"],
        "data_identity_sha256": data_receipt["data_identity_sha256"],
        "e016_checkpoint_sha256": config["parents"]["e016_checkpoint_sha256"],
        "e016_checkpoint_parameter_sha256": config["parents"][
            "e016_checkpoint_parameter_sha256"
        ],
        "candidate_initializations": candidate_initializations,
        "candidate_results": candidate_results,
        "smoke_seeds": list(config["sampling"]["smoke_only_seeds"]),
        "training_seeds": training_seeds,
        "prediction_freeze_seed": evaluation_seed,
        "view_order": list(G2C_VIEW_ORDER),
        "eligible_capture_count": data_result["eligible_capture_count"],
        "raw_reset_diagnostic_count": data_result[
            "raw_reset_diagnostic_count"
        ],
        "post_warmup_diagnostic_count": data_result[
            "post_warmup_diagnostic_count"
        ],
        "reset_diagnostic_count": data_result["reset_diagnostic_count"],
        "static_capture_timestamp_source": data_result[
            "static_capture_timestamp_source"
        ],
        "static_capture_simulation_control_time_s": data_result[
            "static_capture_simulation_control_time_s"
        ],
        "static_capture_sequence_field": data_result[
            "static_capture_sequence_field"
        ],
        "static_views_share_timestamp_without_environment_step": data_result[
            "static_views_share_timestamp_without_environment_step"
        ],
        "sampler_seed": G2C_SHARED_SAMPLER_SEED,
        "sampler_order_identical": sampler_order_identical,
        "prediction_before_label": {
            "passed": bool(
                scoring_audit["privileged_label_opened_after_freeze"]
                and freeze["privileged_label_open_count_before_freeze"] == 0
            ),
            "prediction_count": freeze["prediction_count"],
            "prediction_ledger_sha256": freeze[
                "prediction_ledger_raw_sha256"
            ],
            "prediction_freeze_sha256": freeze["freeze_sha256"],
            "privileged_label_bundle_open_count_after_freeze": scoring_audit[
                "privileged_label_bundle_open_count"
            ],
        },
        "model_val_smoke": {
            "row_count": len(scoring_rows),
            "view_summary_count": len(view_summaries),
            "support_gate_expected_to_fail": True,
            "support_gate_failure_is_protocol_invalid": False,
            "formal_selected_checkpoint": None,
        },
        "checkpoint_persisted": False,
        "dynamic_qualification_plan": g2c_dynamic_qualification_plan(),
        "permissions": permissions,
    }
    _atomic_json(root / "smoke_summary.json", summary)
    artifact_names = (
        "data/data_receipt.json",
        "smoke_summary.json",
        "provider_smoke/training_protocol_snapshot.json",
        "provider_smoke/candidate_initializations.json",
        "provider_smoke/training_trace.json",
        "provider_smoke/candidate_results.json",
        "provider_smoke/sampler_orders.json",
        "provider_smoke/prediction_audits.json",
        "provider_smoke/model_val_prediction_ledger.jsonl",
        "provider_smoke/model_val_prediction_freeze.json",
        "provider_smoke/model_val_scoring_ledger.jsonl",
        "provider_smoke/model_val_scoring_audit.json",
        "provider_smoke/model_val_view_summaries.json",
    )
    receipt = {
        "version": E018_P1_G2C_SMOKE_RESULT_VERSION,
        "status": summary["status"],
        "classification": summary["classification"],
        "config_sha256": summary["config_sha256"],
        "source_git_commit": summary["source_git_commit"],
        "source_identity_sha256": summary["source_identity_sha256"],
        "data_receipt_raw_sha256": summary["data_receipt_raw_sha256"],
        "data_receipt_internal_sha256": summary[
            "data_receipt_internal_sha256"
        ],
        "data_identity_sha256": summary["data_identity_sha256"],
        "prediction_ledger_sha256": freeze["prediction_ledger_raw_sha256"],
        "raw_reset_diagnostic_count": summary[
            "raw_reset_diagnostic_count"
        ],
        "post_warmup_diagnostic_count": summary[
            "post_warmup_diagnostic_count"
        ],
        "reset_diagnostic_count": summary["reset_diagnostic_count"],
        "static_capture_timestamp_source": summary[
            "static_capture_timestamp_source"
        ],
        "static_capture_simulation_control_time_s": summary[
            "static_capture_simulation_control_time_s"
        ],
        "static_capture_sequence_field": summary[
            "static_capture_sequence_field"
        ],
        "static_views_share_timestamp_without_environment_step": summary[
            "static_views_share_timestamp_without_environment_step"
        ],
        "artifact_sha256": {
            name: file_sha256(root / name) for name in artifact_names
        },
        **permissions,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_json(root / "smoke_receipt.json", receipt)
    verification = verify_g2c_smoke_receipt(root)
    return {**summary, "receipt": receipt, "verification": verification}


__all__ = [
    "E018_P1_G2C_SMOKE_RESULT_VERSION",
    "E018_P1_G2C_TRAIN_PROTOCOL_VERSION",
    "G2C_CANDIDATE_EPOCHS",
    "G2C_CANDIDATE_IDS",
    "G2C_CANDIDATE_INITIALIZATION_SEEDS",
    "G2C_SHARED_SAMPLER_SEED",
    "assert_g2c_prediction_ledger_deployable_only",
    "build_g2c_train_config_payload",
    "calibrate_g2c_viewpoint",
    "freeze_g2c_prediction_ledger",
    "g2c_dynamic_qualification_plan",
    "g2c_training_protocol",
    "load_frozen_g2c_prediction_ledger",
    "run_e018_p1_g2c_smoke",
    "select_g2c_checkpoint",
    "summarize_g2c_model_val_view",
    "validate_g2c_dynamic_qualification_counters",
    "verify_g2c_smoke_receipt",
]
