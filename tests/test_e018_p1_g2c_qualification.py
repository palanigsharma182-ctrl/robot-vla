from __future__ import annotations

import json
import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from robot_vla.precision import e018_p1_g2c_qualification as qualification
from robot_vla.precision.active_front_camera import ExternalCameraMotionState
from robot_vla.precision.e018_p1_g2a import canonical_sha256
from robot_vla.precision.e018_p1_g2c_qualification import (
    FORMAL_QUALIFICATION_SEEDS,
    QUALIFICATION_VIEW_ORDER,
    QualificationJournal,
    _type7_quantile,
    assert_qualification_prediction_deployable_only,
    build_qualification_deployable_capture,
    finalize_qualification_prediction,
    load_g2c_dynamic_qualification_config,
    process_qualification_hook_frame,
    qualification_scored_frame_identity,
    run_e018_p1_g2c_qualification_capture,
    run_e018_p1_g2c_qualification_smoke,
    score_e018_p1_g2c_qualification,
    score_qualification_prediction,
    select_qualification_primary,
    summarize_qualification_viewpoint,
    validate_qualification_prediction_mechanics,
    verify_g2c_qualification_result,
)
from robot_vla.precision.e018_p1_g2c_data import (
    load_e018_p1_g2c_data_config,
)

CONFIG_PATH = (
    Path(__file__).parents[1] / "configs/e018_p1_g2c_dynamic_qualification_development_v1.json"
)
DATA_CONFIG_PATH = (
    Path(__file__).parents[1] / "configs/e018_p1_g2c_front_provider_data_development_v1.json"
)
G0C_CONFIG_PATH = (
    Path(__file__).parents[1] / "configs/e018_p1_g0c_rotated_motion_development_v1.json"
)


class _PoisonMapping:
    def __getitem__(self, key: object) -> object:
        raise AssertionError(f"非评分帧不应读取 observation: {key}")


def _minimal_privileged_label() -> dict[str, object]:
    observability = qualification.ObjectObservabilityLabel(
        object_exists=True,
        projection_valid=True,
        in_fov=True,
        observable=True,
        legacy_visible=True,
        center_inside_object_mask=True,
        center_inside_goal_mask=False,
        local_object_visible_fraction=1.0,
        object_mask_area_fraction=1.0 / (128 * 128),
        occlusion_type="observable",
    )
    return {
        "gt_object_exists": True,
        "gt_observable": True,
        "gt_object_position_base_m": [0.0, 0.0, 0.02],
        "gt_object_projection_valid": True,
        "gt_object_projected_normalized_uv": [0.5, 0.5],
        "gt_object_mask_sha256": "a" * 64,
        "gt_object_visible_pixel_count": 1,
        "gt_object_observability": observability.to_dict(),
        "is_grasped": False,
        "robot_object_contact_force_n": 0.0,
        "goal_gt_read_count": 0,
        "test_data_read": False,
    }


def _prediction_capture_and_raw(
    *,
    row_index: int = 0,
    seed: int = 76801,
    sample_index: int = 1,
    viewpoint_id: str = "LEFT_LOW__CENTER",
    route_alternate_index: int = 0,
    route_frame_index: int = 47,
    input_sha256: str = "f" * 64,
) -> tuple[dict[str, object], dict[str, object]]:
    config = load_g2c_dynamic_qualification_config(CONFIG_PATH)
    checkpoint = config["parents"]["selected_checkpoint"]
    intrinsic = np.asarray(
        [[100.0, 0.0, 63.5], [0.0, 100.0, 63.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    base_from_camera = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    identity = {
        "row_index": row_index,
        "seed": seed,
        "sample_index": sample_index,
        "viewpoint_id": viewpoint_id,
        "frame_role": (
            "home-anchor-first-route-only/v1"
            if route_frame_index == 0
            else "alternate-final-collect/v1"
        ),
        "route_alternate_index": route_alternate_index,
        "route_alternate_viewpoint_id": (
            "LEFT_LOW__CENTER" if route_frame_index == 0 else viewpoint_id
        ),
        "route_frame_index": route_frame_index,
    }
    safety = {
        "eligible_capture": True,
        "finger_force_n": [0.0, 0.0],
        "finger_force_valid": True,
        "raw_gripper_opening_ratio": 1.0,
        "arm_joint_drift_rad": 0.0,
        "tcp_position_drift_m": 0.0,
        "tcp_orientation_drift_rad": 0.0,
        "rgb_timestamp_s": route_frame_index / 20.0,
        "pose_timestamp_s": route_frame_index / 20.0,
        "camera_position_tracking_error_m": 0.0,
        "camera_orientation_tracking_error_rad": 0.0,
        "rotation_projection_error_frobenius": 0.0,
    }
    capture = {
        "identity": identity,
        "input_sha256": input_sha256,
        "external_intrinsic_cv": intrinsic,
        "base_from_external_camera_cv": base_from_camera,
        "deployable_safety": safety,
    }
    sigma = np.asarray([0.1, 0.1], dtype=np.float64)
    geometry = qualification._recompute_prediction_geometry(
        normalized_uv=[0.5, 0.5],
        intrinsic_cv=intrinsic,
        base_from_camera_cv=base_from_camera,
        sigma_xy_px=sigma,
        plane_base_z_m=0.02,
    )
    evidence = qualification.ObjectWriteEvidence(
        visibility_probability=0.9,
        projection_validity_probability=0.9,
        object_mask_probability=0.9,
        goal_mask_probability=0.1,
        normalized_entropy=0.1,
        radial_sigma_px=float(np.linalg.norm(sigma)),
        geometry_valid=True,
    )
    raw = {
        "version": "e018-p1-g2c-prediction-freeze/v1",
        "phase": "deployable-model-val-before-privileged-label-open/v1",
        "candidate_id": checkpoint["candidate_id"],
        "epoch": checkpoint["epoch"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_parameter_sha256": checkpoint["parameter_state_sha256"],
        "checkpoint_provenance_sha256": checkpoint["provenance_sha256"],
        "checkpoint_model_config_sha256": checkpoint["model_config_sha256"],
        "row_index": row_index,
        "batch_index": row_index,
        "batch_offset": 0,
        "seed": seed,
        "split": "engineering_smoke",
        "sample_index": sample_index,
        "viewpoint_id": viewpoint_id,
        "input_sha256": input_sha256,
        "predicted_object_normalized_uv": [0.5, 0.5],
        "predicted_goal_normalized_uv": [0.5, 0.5],
        "object_visibility_probability": 0.9,
        "goal_visibility_probability": 0.1,
        "projection_validity_probability": 0.9,
        "object_normalized_entropy": 0.1,
        "object_sigma_xy_px": sigma.tolist(),
        "object_mask_probability_at_prediction": 0.9,
        "goal_mask_probability_at_prediction": 0.1,
        "predicted_observable": True,
        **geometry,
        "write_score": evidence.score,
        "memory_write_allowed": False,
        "actuation_allowed": False,
    }
    return capture, raw


def _rewrite_config(path: Path, config: dict[str, object]) -> None:
    unsigned = dict(config)
    unsigned.pop("config_sha256")
    config["config_sha256"] = canonical_sha256(unsigned)
    path.write_text(json.dumps(config), encoding="utf-8")


def _formal_decision_receipt() -> dict[str, object]:
    config = load_g2c_dynamic_qualification_config(CONFIG_PATH)
    smoke_gpu_seconds = 1.0
    smoke_bytes = 1_000
    pre_formal_gpu = (
        qualification._D048_PRE_SMOKE_GPU_SECONDS_CONSERVATIVE_UPPER + smoke_gpu_seconds
    )
    pre_formal_bytes = qualification._D048_PRE_SMOKE_ARTIFACT_BYTES_CONSERVATIVE_UPPER + smoke_bytes
    receipt: dict[str, object] = {
        "version": qualification._FORMAL_EXECUTION_DECISION_VERSION,
        "decision_id": "D048",
        "status": "GO-formal-dynamic-qualification-execution-and-one-shot-scoring",
        "qualification_config": {
            "raw_sha256": qualification.file_sha256(CONFIG_PATH),
            "internal_sha256": config["config_sha256"],
        },
        "source": {"git_commit": "4" * 40, "identity_sha256": "5" * 64},
        "parent_identities": {
            "g0c_config_sha256": config["parents"]["g0c_config_sha256"],
            "g0c_receipt_internal_sha256": config["parents"]["g0c_receipt_internal_sha256"],
            "d046_calibration_result_receipt_raw_sha256": config["parents"][
                "calibration_result_receipt_raw_sha256"
            ],
            "d046_calibration_result_receipt_internal_sha256": config["parents"][
                "calibration_result_receipt_internal_sha256"
            ],
            "d046_calibration_result_verification_sha256": config["parents"][
                "calibration_result_verification_sha256"
            ],
            "d046_replicated_persistence": {
                "artifact_id": qualification._D046_ARTIFACT_ID,
                "status": "REPLICATED",
                "local_verified_receipt_sha256": "a" * 64,
                "drive_verified_receipt_sha256": "b" * 64,
                "replicated_receipt_sha256": "c" * 64,
            },
            "selected_checkpoint": config["parents"]["selected_checkpoint"],
        },
        "d047_smoke": {
            "experiment_id": "E018-P1-G2C-D047-PREFLIGHT",
            "seed": 76801,
            "alternate_viewpoint_id": "LEFT_LOW__CENTER",
            "classification": qualification.QUALIFICATION_CLASSIFICATION_SMOKE,
            "execution_status": "complete-execution-freeze-context-destroyed",
            "result_status": "complete-preflight-no-qualification-claim",
            "execution_receipt_raw_sha256": "d" * 64,
            "execution_receipt_internal_sha256": "e" * 64,
            "execution_verification_sha256": "f" * 64,
            "result_receipt_raw_sha256": "1" * 64,
            "result_receipt_internal_sha256": "2" * 64,
            "result_verification_sha256": "3" * 64,
            "combined_artifact_verification_sha256": "4" * 64,
            "public_output_identity_sha256": "6" * 64,
            "private_output_identity_sha256": "7" * 64,
            "result_output_identity_sha256": "8" * 64,
            "started_at_unix_ns": 1,
            "completed_at_unix_ns": 2,
            "wall_elapsed_seconds": 1.0,
            "gpu_elapsed_seconds": smoke_gpu_seconds,
            "total_artifact_bytes": smoke_bytes,
            "formal_claim_allowed": False,
            "rerun_under_same_identity_allowed": False,
        },
        "formal_execution": {
            "experiment_id": "E018-P1-G2C-FORMAL-DYNAMIC-QUALIFICATION",
            "execution_id": "synthetic-d048-unit-test",
            "classification": qualification.QUALIFICATION_CLASSIFICATION_FORMAL,
            "public_output_identity_sha256": "9" * 64,
            "private_output_identity_sha256": "a" * 64,
            "result_output_identity_sha256": "b" * 64,
            "seed_start": 76701,
            "seed_end": 76750,
            "seed_count": 50,
            "alternate_order": list(qualification.FRONT_ALTERNATE_IDS),
            "route_count": 500,
            "expected_counts": {
                "camera_pose_set_count": 48_500,
                "ledger_frame_count": 46_000,
                "moving_interpolation_command_count": 40_000,
                "safe_hold_open_step_count": 48_000,
                "prediction_count": 550,
                "home_prediction_count": 50,
                "alternate_prediction_count": 500,
                "privileged_object_label_capture_count": 550,
                "object_contact_event_count": 0,
            },
            "capture_attempt_count": 1,
            "scoring_attempt_count": 1,
            "test_split_status": "prohibited-unread",
            "memory_and_active_loop": "HOLD",
            "actuator_and_manipulation": "HOLD",
        },
        "permissions": {name: 0 for name in qualification._FORMAL_DECISION_PERMISSION_KEYS},
        "budgets": {
            "version": "e018-p1-g2c-d036-conservative-cumulative-budget/v1",
            "audit": {
                "source": "read-only-receipt-mtime-and-filesystem-inventory/v1",
                "clock": "UTC-unix-time/v1",
                "worker_region": "US",
                "worker_gpu_class": "single-NVIDIA-RTX-6000-Ada-Generation",
                "audited_at_unix_ns": 1,
                "gpu_envelope_start_unix_s": 1_788_561_330.569288,
                "gpu_envelope_end_unix_s": 1_788_585_782.402512,
                "known_gpu_components_seconds": dict(qualification._D048_KNOWN_GPU_COMPONENTS),
                "unknown_gpu_components": list(qualification._D048_UNKNOWN_GPU_COMPONENTS),
                "artifact_inventory_item_count": 42,
                "artifact_duplicate_inclusive_bytes": 2_118_325_603,
                "artifact_scope": "all-g2c-run-and-drive-staging-trees-on-the-single-worker/v1",
                "artifact_counting_semantics": (
                    "retained-filesystem-bytes-with-cross-tree-duplicates-"
                    "conservatively-counted/v1"
                ),
            },
            "gpu": {
                "hard_limit_seconds": 36_000.0,
                "known_lower_bound_seconds": sum(qualification._D048_KNOWN_GPU_COMPONENTS.values()),
                "unknown_component_conservative_envelope_seconds": (
                    qualification._D048_UNKNOWN_GPU_SECONDS_CONSERVATIVE_ENVELOPE
                ),
                "pre_smoke_conservative_upper_bound_seconds": (
                    qualification._D048_PRE_SMOKE_GPU_SECONDS_CONSERVATIVE_UPPER
                ),
                "d047_smoke_actual_seconds": smoke_gpu_seconds,
                "pre_formal_conservative_upper_bound_seconds": pre_formal_gpu,
                "formal_wall_seconds_max": 1_000.0,
                "formal_gpu_seconds_reserved_max": 1_000.0,
                "projected_conservative_upper_bound_seconds": pre_formal_gpu + 1_000.0,
            },
            "artifact": {
                "hard_limit_bytes": 21_474_836_480,
                "audited_duplicate_inclusive_bytes": 2_118_325_603,
                "unknown_component_conservative_envelope_bytes": 2_176_641_693,
                "pre_smoke_conservative_upper_bound_bytes": 4_294_967_296,
                "d047_smoke_actual_bytes": smoke_bytes,
                "pre_formal_conservative_upper_bound_bytes": pre_formal_bytes,
                "formal_combined_artifact_bytes_reserved_max": 4_294_967_296,
                "projected_conservative_upper_bound_bytes": pre_formal_bytes + 4_294_967_296,
            },
        },
        "stopping_rules": {
            "route_or_gate_failure": "stop-consumed-failure-no-retry-no-replacement",
            "partial_route_failure": "freeze-partial-evidence-stop-no-rerun",
            "contact_or_ownership_violation": "stop-consumed-failure-no-actuation",
            "budget_exceeded": "stop-consumed-failure-no-promotion",
            "unsafe_or_catastrophic_result": "complete-scoring-fail-no-promotion",
            "rerun_after_consumption": "prohibited",
        },
        "allowed_claim": (
            "dynamic-front-provider-qualified-for-shadow-planning-only-"
            "no-memory-no-actuator-no-test/v1"
        ),
        "issued_at_unix_ns": 1,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _hook_row(frame_index: int, *, alternate: str) -> dict[str, object]:
    if frame_index == 0:
        state = ExternalCameraMotionState.HOME_ANCHOR.value
        viewpoint = "HOME"
    elif frame_index <= 40:
        state = ExternalCameraMotionState.MOVE_TO_VIEW.value
        viewpoint = alternate
    elif frame_index <= 44:
        state = ExternalCameraMotionState.SETTLE_AT_VIEW.value
        viewpoint = alternate
    elif frame_index <= 47:
        state = ExternalCameraMotionState.COLLECT.value
        viewpoint = alternate
    elif frame_index <= 87:
        state = ExternalCameraMotionState.RETURN_HOME.value
        viewpoint = "HOME"
    else:
        state = ExternalCameraMotionState.VERIFY_HOME_AND_ARM_HOLD.value
        viewpoint = "HOME"
    return {
        "frame_index": frame_index,
        "camera_motion_state": state,
        "viewpoint_primitive_id": viewpoint,
        "settled": frame_index in (45, 46, 47),
        "measurement_write_eligible": frame_index in (45, 46, 47),
    }


def _rotation_to_quaternion_wxyz(rotation: np.ndarray) -> list[float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    value = np.asarray([w, x, y, z], dtype=np.float64)
    value /= np.linalg.norm(value)
    return value.tolist()


def _synthetic_raw_route(
    *, seed: int = 76801, alternate_viewpoint_id: str = "LEFT_LOW__CENTER"
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, np.ndarray], dict[str, object]]:
    g0c_config = qualification._g0c.load_e018_p1_g0c_config(G0C_CONFIG_PATH)
    home, anchors, orientations = qualification._g0c._parse_library(g0c_config)
    primitives = qualification._g0c._expand_primitives(anchors, orientations)
    primitive, orientation = {item.viewpoint_id: (item, mode) for item, mode in primitives}[
        alternate_viewpoint_id
    ]
    motion = g0c_config["motion"]
    safety = g0c_config["safety"]
    move_steps = 40
    period = 1.0 / 20.0
    home_position = np.asarray(home.position_world_m, dtype=np.float64)
    alternate_position = np.asarray(primitive.position_world_m, dtype=np.float64)
    positions = [
        home_position,
        *qualification._g0.sample_translation_path(
            home_position, alternate_position, steps=move_steps
        ),
        *([alternate_position] * (motion["settle_ticks"] + motion["collect_ticks"])),
        *qualification._g0.sample_translation_path(
            alternate_position, home_position, steps=move_steps
        ),
        *([home_position] * motion["settle_ticks"]),
    ]
    states = tuple(
        state for state, count in qualification._ROUTE_STATE_PATTERN for _ in range(count)
    )
    target_offsets = np.asarray(
        [
            orientation.yaw_offset_rad,
            orientation.pitch_offset_rad,
            orientation.roll_offset_rad,
        ],
        dtype=np.float64,
    )
    mount = qualification._quaternion_to_rotation_wxyz([-0.5, -0.5, 0.5, 0.5])
    world_from_base = np.eye(4, dtype=np.float64)
    intrinsic = np.asarray(
        [[100.0, 0.0, 63.5], [0.0, 100.0, 63.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    home_image = np.zeros((128, 128, 3), dtype=np.uint8)
    alternate_image = np.full((128, 128, 3), 20, dtype=np.uint8)
    rgb_images = {
        "home_before": home_image,
        "alternate": alternate_image,
        "home_after": home_image.copy(),
    }
    home_sha = qualification._array_sha256(home_image)
    alternate_sha = qualification._array_sha256(alternate_image)
    episode_id = (
        f"g2c-qualification-seed-{seed:06d}-" f"{alternate_viewpoint_id.lower().replace('_', '-')}"
    )
    rows: list[dict[str, object]] = []
    previous_position: np.ndarray | None = None
    previous_quaternion: np.ndarray | None = None
    previous_velocity: np.ndarray | None = None
    previous_angular_speed: float | None = None
    settle_streak = int(motion["warmup_ticks"])
    for index, (state, position) in enumerate(zip(states, positions, strict=True)):
        if index == 0:
            viewpoint_id = "HOME"
            orientation_id = "CENTER"
            progress = 1.0
        elif index <= 40:
            viewpoint_id = alternate_viewpoint_id
            orientation_id = orientation.orientation_id
            progress = qualification._g0.smootherstep(index / move_steps)
        elif index <= 47:
            viewpoint_id = alternate_viewpoint_id
            orientation_id = orientation.orientation_id
            progress = 1.0
        elif index <= 87:
            viewpoint_id = "HOME"
            orientation_id = orientation.orientation_id
            progress = 1.0 - qualification._g0.smootherstep((index - 47) / move_steps)
        else:
            viewpoint_id = "HOME"
            orientation_id = "CENTER"
            progress = 1.0
        offset_scale = 0.0 if orientation_id == "CENTER" else progress
        offsets = target_offsets * offset_scale
        look_at = (
            primitive.look_at_world_m
            if viewpoint_id == alternate_viewpoint_id
            else home.look_at_world_m
        )
        rotation = qualification._expected_sapien_camera_rotation(
            position,
            look_at,
            yaw_offset_rad=float(offsets[0]),
            pitch_offset_rad=float(offsets[1]),
        )
        quaternion = np.asarray(_rotation_to_quaternion_wxyz(rotation), dtype=np.float64)
        world_from_gl = qualification._pose_matrix(position, rotation @ mount)
        base_from_cv = qualification.opengl_camera_to_opencv(world_from_gl)
        velocity = (
            np.zeros(3, dtype=np.float64)
            if previous_position is None
            else (position - previous_position) / period
        )
        linear_speed = float(np.linalg.norm(velocity))
        angular_speed = (
            0.0
            if previous_quaternion is None
            else qualification._g0.quaternion_angular_distance_rad(previous_quaternion, quaternion)
            / period
        )
        linear_acceleration = (
            0.0
            if previous_velocity is None
            else float(np.linalg.norm(velocity - previous_velocity) / period)
        )
        angular_acceleration = (
            0.0
            if previous_angular_speed is None
            else abs(angular_speed - previous_angular_speed) / period
        )
        stationary = state in {
            ExternalCameraMotionState.HOME_ANCHOR.value,
            ExternalCameraMotionState.SETTLE_AT_VIEW.value,
            ExternalCameraMotionState.COLLECT.value,
            ExternalCameraMotionState.VERIFY_HOME_AND_ARM_HOLD.value,
        }
        settle_evidence = bool(
            linear_speed <= safety["settled_linear_velocity_max_m_s"]
            and angular_speed <= safety["settled_angular_velocity_max_rad_s"]
        )
        settle_streak = settle_streak + 1 if stationary and settle_evidence else 0
        settled = settle_streak >= safety["required_consecutive_settled_ticks"]
        write_eligible = bool(state == ExternalCameraMotionState.COLLECT.value and settled)
        rgb_sha = alternate_sha if index == 47 else home_sha
        rows.append(
            {
                "version": qualification.E018_P1_G2C_QUALIFICATION_EXECUTION_VERSION,
                "episode_id": episode_id,
                "request_id": f"{episode_id}-request-00",
                "camera_command_sequence_id": f"{episode_id}-camera-sequence-00",
                "frame_index": index,
                "control_tick": index,
                "timestamp_s": index * period,
                "external_rgb_timestamp_s": index * period,
                "external_pose_timestamp_s": index * period,
                "timestamp_source": "synchronous-simulator-control-tick-derived/v1",
                "external_rgb_pose_skew_s": 0.0,
                "source_phase": qualification.QUALIFICATION_SOURCE_PHASE,
                "camera_motion_state": state,
                "viewpoint_primitive_id": viewpoint_id,
                "target_orientation_id": orientation_id,
                "orientation_progress": progress,
                "commanded_yaw_offset_rad": float(offsets[0]),
                "commanded_pitch_offset_rad": float(offsets[1]),
                "commanded_roll_offset_rad": float(offsets[2]),
                "arm_owner": "SAFE_HOLD",
                "gripper_owner": "SAFE_HOLD_OPEN",
                "external_camera_owner": qualification.QUALIFICATION_CAMERA_OWNER,
                "arm_motion_command_max_abs": 0.0,
                "gripper_hold_open_command": 1.0,
                "commanded_external_position_world_m": position.tolist(),
                "commanded_external_quaternion_sapien": quaternion.tolist(),
                "actual_external_position_world_m": position.tolist(),
                "actual_external_quaternion_sapien": quaternion.tolist(),
                "commanded_world_from_external_camera_gl": world_from_gl.tolist(),
                "actual_world_from_external_camera_gl": world_from_gl.tolist(),
                "commanded_base_from_external_camera_cv": base_from_cv.tolist(),
                "actual_base_from_external_camera_cv": base_from_cv.tolist(),
                "external_intrinsic_cv": intrinsic.tolist(),
                "external_pose_valid": True,
                "external_position_tracking_error_m": 0.0,
                "external_orientation_tracking_error_rad": 0.0,
                "external_linear_velocity_m_s": velocity.tolist(),
                "external_linear_speed_m_s": linear_speed,
                "external_linear_acceleration_m_s2": linear_acceleration,
                "external_angular_speed_rad_s": angular_speed,
                "external_angular_acceleration_rad_s2": angular_acceleration,
                "settle_evidence_passed": settle_evidence,
                "settle_streak": settle_streak,
                "settled": settled,
                "measurement_write_eligible": write_eligible,
                "memory_write_executed": False,
                "arm_anchor_q_rad": [0.0] * 7,
                "arm_current_q_rad": [0.0] * 7,
                "arm_joint_max_drift_rad": 0.0,
                "tcp_anchor_world": np.eye(4).tolist(),
                "tcp_current_world": np.eye(4).tolist(),
                "tcp_position_drift_m": 0.0,
                "tcp_orientation_drift_rad": 0.0,
                "world_from_robot_base": world_from_base.tolist(),
                "finger_joint_positions_m": [0.04, 0.04],
                "minimum_finger_joint_position_m": 0.04,
                "finger_object_contact_force_n": 0.0,
                "robot_object_contact_force_n": 0.0,
                "robot_object_contact_by_link": [
                    {
                        "link_name": "panda_link0",
                        "force_xyz_n": [0.0, 0.0, 0.0],
                        "force_magnitude_n": 0.0,
                    },
                    {
                        "link_name": "panda_hand",
                        "force_xyz_n": [0.0, 0.0, 0.0],
                        "force_magnitude_n": 0.0,
                    },
                ],
                "is_grasping": False,
                "terminated": False,
                "truncated": False,
                "rgb_sha256": rgb_sha,
                "offline_segmentation_diagnostics": None,
            }
        )
        previous_position = position
        previous_quaternion = quaternion
        previous_velocity = velocity
        previous_angular_speed = angular_speed

    gates = qualification._recompute_qualification_route_witnesses(
        rows,
        seed=seed,
        alternate_viewpoint_id=alternate_viewpoint_id,
        g0c_config=g0c_config,
        capture_safety=load_g2c_dynamic_qualification_config(CONFIG_PATH)["capture_safety"],
        rgb_images=rgb_images,
    )
    summary = {
        "version": qualification.E018_P1_G2C_QUALIFICATION_EXECUTION_VERSION,
        "episode_id": episode_id,
        "seed": seed,
        "alternate_viewpoint_id": alternate_viewpoint_id,
        "alternate_orientation_id": orientation.orientation_id,
        "yaw_offset_rad": orientation.yaw_offset_rad,
        "pitch_offset_rad": orientation.pitch_offset_rad,
        "roll_offset_rad": orientation.roll_offset_rad,
        "frame_count": 92,
        "control_hz": 20,
        "motion_ticks_each_leg": 40,
        "route_simulated_duration_s": 91 / 20,
        "passed": True,
        "status": "passed",
        "test_split_status": "prohibited-unread",
        "provider_forward_count": 2,
        "privileged_capture_count": 2,
        "memory_write_count": 0,
        "formal_claim_allowed": False,
        "qualification_classification": "preflight/no-qualification-claim",
        "offline_segmentation_diagnostics": False,
        "gates": gates,
        "diagnostics": {
            "alternate_rgb_mean_abs_difference": gates["rendered_view_changed"]["actual"],
            "return_home_rgb_mean_abs_difference": gates["return_home_render_recovered"]["actual"],
            "alternate_displacement_m": gates["actual_dynamic_pose_observed"]["actual"][
                "alternate_displacement_m"
            ],
            "requested_orientation_offset_rad": gates["alternate_orientation_target_reached"][
                "actual"
            ]["requested_offset_rad"],
            "actual_orientation_offset_rad": gates["alternate_orientation_target_reached"][
                "actual"
            ]["actual_offset_rad"],
            "alternate_target_orientation_error_rad": gates["alternate_orientation_target_reached"][
                "actual"
            ]["target_error_rad"],
            "object_visible_pixels_collect_min": None,
            "goal_visible_pixels_collect_min": None,
            "rgb_numeric_evidence_source": ("three-public-png-pixel-witnesses-recomputed/v1"),
        },
    }
    return rows, summary, rgb_images, g0c_config


def _build_complete_synthetic_smoke_execution(
    tmp_path: Path,
) -> tuple[Path, Path]:
    from PIL import Image

    config = load_g2c_dynamic_qualification_config(CONFIG_PATH)
    data_config = load_e018_p1_g2c_data_config(DATA_CONFIG_PATH)
    public = tmp_path / "execution"
    private = tmp_path / "private"
    rows, summary, rgb_images, g0c_config = _synthetic_raw_route()
    summary = {**summary, "route_index": 0}
    journal = QualificationJournal(
        public_root=public,
        private_label_root=private,
        config_sha256=config["config_sha256"],
        classification=qualification.QUALIFICATION_CLASSIFICATION_SMOKE,
    )
    qualification._atomic_create_json(public / "config_snapshot.json", config)
    g0c_path = public / "g0c_config_snapshot.json"
    g0c_raw_sha256, _ = qualification._atomic_create_json(g0c_path, g0c_config)
    source = {"git_commit": "4" * 40, "source_tree_sha256": "5" * 64}
    source["identity_sha256"] = canonical_sha256(source)
    qualification._atomic_create_json(public / "source_identity.json", source)
    parent = {
        "config_sha256": config["config_sha256"],
        "g0c_config_sha256": config["parents"]["g0c_config_sha256"],
        "g0c_receipt_internal_sha256": config["parents"]["g0c_receipt_internal_sha256"],
        "g0c_config_snapshot_raw_sha256": g0c_raw_sha256,
        "calibration_verification": {
            "verification_sha256": config["parents"]["calibration_result_verification_sha256"]
        },
        "calibration_identities": qualification._EXPECTED_CALIBRATION_IDENTITIES,
        "normalizer_identity": {
            name: config["parents"][name]
            for name in (
                "proprio_stats_sha256",
                "proprio_normalizer_sha256",
                "finger_force_stats_sha256",
                "finger_force_normalizer_sha256",
            )
        },
    }
    parent["verification_sha256"] = canonical_sha256(parent)
    qualification._atomic_create_json(public / "parent_verification.json", parent)

    image_root = public / "images"
    image_root.mkdir(mode=0o700)
    for role, value in rgb_images.items():
        Image.fromarray(value, mode="RGB").save(image_root / f"{summary['episode_id']}__{role}.png")
    _, rgb_inventory_row = qualification._load_route_rgb_witnesses(
        public, route_index=0, episode_id=str(summary["episode_id"])
    )

    def commit_prediction(*, row_index: int, route_frame_index: int, viewpoint_id: str) -> None:
        sample_index = 0 if route_frame_index == 0 else 1
        capture, raw = _prediction_capture_and_raw(
            row_index=row_index,
            sample_index=sample_index,
            viewpoint_id=viewpoint_id,
            route_frame_index=route_frame_index,
            input_sha256=str(row_index + 1) * 64,
        )
        route_row = rows[route_frame_index]
        intrinsic = np.asarray(route_row["external_intrinsic_cv"], dtype=np.float64)
        base_from_camera = np.asarray(
            route_row["actual_base_from_external_camera_cv"], dtype=np.float64
        )
        capture["external_intrinsic_cv"] = intrinsic
        capture["base_from_external_camera_cv"] = base_from_camera
        capture["deployable_safety"] = {
            "eligible_capture": True,
            "finger_force_n": [0.0, 0.0],
            "finger_force_valid": True,
            "raw_gripper_opening_ratio": 1.0,
            "arm_joint_drift_rad": route_row["arm_joint_max_drift_rad"],
            "tcp_position_drift_m": route_row["tcp_position_drift_m"],
            "tcp_orientation_drift_rad": route_row["tcp_orientation_drift_rad"],
            "rgb_timestamp_s": route_row["external_rgb_timestamp_s"],
            "pose_timestamp_s": route_row["external_pose_timestamp_s"],
            "camera_position_tracking_error_m": route_row["external_position_tracking_error_m"],
            "camera_orientation_tracking_error_rad": route_row[
                "external_orientation_tracking_error_rad"
            ],
            "rotation_projection_error_frobenius": 0.0,
        }
        raw.update(
            qualification._recompute_prediction_geometry(
                normalized_uv=raw["predicted_object_normalized_uv"],
                intrinsic_cv=intrinsic,
                base_from_camera_cv=base_from_camera,
                sigma_xy_px=raw["object_sigma_xy_px"],
                plane_base_z_m=config["capture_safety"]["object_center_base_z_m"],
            )
        )
        prediction = finalize_qualification_prediction(
            raw,
            capture=capture,
            calibration=config["calibration"]["values"][viewpoint_id],
            data_config=data_config,
            qualification_config=config,
            classification=qualification.QUALIFICATION_CLASSIFICATION_SMOKE,
        )
        journal.commit_prediction_then_capture_label(
            prediction, privileged_getter=_minimal_privileged_label
        )

    commit_prediction(
        row_index=0,
        route_frame_index=0,
        viewpoint_id=qualification.FRONT_HOME_ID,
    )
    commit_prediction(
        row_index=1,
        route_frame_index=47,
        viewpoint_id="LEFT_LOW__CENTER",
    )

    motion_writer = qualification._AppendOnlyJsonl(public / "camera_pose_ledger.jsonl")
    motion_writer.append(rows)
    summary_writer = qualification._AppendOnlyJsonl(public / "route_summaries.jsonl")
    summary_writer.append([summary])
    predictions, commits = qualification._load_public_prediction_chain(journal)
    prediction_writer = qualification._AppendOnlyJsonl(public / "prediction_ledger.jsonl")
    prediction_writer.append(predictions)
    commit_writer = qualification._AppendOnlyJsonl(public / "prediction_commit_ledger.jsonl")
    commit_writer.append(commits)
    journal.freeze_route_completion()
    counters = qualification.validate_qualification_route_rows(
        rows,
        seed=76801,
        alternate_index=0,
        alternate_viewpoint_id="LEFT_LOW__CENTER",
        summary=summary,
        g0c_config=g0c_config,
        capture_safety=config["capture_safety"],
        rgb_images=rgb_images,
    )
    output_identities = qualification._output_root_identities(
        public_root=public, private_root=private
    )
    execution_freeze = {
        "version": qualification.E018_P1_G2C_QUALIFICATION_EXECUTION_VERSION,
        "status": "routes-complete-context-destroy-pending",
        "classification": qualification.QUALIFICATION_CLASSIFICATION_SMOKE,
        "config_sha256": config["config_sha256"],
        "source_identity_sha256": source["identity_sha256"],
        "parent_verification_sha256": parent["verification_sha256"],
        "g0c_config_snapshot": {
            "canonical_sha256": canonical_sha256(g0c_config),
            "raw_sha256": g0c_raw_sha256,
            "size_bytes": g0c_path.stat().st_size,
        },
        "route_count": 1,
        "seed_count": 1,
        "alternate_count": 1,
        "seeds": [76801],
        "alternate_order": ["LEFT_LOW__CENTER"],
        "counters": counters,
        "motion_ledger": motion_writer.freeze(),
        "route_summaries": summary_writer.freeze(),
        "prediction_ledger": prediction_writer.freeze(),
        "prediction_commit_ledger": commit_writer.freeze(),
        "route_rgb_inventory": qualification._freeze_ordered_inventory(
            [rgb_inventory_row], version=qualification._ROUTE_RGB_INVENTORY_VERSION
        ),
        "private_label_journal_inventory": journal.private_label_inventory(),
        "output_identities": output_identities,
        "budget_limits": {
            "wall_seconds_max": 900.0,
            "gpu_seconds_max": 900.0,
            "combined_artifact_bytes_max": 1_073_741_824,
        },
        "formal_execution_decision": {"status": "not-applicable-preflight-no-qualification-claim"},
        "last_prediction_sha256": journal.previous_prediction_sha256,
        "permission_counters": {
            **config["permissions"],
            "simulator_isolated_camera_pose_set_count": 97,
            "safe_hold_open_step_count": 96,
            "object_contact_event_count": 0,
            "privileged_object_label_capture_count": 2,
            "goal_gt_read_count": 0,
        },
        "test_split_status": "prohibited-unread",
        "checkpoint_write_count": 0,
        "scoring_started": False,
    }
    execution_freeze["freeze_sha256"] = canonical_sha256(execution_freeze)
    qualification._atomic_create_json(public / "execution_freeze.json", execution_freeze)

    class Env:
        def close(self) -> None:
            return None

    class Provider:
        def destroy(self) -> None:
            return None

    qualification._finalize_qualification_execution(
        env=Env(),
        provider=Provider(),
        journal=journal,
        public_root=public,
        private_root=private,
        combined_artifact_bytes_max=1_073_741_824,
        execution_freeze=execution_freeze,
        environment_identity={"synthetic": True},
        capture_started_at_unix_ns=qualification.time.time_ns() - 1_000_000,
        capture_started_monotonic_s=qualification.time.monotonic() - 0.001,
        wall_seconds_max=900.0,
        gpu_seconds_max=900.0,
    )
    return public, private


def _resign_execution_after_motion_tamper(public: Path, rows: list[dict[str, object]]) -> None:
    motion_path = public / "camera_pose_ledger.jsonl"
    motion_path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    freeze_path = public / "execution_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["motion_ledger"] = {
        "row_count": len(rows),
        "raw_sha256": qualification.file_sha256(motion_path),
        "size_bytes": motion_path.stat().st_size,
    }
    freeze["freeze_sha256"] = canonical_sha256(
        {key: value for key, value in freeze.items() if key != "freeze_sha256"}
    )
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    receipt_path = public / "execution_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["motion_ledger"] = freeze["motion_ledger"]
    receipt["freeze_sha256"] = freeze["freeze_sha256"]
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _resign_execution_after_summary_tamper(
    public: Path, summaries: list[dict[str, object]]
) -> None:
    summary_path = public / "route_summaries.jsonl"
    summary_path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in summaries),
        encoding="utf-8",
    )
    freeze_path = public / "execution_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["route_summaries"] = {
        "row_count": len(summaries),
        "raw_sha256": qualification.file_sha256(summary_path),
        "size_bytes": summary_path.stat().st_size,
    }
    freeze["freeze_sha256"] = canonical_sha256(
        {key: value for key, value in freeze.items() if key != "freeze_sha256"}
    )
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    receipt_path = public / "execution_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["route_summaries"] = freeze["route_summaries"]
    receipt["freeze_sha256"] = freeze["freeze_sha256"]
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_config_binds_formal_seed_view_count_and_calibration() -> None:
    config = load_g2c_dynamic_qualification_config(CONFIG_PATH)

    assert (
        tuple(
            range(
                config["route"]["seed_start"],
                config["route"]["seed_end"] + 1,
            )
        )
        == FORMAL_QUALIFICATION_SEEDS
    )
    assert tuple(config["calibration"]["values"]) == QUALIFICATION_VIEW_ORDER
    assert config["route"]["totals"]["provider_scored_frame_count"] == 550
    assert config["route"]["per_route"]["ledger_frame_count"] == 92


def test_g0c_parent_receipt_recomputes_internal_hash_before_trusting_fields() -> None:
    receipt = {
        "version": qualification._g0c.E018_P1_G0C_RESULT_VERSION,
        "status": "complete-development-only",
        "gate_passed": True,
        "config_sha256": "a" * 64,
        "test_split_status": "prohibited-unread",
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    qualification._verify_g0c_parent_receipt(
        receipt,
        expected_config_sha256="a" * 64,
        expected_internal_sha256=receipt["receipt_sha256"],
    )

    original = receipt["receipt_sha256"]
    receipt["gate_passed"] = False
    with pytest.raises(RuntimeError, match="G0C receipt parent"):
        qualification._verify_g0c_parent_receipt(
            receipt,
            expected_config_sha256="a" * 64,
            expected_internal_sha256=original,
        )
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    with pytest.raises(RuntimeError, match="G0C receipt parent"):
        qualification._verify_g0c_parent_receipt(
            receipt,
            expected_config_sha256="a" * 64,
            expected_internal_sha256=original,
        )


def test_d048_receipt_binds_three_segment_conservative_budget_and_resigning(
    tmp_path: Path,
) -> None:
    config = load_g2c_dynamic_qualification_config(CONFIG_PATH)
    receipt = _formal_decision_receipt()
    path = tmp_path / "d048.json"
    raw_sha256, _ = qualification._atomic_create_json(path, receipt)

    verification = qualification.verify_g2c_formal_execution_decision_receipt(
        decision_receipt_path=path,
        expected_raw_sha256=raw_sha256,
        expected_internal_sha256=receipt["receipt_sha256"],
        qualification_config=config,
        qualification_config_raw_sha256=qualification.file_sha256(CONFIG_PATH),
        expected_source_git_commit="4" * 40,
        expected_source_identity_sha256="5" * 64,
    )

    assert verification["verified"] is True
    gpu = receipt["budgets"]["gpu"]
    artifact = receipt["budgets"]["artifact"]
    assert gpu["pre_formal_conservative_upper_bound_seconds"] == (
        gpu["pre_smoke_conservative_upper_bound_seconds"] + gpu["d047_smoke_actual_seconds"]
    )
    assert artifact["projected_conservative_upper_bound_bytes"] == (
        artifact["pre_smoke_conservative_upper_bound_bytes"]
        + artifact["d047_smoke_actual_bytes"]
        + artifact["formal_combined_artifact_bytes_reserved_max"]
    )

    tampered = _formal_decision_receipt()
    tampered["budgets"]["gpu"]["pre_formal_conservative_upper_bound_seconds"] += 1.0
    tampered["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    )
    with pytest.raises(RuntimeError, match="formal decision receipt"):
        qualification._validate_g2c_formal_execution_decision_receipt(
            tampered,
            config=config,
            qualification_config_raw_sha256=qualification.file_sha256(CONFIG_PATH),
            expected_source_git_commit="4" * 40,
            expected_source_identity_sha256="5" * 64,
        )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda value: value["calibration"]["values"]["HOME__CENTER"].__setitem__(
                "write_threshold", 0.6
            ),
            "calibration",
        ),
        (
            lambda value: value["qualification"].__setitem__("maximum_unsafe_accepted_count", 1),
            "metric",
        ),
        (
            lambda value: value["permissions"].__setitem__("memory_reads", 1),
            "permission",
        ),
    ],
)
def test_config_rejects_rehashed_policy_drift(tmp_path: Path, mutate: object, match: str) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(config)
    path = tmp_path / "drift.json"
    _rewrite_config(path, config)

    with pytest.raises(RuntimeError, match=match):
        load_g2c_dynamic_qualification_config(path)


def test_route_png_reopen_requires_native_rgb_uint8_shape(tmp_path: Path) -> None:
    from PIL import Image

    episode_id = "synthetic-route"
    image_root = tmp_path / "images"
    image_root.mkdir()
    for role in ("home_before", "alternate", "home_after"):
        Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8), mode="RGB").save(
            image_root / f"{episode_id}__{role}.png"
        )

    images, inventory = qualification._load_route_rgb_witnesses(
        tmp_path, route_index=0, episode_id=episode_id
    )
    assert all(value.shape == (128, 128, 3) for value in images.values())
    assert all(value.dtype == np.uint8 for value in images.values())
    assert inventory["route_index"] == 0

    Image.fromarray(np.zeros((128, 128), dtype=np.uint8), mode="L").save(
        image_root / f"{episode_id}__alternate.png"
    )
    with pytest.raises(RuntimeError, match="原生 8-bit RGB PNG"):
        qualification._load_route_rgb_witnesses(tmp_path, route_index=0, episode_id=episode_id)


@pytest.mark.parametrize("tamper_kind", ["jpeg", "symlink", "hardlink"])
def test_route_rgb_witness_rejects_format_and_link_aliases(
    tmp_path: Path, tamper_kind: str
) -> None:
    from PIL import Image

    episode_id = "synthetic-route"
    image_root = tmp_path / "images"
    image_root.mkdir()
    paths = {
        role: image_root / f"{episode_id}__{role}.png"
        for role in ("home_before", "alternate", "home_after")
    }
    image = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8), mode="RGB")
    for path in paths.values():
        image.save(path)
    alternate = paths["alternate"]
    alternate.unlink()
    if tamper_kind == "jpeg":
        image.save(alternate, format="JPEG")
        match = "原生 8-bit RGB PNG"
    elif tamper_kind == "symlink":
        alternate.symlink_to(paths["home_before"])
        match = "file/link"
    else:
        alternate.hardlink_to(paths["home_before"])
        match = "file/link"

    with pytest.raises(RuntimeError, match=match):
        qualification._load_route_rgb_witnesses(tmp_path, route_index=0, episode_id=episode_id)


def test_complete_92_row_raw_route_recomputes_look_at_mechanics_and_contact() -> None:
    rows, summary, rgb_images, g0c_config = _synthetic_raw_route()
    config = load_g2c_dynamic_qualification_config(CONFIG_PATH)

    counters = qualification.validate_qualification_route_rows(
        rows,
        seed=76801,
        alternate_index=0,
        alternate_viewpoint_id="LEFT_LOW__CENTER",
        summary=summary,
        g0c_config=g0c_config,
        capture_safety=config["capture_safety"],
        rgb_images=rgb_images,
    )

    assert counters["ledger_frame_count"] == 92
    assert counters["provider_scored_frame_count"] == 2
    assert all(row["robot_object_contact_force_n"] == 0.0 for row in rows)

    rows[47]["robot_object_contact_by_link"][0]["force_xyz_n"] = [0.02, 0.0, 0.0]
    rows[47]["robot_object_contact_by_link"][0]["force_magnitude_n"] = 0.02
    rows[47]["robot_object_contact_force_n"] = 0.02
    with pytest.raises(RuntimeError, match="contact safety gate"):
        qualification.validate_qualification_route_rows(
            rows,
            seed=76801,
            alternate_index=0,
            alternate_viewpoint_id="LEFT_LOW__CENTER",
            summary=summary,
            g0c_config=g0c_config,
            capture_safety=config["capture_safety"],
            rgb_images=rgb_images,
        )


def test_complete_smoke_execution_tree_scores_and_verifies_without_private_reopen(
    tmp_path: Path,
) -> None:
    public, private = _build_complete_synthetic_smoke_execution(tmp_path)
    execution = qualification.verify_g2c_qualification_execution(
        qualification_config_path=CONFIG_PATH,
        public_execution_root=public,
    )
    result = tmp_path / "result"
    receipt = score_e018_p1_g2c_qualification(
        qualification_config_path=CONFIG_PATH,
        public_execution_root=public,
        private_label_root=private,
        result_output_root=result,
        decision_scoring_go=True,
    )
    verification = verify_g2c_qualification_result(
        qualification_config_path=CONFIG_PATH,
        public_execution_root=public,
        result_root=result,
    )
    combined = qualification.verify_g2c_qualification_combined_artifacts(
        qualification_config_path=CONFIG_PATH,
        public_execution_root=public,
        private_label_root=private,
        result_root=result,
    )

    assert execution["route_count"] == 1
    assert execution["prediction_count"] == 2
    assert receipt["status"] == "complete-preflight-no-qualification-claim"
    assert verification["verified"] is True
    assert combined["verified"] is True
    assert combined["label_content_reopen_count"] == 0


@pytest.mark.parametrize(
    "tamper_kind",
    [
        "command_position",
        "actual_quaternion",
        "world_from_gl",
        "base_from_cv",
        "velocity",
        "acceleration",
        "settle_streak",
        "arm_state",
        "tcp_state",
        "full_robot_contact",
    ],
)
def test_complete_execution_tree_rejects_resigned_motion_safety_tamper(
    tmp_path: Path, tamper_kind: str
) -> None:
    public, _ = _build_complete_synthetic_smoke_execution(tmp_path)
    motion_path = public / "camera_pose_ledger.jsonl"
    rows = [json.loads(line) for line in motion_path.read_text(encoding="utf-8").splitlines()]
    row = rows[10]
    if tamper_kind == "command_position":
        row["commanded_external_position_world_m"][0] += 0.01
    elif tamper_kind == "actual_quaternion":
        row["actual_external_quaternion_sapien"][0] += 0.01
    elif tamper_kind == "world_from_gl":
        row["actual_world_from_external_camera_gl"][0][3] += 0.01
    elif tamper_kind == "base_from_cv":
        row["actual_base_from_external_camera_cv"][0][3] += 0.01
    elif tamper_kind == "velocity":
        row["external_linear_velocity_m_s"][0] += 0.01
    elif tamper_kind == "acceleration":
        row["external_linear_acceleration_m_s2"] += 0.01
    elif tamper_kind == "settle_streak":
        rows[45]["settle_streak"] += 1
    elif tamper_kind == "arm_state":
        row["arm_current_q_rad"][0] += 0.01
    elif tamper_kind == "tcp_state":
        row["tcp_current_world"][0][3] += 0.01
    elif tamper_kind == "full_robot_contact":
        row["robot_object_contact_by_link"][0]["force_xyz_n"] = [0.02, 0.0, 0.0]
        row["robot_object_contact_by_link"][0]["force_magnitude_n"] = 0.02
        row["robot_object_contact_force_n"] = 0.02
    else:  # pragma: no cover - parametrize 已冻结 choices
        raise AssertionError(tamper_kind)
    _resign_execution_after_motion_tamper(public, rows)

    with pytest.raises(RuntimeError, match="qualification"):
        qualification.verify_g2c_qualification_execution(
            qualification_config_path=CONFIG_PATH,
            public_execution_root=public,
        )


@pytest.mark.parametrize("tamper_kind", ["extra", "symlink", "hardlink"])
def test_complete_execution_verifier_rejects_extra_and_link_files(
    tmp_path: Path, tamper_kind: str
) -> None:
    public, _ = _build_complete_synthetic_smoke_execution(tmp_path)
    unexpected = public / "unexpected.bin"
    if tamper_kind == "extra":
        unexpected.write_bytes(b"extra")
    elif tamper_kind == "symlink":
        unexpected.symlink_to(public / "phase_state.json")
    else:
        unexpected.hardlink_to(public / "phase_state.json")

    with pytest.raises(RuntimeError, match="qualification"):
        qualification.verify_g2c_qualification_execution(
            qualification_config_path=CONFIG_PATH,
            public_execution_root=public,
        )


def test_complete_execution_rejects_resigned_route_summary_gate_tamper(
    tmp_path: Path,
) -> None:
    public, _ = _build_complete_synthetic_smoke_execution(tmp_path)
    summary_path = public / "route_summaries.jsonl"
    summaries = [json.loads(line) for line in summary_path.read_text(encoding="utf-8").splitlines()]
    summaries[0]["gates"]["rendered_view_changed"]["passed"] = False
    _resign_execution_after_summary_tamper(public, summaries)

    with pytest.raises(RuntimeError, match="qualification"):
        qualification.verify_g2c_qualification_execution(
            qualification_config_path=CONFIG_PATH,
            public_execution_root=public,
        )


def test_complete_execution_rejects_resigned_timing_budget_tamper(
    tmp_path: Path,
) -> None:
    public, _ = _build_complete_synthetic_smoke_execution(tmp_path)
    receipt_path = public / "execution_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["wall_elapsed_seconds"] = 901.0
    receipt["gpu_elapsed_seconds"] = 901.0
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="execution receipt"):
        qualification.verify_g2c_qualification_execution(
            qualification_config_path=CONFIG_PATH,
            public_execution_root=public,
        )


def test_combined_artifact_budget_is_not_applied_per_tree_independently(
    tmp_path: Path,
) -> None:
    result = tmp_path / "result"
    result.mkdir()
    for name in (
        "scoring_ledger.jsonl",
        "viewpoint_summaries.json",
        "qualification_summary.json",
        "qualification_receipt.json",
    ):
        (result / name).write_bytes(b"x")
    (result / "scoring_state.json").write_bytes(b"in-progress")

    with pytest.raises(RuntimeError, match="combined artifact byte budget"):
        qualification._finalize_combined_artifact_accounting(
            result_root=result,
            classification=qualification.QUALIFICATION_CLASSIFICATION_SMOKE,
            config_sha256="a" * 64,
            output_identities={
                "public_output_identity_sha256": "b" * 64,
                "private_output_identity_sha256": "c" * 64,
                "result_output_identity_sha256": "d" * 64,
            },
            public_execution_bytes=600,
            private_label_commit_bytes=400,
            private_scoring_consumption_marker_bytes=100,
            private_label_total_bytes=500,
            combined_budget_limit_bytes=1_100,
            final_scoring_state={"status": "complete"},
        )
    assert not (result / "artifact_accounting.json").exists()


def test_scored_frame_selector_is_home_first_only_and_final_collect_only() -> None:
    alternate = QUALIFICATION_VIEW_ORDER[1]
    first = [
        qualification_scored_frame_identity(
            _hook_row(index, alternate=alternate),
            seed=FORMAL_QUALIFICATION_SEEDS[0],
            alternate_index=0,
            alternate_viewpoint_id=alternate,
        )
        for index in range(92)
    ]
    later = [
        qualification_scored_frame_identity(
            _hook_row(index, alternate=QUALIFICATION_VIEW_ORDER[2]),
            seed=FORMAL_QUALIFICATION_SEEDS[0],
            alternate_index=1,
            alternate_viewpoint_id=QUALIFICATION_VIEW_ORDER[2],
        )
        for index in range(92)
    ]

    assert [(item["sample_index"], item["route_frame_index"]) for item in first if item] == [
        (0, 0),
        (1, 47),
    ]
    assert [(item["sample_index"], item["route_frame_index"]) for item in later if item] == [
        (2, 47)
    ]


def test_journal_persists_prediction_and_state_before_gt(tmp_path: Path) -> None:
    public = tmp_path / "public"
    private = tmp_path / "private"
    journal = QualificationJournal(
        public_root=public,
        private_label_root=private,
        config_sha256="a" * 64,
        classification="preflight/no-qualification-claim",
    )
    observed: dict[str, object] = {}

    def privileged_getter() -> dict[str, object]:
        prediction_path = public / "prediction_commits/000000.json"
        receipt_path = public / "prediction_commits/000000.commit.json"
        state = json.loads((public / "phase_state.json").read_text(encoding="utf-8"))
        observed.update(
            {
                "prediction_exists": prediction_path.is_file(),
                "receipt_exists": receipt_path.is_file(),
                "state": state,
            }
        )
        return _minimal_privileged_label()

    prediction, label = journal.commit_prediction_then_capture_label(
        {"row_index": 0, "viewpoint_id": "HOME__CENTER"},
        privileged_getter=privileged_getter,
    )

    assert observed["prediction_exists"] is True
    assert observed["receipt_exists"] is True
    assert observed["state"]["label_array_consumed"] is True
    assert observed["state"]["privileged_access_started_count"] == 1
    assert label["prediction_fsync_completed_at_unix_ns"] < label["privileged_captured_at_unix_ns"]
    assert prediction["previous_prediction_sha256"] is None
    assert journal.prediction_count == journal.privileged_capture_count == 1


def test_journal_failed_getter_is_still_consumed(tmp_path: Path) -> None:
    journal = QualificationJournal(
        public_root=tmp_path / "public",
        private_label_root=tmp_path / "private",
        config_sha256="b" * 64,
        classification="preflight/no-qualification-claim",
    )

    def fail_after_read() -> dict[str, object]:
        raise RuntimeError("synthetic GT read failure")

    with pytest.raises(RuntimeError, match="GT read failure"):
        journal.commit_prediction_then_capture_label(
            {"row_index": 0, "viewpoint_id": "HOME__CENTER"},
            privileged_getter=fail_after_read,
        )
    failure = journal.freeze_consumed_failure(RuntimeError("capture failed"))

    assert failure["prediction_commit_count"] == 1
    assert failure["privileged_access_started_count"] == 1
    assert failure["privileged_capture_count"] == 0
    assert failure["label_array_consumed"] is True


def test_prediction_sentinel_rejects_nested_privileged_field() -> None:
    with pytest.raises(ValueError, match="privileged"):
        assert_qualification_prediction_deployable_only(
            {"row_index": 0, "nested": {"world_xyz_error_m": 0.0}}
        )


def test_deployable_capture_interface_cannot_receive_observation_or_segmentation() -> None:
    parameters = set(inspect.signature(build_qualification_deployable_capture).parameters)

    assert "observation" not in parameters
    assert "segmentation" not in parameters


def test_finalize_prediction_applies_d046_scale_and_threshold() -> None:
    qualification_config = load_g2c_dynamic_qualification_config(CONFIG_PATH)
    data_config = load_e018_p1_g2c_data_config(DATA_CONFIG_PATH)
    viewpoint_id = "RIGHT_LOW__CENTER"
    capture, raw = _prediction_capture_and_raw(
        sample_index=6,
        viewpoint_id=viewpoint_id,
        route_alternate_index=5,
        input_sha256="c" * 64,
    )

    result = finalize_qualification_prediction(
        raw,
        capture=capture,
        calibration=qualification_config["calibration"]["values"][viewpoint_id],
        data_config=data_config,
        qualification_config=qualification_config,
        classification="preflight/no-qualification-claim",
    )

    scale = qualification_config["calibration"]["values"][viewpoint_id]["scale_factor"]
    assert np.allclose(
        result["calibrated_covariance_base_m2"],
        np.asarray(raw["raw_covariance_base_m2"]) * scale,
        rtol=0.0,
        atol=0.0,
    )
    assert (
        result["write_threshold"]
        == qualification_config["calibration"]["values"][viewpoint_id]["write_threshold"]
    )
    assert result["write_accepted"] is True


def test_finalize_prediction_recomputes_write_score_bool() -> None:
    qualification_config = load_g2c_dynamic_qualification_config(CONFIG_PATH)
    data_config = load_e018_p1_g2c_data_config(DATA_CONFIG_PATH)
    capture, raw = _prediction_capture_and_raw(input_sha256="d" * 64)
    raw["predicted_observable"] = False
    raw["write_score"] = 0.9

    with pytest.raises(RuntimeError, match="派生字段"):
        finalize_qualification_prediction(
            raw,
            capture=capture,
            calibration=qualification_config["calibration"]["values"]["LEFT_LOW__CENTER"],
            data_config=data_config,
            qualification_config=qualification_config,
            classification="preflight/no-qualification-claim",
        )


def test_formal_hold_fails_before_any_config_or_environment_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("formal HOLD 后不得进入 capture runner")

    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_qualification._run_qualification_capture",
        unexpected,
    )
    with pytest.raises(PermissionError, match="HOLD"):
        run_e018_p1_g2c_qualification_capture(
            qualification_config_path="missing",
            g0c_config_path="missing",
            g0c_receipt_path="missing",
            calibration_config_path="missing",
            calibration_prediction_freeze_root="missing",
            calibration_result_root="missing",
            data_config_path="missing",
            stats_root="missing",
            selected_checkpoint_path="missing",
            repository_root="missing",
            public_output_root="missing",
            private_label_output_root="missing",
            expected_source_git_commit="0" * 40,
            expected_source_identity_sha256="0" * 64,
            decision_execution_go=False,
        )


def test_formal_go_without_exact_d048_receipt_fails_before_config_read() -> None:
    with pytest.raises(PermissionError, match="D048"):
        run_e018_p1_g2c_qualification_capture(
            qualification_config_path="missing",
            g0c_config_path="missing",
            g0c_receipt_path="missing",
            calibration_config_path="missing",
            calibration_prediction_freeze_root="missing",
            calibration_result_root="missing",
            data_config_path="missing",
            stats_root="missing",
            selected_checkpoint_path="missing",
            repository_root="missing",
            public_output_root="missing",
            private_label_output_root="missing",
            expected_source_git_commit="0" * 40,
            expected_source_identity_sha256="0" * 64,
            decision_execution_go=True,
        )


def test_qualification_cli_help_and_verify_config() -> None:
    module = "robot_vla.cli.run_e018_p1_g2c_qualification"
    help_result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "verify-failure" in help_result.stdout
    assert "verify-artifacts" in help_result.stdout

    capture_help = subprocess.run(
        [sys.executable, "-m", module, "capture", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert capture_help.returncode == 0
    assert "--decision-receipt" in capture_help.stdout
    assert "--expected-decision-receipt-raw-sha256" in capture_help.stdout
    assert "--expected-decision-receipt-internal-sha256" in capture_help.stdout

    verify = subprocess.run(
        [sys.executable, "-m", module, "verify-config", "--config", str(CONFIG_PATH)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert verify.returncode == 0, verify.stderr
    assert json.loads(verify.stdout)["verified"] is True


@pytest.mark.parametrize("seed", [76701, 999999])
def test_smoke_rejects_formal_or_unregistered_seed_before_runner(
    monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    def unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("非法 smoke 不得进入 capture runner")

    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_qualification._run_qualification_capture",
        unexpected,
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        run_e018_p1_g2c_qualification_smoke(
            qualification_config_path="missing",
            g0c_config_path="missing",
            g0c_receipt_path="missing",
            calibration_config_path="missing",
            calibration_prediction_freeze_root="missing",
            calibration_result_root="missing",
            data_config_path=DATA_CONFIG_PATH if seed == 999999 else "missing",
            stats_root="missing",
            selected_checkpoint_path="missing",
            repository_root="missing",
            public_output_root="missing",
            private_label_output_root="missing",
            seed=seed,
        )


def test_hook_never_calls_provider_or_gt_on_other_90_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = QualificationJournal(
        public_root=tmp_path / "public",
        private_label_root=tmp_path / "private",
        config_sha256="e" * 64,
        classification="preflight/no-qualification-claim",
    )
    provider_calls: list[int] = []
    gt_calls: list[int] = []

    class Provider:
        def predict(self, capture: dict[str, object]) -> dict[str, object]:
            row_index = int(capture["identity"]["row_index"])
            provider_calls.append(row_index)
            return {"row_index": row_index, "viewpoint_id": capture["identity"]["viewpoint_id"]}

    def fake_capture(**kwargs: object) -> dict[str, object]:
        return {"identity": kwargs["identity"]}

    def fake_label(**kwargs: object) -> dict[str, object]:
        prediction = kwargs["prediction"]
        row_index = int(prediction["row_index"])
        assert (tmp_path / "public/prediction_commits" / f"{row_index:06d}.commit.json").is_file()
        gt_calls.append(row_index)
        return _minimal_privileged_label()

    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_qualification.build_qualification_deployable_capture",
        fake_capture,
    )
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_qualification.capture_qualification_object_label",
        fake_label,
    )
    poison_observation = _PoisonMapping()
    for index in range(92):
        process_qualification_hook_frame(
            motion_row=_hook_row(index, alternate="LEFT_LOW__CENTER"),
            rgb=np.zeros((1,), dtype=np.uint8),
            observation=poison_observation,
            seed=76801,
            alternate_index=0,
            alternate_viewpoint_id="LEFT_LOW__CENTER",
            base_env=object(),
            spec=object(),
            proprio_normalizer=object(),
            finger_force_normalizer=object(),
            data_config={},
            provider=Provider(),
            journal=journal,
        )

    assert provider_calls == [0, 1]
    assert gt_calls == [0, 1]
    assert journal.prediction_count == journal.privileged_capture_count == 2


def _scoring_prediction() -> tuple[dict[str, object], dict[str, object]]:
    config = load_g2c_dynamic_qualification_config(CONFIG_PATH)
    viewpoint_id = "LEFT_LOW__YAW_LEFT"
    capture, raw = _prediction_capture_and_raw(
        sample_index=2,
        viewpoint_id=viewpoint_id,
        route_alternate_index=1,
    )
    prediction = finalize_qualification_prediction(
        raw,
        capture=capture,
        calibration=config["calibration"]["values"][viewpoint_id],
        data_config=load_e018_p1_g2c_data_config(DATA_CONFIG_PATH),
        qualification_config=config,
        classification="preflight/no-qualification-claim",
    )
    prediction.update(
        {
            "previous_prediction_sha256": None,
            "prediction_write_started_at_unix_ns": 1,
            "prediction_sha256": "1" * 64,
        }
    )
    label = {
        **_minimal_privileged_label(),
        "label_sha256": "2" * 64,
        "prediction_fsync_completed_at_unix_ns": 2,
        "privileged_captured_at_unix_ns": 3,
    }
    return prediction, label


def test_stable_type7_p90_is_explicit_linear_interpolation() -> None:
    assert _type7_quantile(list(range(10)), numerator=9, denominator=10) == 8.1


def test_score_row_keeps_safe_rejection_in_coverage_denominator() -> None:
    config = load_g2c_dynamic_qualification_config(CONFIG_PATH)
    prediction, label = _scoring_prediction()
    accepted = score_qualification_prediction(prediction, label, config=config)
    rejected = dict(accepted)
    rejected["write_accepted"] = False
    rows = [accepted] * 49 + [rejected]

    summary = summarize_qualification_viewpoint(
        rows, viewpoint_id=prediction["viewpoint_id"], config=config
    )

    assert accepted["oracle_safe_measurement"] is True
    assert accepted["unsafe_accepted"] is False
    assert summary["oracle_safe_count"] == 50
    assert summary["accepted_and_oracle_safe_count"] == 49
    assert summary["accepted_safe_coverage"] == 49 / 50


def test_prediction_bool_is_exact_and_numeric_tamper_exceeds_tolerance() -> None:
    config = load_g2c_dynamic_qualification_config(CONFIG_PATH)
    prediction, _ = _scoring_prediction()
    prediction["write_accepted"] = False
    with pytest.raises(RuntimeError, match="write_accepted"):
        validate_qualification_prediction_mechanics(prediction, config=config)

    prediction, _ = _scoring_prediction()
    prediction["write_threshold"] += 1e-12
    with pytest.raises(RuntimeError, match="write_threshold"):
        validate_qualification_prediction_mechanics(prediction, config=config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_sha256", "0" * 64),
        ("checkpoint_parameter_sha256", "0" * 64),
        ("memory_write_allowed", True),
        ("memory_write_executed", True),
        ("actuation_allowed", True),
        ("test_data_read", True),
    ],
)
def test_prediction_rejects_checkpoint_or_permission_identity_tamper(
    field: str, value: object
) -> None:
    config = load_g2c_dynamic_qualification_config(CONFIG_PATH)
    prediction, _ = _scoring_prediction()
    prediction[field] = value

    with pytest.raises(RuntimeError, match="identity|schema"):
        validate_qualification_prediction_mechanics(prediction, config=config)


@pytest.mark.parametrize(
    "field,mutate,match",
    [
        (
            "base_from_external_camera_cv",
            lambda value: value.__setitem__(0, [1.0, 0.0, 0.0, 0.01]),
            "geometry/covariance 同源",
        ),
        (
            "predicted_object_position_base_m",
            lambda value: value.__setitem__(0, value[0] + 0.001),
            "geometry/covariance 同源",
        ),
        (
            "raw_covariance_base_m2",
            lambda value: value[0].__setitem__(0, value[0][0] + 1e-6),
            "geometry/covariance 同源",
        ),
    ],
)
def test_prediction_camera_geometry_tamper_is_recomputed_from_public_primitive(
    field: str, mutate: object, match: str
) -> None:
    config = load_g2c_dynamic_qualification_config(CONFIG_PATH)
    prediction, _ = _scoring_prediction()
    mutate(prediction[field])

    with pytest.raises(RuntimeError, match=match):
        validate_qualification_prediction_mechanics(prediction, config=config)


def test_private_label_getter_rejects_resigned_hidden_goal_gt(tmp_path: Path) -> None:
    journal = QualificationJournal(
        public_root=tmp_path / "public",
        private_label_root=tmp_path / "private",
        config_sha256="a" * 64,
        classification="preflight/no-qualification-claim",
    )
    hidden = {**_minimal_privileged_label(), "goal_position_base_m": [0, 0, 0]}

    with pytest.raises(ValueError, match="extra=.*goal_position_base_m"):
        journal.commit_prediction_then_capture_label(
            {"row_index": 0}, privileged_getter=lambda: hidden
        )

    state = json.loads((tmp_path / "public/phase_state.json").read_text(encoding="utf-8"))
    assert state["privileged_access_started_count"] == 1
    assert not (tmp_path / "private/label_commits/000000.json").exists()


def test_scorer_rejects_hidden_private_field_after_full_resigning(
    tmp_path: Path,
) -> None:
    public, private = _build_complete_synthetic_smoke_execution(tmp_path)
    label_path = private / "label_commits/000000.json"
    label = json.loads(label_path.read_text(encoding="utf-8"))
    label["goal_position_base_m"] = [0.0, 0.0, 0.02]
    label["label_sha256"] = canonical_sha256(
        {key: value for key, value in label.items() if key != "label_sha256"}
    )
    label_path.write_text(
        json.dumps(label, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    freeze_path = public / "execution_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    inventory = freeze["private_label_journal_inventory"]
    row = inventory["rows"][0]
    row.update(
        {
            "label_raw_sha256": qualification.file_sha256(label_path),
            "label_internal_sha256": label["label_sha256"],
            "size_bytes": label_path.stat().st_size,
            "public_scoring_primitive_sha256": canonical_sha256(
                qualification._qualification_public_scoring_primitive(label)
            ),
        }
    )
    row["row_sha256"] = canonical_sha256(
        {key: value for key, value in row.items() if key != "row_sha256"}
    )
    inventory["inventory_sha256"] = canonical_sha256(
        {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    )
    freeze["freeze_sha256"] = canonical_sha256(
        {key: value for key, value in freeze.items() if key != "freeze_sha256"}
    )
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    receipt_path = public / "execution_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["private_label_journal_inventory"] = inventory
    receipt["freeze_sha256"] = freeze["freeze_sha256"]
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="goal_position_base_m"):
        score_e018_p1_g2c_qualification(
            qualification_config_path=CONFIG_PATH,
            public_execution_root=public,
            private_label_root=private,
            result_output_root=tmp_path / "result",
            decision_scoring_go=True,
        )


def _route_row_for_prediction(prediction: dict[str, object]) -> dict[str, object]:
    safety = prediction["deployable_safety"]
    return {
        "actual_base_from_external_camera_cv": json.loads(
            json.dumps(prediction["base_from_external_camera_cv"])
        ),
        "external_intrinsic_cv": json.loads(json.dumps(prediction["external_intrinsic_cv"])),
        "finger_joint_positions_m": [0.04, 0.04],
        "finger_object_contact_force_n": max(safety["finger_force_n"]),
        "arm_joint_max_drift_rad": safety["arm_joint_drift_rad"],
        "tcp_position_drift_m": safety["tcp_position_drift_m"],
        "tcp_orientation_drift_rad": safety["tcp_orientation_drift_rad"],
        "external_rgb_timestamp_s": safety["rgb_timestamp_s"],
        "external_pose_timestamp_s": safety["pose_timestamp_s"],
        "external_position_tracking_error_m": safety["camera_position_tracking_error_m"],
        "external_orientation_tracking_error_rad": safety["camera_orientation_tracking_error_rad"],
    }


def test_prediction_camera_and_safety_are_bound_to_same_raw_route_frame() -> None:
    config = load_g2c_dynamic_qualification_config(CONFIG_PATH)
    prediction, _ = _scoring_prediction()
    route_row = _route_row_for_prediction(prediction)
    identity = {
        name: prediction[name]
        for name in (
            "row_index",
            "seed",
            "sample_index",
            "viewpoint_id",
            "frame_role",
            "route_alternate_index",
            "route_alternate_viewpoint_id",
            "route_frame_index",
        )
    }
    qualification._validate_prediction_against_route_row(
        prediction,
        route_row=route_row,
        expected_identity=identity,
        config=config,
    )

    prediction["base_from_external_camera_cv"][0][3] += 0.01
    geometry = qualification._recompute_prediction_geometry(
        normalized_uv=prediction["predicted_object_normalized_uv"],
        intrinsic_cv=prediction["external_intrinsic_cv"],
        base_from_camera_cv=prediction["base_from_external_camera_cv"],
        sigma_xy_px=prediction["object_sigma_xy_px"],
        plane_base_z_m=0.02,
    )
    prediction.update(geometry)
    with pytest.raises(RuntimeError, match="camera/gripper"):
        qualification._validate_prediction_against_route_row(
            prediction,
            route_row=route_row,
            expected_identity=identity,
            config=config,
        )

    prediction, _ = _scoring_prediction()
    prediction["deployable_safety"]["arm_joint_drift_rad"] = 1e-6
    with pytest.raises(RuntimeError, match="arm_joint_drift_rad"):
        qualification._validate_prediction_against_route_row(
            prediction,
            route_row=route_row,
            expected_identity=identity,
            config=config,
        )


def test_primary_uses_shortlist_tier_before_metric_ties() -> None:
    config = load_g2c_dynamic_qualification_config(CONFIG_PATH)

    def summary(viewpoint_id: str, coverage: float) -> dict[str, object]:
        return {
            "viewpoint_id": viewpoint_id,
            "passed": True,
            "accepted_safe_coverage": coverage,
            "observable_world_xyz_p90_m": 0.001,
            "observable_world_xyz_max_m": 0.002,
            "covariance_95_coverage": 0.94,
            "visibility_recall": 1.0,
        }

    selection = select_qualification_primary(
        [
            summary("LEFT_LOW__CENTER", 1.0),
            summary("LEFT_LOW__YAW_LEFT", 0.8),
        ],
        config=config,
    )

    assert selection["primary_viewpoint_id"] == "LEFT_LOW__YAW_LEFT"


def test_public_verifier_signature_excludes_private_and_runtime_inputs() -> None:
    parameters = set(inspect.signature(verify_g2c_qualification_result).parameters)

    assert parameters == {
        "qualification_config_path",
        "public_execution_root",
        "result_root",
    }
    forbidden = {"private", "label", "model", "checkpoint", "simulator", "data"}
    assert not any(token in name for name in parameters for token in forbidden)


def test_scoring_hold_precedes_execution_or_private_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("scoring HOLD 后不得读 execution/private")

    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_qualification.verify_g2c_qualification_execution",
        unexpected,
    )
    with pytest.raises(PermissionError, match="HOLD"):
        score_e018_p1_g2c_qualification(
            qualification_config_path="missing",
            public_execution_root="missing",
            private_label_root="missing",
            result_output_root="missing",
            decision_scoring_go=False,
        )


def test_one_shot_scorer_and_public_verifier_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_g2c_dynamic_qualification_config(CONFIG_PATH)
    public = tmp_path / "execution"
    private = tmp_path / "private"
    journal = QualificationJournal(
        public_root=public,
        private_label_root=private,
        config_sha256=config["config_sha256"],
        classification="preflight/no-qualification-claim",
    )
    base_prediction, _ = _scoring_prediction()
    for name in (
        "previous_prediction_sha256",
        "prediction_write_started_at_unix_ns",
        "prediction_sha256",
    ):
        base_prediction.pop(name, None)
    predictions = []
    for row_index in range(2):
        prediction = dict(base_prediction)
        prediction["row_index"] = row_index
        prediction["batch_index"] = row_index
        committed, _ = journal.commit_prediction_then_capture_label(
            prediction,
            privileged_getter=_minimal_privileged_label,
        )
        predictions.append(committed)
    commits = [
        json.loads(
            (public / f"prediction_commits/{index:06d}.commit.json").read_text(encoding="utf-8")
        )
        for index in range(2)
    ]
    (public / "prediction_ledger.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    (public / "prediction_commit_ledger.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in commits),
        encoding="utf-8",
    )
    result = tmp_path / "result"
    output_identities = qualification._output_root_identities(
        public_root=public,
        private_root=private,
        result_root=result,
    )
    execution = {
        "classification": "preflight/no-qualification-claim",
        "config_sha256": config["config_sha256"],
        "source_git_commit": "4" * 40,
        "source_identity_sha256": "5" * 64,
        "execution_freeze_sha256": "6" * 64,
        "execution_receipt_internal_sha256": "7" * 64,
        "prediction_count": 2,
        "private_label_journal_inventory": journal.private_label_inventory(),
        "output_identities": {
            name: value
            for name, value in output_identities.items()
            if name != "result_output_identity_sha256"
        },
        "formal_execution_decision": {"status": "not-applicable-preflight-no-qualification-claim"},
        "artifact_bytes": sum(path.stat().st_size for path in public.rglob("*") if path.is_file()),
        "combined_artifact_bytes_max": config["budgets"]["artifact_bytes_max"],
        "verification_sha256": "8" * 64,
    }
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_qualification.verify_g2c_qualification_execution",
        lambda **_: execution,
    )
    receipt = score_e018_p1_g2c_qualification(
        qualification_config_path=CONFIG_PATH,
        public_execution_root=public,
        private_label_root=private,
        result_output_root=result,
        decision_scoring_go=True,
    )
    verification = verify_g2c_qualification_result(
        qualification_config_path=CONFIG_PATH,
        public_execution_root=public,
        result_root=result,
    )
    combined = qualification.verify_g2c_qualification_combined_artifacts(
        qualification_config_path=CONFIG_PATH,
        public_execution_root=public,
        private_label_root=private,
        result_root=result,
    )

    assert receipt["status"] == "complete-preflight-no-qualification-claim"
    assert receipt["gate_passed"] is False
    assert verification["verified"] is True
    assert verification["label_journal_reopen_count"] == 0
    assert combined["verified"] is True
    assert combined["label_content_reopen_count"] == 0
    assert combined["combined_artifact_bytes"] == (
        combined["public_execution_bytes"]
        + combined["private_label_bytes"]
        + combined["result_artifact_bytes"]
    )
    assert (private / "SCORING_CONSUMED.json").is_file()
    with pytest.raises(RuntimeError, match="exact tree"):
        score_e018_p1_g2c_qualification(
            qualification_config_path=CONFIG_PATH,
            public_execution_root=public,
            private_label_root=private,
            result_output_root=tmp_path / "second-result",
            decision_scoring_go=True,
        )
    first_label = private / "label_commits/000000.json"
    original_label = first_label.read_bytes()
    tampered_label = original_label.replace(b'"row_index": 0', b'"row_index": 1', 1)
    assert tampered_label != original_label
    assert len(tampered_label) == len(original_label)
    first_label.write_bytes(tampered_label)
    with pytest.raises(RuntimeError, match="label raw SHA"):
        qualification.verify_g2c_qualification_combined_artifacts(
            qualification_config_path=CONFIG_PATH,
            public_execution_root=public,
            private_label_root=private,
            result_root=result,
        )
    first_label.write_bytes(original_label)
    (private / "unexpected.bin").write_bytes(b"tamper")
    with pytest.raises(RuntimeError, match="exact tree"):
        qualification.verify_g2c_qualification_combined_artifacts(
            qualification_config_path=CONFIG_PATH,
            public_execution_root=public,
            private_label_root=private,
            result_root=result,
        )


def test_execution_receipt_write_failure_freezes_consumed_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public = tmp_path / "public"
    journal = QualificationJournal(
        public_root=public,
        private_label_root=tmp_path / "private",
        config_sha256="9" * 64,
        classification="preflight/no-qualification-claim",
    )
    journal.commit_prediction_then_capture_label(
        {"row_index": 0}, privileged_getter=_minimal_privileged_label
    )
    journal.freeze_route_completion()

    class Env:
        close_count = 0

        def close(self) -> None:
            self.close_count += 1

    class Provider:
        destroy_count = 0

        def destroy(self) -> None:
            self.destroy_count += 1

    env = Env()
    provider = Provider()
    original = qualification._atomic_create_json

    def fail_receipt(path: Path, value: object) -> tuple[str, int]:
        if path.name == "execution_receipt.json":
            raise OSError("synthetic receipt fsync failure")
        return original(path, value)

    monkeypatch.setattr(qualification, "_atomic_create_json", fail_receipt)
    with pytest.raises(OSError, match="receipt fsync"):
        qualification._finalize_qualification_execution(
            env=env,
            provider=provider,
            journal=journal,
            public_root=public,
            execution_freeze={"version": "synthetic"},
            environment_identity={},
        )

    failure = json.loads((public / "consumed_failure.json").read_text(encoding="utf-8"))
    state = json.loads((public / "phase_state.json").read_text(encoding="utf-8"))
    assert failure["label_array_consumed"] is True
    assert state["status"] == "consumed-qualification-failure"
    assert env.close_count >= 1
    assert provider.destroy_count >= 1


def test_capture_cleanup_failures_still_freeze_consumed_evidence(
    tmp_path: Path,
) -> None:
    journal = QualificationJournal(
        public_root=tmp_path / "public",
        private_label_root=tmp_path / "private",
        config_sha256="9" * 64,
        classification="preflight/no-qualification-claim",
    )

    class Env:
        def close(self) -> None:
            raise OSError("synthetic env close failure")

    class Provider:
        def destroy(self) -> None:
            raise RuntimeError("synthetic provider destroy failure")

    qualification._best_effort_freeze_capture_failure(
        env=Env(),
        provider=Provider(),
        journal=journal,
        error=ValueError("synthetic route failure"),
        evidence={"partial_row_count": 3},
    )

    failure = json.loads((tmp_path / "public/consumed_failure.json").read_text(encoding="utf-8"))
    assert failure["error_type"] == "ValueError"
    assert "cleanup_errors" in failure["error_message"]
    assert len(failure["cleanup_errors"]) == 2
    assert failure["failure_evidence"] == {"partial_row_count": 3}
    assert failure["rerun_under_same_identity_allowed"] is False


def test_actual_execution_finalizer_preserves_both_cleanup_failures(
    tmp_path: Path,
) -> None:
    journal = QualificationJournal(
        public_root=tmp_path / "public",
        private_label_root=tmp_path / "private",
        config_sha256="9" * 64,
        classification="preflight/no-qualification-claim",
    )

    class Env:
        def close(self) -> None:
            raise OSError("env-close")

    class Provider:
        def destroy(self) -> None:
            raise RuntimeError("provider-destroy")

    with pytest.raises(RuntimeError, match="context cleanup failures"):
        qualification._finalize_qualification_execution(
            env=Env(),
            provider=Provider(),
            journal=journal,
            public_root=tmp_path / "public",
            execution_freeze={"version": "synthetic"},
            environment_identity={},
        )
    failure = json.loads((tmp_path / "public/consumed_failure.json").read_text(encoding="utf-8"))
    assert len(failure["cleanup_errors"]) == 2
    assert failure["cleanup_errors"][0].startswith("env.close:OSError")
    assert failure["cleanup_errors"][1].startswith("provider.destroy:RuntimeError")


def test_failure_evidence_keeps_full_returned_route_before_gate_and_partial_route(
    tmp_path: Path,
) -> None:
    full_root = tmp_path / "full"
    motion = qualification._AppendOnlyJsonl(full_root / "camera_pose_ledger.jsonl")
    summaries = qualification._AppendOnlyJsonl(full_root / "route_summaries.jsonl")
    rows = [{"frame_index": index} for index in range(92)]
    qualification._append_returned_route_evidence(
        motion_writer=motion,
        summary_writer=summaries,
        rows=rows,
        summary={"status": "returned-before-gate"},
    )
    # 模拟紧随 append 的 protocol gate 失败；证据不得回滚或补跑。
    with pytest.raises(RuntimeError, match="synthetic gate"):
        raise RuntimeError("synthetic gate failure")
    full_evidence = qualification._freeze_capture_failure_evidence(
        public_root=full_root,
        motion_writer=motion,
        summary_writer=summaries,
        route_counters=[],
        persisted_route_count=1,
        active_route_identity={"seed": 76801},
        active_route_rows=rows,
        active_route_persisted=True,
        route_rgb_inventory_rows=[],
        journal=None,
    )
    assert full_evidence["persisted_full_route_count"] == 1
    assert full_evidence["motion_ledger"]["writer_row_count"] == 92
    assert full_evidence["route_summaries"]["writer_row_count"] == 1
    assert full_evidence["partial_route_ledger"] is None

    partial_root = tmp_path / "partial"
    partial_root.mkdir()
    partial_rows = [{"frame_index": index} for index in range(3)]
    partial_evidence = qualification._freeze_capture_failure_evidence(
        public_root=partial_root,
        motion_writer=None,
        summary_writer=None,
        route_counters=[],
        persisted_route_count=0,
        active_route_identity={"seed": 76801},
        active_route_rows=partial_rows,
        active_route_persisted=False,
        route_rgb_inventory_rows=[],
        journal=None,
    )
    assert partial_evidence["active_route_row_count"] == 3
    assert partial_evidence["partial_route_ledger"]["row_count"] == 3
    assert (partial_root / "partial_route_ledger.jsonl").is_file()


def test_consumed_failure_public_verifier_reopens_partial_ledger_and_hashes(
    tmp_path: Path,
) -> None:
    config = load_g2c_dynamic_qualification_config(CONFIG_PATH)
    public = tmp_path / "public"
    journal = QualificationJournal(
        public_root=public,
        private_label_root=tmp_path / "private",
        config_sha256=config["config_sha256"],
        classification="preflight/no-qualification-claim",
    )
    partial_rows = [{"frame_index": index} for index in range(3)]
    evidence = qualification._freeze_capture_failure_evidence(
        public_root=public,
        motion_writer=None,
        summary_writer=None,
        route_counters=[],
        persisted_route_count=0,
        active_route_identity={"seed": 76801},
        active_route_rows=partial_rows,
        active_route_persisted=False,
        route_rgb_inventory_rows=[],
        journal=journal,
    )
    journal.freeze_consumed_failure(RuntimeError("synthetic"), evidence=evidence)

    verification = qualification.verify_g2c_qualification_failure(
        qualification_config_path=CONFIG_PATH,
        public_execution_root=public,
    )

    assert verification["verified"] is True
    assert verification["prediction_commit_count"] == 0
    assert verification["rerun_under_same_identity_allowed"] is False


@pytest.mark.parametrize("tamper_kind", ["extra", "symlink", "hardlink"])
def test_consumed_failure_verifier_rejects_extra_and_link_files(
    tmp_path: Path, tamper_kind: str
) -> None:
    config = load_g2c_dynamic_qualification_config(CONFIG_PATH)
    public = tmp_path / "public"
    journal = QualificationJournal(
        public_root=public,
        private_label_root=tmp_path / "private",
        config_sha256=config["config_sha256"],
        classification=qualification.QUALIFICATION_CLASSIFICATION_SMOKE,
    )
    journal.freeze_consumed_failure(RuntimeError("synthetic"), evidence={})
    unexpected = public / "unexpected.bin"
    if tamper_kind == "extra":
        unexpected.write_bytes(b"extra")
    elif tamper_kind == "symlink":
        unexpected.symlink_to(public / "phase_state.json")
    else:
        unexpected.hardlink_to(public / "phase_state.json")

    with pytest.raises(RuntimeError, match="qualification"):
        qualification.verify_g2c_qualification_failure(
            qualification_config_path=CONFIG_PATH,
            public_execution_root=public,
        )
