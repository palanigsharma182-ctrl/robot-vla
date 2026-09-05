"""E018-P1 G2C 动态 front-provider qualification。

本模块把动态运动、逐帧 deployable prediction、privileged label 采集和离线
评分拆成单向的数据流。正式 qualification 仍由 Decision Gate 控制；D047 只
授权实现、测试和单 seed/单 route 的 noncanonical preflight。
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla.adapters import FrankaObservationAdapter
from robot_vla.contracts import RobotSpec
from robot_vla.observation import invert_se3, opengl_camera_to_opencv, validate_se3
from robot_vla.precision import e018_p1_g0 as _g0
from robot_vla.precision import e018_p1_g0c as _g0c
from robot_vla.precision.active_front_provider import (
    build_precision_camera_role_state,
)
from robot_vla.precision.active_front_camera import (
    ExternalCameraMotionState,
    FrontCameraOrientationMode,
    quaternion_multiply_wxyz,
)
from robot_vla.precision.e018_p1_g2a import (
    FRONT_ALTERNATE_IDS,
    FRONT_HOME_ID,
    G0B_SHORTLIST_ORDER,
    _mahalanobis_squared_psd,
    canonical_sha256,
    file_sha256,
)
from robot_vla.precision.e018_p1_g2c import (
    _measurement_covariance,
    g2c_dynamic_qualification_plan,
    validate_g2c_dynamic_qualification_counters,
)
from robot_vla.precision.e018_p1_g2c_calibration import (
    G2C_CALIBRATION_SELECTION_PARENT,
    load_g2c_calibration_config,
    verify_g2c_calibration_result,
)
from robot_vla.precision.e018_p1_g2c_data import (
    _base_point,
    _finger_force_n,
    _load_normalizers,
    _robot_object_contact_force_n,
    _single_bool,
    _single_rigid,
    load_e018_p1_g2c_data_config,
)
from robot_vla.precision.e018_p1_g2c_model_val import (
    _prediction_rows_for_batch,
)
from robot_vla.precision.e018_p1_g2c_training import (
    _git_source_identity,
)
from robot_vla.precision.geometry import project_base_point_to_normalized_uv
from robot_vla.precision.object_observability import (
    ObjectObservabilityLabel,
    ObjectWriteEvidence,
    derive_object_observability,
)
from robot_vla.precision.outliers import geometry_conditioning

E018_P1_G2C_QUALIFICATION_CONFIG_VERSION = "e018-p1-g2c-dynamic-qualification-development/v1"
E018_P1_G2C_QUALIFICATION_EXECUTION_VERSION = "e018-p1-g2c-dynamic-qualification-execution/v1"
E018_P1_G2C_QUALIFICATION_RESULT_VERSION = "e018-p1-g2c-dynamic-qualification-result/v1"

FORMAL_QUALIFICATION_SEEDS = tuple(range(76701, 76751))
QUALIFICATION_VIEW_ORDER = (FRONT_HOME_ID, *FRONT_ALTERNATE_IDS)
QUALIFICATION_CLASSIFICATION_FORMAL = (
    "formal-dynamic-qualification-no-test-no-memory-no-manipulation/v1"
)
QUALIFICATION_CLASSIFICATION_SMOKE = "preflight/no-qualification-claim"
QUALIFICATION_SOURCE_PHASE = "G2C_DYNAMIC_QUALIFICATION_NO_EXECUTIVE_PHASE"
QUALIFICATION_CAMERA_OWNER = "G2C_QUALIFICATION_ISOLATED_SIM_CAMERA"

_G0C_HOME_VIEWPOINT_ID = "HOME"
_ROUTE_STATE_PATTERN = (
    (ExternalCameraMotionState.HOME_ANCHOR.value, 1),
    (ExternalCameraMotionState.MOVE_TO_VIEW.value, 40),
    (ExternalCameraMotionState.SETTLE_AT_VIEW.value, 4),
    (ExternalCameraMotionState.COLLECT.value, 3),
    (ExternalCameraMotionState.RETURN_HOME.value, 40),
    (ExternalCameraMotionState.VERIFY_HOME_AND_ARM_HOLD.value, 4),
)
_FINAL_COLLECT_FRAME_INDEX = 47
_ROUTE_RGB_INVENTORY_VERSION = "e018-p1-g2c-route-rgb-inventory/v1"
_PRIVATE_LABEL_INVENTORY_VERSION = "e018-p1-g2c-private-label-inventory/v1"
_ARTIFACT_ACCOUNTING_VERSION = "e018-p1-g2c-combined-artifact-accounting/v1"
_FORMAL_EXECUTION_DECISION_VERSION = "e018-p1-g2c-formal-execution-decision-receipt/v1"
_FORMAL_DECISION_RECEIPT_KEYS = {
    "version",
    "decision_id",
    "status",
    "qualification_config",
    "source",
    "parent_identities",
    "d047_smoke",
    "formal_execution",
    "permissions",
    "budgets",
    "stopping_rules",
    "allowed_claim",
    "issued_at_unix_ns",
    "receipt_sha256",
}
_FORMAL_DECISION_CONFIG_KEYS = {"raw_sha256", "internal_sha256"}
_FORMAL_DECISION_SOURCE_KEYS = {"git_commit", "identity_sha256"}
_FORMAL_DECISION_PARENT_KEYS = {
    "g0c_config_sha256",
    "g0c_receipt_internal_sha256",
    "d046_calibration_result_receipt_raw_sha256",
    "d046_calibration_result_receipt_internal_sha256",
    "d046_calibration_result_verification_sha256",
    "d046_replicated_persistence",
    "selected_checkpoint",
}
_FORMAL_DECISION_PERSISTENCE_KEYS = {
    "artifact_id",
    "status",
    "drive_persistence_receipt_raw_sha256",
    "drive_persistence_receipt_internal_sha256",
    "replication_verification_sha256",
    "completion_marker_raw_sha256",
    "completion_marker_internal_sha256",
}
_FORMAL_DECISION_SMOKE_KEYS = {
    "experiment_id",
    "seed",
    "alternate_viewpoint_id",
    "classification",
    "source_git_commit",
    "source_identity_sha256",
    "qualification_config_raw_sha256",
    "qualification_config_internal_sha256",
    "execution_status",
    "result_status",
    "execution_receipt_raw_sha256",
    "execution_receipt_internal_sha256",
    "execution_verification_sha256",
    "result_receipt_raw_sha256",
    "result_receipt_internal_sha256",
    "result_verification_sha256",
    "combined_artifact_verification_sha256",
    "public_output_identity_sha256",
    "private_output_identity_sha256",
    "result_output_identity_sha256",
    "started_at_unix_ns",
    "completed_at_unix_ns",
    "wall_elapsed_seconds",
    "gpu_elapsed_seconds",
    "total_artifact_bytes",
    "formal_claim_allowed",
    "rerun_under_same_identity_allowed",
}
_FORMAL_DECISION_EXECUTION_KEYS = {
    "experiment_id",
    "execution_id",
    "classification",
    "public_output_identity_sha256",
    "private_output_identity_sha256",
    "result_output_identity_sha256",
    "seed_start",
    "seed_end",
    "seed_count",
    "alternate_order",
    "route_count",
    "expected_counts",
    "capture_attempt_count",
    "scoring_attempt_count",
    "test_split_status",
    "memory_and_active_loop",
    "actuator_and_manipulation",
}
_FORMAL_DECISION_EXPECTED_COUNT_KEYS = {
    "camera_pose_set_count",
    "ledger_frame_count",
    "moving_interpolation_command_count",
    "safe_hold_open_step_count",
    "prediction_count",
    "home_prediction_count",
    "alternate_prediction_count",
    "privileged_object_label_capture_count",
    "object_contact_event_count",
}
_FORMAL_DECISION_PERMISSION_KEYS = {
    "test_array_reads",
    "memory_reads",
    "memory_writes",
    "runtime_camera_actuation",
    "physical_camera_actuation",
    "nonzero_arm_motion_commands",
    "gripper_close_commands",
    "manipulation_progression",
    "checkpoint_writes",
    "goal_gt_read_count",
    "object_contact_event_count",
    "fresh_test_reads",
    "retry_count",
    "seed_replacement_count",
    "route_replacement_count",
}
_FORMAL_DECISION_BUDGET_KEYS = {
    "version",
    "audit",
    "gpu",
    "artifact",
}
_FORMAL_DECISION_BUDGET_AUDIT_KEYS = {
    "source",
    "clock",
    "worker_region",
    "worker_gpu_class",
    "audited_at_unix_ns",
    "gpu_envelope_start_unix_s",
    "gpu_envelope_end_unix_s",
    "known_gpu_components_seconds",
    "unknown_gpu_components",
    "artifact_inventory_item_count",
    "artifact_duplicate_inclusive_bytes",
    "artifact_scope",
    "artifact_counting_semantics",
}
_FORMAL_DECISION_KNOWN_GPU_COMPONENT_KEYS = {
    "D040_failed_train",
    "D041_formal_train",
    "D042_model_validation_phase_a",
    "D045_calibration_phase_a",
}
_FORMAL_DECISION_GPU_BUDGET_KEYS = {
    "hard_limit_seconds",
    "known_lower_bound_seconds",
    "unknown_component_conservative_envelope_seconds",
    "pre_smoke_conservative_upper_bound_seconds",
    "d047_smoke_actual_seconds",
    "pre_formal_conservative_upper_bound_seconds",
    "formal_wall_seconds_max",
    "formal_gpu_seconds_reserved_max",
    "projected_conservative_upper_bound_seconds",
}
_FORMAL_DECISION_ARTIFACT_BUDGET_KEYS = {
    "hard_limit_bytes",
    "audited_duplicate_inclusive_bytes",
    "unknown_component_conservative_envelope_bytes",
    "pre_smoke_conservative_upper_bound_bytes",
    "d047_smoke_actual_bytes",
    "pre_formal_conservative_upper_bound_bytes",
    "formal_combined_artifact_bytes_reserved_max",
    "projected_conservative_upper_bound_bytes",
}
_FORMAL_DECISION_STOPPING_RULE_KEYS = {
    "route_or_gate_failure",
    "partial_route_failure",
    "contact_or_ownership_violation",
    "budget_exceeded",
    "unsafe_or_catastrophic_result",
    "rerun_after_consumption",
}
_FORMAL_DECISION_VERIFICATION_KEYS = {
    "verified",
    "receipt",
    "receipt_raw_sha256",
    "receipt_internal_sha256",
    "verification_sha256",
}
_D036_CUMULATIVE_GPU_SECONDS_MAX = 36_000.0
_D036_CUMULATIVE_ARTIFACT_BYTES_MAX = 21_474_836_480
_D047_SMOKE_ARTIFACT_BYTES_MAX = 1_073_741_824
_D047_SMOKE_SECONDS_MAX = 900.0
_D048_FORMAL_GPU_SECONDS_RESERVE_MAX = 9_000.0
_D048_FORMAL_ARTIFACT_BYTES_RESERVE_MAX = 4_294_967_296
_D048_PRE_SMOKE_GPU_SECONDS_CONSERVATIVE_UPPER = 24_451.833224
_D048_PRE_SMOKE_ARTIFACT_BYTES_CONSERVATIVE_UPPER = 4_294_967_296
_D048_AUDITED_ARTIFACT_BYTES = 2_118_325_603
_D048_UNKNOWN_GPU_SECONDS_CONSERVATIVE_ENVELOPE = 24_019.513831184
_D048_UNKNOWN_ARTIFACT_BYTES_CONSERVATIVE_ENVELOPE = 2_176_641_693
_D048_KNOWN_GPU_COMPONENTS = {
    "D040_failed_train": 10.46814069100219,
    "D041_formal_train": 244.03758283700154,
    "D042_model_validation_phase_a": 174.83248984300008,
    "D045_calibration_phase_a": 2.981179444999725,
}
_D048_UNKNOWN_GPU_COMPONENTS = [
    "D037_DATA",
    "D043_MODEL_VALIDATION_PHASE_B",
    "D046_CALIBRATION_PHASE_B",
    "engineering_smoke_and_code_validation",
    "inter_run_idle_gaps_inside_mtime_envelope",
]
_ARTIFACT_ACCOUNTING_KEYS = {
    "version",
    "status",
    "classification",
    "config_sha256",
    "output_identities",
    "public_execution_bytes",
    "private_label_commit_bytes",
    "private_scoring_consumption_marker_bytes",
    "private_label_total_bytes",
    "result_total_bytes",
    "combined_total_bytes",
    "combined_budget_limit_bytes",
    "accounting_semantics",
    "accounting_sha256",
}
_D046_ARTIFACT_ID = "g2c-calibration-phase-b-score-calibrate-d046-4158a02-20260905-v1"
_D046_REPLICATED_PERSISTENCE = {
    "artifact_id": _D046_ARTIFACT_ID,
    "status": "REPLICATED",
    "drive_persistence_receipt_raw_sha256": (
        "5837dd74536ac9795625a5defb5157715b20cd20cc700f9fb9a60ca1e0038a59"
    ),
    "drive_persistence_receipt_internal_sha256": (
        "192c4b2a8370f32c1ea748283bebc03d096cf93b44d7c798d20a229e26ed3cdc"
    ),
    "replication_verification_sha256": (
        "20b3552489bb33e1081e8d90d2a65317a3be807442c3b8c0980a8082256c5d20"
    ),
    "completion_marker_raw_sha256": (
        "fe5ddf258a630e578b877170cd05460b0ff3f2fbc5ed17eb472003a970973530"
    ),
    "completion_marker_internal_sha256": (
        "47021fc7217092992e6326744f6ea20dfd7f0ab0b908d4907b25bec511d456be"
    ),
}
_PROVIDER_RAW_PREDICTION_KEYS = {
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
_QUALIFICATION_PREDICTION_KEYS = _PROVIDER_RAW_PREDICTION_KEYS | {
    "classification",
    "frame_role",
    "route_alternate_index",
    "route_alternate_viewpoint_id",
    "route_frame_index",
    "external_intrinsic_cv",
    "base_from_external_camera_cv",
    "deployable_safety",
    "deployable_free_static_safe",
    "object_write_structurally_eligible",
    "structurally_eligible",
    "calibration_scale_factor",
    "calibrated_covariance_base_m2",
    "calibrated_position_std_max_m",
    "write_threshold",
    "write_accepted",
    "memory_write_executed",
    "test_data_read",
}
_COMMITTED_QUALIFICATION_PREDICTION_KEYS = _QUALIFICATION_PREDICTION_KEYS | {
    "previous_prediction_sha256",
    "prediction_write_started_at_unix_ns",
    "prediction_sha256",
}
_DEPLOYABLE_SAFETY_KEYS = {
    "eligible_capture",
    "finger_force_n",
    "finger_force_valid",
    "raw_gripper_opening_ratio",
    "arm_joint_drift_rad",
    "tcp_position_drift_m",
    "tcp_orientation_drift_rad",
    "rgb_timestamp_s",
    "pose_timestamp_s",
    "camera_position_tracking_error_m",
    "camera_orientation_tracking_error_rad",
    "rotation_projection_error_frobenius",
}
_PRIVILEGED_OBJECT_LABEL_KEYS = {
    "gt_object_exists",
    "gt_observable",
    "gt_object_position_base_m",
    "gt_object_projection_valid",
    "gt_object_projected_normalized_uv",
    "gt_object_mask_sha256",
    "gt_object_visible_pixel_count",
    "gt_object_observability",
    "is_grasped",
    "robot_object_contact_force_n",
    "goal_gt_read_count",
    "test_data_read",
}
_PRIVILEGED_OBJECT_OBSERVABILITY_KEYS = {
    "object_exists",
    "projection_valid",
    "in_fov",
    "observable",
    "legacy_visible",
    "center_inside_object_mask",
    "center_inside_goal_mask",
    "local_object_visible_fraction",
    "object_mask_area_fraction",
    "occlusion_type",
    "semantics",
    "version",
    "legacy_contract_mismatch",
}
_PRIVATE_LABEL_COMMIT_KEYS = _PRIVILEGED_OBJECT_LABEL_KEYS | {
    "version",
    "row_index",
    "prediction_sha256",
    "prediction_raw_sha256",
    "prediction_commit_receipt_sha256",
    "prediction_write_started_at_unix_ns",
    "prediction_fsync_completed_at_unix_ns",
    "privileged_captured_at_unix_ns",
    "label_sha256",
}
_QUALIFICATION_ROUTE_ROW_KEYS = {
    "version",
    "episode_id",
    "request_id",
    "camera_command_sequence_id",
    "frame_index",
    "control_tick",
    "timestamp_s",
    "external_rgb_timestamp_s",
    "external_pose_timestamp_s",
    "timestamp_source",
    "external_rgb_pose_skew_s",
    "source_phase",
    "camera_motion_state",
    "viewpoint_primitive_id",
    "target_orientation_id",
    "orientation_progress",
    "commanded_yaw_offset_rad",
    "commanded_pitch_offset_rad",
    "commanded_roll_offset_rad",
    "arm_owner",
    "gripper_owner",
    "external_camera_owner",
    "arm_motion_command_max_abs",
    "gripper_hold_open_command",
    "commanded_external_position_world_m",
    "commanded_external_quaternion_sapien",
    "actual_external_position_world_m",
    "actual_external_quaternion_sapien",
    "commanded_world_from_external_camera_gl",
    "actual_world_from_external_camera_gl",
    "commanded_base_from_external_camera_cv",
    "actual_base_from_external_camera_cv",
    "external_intrinsic_cv",
    "external_pose_valid",
    "external_position_tracking_error_m",
    "external_orientation_tracking_error_rad",
    "external_linear_velocity_m_s",
    "external_linear_speed_m_s",
    "external_linear_acceleration_m_s2",
    "external_angular_speed_rad_s",
    "external_angular_acceleration_rad_s2",
    "settle_evidence_passed",
    "settle_streak",
    "settled",
    "measurement_write_eligible",
    "memory_write_executed",
    "arm_anchor_q_rad",
    "arm_current_q_rad",
    "arm_joint_max_drift_rad",
    "tcp_anchor_world",
    "tcp_current_world",
    "tcp_position_drift_m",
    "tcp_orientation_drift_rad",
    "world_from_robot_base",
    "finger_joint_positions_m",
    "minimum_finger_joint_position_m",
    "finger_object_contact_force_n",
    "robot_object_contact_force_n",
    "robot_object_contact_by_link",
    "is_grasping",
    "terminated",
    "truncated",
    "rgb_sha256",
    "offline_segmentation_diagnostics",
}

_EXPECTED_DECISION = {
    "implementation_gate": "D047",
    "formal_execution": "HOLD-until-exact-source-r2-and-explicit-go",
    "noncanonical_smoke": "GO-one-seed-one-route-maximum",
    "fresh_test": "HOLD",
    "memory_and_active_loop": "HOLD",
}
_EXPECTED_PARENT_SCALARS = {
    "g0c_config_sha256": ("c93bbfd48b6d9bc2fc75b5b87e4ded7161efebd7eda50cd81cc2ded47810e965"),
    "g0c_receipt_internal_sha256": (
        "bf8232b620cd5ff8de8c0007391252b8829c3ebbac320a7d5a60507beaca258e"
    ),
    "calibration_config_raw_sha256": (
        "b2ff7ee79a87a65bc080c5b5411a8989971fd262ad8226f8f51b1f055937f75f"
    ),
    "calibration_config_internal_sha256": (
        "98a5727766cfe46f133bc4945d154be58da52be8c7d341e0d857c84a65aeaa74"
    ),
    "calibration_prediction_freeze_internal_sha256": (
        "f739d35b7a9983fc2016b47584fe9739d7e1ffac39179c17ebe18f01db43a7a6"
    ),
    "calibration_result_receipt_raw_sha256": (
        "16b377e8da6a3539883101c47f53a766d1efc60da4ff4b44b302e0db827698a9"
    ),
    "calibration_result_receipt_internal_sha256": (
        "6e67503d1894670c8b5a8ea5f0139453eaa85f8114025b0172a112345926c837"
    ),
    "calibration_result_verification_sha256": (
        "f8c8c8fb68aea11da471972046a0ee24791d475b6c7d1ecf23a193f247bc8ed7"
    ),
    "calibration_viewpoints_raw_sha256": (
        "a51c879de45208d540bb3c5db8d6389ab3e2f8dace5e424ecef9c142578659b6"
    ),
    "data_config_raw_sha256": ("5b825bbd1034e10801617d19dc10fca1e15f5c3253ca77571192d627e2d9e4ef"),
    "data_config_internal_sha256": (
        "56718c0611fc620ccfb767141d8d0867ea5d03806348396d0a2e201fbff3d5de"
    ),
    "proprio_stats_sha256": ("2a1061b3a56edfcfeb6e955a1910dc309ff9b776dc4eb355192661fe628de01e"),
    "proprio_normalizer_sha256": (
        "eb39fa6750a80d4781559e465d694fca411e3a4c11f74cbd09423886735a219e"
    ),
    "finger_force_stats_sha256": (
        "fcc5b4b87aa13919ec261fc5e71a24e1b6446f47abdbc87d4b1bf4f93fe7a9e8"
    ),
    "finger_force_normalizer_sha256": (
        "6de38cb2a3d74c7da581a96712c1eafc974ccf52f4cefb5a339708c30b3e79d5"
    ),
}
_EXPECTED_QUALIFICATION = {
    "minimum_visibility_precision": 0.95,
    "minimum_visibility_recall": 0.9,
    "maximum_observable_world_xyz_p90_m": 0.005,
    "maximum_observable_world_xyz_max_m": 0.02,
    "maximum_unsafe_accepted_count": 0,
    "maximum_catastrophic_accepted_count": 0,
    "minimum_accepted_safe_coverage": 0.1,
    "minimum_covariance_95_coverage": 0.9,
    "minimum_covariance_evaluable_count": 30,
    "maximum_calibrated_position_std_m": 0.02,
    "covariance_chi_square_threshold": 5.991,
    "maximum_oracle_safe_error_m": 0.005,
    "catastrophic_world_xyz_error_m": 0.02,
    "metric_float_recompute_tolerance": 1e-14,
    "quantile_algorithm": "sorted-linear-type7-rational-index/v1",
    "g0b_shortlist_order": list(G0B_SHORTLIST_ORDER),
    "primary_selection_policy": (
        "qualified-then-shortlist-tier-coverage-p90-max-cov95-recall-" "frozen-order/v1"
    ),
}
_EXPECTED_CAPTURE_SAFETY = {
    "maximum_rgb_pose_skew_s": 0.01,
    "maximum_rotation_projection_error_frobenius": 0.000001,
    "maximum_camera_position_tracking_error_m": 0.00001,
    "maximum_camera_orientation_tracking_error_rad": 0.0001,
    "support_radius_px": 2,
    "object_center_base_z_m": 0.02,
    "object_center_base_z_tolerance_m": 0.00001,
    "require_not_grasped": True,
    "require_finger_force_valid": True,
    "maximum_finger_force_n": 0.01,
    "minimum_raw_gripper_opening_ratio": 0.95,
    "maximum_arm_joint_drift_rad": 0.00001,
    "maximum_tcp_position_drift_m": 0.00001,
    "maximum_tcp_orientation_drift_rad": 0.0001,
    "maximum_robot_object_contact_force_n": 0.01,
}
_EXPECTED_CALIBRATION_VALUES = {
    "HOME__CENTER": {
        "scale_factor": 1.0,
        "write_threshold": 0.6123920381069183,
    },
    "LEFT_LOW__CENTER": {
        "scale_factor": 1.0,
        "write_threshold": 0.6152185201644897,
    },
    "LEFT_LOW__YAW_LEFT": {
        "scale_factor": 1.0,
        "write_threshold": 0.6121437251567841,
    },
    "LEFT_LOW__YAW_RIGHT": {
        "scale_factor": 1.0,
        "write_threshold": 0.6172557175159454,
    },
    "LEFT_LOW__PITCH_UP": {
        "scale_factor": 1.0,
        "write_threshold": 0.6127982139587402,
    },
    "LEFT_LOW__PITCH_DOWN": {
        "scale_factor": 1.0,
        "write_threshold": 0.6146058142185211,
    },
    "RIGHT_LOW__CENTER": {
        "scale_factor": 1.4244880966132178,
        "write_threshold": 0.6158104538917542,
    },
    "RIGHT_LOW__YAW_LEFT": {
        "scale_factor": 1.0,
        "write_threshold": 0.6134378910064697,
    },
    "RIGHT_LOW__YAW_RIGHT": {
        "scale_factor": 1.2340519915495893,
        "write_threshold": 0.6161833703517914,
    },
    "RIGHT_LOW__PITCH_UP": {
        "scale_factor": 1.0,
        "write_threshold": 0.6161805391311646,
    },
    "RIGHT_LOW__PITCH_DOWN": {
        "scale_factor": 1.5121781332987676,
        "write_threshold": 0.6172976791858673,
    },
}
_EXPECTED_CALIBRATION_IDENTITIES = {
    "HOME__CENTER": "7091db69c2bdc65dc904c5ebb3fcd048beb51dc37e13acf484c072f5bece2d8e",
    "LEFT_LOW__CENTER": "67a2eb07c56fe704c02c31e2a372c27d4bed2ce6ddbcf654ef079b3b475c8005",
    "LEFT_LOW__YAW_LEFT": "180b38597260db84c1ab90ca0eddef43b9124da83b52ab0ccfa024a08a4ac139",
    "LEFT_LOW__YAW_RIGHT": "b8b37f013320c0d99d497d1fcc128dd7f68f84754bbdb0787dbec6c41d3aae53",
    "LEFT_LOW__PITCH_UP": "fcc5531ad989172a124c2cb16ee60283fdac36334c905ae9604f814bb323ca97",
    "LEFT_LOW__PITCH_DOWN": "7f9165b1f386b636f20749d47a9dfce369876c090711168e55651a6a79f12037",
    "RIGHT_LOW__CENTER": "dc6f7698608c00bd15b54f9a93cace6ea5cdf72ccf2f6d370802d50ed9dc6bd2",
    "RIGHT_LOW__YAW_LEFT": "8ebfff91add880b7320709225319a03988987030cd36b74b7c20099da12b9452",
    "RIGHT_LOW__YAW_RIGHT": "d6bcbf38e38efd4036dbbf441e0654cc5c55eaa242017760bb1db2b1439abe37",
    "RIGHT_LOW__PITCH_UP": "8b106a08d8c0f069b1d5b8944800b2d1ccc74423f1200634110df1d314dc54f2",
    "RIGHT_LOW__PITCH_DOWN": "42a6e1a85ca390252bd92dd6cc324d7a5254f4c2584b33a0bbb5cdb9fcb206cb",
}
_EXPECTED_JOURNAL = {
    "prediction_commit": ("one-atomic-json-file-per-scored-frame-fsync-file-and-parent/v1"),
    "prediction_hash_chain": True,
    "phase_state_before_first_privileged_read": True,
    "private_label_journal": ("write-only-during-routes-read-once-after-context-destroy/v1"),
    "rerun_after_consumption_allowed": False,
}
_PERMISSION_KEYS = {
    "test_array_reads",
    "memory_reads",
    "memory_writes",
    "runtime_camera_actuation",
    "physical_camera_actuation",
    "nonzero_arm_motion_commands",
    "gripper_close_commands",
    "manipulation_progression",
    "checkpoint_writes",
}


def _require_exact_keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是 JSON object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys 漂移: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise RuntimeError(f"{name} file/link 漂移: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是 JSON object")
    return value


def _read_json_once_with_raw_sha(path: Path, name: str) -> tuple[dict[str, Any], str]:
    """一次文件打开同时完成 JSON 解析和 raw SHA，供 one-shot private scorer。"""

    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise RuntimeError(f"{name} file/link 漂移: {path}")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是 JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _read_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise RuntimeError(f"{name} file/link 漂移: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{name} 第 {line_number} 行不是 object")
        rows.append(value)
    return rows


def load_g2c_dynamic_qualification_config(path: str | Path) -> dict[str, Any]:
    """严格加载 D047 config；不接受运行时阈值或 seed override。"""

    config = _read_json(Path(path), "G2C dynamic qualification config")
    _require_exact_keys(
        config,
        {
            "version",
            "status",
            "config_sha256",
            "decision",
            "parents",
            "route",
            "capture_safety",
            "calibration",
            "qualification",
            "journal",
            "permissions",
            "budgets",
        },
        "G2C dynamic qualification config",
    )
    unsigned = dict(config)
    internal = unsigned.pop("config_sha256")
    if internal != canonical_sha256(unsigned):
        raise RuntimeError("G2C qualification config self SHA 漂移")
    if (
        config["version"] != E018_P1_G2C_QUALIFICATION_CONFIG_VERSION
        or config["status"] != "implementation-ready-formal-execution-hold/v1"
        or config["decision"] != _EXPECTED_DECISION
    ):
        raise RuntimeError("G2C qualification version/status/decision 漂移")

    parents = _require_exact_keys(
        config["parents"],
        {*_EXPECTED_PARENT_SCALARS, "selected_checkpoint"},
        "G2C qualification parents",
    )
    if any(parents[name] != value for name, value in _EXPECTED_PARENT_SCALARS.items()):
        raise RuntimeError("G2C qualification frozen parent identity 漂移")
    if parents["selected_checkpoint"] != {
        key: G2C_CALIBRATION_SELECTION_PARENT[key]
        for key in (
            "candidate_id",
            "epoch",
            "checkpoint_sha256",
            "parameter_state_sha256",
            "provenance_sha256",
            "model_config_sha256",
        )
    }:
        raise RuntimeError("G2C qualification selected checkpoint identity 漂移")

    plan = g2c_dynamic_qualification_plan()
    route = _require_exact_keys(
        config["route"],
        {
            "seed_start",
            "seed_end",
            "seed_count",
            "alternate_count",
            "route_count",
            "home_viewpoint_id",
            "alternate_order",
            "per_route",
            "totals",
            "score_home_only_on_first_alternate_route_per_seed",
            "score_only_final_collect_frame",
            "one_attempt_per_route",
            "retry_allowed",
            "seed_or_route_replacement_allowed",
        },
        "G2C qualification route",
    )
    if (
        (route["seed_start"], route["seed_end"], route["seed_count"])
        != (FORMAL_QUALIFICATION_SEEDS[0], FORMAL_QUALIFICATION_SEEDS[-1], 50)
        or route["alternate_count"] != plan["alternate_count"]
        or route["route_count"] != plan["route_count"]
        or route["home_viewpoint_id"] != FRONT_HOME_ID
        or route["alternate_order"] != list(FRONT_ALTERNATE_IDS)
        or route["per_route"] != plan["per_route"]
        or route["totals"] != plan["totals"]
        or route["score_home_only_on_first_alternate_route_per_seed"] is not True
        or route["score_only_final_collect_frame"] is not True
        or route["one_attempt_per_route"] is not True
        or route["retry_allowed"] is not False
        or route["seed_or_route_replacement_allowed"] is not False
    ):
        raise RuntimeError("G2C qualification route/count/order 漂移")

    if config["capture_safety"] != _EXPECTED_CAPTURE_SAFETY:
        raise RuntimeError("G2C qualification capture safety contract 漂移")

    calibration = _require_exact_keys(
        config["calibration"],
        {"load_and_verify_d046_result_at_runtime", "values"},
        "G2C qualification calibration",
    )
    values = calibration["values"]
    if (
        calibration["load_and_verify_d046_result_at_runtime"] is not True
        or not isinstance(values, dict)
        or tuple(values) != QUALIFICATION_VIEW_ORDER
        or values != _EXPECTED_CALIBRATION_VALUES
    ):
        raise RuntimeError("G2C qualification calibration viewpoint order 漂移")
    for viewpoint_id, value in values.items():
        _require_exact_keys(
            value,
            {"scale_factor", "write_threshold"},
            f"G2C qualification calibration {viewpoint_id}",
        )
        scale = value["scale_factor"]
        threshold = value["write_threshold"]
        if (
            not isinstance(scale, (int, float))
            or isinstance(scale, bool)
            or not math.isfinite(float(scale))
            or float(scale) < 1.0
            or not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise RuntimeError("G2C qualification calibration value 漂移")

    if config["qualification"] != _EXPECTED_QUALIFICATION:
        raise RuntimeError("G2C qualification metric/PRIMARY contract 漂移")
    if config["journal"] != _EXPECTED_JOURNAL:
        raise RuntimeError("G2C qualification journal contract 漂移")
    permissions = _require_exact_keys(
        config["permissions"], _PERMISSION_KEYS, "G2C qualification permissions"
    )
    if any(type(value) is not int or value != 0 for value in permissions.values()):
        raise RuntimeError("G2C qualification permission 必须全部为 0")
    if config["budgets"] != {
        "noncanonical_smoke_seed_count_max": 1,
        "noncanonical_smoke_route_count_max": 1,
        "noncanonical_smoke_gpu_seconds_max": 900.0,
        "artifact_bytes_max": 1_073_741_824,
    }:
        raise RuntimeError("G2C qualification smoke/artifact budget 漂移")
    return config


def _validate_g2c_formal_execution_decision_receipt(
    receipt: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    qualification_config_raw_sha256: str,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
) -> str:
    """验证 D048 的嵌套 exact contract，并返回 internal SHA。"""

    _require_exact_keys(
        dict(receipt),
        _FORMAL_DECISION_RECEIPT_KEYS,
        "G2C formal decision receipt",
    )
    config_identity = _require_exact_keys(
        receipt.get("qualification_config"),
        _FORMAL_DECISION_CONFIG_KEYS,
        "G2C formal decision qualification config",
    )
    source = _require_exact_keys(
        receipt.get("source"),
        _FORMAL_DECISION_SOURCE_KEYS,
        "G2C formal decision source",
    )
    parents = _require_exact_keys(
        receipt.get("parent_identities"),
        _FORMAL_DECISION_PARENT_KEYS,
        "G2C formal decision parents",
    )
    persistence = _require_exact_keys(
        parents.get("d046_replicated_persistence"),
        _FORMAL_DECISION_PERSISTENCE_KEYS,
        "G2C formal decision D046 persistence",
    )
    smoke = _require_exact_keys(
        receipt.get("d047_smoke"),
        _FORMAL_DECISION_SMOKE_KEYS,
        "G2C formal decision D047 smoke",
    )
    formal = _require_exact_keys(
        receipt.get("formal_execution"),
        _FORMAL_DECISION_EXECUTION_KEYS,
        "G2C formal decision execution",
    )
    counts = _require_exact_keys(
        formal.get("expected_counts"),
        _FORMAL_DECISION_EXPECTED_COUNT_KEYS,
        "G2C formal decision expected counts",
    )
    permissions = _require_exact_keys(
        receipt.get("permissions"),
        _FORMAL_DECISION_PERMISSION_KEYS,
        "G2C formal decision permissions",
    )
    budgets = _require_exact_keys(
        receipt.get("budgets"),
        _FORMAL_DECISION_BUDGET_KEYS,
        "G2C formal decision budgets",
    )
    budget_audit = _require_exact_keys(
        budgets.get("audit"),
        _FORMAL_DECISION_BUDGET_AUDIT_KEYS,
        "G2C formal decision budget audit",
    )
    known_gpu_components = _require_exact_keys(
        budget_audit.get("known_gpu_components_seconds"),
        _FORMAL_DECISION_KNOWN_GPU_COMPONENT_KEYS,
        "G2C formal decision known GPU components",
    )
    gpu_budget = _require_exact_keys(
        budgets.get("gpu"),
        _FORMAL_DECISION_GPU_BUDGET_KEYS,
        "G2C formal decision GPU budget",
    )
    artifact_budget = _require_exact_keys(
        budgets.get("artifact"),
        _FORMAL_DECISION_ARTIFACT_BUDGET_KEYS,
        "G2C formal decision artifact budget",
    )
    stopping = _require_exact_keys(
        receipt.get("stopping_rules"),
        _FORMAL_DECISION_STOPPING_RULE_KEYS,
        "G2C formal decision stopping rules",
    )
    unsigned = dict(receipt)
    internal = unsigned.pop("receipt_sha256", None)
    expected_parents = config["parents"]
    expected_parent_subset = {
        "g0c_config_sha256": expected_parents["g0c_config_sha256"],
        "g0c_receipt_internal_sha256": expected_parents["g0c_receipt_internal_sha256"],
        "d046_calibration_result_receipt_raw_sha256": expected_parents[
            "calibration_result_receipt_raw_sha256"
        ],
        "d046_calibration_result_receipt_internal_sha256": expected_parents[
            "calibration_result_receipt_internal_sha256"
        ],
        "d046_calibration_result_verification_sha256": expected_parents[
            "calibration_result_verification_sha256"
        ],
        "selected_checkpoint": expected_parents["selected_checkpoint"],
    }
    expected_counts = {
        "camera_pose_set_count": 48_500,
        "ledger_frame_count": 46_000,
        "moving_interpolation_command_count": 40_000,
        "safe_hold_open_step_count": 48_000,
        "prediction_count": 550,
        "home_prediction_count": 50,
        "alternate_prediction_count": 500,
        "privileged_object_label_capture_count": 550,
        "object_contact_event_count": 0,
    }
    sha_fields = [
        *(
            smoke.get(name)
            for name in (
                "execution_receipt_raw_sha256",
                "execution_receipt_internal_sha256",
                "execution_verification_sha256",
                "result_receipt_raw_sha256",
                "result_receipt_internal_sha256",
                "result_verification_sha256",
                "combined_artifact_verification_sha256",
                "public_output_identity_sha256",
                "private_output_identity_sha256",
                "result_output_identity_sha256",
            )
        ),
        *(
            formal.get(name)
            for name in (
                "public_output_identity_sha256",
                "private_output_identity_sha256",
                "result_output_identity_sha256",
            )
        ),
    ]
    known_gpu_lower = gpu_budget.get("known_lower_bound_seconds")
    unknown_gpu_envelope = gpu_budget.get("unknown_component_conservative_envelope_seconds")
    pre_smoke_gpu_upper = gpu_budget.get("pre_smoke_conservative_upper_bound_seconds")
    smoke_gpu = gpu_budget.get("d047_smoke_actual_seconds")
    pre_formal_gpu_upper = gpu_budget.get("pre_formal_conservative_upper_bound_seconds")
    qualification_gpu = gpu_budget.get("formal_gpu_seconds_reserved_max")
    qualification_wall = gpu_budget.get("formal_wall_seconds_max")
    projected_gpu_upper = gpu_budget.get("projected_conservative_upper_bound_seconds")
    audited_bytes = artifact_budget.get("audited_duplicate_inclusive_bytes")
    unknown_bytes_envelope = artifact_budget.get("unknown_component_conservative_envelope_bytes")
    pre_smoke_bytes_upper = artifact_budget.get("pre_smoke_conservative_upper_bound_bytes")
    smoke_bytes = artifact_budget.get("d047_smoke_actual_bytes")
    pre_formal_bytes_upper = artifact_budget.get("pre_formal_conservative_upper_bound_bytes")
    qualification_bytes = artifact_budget.get("formal_combined_artifact_bytes_reserved_max")
    projected_bytes_upper = artifact_budget.get("projected_conservative_upper_bound_bytes")
    numeric_budget_values = (
        known_gpu_lower,
        unknown_gpu_envelope,
        pre_smoke_gpu_upper,
        smoke_gpu,
        pre_formal_gpu_upper,
        qualification_gpu,
        qualification_wall,
        projected_gpu_upper,
    )
    expected_stopping = {
        "route_or_gate_failure": "stop-consumed-failure-no-retry-no-replacement",
        "partial_route_failure": "freeze-partial-evidence-stop-no-rerun",
        "contact_or_ownership_violation": "stop-consumed-failure-no-actuation",
        "budget_exceeded": "stop-consumed-failure-no-promotion",
        "unsafe_or_catastrophic_result": "complete-scoring-fail-no-promotion",
        "rerun_after_consumption": "prohibited",
    }
    if (
        internal != canonical_sha256(unsigned)
        or receipt.get("version") != _FORMAL_EXECUTION_DECISION_VERSION
        or receipt.get("decision_id") != "D048"
        or receipt.get("status") != "GO-formal-dynamic-qualification-execution-and-one-shot-scoring"
        or config_identity
        != {
            "raw_sha256": qualification_config_raw_sha256,
            "internal_sha256": config["config_sha256"],
        }
        or source
        != {
            "git_commit": expected_source_git_commit,
            "identity_sha256": expected_source_identity_sha256,
        }
        or any(parents.get(name) != value for name, value in expected_parent_subset.items())
        or persistence != _D046_REPLICATED_PERSISTENCE
        or any(not _is_sha256(value) for value in sha_fields)
        or smoke.get("experiment_id") != "E018-P1-G2C-D047-PREFLIGHT"
        or smoke.get("seed") != 76801
        or smoke.get("alternate_viewpoint_id") != FRONT_ALTERNATE_IDS[0]
        or smoke.get("classification") != QUALIFICATION_CLASSIFICATION_SMOKE
        or smoke.get("source_git_commit") != source["git_commit"]
        or smoke.get("source_identity_sha256") != source["identity_sha256"]
        or smoke.get("qualification_config_raw_sha256") != config_identity["raw_sha256"]
        or smoke.get("qualification_config_internal_sha256") != config_identity["internal_sha256"]
        or smoke.get("execution_status") != "complete-execution-freeze-context-destroyed"
        or smoke.get("result_status") != "complete-preflight-no-qualification-claim"
        or type(smoke.get("started_at_unix_ns")) is not int
        or type(smoke.get("completed_at_unix_ns")) is not int
        or smoke["completed_at_unix_ns"] <= smoke["started_at_unix_ns"]
        or any(
            not isinstance(smoke.get(name), (int, float))
            or isinstance(smoke.get(name), bool)
            or not math.isfinite(float(smoke[name]))
            or not 0.0 <= float(smoke[name]) <= _D047_SMOKE_SECONDS_MAX
            for name in ("wall_elapsed_seconds", "gpu_elapsed_seconds")
        )
        or type(smoke.get("total_artifact_bytes")) is not int
        or not 0 < smoke["total_artifact_bytes"] <= _D047_SMOKE_ARTIFACT_BYTES_MAX
        or smoke.get("formal_claim_allowed") is not False
        or smoke.get("rerun_under_same_identity_allowed") is not False
        or formal.get("experiment_id") != "E018-P1-G2C-FORMAL-DYNAMIC-QUALIFICATION"
        or not isinstance(formal.get("execution_id"), str)
        or not formal["execution_id"].strip()
        or formal.get("classification") != QUALIFICATION_CLASSIFICATION_FORMAL
        or len(
            {
                formal["public_output_identity_sha256"],
                formal["private_output_identity_sha256"],
                formal["result_output_identity_sha256"],
            }
        )
        != 3
        or (formal.get("seed_start"), formal.get("seed_end"), formal.get("seed_count"))
        != (FORMAL_QUALIFICATION_SEEDS[0], FORMAL_QUALIFICATION_SEEDS[-1], 50)
        or formal.get("alternate_order") != list(FRONT_ALTERNATE_IDS)
        or formal.get("route_count") != 500
        or counts != expected_counts
        or formal.get("capture_attempt_count") != 1
        or formal.get("scoring_attempt_count") != 1
        or formal.get("test_split_status") != "prohibited-unread"
        or formal.get("memory_and_active_loop") != "HOLD"
        or formal.get("actuator_and_manipulation") != "HOLD"
        or any(type(value) is not int or value != 0 for value in permissions.values())
        or budgets.get("version") != "e018-p1-g2c-d036-conservative-cumulative-budget/v1"
        or budget_audit.get("source") != "read-only-receipt-mtime-and-filesystem-inventory/v1"
        or budget_audit.get("clock") != "UTC-unix-time/v1"
        or budget_audit.get("worker_region") != "US"
        or budget_audit.get("worker_gpu_class") != "single-NVIDIA-RTX-6000-Ada-Generation"
        or type(budget_audit.get("audited_at_unix_ns")) is not int
        or budget_audit["audited_at_unix_ns"] <= 0
        or not math.isclose(
            float(budget_audit.get("gpu_envelope_start_unix_s", math.nan)),
            1_788_561_330.569288,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or not math.isclose(
            float(budget_audit.get("gpu_envelope_end_unix_s", math.nan)),
            1_788_585_782.402512,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or any(
            not math.isclose(
                float(known_gpu_components.get(name, math.nan)),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for name, expected in _D048_KNOWN_GPU_COMPONENTS.items()
        )
        or budget_audit.get("unknown_gpu_components") != _D048_UNKNOWN_GPU_COMPONENTS
        or budget_audit.get("artifact_inventory_item_count") != 42
        or budget_audit.get("artifact_duplicate_inclusive_bytes") != _D048_AUDITED_ARTIFACT_BYTES
        or budget_audit.get("artifact_scope")
        != "all-g2c-run-and-drive-staging-trees-on-the-single-worker/v1"
        or budget_audit.get("artifact_counting_semantics")
        != "retained-filesystem-bytes-with-cross-tree-duplicates-conservatively-counted/v1"
        or gpu_budget.get("hard_limit_seconds") != _D036_CUMULATIVE_GPU_SECONDS_MAX
        or artifact_budget.get("hard_limit_bytes") != _D036_CUMULATIVE_ARTIFACT_BYTES_MAX
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in numeric_budget_values
        )
        or not math.isclose(
            float(known_gpu_lower),
            sum(_D048_KNOWN_GPU_COMPONENTS.values()),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(pre_smoke_gpu_upper),
            _D048_PRE_SMOKE_GPU_SECONDS_CONSERVATIVE_UPPER,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(unknown_gpu_envelope),
            _D048_UNKNOWN_GPU_SECONDS_CONSERVATIVE_ENVELOPE,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(pre_smoke_gpu_upper),
            float(known_gpu_lower) + float(unknown_gpu_envelope),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(smoke_gpu),
            float(smoke["gpu_elapsed_seconds"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(pre_formal_gpu_upper),
            float(pre_smoke_gpu_upper) + float(smoke_gpu),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not 0.0 < float(qualification_gpu) <= _D048_FORMAL_GPU_SECONDS_RESERVE_MAX
        or not 0.0 < float(qualification_wall) <= _D048_FORMAL_GPU_SECONDS_RESERVE_MAX
        or not math.isclose(
            float(projected_gpu_upper),
            float(pre_formal_gpu_upper) + float(qualification_gpu),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or float(projected_gpu_upper) > _D036_CUMULATIVE_GPU_SECONDS_MAX
        or type(audited_bytes) is not int
        or type(unknown_bytes_envelope) is not int
        or type(pre_smoke_bytes_upper) is not int
        or type(smoke_bytes) is not int
        or type(pre_formal_bytes_upper) is not int
        or type(qualification_bytes) is not int
        or type(projected_bytes_upper) is not int
        or audited_bytes != _D048_AUDITED_ARTIFACT_BYTES
        or unknown_bytes_envelope != _D048_UNKNOWN_ARTIFACT_BYTES_CONSERVATIVE_ENVELOPE
        or pre_smoke_bytes_upper != _D048_PRE_SMOKE_ARTIFACT_BYTES_CONSERVATIVE_UPPER
        or pre_smoke_bytes_upper != audited_bytes + unknown_bytes_envelope
        or smoke_bytes != smoke["total_artifact_bytes"]
        or pre_formal_bytes_upper != pre_smoke_bytes_upper + smoke_bytes
        or not 0 < qualification_bytes <= _D048_FORMAL_ARTIFACT_BYTES_RESERVE_MAX
        or projected_bytes_upper != pre_formal_bytes_upper + qualification_bytes
        or projected_bytes_upper > _D036_CUMULATIVE_ARTIFACT_BYTES_MAX
        or stopping != expected_stopping
        or receipt.get("allowed_claim")
        != (
            "dynamic-front-provider-qualified-for-shadow-planning-only-"
            "no-memory-no-actuator-no-test/v1"
        )
        or type(receipt.get("issued_at_unix_ns")) is not int
        or receipt["issued_at_unix_ns"] <= 0
        or smoke["started_at_unix_ns"] < budget_audit["audited_at_unix_ns"]
        or receipt["issued_at_unix_ns"] <= smoke["completed_at_unix_ns"]
    ):
        raise RuntimeError("G2C formal decision receipt contract/identity 漂移")
    return str(internal)


def verify_g2c_formal_execution_decision_receipt(
    *,
    decision_receipt_path: str | Path,
    expected_raw_sha256: str,
    expected_internal_sha256: str,
    qualification_config: Mapping[str, Any],
    qualification_config_raw_sha256: str,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
) -> dict[str, Any]:
    """验证独立 D048 GO receipt；bool 不能单独授权 formal execution。"""

    path = Path(decision_receipt_path)
    if file_sha256(path) != expected_raw_sha256:
        raise RuntimeError("G2C formal decision receipt raw SHA 漂移")
    receipt = _read_json(path, "G2C formal execution decision receipt")
    internal = _validate_g2c_formal_execution_decision_receipt(
        receipt,
        config=qualification_config,
        qualification_config_raw_sha256=qualification_config_raw_sha256,
        expected_source_git_commit=expected_source_git_commit,
        expected_source_identity_sha256=expected_source_identity_sha256,
    )
    if internal != expected_internal_sha256:
        raise RuntimeError("G2C formal decision receipt internal SHA 漂移")
    verification = {
        "verified": True,
        "receipt": receipt,
        "receipt_raw_sha256": expected_raw_sha256,
        "receipt_internal_sha256": internal,
    }
    verification["verification_sha256"] = canonical_sha256(verification)
    return verification


def _verify_embedded_formal_execution_decision(
    value: Any,
    *,
    config: Mapping[str, Any],
    qualification_config_raw_sha256: str,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
) -> dict[str, Any]:
    """公开 verifier 重验 capture 中嵌入的 D048 exact contract。"""

    embedded = _require_exact_keys(
        value,
        _FORMAL_DECISION_VERIFICATION_KEYS,
        "G2C embedded formal decision verification",
    )
    receipt = embedded.get("receipt")
    if not isinstance(receipt, Mapping):
        raise RuntimeError("G2C embedded formal decision receipt 类型漂移")
    internal = _validate_g2c_formal_execution_decision_receipt(
        receipt,
        config=config,
        qualification_config_raw_sha256=qualification_config_raw_sha256,
        expected_source_git_commit=expected_source_git_commit,
        expected_source_identity_sha256=expected_source_identity_sha256,
    )
    unsigned = dict(embedded)
    verification_sha256 = unsigned.pop("verification_sha256", None)
    if (
        embedded.get("verified") is not True
        or not _is_sha256(embedded.get("receipt_raw_sha256"))
        or embedded.get("receipt_internal_sha256") != internal
        or verification_sha256 != canonical_sha256(unsigned)
    ):
        raise RuntimeError("G2C embedded formal decision verification 漂移")
    return dict(embedded)


def _validate_formal_execution_decision_time_order(
    *,
    execution_started_at_unix_ns: Any,
    classification: str,
    formal_decision_verification: Mapping[str, Any] | None,
) -> None:
    """formal capture 必须严格晚于已验证 D048 签发；smoke 不适用。"""

    if classification != QUALIFICATION_CLASSIFICATION_FORMAL:
        return
    decision_receipt = (
        None
        if formal_decision_verification is None
        else formal_decision_verification.get("receipt")
    )
    issued_at = (
        decision_receipt.get("issued_at_unix_ns") if isinstance(decision_receipt, Mapping) else None
    )
    if (
        type(execution_started_at_unix_ns) is not int
        or type(issued_at) is not int
        or execution_started_at_unix_ns <= issued_at
    ):
        raise RuntimeError("formal qualification execution 必须严格晚于 D048 GO 签发")


def _verify_g0c_parent_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_config_sha256: str,
    expected_internal_sha256: str,
) -> None:
    unsigned = dict(receipt)
    internal = unsigned.pop("receipt_sha256", None)
    if (
        internal != canonical_sha256(unsigned)
        or internal != expected_internal_sha256
        or receipt.get("version") != _g0c.E018_P1_G0C_RESULT_VERSION
        or receipt.get("status") != "complete-development-only"
        or receipt.get("gate_passed") is not True
        or receipt.get("config_sha256") != expected_config_sha256
        or receipt.get("test_split_status") != "prohibited-unread"
    ):
        raise RuntimeError("G2C qualification G0C receipt parent 漂移")


def verify_g2c_qualification_parents(
    *,
    qualification_config_path: str | Path,
    g0c_config_path: str | Path,
    g0c_receipt_path: str | Path,
    calibration_config_path: str | Path,
    calibration_prediction_freeze_root: str | Path,
    calibration_result_root: str | Path,
    data_config_path: str | Path,
) -> dict[str, Any]:
    """在加载 provider 前验证 G0C、D046、normalizer 和 calibration 绑定。"""

    config = load_g2c_dynamic_qualification_config(qualification_config_path)
    parents = config["parents"]
    g0c_config = _g0c.load_e018_p1_g0c_config(g0c_config_path)
    if canonical_sha256(g0c_config) != parents["g0c_config_sha256"]:
        raise RuntimeError("G2C qualification G0C config parent 漂移")
    g0c_receipt = _read_json(Path(g0c_receipt_path), "G2C qualification G0C receipt")
    _verify_g0c_parent_receipt(
        g0c_receipt,
        expected_config_sha256=parents["g0c_config_sha256"],
        expected_internal_sha256=parents["g0c_receipt_internal_sha256"],
    )

    calibration_config_path = Path(calibration_config_path)
    if file_sha256(calibration_config_path) != parents["calibration_config_raw_sha256"]:
        raise RuntimeError("G2C qualification calibration config raw SHA 漂移")
    calibration_config = load_g2c_calibration_config(calibration_config_path)
    if calibration_config["config_sha256"] != parents["calibration_config_internal_sha256"]:
        raise RuntimeError("G2C qualification calibration config identity 漂移")
    calibration_verification = verify_g2c_calibration_result(
        calibration_config_path=calibration_config_path,
        prediction_freeze_root=calibration_prediction_freeze_root,
        output_root=calibration_result_root,
    )
    if (
        calibration_verification.get("status") != "complete-calibration-pass"
        or calibration_verification.get("gate_passed") is not True
        or calibration_verification.get("receipt_raw_sha256")
        != parents["calibration_result_receipt_raw_sha256"]
        or calibration_verification.get("receipt_internal_sha256")
        != parents["calibration_result_receipt_internal_sha256"]
        or calibration_verification.get("verification_sha256")
        != parents["calibration_result_verification_sha256"]
        or calibration_verification.get("prediction_freeze_internal_sha256")
        != parents["calibration_prediction_freeze_internal_sha256"]
        or tuple(calibration_verification.get("qualified_non_home_viewpoint_ids", ()))
        != FRONT_ALTERNATE_IDS
    ):
        raise RuntimeError("G2C qualification D046 result parent 漂移")
    calibrations_path = Path(calibration_result_root) / "viewpoint_calibrations.json"
    if file_sha256(calibrations_path) != parents["calibration_viewpoints_raw_sha256"]:
        raise RuntimeError("G2C qualification D046 calibrations raw SHA 漂移")
    calibration_rows = json.loads(calibrations_path.read_text(encoding="utf-8"))
    if not isinstance(calibration_rows, list) or len(calibration_rows) != 11:
        raise RuntimeError("G2C qualification D046 calibration count 漂移")
    by_view = {str(row.get("viewpoint_id")): row for row in calibration_rows}
    if tuple(by_view) != QUALIFICATION_VIEW_ORDER:
        raise RuntimeError("G2C qualification D046 calibration order 漂移")
    for viewpoint_id in QUALIFICATION_VIEW_ORDER:
        row = by_view[viewpoint_id]
        stored = config["calibration"]["values"][viewpoint_id]
        if (
            row.get("passed") is not True
            or not isinstance(row.get("calibration"), dict)
            or row["calibration"].get("scale_factor") != stored["scale_factor"]
            or row.get("write_threshold") != stored["write_threshold"]
        ):
            raise RuntimeError(f"G2C qualification {viewpoint_id} calibration 漂移")

    data_config_path = Path(data_config_path)
    if file_sha256(data_config_path) != parents["data_config_raw_sha256"]:
        raise RuntimeError("G2C qualification DATA config raw SHA 漂移")
    data_config = load_e018_p1_g2c_data_config(data_config_path)
    if canonical_sha256(data_config) != parents["data_config_internal_sha256"]:
        raise RuntimeError("G2C qualification DATA config identity 漂移")
    if any(
        data_config["data_identity"][name] != parents[name]
        for name in (
            "proprio_stats_sha256",
            "proprio_normalizer_sha256",
            "finger_force_stats_sha256",
            "finger_force_normalizer_sha256",
        )
    ):
        raise RuntimeError("G2C qualification normalizer parent 漂移")
    data_capture = data_config["capture"]
    lifecycle = data_capture["eligible_lifecycle_invariants"]
    expected_from_data = {
        "maximum_rgb_pose_skew_s": data_capture["maximum_rgb_pose_skew_s"],
        "maximum_rotation_projection_error_frobenius": data_capture[
            "maximum_rotation_projection_error_frobenius"
        ],
        "maximum_camera_position_tracking_error_m": data_capture[
            "maximum_camera_position_tracking_error_m"
        ],
        "maximum_camera_orientation_tracking_error_rad": data_capture[
            "maximum_camera_orientation_tracking_error_rad"
        ],
        "support_radius_px": data_capture["support_radius_px"],
        **{
            name: lifecycle[name]
            for name in (
                "object_center_base_z_m",
                "object_center_base_z_tolerance_m",
                "require_not_grasped",
                "require_finger_force_valid",
                "maximum_finger_force_n",
                "minimum_raw_gripper_opening_ratio",
                "maximum_arm_joint_drift_rad",
                "maximum_tcp_position_drift_m",
                "maximum_tcp_orientation_drift_rad",
                "maximum_robot_object_contact_force_n",
            )
        },
    }
    if config["capture_safety"] != expected_from_data:
        raise RuntimeError("G2C qualification capture safety 与 DATA parent 漂移")
    calibration_identities = {
        viewpoint_id: canonical_sha256(by_view[viewpoint_id])
        for viewpoint_id in QUALIFICATION_VIEW_ORDER
    }
    if calibration_identities != _EXPECTED_CALIBRATION_IDENTITIES:
        raise RuntimeError("G2C qualification D046 per-view identity 漂移")
    return {
        "config_sha256": config["config_sha256"],
        "g0c_config_sha256": parents["g0c_config_sha256"],
        "g0c_receipt_internal_sha256": parents["g0c_receipt_internal_sha256"],
        "calibration_verification": calibration_verification,
        "calibrations": by_view,
        "calibration_identities": calibration_identities,
        "data_config": data_config,
    }


def qualification_scored_frame_identity(
    row: Mapping[str, Any],
    *,
    seed: int,
    alternate_index: int,
    alternate_viewpoint_id: str,
) -> dict[str, Any] | None:
    """把 92 帧 route hook 限缩为冻结的 HOME/末个 COLLECT 两类样本。"""

    if (
        type(seed) is not int
        or seed <= 0
        or type(alternate_index) is not int
        or not 0 <= alternate_index < len(FRONT_ALTERNATE_IDS)
        or alternate_viewpoint_id != FRONT_ALTERNATE_IDS[alternate_index]
    ):
        raise ValueError("qualification seed/alternate index/order 漂移")
    frame_index = row.get("frame_index")
    state = row.get("camera_motion_state")
    viewpoint_id = row.get("viewpoint_primitive_id")
    if type(frame_index) is not int or not 0 <= frame_index < 92:
        raise RuntimeError("qualification hook frame_index 漂移")

    if frame_index == 0:
        if (
            state != ExternalCameraMotionState.HOME_ANCHOR.value
            or viewpoint_id != _G0C_HOME_VIEWPOINT_ID
        ):
            raise RuntimeError("qualification HOME_ANCHOR identity 漂移")
        if alternate_index != 0:
            return None
        return {
            "seed": seed,
            "sample_index": 0,
            "viewpoint_id": FRONT_HOME_ID,
            "frame_role": "home-anchor-first-route-only/v1",
            "route_alternate_index": alternate_index,
            "route_alternate_viewpoint_id": alternate_viewpoint_id,
            "route_frame_index": frame_index,
        }

    if frame_index == _FINAL_COLLECT_FRAME_INDEX:
        if (
            state != ExternalCameraMotionState.COLLECT.value
            or viewpoint_id != alternate_viewpoint_id
            or row.get("settled") is not True
            or row.get("measurement_write_eligible") is not True
        ):
            raise RuntimeError("qualification final COLLECT identity/settle 漂移")
        return {
            "seed": seed,
            "sample_index": alternate_index + 1,
            "viewpoint_id": alternate_viewpoint_id,
            "frame_role": "alternate-final-collect/v1",
            "route_alternate_index": alternate_index,
            "route_alternate_viewpoint_id": alternate_viewpoint_id,
            "route_frame_index": frame_index,
        }

    # 前两个 COLLECT 帧只用于 settle/safety ledger，绝不进入 provider。
    if state == ExternalCameraMotionState.COLLECT.value and frame_index not in (45, 46):
        raise RuntimeError("qualification COLLECT frame index 漂移")
    return None


def _qualification_finite_vector(row: Mapping[str, Any], key: str, size: int) -> np.ndarray:
    value = np.asarray(row.get(key), dtype=np.float64)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise RuntimeError(f"qualification route {key} 必须是有限 [{size}] witness")
    return value


def _qualification_finite_matrix(row: Mapping[str, Any], key: str) -> np.ndarray:
    value = np.asarray(row.get(key), dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise RuntimeError(f"qualification route {key} 必须是有限 [4,4] witness")
    return validate_se3(value, f"qualification route {key}")


def _qualification_close(actual: Any, expected: float, name: str) -> None:
    if (
        not isinstance(actual, (int, float))
        or isinstance(actual, bool)
        or not math.isfinite(float(actual))
        or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise RuntimeError(f"qualification route {name} 原始 witness 重算漂移")


def _quaternion_to_rotation_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise RuntimeError("qualification quaternion witness 无效")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise RuntimeError("qualification quaternion witness 范数为零")
    w, x, y, z = value / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _expected_sapien_camera_rotation(
    position_world_m: np.ndarray,
    look_at_world_m: Sequence[float],
    *,
    yaw_offset_rad: float,
    pitch_offset_rad: float,
) -> np.ndarray:
    """纯 NumPy 复现 ManiSkill look_at + SAPIEN local yaw/pitch 语义。"""

    target = np.asarray(look_at_world_m, dtype=np.float64)
    forward = target - position_world_m
    forward /= np.linalg.norm(forward)
    up_hint = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    left = np.cross(up_hint, forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    nominal = np.stack((forward, left, up), axis=-1)
    half_yaw = yaw_offset_rad * 0.5
    half_pitch = pitch_offset_rad * 0.5
    local_yaw = np.asarray(
        (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
        dtype=np.float64,
    )
    local_pitch = np.asarray(
        (math.cos(half_pitch), 0.0, -math.sin(half_pitch), 0.0),
        dtype=np.float64,
    )
    local_offset = quaternion_multiply_wxyz(local_yaw, local_pitch)
    return nominal @ _quaternion_to_rotation_wxyz(local_offset)


def _pose_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def _recompute_qualification_route_witnesses(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    alternate_viewpoint_id: str,
    g0c_config: Mapping[str, Any],
    capture_safety: Mapping[str, Any],
    rgb_images: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """从 raw pose/q/TCP/RGB witness 重算逐帧 mechanics 与完整 G0C gates。"""

    home, anchors, orientations = _g0c._parse_library(dict(g0c_config))
    primitives = _g0c._expand_primitives(anchors, orientations)
    by_id = {item.viewpoint_id: (item, orientation) for item, orientation in primitives}
    if tuple(by_id) != FRONT_ALTERNATE_IDS or alternate_viewpoint_id not in by_id:
        raise RuntimeError("qualification G0C primitive identity/order 漂移")
    alternate, orientation = by_id[alternate_viewpoint_id]
    environment = g0c_config["environment"]
    motion = g0c_config["motion"]
    safety = g0c_config["safety"]
    control_hz = environment["control_hz"]
    control_period = 1.0 / float(control_hz)
    move_steps = round(float(motion["move_duration_s"]) * control_hz)
    if move_steps != 40:
        raise RuntimeError("qualification G0C move step 漂移")

    home_position = np.asarray(home.position_world_m, dtype=np.float64)
    alternate_position = np.asarray(alternate.position_world_m, dtype=np.float64)
    outward = _g0.sample_translation_path(home_position, alternate_position, steps=move_steps)
    returning = _g0.sample_translation_path(alternate_position, home_position, steps=move_steps)
    expected_commanded_positions = [
        home_position,
        *outward,
        *([alternate_position] * (motion["settle_ticks"] + motion["collect_ticks"])),
        *returning,
        *([home_position] * motion["settle_ticks"]),
    ]
    if len(expected_commanded_positions) != len(rows):
        raise RuntimeError("qualification command trajectory count 漂移")

    episode_id = (
        f"g2c-qualification-seed-{seed:06d}-" f"{alternate_viewpoint_id.lower().replace('_', '-')}"
    )
    request_id = f"{episode_id}-request-00"
    command_id = f"{episode_id}-camera-sequence-00"
    target_offsets = np.asarray(
        (
            orientation.yaw_offset_rad,
            orientation.pitch_offset_rad,
            orientation.roll_offset_rad,
        ),
        dtype=np.float64,
    )
    expected_states = tuple(state for state, count in _ROUTE_STATE_PATTERN for _ in range(count))
    previous_position: np.ndarray | None = None
    previous_quaternion: np.ndarray | None = None
    previous_velocity: np.ndarray | None = None
    previous_angular_speed: float | None = None
    settle_streak = int(motion["warmup_ticks"])
    home_command_quaternion: np.ndarray | None = None
    contact_link_names: tuple[str, ...] | None = None

    for index, (row, state, commanded_expected) in enumerate(
        zip(rows, expected_states, expected_commanded_positions, strict=True)
    ):
        if index == 0:
            viewpoint_expected = _G0C_HOME_VIEWPOINT_ID
            orientation_id_expected = "CENTER"
            progress_expected = 1.0
        elif index <= 40:
            viewpoint_expected = alternate_viewpoint_id
            orientation_id_expected = orientation.orientation_id
            progress_expected = _g0.smootherstep(index / move_steps)
        elif index <= 47:
            viewpoint_expected = alternate_viewpoint_id
            orientation_id_expected = orientation.orientation_id
            progress_expected = 1.0
        elif index <= 87:
            viewpoint_expected = _G0C_HOME_VIEWPOINT_ID
            orientation_id_expected = orientation.orientation_id
            progress_expected = 1.0 - _g0.smootherstep((index - 47) / move_steps)
        else:
            viewpoint_expected = _G0C_HOME_VIEWPOINT_ID
            orientation_id_expected = "CENTER"
            progress_expected = 1.0
        offset_scale = 0.0 if orientation_id_expected == "CENTER" else progress_expected
        expected_offsets = target_offsets * offset_scale
        expected_timestamp = index * control_period
        if (
            row.get("version") != E018_P1_G2C_QUALIFICATION_EXECUTION_VERSION
            or row.get("episode_id") != episode_id
            or row.get("request_id") != request_id
            or row.get("camera_command_sequence_id") != command_id
            or row.get("frame_index") != index
            or row.get("control_tick") != index
            or row.get("timestamp_source") != "synchronous-simulator-control-tick-derived/v1"
            or row.get("source_phase") != QUALIFICATION_SOURCE_PHASE
            or row.get("external_camera_owner") != QUALIFICATION_CAMERA_OWNER
            or row.get("camera_motion_state") != state
            or row.get("viewpoint_primitive_id") != viewpoint_expected
            or row.get("target_orientation_id") != orientation_id_expected
            or row.get("external_pose_valid") is not True
        ):
            raise RuntimeError("qualification route identity/state/orientation pattern 漂移")
        for name in ("timestamp_s", "external_rgb_timestamp_s", "external_pose_timestamp_s"):
            _qualification_close(row.get(name), expected_timestamp, name)
        _qualification_close(
            row.get("external_rgb_pose_skew_s"),
            abs(float(row["external_rgb_timestamp_s"]) - float(row["external_pose_timestamp_s"])),
            "external_rgb_pose_skew_s",
        )
        _qualification_close(
            row.get("orientation_progress"),
            progress_expected,
            "orientation_progress",
        )
        for name, expected in zip(
            (
                "commanded_yaw_offset_rad",
                "commanded_pitch_offset_rad",
                "commanded_roll_offset_rad",
            ),
            expected_offsets,
            strict=True,
        ):
            _qualification_close(row.get(name), float(expected), name)

        commanded_position = _qualification_finite_vector(
            row, "commanded_external_position_world_m", 3
        )
        actual_position = _qualification_finite_vector(row, "actual_external_position_world_m", 3)
        commanded_quaternion = _qualification_finite_vector(
            row, "commanded_external_quaternion_sapien", 4
        )
        actual_quaternion = _qualification_finite_vector(
            row, "actual_external_quaternion_sapien", 4
        )
        if (
            not np.allclose(commanded_position, commanded_expected, rtol=0.0, atol=1e-12)
            or not math.isclose(
                float(np.linalg.norm(commanded_quaternion)),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or not math.isclose(
                float(np.linalg.norm(actual_quaternion)),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise RuntimeError("qualification commanded/actual camera pose witness 漂移")
        expected_command_rotation = _expected_sapien_camera_rotation(
            commanded_expected,
            alternate.look_at_world_m
            if viewpoint_expected == alternate_viewpoint_id
            else home.look_at_world_m,
            yaw_offset_rad=float(expected_offsets[0]),
            pitch_offset_rad=float(expected_offsets[1]),
        )
        commanded_rotation = _quaternion_to_rotation_wxyz(commanded_quaternion)
        actual_rotation = _quaternion_to_rotation_wxyz(actual_quaternion)
        mount_rotation = _quaternion_to_rotation_wxyz((-0.5, -0.5, 0.5, 0.5))
        expected_command_world_from_gl = _pose_matrix(
            commanded_expected, expected_command_rotation @ mount_rotation
        )
        expected_actual_world_from_gl = _pose_matrix(
            actual_position, actual_rotation @ mount_rotation
        )
        commanded_world_from_gl = _qualification_finite_matrix(
            row, "commanded_world_from_external_camera_gl"
        )
        actual_world_from_gl = _qualification_finite_matrix(
            row, "actual_world_from_external_camera_gl"
        )
        world_from_base = _qualification_finite_matrix(row, "world_from_robot_base")
        base_from_world = invert_se3(world_from_base, "qualification world_from_robot_base")
        expected_command_base_from_cv = validate_se3(
            base_from_world @ opengl_camera_to_opencv(commanded_world_from_gl),
            "qualification expected commanded base_from_camera_cv",
        )
        expected_actual_base_from_cv = validate_se3(
            base_from_world @ opengl_camera_to_opencv(actual_world_from_gl),
            "qualification expected actual base_from_camera_cv",
        )
        stored_command_base_from_cv = _qualification_finite_matrix(
            row, "commanded_base_from_external_camera_cv"
        )
        stored_actual_base_from_cv = _qualification_finite_matrix(
            row, "actual_base_from_external_camera_cv"
        )
        intrinsic = np.asarray(row.get("external_intrinsic_cv"), dtype=np.float64)
        if (
            not np.allclose(
                commanded_rotation,
                expected_command_rotation,
                rtol=0.0,
                atol=2e-6,
            )
            or not np.allclose(
                commanded_world_from_gl,
                expected_command_world_from_gl,
                rtol=0.0,
                atol=2e-6,
            )
            or not np.allclose(
                actual_world_from_gl,
                expected_actual_world_from_gl,
                rtol=0.0,
                atol=2e-6,
            )
            or not np.allclose(
                stored_command_base_from_cv,
                expected_command_base_from_cv,
                rtol=0.0,
                atol=2e-6,
            )
            or not np.allclose(
                stored_actual_base_from_cv,
                expected_actual_base_from_cv,
                rtol=0.0,
                atol=2e-6,
            )
            or intrinsic.shape != (3, 3)
            or not np.isfinite(intrinsic).all()
        ):
            raise RuntimeError("qualification frozen look-at/orientation/transform witness 漂移")
        if index == 0:
            home_command_quaternion = commanded_quaternion
        elif (
            index >= 88
            and _g0.quaternion_angular_distance_rad(commanded_quaternion, home_command_quaternion)
            > 1e-12
        ):
            raise RuntimeError("qualification VERIFY HOME command orientation 漂移")

        position_error = float(np.linalg.norm(actual_position - commanded_position))
        orientation_error = _g0.quaternion_angular_distance_rad(
            commanded_quaternion, actual_quaternion
        )
        velocity = (
            np.zeros(3, dtype=np.float64)
            if previous_position is None
            else (actual_position - previous_position) / control_period
        )
        linear_speed = float(np.linalg.norm(velocity))
        angular_speed = (
            0.0
            if previous_quaternion is None
            else _g0.quaternion_angular_distance_rad(previous_quaternion, actual_quaternion)
            / control_period
        )
        linear_acceleration = (
            0.0
            if previous_velocity is None
            else float(np.linalg.norm(velocity - previous_velocity) / control_period)
        )
        angular_acceleration = (
            0.0
            if previous_angular_speed is None
            else abs(angular_speed - previous_angular_speed) / control_period
        )
        stored_velocity = _qualification_finite_vector(row, "external_linear_velocity_m_s", 3)
        if not np.allclose(stored_velocity, velocity, rtol=0.0, atol=1e-12):
            raise RuntimeError("qualification camera linear velocity 重算漂移")
        for name, expected in (
            ("external_position_tracking_error_m", position_error),
            ("external_orientation_tracking_error_rad", orientation_error),
            ("external_linear_speed_m_s", linear_speed),
            ("external_linear_acceleration_m_s2", linear_acceleration),
            ("external_angular_speed_rad_s", angular_speed),
            ("external_angular_acceleration_rad_s2", angular_acceleration),
        ):
            _qualification_close(row.get(name), expected, name)

        settle_evidence = bool(
            position_error <= safety["camera_position_tracking_tolerance_m"]
            and orientation_error <= safety["camera_orientation_tracking_tolerance_rad"]
            and linear_speed <= safety["settled_linear_velocity_max_m_s"]
            and angular_speed <= safety["settled_angular_velocity_max_rad_s"]
        )
        stationary = state in {
            ExternalCameraMotionState.HOME_ANCHOR.value,
            ExternalCameraMotionState.SETTLE_AT_VIEW.value,
            ExternalCameraMotionState.COLLECT.value,
            ExternalCameraMotionState.VERIFY_HOME_AND_ARM_HOLD.value,
        }
        if stationary and settle_evidence:
            settle_streak += 1
        else:
            settle_streak = 0
        settled = settle_streak >= safety["required_consecutive_settled_ticks"]
        write_eligible = _g0.measurement_write_eligible(
            ExternalCameraMotionState(state), settled=settled
        )
        if (
            row.get("settle_evidence_passed") is not settle_evidence
            or row.get("settle_streak") != settle_streak
            or row.get("settled") is not settled
            or row.get("measurement_write_eligible") is not write_eligible
        ):
            raise RuntimeError("qualification settle/write eligibility 重算漂移")

        arm_anchor = _qualification_finite_vector(row, "arm_anchor_q_rad", 7)
        arm_current = _qualification_finite_vector(row, "arm_current_q_rad", 7)
        if index and not np.array_equal(
            arm_anchor,
            _qualification_finite_vector(rows[0], "arm_anchor_q_rad", 7),
        ):
            raise RuntimeError("qualification arm anchor witness 漂移")
        arm_drift = float(np.max(np.abs(arm_current - arm_anchor)))
        tcp_anchor = _qualification_finite_matrix(row, "tcp_anchor_world")
        tcp_current = _qualification_finite_matrix(row, "tcp_current_world")
        if index and not np.array_equal(
            tcp_anchor,
            _qualification_finite_matrix(rows[0], "tcp_anchor_world"),
        ):
            raise RuntimeError("qualification TCP anchor witness 漂移")
        tcp_position_drift = float(np.linalg.norm(tcp_current[:3, 3] - tcp_anchor[:3, 3]))
        tcp_orientation_drift = _g0.rotation_angular_distance_rad(
            tcp_anchor[:3, :3], tcp_current[:3, :3]
        )
        finger_positions = _qualification_finite_vector(row, "finger_joint_positions_m", 2)
        finger_contact = row.get("finger_object_contact_force_n")
        if (
            not isinstance(finger_contact, (int, float))
            or isinstance(finger_contact, bool)
            or not math.isfinite(float(finger_contact))
            or float(finger_contact) < 0.0
        ):
            raise RuntimeError("qualification finger contact witness 漂移")
        link_witnesses = row.get("robot_object_contact_by_link")
        if not isinstance(link_witnesses, list) or not link_witnesses:
            raise RuntimeError("qualification full robot-object contact witness 缺失")
        link_names: list[str] = []
        link_magnitudes: list[float] = []
        for link_witness in link_witnesses:
            if not isinstance(link_witness, dict) or set(link_witness) != {
                "link_name",
                "force_xyz_n",
                "force_magnitude_n",
            }:
                raise RuntimeError("qualification robot-object link witness schema 漂移")
            link_name = link_witness["link_name"]
            force = np.asarray(link_witness["force_xyz_n"], dtype=np.float64)
            if (
                not isinstance(link_name, str)
                or not link_name
                or force.shape != (3,)
                or not np.isfinite(force).all()
            ):
                raise RuntimeError("qualification robot-object link witness value 漂移")
            magnitude = float(np.linalg.norm(force))
            _qualification_close(
                link_witness.get("force_magnitude_n"),
                magnitude,
                "robot_object_link_force_magnitude_n",
            )
            link_names.append(link_name)
            link_magnitudes.append(magnitude)
        if len(set(link_names)) != len(link_names):
            raise RuntimeError("qualification robot-object link name 重复")
        if contact_link_names is None:
            contact_link_names = tuple(link_names)
        elif tuple(link_names) != contact_link_names:
            raise RuntimeError("qualification robot-object link order 漂移")
        robot_object_contact = max(link_magnitudes)
        _qualification_close(
            row.get("robot_object_contact_force_n"),
            robot_object_contact,
            "robot_object_contact_force_n",
        )
        if robot_object_contact > capture_safety["maximum_robot_object_contact_force_n"]:
            raise RuntimeError("qualification full robot-object contact safety gate 未通过")
        for name, expected in (
            ("arm_joint_max_drift_rad", arm_drift),
            ("tcp_position_drift_m", tcp_position_drift),
            ("tcp_orientation_drift_rad", tcp_orientation_drift),
            ("minimum_finger_joint_position_m", float(np.min(finger_positions))),
        ):
            _qualification_close(row.get(name), expected, name)
        previous_position = actual_position
        previous_quaternion = actual_quaternion
        previous_velocity = velocity
        previous_angular_speed = angular_speed

    expected_roles = {
        "home_before": 0,
        "alternate": _FINAL_COLLECT_FRAME_INDEX,
        "home_after": 91,
    }
    if set(rgb_images) != set(expected_roles):
        raise RuntimeError("qualification route RGB role schema 漂移")
    rgb_arrays: dict[str, np.ndarray] = {}
    for role, row_index in expected_roles.items():
        image = np.asarray(rgb_images[role])
        if image.dtype != np.uint8 or image.shape != (128, 128, 3):
            raise RuntimeError("qualification route RGB witness shape/dtype 漂移")
        image = np.ascontiguousarray(image)
        if hashlib.sha256(image.tobytes()).hexdigest() != rows[row_index].get("rgb_sha256"):
            raise RuntimeError("qualification route RGB pixel/ledger SHA 漂移")
        rgb_arrays[role] = image
    alternate_diff = float(
        np.mean(
            np.abs(
                rgb_arrays["alternate"].astype(np.float64)
                - rgb_arrays["home_before"].astype(np.float64)
            )
        )
    )
    return_diff = float(
        np.mean(
            np.abs(
                rgb_arrays["home_after"].astype(np.float64)
                - rgb_arrays["home_before"].astype(np.float64)
            )
        )
    )
    target_row = rows[_FINAL_COLLECT_FRAME_INDEX]
    requested_offset = float(np.linalg.norm(target_offsets))
    inverse_orientation = FrontCameraOrientationMode(
        orientation_id=orientation.orientation_id,
        yaw_offset_rad=-orientation.yaw_offset_rad,
        pitch_offset_rad=-orientation.pitch_offset_rad,
        roll_offset_rad=-orientation.roll_offset_rad,
    )
    target_command_quaternion = _qualification_finite_vector(
        target_row, "commanded_external_quaternion_sapien", 4
    )
    nominal_quaternion = _g0.compose_camera_orientation_wxyz(
        target_command_quaternion, inverse_orientation
    )
    target_actual_quaternion = _qualification_finite_vector(
        target_row, "actual_external_quaternion_sapien", 4
    )
    actual_offset = _g0.quaternion_angular_distance_rad(
        nominal_quaternion, target_actual_quaternion
    )
    target_error = _g0.quaternion_angular_distance_rad(
        target_command_quaternion, target_actual_quaternion
    )
    return _g0.recompute_route_gates(
        rows,
        config=g0c_config,
        home_position_world_m=home.position_world_m,
        home_quaternion_sapien=rows[0]["commanded_external_quaternion_sapien"],
        alternate_orientation_id=orientation.orientation_id,
        requested_orientation_offset_rad=requested_offset,
        actual_orientation_offset_rad=actual_offset,
        alternate_target_orientation_error_rad=target_error,
        alternate_rgb_mean_abs_difference=alternate_diff,
        return_home_rgb_mean_abs_difference=return_diff,
    )


def validate_qualification_route_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    alternate_index: int,
    alternate_viewpoint_id: str,
    summary: Mapping[str, Any],
    g0c_config: Mapping[str, Any],
    capture_safety: Mapping[str, Any],
    rgb_images: Mapping[str, np.ndarray],
) -> dict[str, int]:
    """从 raw witness 重验一条 G0C-v2 route、SafeHold 与完整 gate。"""

    if alternate_viewpoint_id != FRONT_ALTERNATE_IDS[alternate_index]:
        raise RuntimeError("qualification route alternate order 漂移")
    expected_states = tuple(state for state, count in _ROUTE_STATE_PATTERN for _ in range(count))
    if len(rows) != len(expected_states) or len(rows) != 92:
        raise RuntimeError("qualification route ledger 必须精确为 92 帧")
    if tuple(row.get("camera_motion_state") for row in rows) != expected_states:
        raise RuntimeError("qualification route state pattern 漂移")
    if [row.get("frame_index") for row in rows] != list(range(92)):
        raise RuntimeError("qualification route frame_index 不连续")
    if [row.get("control_tick") for row in rows] != list(range(92)):
        raise RuntimeError("qualification route control_tick 不连续")
    expected_provider_count = 1 + int(alternate_index == 0)
    if (
        summary.get("version") != E018_P1_G2C_QUALIFICATION_EXECUTION_VERSION
        or summary.get("episode_id") != rows[0].get("episode_id")
        or summary.get("seed") != seed
        or summary.get("alternate_viewpoint_id") != alternate_viewpoint_id
        or summary.get("alternate_orientation_id")
        != rows[_FINAL_COLLECT_FRAME_INDEX].get("target_orientation_id")
        or summary.get("yaw_offset_rad")
        != rows[_FINAL_COLLECT_FRAME_INDEX].get("commanded_yaw_offset_rad")
        or summary.get("pitch_offset_rad")
        != rows[_FINAL_COLLECT_FRAME_INDEX].get("commanded_pitch_offset_rad")
        or summary.get("roll_offset_rad")
        != rows[_FINAL_COLLECT_FRAME_INDEX].get("commanded_roll_offset_rad")
        or summary.get("frame_count") != 92
        or summary.get("control_hz") != 20
        or summary.get("motion_ticks_each_leg") != 40
        or summary.get("route_simulated_duration_s") != 91 / 20
        or summary.get("passed") is not True
        or summary.get("status") != "passed"
        or summary.get("test_split_status") != "prohibited-unread"
        or summary.get("provider_forward_count") != expected_provider_count
        or summary.get("privileged_capture_count") != expected_provider_count
        or summary.get("memory_write_count") != 0
        or summary.get("formal_claim_allowed") is not False
        or summary.get("qualification_classification")
        not in {
            QUALIFICATION_CLASSIFICATION_FORMAL,
            QUALIFICATION_CLASSIFICATION_SMOKE,
        }
        or summary.get("offline_segmentation_diagnostics") is not False
    ):
        raise RuntimeError("qualification route summary identity/permission 漂移")
    for row in rows:
        if (
            set(row) != _QUALIFICATION_ROUTE_ROW_KEYS
            or row.get("arm_owner") != "SAFE_HOLD"
            or row.get("gripper_owner") != "SAFE_HOLD_OPEN"
            or row.get("arm_motion_command_max_abs") != 0.0
            or row.get("gripper_hold_open_command") != 1.0
            or row.get("memory_write_executed") is not False
            or row.get("is_grasping") is not False
            or row.get("terminated") is not False
            or row.get("truncated") is not False
            or row.get("offline_segmentation_diagnostics") is not None
        ):
            raise RuntimeError("qualification route ownership/permission/safety 漂移")
    if any(
        row.get("measurement_write_eligible") is not False
        for row in rows
        if row.get("camera_motion_state") != ExternalCameraMotionState.COLLECT.value
    ):
        raise RuntimeError("qualification 非 COLLECT 帧错误获得 write eligibility")
    if any(
        row.get("measurement_write_eligible") is not True or row.get("settled") is not True
        for row in rows[45:48]
    ):
        raise RuntimeError("qualification COLLECT settle/write eligibility 漂移")

    gates = _recompute_qualification_route_witnesses(
        rows,
        seed=seed,
        alternate_viewpoint_id=alternate_viewpoint_id,
        g0c_config=g0c_config,
        capture_safety=capture_safety,
        rgb_images=rgb_images,
    )
    _assert_structured_derived_equal(
        summary.get("gates"),
        gates,
        tolerance=1e-7,
        path="qualification_route.gates",
    )
    if any(gate["passed"] is not True for gate in gates.values()):
        raise RuntimeError("qualification route raw-witness safety gate 未通过")
    diagnostics = summary.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != {
        "alternate_rgb_mean_abs_difference",
        "return_home_rgb_mean_abs_difference",
        "alternate_displacement_m",
        "requested_orientation_offset_rad",
        "actual_orientation_offset_rad",
        "alternate_target_orientation_error_rad",
        "object_visible_pixels_collect_min",
        "goal_visible_pixels_collect_min",
        "rgb_numeric_evidence_source",
    }:
        raise RuntimeError("qualification route diagnostics schema 漂移")
    expected_diagnostics = {
        "alternate_rgb_mean_abs_difference": gates["rendered_view_changed"]["actual"],
        "return_home_rgb_mean_abs_difference": gates["return_home_render_recovered"]["actual"],
        "alternate_displacement_m": gates["actual_dynamic_pose_observed"]["actual"][
            "alternate_displacement_m"
        ],
        "requested_orientation_offset_rad": gates["alternate_orientation_target_reached"]["actual"][
            "requested_offset_rad"
        ],
        "actual_orientation_offset_rad": gates["alternate_orientation_target_reached"]["actual"][
            "actual_offset_rad"
        ],
        "alternate_target_orientation_error_rad": gates["alternate_orientation_target_reached"][
            "actual"
        ]["target_error_rad"],
        "object_visible_pixels_collect_min": None,
        "goal_visible_pixels_collect_min": None,
        "rgb_numeric_evidence_source": ("three-public-png-pixel-witnesses-recomputed/v1"),
    }
    _assert_structured_derived_equal(
        diagnostics,
        expected_diagnostics,
        tolerance=1e-7,
        path="qualification_route.diagnostics",
    )
    return {
        "camera_pose_set_count": 97,
        "moving_interpolation_command_count": 80,
        "safe_hold_open_step_count": 96,
        "ledger_frame_count": 92,
        "provider_scored_home_frame_count": int(alternate_index == 0),
        "provider_scored_alternate_frame_count": 1,
        "provider_scored_frame_count": 1 + int(alternate_index == 0),
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _deployable_physical_state(
    *,
    base_env: Any,
    spec: RobotSpec,
    maximum_projection_error: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 provider 必需状态；有意不读取 robot-object oracle contact。"""

    robot = base_env.agent.robot
    qpos = np.asarray(_g0._numpy(robot.get_qpos()))
    qvel = np.asarray(_g0._numpy(robot.get_qvel()))
    if qpos.shape != (1, 9) or qvel.shape != (1, 9):
        raise RuntimeError("qualification 只支持单环境 Panda 9-DoF state")
    joint_names = tuple(joint.name for joint in robot.active_joints)
    proprio = FrankaObservationAdapter(spec).from_maniskill(qpos[0], qvel[0], joint_names)
    world_from_base, _ = _single_rigid(
        robot.pose,
        "qualification world_from_robot_base",
        maximum_projection_error=maximum_projection_error,
    )
    world_from_tcp, _ = _single_rigid(
        base_env.agent.tcp_pose,
        "qualification world_from_tcp",
        maximum_projection_error=maximum_projection_error,
    )
    base_from_tcp = validate_se3(
        invert_se3(world_from_base, "qualification world_from_robot_base") @ world_from_tcp,
        "qualification base_from_tcp",
    )
    return proprio, base_from_tcp, _finger_force_n(base_env)


def build_qualification_deployable_capture(
    *,
    identity: Mapping[str, Any],
    motion_row: Mapping[str, Any],
    rgb: np.ndarray,
    base_env: Any,
    spec: RobotSpec,
    proprio_normalizer: Any,
    finger_force_normalizer: Any,
    data_config: Mapping[str, Any],
) -> dict[str, Any]:
    """从 RGB/pose/robot state 构建单帧输入；接口不接 observation/segmentation。"""

    required_identity = {
        "seed",
        "sample_index",
        "viewpoint_id",
        "frame_role",
        "route_alternate_index",
        "route_alternate_viewpoint_id",
        "route_frame_index",
        "row_index",
    }
    _require_exact_keys(dict(identity), required_identity, "qualification scored identity")
    image = np.asarray(rgb)
    expected_image_shape = tuple(data_config["environment"]["image_shape_hwc"])
    if (
        image.shape != expected_image_shape
        or image.dtype != np.uint8
        or _array_sha256(image) != motion_row.get("rgb_sha256")
    ):
        raise RuntimeError("qualification RGB shape/dtype/ledger SHA 漂移")
    capture_config = data_config["capture"]
    maximum_projection = float(capture_config["maximum_rotation_projection_error_frobenius"])
    proprio, base_from_tcp, finger_force = _deployable_physical_state(
        base_env=base_env,
        spec=spec,
        maximum_projection_error=maximum_projection,
    )
    base_from_camera, camera_projection_error = _single_rigid(
        motion_row["actual_base_from_external_camera_cv"],
        "qualification actual_base_from_external_camera_cv",
        maximum_projection_error=maximum_projection,
    )
    intrinsic = np.asarray(motion_row["external_intrinsic_cv"], dtype=np.float64)
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
    ):
        raise RuntimeError("qualification intrinsic_cv 漂移")
    structured_state = build_precision_camera_role_state(
        spec=spec,
        proprio_normalizer=proprio_normalizer,
        finger_force_normalizer=finger_force_normalizer,
        physical_proprio=proprio,
        base_from_tcp=base_from_tcp,
        base_from_camera_cv=base_from_camera,
        finger_force_n=finger_force,
    )
    finger_force_max = float(np.max(finger_force))
    if not math.isclose(
        finger_force_max,
        float(motion_row["finger_object_contact_force_n"]),
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise RuntimeError("qualification finger-force 与 motion ledger 漂移")
    rgb_timestamp = float(motion_row["external_rgb_timestamp_s"])
    pose_timestamp = float(motion_row["external_pose_timestamp_s"])
    safety = {
        "eligible_capture": bool(
            motion_row.get("settled") is True
            and identity["route_frame_index"] in (0, _FINAL_COLLECT_FRAME_INDEX)
        ),
        "finger_force_n": finger_force.astype(float).tolist(),
        "finger_force_valid": True,
        "raw_gripper_opening_ratio": float(proprio[-1]),
        "arm_joint_drift_rad": float(motion_row["arm_joint_max_drift_rad"]),
        "tcp_position_drift_m": float(motion_row["tcp_position_drift_m"]),
        "tcp_orientation_drift_rad": float(motion_row["tcp_orientation_drift_rad"]),
        "rgb_timestamp_s": rgb_timestamp,
        "pose_timestamp_s": pose_timestamp,
        "camera_position_tracking_error_m": float(motion_row["external_position_tracking_error_m"]),
        "camera_orientation_tracking_error_rad": float(
            motion_row["external_orientation_tracking_error_rad"]
        ),
        "rotation_projection_error_frobenius": camera_projection_error,
    }
    input_identity = {
        **dict(identity),
        "rgb_sha256": _array_sha256(image),
        "physical_proprio_sha256": _array_sha256(proprio),
        "structured_state_sha256": _array_sha256(structured_state),
        "base_from_tcp_sha256": _array_sha256(base_from_tcp),
        "base_from_external_camera_cv_sha256": _array_sha256(base_from_camera),
        "external_intrinsic_cv_sha256": _array_sha256(intrinsic),
        "finger_force_n_sha256": _array_sha256(finger_force),
        "rgb_timestamp_s": rgb_timestamp,
        "pose_timestamp_s": pose_timestamp,
    }
    return {
        "identity": dict(identity),
        "input_sha256": canonical_sha256(input_identity),
        "rgb_external": np.ascontiguousarray(image),
        "physical_proprio": proprio.astype(np.float32, copy=False),
        "structured_state": structured_state.astype(np.float32, copy=False),
        "geometric_motion": np.asarray(capture_config["geometric_motion_value"], dtype=np.float32),
        "base_from_tcp": base_from_tcp.astype(np.float64, copy=False),
        "base_from_external_camera_cv": base_from_camera.astype(np.float64, copy=False),
        "external_intrinsic_cv": intrinsic,
        "deployable_safety": safety,
    }


def qualification_deployable_safe(
    safety: Mapping[str, Any], *, qualification_config: Mapping[str, Any]
) -> bool:
    """只用 deployable state 重算 free-static safety。"""

    limits = qualification_config["capture_safety"]
    force = np.asarray(safety["finger_force_n"], dtype=np.float64)
    rgb_timestamp = float(safety["rgb_timestamp_s"])
    pose_timestamp = float(safety["pose_timestamp_s"])
    return bool(
        safety["eligible_capture"] is True
        and safety["finger_force_valid"] is True
        and force.shape == (2,)
        and np.isfinite(force).all()
        and np.all(force >= 0.0)
        and float(force.max()) <= limits["maximum_finger_force_n"]
        and float(safety["raw_gripper_opening_ratio"])
        >= limits["minimum_raw_gripper_opening_ratio"]
        and float(safety["arm_joint_drift_rad"]) <= limits["maximum_arm_joint_drift_rad"]
        and float(safety["tcp_position_drift_m"]) <= limits["maximum_tcp_position_drift_m"]
        and float(safety["tcp_orientation_drift_rad"])
        <= limits["maximum_tcp_orientation_drift_rad"]
        and abs(rgb_timestamp - pose_timestamp) <= limits["maximum_rgb_pose_skew_s"]
        and float(safety["camera_position_tracking_error_m"])
        <= limits["maximum_camera_position_tracking_error_m"]
        and float(safety["camera_orientation_tracking_error_rad"])
        <= limits["maximum_camera_orientation_tracking_error_rad"]
        and float(safety["rotation_projection_error_frobenius"])
        <= limits["maximum_rotation_projection_error_frobenius"]
    )


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _recompute_prediction_geometry(
    *,
    normalized_uv: Any,
    intrinsic_cv: Any,
    base_from_camera_cv: Any,
    sigma_xy_px: Any,
    plane_base_z_m: float,
) -> dict[str, Any]:
    """从公开的 UV、相机几何与 sigma 重算 world point/covariance。"""

    uv = np.asarray(normalized_uv, dtype=np.float64)
    intrinsic = np.asarray(intrinsic_cv, dtype=np.float64)
    transform = np.asarray(base_from_camera_cv, dtype=np.float64)
    sigma = np.asarray(sigma_xy_px, dtype=np.float64)
    if (
        uv.shape != (2,)
        or not np.isfinite(uv).all()
        or np.any(uv < 0.0)
        or np.any(uv > 1.0)
        or intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
        or transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or sigma.shape != (2,)
        or not np.isfinite(sigma).all()
        or np.any(sigma < 0.0)
    ):
        raise RuntimeError("qualification prediction geometry primitive shape/value 漂移")
    transform = validate_se3(transform, "qualification prediction base_from_external_camera_cv")
    try:
        geometry = geometry_conditioning(
            normalized_uv=uv,
            intrinsic_cv=intrinsic,
            base_from_camera_cv=transform,
            image_size_hw=(128, 128),
            plane_base_z_m=float(plane_base_z_m),
        )
        position = np.asarray(geometry["predicted_world_point_base_m"], dtype=np.float64)
        covariance = _measurement_covariance(geometry["local_jacobian_xy_m_per_px"], sigma)
    except (ValueError, np.linalg.LinAlgError):
        return {
            "geometry_valid": False,
            "predicted_object_position_base_m": None,
            "raw_covariance_base_m2": None,
        }
    return {
        "geometry_valid": True,
        "predicted_object_position_base_m": position.astype(float).tolist(),
        "raw_covariance_base_m2": covariance.astype(float).tolist(),
    }


def _validate_prediction_geometry_same_source(
    prediction: Mapping[str, Any],
    *,
    intrinsic_cv: Any,
    base_from_camera_cv: Any,
    plane_base_z_m: float,
    tolerance: float,
) -> dict[str, Any]:
    recomputed = _recompute_prediction_geometry(
        normalized_uv=prediction.get("predicted_object_normalized_uv"),
        intrinsic_cv=intrinsic_cv,
        base_from_camera_cv=base_from_camera_cv,
        sigma_xy_px=prediction.get("object_sigma_xy_px"),
        plane_base_z_m=plane_base_z_m,
    )
    for name, expected in recomputed.items():
        actual = prediction.get(name)
        if type(expected) is bool or expected is None:
            matches = actual is expected
        else:
            try:
                left = np.asarray(actual, dtype=np.float64)
                right = np.asarray(expected, dtype=np.float64)
            except (TypeError, ValueError):
                matches = False
            else:
                matches = left.shape == right.shape and bool(
                    np.allclose(left, right, rtol=0.0, atol=tolerance)
                )
        if not matches:
            raise RuntimeError(f"qualification prediction geometry/covariance 同源漂移: {name}")
    return recomputed


def _validate_qualification_prediction_static_identity(
    prediction: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    classification: str,
    committed: bool,
) -> None:
    """验证 provider/checkpoint/权限身份；数值 mechanics 由独立重算负责。"""

    expected_keys = (
        _COMMITTED_QUALIFICATION_PREDICTION_KEYS if committed else _QUALIFICATION_PREDICTION_KEYS
    )
    _require_exact_keys(dict(prediction), expected_keys, "qualification prediction")
    checkpoint = config["parents"]["selected_checkpoint"]
    expected_split = (
        "qualification"
        if classification == QUALIFICATION_CLASSIFICATION_FORMAL
        else "engineering_smoke"
    )
    safety = _require_exact_keys(
        prediction.get("deployable_safety"),
        _DEPLOYABLE_SAFETY_KEYS,
        "qualification prediction deployable_safety",
    )
    numeric_probabilities = (
        "object_visibility_probability",
        "goal_visibility_probability",
        "projection_validity_probability",
        "object_normalized_entropy",
        "object_mask_probability_at_prediction",
        "goal_mask_probability_at_prediction",
    )
    try:
        probability_values = [float(prediction[name]) for name in numeric_probabilities]
        goal_uv = np.asarray(prediction["predicted_goal_normalized_uv"], dtype=np.float64)
        force = np.asarray(safety["finger_force_n"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("qualification prediction primitive 类型漂移") from error
    if (
        prediction.get("version") != E018_P1_G2C_QUALIFICATION_EXECUTION_VERSION
        or prediction.get("phase") != "dynamic-deployable-prediction-before-privileged-capture/v1"
        or prediction.get("classification") != classification
        or prediction.get("candidate_id") != checkpoint["candidate_id"]
        or prediction.get("epoch") != checkpoint["epoch"]
        or prediction.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]
        or prediction.get("checkpoint_parameter_sha256") != checkpoint["parameter_state_sha256"]
        or prediction.get("checkpoint_provenance_sha256") != checkpoint["provenance_sha256"]
        or prediction.get("checkpoint_model_config_sha256") != checkpoint["model_config_sha256"]
        or prediction.get("split") != expected_split
        or type(prediction.get("row_index")) is not int
        or prediction.get("batch_index") != prediction.get("row_index")
        or prediction.get("batch_offset") != 0
        or type(prediction.get("seed")) is not int
        or type(prediction.get("sample_index")) is not int
        or prediction.get("viewpoint_id") not in QUALIFICATION_VIEW_ORDER
        or not _is_sha256(prediction.get("input_sha256"))
        or goal_uv.shape != (2,)
        or not np.isfinite(goal_uv).all()
        or np.any(goal_uv < 0.0)
        or np.any(goal_uv > 1.0)
        or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probability_values)
        or type(prediction.get("predicted_observable")) is not bool
        or type(prediction.get("geometry_valid")) is not bool
        or prediction.get("memory_write_allowed") is not False
        or prediction.get("memory_write_executed") is not False
        or prediction.get("actuation_allowed") is not False
        or prediction.get("test_data_read") is not False
        or type(safety.get("eligible_capture")) is not bool
        or safety.get("finger_force_valid") is not True
        or force.shape != (2,)
        or not np.isfinite(force).all()
        or np.any(force < 0.0)
    ):
        raise RuntimeError("qualification prediction static identity/permission 漂移")
    if committed:
        if (
            prediction.get("previous_prediction_sha256") is not None
            and not _is_sha256(prediction.get("previous_prediction_sha256"))
        ) or (
            type(prediction.get("prediction_write_started_at_unix_ns")) is not int
            or prediction["prediction_write_started_at_unix_ns"] <= 0
            or not _is_sha256(prediction.get("prediction_sha256"))
        ):
            raise RuntimeError("qualification committed prediction identity 漂移")


def finalize_qualification_prediction(
    raw_prediction: Mapping[str, Any],
    *,
    capture: Mapping[str, Any],
    calibration: Mapping[str, Any],
    data_config: Mapping[str, Any],
    qualification_config: Mapping[str, Any],
    classification: str,
) -> dict[str, Any]:
    """应用 D046 covariance scale/threshold；整个函数不接 GT。"""

    identity = capture["identity"]
    _require_exact_keys(
        dict(raw_prediction),
        _PROVIDER_RAW_PREDICTION_KEYS,
        "qualification raw provider prediction",
    )
    if (
        raw_prediction.get("row_index") != identity.get("row_index")
        or raw_prediction.get("seed") != identity.get("seed")
        or raw_prediction.get("sample_index") != identity.get("sample_index")
        or raw_prediction.get("viewpoint_id") != identity.get("viewpoint_id")
        or raw_prediction.get("input_sha256") != capture.get("input_sha256")
    ):
        raise RuntimeError("qualification provider output/input identity 漂移")
    _require_exact_keys(
        dict(calibration),
        {"scale_factor", "write_threshold"},
        "qualification per-view calibration",
    )
    scale = float(calibration["scale_factor"])
    threshold = float(calibration["write_threshold"])
    if not math.isfinite(scale) or scale < 1.0 or not 0.0 <= threshold <= 1.0:
        raise RuntimeError("qualification calibration scale/threshold 非法")
    _validate_prediction_geometry_same_source(
        raw_prediction,
        intrinsic_cv=capture["external_intrinsic_cv"],
        base_from_camera_cv=capture["base_from_external_camera_cv"],
        plane_base_z_m=float(qualification_config["capture_safety"]["object_center_base_z_m"]),
        tolerance=float(qualification_config["qualification"]["metric_float_recompute_tolerance"]),
    )
    evidence = ObjectWriteEvidence(
        visibility_probability=float(raw_prediction["object_visibility_probability"]),
        projection_validity_probability=float(raw_prediction["projection_validity_probability"]),
        object_mask_probability=float(raw_prediction["object_mask_probability_at_prediction"]),
        goal_mask_probability=float(raw_prediction["goal_mask_probability_at_prediction"]),
        normalized_entropy=float(raw_prediction["object_normalized_entropy"]),
        radial_sigma_px=float(np.linalg.norm(raw_prediction["object_sigma_xy_px"])),
        geometry_valid=bool(raw_prediction["geometry_valid"]),
    )
    if (
        type(raw_prediction["predicted_observable"]) is not bool
        or raw_prediction["predicted_observable"]
        != (float(raw_prediction["object_visibility_probability"]) >= 0.5)
        or not math.isclose(
            float(raw_prediction["write_score"]),
            evidence.score,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise RuntimeError("qualification provider observable/write-score 派生字段漂移")
    raw_covariance_value = raw_prediction["raw_covariance_base_m2"]
    if raw_covariance_value is None:
        calibrated_covariance = None
        maximum_std = None
    else:
        raw_covariance = np.asarray(raw_covariance_value, dtype=np.float64)
        if (
            raw_covariance.shape != (3, 3)
            or not np.isfinite(raw_covariance).all()
            or not np.allclose(raw_covariance, raw_covariance.T, rtol=0.0, atol=1e-12)
            or float(np.linalg.eigvalsh(raw_covariance).min()) < -1e-12
        ):
            raise RuntimeError("qualification raw covariance shape/finite/PSD 漂移")
        calibrated_covariance = raw_covariance * scale
        maximum_std = float(
            np.sqrt(max(0.0, float(np.linalg.eigvalsh(calibrated_covariance).max())))
        )
    deployable_safe = qualification_deployable_safe(
        capture["deployable_safety"], qualification_config=qualification_config
    )
    structurally_eligible = bool(
        raw_prediction["predicted_observable"]
        and evidence.structurally_eligible
        and deployable_safe
    )
    accepted = bool(
        structurally_eligible
        and evidence.score >= threshold
        and maximum_std is not None
        and maximum_std
        <= qualification_config["qualification"]["maximum_calibrated_position_std_m"]
    )
    result = {
        **dict(raw_prediction),
        "version": E018_P1_G2C_QUALIFICATION_EXECUTION_VERSION,
        "phase": "dynamic-deployable-prediction-before-privileged-capture/v1",
        "classification": classification,
        "frame_role": identity["frame_role"],
        "route_alternate_index": identity["route_alternate_index"],
        "route_alternate_viewpoint_id": identity["route_alternate_viewpoint_id"],
        "route_frame_index": identity["route_frame_index"],
        "external_intrinsic_cv": np.asarray(
            capture["external_intrinsic_cv"], dtype=np.float64
        ).tolist(),
        "base_from_external_camera_cv": np.asarray(
            capture["base_from_external_camera_cv"], dtype=np.float64
        ).tolist(),
        "deployable_safety": dict(capture["deployable_safety"]),
        "deployable_free_static_safe": deployable_safe,
        "object_write_structurally_eligible": evidence.structurally_eligible,
        "structurally_eligible": structurally_eligible,
        "calibration_scale_factor": scale,
        "calibrated_covariance_base_m2": (
            None if calibrated_covariance is None else calibrated_covariance.astype(float).tolist()
        ),
        "calibrated_position_std_max_m": maximum_std,
        "write_threshold": threshold,
        "write_accepted": accepted,
        "memory_write_allowed": False,
        "memory_write_executed": False,
        "actuation_allowed": False,
        "test_data_read": False,
    }
    assert_qualification_prediction_deployable_only(result)
    _validate_qualification_prediction_static_identity(
        result,
        config=qualification_config,
        classification=classification,
        committed=False,
    )
    return result


class QualificationProvider:
    """D046 selected checkpoint 的单帧、object-only qualification provider。"""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        qualification_config: Mapping[str, Any],
        data_config: Mapping[str, Any],
        classification: str,
    ) -> None:
        import torch

        from robot_vla.precision.checkpoint import (
            PrecisionCheckpointRole,
            load_precision_checkpoint,
        )

        if classification not in {
            QUALIFICATION_CLASSIFICATION_FORMAL,
            QUALIFICATION_CLASSIFICATION_SMOKE,
        }:
            raise ValueError("qualification classification 未冻结")
        if not torch.cuda.is_available():
            raise RuntimeError("qualification provider 要求 CUDA")
        expected = qualification_config["parents"]["selected_checkpoint"]
        loaded = load_precision_checkpoint(
            checkpoint_path,
            expected_checkpoint_sha256=expected["checkpoint_sha256"],
            expected_provenance_sha256=expected["provenance_sha256"],
            expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
        )
        if (
            loaded.receipt.parameter_state_sha256 != expected["parameter_state_sha256"]
            or loaded.receipt.model_config_sha256 != expected["model_config_sha256"]
        ):
            raise RuntimeError("qualification selected checkpoint semantic identity 漂移")
        self._torch = torch
        self.model = loaded.model.to(torch.device("cuda"))
        self.model.eval()
        self.checkpoint_identity = dict(expected)
        self.qualification_config = qualification_config
        self.data_config = data_config
        self.classification = classification
        self.forward_count = 0
        self.destroyed = False

    def predict(self, capture: Mapping[str, Any]) -> dict[str, Any]:
        if self.destroyed:
            raise RuntimeError("qualification provider context 已销毁")
        torch = self._torch
        image = np.asarray(capture["rgb_external"], dtype=np.uint8)
        state = np.asarray(capture["structured_state"], dtype=np.float32)
        motion = np.asarray(capture["geometric_motion"], dtype=np.float32)
        if image.shape != (128, 128, 3) or state.ndim != 1 or motion.shape != (4,):
            raise RuntimeError("qualification provider input shape 漂移")
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)[None]
            / np.float32(255.0)
        ).to(torch.device("cuda"))
        state_tensor = torch.from_numpy(state[None]).to(torch.device("cuda"))
        motion_tensor = torch.from_numpy(motion[None]).to(torch.device("cuda"))
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True),
        ):
            output = self.model(image_tensor, state_tensor, motion_tensor)
        identity = capture["identity"]
        split = (
            "qualification"
            if self.classification == QUALIFICATION_CLASSIFICATION_FORMAL
            else "engineering_smoke"
        )
        sample = {
            "model_inputs": {
                "rgb_external": image,
                "structured_state": state,
                "geometric_motion": motion,
            },
            "capture": {
                "seed": identity["seed"],
                "split": split,
                "sample_index": identity["sample_index"],
                "viewpoint_id": identity["viewpoint_id"],
                "input_sha256": capture["input_sha256"],
                "external_intrinsic_cv": capture["external_intrinsic_cv"],
                "base_from_external_camera_cv": capture["base_from_external_camera_cv"],
            },
        }
        raw_rows = _prediction_rows_for_batch(
            samples=[sample],
            output=output,
            candidate_id=self.checkpoint_identity["candidate_id"],
            epoch=self.checkpoint_identity["epoch"],
            checkpoint_identity=self.checkpoint_identity,
            global_start=identity["row_index"],
            batch_index=identity["row_index"],
        )
        if len(raw_rows) != 1:
            raise RuntimeError("qualification provider 必须精确返回一行")
        self.forward_count += 1
        return finalize_qualification_prediction(
            raw_rows[0],
            capture=capture,
            calibration=self.qualification_config["calibration"]["values"][
                identity["viewpoint_id"]
            ],
            data_config=self.data_config,
            qualification_config=self.qualification_config,
            classification=self.classification,
        )

    def destroy(self) -> None:
        if self.destroyed:
            return
        torch = self._torch
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.destroyed = True


def capture_qualification_object_label(
    *,
    observation: Mapping[str, Any],
    base_env: Any,
    prediction: Mapping[str, Any],
    data_config: Mapping[str, Any],
) -> dict[str, Any]:
    """commit 后读取 object-only simulator GT；不读取 goal actor/position/mask。"""

    camera_uid = data_config["environment"]["external_camera_uid"]
    sensor = observation["sensor_data"][camera_uid]
    segmentation = np.asarray(_g0._numpy(sensor["segmentation"]))
    if segmentation.ndim != 4 or segmentation.shape[:3] != (1, 128, 128):
        raise RuntimeError("qualification object segmentation schema 漂移")
    actor_ids = np.asarray(segmentation[0, ..., 0])
    if not np.issubdtype(actor_ids.dtype, np.integer):
        raise RuntimeError("qualification object actor-id channel 必须是整数")
    object_actor_id = int(_g0._numpy(base_env.cube.per_scene_id).reshape(-1)[0])
    object_mask = np.asarray(actor_ids == object_actor_id, dtype=np.bool_)
    maximum_projection = float(
        data_config["capture"]["maximum_rotation_projection_error_frobenius"]
    )
    object_position = _base_point(
        base_env,
        base_env.cube,
        maximum_projection_error=maximum_projection,
    )
    intrinsic = np.asarray(prediction["external_intrinsic_cv"], dtype=np.float64)
    base_from_camera = np.asarray(prediction["base_from_external_camera_cv"], dtype=np.float64)
    projected_uv: np.ndarray | None
    projection_valid = False
    try:
        candidate_uv = project_base_point_to_normalized_uv(
            object_position,
            intrinsic,
            base_from_camera,
            object_mask.shape,
        )
        projection_valid = bool(np.all((candidate_uv >= 0.0) & (candidate_uv <= 1.0)))
        projected_uv = candidate_uv if projection_valid else None
    except ValueError:
        projected_uv = None
    observability = derive_object_observability(
        object_exists=True,
        projection_valid=projection_valid,
        projected_normalized_uv=projected_uv,
        object_mask=object_mask,
        # Object-only qualification intentionally does not read goal identity.
        goal_mask=np.zeros_like(object_mask, dtype=np.bool_),
        legacy_visible=bool(object_mask.any()),
        support_radius_px=int(data_config["capture"]["support_radius_px"]),
    )
    return {
        "gt_object_exists": True,
        "gt_observable": observability.observable,
        "gt_object_position_base_m": object_position.astype(float).tolist(),
        "gt_object_projection_valid": projection_valid,
        "gt_object_projected_normalized_uv": (
            None if projected_uv is None else projected_uv.astype(float).tolist()
        ),
        "gt_object_mask_sha256": _array_sha256(object_mask),
        "gt_object_visible_pixel_count": int(np.count_nonzero(object_mask)),
        "gt_object_observability": observability.to_dict(),
        "is_grasped": _single_bool(
            base_env.agent.is_grasping(base_env.cube), "qualification is_grasped"
        ),
        "robot_object_contact_force_n": _robot_object_contact_force_n(base_env),
        "goal_gt_read_count": 0,
        "test_data_read": False,
    }


def process_qualification_hook_frame(
    *,
    motion_row: Mapping[str, Any],
    rgb: np.ndarray,
    observation: Mapping[str, Any],
    seed: int,
    alternate_index: int,
    alternate_viewpoint_id: str,
    base_env: Any,
    spec: RobotSpec,
    proprio_normalizer: Any,
    finger_force_normalizer: Any,
    data_config: Mapping[str, Any],
    provider: Any,
    journal: "QualificationJournal",
) -> dict[str, Any] | None:
    """route hook 的唯一 provider/GT 入口；非评分帧立即返回。"""

    selected = qualification_scored_frame_identity(
        motion_row,
        seed=seed,
        alternate_index=alternate_index,
        alternate_viewpoint_id=alternate_viewpoint_id,
    )
    if selected is None:
        return None
    identity = {**selected, "row_index": journal.prediction_count}
    capture = build_qualification_deployable_capture(
        identity=identity,
        motion_row=motion_row,
        rgb=rgb,
        base_env=base_env,
        spec=spec,
        proprio_normalizer=proprio_normalizer,
        finger_force_normalizer=finger_force_normalizer,
        data_config=data_config,
    )
    prediction = provider.predict(capture)
    committed, _ = journal.commit_prediction_then_capture_label(
        prediction,
        privileged_getter=lambda: capture_qualification_object_label(
            observation=observation,
            base_env=base_env,
            prediction=prediction,
            data_config=data_config,
        ),
    )
    return committed


def _fsync_parent(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _serialized_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> tuple[str, int]:
    """以 link-no-replace 发布完整 JSON，并返回 raw SHA 与 parent-fsync 后时间。"""

    if path.exists():
        raise FileExistsError(f"拒绝覆盖 immutable qualification artifact: {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    raw = _serialized_json_bytes(value)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
        _fsync_parent(path)
        completed_at_unix_ns = time.time_ns()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(raw).hexdigest(), completed_at_unix_ns


def _atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(_serialized_json_bytes(value).decode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        temporary = None
        _fsync_parent(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class _AppendOnlyJsonl:
    """执行期逐 route fsync，完成后只读并以 raw SHA 冻结。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        _fsync_parent(path)
        self.row_count = 0
        self.frozen = False

    def append(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if self.frozen:
            raise RuntimeError("qualification JSONL 已冻结")
        if self.path.is_symlink() or self.path.stat().st_nlink != 1:
            raise RuntimeError("qualification JSONL file/link 漂移")
        raw = b"".join(
            (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
            for row in rows
        )
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND)
        try:
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("qualification JSONL append 未写入数据")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.row_count += len(rows)

    def freeze(self) -> dict[str, Any]:
        if self.frozen:
            raise RuntimeError("qualification JSONL 不得重复冻结")
        self.frozen = True
        return {
            "row_count": self.row_count,
            "raw_sha256": file_sha256(self.path),
            "size_bytes": self.path.stat().st_size,
        }


_PRIVILEGED_PREDICTION_KEYS = {
    "gt_observable",
    "gt_object_position_base_m",
    "object_mask",
    "keypoint_observable",
    "oracle_safe_measurement",
    "world_xyz_error_m",
    "world_xy_error_vector_m",
    "catastrophic_measurement",
    "robot_object_contact_force_n",
    "is_grasped",
    "segmentation",
    "object_actor_id",
    "goal_actor_id",
    "goal_position_base_m",
}


def assert_qualification_prediction_deployable_only(row: Mapping[str, Any]) -> None:
    """拒绝 prediction commit 中的显式 label/GT 派生字段。"""

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).lower() in _PRIVILEGED_PREDICTION_KEYS:
                    raise ValueError(f"qualification prediction 含 privileged 字段: {path}.{key}")
                walk(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(row, "prediction")


def _validate_qualification_object_label(label: Mapping[str, Any], *, committed: bool) -> None:
    """只允许 object label 与其显式 observability diagnostics。"""

    _require_exact_keys(
        dict(label),
        _PRIVATE_LABEL_COMMIT_KEYS if committed else _PRIVILEGED_OBJECT_LABEL_KEYS,
        "qualification private object label",
    )
    diagnostics = _require_exact_keys(
        label.get("gt_object_observability"),
        _PRIVILEGED_OBJECT_OBSERVABILITY_KEYS,
        "qualification object observability diagnostics",
    )
    base_diagnostics = dict(diagnostics)
    legacy_mismatch = base_diagnostics.pop("legacy_contract_mismatch", None)
    try:
        observability = ObjectObservabilityLabel(**base_diagnostics)
        position = np.asarray(label.get("gt_object_position_base_m"), dtype=np.float64)
        projected_value = label.get("gt_object_projected_normalized_uv")
        projected = (
            None if projected_value is None else np.asarray(projected_value, dtype=np.float64)
        )
        contact = float(label.get("robot_object_contact_force_n"))
    except (TypeError, ValueError) as error:
        raise RuntimeError("qualification object-only label 类型/语义漂移") from error
    if (
        diagnostics != observability.to_dict()
        or legacy_mismatch is not observability.legacy_contract_mismatch
        or label.get("gt_object_exists") is not True
        or type(label.get("gt_observable")) is not bool
        or label.get("gt_observable") is not observability.observable
        or label.get("gt_object_projection_valid") is not observability.projection_valid
        or observability.object_exists is not True
        or observability.center_inside_goal_mask is not False
        or position.shape != (3,)
        or not np.isfinite(position).all()
        or (
            observability.projection_valid
            and (
                projected is None
                or projected.shape != (2,)
                or not np.isfinite(projected).all()
                or np.any(projected < 0.0)
                or np.any(projected > 1.0)
            )
        )
        or (not observability.projection_valid and projected is not None)
        or not _is_sha256(label.get("gt_object_mask_sha256"))
        or type(label.get("gt_object_visible_pixel_count")) is not int
        or label["gt_object_visible_pixel_count"] < 0
        or observability.legacy_visible is not (label["gt_object_visible_pixel_count"] > 0)
        or type(label.get("is_grasped")) is not bool
        or not math.isfinite(contact)
        or contact < 0.0
        or label.get("goal_gt_read_count") != 0
        or label.get("test_data_read") is not False
    ):
        raise RuntimeError("qualification object-only label primitive 漂移")
    if committed:
        unsigned = dict(label)
        internal = unsigned.pop("label_sha256", None)
        if (
            label.get("version") != E018_P1_G2C_QUALIFICATION_EXECUTION_VERSION
            or type(label.get("row_index")) is not int
            or not _is_sha256(label.get("prediction_sha256"))
            or not _is_sha256(label.get("prediction_raw_sha256"))
            or not _is_sha256(label.get("prediction_commit_receipt_sha256"))
            or type(label.get("prediction_write_started_at_unix_ns")) is not int
            or type(label.get("prediction_fsync_completed_at_unix_ns")) is not int
            or type(label.get("privileged_captured_at_unix_ns")) is not int
            or label["prediction_write_started_at_unix_ns"] <= 0
            or label["prediction_fsync_completed_at_unix_ns"]
            <= label["prediction_write_started_at_unix_ns"]
            or label["privileged_captured_at_unix_ns"]
            <= label["prediction_fsync_completed_at_unix_ns"]
            or internal != canonical_sha256(unsigned)
        ):
            raise RuntimeError("qualification committed object label identity/hash 漂移")


def _qualification_public_scoring_primitive(
    label: Mapping[str, Any],
) -> dict[str, Any]:
    """只承诺 scorer 真正消费的 GT primitive，避免公开无关私有字段。"""

    keys = (
        "row_index",
        "prediction_sha256",
        "gt_object_exists",
        "gt_observable",
        "gt_object_position_base_m",
        "is_grasped",
        "robot_object_contact_force_n",
        "goal_gt_read_count",
        "test_data_read",
        "prediction_fsync_completed_at_unix_ns",
        "privileged_captured_at_unix_ns",
    )
    missing = [key for key in keys if key not in label]
    if missing:
        raise RuntimeError(f"qualification public scoring primitive 缺字段: {missing}")
    return {key: label[key] for key in keys}


class QualificationJournal:
    """强制 prediction fsync 先于 GT getter，并维护不可覆盖 hash chain。"""

    def __init__(
        self,
        *,
        public_root: str | Path,
        private_label_root: str | Path,
        config_sha256: str,
        classification: str,
    ) -> None:
        self.public_root = Path(public_root)
        self.private_label_root = Path(private_label_root)
        if self.public_root.exists() or self.private_label_root.exists():
            raise FileExistsError("qualification public/private output 必须同时全新")
        self.public_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.private_label_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        (self.public_root / "prediction_commits").mkdir(mode=0o700)
        (self.private_label_root / "label_commits").mkdir(mode=0o700)
        self.config_sha256 = config_sha256
        self.classification = classification
        self.started_at_unix_ns = time.time_ns()
        self.consumption_started_at_unix_ns: int | None = None
        self.previous_prediction_sha256: str | None = None
        self.prediction_count = 0
        self.privileged_access_started_count = 0
        self.privileged_capture_count = 0
        self.private_label_inventory_rows: list[dict[str, Any]] = []
        self._write_state("initialized-before-privileged-read")

    @property
    def state_path(self) -> Path:
        return self.public_root / "phase_state.json"

    def _state(self, status: str) -> dict[str, Any]:
        return {
            "version": E018_P1_G2C_QUALIFICATION_EXECUTION_VERSION,
            "status": status,
            "classification": self.classification,
            "config_sha256": self.config_sha256,
            "created_at_unix_ns": self.started_at_unix_ns,
            "qualification_consumption_started_at_unix_ns": (self.consumption_started_at_unix_ns),
            "label_array_consumed": self.privileged_access_started_count > 0,
            "prediction_commit_count": self.prediction_count,
            "privileged_access_started_count": (self.privileged_access_started_count),
            "privileged_capture_count": self.privileged_capture_count,
            "rerun_under_same_identity_allowed": False,
        }

    def _write_state(self, status: str) -> None:
        _atomic_replace_json(self.state_path, self._state(status))

    def _start_consumption(self) -> None:
        if self.consumption_started_at_unix_ns is None:
            self.consumption_started_at_unix_ns = time.time_ns()
            self._write_state("qualification-consumption-started")

    def commit_prediction_then_capture_label(
        self,
        prediction: Mapping[str, Any],
        *,
        privileged_getter: Callable[[], Mapping[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """提交一行 prediction 并 fsync 后，才调用 privileged getter。"""

        self._start_consumption()
        row_index = self.prediction_count
        if prediction.get("row_index") != row_index:
            raise RuntimeError("qualification prediction row_index/order 漂移")
        assert_qualification_prediction_deployable_only(prediction)
        write_started_at = time.time_ns()
        committed = {
            **dict(prediction),
            "previous_prediction_sha256": self.previous_prediction_sha256,
            "prediction_write_started_at_unix_ns": write_started_at,
        }
        committed["prediction_sha256"] = canonical_sha256(committed)
        prediction_path = self.public_root / "prediction_commits" / f"{row_index:06d}.json"
        prediction_raw_sha256, prediction_fsync_completed_at = _atomic_create_json(
            prediction_path, committed
        )
        reopened = _read_json(prediction_path, "qualification prediction commit")
        reopened_unsigned = dict(reopened)
        reopened_internal = reopened_unsigned.pop("prediction_sha256", None)
        if (
            reopened != committed
            or reopened_internal != canonical_sha256(reopened_unsigned)
            or file_sha256(prediction_path) != prediction_raw_sha256
        ):
            raise RuntimeError("qualification prediction commit reopen 漂移")

        commit_receipt = {
            "version": E018_P1_G2C_QUALIFICATION_EXECUTION_VERSION,
            "row_index": row_index,
            "prediction_sha256": committed["prediction_sha256"],
            "prediction_raw_sha256": prediction_raw_sha256,
            "prediction_fsync_completed_at_unix_ns": (prediction_fsync_completed_at),
        }
        commit_receipt["commit_receipt_sha256"] = canonical_sha256(commit_receipt)
        _atomic_create_json(
            self.public_root / "prediction_commits" / f"{row_index:06d}.commit.json",
            commit_receipt,
        )
        self.previous_prediction_sha256 = committed["prediction_sha256"]
        self.prediction_count += 1

        # 先把“GT access 已开始”持久化；即使 getter 在读取后抛错，也不能把
        # 本 identity 误报成未消费。
        self.privileged_access_started_count += 1
        self._write_state("qualification-privileged-access-in-progress")
        privileged = dict(privileged_getter())
        _validate_qualification_object_label(privileged, committed=False)
        captured_at = time.time_ns()
        if captured_at <= prediction_fsync_completed_at:
            raise RuntimeError("qualification GT wall-clock 未严格晚于 prediction fsync")
        label = {
            **privileged,
            "version": E018_P1_G2C_QUALIFICATION_EXECUTION_VERSION,
            "row_index": row_index,
            "prediction_sha256": committed["prediction_sha256"],
            "prediction_raw_sha256": prediction_raw_sha256,
            "prediction_commit_receipt_sha256": commit_receipt["commit_receipt_sha256"],
            "prediction_write_started_at_unix_ns": write_started_at,
            "prediction_fsync_completed_at_unix_ns": (prediction_fsync_completed_at),
            "privileged_captured_at_unix_ns": captured_at,
        }
        label["label_sha256"] = canonical_sha256(label)
        _validate_qualification_object_label(label, committed=True)
        label_path = self.private_label_root / "label_commits" / f"{row_index:06d}.json"
        label_raw_sha256, _ = _atomic_create_json(label_path, label)
        inventory_row = {
            "row_index": row_index,
            "prediction_sha256": committed["prediction_sha256"],
            "label_raw_sha256": label_raw_sha256,
            "label_internal_sha256": label["label_sha256"],
            "size_bytes": label_path.stat().st_size,
            "public_scoring_primitive_sha256": canonical_sha256(
                _qualification_public_scoring_primitive(label)
            ),
        }
        inventory_row["row_sha256"] = canonical_sha256(inventory_row)
        self.private_label_inventory_rows.append(inventory_row)
        self.privileged_capture_count += 1
        self._write_state("qualification-routes-in-progress")
        return committed, label

    def private_label_inventory(self) -> dict[str, Any]:
        if (
            self.prediction_count == 0
            or self.prediction_count != self.privileged_capture_count
            or len(self.private_label_inventory_rows) != self.privileged_capture_count
        ):
            raise RuntimeError("qualification private label inventory 不完整")
        return _freeze_ordered_inventory(
            self.private_label_inventory_rows,
            version=_PRIVATE_LABEL_INVENTORY_VERSION,
        )

    def freeze_route_completion(self) -> dict[str, Any]:
        self.private_label_inventory()
        self._write_state("routes-complete-context-destroy-pending")
        return self._state("routes-complete-context-destroy-pending")

    def mark_context_destroyed(self) -> dict[str, Any]:
        self._write_state("complete-execution-freeze-context-destroyed")
        return self._state("complete-execution-freeze-context-destroyed")

    def freeze_consumed_failure(
        self,
        error: BaseException,
        *,
        evidence: Mapping[str, Any] | None = None,
        cleanup_errors: Sequence[str] = (),
    ) -> dict[str, Any]:
        failure = {
            "version": E018_P1_G2C_QUALIFICATION_EXECUTION_VERSION,
            "status": "consumed-qualification-failure",
            "classification": self.classification,
            "config_sha256": self.config_sha256,
            "prediction_commit_count": self.prediction_count,
            "privileged_access_started_count": (self.privileged_access_started_count),
            "privileged_capture_count": self.privileged_capture_count,
            "label_array_consumed": self.privileged_access_started_count > 0,
            "rerun_under_same_identity_allowed": False,
            "error_type": type(error).__name__,
            "error_message": (
                str(error)
                if not cleanup_errors
                else f"{error}; cleanup_errors={list(cleanup_errors)}"
            ),
            "cleanup_errors": list(cleanup_errors),
            "failure_evidence": dict(evidence or {}),
            "failed_at_unix_ns": time.time_ns(),
        }
        failure["failure_sha256"] = canonical_sha256(failure)
        _atomic_create_json(self.public_root / "consumed_failure.json", failure)
        self._write_state("consumed-qualification-failure")
        return failure


def _finalize_qualification_execution(
    *,
    env: Any,
    provider: QualificationProvider,
    journal: QualificationJournal,
    public_root: Path,
    execution_freeze: Mapping[str, Any],
    environment_identity: Mapping[str, Any],
    private_root: Path | None = None,
    combined_artifact_bytes_max: int = 2**63 - 1,
    capture_started_at_unix_ns: int | None = None,
    capture_started_monotonic_s: float | None = None,
    wall_seconds_max: float = math.inf,
    gpu_seconds_max: float = math.inf,
) -> dict[str, Any]:
    """销毁 context 后发布 receipt；任一后处理失败都冻结 consumed failure。"""

    cleanup_errors = _collect_context_cleanup_errors(env=env, provider=provider)
    if cleanup_errors:
        error = RuntimeError(f"qualification context cleanup failures: {cleanup_errors}")
        journal.freeze_consumed_failure(error, cleanup_errors=cleanup_errors)
        raise error
    try:
        if capture_started_at_unix_ns is None:
            capture_started_at_unix_ns = time.time_ns() - 1
        if capture_started_monotonic_s is None:
            capture_started_monotonic_s = time.monotonic()
        completed_at_unix_ns = time.time_ns()
        wall_elapsed_seconds = time.monotonic() - capture_started_monotonic_s
        # D036 的 active-GPU 口径是进程 monotonic elapsed。这里用完整
        # capture wall 作为 GPU 占用保守上界，不以 CUDA kernel event 缩短预算。
        gpu_elapsed_seconds = wall_elapsed_seconds
        if (
            not math.isfinite(wall_elapsed_seconds)
            or wall_elapsed_seconds < 0.0
            or wall_elapsed_seconds > wall_seconds_max
            or gpu_elapsed_seconds > gpu_seconds_max
        ):
            raise RuntimeError("qualification capture wall/GPU budget 超限")
        context_state = journal.mark_context_destroyed()
        receipt = {
            **dict(execution_freeze),
            "status": "complete-execution-freeze-context-destroyed",
            "environment_identity": dict(environment_identity),
            "context_destroyed": True,
            "phase_state_status": context_state["status"],
            "started_at_unix_ns": capture_started_at_unix_ns,
            "completed_at_unix_ns": completed_at_unix_ns,
            "wall_elapsed_seconds": wall_elapsed_seconds,
            "gpu_elapsed_seconds": gpu_elapsed_seconds,
            "gpu_elapsed_accounting": (
                "full-capture-process-monotonic-wall-conservative-upper-bound/v1"
            ),
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        if private_root is not None:
            capture_artifact_bytes = (
                _regular_tree_bytes(public_root, name="qualification public capture before receipt")
                + _regular_tree_bytes(private_root, name="qualification private capture")
                + len(_serialized_json_bytes(receipt))
            )
            if capture_artifact_bytes > combined_artifact_bytes_max:
                raise RuntimeError("qualification capture public/private artifact budget 超限")
        _atomic_create_json(public_root / "execution_receipt.json", receipt)
        return receipt
    except Exception as error:
        journal.freeze_consumed_failure(error)
        raise


def _qualification_execution_budget_limits(
    *,
    classification: str,
    config: Mapping[str, Any],
    execution_decision_verification: Mapping[str, Any] | None,
) -> tuple[float, float, int]:
    """返回 wall/GPU/三树 artifact 上限；formal 只信已验证 D048。"""

    if classification == QUALIFICATION_CLASSIFICATION_SMOKE:
        return (
            float(config["budgets"]["noncanonical_smoke_gpu_seconds_max"]),
            float(config["budgets"]["noncanonical_smoke_gpu_seconds_max"]),
            int(config["budgets"]["artifact_bytes_max"]),
        )
    if (
        classification != QUALIFICATION_CLASSIFICATION_FORMAL
        or execution_decision_verification is None
        or execution_decision_verification.get("verified") is not True
    ):
        raise RuntimeError("qualification formal budget 缺已验证 D048")
    budgets = execution_decision_verification["receipt"]["budgets"]
    return (
        float(budgets["gpu"]["formal_wall_seconds_max"]),
        float(budgets["gpu"]["formal_gpu_seconds_reserved_max"]),
        int(budgets["artifact"]["formal_combined_artifact_bytes_reserved_max"]),
    )


def _collect_context_cleanup_errors(*, env: Any | None, provider: Any) -> list[str]:
    """独立尝试 env/provider cleanup，保留两侧全部错误。"""

    cleanup_errors: list[str] = []
    if env is not None:
        try:
            env.close()
        except Exception as cleanup_error:
            cleanup_errors.append(f"env.close:{type(cleanup_error).__name__}:{cleanup_error}")
    try:
        provider.destroy()
    except Exception as cleanup_error:
        cleanup_errors.append(f"provider.destroy:{type(cleanup_error).__name__}:{cleanup_error}")
    return cleanup_errors


def _best_effort_freeze_capture_failure(
    *,
    env: Any | None,
    provider: Any,
    journal: QualificationJournal | None,
    error: BaseException,
    evidence: Mapping[str, Any] | None = None,
) -> None:
    """route-time 异常时两侧 cleanup 独立尝试，并最终冻结 consumed failure。"""

    cleanup_errors = _collect_context_cleanup_errors(env=env, provider=provider)
    if journal is None:
        return
    journal.freeze_consumed_failure(
        error,
        evidence=evidence,
        cleanup_errors=cleanup_errors,
    )


def _append_returned_route_evidence(
    *,
    motion_writer: _AppendOnlyJsonl,
    summary_writer: _AppendOnlyJsonl,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    """完整 route 返回后立即落盘；任何后续 gate 都在此函数之后。"""

    motion_writer.append(rows)
    summary_writer.append([summary])


def _writer_failure_snapshot(
    writer: _AppendOnlyJsonl | None,
) -> dict[str, Any] | None:
    if writer is None or not writer.path.is_file():
        return None
    return {
        "writer_row_count": writer.row_count,
        "raw_sha256": file_sha256(writer.path),
        "size_bytes": writer.path.stat().st_size,
    }


def _freeze_capture_failure_evidence(
    *,
    public_root: Path,
    motion_writer: _AppendOnlyJsonl | None,
    summary_writer: _AppendOnlyJsonl | None,
    route_counters: Sequence[Mapping[str, int]],
    persisted_route_count: int,
    active_route_identity: Mapping[str, Any] | None,
    active_route_rows: Sequence[Mapping[str, Any]],
    active_route_persisted: bool,
    route_rgb_inventory_rows: Sequence[Mapping[str, Any]],
    journal: QualificationJournal | None,
) -> dict[str, Any]:
    """失败时冻结 completed/full-returned/partial 三类可恢复证据。"""

    partial_ledger: dict[str, Any] | None = None
    partial_error: str | None = None
    if active_route_rows and not active_route_persisted:
        try:
            partial_writer = _AppendOnlyJsonl(public_root / "partial_route_ledger.jsonl")
            partial_writer.append(active_route_rows)
            partial_ledger = partial_writer.freeze()
        except Exception as partial_exception:
            partial_error = f"{type(partial_exception).__name__}:{partial_exception}"
    evidence = {
        "version": "e018-p1-g2c-capture-failure-evidence/v1",
        "validated_route_count": len(route_counters),
        "persisted_full_route_count": persisted_route_count,
        "active_route_identity": (
            None if active_route_identity is None else dict(active_route_identity)
        ),
        "active_route_row_count": len(active_route_rows),
        "active_route_persisted_in_main_ledger": active_route_persisted,
        "partial_route_ledger": partial_ledger,
        "partial_route_persistence_error": partial_error,
        "motion_ledger": _writer_failure_snapshot(motion_writer),
        "route_summaries": _writer_failure_snapshot(summary_writer),
        "route_rgb_inventory_rows": [dict(item) for item in route_rgb_inventory_rows],
        "route_rgb_inventory_rows_sha256": canonical_sha256(route_rgb_inventory_rows),
        "prediction_commit_count": (None if journal is None else journal.prediction_count),
        "privileged_access_started_count": (
            None if journal is None else journal.privileged_access_started_count
        ),
        "privileged_capture_count": (None if journal is None else journal.privileged_capture_count),
        "private_label_inventory_rows": (
            [] if journal is None else list(journal.private_label_inventory_rows)
        ),
        "last_prediction_sha256": (None if journal is None else journal.previous_prediction_sha256),
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return evidence


def _type7_quantile(values: Sequence[float], *, numerator: int, denominator: int) -> float:
    """冻结的 sorted linear type-7 quantile，避免 NumPy/LAPACK 路径差异。"""

    if (
        not values
        or type(numerator) is not int
        or type(denominator) is not int
        or denominator <= 0
        or not 0 <= numerator <= denominator
    ):
        raise ValueError("qualification quantile 输入非法")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("qualification quantile 只接受有限数值")
    scaled = (len(ordered) - 1) * numerator
    lower = scaled // denominator
    remainder = scaled % denominator
    if lower == len(ordered) - 1 or remainder == 0:
        return ordered[lower]
    return ordered[lower] + ((ordered[lower + 1] - ordered[lower]) * remainder / denominator)


def _recompute_qualification_prediction(
    prediction: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """从原始 provider 输出与冻结 config 重算 scale/threshold/accept bool。"""

    viewpoint_id = prediction.get("viewpoint_id")
    if viewpoint_id not in QUALIFICATION_VIEW_ORDER:
        raise RuntimeError("qualification prediction viewpoint 漂移")
    calibration = config["calibration"]["values"][viewpoint_id]
    scale = float(calibration["scale_factor"])
    threshold = float(calibration["write_threshold"])
    evidence = ObjectWriteEvidence(
        visibility_probability=float(prediction["object_visibility_probability"]),
        projection_validity_probability=float(prediction["projection_validity_probability"]),
        object_mask_probability=float(prediction["object_mask_probability_at_prediction"]),
        goal_mask_probability=float(prediction["goal_mask_probability_at_prediction"]),
        normalized_entropy=float(prediction["object_normalized_entropy"]),
        radial_sigma_px=float(np.linalg.norm(prediction["object_sigma_xy_px"])),
        geometry_valid=bool(prediction["geometry_valid"]),
    )
    raw_value = prediction["raw_covariance_base_m2"]
    if raw_value is None:
        calibrated = None
        maximum_std = None
    else:
        raw = np.asarray(raw_value, dtype=np.float64)
        if (
            raw.shape != (3, 3)
            or not np.isfinite(raw).all()
            or not np.allclose(raw, raw.T, rtol=0.0, atol=1e-12)
            or float(np.linalg.eigvalsh(raw).min()) < -1e-12
        ):
            raise RuntimeError("qualification raw covariance 漂移")
        calibrated = raw * scale
        maximum_std = float(np.sqrt(max(0.0, float(np.linalg.eigvalsh(calibrated).max()))))
    deployable_safe = qualification_deployable_safe(
        prediction["deployable_safety"], qualification_config=config
    )
    predicted_observable = bool(float(prediction["object_visibility_probability"]) >= 0.5)
    structurally_eligible = bool(
        predicted_observable and evidence.structurally_eligible and deployable_safe
    )
    accepted = bool(
        structurally_eligible
        and evidence.score >= threshold
        and maximum_std is not None
        and maximum_std <= config["qualification"]["maximum_calibrated_position_std_m"]
    )
    return {
        "predicted_observable": predicted_observable,
        "write_score": evidence.score,
        "object_write_structurally_eligible": evidence.structurally_eligible,
        "deployable_free_static_safe": deployable_safe,
        "structurally_eligible": structurally_eligible,
        "calibration_scale_factor": scale,
        "calibrated_covariance_base_m2": (
            None if calibrated is None else calibrated.astype(float).tolist()
        ),
        "calibrated_position_std_max_m": maximum_std,
        "write_threshold": threshold,
        "write_accepted": accepted,
    }


def _same_derived_value(
    actual: Any,
    expected: Any,
    *,
    tolerance: float,
) -> bool:
    if type(expected) is bool or expected is None:
        return actual is expected
    if isinstance(expected, str):
        return actual == expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)
        )
    if isinstance(expected, list):
        try:
            left = np.asarray(actual, dtype=np.float64)
            right = np.asarray(expected, dtype=np.float64)
        except (TypeError, ValueError):
            return actual == expected
        return left.shape == right.shape and bool(
            np.allclose(left, right, rtol=0.0, atol=tolerance)
        )
    return actual == expected


def validate_qualification_prediction_mechanics(
    prediction: Mapping[str, Any], *, config: Mapping[str, Any]
) -> dict[str, Any]:
    classification = prediction.get("classification")
    if classification not in {
        QUALIFICATION_CLASSIFICATION_FORMAL,
        QUALIFICATION_CLASSIFICATION_SMOKE,
    }:
        raise RuntimeError("qualification prediction classification 漂移")
    _validate_qualification_prediction_static_identity(
        prediction,
        config=config,
        classification=classification,
        committed="prediction_sha256" in prediction,
    )
    _validate_prediction_geometry_same_source(
        prediction,
        intrinsic_cv=prediction["external_intrinsic_cv"],
        base_from_camera_cv=prediction["base_from_external_camera_cv"],
        plane_base_z_m=float(config["capture_safety"]["object_center_base_z_m"]),
        tolerance=float(config["qualification"]["metric_float_recompute_tolerance"]),
    )
    recomputed = _recompute_qualification_prediction(prediction, config=config)
    tolerance = float(config["qualification"]["metric_float_recompute_tolerance"])
    for name, expected in recomputed.items():
        actual = prediction.get(name)
        # Safety/accept/PRIMARY bool 永远 exact；1e-14 只给数值派生量。
        field_tolerance = 0.0 if type(expected) is bool else tolerance
        if not _same_derived_value(actual, expected, tolerance=field_tolerance):
            raise RuntimeError(f"qualification prediction 派生字段漂移: {name}")
    return recomputed


def _validate_prediction_against_route_row(
    prediction: Mapping[str, Any],
    *,
    route_row: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """把评分 prediction 的相机/安全 primitive 绑定到同一 raw route 帧。"""

    safety = prediction["deployable_safety"]
    finger_positions = _qualification_finite_vector(route_row, "finger_joint_positions_m", 2)
    lower, upper = RobotSpec().gripper_joint_position_range_m
    expected_opening = float(
        np.mean((finger_positions - lower) / (upper - lower), dtype=np.float64)
    )
    force = np.asarray(safety["finger_force_n"], dtype=np.float64)
    _, expected_projection_error = _single_rigid(
        route_row["actual_base_from_external_camera_cv"],
        "qualification prediction-route camera transform",
        maximum_projection_error=float(
            config["capture_safety"]["maximum_rotation_projection_error_frobenius"]
        ),
    )
    if any(prediction.get(name) != value for name, value in expected_identity.items()):
        raise RuntimeError("qualification prediction/raw-route frame identity 漂移")
    if (
        not np.allclose(
            np.asarray(prediction["external_intrinsic_cv"], dtype=np.float64),
            np.asarray(route_row["external_intrinsic_cv"], dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )
        or not np.allclose(
            np.asarray(prediction["base_from_external_camera_cv"], dtype=np.float64),
            np.asarray(route_row["actual_base_from_external_camera_cv"], dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )
        or safety.get("eligible_capture") is not True
        or force.shape != (2,)
        or not math.isclose(
            float(np.max(force)),
            float(route_row["finger_object_contact_force_n"]),
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        or not math.isclose(
            float(safety["raw_gripper_opening_ratio"]),
            expected_opening,
            rel_tol=0.0,
            abs_tol=1e-7,
        )
    ):
        raise RuntimeError("qualification prediction/raw-route camera/gripper 漂移")
    linked_scalars = {
        "arm_joint_drift_rad": "arm_joint_max_drift_rad",
        "tcp_position_drift_m": "tcp_position_drift_m",
        "tcp_orientation_drift_rad": "tcp_orientation_drift_rad",
        "rgb_timestamp_s": "external_rgb_timestamp_s",
        "pose_timestamp_s": "external_pose_timestamp_s",
        "camera_position_tracking_error_m": ("external_position_tracking_error_m"),
        "camera_orientation_tracking_error_rad": ("external_orientation_tracking_error_rad"),
    }
    for safety_name, route_name in linked_scalars.items():
        if not math.isclose(
            float(safety[safety_name]),
            float(route_row[route_name]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"qualification prediction/raw-route safety 漂移: {safety_name}")
    if not math.isclose(
        float(safety["rotation_projection_error_frobenius"]),
        expected_projection_error,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("qualification prediction/raw-route rotation projection 漂移")


def score_qualification_prediction(
    prediction: Mapping[str, Any],
    label: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """独立 scorer 的单行纯函数；bool 均从 prediction+label 重算。"""

    mechanics = validate_qualification_prediction_mechanics(prediction, config=config)
    gt_position = np.asarray(label["gt_object_position_base_m"], dtype=np.float64)
    contact = float(label["robot_object_contact_force_n"])
    if (
        type(label["gt_object_exists"]) is not bool
        or type(label["gt_observable"]) is not bool
        or type(label["is_grasped"]) is not bool
        or gt_position.shape != (3,)
        or not np.isfinite(gt_position).all()
        or not math.isfinite(contact)
        or contact < 0.0
        or label.get("goal_gt_read_count") != 0
        or label.get("test_data_read") is not False
    ):
        raise RuntimeError("qualification object-only GT primitive 漂移")
    predicted_value = prediction["predicted_object_position_base_m"]
    predicted_position = (
        None if predicted_value is None else np.asarray(predicted_value, dtype=np.float64)
    )
    if predicted_position is not None and (
        predicted_position.shape != (3,) or not np.isfinite(predicted_position).all()
    ):
        raise RuntimeError("qualification predicted object position 漂移")
    world_error = (
        None
        if predicted_position is None
        else float(np.linalg.norm(predicted_position - gt_position))
    )
    error_xy = (
        None
        if predicted_position is None
        else (predicted_position[:2] - gt_position[:2]).astype(float).tolist()
    )
    limits = config["capture_safety"]
    privileged_safe = bool(
        label["gt_object_exists"]
        and not label["is_grasped"]
        and contact <= limits["maximum_robot_object_contact_force_n"]
        and abs(float(gt_position[2]) - limits["object_center_base_z_m"])
        <= limits["object_center_base_z_tolerance_m"]
    )
    oracle_safe = bool(
        mechanics["deployable_free_static_safe"]
        and privileged_safe
        and label["gt_observable"]
        and prediction["geometry_valid"]
        and world_error is not None
        and world_error <= config["qualification"]["maximum_oracle_safe_error_m"]
    )
    catastrophic = bool(
        world_error is not None
        and world_error > config["qualification"]["catastrophic_world_xyz_error_m"]
    )
    covariance_evaluable = bool(
        label["gt_observable"]
        and prediction["geometry_valid"]
        and error_xy is not None
        and mechanics["calibrated_covariance_base_m2"] is not None
    )
    mahalanobis_squared = None
    covariance_95_covered = None
    if covariance_evaluable:
        covariance = np.asarray(mechanics["calibrated_covariance_base_m2"], dtype=np.float64)
        mahalanobis_squared = _mahalanobis_squared_psd(
            np.asarray(error_xy, dtype=np.float64), covariance[:2, :2]
        )
        covariance_95_covered = bool(
            math.isfinite(mahalanobis_squared)
            and mahalanobis_squared <= config["qualification"]["covariance_chi_square_threshold"]
        )
    accepted = mechanics["write_accepted"]
    return {
        "version": E018_P1_G2C_QUALIFICATION_RESULT_VERSION,
        "phase": "offline-after-context-destroy-object-only-scoring/v1",
        "row_index": prediction["row_index"],
        "seed": prediction["seed"],
        "sample_index": prediction["sample_index"],
        "viewpoint_id": prediction["viewpoint_id"],
        "prediction_sha256": prediction["prediction_sha256"],
        "label_sha256": label["label_sha256"],
        "prediction_fsync_completed_at_unix_ns": label["prediction_fsync_completed_at_unix_ns"],
        "privileged_captured_at_unix_ns": label["privileged_captured_at_unix_ns"],
        "gt_object_exists": label["gt_object_exists"],
        "gt_observable": label["gt_observable"],
        "gt_object_position_base_m": gt_position.astype(float).tolist(),
        "is_grasped": label["is_grasped"],
        "robot_object_contact_force_n": contact,
        "predicted_observable": mechanics["predicted_observable"],
        "geometry_valid": bool(prediction["geometry_valid"]),
        "predicted_object_position_base_m": predicted_value,
        "world_xyz_error_m": world_error,
        "world_xy_error_vector_m": error_xy,
        "raw_covariance_base_m2": prediction["raw_covariance_base_m2"],
        "calibrated_covariance_base_m2": mechanics["calibrated_covariance_base_m2"],
        "calibrated_position_std_max_m": mechanics["calibrated_position_std_max_m"],
        "write_score": mechanics["write_score"],
        "write_threshold": mechanics["write_threshold"],
        "deployable_free_static_safe": mechanics["deployable_free_static_safe"],
        "privileged_free_static_safe": privileged_safe,
        "structurally_eligible": mechanics["structurally_eligible"],
        "write_accepted": accepted,
        "oracle_safe_measurement": oracle_safe,
        "unsafe_accepted": bool(accepted and not oracle_safe),
        "catastrophic_measurement": catastrophic,
        "catastrophic_accepted": bool(accepted and catastrophic),
        "covariance_evaluable": covariance_evaluable,
        "mahalanobis_squared": mahalanobis_squared,
        "covariance_95_covered": covariance_95_covered,
        "test_data_read": False,
        "goal_gt_read_count": 0,
    }


def summarize_qualification_viewpoint(
    rows: Sequence[Mapping[str, Any]],
    *,
    viewpoint_id: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if not rows or any(row.get("viewpoint_id") != viewpoint_id for row in rows):
        raise ValueError("qualification summary rows 必须属于单一非空 viewpoint")
    predicted_positive = sum(bool(row["predicted_observable"]) for row in rows)
    observable_positive = sum(bool(row["gt_observable"]) for row in rows)
    true_positive = sum(bool(row["predicted_observable"] and row["gt_observable"]) for row in rows)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / observable_positive if observable_positive else 0.0
    errors = [
        float(row["world_xyz_error_m"])
        for row in rows
        if row["gt_observable"] and row["geometry_valid"] and row["world_xyz_error_m"] is not None
    ]
    p90 = _type7_quantile(errors, numerator=9, denominator=10) if errors else None
    maximum = max(errors) if errors else None
    oracle_safe_count = sum(bool(row["oracle_safe_measurement"]) for row in rows)
    accepted_safe_count = sum(
        bool(row["write_accepted"] and row["oracle_safe_measurement"]) for row in rows
    )
    accepted_safe_coverage = accepted_safe_count / oracle_safe_count if oracle_safe_count else 0.0
    covariance_rows = [row for row in rows if row["covariance_evaluable"]]
    covariance_covered_count = sum(row["covariance_95_covered"] is True for row in covariance_rows)
    covariance_coverage = (
        covariance_covered_count / len(covariance_rows) if covariance_rows else 0.0
    )
    std_values = [
        float(row["calibrated_position_std_max_m"])
        for row in rows
        if row["calibrated_position_std_max_m"] is not None
    ]
    maximum_std = max(std_values) if std_values else None
    unsafe_count = sum(bool(row["unsafe_accepted"]) for row in rows)
    catastrophic_accepted_count = sum(bool(row["catastrophic_accepted"]) for row in rows)
    rules = config["qualification"]
    gates = {
        "visibility_precision": precision >= rules["minimum_visibility_precision"],
        "visibility_recall": recall >= rules["minimum_visibility_recall"],
        "observable_world_xyz_p90": p90 is not None
        and p90 <= rules["maximum_observable_world_xyz_p90_m"],
        "observable_world_xyz_max": maximum is not None
        and maximum <= rules["maximum_observable_world_xyz_max_m"],
        "unsafe_accepted": unsafe_count <= rules["maximum_unsafe_accepted_count"],
        "catastrophic_accepted": catastrophic_accepted_count
        <= rules["maximum_catastrophic_accepted_count"],
        "accepted_safe_coverage": accepted_safe_coverage >= rules["minimum_accepted_safe_coverage"],
        "covariance_95_coverage": covariance_coverage >= rules["minimum_covariance_95_coverage"],
        "covariance_support": len(covariance_rows) >= rules["minimum_covariance_evaluable_count"],
        "maximum_calibrated_position_std": maximum_std is not None
        and maximum_std <= rules["maximum_calibrated_position_std_m"],
    }
    passed = all(gates.values())
    return {
        "viewpoint_id": viewpoint_id,
        "row_count": len(rows),
        "observable_positive_count": observable_positive,
        "predicted_positive_count": predicted_positive,
        "true_positive_count": true_positive,
        "visibility_precision": precision,
        "visibility_recall": recall,
        "observable_geometry_evaluable_count": len(errors),
        "observable_world_xyz_p90_m": p90,
        "observable_world_xyz_max_m": maximum,
        "oracle_safe_count": oracle_safe_count,
        "accepted_count": sum(bool(row["write_accepted"]) for row in rows),
        "accepted_and_oracle_safe_count": accepted_safe_count,
        "accepted_safe_coverage": accepted_safe_coverage,
        "unsafe_accepted_count": unsafe_count,
        "catastrophic_measurement_count": sum(
            bool(row["catastrophic_measurement"]) for row in rows
        ),
        "catastrophic_accepted_count": catastrophic_accepted_count,
        "covariance_evaluable_count": len(covariance_rows),
        "covariance_95_covered_count": covariance_covered_count,
        "covariance_95_coverage": covariance_coverage,
        "maximum_calibrated_position_std_m": maximum_std,
        "gates": gates,
        "failure_reasons": [name for name, passed_gate in gates.items() if not passed_gate],
        "passed": passed,
    }


def select_qualification_primary(
    summaries: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    qualified = [
        row for row in summaries if row["viewpoint_id"] != FRONT_HOME_ID and row["passed"] is True
    ]
    if not qualified:
        return {
            "status": "protocol-valid-negative-zero-qualified-alternates",
            "primary_viewpoint_id": None,
            "qualified_non_home_viewpoint_ids": [],
            "selection_key": None,
        }
    shortlist = tuple(config["qualification"]["g0b_shortlist_order"])

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        viewpoint_id = str(row["viewpoint_id"])
        return (
            0 if viewpoint_id in shortlist else 1,
            -float(row["accepted_safe_coverage"]),
            float(row["observable_world_xyz_p90_m"]),
            float(row["observable_world_xyz_max_m"]),
            abs(float(row["covariance_95_coverage"]) - 0.95),
            -float(row["visibility_recall"]),
            FRONT_ALTERNATE_IDS.index(viewpoint_id),
        )

    selected = min(qualified, key=key)
    selected_key = key(selected)
    return {
        "status": "primary-selected-from-qualified-alternates",
        "primary_viewpoint_id": selected["viewpoint_id"],
        "qualified_non_home_viewpoint_ids": [
            viewpoint_id
            for viewpoint_id in FRONT_ALTERNATE_IDS
            if any(row["viewpoint_id"] == viewpoint_id for row in qualified)
        ],
        "selection_key": list(selected_key),
    }


def build_qualification_result_summary(
    *,
    execution_verification: Mapping[str, Any],
    viewpoint_summaries: Sequence[Mapping[str, Any]],
    primary: Mapping[str, Any],
    label_open_count: int,
    private_label_consumption_marker_raw_sha256: str,
    private_label_consumption_marker_internal_sha256: str,
) -> dict[str, Any]:
    classification = execution_verification["classification"]
    formal = classification == QUALIFICATION_CLASSIFICATION_FORMAL
    qualified = [
        row["viewpoint_id"]
        for row in viewpoint_summaries
        if row["viewpoint_id"] != FRONT_HOME_ID and row["passed"] is True
    ]
    gate_passed = bool(formal and qualified)
    status = (
        "complete-dynamic-qualification-pass"
        if gate_passed
        else (
            "complete-dynamic-qualification-protocol-valid-negative"
            if formal
            else "complete-preflight-no-qualification-claim"
        )
    )
    return {
        "version": E018_P1_G2C_QUALIFICATION_RESULT_VERSION,
        "status": status,
        "classification": classification,
        "protocol_valid": True,
        "gate_passed": gate_passed,
        "config_sha256": execution_verification["config_sha256"],
        "source_git_commit": execution_verification["source_git_commit"],
        "source_identity_sha256": execution_verification["source_identity_sha256"],
        "execution_freeze_sha256": execution_verification["execution_freeze_sha256"],
        "execution_receipt_internal_sha256": execution_verification[
            "execution_receipt_internal_sha256"
        ],
        "private_label_journal_inventory_sha256": execution_verification[
            "private_label_journal_inventory"
        ]["inventory_sha256"],
        "scoring_row_count": sum(int(row["row_count"]) for row in viewpoint_summaries),
        "viewpoint_summary_count": len(viewpoint_summaries),
        "label_journal_open_count": label_open_count,
        "label_journal_reopen_count_for_public_verification": 0,
        "private_label_consumption_marker_raw_sha256": (
            private_label_consumption_marker_raw_sha256
        ),
        "private_label_consumption_marker_internal_sha256": (
            private_label_consumption_marker_internal_sha256
        ),
        "qualified_non_home_viewpoint_ids": qualified,
        "qualified_non_home_viewpoint_count": len(qualified),
        "primary": dict(primary),
        "unsafe_accepted_count": sum(
            int(row["unsafe_accepted_count"]) for row in viewpoint_summaries
        ),
        "catastrophic_measurement_count": sum(
            int(row["catastrophic_measurement_count"]) for row in viewpoint_summaries
        ),
        "catastrophic_accepted_count": sum(
            int(row["catastrophic_accepted_count"]) for row in viewpoint_summaries
        ),
        "test_array_read_count": 0,
        "memory_read_count": 0,
        "memory_write_count": 0,
        "runtime_camera_actuation_count": 0,
        "physical_camera_actuation_count": 0,
        "arm_motion_command_count": 0,
        "gripper_close_command_count": 0,
        "manipulation_progression_count": 0,
        "checkpoint_write_count": 0,
        "goal_gt_read_count": 0,
        "allowed_claim": (
            "dynamic-front-provider-qualification-only"
            if formal
            else "engineering-preflight-only-no-qualification-claim"
        ),
    }


def _load_public_prediction_chain(
    journal: QualificationJournal,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    previous: str | None = None
    for row_index in range(journal.prediction_count):
        prediction = _read_json(
            journal.public_root / "prediction_commits" / f"{row_index:06d}.json",
            "qualification prediction chain",
        )
        receipt = _read_json(
            journal.public_root / "prediction_commits" / f"{row_index:06d}.commit.json",
            "qualification prediction commit receipt",
        )
        prediction_unsigned = dict(prediction)
        internal = prediction_unsigned.pop("prediction_sha256", None)
        receipt_unsigned = dict(receipt)
        receipt_internal = receipt_unsigned.pop("commit_receipt_sha256", None)
        if (
            prediction.get("row_index") != row_index
            or prediction.get("previous_prediction_sha256") != previous
            or internal != canonical_sha256(prediction_unsigned)
            or receipt.get("row_index") != row_index
            or receipt.get("prediction_sha256") != internal
            or receipt.get("prediction_raw_sha256")
            != file_sha256(journal.public_root / "prediction_commits" / f"{row_index:06d}.json")
            or receipt_internal != canonical_sha256(receipt_unsigned)
        ):
            raise RuntimeError("qualification public prediction hash chain 漂移")
        previous = str(internal)
        predictions.append(prediction)
        receipts.append(receipt)
    return predictions, receipts


def _sum_route_counters(
    counters: Sequence[Mapping[str, int]],
) -> dict[str, int]:
    names = {
        "camera_pose_set_count",
        "moving_interpolation_command_count",
        "safe_hold_open_step_count",
        "ledger_frame_count",
        "provider_scored_home_frame_count",
        "provider_scored_alternate_frame_count",
        "provider_scored_frame_count",
    }
    if any(set(item) != names for item in counters):
        raise RuntimeError("qualification per-route counter schema 漂移")
    return {name: sum(int(item[name]) for item in counters) for name in names}


def _load_route_rgb_witnesses(
    root: Path,
    *,
    route_index: int,
    episode_id: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """重开三张公开 PNG，返回像素 witness 与可冻结的逐文件 inventory。"""

    from PIL import Image

    images: dict[str, np.ndarray] = {}
    files: dict[str, Any] = {}
    for role in ("home_before", "alternate", "home_after"):
        relative = f"images/{episode_id}__{role}.png"
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise RuntimeError(f"qualification route RGB file/link 漂移: {relative}")
        with Image.open(path) as opened:
            if opened.format != "PNG" or opened.mode != "RGB":
                raise RuntimeError(f"qualification route RGB 必须是原生 8-bit RGB PNG: {relative}")
            opened.load()
            array = np.asarray(opened)
        if array.dtype != np.uint8 or array.shape != (128, 128, 3):
            raise RuntimeError(f"qualification route RGB shape/dtype 漂移: {relative}")
        array = np.ascontiguousarray(array)
        pixel_sha256 = hashlib.sha256(array.tobytes()).hexdigest()
        images[role] = array
        files[role] = {
            "relative_path": relative,
            "raw_sha256": file_sha256(path),
            "pixel_sha256": pixel_sha256,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "size_bytes": path.stat().st_size,
        }
    row = {
        "route_index": route_index,
        "episode_id": episode_id,
        "files": files,
    }
    row["row_sha256"] = canonical_sha256(row)
    return images, row


def _freeze_ordered_inventory(rows: Sequence[Mapping[str, Any]], *, version: str) -> dict[str, Any]:
    document = {
        "version": version,
        "row_count": len(rows),
        "rows": [dict(row) for row in rows],
    }
    document["inventory_sha256"] = canonical_sha256(document)
    return document


def _verify_ordered_inventory(
    value: Any,
    *,
    version: str,
    expected_count: int,
    name: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {
        "version",
        "row_count",
        "rows",
        "inventory_sha256",
    }:
        raise RuntimeError(f"{name} schema 漂移")
    unsigned = dict(value)
    internal = unsigned.pop("inventory_sha256", None)
    rows = value.get("rows")
    if (
        value.get("version") != version
        or value.get("row_count") != expected_count
        or not isinstance(rows, list)
        or len(rows) != expected_count
        or internal != canonical_sha256(unsigned)
    ):
        raise RuntimeError(f"{name} identity/count/hash 漂移")
    return rows


def _verify_exact_regular_files(
    root: Path,
    *,
    expected_files: set[str],
    expected_directories: set[str] = frozenset(),
    name: str,
) -> int:
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    total_bytes = 0
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"{name} root 类型漂移")
    for path in root.rglob("*"):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            raise RuntimeError(f"{name} 禁止 symlink: {relative}")
        if path.is_file():
            if path.stat().st_nlink != 1:
                raise RuntimeError(f"{name} 禁止 hardlink: {relative}")
            actual_files.add(relative)
            total_bytes += path.stat().st_size
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            raise RuntimeError(f"{name} 禁止特殊文件: {relative}")
    if actual_files != expected_files or actual_directories != expected_directories:
        raise RuntimeError(
            f"{name} exact tree 漂移: files={sorted(actual_files ^ expected_files)}, "
            f"dirs={sorted(actual_directories ^ expected_directories)}"
        )
    return total_bytes


def _regular_tree_bytes(root: Path, *, name: str) -> int:
    """只统计普通单链接文件；用于成功 receipt 发布前的保守预算检查。"""

    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"{name} root 类型漂移")
    total_bytes = 0
    for path in root.rglob("*"):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            raise RuntimeError(f"{name} 禁止 symlink: {relative}")
        if path.is_file():
            if path.stat().st_nlink != 1:
                raise RuntimeError(f"{name} 禁止 hardlink: {relative}")
            total_bytes += path.stat().st_size
        elif not path.is_dir():
            raise RuntimeError(f"{name} 禁止特殊文件: {relative}")
    return total_bytes


def _output_root_identities(
    *,
    public_root: str | Path,
    private_root: str | Path,
    result_root: str | Path | None = None,
) -> dict[str, str]:
    """拒绝重复/嵌套输出，并只公开 realpath 的不可逆 identity。"""

    values: dict[str, Path] = {
        "public_output_identity_sha256": Path(public_root).resolve(),
        "private_output_identity_sha256": Path(private_root).resolve(),
    }
    if result_root is not None:
        values["result_output_identity_sha256"] = Path(result_root).resolve()
    paths = list(values.values())
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise RuntimeError("qualification public/private/result output 必须去重且互不嵌套")
    return {name: canonical_sha256({"absolute_path": str(path)}) for name, path in values.items()}


def _finalize_combined_artifact_accounting(
    *,
    result_root: Path,
    classification: str,
    config_sha256: str,
    output_identities: Mapping[str, str],
    public_execution_bytes: int,
    private_label_commit_bytes: int,
    private_scoring_consumption_marker_bytes: int,
    private_label_total_bytes: int,
    combined_budget_limit_bytes: int,
    final_scoring_state: Mapping[str, Any],
) -> dict[str, Any]:
    """以长度 fixed point 发布 accounting 与最终 state，避免自计数字节循环。"""

    stable_result_files = {
        "scoring_ledger.jsonl",
        "viewpoint_summaries.json",
        "qualification_summary.json",
        "qualification_receipt.json",
    }
    stable_result_bytes = sum((result_root / name).stat().st_size for name in stable_result_files)
    candidate_result_bytes = stable_result_bytes
    accounting: dict[str, Any] | None = None
    finalized_state: dict[str, Any] | None = None
    for _ in range(32):
        candidate_combined_bytes = (
            public_execution_bytes + private_label_total_bytes + candidate_result_bytes
        )
        accounting = {
            "version": _ARTIFACT_ACCOUNTING_VERSION,
            "status": "complete-combined-artifact-accounting",
            "classification": classification,
            "config_sha256": config_sha256,
            "output_identities": dict(output_identities),
            "public_execution_bytes": public_execution_bytes,
            "private_label_commit_bytes": private_label_commit_bytes,
            "private_scoring_consumption_marker_bytes": (private_scoring_consumption_marker_bytes),
            "private_label_total_bytes": private_label_total_bytes,
            "result_total_bytes": candidate_result_bytes,
            "combined_total_bytes": candidate_combined_bytes,
            "combined_budget_limit_bytes": combined_budget_limit_bytes,
            "accounting_semantics": (
                "exact-three-disjoint-trees-private-content-not-reopened-by-public-verifier/v1"
            ),
        }
        accounting["accounting_sha256"] = canonical_sha256(accounting)
        finalized_state = {
            **dict(final_scoring_state),
            "artifact_accounting_sha256": accounting["accounting_sha256"],
        }
        recomputed_result_bytes = (
            stable_result_bytes
            + len(_serialized_json_bytes(accounting))
            + len(_serialized_json_bytes(finalized_state))
        )
        if recomputed_result_bytes == candidate_result_bytes:
            break
        candidate_result_bytes = recomputed_result_bytes
    else:  # pragma: no cover - integer digit-length fixed point 必然快速收敛
        raise RuntimeError("qualification result artifact accounting 未收敛")
    assert accounting is not None and finalized_state is not None
    if accounting["combined_total_bytes"] > combined_budget_limit_bytes:
        raise RuntimeError("qualification combined artifact byte budget 超限")
    _atomic_create_json(result_root / "artifact_accounting.json", accounting)
    _atomic_replace_json(result_root / "scoring_state.json", finalized_state)
    actual_result_bytes = _verify_exact_regular_files(
        result_root,
        expected_files={
            *stable_result_files,
            "artifact_accounting.json",
            "scoring_state.json",
        },
        name="qualification result after combined accounting",
    )
    if actual_result_bytes != accounting["result_total_bytes"]:
        raise RuntimeError("qualification result artifact accounting 漂移")
    return accounting


def verify_g2c_qualification_execution(
    *,
    qualification_config_path: str | Path,
    public_execution_root: str | Path,
) -> dict[str, Any]:
    """公开重算 route/prediction execution；接口不接 labels/model/DATA。"""

    config = load_g2c_dynamic_qualification_config(qualification_config_path)
    root = Path(public_execution_root)
    freeze = _read_json(root / "execution_freeze.json", "qualification execution freeze")
    freeze_unsigned = dict(freeze)
    freeze_internal = freeze_unsigned.pop("freeze_sha256", None)
    if freeze_internal != canonical_sha256(freeze_unsigned):
        raise RuntimeError("qualification execution freeze internal SHA 漂移")
    classification = freeze.get("classification")
    if classification == QUALIFICATION_CLASSIFICATION_FORMAL:
        expected_seeds = FORMAL_QUALIFICATION_SEEDS
        expected_alternates = FRONT_ALTERNATE_IDS
        expected_route_count = 500
        expected_prediction_count = 550
    elif classification == QUALIFICATION_CLASSIFICATION_SMOKE:
        seeds = freeze.get("seeds")
        alternates = freeze.get("alternate_order")
        if (
            not isinstance(seeds, list)
            or len(seeds) != 1
            or type(seeds[0]) is not int
            or seeds[0] in FORMAL_QUALIFICATION_SEEDS
            or alternates != [FRONT_ALTERNATE_IDS[0]]
        ):
            raise RuntimeError("qualification smoke seed/alternate identity 漂移")
        expected_seeds = tuple(seeds)
        expected_alternates = (FRONT_ALTERNATE_IDS[0],)
        expected_route_count = 1
        expected_prediction_count = 2
    else:
        raise RuntimeError("qualification execution classification 漂移")
    if (
        freeze.get("version") != E018_P1_G2C_QUALIFICATION_EXECUTION_VERSION
        or freeze.get("status") != "routes-complete-context-destroy-pending"
        or freeze.get("config_sha256") != config["config_sha256"]
        or freeze.get("route_count") != expected_route_count
        or freeze.get("seed_count") != len(expected_seeds)
        or freeze.get("alternate_count") != len(expected_alternates)
        or freeze.get("seeds") != list(expected_seeds)
        or freeze.get("alternate_order") != list(expected_alternates)
        or freeze.get("test_split_status") != "prohibited-unread"
        or freeze.get("checkpoint_write_count") != 0
        or freeze.get("scoring_started") is not False
    ):
        raise RuntimeError("qualification execution freeze contract 漂移")
    if _read_json(root / "config_snapshot.json", "qualification config snapshot") != config:
        raise RuntimeError("qualification config snapshot 漂移")
    g0c_snapshot_path = root / "g0c_config_snapshot.json"
    g0c_config = _g0c.load_e018_p1_g0c_config(g0c_snapshot_path)
    g0c_snapshot = freeze.get("g0c_config_snapshot")
    expected_g0c_snapshot = {
        "canonical_sha256": config["parents"]["g0c_config_sha256"],
        "raw_sha256": file_sha256(g0c_snapshot_path),
        "size_bytes": g0c_snapshot_path.stat().st_size,
    }
    if (
        canonical_sha256(g0c_config) != config["parents"]["g0c_config_sha256"]
        or g0c_snapshot != expected_g0c_snapshot
    ):
        raise RuntimeError("qualification frozen G0C config snapshot 漂移")
    source = _read_json(root / "source_identity.json", "qualification source identity")
    source_unsigned = dict(source)
    source_internal = source_unsigned.pop("identity_sha256", None)
    if source_internal != canonical_sha256(source_unsigned) or source_internal != freeze.get(
        "source_identity_sha256"
    ):
        raise RuntimeError("qualification source identity 漂移")
    output_identities = _require_exact_keys(
        freeze.get("output_identities"),
        {"public_output_identity_sha256", "private_output_identity_sha256"},
        "qualification output identities",
    )
    if output_identities.get("public_output_identity_sha256") != canonical_sha256(
        {"absolute_path": str(root.resolve())}
    ):
        raise RuntimeError("qualification public output identity 漂移")
    if classification == QUALIFICATION_CLASSIFICATION_FORMAL:
        formal_decision = _verify_embedded_formal_execution_decision(
            freeze.get("formal_execution_decision"),
            config=config,
            qualification_config_raw_sha256=file_sha256(Path(qualification_config_path)),
            expected_source_git_commit=str(source.get("git_commit")),
            expected_source_identity_sha256=str(source_internal),
        )
        frozen_formal = formal_decision["receipt"]["formal_execution"]
        if any(frozen_formal.get(name) != value for name, value in output_identities.items()):
            raise RuntimeError("qualification D048/output identity 漂移")
    else:
        formal_decision = freeze.get("formal_execution_decision")
        if formal_decision != {"status": "not-applicable-preflight-no-qualification-claim"}:
            raise RuntimeError("qualification smoke formal decision 字段漂移")
    wall_seconds_max, gpu_seconds_max, combined_artifact_bytes_max = (
        _qualification_execution_budget_limits(
            classification=classification,
            config=config,
            execution_decision_verification=(
                formal_decision if classification == QUALIFICATION_CLASSIFICATION_FORMAL else None
            ),
        )
    )
    if freeze.get("budget_limits") != {
        "wall_seconds_max": wall_seconds_max,
        "gpu_seconds_max": gpu_seconds_max,
        "combined_artifact_bytes_max": combined_artifact_bytes_max,
    }:
        raise RuntimeError("qualification execution budget limits 漂移")
    parent = _read_json(root / "parent_verification.json", "qualification parent verification")
    parent_unsigned = dict(parent)
    parent_internal = parent_unsigned.pop("verification_sha256", None)
    expected_normalizer_identity = {
        name: config["parents"][name]
        for name in (
            "proprio_stats_sha256",
            "proprio_normalizer_sha256",
            "finger_force_stats_sha256",
            "finger_force_normalizer_sha256",
        )
    }
    if (
        set(parent)
        != {
            "config_sha256",
            "g0c_config_sha256",
            "g0c_receipt_internal_sha256",
            "g0c_config_snapshot_raw_sha256",
            "calibration_verification",
            "calibration_identities",
            "normalizer_identity",
            "verification_sha256",
        }
        or parent.get("normalizer_identity") != expected_normalizer_identity
        or parent_internal != canonical_sha256(parent_unsigned)
        or parent.get("config_sha256") != config["config_sha256"]
        or parent.get("g0c_config_sha256") != config["parents"]["g0c_config_sha256"]
        or parent.get("g0c_receipt_internal_sha256")
        != config["parents"]["g0c_receipt_internal_sha256"]
        or parent.get("g0c_config_snapshot_raw_sha256") != expected_g0c_snapshot["raw_sha256"]
        or parent.get("calibration_verification", {}).get("verification_sha256")
        != config["parents"]["calibration_result_verification_sha256"]
        or parent.get("calibration_identities") != _EXPECTED_CALIBRATION_IDENTITIES
        or freeze.get("parent_verification_sha256") != parent_internal
    ):
        raise RuntimeError("qualification parent verification 漂移")

    motion_rows = _read_jsonl(root / "camera_pose_ledger.jsonl", "qualification motion ledger")
    summaries = _read_jsonl(root / "route_summaries.jsonl", "qualification route summaries")
    predictions = _read_jsonl(root / "prediction_ledger.jsonl", "qualification prediction ledger")
    commit_receipts = _read_jsonl(
        root / "prediction_commit_ledger.jsonl",
        "qualification prediction commit ledger",
    )
    if (
        len(motion_rows) != expected_route_count * 92
        or len(summaries) != expected_route_count
        or len(predictions) != expected_prediction_count
        or len(commit_receipts) != expected_prediction_count
    ):
        raise RuntimeError("qualification execution ledger count 漂移")
    route_counters: list[dict[str, int]] = []
    route_rgb_inventory_rows = _verify_ordered_inventory(
        freeze.get("route_rgb_inventory"),
        version=_ROUTE_RGB_INVENTORY_VERSION,
        expected_count=expected_route_count,
        name="qualification route RGB inventory",
    )
    expected_routes = [
        (seed, alternate_index, viewpoint_id)
        for seed in expected_seeds
        for alternate_index, viewpoint_id in enumerate(expected_alternates)
    ]
    route_rows_by_identity: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for route_index, ((seed, alternate_index, viewpoint_id), summary) in enumerate(
        zip(expected_routes, summaries, strict=True)
    ):
        route_rows = motion_rows[route_index * 92 : (route_index + 1) * 92]
        route_rows_by_identity[(seed, alternate_index)] = route_rows
        if (
            summary.get("route_index") != route_index
            or summary.get("qualification_classification") != classification
        ):
            raise RuntimeError("qualification route summary order/classification 漂移")
        rgb_images, expected_rgb_inventory_row = _load_route_rgb_witnesses(
            root,
            route_index=route_index,
            episode_id=str(summary.get("episode_id")),
        )
        if route_rgb_inventory_rows[route_index] != expected_rgb_inventory_row:
            raise RuntimeError("qualification route RGB inventory/file 漂移")
        route_counters.append(
            validate_qualification_route_rows(
                route_rows,
                seed=seed,
                alternate_index=alternate_index,
                alternate_viewpoint_id=viewpoint_id,
                summary=summary,
                g0c_config=g0c_config,
                capture_safety=config["capture_safety"],
                rgb_images=rgb_images,
            )
        )
    totals = _sum_route_counters(route_counters)
    if totals != freeze.get("counters"):
        raise RuntimeError("qualification route totals 重算漂移")
    if classification == QUALIFICATION_CLASSIFICATION_FORMAL:
        validate_g2c_dynamic_qualification_counters(totals)

    expected_prediction_identities: list[tuple[dict[str, Any], Mapping[str, Any]]] = []
    for seed in expected_seeds:
        first_alternate = expected_alternates[0]
        expected_prediction_identities.append(
            (
                {
                    "seed": seed,
                    "sample_index": 0,
                    "viewpoint_id": FRONT_HOME_ID,
                    "frame_role": "home-anchor-first-route-only/v1",
                    "route_alternate_index": 0,
                    "route_alternate_viewpoint_id": first_alternate,
                    "route_frame_index": 0,
                },
                route_rows_by_identity[(seed, 0)][0],
            )
        )
        for alternate_index, viewpoint_id in enumerate(expected_alternates):
            expected_prediction_identities.append(
                (
                    {
                        "seed": seed,
                        "sample_index": alternate_index + 1,
                        "viewpoint_id": viewpoint_id,
                        "frame_role": "alternate-final-collect/v1",
                        "route_alternate_index": alternate_index,
                        "route_alternate_viewpoint_id": viewpoint_id,
                        "route_frame_index": _FINAL_COLLECT_FRAME_INDEX,
                    },
                    route_rows_by_identity[(seed, alternate_index)][_FINAL_COLLECT_FRAME_INDEX],
                )
            )
    previous: str | None = None
    for row_index, (prediction, commit, expected_identity) in enumerate(
        zip(predictions, commit_receipts, expected_prediction_identities, strict=True)
    ):
        prediction_unsigned = dict(prediction)
        internal = prediction_unsigned.pop("prediction_sha256", None)
        commit_unsigned = dict(commit)
        commit_internal = commit_unsigned.pop("commit_receipt_sha256", None)
        identity, route_row = expected_identity
        seed = identity["seed"]
        sample_index = identity["sample_index"]
        viewpoint_id = identity["viewpoint_id"]
        route_frame_index = identity["route_frame_index"]
        prediction_path = root / "prediction_commits" / f"{row_index:06d}.json"
        commit_path = root / "prediction_commits" / f"{row_index:06d}.commit.json"
        if (
            prediction.get("row_index") != row_index
            or prediction.get("seed") != seed
            or prediction.get("sample_index") != sample_index
            or prediction.get("viewpoint_id") != viewpoint_id
            or prediction.get("route_frame_index") != route_frame_index
            or prediction.get("previous_prediction_sha256") != previous
            or internal != canonical_sha256(prediction_unsigned)
            or _read_json(prediction_path, "qualification prediction commit") != prediction
            or file_sha256(prediction_path) != commit.get("prediction_raw_sha256")
            or _read_json(commit_path, "qualification commit receipt") != commit
            or commit.get("row_index") != row_index
            or commit.get("prediction_sha256") != internal
            or commit_internal != canonical_sha256(commit_unsigned)
            or type(commit.get("prediction_fsync_completed_at_unix_ns")) is not int
            or commit["prediction_fsync_completed_at_unix_ns"]
            <= prediction.get("prediction_write_started_at_unix_ns", -1)
        ):
            raise RuntimeError("qualification prediction identity/hash/commit 漂移")
        _validate_prediction_against_route_row(
            prediction,
            route_row=route_row,
            expected_identity={"row_index": row_index, **identity},
            config=config,
        )
        validate_qualification_prediction_mechanics(prediction, config=config)
        previous = str(internal)
    if previous != freeze.get("last_prediction_sha256"):
        raise RuntimeError("qualification final prediction hash 漂移")
    private_label_inventory_rows = _verify_ordered_inventory(
        freeze.get("private_label_journal_inventory"),
        version=_PRIVATE_LABEL_INVENTORY_VERSION,
        expected_count=expected_prediction_count,
        name="qualification private label journal inventory",
    )
    for row_index, (inventory_row, prediction) in enumerate(
        zip(private_label_inventory_rows, predictions, strict=True)
    ):
        if not isinstance(inventory_row, dict):
            raise RuntimeError("qualification private label inventory row 类型漂移")
        unsigned_inventory_row = dict(inventory_row)
        row_internal = unsigned_inventory_row.pop("row_sha256", None)
        hashes = (
            inventory_row.get("label_raw_sha256"),
            inventory_row.get("label_internal_sha256"),
            inventory_row.get("public_scoring_primitive_sha256"),
        )
        if (
            set(inventory_row)
            != {
                "row_index",
                "prediction_sha256",
                "label_raw_sha256",
                "label_internal_sha256",
                "size_bytes",
                "public_scoring_primitive_sha256",
                "row_sha256",
            }
            or inventory_row.get("row_index") != row_index
            or inventory_row.get("prediction_sha256") != prediction.get("prediction_sha256")
            or type(inventory_row.get("size_bytes")) is not int
            or inventory_row["size_bytes"] <= 0
            or row_internal != canonical_sha256(unsigned_inventory_row)
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes
            )
        ):
            raise RuntimeError("qualification private label inventory row/hash 漂移")

    for name, rows, path in (
        ("motion_ledger", motion_rows, root / "camera_pose_ledger.jsonl"),
        ("route_summaries", summaries, root / "route_summaries.jsonl"),
        ("prediction_ledger", predictions, root / "prediction_ledger.jsonl"),
        (
            "prediction_commit_ledger",
            commit_receipts,
            root / "prediction_commit_ledger.jsonl",
        ),
    ):
        frozen = freeze[name]
        if (
            frozen.get("row_count") != len(rows)
            or frozen.get("raw_sha256") != file_sha256(path)
            or frozen.get("size_bytes") != path.stat().st_size
        ):
            raise RuntimeError(f"qualification {name} freeze 漂移")
    contact_event_count = sum(
        bool(
            row.get("is_grasping") is True
            or float(row.get("robot_object_contact_force_n", math.inf))
            > config["capture_safety"]["maximum_robot_object_contact_force_n"]
        )
        for row in motion_rows
    )
    expected_permissions = {
        **config["permissions"],
        "simulator_isolated_camera_pose_set_count": totals["camera_pose_set_count"],
        "safe_hold_open_step_count": totals["safe_hold_open_step_count"],
        "object_contact_event_count": contact_event_count,
        "privileged_object_label_capture_count": expected_prediction_count,
        "goal_gt_read_count": 0,
    }
    if freeze.get("permission_counters") != expected_permissions:
        raise RuntimeError("qualification permission/contact counters 漂移")
    phase_state = _read_json(root / "phase_state.json", "qualification phase state")
    if (
        phase_state.get("status") != "complete-execution-freeze-context-destroyed"
        or phase_state.get("classification") != classification
        or phase_state.get("config_sha256") != config["config_sha256"]
        or phase_state.get("label_array_consumed") is not True
        or phase_state.get("prediction_commit_count") != expected_prediction_count
        or phase_state.get("privileged_access_started_count") != expected_prediction_count
        or phase_state.get("privileged_capture_count") != expected_prediction_count
        or phase_state.get("rerun_under_same_identity_allowed") is not False
    ):
        raise RuntimeError("qualification final phase state 漂移")
    receipt_path = root / "execution_receipt.json"
    receipt = _read_json(receipt_path, "qualification execution receipt")
    receipt_unsigned = dict(receipt)
    receipt_internal = receipt_unsigned.pop("receipt_sha256", None)
    timing_values = (
        receipt.get("wall_elapsed_seconds"),
        receipt.get("gpu_elapsed_seconds"),
    )
    if (
        receipt_internal != canonical_sha256(receipt_unsigned)
        or receipt.get("status") != "complete-execution-freeze-context-destroyed"
        or receipt.get("context_destroyed") is not True
        or receipt.get("phase_state_status") != phase_state["status"]
        or type(receipt.get("started_at_unix_ns")) is not int
        or type(receipt.get("completed_at_unix_ns")) is not int
        or receipt["completed_at_unix_ns"] <= receipt["started_at_unix_ns"]
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in timing_values
        )
        or float(receipt["wall_elapsed_seconds"]) > wall_seconds_max
        or float(receipt["gpu_elapsed_seconds"]) > gpu_seconds_max
        or not math.isclose(
            float(receipt["gpu_elapsed_seconds"]),
            float(receipt["wall_elapsed_seconds"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or receipt.get("gpu_elapsed_accounting")
        != "full-capture-process-monotonic-wall-conservative-upper-bound/v1"
        or any(receipt.get(name) != value for name, value in freeze.items() if name != "status")
    ):
        raise RuntimeError("qualification execution receipt 漂移")
    _validate_formal_execution_decision_time_order(
        execution_started_at_unix_ns=receipt["started_at_unix_ns"],
        classification=classification,
        formal_decision_verification=(
            formal_decision if classification == QUALIFICATION_CLASSIFICATION_FORMAL else None
        ),
    )
    commit_files = {
        f"prediction_commits/{index:06d}{suffix}.json"
        for index in range(expected_prediction_count)
        for suffix in ("", ".commit")
    }
    expected_top = {
        "config_snapshot.json",
        "g0c_config_snapshot.json",
        "source_identity.json",
        "parent_verification.json",
        "phase_state.json",
        "camera_pose_ledger.jsonl",
        "route_summaries.jsonl",
        "prediction_ledger.jsonl",
        "prediction_commit_ledger.jsonl",
        "execution_freeze.json",
        "execution_receipt.json",
    }
    image_files = {
        value["relative_path"]
        for row in route_rgb_inventory_rows
        for value in row["files"].values()
    }
    artifact_bytes = _verify_exact_regular_files(
        root,
        expected_files=expected_top | commit_files | image_files,
        expected_directories={"prediction_commits", "images"},
        name="qualification public execution",
    )
    if artifact_bytes > combined_artifact_bytes_max:
        raise RuntimeError("qualification execution artifact byte budget 超限")
    result = {
        "version": E018_P1_G2C_QUALIFICATION_EXECUTION_VERSION,
        "status": receipt["status"],
        "verified": True,
        "classification": classification,
        "config_sha256": config["config_sha256"],
        "source_git_commit": source["git_commit"],
        "source_identity_sha256": source_internal,
        "execution_freeze_sha256": freeze_internal,
        "execution_receipt_raw_sha256": file_sha256(receipt_path),
        "execution_receipt_internal_sha256": receipt_internal,
        "route_count": expected_route_count,
        "prediction_count": expected_prediction_count,
        "counters": totals,
        "permission_counters": expected_permissions,
        "started_at_unix_ns": receipt["started_at_unix_ns"],
        "completed_at_unix_ns": receipt["completed_at_unix_ns"],
        "wall_elapsed_seconds": receipt["wall_elapsed_seconds"],
        "gpu_elapsed_seconds": receipt["gpu_elapsed_seconds"],
        "combined_artifact_bytes_max": combined_artifact_bytes_max,
        "output_identities": dict(output_identities),
        "formal_execution_decision": formal_decision,
        "private_label_journal_inventory": freeze["private_label_journal_inventory"],
        "artifact_bytes": artifact_bytes,
    }
    result["verification_sha256"] = canonical_sha256(result)
    return result


def verify_g2c_qualification_failure(
    *,
    qualification_config_path: str | Path,
    public_execution_root: str | Path,
) -> dict[str, Any]:
    """机械验证 consumed failure、已落盘 full/partial route 与 prediction chain。"""

    config = load_g2c_dynamic_qualification_config(qualification_config_path)
    root = Path(public_execution_root)
    failure_path = root / "consumed_failure.json"
    failure = _read_json(failure_path, "qualification consumed failure")
    _require_exact_keys(
        failure,
        {
            "version",
            "status",
            "classification",
            "config_sha256",
            "prediction_commit_count",
            "privileged_access_started_count",
            "privileged_capture_count",
            "label_array_consumed",
            "rerun_under_same_identity_allowed",
            "error_type",
            "error_message",
            "cleanup_errors",
            "failure_evidence",
            "failed_at_unix_ns",
            "failure_sha256",
        },
        "qualification consumed failure",
    )
    unsigned = dict(failure)
    failure_internal = unsigned.pop("failure_sha256", None)
    classification = failure.get("classification")
    if (
        failure_internal != canonical_sha256(unsigned)
        or failure.get("version") != E018_P1_G2C_QUALIFICATION_EXECUTION_VERSION
        or failure.get("status") != "consumed-qualification-failure"
        or classification
        not in {
            QUALIFICATION_CLASSIFICATION_FORMAL,
            QUALIFICATION_CLASSIFICATION_SMOKE,
        }
        or failure.get("config_sha256") != config["config_sha256"]
        or type(failure.get("prediction_commit_count")) is not int
        or type(failure.get("privileged_access_started_count")) is not int
        or type(failure.get("privileged_capture_count")) is not int
        or min(
            failure["prediction_commit_count"],
            failure["privileged_access_started_count"],
            failure["privileged_capture_count"],
        )
        < 0
        or failure["privileged_capture_count"] > failure["privileged_access_started_count"]
        or failure["privileged_access_started_count"] > failure["prediction_commit_count"]
        or failure.get("label_array_consumed")
        is not (failure["privileged_access_started_count"] > 0)
        or failure.get("rerun_under_same_identity_allowed") is not False
        or not isinstance(failure.get("error_type"), str)
        or not isinstance(failure.get("error_message"), str)
        or not isinstance(failure.get("cleanup_errors"), list)
        or any(not isinstance(value, str) for value in failure["cleanup_errors"])
        or type(failure.get("failed_at_unix_ns")) is not int
        or failure["failed_at_unix_ns"] <= 0
    ):
        raise RuntimeError("qualification consumed failure identity/state 漂移")
    state = _read_json(root / "phase_state.json", "qualification failure phase state")
    if (
        state.get("status") != "consumed-qualification-failure"
        or state.get("classification") != classification
        or state.get("config_sha256") != config["config_sha256"]
        or state.get("prediction_commit_count") != failure["prediction_commit_count"]
        or state.get("privileged_access_started_count")
        != failure["privileged_access_started_count"]
        or state.get("privileged_capture_count") != failure["privileged_capture_count"]
        or state.get("label_array_consumed") is not failure["label_array_consumed"]
        or state.get("rerun_under_same_identity_allowed") is not False
    ):
        raise RuntimeError("qualification failure phase-state 漂移")

    evidence = failure.get("failure_evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError("qualification failure evidence 类型漂移")
    if evidence:
        evidence_unsigned = dict(evidence)
        evidence_internal = evidence_unsigned.pop("evidence_sha256", None)
        if evidence_internal != canonical_sha256(evidence_unsigned):
            raise RuntimeError("qualification failure evidence hash 漂移")

        def verify_ledger_snapshot(name: str, snapshot: Any, expected_path: Path) -> int:
            if snapshot is None:
                if expected_path.exists():
                    raise RuntimeError(f"qualification failure {name} 未绑定现存文件")
                return 0
            _require_exact_keys(
                snapshot,
                {"writer_row_count", "raw_sha256", "size_bytes"},
                f"qualification failure {name}",
            )
            if (
                expected_path.is_symlink()
                or not expected_path.is_file()
                or expected_path.stat().st_nlink != 1
                or snapshot.get("raw_sha256") != file_sha256(expected_path)
                or snapshot.get("size_bytes") != expected_path.stat().st_size
                or type(snapshot.get("writer_row_count")) is not int
                or snapshot["writer_row_count"] < 0
            ):
                raise RuntimeError(f"qualification failure {name} file/hash 漂移")
            rows = _read_jsonl(expected_path, f"qualification failure {name}")
            if len(rows) != snapshot["writer_row_count"]:
                raise RuntimeError(f"qualification failure {name} row count 漂移")
            return len(rows)

        motion_count = verify_ledger_snapshot(
            "motion ledger",
            evidence.get("motion_ledger"),
            root / "camera_pose_ledger.jsonl",
        )
        summary_count = verify_ledger_snapshot(
            "route summaries",
            evidence.get("route_summaries"),
            root / "route_summaries.jsonl",
        )
        partial = evidence.get("partial_route_ledger")
        if partial is None:
            partial_count = 0
            if (root / "partial_route_ledger.jsonl").exists():
                raise RuntimeError("qualification failure partial ledger 未绑定")
        else:
            # partial freeze 使用 AppendOnlyJsonl.freeze 的字段名。
            translated = {
                "writer_row_count": partial.get("row_count"),
                "raw_sha256": partial.get("raw_sha256"),
                "size_bytes": partial.get("size_bytes"),
            }
            partial_count = verify_ledger_snapshot(
                "partial route ledger",
                translated,
                root / "partial_route_ledger.jsonl",
            )
        if (
            type(evidence.get("validated_route_count")) is not int
            or type(evidence.get("persisted_full_route_count")) is not int
            or evidence["validated_route_count"] < 0
            or evidence["persisted_full_route_count"] < evidence["validated_route_count"]
            or motion_count != evidence["persisted_full_route_count"] * 92
            or summary_count != evidence["persisted_full_route_count"]
            or type(evidence.get("active_route_row_count")) is not int
            or evidence["active_route_row_count"] < 0
            or (
                evidence.get("active_route_persisted_in_main_ledger") is False
                and partial_count != evidence["active_route_row_count"]
            )
            or (
                evidence.get("active_route_persisted_in_main_ledger") is True and partial_count != 0
            )
        ):
            raise RuntimeError("qualification failure full/partial route count 漂移")

    previous: str | None = None
    for row_index in range(failure["prediction_commit_count"]):
        prediction_path = root / "prediction_commits" / f"{row_index:06d}.json"
        receipt_path = root / "prediction_commits" / f"{row_index:06d}.commit.json"
        prediction = _read_json(prediction_path, "qualification failed prediction")
        receipt = _read_json(receipt_path, "qualification failed prediction receipt")
        prediction_unsigned = dict(prediction)
        prediction_internal = prediction_unsigned.pop("prediction_sha256", None)
        receipt_unsigned = dict(receipt)
        receipt_internal = receipt_unsigned.pop("commit_receipt_sha256", None)
        if (
            prediction.get("row_index") != row_index
            or prediction.get("previous_prediction_sha256") != previous
            or prediction_internal != canonical_sha256(prediction_unsigned)
            or receipt.get("row_index") != row_index
            or receipt.get("prediction_sha256") != prediction_internal
            or receipt.get("prediction_raw_sha256") != file_sha256(prediction_path)
            or receipt_internal != canonical_sha256(receipt_unsigned)
        ):
            raise RuntimeError("qualification failed prediction chain 漂移")
        validate_qualification_prediction_mechanics(prediction, config=config)
        previous = prediction_internal

    expected_files = {"phase_state.json", "consumed_failure.json"}
    for name in (
        "config_snapshot.json",
        "g0c_config_snapshot.json",
        "source_identity.json",
        "parent_verification.json",
        "camera_pose_ledger.jsonl",
        "route_summaries.jsonl",
        "partial_route_ledger.jsonl",
        "prediction_ledger.jsonl",
        "prediction_commit_ledger.jsonl",
        "execution_freeze.json",
    ):
        if (root / name).exists():
            expected_files.add(name)
    expected_directories: set[str] = set()
    if (root / "prediction_commits").exists():
        expected_directories.add("prediction_commits")
        expected_files.update(
            f"prediction_commits/{index:06d}{suffix}.json"
            for index in range(failure["prediction_commit_count"])
            for suffix in ("", ".commit")
        )
    rgb_inventory_rows = evidence.get("route_rgb_inventory_rows", [])
    if not isinstance(rgb_inventory_rows, list):
        raise RuntimeError("qualification failure RGB inventory rows 类型漂移")
    image_files: set[str] = set()
    for inventory_row in rgb_inventory_rows:
        if not isinstance(inventory_row, Mapping):
            raise RuntimeError("qualification failure RGB inventory row 类型漂移")
        files = inventory_row.get("files")
        if not isinstance(files, Mapping):
            raise RuntimeError("qualification failure RGB inventory files 类型漂移")
        for item in files.values():
            if (
                not isinstance(item, Mapping)
                or not isinstance(item.get("relative_path"), str)
                or not item["relative_path"].startswith("images/")
            ):
                raise RuntimeError("qualification failure RGB inventory path 漂移")
            image_files.add(str(item["relative_path"]))
    if (root / "images").exists():
        expected_directories.add("images")
    expected_files.update(image_files)
    artifact_bytes = _verify_exact_regular_files(
        root,
        expected_files=expected_files,
        expected_directories=expected_directories,
        name="qualification consumed failure",
    )

    result = {
        "version": E018_P1_G2C_QUALIFICATION_EXECUTION_VERSION,
        "status": "verified-consumed-qualification-failure",
        "verified": True,
        "classification": classification,
        "config_sha256": config["config_sha256"],
        "failure_raw_sha256": file_sha256(failure_path),
        "failure_internal_sha256": failure_internal,
        "prediction_commit_count": failure["prediction_commit_count"],
        "artifact_bytes": artifact_bytes,
        "rerun_under_same_identity_allowed": False,
    }
    result["verification_sha256"] = canonical_sha256(result)
    return result


def _run_qualification_capture(
    *,
    qualification_config_path: str | Path,
    g0c_config_path: str | Path,
    g0c_receipt_path: str | Path,
    calibration_config_path: str | Path,
    calibration_prediction_freeze_root: str | Path,
    calibration_result_root: str | Path,
    data_config_path: str | Path,
    stats_root: str | Path,
    selected_checkpoint_path: str | Path,
    repository_root: str | Path,
    public_output_root: str | Path,
    private_label_output_root: str | Path,
    seeds: Sequence[int],
    alternate_ids: Sequence[str],
    classification: str,
    expected_source_git_commit: str | None,
    expected_source_identity_sha256: str | None,
    execution_decision_verification: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """共享 formal/smoke capture；不评分、不读取已写 private label。"""

    capture_started_at_unix_ns = time.time_ns()
    capture_started_monotonic_s = time.monotonic()
    _validate_formal_execution_decision_time_order(
        execution_started_at_unix_ns=capture_started_at_unix_ns,
        classification=classification,
        formal_decision_verification=execution_decision_verification,
    )
    import gymnasium as gym
    import mani_skill
    import sapien
    import torch
    from mani_skill.utils import sapien_utils

    from robot_vla.sim import register_robot_vla_maniskill_envs

    public_root = Path(public_output_root)
    private_root = Path(private_label_output_root)
    output_identities = _output_root_identities(public_root=public_root, private_root=private_root)
    if public_root.exists() or private_root.exists():
        raise FileExistsError("qualification public/private output 必须全新")
    config = load_g2c_dynamic_qualification_config(qualification_config_path)
    parent_verification = verify_g2c_qualification_parents(
        qualification_config_path=qualification_config_path,
        g0c_config_path=g0c_config_path,
        g0c_receipt_path=g0c_receipt_path,
        calibration_config_path=calibration_config_path,
        calibration_prediction_freeze_root=calibration_prediction_freeze_root,
        calibration_result_root=calibration_result_root,
        data_config_path=data_config_path,
    )
    g0c_config = _g0c.load_e018_p1_g0c_config(g0c_config_path)
    data_config = parent_verification["data_config"]
    source_identity = _git_source_identity(Path(repository_root))
    if classification == QUALIFICATION_CLASSIFICATION_FORMAL:
        if (
            source_identity["git_commit"] != expected_source_git_commit
            or source_identity["identity_sha256"] != expected_source_identity_sha256
            or execution_decision_verification is None
            or execution_decision_verification.get("verified") is not True
            or execution_decision_verification.get("receipt", {})
            .get("source", {})
            .get("git_commit")
            != source_identity["git_commit"]
            or execution_decision_verification.get("receipt", {})
            .get("source", {})
            .get("identity_sha256")
            != source_identity["identity_sha256"]
            or any(
                execution_decision_verification.get("receipt", {})
                .get("formal_execution", {})
                .get(name)
                != value
                for name, value in output_identities.items()
            )
        ):
            raise RuntimeError("formal qualification D048/exact-source R2 identity 漂移")
    elif classification != QUALIFICATION_CLASSIFICATION_SMOKE:
        raise ValueError("qualification classification 未冻结")
    elif execution_decision_verification is not None:
        raise RuntimeError("qualification smoke 不得绑定 formal decision receipt")
    wall_seconds_max, gpu_seconds_max, combined_artifact_bytes_max = (
        _qualification_execution_budget_limits(
            classification=classification,
            config=config,
            execution_decision_verification=execution_decision_verification,
        )
    )
    if (
        mani_skill.__version__ != data_config["software"]["expected_mani_skill_version"]
        or sapien.__version__ != data_config["software"]["expected_sapien_version"]
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("qualification CUDA/ManiSkill/SAPIEN environment 漂移")
    spec, proprio_normalizer, force_normalizer, normalizer_identity = _load_normalizers(
        stats_root=Path(stats_root), config=data_config
    )
    provider = QualificationProvider(
        checkpoint_path=selected_checkpoint_path,
        qualification_config=config,
        data_config=data_config,
        classification=classification,
    )
    home, anchors, orientations = _g0c._parse_library(g0c_config)
    primitives = _g0c._expand_primitives(anchors, orientations)
    primitive_by_id = {item.viewpoint_id: (item, orientation) for item, orientation in primitives}
    if tuple(primitive_by_id) != FRONT_ALTERNATE_IDS:
        provider.destroy()
        raise RuntimeError("qualification G0C primitive order 漂移")
    route_config = json.loads(json.dumps(g0c_config))
    route_config["experiment"]["offline_segmentation_diagnostics"] = False
    route_config["experiment"]["save_settled_rgb"] = True
    journal: QualificationJournal | None = None
    env: Any | None = None
    motion_writer: _AppendOnlyJsonl | None = None
    summary_writer: _AppendOnlyJsonl | None = None
    route_rgb_inventory_rows: list[dict[str, Any]] = []
    route_counters: list[dict[str, int]] = []
    active_route_rows: list[dict[str, Any]] = []
    active_route_identity: dict[str, Any] | None = None
    active_route_persisted = False
    persisted_route_count = 0
    try:
        register_robot_vla_maniskill_envs()
        environment = g0c_config["environment"]
        env = gym.make(
            environment["environment_id"],
            obs_mode=environment["obs_mode"],
            control_mode=environment["control_mode"],
            num_envs=environment["num_envs"],
            robot_uids=environment["robot_uid"],
        )
        base_env = env.unwrapped
        if (
            base_env.control_freq != environment["control_hz"]
            or environment["camera_uid"] != data_config["environment"]["external_camera_uid"]
            or environment["camera_uid"] not in base_env._sensors
        ):
            raise RuntimeError("qualification environment/camera/control identity 漂移")
        sensor = base_env._sensors[environment["camera_uid"]]
        camera = sensor.camera
        if sensor.entity is not None or not callable(getattr(camera, "set_local_pose", None)):
            raise RuntimeError("qualification 要求 isolated unmounted RenderCamera")

        journal = QualificationJournal(
            public_root=public_root,
            private_label_root=private_root,
            config_sha256=config["config_sha256"],
            classification=classification,
        )
        _atomic_create_json(public_root / "config_snapshot.json", config)
        g0c_snapshot_path = public_root / "g0c_config_snapshot.json"
        g0c_snapshot_raw_sha256, _ = _atomic_create_json(g0c_snapshot_path, g0c_config)
        _atomic_create_json(public_root / "source_identity.json", source_identity)
        parent_public = {
            "config_sha256": parent_verification["config_sha256"],
            "g0c_config_sha256": parent_verification["g0c_config_sha256"],
            "g0c_receipt_internal_sha256": parent_verification["g0c_receipt_internal_sha256"],
            "g0c_config_snapshot_raw_sha256": g0c_snapshot_raw_sha256,
            "calibration_verification": parent_verification["calibration_verification"],
            "calibration_identities": parent_verification["calibration_identities"],
            "normalizer_identity": normalizer_identity,
        }
        parent_public["verification_sha256"] = canonical_sha256(parent_public)
        _atomic_create_json(public_root / "parent_verification.json", parent_public)
        motion_writer = _AppendOnlyJsonl(public_root / "camera_pose_ledger.jsonl")
        summary_writer = _AppendOnlyJsonl(public_root / "route_summaries.jsonl")
        object_contact_event_count = 0
        route_index = 0
        for seed in seeds:
            for alternate_viewpoint_id in alternate_ids:
                alternate_index = FRONT_ALTERNATE_IDS.index(alternate_viewpoint_id)
                primitive, orientation = primitive_by_id[alternate_viewpoint_id]
                before = journal.prediction_count
                active_route_rows = []
                active_route_identity = {
                    "seed": seed,
                    "alternate_index": alternate_index,
                    "alternate_viewpoint_id": alternate_viewpoint_id,
                    "intended_route_index": route_index,
                }
                active_route_persisted = False

                def frame_hook(
                    row: dict[str, Any],
                    rgb: np.ndarray,
                    observation: dict[str, Any],
                    *,
                    current_seed: int = seed,
                    current_alternate_index: int = alternate_index,
                    current_viewpoint: str = alternate_viewpoint_id,
                ) -> None:
                    active_route_rows.append(row)
                    process_qualification_hook_frame(
                        motion_row=row,
                        rgb=rgb,
                        observation=observation,
                        seed=current_seed,
                        alternate_index=current_alternate_index,
                        alternate_viewpoint_id=current_viewpoint,
                        base_env=base_env,
                        spec=spec,
                        proprio_normalizer=proprio_normalizer,
                        finger_force_normalizer=force_normalizer,
                        data_config=data_config,
                        provider=provider,
                        journal=journal,
                    )

                rows, summary, _ = _g0._run_route(
                    env=env,
                    base_env=base_env,
                    camera=camera,
                    config=route_config,
                    seed=seed,
                    home=home,
                    alternate=primitive,
                    output_root=public_root,
                    sapien_module=sapien,
                    sapien_utils_module=sapien_utils,
                    alternate_orientation=orientation,
                    result_version=E018_P1_G2C_QUALIFICATION_EXECUTION_VERSION,
                    episode_prefix="g2c-qualification",
                    source_phase=QUALIFICATION_SOURCE_PHASE,
                    camera_owner=QUALIFICATION_CAMERA_OWNER,
                    frame_hook=frame_hook,
                    include_raw_safety_witnesses=True,
                )
                expected_provider_count = 1 + int(alternate_index == 0)
                actual_provider_count = journal.prediction_count - before
                qualification_summary = {
                    **summary,
                    "diagnostics": {
                        **summary["diagnostics"],
                        "rgb_numeric_evidence_source": (
                            "three-public-png-pixel-witnesses-recomputed/v1"
                        ),
                    },
                    "route_index": route_index,
                    "qualification_classification": classification,
                    "provider_forward_count": actual_provider_count,
                    "privileged_capture_count": actual_provider_count,
                    "offline_segmentation_diagnostics": False,
                }
                # 完整 route 一旦返回就先 append+fsync；后续 protocol/gate
                # 失败也必须保留原始 92 帧与当时 summary，禁止重跑替换。
                _append_returned_route_evidence(
                    motion_writer=motion_writer,
                    summary_writer=summary_writer,
                    rows=rows,
                    summary=qualification_summary,
                )
                active_route_persisted = True
                persisted_route_count += 1
                if actual_provider_count != expected_provider_count:
                    raise RuntimeError("qualification per-route provider count 漂移")
                rgb_images, rgb_inventory_row = _load_route_rgb_witnesses(
                    public_root,
                    route_index=route_index,
                    episode_id=str(summary["episode_id"]),
                )
                route_rgb_inventory_rows.append(rgb_inventory_row)
                counter = validate_qualification_route_rows(
                    rows,
                    seed=seed,
                    alternate_index=alternate_index,
                    alternate_viewpoint_id=alternate_viewpoint_id,
                    summary=qualification_summary,
                    g0c_config=g0c_config,
                    capture_safety=config["capture_safety"],
                    rgb_images=rgb_images,
                )
                route_counters.append(counter)
                object_contact_event_count += sum(
                    bool(
                        row["is_grasping"]
                        or float(row["robot_object_contact_force_n"])
                        > config["capture_safety"]["maximum_robot_object_contact_force_n"]
                    )
                    for row in rows
                )
                route_index += 1
                active_route_rows = []
                active_route_identity = None
                active_route_persisted = False
                if time.monotonic() - capture_started_monotonic_s > min(
                    wall_seconds_max, gpu_seconds_max
                ):
                    raise RuntimeError("qualification capture wall/GPU budget 超限")

        totals = _sum_route_counters(route_counters)
        expected_route_count = len(seeds) * len(alternate_ids)
        if route_index != expected_route_count:
            raise RuntimeError("qualification route count 漂移")
        if classification == QUALIFICATION_CLASSIFICATION_FORMAL:
            validate_g2c_dynamic_qualification_counters(totals)
            if (
                tuple(seeds) != FORMAL_QUALIFICATION_SEEDS
                or tuple(alternate_ids) != FRONT_ALTERNATE_IDS
                or route_index != 500
            ):
                raise RuntimeError("formal qualification seed/view/route identity 漂移")
        else:
            if totals != {
                "camera_pose_set_count": 97,
                "moving_interpolation_command_count": 80,
                "safe_hold_open_step_count": 96,
                "ledger_frame_count": 92,
                "provider_scored_home_frame_count": 1,
                "provider_scored_alternate_frame_count": 1,
                "provider_scored_frame_count": 2,
            }:
                raise RuntimeError("qualification smoke exact counters 漂移")
        if (
            provider.forward_count != totals["provider_scored_frame_count"]
            or journal.prediction_count != provider.forward_count
            or journal.privileged_capture_count != provider.forward_count
            or journal.privileged_access_started_count != provider.forward_count
        ):
            raise RuntimeError("qualification provider/prediction/GT count 漂移")

        motion_freeze = motion_writer.freeze()
        summary_freeze = summary_writer.freeze()
        predictions, commit_receipts = _load_public_prediction_chain(journal)
        prediction_writer = _AppendOnlyJsonl(public_root / "prediction_ledger.jsonl")
        prediction_writer.append(predictions)
        prediction_freeze = prediction_writer.freeze()
        commit_writer = _AppendOnlyJsonl(public_root / "prediction_commit_ledger.jsonl")
        commit_writer.append(commit_receipts)
        commit_freeze = commit_writer.freeze()
        journal.freeze_route_completion()
        route_rgb_inventory = _freeze_ordered_inventory(
            route_rgb_inventory_rows,
            version=_ROUTE_RGB_INVENTORY_VERSION,
        )
        private_label_inventory = journal.private_label_inventory()
        execution_freeze = {
            "version": E018_P1_G2C_QUALIFICATION_EXECUTION_VERSION,
            "status": "routes-complete-context-destroy-pending",
            "classification": classification,
            "config_sha256": config["config_sha256"],
            "source_identity_sha256": source_identity["identity_sha256"],
            "parent_verification_sha256": parent_public["verification_sha256"],
            "g0c_config_snapshot": {
                "canonical_sha256": canonical_sha256(g0c_config),
                "raw_sha256": g0c_snapshot_raw_sha256,
                "size_bytes": g0c_snapshot_path.stat().st_size,
            },
            "route_count": route_index,
            "seed_count": len(seeds),
            "alternate_count": len(alternate_ids),
            "seeds": list(seeds),
            "alternate_order": list(alternate_ids),
            "counters": totals,
            "motion_ledger": motion_freeze,
            "route_summaries": summary_freeze,
            "prediction_ledger": prediction_freeze,
            "prediction_commit_ledger": commit_freeze,
            "route_rgb_inventory": route_rgb_inventory,
            "private_label_journal_inventory": private_label_inventory,
            "output_identities": output_identities,
            "budget_limits": {
                "wall_seconds_max": wall_seconds_max,
                "gpu_seconds_max": gpu_seconds_max,
                "combined_artifact_bytes_max": combined_artifact_bytes_max,
            },
            "formal_execution_decision": (
                dict(execution_decision_verification)
                if execution_decision_verification is not None
                else {"status": "not-applicable-preflight-no-qualification-claim"}
            ),
            "last_prediction_sha256": journal.previous_prediction_sha256,
            "permission_counters": {
                **config["permissions"],
                "simulator_isolated_camera_pose_set_count": totals["camera_pose_set_count"],
                "safe_hold_open_step_count": totals["safe_hold_open_step_count"],
                "object_contact_event_count": object_contact_event_count,
                "privileged_object_label_capture_count": journal.privileged_capture_count,
                "goal_gt_read_count": 0,
            },
            "test_split_status": "prohibited-unread",
            "checkpoint_write_count": 0,
            "scoring_started": False,
        }
        execution_freeze["freeze_sha256"] = canonical_sha256(execution_freeze)
        _atomic_create_json(public_root / "execution_freeze.json", execution_freeze)
        environment_identity = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(torch.device("cuda")),
            "mani_skill": mani_skill.__version__,
            "sapien": sapien.__version__,
            "external_camera_unmounted": sensor.entity is None,
            "camera_class": type(camera).__module__ + "." + type(camera).__name__,
        }
    except Exception as error:
        failure_evidence: dict[str, Any]
        try:
            failure_evidence = _freeze_capture_failure_evidence(
                public_root=public_root,
                motion_writer=motion_writer,
                summary_writer=summary_writer,
                route_counters=route_counters,
                persisted_route_count=persisted_route_count,
                active_route_identity=active_route_identity,
                active_route_rows=active_route_rows,
                active_route_persisted=active_route_persisted,
                route_rgb_inventory_rows=route_rgb_inventory_rows,
                journal=journal,
            )
        except Exception as evidence_error:
            failure_evidence = {
                "version": "e018-p1-g2c-capture-failure-evidence/v1",
                "evidence_capture_error": (f"{type(evidence_error).__name__}:{evidence_error}"),
            }
            failure_evidence["evidence_sha256"] = canonical_sha256(failure_evidence)
        _best_effort_freeze_capture_failure(
            env=env,
            provider=provider,
            journal=journal,
            error=error,
            evidence=failure_evidence,
        )
        raise
    else:
        return _finalize_qualification_execution(
            env=env,
            provider=provider,
            journal=journal,
            public_root=public_root,
            execution_freeze=execution_freeze,
            environment_identity=environment_identity,
            private_root=private_root,
            combined_artifact_bytes_max=combined_artifact_bytes_max,
            capture_started_at_unix_ns=capture_started_at_unix_ns,
            capture_started_monotonic_s=capture_started_monotonic_s,
            wall_seconds_max=wall_seconds_max,
            gpu_seconds_max=gpu_seconds_max,
        )


def run_e018_p1_g2c_qualification_capture(
    *,
    qualification_config_path: str | Path,
    g0c_config_path: str | Path,
    g0c_receipt_path: str | Path,
    calibration_config_path: str | Path,
    calibration_prediction_freeze_root: str | Path,
    calibration_result_root: str | Path,
    data_config_path: str | Path,
    stats_root: str | Path,
    selected_checkpoint_path: str | Path,
    repository_root: str | Path,
    public_output_root: str | Path,
    private_label_output_root: str | Path,
    expected_source_git_commit: str,
    expected_source_identity_sha256: str,
    decision_execution_go: bool,
    decision_receipt_path: str | Path | None = None,
    expected_decision_receipt_raw_sha256: str | None = None,
    expected_decision_receipt_internal_sha256: str | None = None,
) -> dict[str, Any]:
    """执行一次 formal 500-route capture；D047 阶段入口保持 HOLD。"""

    if decision_execution_go is not True:
        raise PermissionError("G2C formal dynamic qualification 仍为 HOLD")
    if (
        decision_receipt_path is None
        or expected_decision_receipt_raw_sha256 is None
        or expected_decision_receipt_internal_sha256 is None
    ):
        raise PermissionError("G2C formal qualification 缺独立 D048 GO receipt")
    config = load_g2c_dynamic_qualification_config(qualification_config_path)
    decision_verification = verify_g2c_formal_execution_decision_receipt(
        decision_receipt_path=decision_receipt_path,
        expected_raw_sha256=expected_decision_receipt_raw_sha256,
        expected_internal_sha256=expected_decision_receipt_internal_sha256,
        qualification_config=config,
        qualification_config_raw_sha256=file_sha256(Path(qualification_config_path)),
        expected_source_git_commit=expected_source_git_commit,
        expected_source_identity_sha256=expected_source_identity_sha256,
    )
    return _run_qualification_capture(
        qualification_config_path=qualification_config_path,
        g0c_config_path=g0c_config_path,
        g0c_receipt_path=g0c_receipt_path,
        calibration_config_path=calibration_config_path,
        calibration_prediction_freeze_root=calibration_prediction_freeze_root,
        calibration_result_root=calibration_result_root,
        data_config_path=data_config_path,
        stats_root=stats_root,
        selected_checkpoint_path=selected_checkpoint_path,
        repository_root=repository_root,
        public_output_root=public_output_root,
        private_label_output_root=private_label_output_root,
        seeds=FORMAL_QUALIFICATION_SEEDS,
        alternate_ids=FRONT_ALTERNATE_IDS,
        classification=QUALIFICATION_CLASSIFICATION_FORMAL,
        expected_source_git_commit=expected_source_git_commit,
        expected_source_identity_sha256=expected_source_identity_sha256,
        execution_decision_verification=decision_verification,
    )


def run_e018_p1_g2c_qualification_smoke(
    *,
    qualification_config_path: str | Path,
    g0c_config_path: str | Path,
    g0c_receipt_path: str | Path,
    calibration_config_path: str | Path,
    calibration_prediction_freeze_root: str | Path,
    calibration_result_root: str | Path,
    data_config_path: str | Path,
    stats_root: str | Path,
    selected_checkpoint_path: str | Path,
    repository_root: str | Path,
    public_output_root: str | Path,
    private_label_output_root: str | Path,
    seed: int,
    alternate_viewpoint_id: str = "LEFT_LOW__CENTER",
) -> dict[str, Any]:
    """最多 1 noncanonical seed x 1 route 的 D047 preflight。"""

    if (
        type(seed) is not int
        or seed in FORMAL_QUALIFICATION_SEEDS
        or alternate_viewpoint_id != FRONT_ALTERNATE_IDS[0]
    ):
        raise ValueError("qualification smoke 必须是 noncanonical seed 与首个冻结 alternate")
    data_config = load_e018_p1_g2c_data_config(data_config_path)
    if seed not in data_config["sampling"]["smoke_only_seeds"]:
        raise ValueError("qualification smoke seed 不属于冻结 smoke-only seeds")
    return _run_qualification_capture(
        qualification_config_path=qualification_config_path,
        g0c_config_path=g0c_config_path,
        g0c_receipt_path=g0c_receipt_path,
        calibration_config_path=calibration_config_path,
        calibration_prediction_freeze_root=calibration_prediction_freeze_root,
        calibration_result_root=calibration_result_root,
        data_config_path=data_config_path,
        stats_root=stats_root,
        selected_checkpoint_path=selected_checkpoint_path,
        repository_root=repository_root,
        public_output_root=public_output_root,
        private_label_output_root=private_label_output_root,
        seeds=(seed,),
        alternate_ids=(alternate_viewpoint_id,),
        classification=QUALIFICATION_CLASSIFICATION_SMOKE,
        expected_source_git_commit=None,
        expected_source_identity_sha256=None,
        execution_decision_verification=None,
    )


def score_e018_p1_g2c_qualification(
    *,
    qualification_config_path: str | Path,
    public_execution_root: str | Path,
    private_label_root: str | Path,
    result_output_root: str | Path,
    decision_scoring_go: bool,
) -> dict[str, Any]:
    """context destroy 后一次性读取 private journal；接口不接模型/checkpoint/DATA。"""

    if decision_scoring_go is not True:
        raise PermissionError("G2C qualification scoring 仍为 HOLD")
    config = load_g2c_dynamic_qualification_config(qualification_config_path)
    execution = verify_g2c_qualification_execution(
        qualification_config_path=qualification_config_path,
        public_execution_root=public_execution_root,
    )
    prediction_rows = _read_jsonl(
        Path(public_execution_root) / "prediction_ledger.jsonl",
        "qualification scoring predictions",
    )
    expected_count = execution["prediction_count"]
    label_inventory = execution["private_label_journal_inventory"]
    label_inventory_rows = label_inventory["rows"]
    public_root = Path(public_execution_root)
    private_root = Path(private_label_root)
    result_root = Path(result_output_root)
    output_identities = _output_root_identities(
        public_root=public_root,
        private_root=private_root,
        result_root=result_root,
    )
    if any(
        output_identities.get(name) != value
        for name, value in execution["output_identities"].items()
    ):
        raise RuntimeError("qualification scorer public/private output identity 漂移")
    if execution["classification"] == QUALIFICATION_CLASSIFICATION_FORMAL:
        frozen_formal = execution["formal_execution_decision"]["receipt"]["formal_execution"]
        if any(frozen_formal.get(name) != value for name, value in output_identities.items()):
            raise RuntimeError("qualification scorer D048 result output identity 漂移")
    if result_root.exists():
        raise FileExistsError("qualification result output 必须全新")
    expected_label_files = {f"label_commits/{index:06d}.json" for index in range(expected_count)}
    private_label_commit_bytes = _verify_exact_regular_files(
        private_root,
        expected_files=expected_label_files,
        expected_directories={"label_commits"},
        name="qualification private label journal before scoring",
    )
    if private_label_commit_bytes != sum(int(row["size_bytes"]) for row in label_inventory_rows):
        raise RuntimeError("qualification private label inventory byte accounting 漂移")
    result_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    scoring_state = {
        "version": E018_P1_G2C_QUALIFICATION_RESULT_VERSION,
        "status": "initialized-before-private-label-open",
        "classification": execution["classification"],
        "config_sha256": config["config_sha256"],
        "execution_receipt_internal_sha256": execution["execution_receipt_internal_sha256"],
        "private_label_journal_inventory_sha256": label_inventory["inventory_sha256"],
        "label_journal_consumed": False,
        "label_journal_open_count": 0,
        "rerun_under_same_identity_allowed": False,
        "created_at_unix_ns": time.time_ns(),
    }
    _atomic_replace_json(result_root / "scoring_state.json", scoring_state)
    marker = {
        "version": E018_P1_G2C_QUALIFICATION_RESULT_VERSION,
        "status": "private-label-scoring-consumption-started",
        "config_sha256": config["config_sha256"],
        "execution_receipt_internal_sha256": execution["execution_receipt_internal_sha256"],
        "private_label_journal_inventory_sha256": label_inventory["inventory_sha256"],
        "result_output_identity_sha256": canonical_sha256(
            {"absolute_path": str(result_root.resolve())}
        ),
        "rerun_under_same_identity_allowed": False,
        "consumption_started_at_unix_ns": time.time_ns(),
    }
    marker["marker_sha256"] = canonical_sha256(marker)
    marker_path = private_root / "SCORING_CONSUMED.json"
    try:
        marker_raw_sha256, _ = _atomic_create_json(marker_path, marker)
        marker_size_bytes = marker_path.stat().st_size
        private_label_total_bytes = _verify_exact_regular_files(
            private_root,
            expected_files={*expected_label_files, "SCORING_CONSUMED.json"},
            expected_directories={"label_commits"},
            name="qualification private label journal after scoring marker",
        )
        if private_label_total_bytes != (private_label_commit_bytes + marker_size_bytes):
            raise RuntimeError("qualification private label marker byte accounting 漂移")
        scoring_state.update(
            {
                "status": "private-label-scoring-in-progress",
                "label_journal_consumed": True,
                "private_label_consumption_marker_raw_sha256": marker_raw_sha256,
                "private_label_consumption_marker_internal_sha256": marker["marker_sha256"],
            }
        )
        _atomic_replace_json(result_root / "scoring_state.json", scoring_state)
        scoring_rows: list[dict[str, Any]] = []
        for row_index, prediction in enumerate(prediction_rows):
            label_path = private_root / "label_commits" / f"{row_index:06d}.json"
            label, label_raw_sha256 = _read_json_once_with_raw_sha(
                label_path, "qualification private label"
            )
            _validate_qualification_object_label(label, committed=True)
            label_unsigned = dict(label)
            label_internal = label_unsigned.pop("label_sha256", None)
            inventory_row = label_inventory_rows[row_index]
            public_scoring_primitive_sha256 = canonical_sha256(
                _qualification_public_scoring_primitive(label)
            )
            prediction_commit = _read_json(
                Path(public_execution_root) / "prediction_commits" / f"{row_index:06d}.commit.json",
                "qualification scoring prediction commit",
            )
            if (
                label_internal != canonical_sha256(label_unsigned)
                or inventory_row.get("row_index") != row_index
                or inventory_row.get("prediction_sha256") != prediction.get("prediction_sha256")
                or inventory_row.get("label_raw_sha256") != label_raw_sha256
                or inventory_row.get("label_internal_sha256") != label_internal
                or inventory_row.get("size_bytes") != label_path.stat().st_size
                or inventory_row.get("public_scoring_primitive_sha256")
                != public_scoring_primitive_sha256
                or label.get("row_index") != row_index
                or label.get("prediction_sha256") != prediction.get("prediction_sha256")
                or label.get("prediction_raw_sha256")
                != prediction_commit.get("prediction_raw_sha256")
                or label.get("prediction_commit_receipt_sha256")
                != prediction_commit.get("commit_receipt_sha256")
                or label.get("prediction_write_started_at_unix_ns")
                != prediction.get("prediction_write_started_at_unix_ns")
                or label.get("prediction_fsync_completed_at_unix_ns")
                != prediction_commit.get("prediction_fsync_completed_at_unix_ns")
                or type(label.get("privileged_captured_at_unix_ns")) is not int
                or label["privileged_captured_at_unix_ns"]
                <= label["prediction_fsync_completed_at_unix_ns"]
            ):
                raise RuntimeError("qualification label identity/commit order 漂移")
            scored = score_qualification_prediction(prediction, label, config=config)
            scored["label_raw_sha256"] = label_raw_sha256
            scored["public_scoring_primitive_sha256"] = public_scoring_primitive_sha256
            scoring_rows.append(scored)
            scoring_state["label_journal_open_count"] = row_index + 1
            _atomic_replace_json(result_root / "scoring_state.json", scoring_state)
        if len(scoring_rows) != expected_count:
            raise RuntimeError("qualification scoring row count 漂移")
        present_viewpoints = tuple(
            viewpoint_id
            for viewpoint_id in QUALIFICATION_VIEW_ORDER
            if any(row["viewpoint_id"] == viewpoint_id for row in scoring_rows)
        )
        viewpoint_summaries = [
            summarize_qualification_viewpoint(
                [row for row in scoring_rows if row["viewpoint_id"] == viewpoint_id],
                viewpoint_id=viewpoint_id,
                config=config,
            )
            for viewpoint_id in present_viewpoints
        ]
        if execution["classification"] == QUALIFICATION_CLASSIFICATION_FORMAL:
            if present_viewpoints != QUALIFICATION_VIEW_ORDER:
                raise RuntimeError("formal qualification viewpoint summary 不完整")
            primary = select_qualification_primary(viewpoint_summaries, config=config)
        else:
            primary = {
                "status": "not-applicable-preflight-no-qualification-claim",
                "primary_viewpoint_id": None,
                "qualified_non_home_viewpoint_ids": [],
                "selection_key": None,
            }
        summary = build_qualification_result_summary(
            execution_verification=execution,
            viewpoint_summaries=viewpoint_summaries,
            primary=primary,
            label_open_count=expected_count,
            private_label_consumption_marker_raw_sha256=marker_raw_sha256,
            private_label_consumption_marker_internal_sha256=marker["marker_sha256"],
        )
        scoring_writer = _AppendOnlyJsonl(result_root / "scoring_ledger.jsonl")
        scoring_writer.append(scoring_rows)
        scoring_writer.freeze()
        _atomic_create_json(
            result_root / "viewpoint_summaries.json",
            {
                "version": E018_P1_G2C_QUALIFICATION_RESULT_VERSION,
                "viewpoint_order": list(present_viewpoints),
                "summaries": viewpoint_summaries,
            },
        )
        _atomic_create_json(result_root / "qualification_summary.json", summary)
        receipt = {
            **summary,
            "artifact_sha256": {
                name: file_sha256(result_root / name)
                for name in (
                    "scoring_ledger.jsonl",
                    "viewpoint_summaries.json",
                    "qualification_summary.json",
                )
            },
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        _atomic_create_json(result_root / "qualification_receipt.json", receipt)
        final_scoring_state = {
            **scoring_state,
            "status": "complete-qualification-scoring",
            "qualification_receipt_internal_sha256": receipt["receipt_sha256"],
            "completed_at_unix_ns": time.time_ns(),
        }
        _finalize_combined_artifact_accounting(
            result_root=result_root,
            classification=execution["classification"],
            config_sha256=config["config_sha256"],
            output_identities=output_identities,
            public_execution_bytes=int(execution["artifact_bytes"]),
            private_label_commit_bytes=private_label_commit_bytes,
            private_scoring_consumption_marker_bytes=marker_size_bytes,
            private_label_total_bytes=private_label_total_bytes,
            combined_budget_limit_bytes=int(execution["combined_artifact_bytes_max"]),
            final_scoring_state=final_scoring_state,
        )
        return receipt
    except Exception as error:
        failure = {
            "version": E018_P1_G2C_QUALIFICATION_RESULT_VERSION,
            "status": "consumed-qualification-scoring-failure",
            "config_sha256": config["config_sha256"],
            "execution_receipt_internal_sha256": execution["execution_receipt_internal_sha256"],
            "label_journal_consumed": marker_path.exists(),
            "label_journal_open_count": scoring_state["label_journal_open_count"],
            "rerun_under_same_identity_allowed": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "failed_at_unix_ns": time.time_ns(),
        }
        failure["failure_sha256"] = canonical_sha256(failure)
        _atomic_create_json(result_root / "consumed_failure.json", failure)
        scoring_state.update(
            {
                "status": failure["status"],
                "label_journal_consumed": marker_path.exists(),
                "failure_sha256": failure["failure_sha256"],
            }
        )
        _atomic_replace_json(result_root / "scoring_state.json", scoring_state)
        raise


def _assert_structured_derived_equal(
    actual: Any,
    expected: Any,
    *,
    tolerance: float,
    path: str,
) -> None:
    if type(expected) is bool:
        if type(actual) is not bool or actual is not expected:
            raise RuntimeError(f"qualification derived bool 漂移: {path}")
        return
    if expected is None or isinstance(expected, str):
        if actual != expected:
            raise RuntimeError(f"qualification derived field 漂移: {path}")
        return
    if isinstance(expected, int):
        if type(actual) is not int or actual != expected:
            raise RuntimeError(f"qualification derived integer 漂移: {path}")
        return
    if isinstance(expected, float):
        if (
            not isinstance(actual, (int, float))
            or isinstance(actual, bool)
            or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tolerance)
        ):
            raise RuntimeError(f"qualification derived numeric 漂移: {path}")
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise RuntimeError(f"qualification derived object keys 漂移: {path}")
        for key, value in expected.items():
            _assert_structured_derived_equal(
                actual[key],
                value,
                tolerance=tolerance,
                path=f"{path}.{key}",
            )
        return
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if (
            not isinstance(actual, Sequence)
            or isinstance(actual, (str, bytes))
            or len(actual) != len(expected)
        ):
            raise RuntimeError(f"qualification derived sequence 漂移: {path}")
        for index, value in enumerate(expected):
            _assert_structured_derived_equal(
                actual[index],
                value,
                tolerance=tolerance,
                path=f"{path}[{index}]",
            )
        return
    if actual != expected:
        raise RuntimeError(f"qualification derived value 漂移: {path}")


def verify_g2c_qualification_result(
    *,
    qualification_config_path: str | Path,
    public_execution_root: str | Path,
    result_root: str | Path,
) -> dict[str, Any]:
    """公开重算最终结果；签名不接 private label/model/checkpoint/DATA。"""

    config = load_g2c_dynamic_qualification_config(qualification_config_path)
    execution = verify_g2c_qualification_execution(
        qualification_config_path=qualification_config_path,
        public_execution_root=public_execution_root,
    )
    execution_root = Path(public_execution_root)
    root = Path(result_root)
    predictions = _read_jsonl(
        execution_root / "prediction_ledger.jsonl",
        "qualification result predictions",
    )
    commits = _read_jsonl(
        execution_root / "prediction_commit_ledger.jsonl",
        "qualification result prediction commits",
    )
    scoring_rows = _read_jsonl(root / "scoring_ledger.jsonl", "qualification scoring ledger")
    if len(scoring_rows) != execution["prediction_count"]:
        raise RuntimeError("qualification scoring ledger count 漂移")
    tolerance = float(config["qualification"]["metric_float_recompute_tolerance"])
    label_inventory_rows = execution["private_label_journal_inventory"]["rows"]
    recomputed_rows: list[dict[str, Any]] = []
    for row_index, (prediction, commit, stored) in enumerate(
        zip(predictions, commits, scoring_rows, strict=True)
    ):
        label_raw = stored.get("label_raw_sha256")
        if (
            not isinstance(label_raw, str)
            or len(label_raw) != 64
            or any(character not in "0123456789abcdef" for character in label_raw)
            or stored.get("prediction_fsync_completed_at_unix_ns")
            != commit.get("prediction_fsync_completed_at_unix_ns")
            or type(stored.get("privileged_captured_at_unix_ns")) is not int
            or stored["privileged_captured_at_unix_ns"]
            <= stored["prediction_fsync_completed_at_unix_ns"]
        ):
            raise RuntimeError("qualification public scoring label/order evidence 漂移")
        label = {
            "row_index": row_index,
            "prediction_sha256": prediction["prediction_sha256"],
            "gt_object_exists": stored["gt_object_exists"],
            "gt_observable": stored["gt_observable"],
            "gt_object_position_base_m": stored["gt_object_position_base_m"],
            "is_grasped": stored["is_grasped"],
            "robot_object_contact_force_n": stored["robot_object_contact_force_n"],
            "goal_gt_read_count": stored["goal_gt_read_count"],
            "test_data_read": stored["test_data_read"],
            "label_sha256": stored["label_sha256"],
            "prediction_fsync_completed_at_unix_ns": stored[
                "prediction_fsync_completed_at_unix_ns"
            ],
            "privileged_captured_at_unix_ns": stored["privileged_captured_at_unix_ns"],
        }
        primitive_sha256 = canonical_sha256(_qualification_public_scoring_primitive(label))
        inventory_row = label_inventory_rows[row_index]
        if (
            stored.get("label_sha256") != inventory_row.get("label_internal_sha256")
            or label_raw != inventory_row.get("label_raw_sha256")
            or primitive_sha256 != inventory_row.get("public_scoring_primitive_sha256")
            or stored.get("public_scoring_primitive_sha256") != primitive_sha256
        ):
            raise RuntimeError("qualification public scoring primitive commitment 漂移")
        expected = score_qualification_prediction(prediction, label, config=config)
        expected["label_raw_sha256"] = label_raw
        expected["public_scoring_primitive_sha256"] = primitive_sha256
        if expected["row_index"] != row_index:
            raise RuntimeError("qualification scoring row_index 漂移")
        _assert_structured_derived_equal(
            stored,
            expected,
            tolerance=tolerance,
            path=f"scoring_rows[{row_index}]",
        )
        recomputed_rows.append(expected)
    viewpoint_document = _read_json(
        root / "viewpoint_summaries.json", "qualification viewpoint summaries"
    )
    present_viewpoints = tuple(
        viewpoint_id
        for viewpoint_id in QUALIFICATION_VIEW_ORDER
        if any(row["viewpoint_id"] == viewpoint_id for row in recomputed_rows)
    )
    recomputed_summaries = [
        summarize_qualification_viewpoint(
            [row for row in recomputed_rows if row["viewpoint_id"] == viewpoint_id],
            viewpoint_id=viewpoint_id,
            config=config,
        )
        for viewpoint_id in present_viewpoints
    ]
    expected_viewpoint_document = {
        "version": E018_P1_G2C_QUALIFICATION_RESULT_VERSION,
        "viewpoint_order": list(present_viewpoints),
        "summaries": recomputed_summaries,
    }
    _assert_structured_derived_equal(
        viewpoint_document,
        expected_viewpoint_document,
        tolerance=tolerance,
        path="viewpoint_summaries",
    )
    if execution["classification"] == QUALIFICATION_CLASSIFICATION_FORMAL:
        recomputed_primary = select_qualification_primary(recomputed_summaries, config=config)
    else:
        recomputed_primary = {
            "status": "not-applicable-preflight-no-qualification-claim",
            "primary_viewpoint_id": None,
            "qualified_non_home_viewpoint_ids": [],
            "selection_key": None,
        }
    stored_summary = _read_json(root / "qualification_summary.json", "qualification summary")
    for name in (
        "private_label_consumption_marker_raw_sha256",
        "private_label_consumption_marker_internal_sha256",
    ):
        value = stored_summary.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"qualification {name} 漂移")
    recomputed_summary = build_qualification_result_summary(
        execution_verification=execution,
        viewpoint_summaries=recomputed_summaries,
        primary=recomputed_primary,
        label_open_count=execution["prediction_count"],
        private_label_consumption_marker_raw_sha256=stored_summary[
            "private_label_consumption_marker_raw_sha256"
        ],
        private_label_consumption_marker_internal_sha256=stored_summary[
            "private_label_consumption_marker_internal_sha256"
        ],
    )
    _assert_structured_derived_equal(
        stored_summary,
        recomputed_summary,
        tolerance=tolerance,
        path="qualification_summary",
    )
    receipt_path = root / "qualification_receipt.json"
    receipt = _read_json(receipt_path, "qualification result receipt")
    expected_artifacts = {
        name: file_sha256(root / name)
        for name in (
            "scoring_ledger.jsonl",
            "viewpoint_summaries.json",
            "qualification_summary.json",
        )
    }
    expected_receipt = {
        **stored_summary,
        "artifact_sha256": expected_artifacts,
    }
    expected_receipt["receipt_sha256"] = canonical_sha256(expected_receipt)
    _assert_structured_derived_equal(
        receipt,
        expected_receipt,
        tolerance=tolerance,
        path="qualification_receipt",
    )
    accounting_path = root / "artifact_accounting.json"
    accounting = _read_json(accounting_path, "qualification combined artifact accounting")
    _require_exact_keys(
        accounting,
        _ARTIFACT_ACCOUNTING_KEYS,
        "qualification combined artifact accounting",
    )
    accounting_unsigned = dict(accounting)
    accounting_internal = accounting_unsigned.pop("accounting_sha256", None)
    scoring_state = _read_json(root / "scoring_state.json", "qualification scoring state")
    if (
        scoring_state.get("status") != "complete-qualification-scoring"
        or scoring_state.get("classification") != execution["classification"]
        or scoring_state.get("config_sha256") != config["config_sha256"]
        or scoring_state.get("execution_receipt_internal_sha256")
        != execution["execution_receipt_internal_sha256"]
        or scoring_state.get("private_label_journal_inventory_sha256")
        != execution["private_label_journal_inventory"]["inventory_sha256"]
        or scoring_state.get("label_journal_consumed") is not True
        or scoring_state.get("label_journal_open_count") != execution["prediction_count"]
        or scoring_state.get("rerun_under_same_identity_allowed") is not False
        or scoring_state.get("qualification_receipt_internal_sha256") != receipt["receipt_sha256"]
        or scoring_state.get("artifact_accounting_sha256") != accounting_internal
        or type(scoring_state.get("created_at_unix_ns")) is not int
        or type(scoring_state.get("completed_at_unix_ns")) is not int
        or scoring_state["completed_at_unix_ns"] < scoring_state["created_at_unix_ns"]
    ):
        raise RuntimeError("qualification scoring completion state 漂移")
    artifact_bytes = _verify_exact_regular_files(
        root,
        expected_files={
            "scoring_ledger.jsonl",
            "viewpoint_summaries.json",
            "qualification_summary.json",
            "qualification_receipt.json",
            "artifact_accounting.json",
            "scoring_state.json",
        },
        name="qualification result",
    )
    expected_output_identities = {
        **execution["output_identities"],
        "result_output_identity_sha256": canonical_sha256({"absolute_path": str(root.resolve())}),
    }
    private_commit_bytes = sum(
        int(row["size_bytes"]) for row in execution["private_label_journal_inventory"]["rows"]
    )
    marker_bytes = accounting.get("private_scoring_consumption_marker_bytes")
    private_total_bytes = accounting.get("private_label_total_bytes")
    combined_total_bytes = accounting.get("combined_total_bytes")
    if (
        accounting_internal != canonical_sha256(accounting_unsigned)
        or accounting.get("version") != _ARTIFACT_ACCOUNTING_VERSION
        or accounting.get("status") != "complete-combined-artifact-accounting"
        or accounting.get("classification") != execution["classification"]
        or accounting.get("config_sha256") != config["config_sha256"]
        or accounting.get("output_identities") != expected_output_identities
        or accounting.get("public_execution_bytes") != execution["artifact_bytes"]
        or accounting.get("private_label_commit_bytes") != private_commit_bytes
        or type(marker_bytes) is not int
        or marker_bytes <= 0
        or type(private_total_bytes) is not int
        or private_total_bytes != private_commit_bytes + marker_bytes
        or accounting.get("result_total_bytes") != artifact_bytes
        or type(combined_total_bytes) is not int
        or combined_total_bytes
        != execution["artifact_bytes"] + private_total_bytes + artifact_bytes
        or accounting.get("combined_budget_limit_bytes") != execution["combined_artifact_bytes_max"]
        or combined_total_bytes > execution["combined_artifact_bytes_max"]
        or accounting.get("accounting_semantics")
        != "exact-three-disjoint-trees-private-content-not-reopened-by-public-verifier/v1"
    ):
        raise RuntimeError("qualification combined artifact accounting 漂移")
    result = {
        "version": E018_P1_G2C_QUALIFICATION_RESULT_VERSION,
        "status": stored_summary["status"],
        "verified": True,
        "protocol_valid": stored_summary["protocol_valid"],
        "gate_passed": stored_summary["gate_passed"],
        "classification": execution["classification"],
        "config_sha256": config["config_sha256"],
        "source_git_commit": execution["source_git_commit"],
        "source_identity_sha256": execution["source_identity_sha256"],
        "execution_verification_sha256": execution["verification_sha256"],
        "receipt_raw_sha256": file_sha256(receipt_path),
        "receipt_internal_sha256": receipt["receipt_sha256"],
        "scoring_row_count": len(scoring_rows),
        "viewpoint_summary_count": len(recomputed_summaries),
        "primary_viewpoint_id": recomputed_primary["primary_viewpoint_id"],
        "qualified_non_home_viewpoint_ids": stored_summary["qualified_non_home_viewpoint_ids"],
        "unsafe_accepted_count": stored_summary["unsafe_accepted_count"],
        "catastrophic_accepted_count": stored_summary["catastrophic_accepted_count"],
        "label_journal_reopen_count": 0,
        "public_execution_bytes": execution["artifact_bytes"],
        "private_label_bytes": private_total_bytes,
        "result_artifact_bytes": artifact_bytes,
        "combined_artifact_bytes": combined_total_bytes,
        "combined_artifact_budget_bytes": execution["combined_artifact_bytes_max"],
        "artifact_accounting_sha256": accounting_internal,
    }
    result["verification_sha256"] = canonical_sha256(result)
    return result


def verify_g2c_qualification_combined_artifacts(
    *,
    qualification_config_path: str | Path,
    public_execution_root: str | Path,
    private_label_root: str | Path,
    result_root: str | Path,
) -> dict[str, Any]:
    """R2 artifact verifier：只审文件树/hash/bytes，不重开 private label 内容。"""

    execution = verify_g2c_qualification_execution(
        qualification_config_path=qualification_config_path,
        public_execution_root=public_execution_root,
    )
    result = verify_g2c_qualification_result(
        qualification_config_path=qualification_config_path,
        public_execution_root=public_execution_root,
        result_root=result_root,
    )
    expected_count = int(execution["prediction_count"])
    private_root = Path(private_label_root)
    expected_label_files = {f"label_commits/{index:06d}.json" for index in range(expected_count)}
    private_bytes = _verify_exact_regular_files(
        private_root,
        expected_files={*expected_label_files, "SCORING_CONSUMED.json"},
        expected_directories={"label_commits"},
        name="qualification private label journal artifact-only verification",
    )
    # 这里只重算原始文件摘要，不解析 private label JSON。这样既能保持
    # public verifier 的 label-content reopen count 为 0，又能发现评分后对
    # private tree 的等长篡改，避免“文件数/总字节未变”被误报为三树完整。
    label_inventory_rows = execution["private_label_journal_inventory"]["rows"]
    for row_index, inventory_row in enumerate(label_inventory_rows):
        label_path = private_root / "label_commits" / f"{row_index:06d}.json"
        if (
            inventory_row.get("row_index") != row_index
            or inventory_row.get("size_bytes") != label_path.stat().st_size
            or inventory_row.get("label_raw_sha256") != file_sha256(label_path)
        ):
            raise RuntimeError("qualification private label raw SHA/size 漂移")
    summary = _read_json(
        Path(result_root) / "qualification_summary.json",
        "qualification summary for private marker commitment",
    )
    marker_path = private_root / "SCORING_CONSUMED.json"
    combined_bytes = (
        int(execution["artifact_bytes"]) + private_bytes + int(result["result_artifact_bytes"])
    )
    if (
        file_sha256(marker_path) != summary.get("private_label_consumption_marker_raw_sha256")
        or private_bytes != result["private_label_bytes"]
        or combined_bytes != result["combined_artifact_bytes"]
        or combined_bytes > result["combined_artifact_budget_bytes"]
    ):
        raise RuntimeError("qualification exact three-tree artifact accounting 漂移")
    verification = {
        "version": _ARTIFACT_ACCOUNTING_VERSION,
        "status": "verified-exact-three-tree-artifact-accounting",
        "verified": True,
        "classification": execution["classification"],
        "public_execution_bytes": execution["artifact_bytes"],
        "private_label_bytes": private_bytes,
        "result_artifact_bytes": result["result_artifact_bytes"],
        "combined_artifact_bytes": combined_bytes,
        "combined_artifact_budget_bytes": result["combined_artifact_budget_bytes"],
        "label_content_reopen_count": 0,
        "result_verification_sha256": result["verification_sha256"],
    }
    verification["verification_sha256"] = canonical_sha256(verification)
    return verification


__all__ = [
    "E018_P1_G2C_QUALIFICATION_CONFIG_VERSION",
    "E018_P1_G2C_QUALIFICATION_EXECUTION_VERSION",
    "E018_P1_G2C_QUALIFICATION_RESULT_VERSION",
    "FORMAL_QUALIFICATION_SEEDS",
    "QUALIFICATION_VIEW_ORDER",
    "QUALIFICATION_CLASSIFICATION_FORMAL",
    "QUALIFICATION_CLASSIFICATION_SMOKE",
    "QUALIFICATION_SOURCE_PHASE",
    "QUALIFICATION_CAMERA_OWNER",
    "QualificationProvider",
    "QualificationJournal",
    "assert_qualification_prediction_deployable_only",
    "build_qualification_deployable_capture",
    "capture_qualification_object_label",
    "finalize_qualification_prediction",
    "load_g2c_dynamic_qualification_config",
    "process_qualification_hook_frame",
    "qualification_deployable_safe",
    "qualification_scored_frame_identity",
    "run_e018_p1_g2c_qualification_capture",
    "run_e018_p1_g2c_qualification_smoke",
    "score_e018_p1_g2c_qualification",
    "score_qualification_prediction",
    "select_qualification_primary",
    "summarize_qualification_viewpoint",
    "validate_qualification_route_rows",
    "validate_qualification_prediction_mechanics",
    "verify_g2c_formal_execution_decision_receipt",
    "verify_g2c_qualification_combined_artifacts",
    "verify_g2c_qualification_execution",
    "verify_g2c_qualification_failure",
    "verify_g2c_qualification_parents",
    "verify_g2c_qualification_result",
]
