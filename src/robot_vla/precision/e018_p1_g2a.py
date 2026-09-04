"""E018-P1 G2A：冻结 wrist checkpoint 的 front camera 逐视角资格验证。

本模块只允许 development qualification。运行分成不可交换的两段：先仅用可部署
输入生成并冻结 prediction ledger；随后才重置同一组 simulator seeds，读取 privileged
segmentation/pose 做离线评分。它不读 test arrays、不访问 live Memory，也不执行机器人或
运行时相机动作。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla.adapters import (
    FingerForceNormalizer,
    FingerForceStats,
    FrankaObservationAdapter,
    ProprioNormalizer,
    ProprioStats,
)
from robot_vla.contracts import RobotSpec
from robot_vla.data.trajectory import load_manifest, resolve_trajectory_path
from robot_vla.observation import (
    invert_se3,
    opengl_camera_to_opencv,
    rotation_matrix_to_6d,
    validate_se3,
)
from robot_vla.precision.active_external_observation import (
    ACTUAL_EXTERNAL_POSE_SOURCE,
    _closest_rigid_transform,
)
from robot_vla.precision.active_front_camera import (
    ExternalCameraMotionState,
    FrontCameraOrientationMode,
    FrontCameraViewpoint,
)
from robot_vla.precision.active_front_provider import (
    ACTIVE_FRONT_PROVIDER_ADAPTER_VERSION,
    FRONT_PROVIDER_FRAME_CONVENTION,
    FRONT_PROVIDER_ROLE_SUBSTITUTION_SEMANTICS,
    ActiveFrontProviderAdapterConfig,
    ActiveFrontProviderIdentity,
    build_active_front_model_input,
    build_precision_camera_role_state,
)
from robot_vla.precision.geometry import project_base_point_to_normalized_uv
from robot_vla.precision.object_observability import (
    ObjectWriteEvidence,
    derive_object_observability,
)
from robot_vla.precision.observability import mask_probability_at_normalized_uv
from robot_vla.precision.outliers import geometry_conditioning
from robot_vla.precision.provider import PrecisionGeometricMotionInput

E018_P1_G2A_CONFIG_VERSION = (
    "e018-p1-g2a-front-provider-qualification-development/v1"
)
E018_P1_G2A_RESULT_VERSION = "e018-p1-g2a-front-provider-qualification-result/v1"
E018_P1_G2A_GATE = "G2A_FRONT_PROVIDER_QUALIFICATION"

NATIVE_WRIST_CONTROL_ID = "WRIST_NATIVE"
FRONT_HOME_ID = "HOME__CENTER"
FRONT_ALTERNATE_IDS = (
    "LEFT_LOW__CENTER",
    "LEFT_LOW__YAW_LEFT",
    "LEFT_LOW__YAW_RIGHT",
    "LEFT_LOW__PITCH_UP",
    "LEFT_LOW__PITCH_DOWN",
    "RIGHT_LOW__CENTER",
    "RIGHT_LOW__YAW_LEFT",
    "RIGHT_LOW__YAW_RIGHT",
    "RIGHT_LOW__PITCH_UP",
    "RIGHT_LOW__PITCH_DOWN",
)
PER_SCENE_CAPTURE_ORDER = (NATIVE_WRIST_CONTROL_ID, FRONT_HOME_ID, *FRONT_ALTERNATE_IDS)
G0B_SHORTLIST_ORDER = (
    "LEFT_LOW__YAW_LEFT",
    "LEFT_LOW__PITCH_UP",
    "RIGHT_LOW__YAW_RIGHT",
    "RIGHT_LOW__YAW_LEFT",
)
PRIMARY_TIE_BREAK_FIELDS = (
    "shortlist_tier_ascending",
    "accepted_safe_coverage_descending",
    "observable_world_xyz_p90_m_ascending",
    "observable_world_xyz_max_m_ascending",
    "absolute_covariance_coverage_minus_0_95_ascending",
    "visibility_recall_descending",
    "frozen_shortlist_order_then_primitive_id_ascending",
)
_SOURCE_FILES = (
    "src/robot_vla/precision/active_front_provider.py",
    "src/robot_vla/precision/e018_p1_g2a.py",
    "src/robot_vla/cli/run_e018_p1_g2a.py",
    "configs/e018_p1_g2a_front_provider_qualification_development_v1.json",
)
_ARTIFACT_NAMES = (
    "config_snapshot.json",
    "source_identity.json",
    "seed_audit.json",
    "wrist_pose_envelope.json",
    "deployable_capture_audit.json",
    "inference_audit.json",
    "prediction_ledger.jsonl",
    "prediction_freeze.json",
    "offline_scoring_ledger.jsonl",
    "offline_scoring_audit.json",
    "summary.json",
    "report.md",
)
_SHA256_PATTERN = __import__("re").compile(r"[0-9a-f]{64}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    dtype = str(array.dtype).encode("ascii")
    digest.update(len(dtype).to_bytes(2, "big"))
    digest.update(dtype)
    digest.update(len(array.shape).to_bytes(2, "big"))
    for dimension in array.shape:
        digest.update(int(dimension).to_bytes(8, "big", signed=False))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
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
        _fsync_parent_directory(path)
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
        _fsync_parent_directory(path)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_parent_directory(path: Path) -> None:
    """持久化原子 replace 的目录项，避免只落盘临时文件内容。"""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    directory_fd = os.open(path.parent, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _require_keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是 JSON object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys 漂移: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} 必须是 64 位小写 SHA-256")
    return value


def _probability(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} 必须是 [0,1] 内有限概率")
    return result


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} 必须是有限正数")
    return result


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")
    return value


def load_e018_p1_g2a_config(
    path: str | Path,
    *,
    parent_g0c_config_path: str | Path,
) -> dict[str, Any]:
    """严格加载 pre-result G2A 配置，并绑定通过的 G0C parent。"""

    from robot_vla.precision.e018_p1_g0c import load_e018_p1_g0c_config

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"E018-P1 G2A config 不存在: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = _require_keys(
        config,
        {
            "version",
            "status",
            "scope",
            "parents",
            "data_identity",
            "software",
            "environment",
            "sampling",
            "viewpoints",
            "adapter",
            "geometry",
            "observability",
            "qualification",
            "execution",
        },
        "E018-P1 G2A config",
    )
    if config["version"] != E018_P1_G2A_CONFIG_VERSION:
        raise ValueError("E018-P1 G2A config version 漂移")
    if config["status"] != "development-only-provider-qualification-no-formal-claim":
        raise ValueError("E018-P1 G2A 只能以 development qualification-only 运行")

    scope = _require_keys(
        config["scope"],
        {
            "gate",
            "test_manifest_metadata_read_allowed",
            "test_trajectory_array_read_allowed",
            "test_label_array_read_allowed",
            "formal_claim_allowed",
            "runtime_gt_control_allowed",
            "live_memory_read_allowed",
            "live_memory_write_allowed",
            "runtime_camera_actuation_allowed",
            "arm_actuation_allowed",
            "manipulation_progression_allowed",
        },
        "scope",
    )
    if scope != {
        "gate": E018_P1_G2A_GATE,
        "test_manifest_metadata_read_allowed": True,
        "test_trajectory_array_read_allowed": False,
        "test_label_array_read_allowed": False,
        "formal_claim_allowed": False,
        "runtime_gt_control_allowed": False,
        "live_memory_read_allowed": False,
        "live_memory_write_allowed": False,
        "runtime_camera_actuation_allowed": False,
        "arm_actuation_allowed": False,
        "manipulation_progression_allowed": False,
    }:
        raise ValueError("G2A scope 必须保持 metadata-only/no-test-array/no-actuation")

    parents = _require_keys(
        config["parents"],
        {
            "g0c_config_version",
            "g0c_config_sha256",
            "g0c_receipt_raw_sha256",
            "g0c_receipt_internal_sha256",
            "g0c_gate_passed",
            "e016_config_sha256",
            "checkpoint_receipt_raw_sha256",
            "checkpoint_sha256",
            "checkpoint_parameter_sha256",
            "checkpoint_provenance_sha256",
            "model_config_sha256",
            "selected_epoch",
            "source_training_camera",
        },
        "parents",
    )
    for name in (
        "g0c_config_sha256",
        "g0c_receipt_raw_sha256",
        "g0c_receipt_internal_sha256",
        "e016_config_sha256",
        "checkpoint_receipt_raw_sha256",
        "checkpoint_sha256",
        "checkpoint_parameter_sha256",
        "checkpoint_provenance_sha256",
        "model_config_sha256",
    ):
        _require_sha256(parents[name], f"parents.{name}")
    if (
        parents["g0c_config_version"]
        != "e018-p1-g0c-rotated-motion-development/v1"
        or parents["g0c_gate_passed"] is not True
        or parents["selected_epoch"] != 12
        or parents["source_training_camera"] != "hand_camera"
    ):
        raise ValueError("G2A parent version/gate/epoch/training-camera 漂移")
    parent_g0c = load_e018_p1_g0c_config(parent_g0c_config_path)
    if canonical_sha256(parent_g0c) != parents["g0c_config_sha256"]:
        raise ValueError("G2A parent G0C config SHA-256 漂移")

    data = _require_keys(
        config["data_identity"],
        {
            "e013_manifest_sha256",
            "e016_fresh_manifest_sha256",
            "proprio_stats_sha256",
            "proprio_normalizer_sha256",
            "finger_force_stats_sha256",
            "finger_force_normalizer_sha256",
            "wrist_pose_envelope_source_splits",
            "wrist_pose_envelope_policy",
        },
        "data_identity",
    )
    for name in (
        "e013_manifest_sha256",
        "e016_fresh_manifest_sha256",
        "proprio_stats_sha256",
        "proprio_normalizer_sha256",
        "finger_force_stats_sha256",
        "finger_force_normalizer_sha256",
    ):
        _require_sha256(data[name], f"data_identity.{name}")
    if (
        data["wrist_pose_envelope_source_splits"] != ["train", "val"]
        or data["wrist_pose_envelope_policy"]
        != "componentwise-min-max-over-valid-e013-train-val/v1"
    ):
        raise ValueError("G2A wrist pose envelope source/policy 漂移")

    software = _require_keys(
        config["software"],
        {"expected_mani_skill_version", "expected_sapien_version"},
        "software",
    )
    if software != {
        "expected_mani_skill_version": "3.0.1",
        "expected_sapien_version": "3.0.3",
    }:
        raise ValueError("G2A simulator software identity 漂移")

    environment = _require_keys(
        config["environment"],
        {
            "environment_id",
            "robot_uid",
            "external_camera_uid",
            "wrist_camera_uid",
            "obs_mode",
            "control_mode",
            "num_envs",
            "image_shape_hwc",
            "capture_mode",
        },
        "environment",
    )
    if environment != {
        "environment_id": "RobotVLAPickCubeToRegion-v1",
        "robot_uid": "panda_wristcam",
        "external_camera_uid": "base_camera",
        "wrist_camera_uid": "hand_camera",
        "obs_mode": "rgb+segmentation",
        "control_mode": "pd_joint_delta_pos",
        "num_envs": 1,
        "image_shape_hwc": [128, 128, 3],
        "capture_mode": "static-render-only-no-environment-step/v1",
    }:
        raise ValueError("G2A environment identity 漂移")

    sampling = _require_keys(
        config["sampling"],
        {
            "seed_policy",
            "seeds",
            "known_development_seeds",
            "required_seed_count",
            "require_disjoint_from_all_manifest_splits",
        },
        "sampling",
    )
    expected_seeds = list(range(75001, 75051))
    if (
        sampling["seed_policy"] != "contiguous-unused-development-range/v1"
        or sampling["seeds"] != expected_seeds
        or sampling["required_seed_count"] != 50
        or sampling["require_disjoint_from_all_manifest_splits"] is not True
    ):
        raise ValueError("G2A 50-seed development identity 漂移")
    known = _require_keys(
        sampling["known_development_seeds"],
        {"g0", "g0b", "g0c", "g1a"},
        "sampling.known_development_seeds",
    )
    expected_known = {
        "g0": [71001, 71013, 71027, 71039],
        "g0b": list(range(72001, 72051)),
        "g0c": [73001, 73013, 73027, 73039],
        "g1a": [74101],
    }
    if known != expected_known:
        raise ValueError("G2A 已用 development seed registry 漂移")

    viewpoints = _require_keys(
        config["viewpoints"],
        {
            "native_wrist_control_id",
            "front_home_id",
            "per_scene_capture_order",
            "front_qualification_ids",
            "g0b_shortlist_order",
            "primary_tie_break_fields",
            "primary_selection_policy",
        },
        "viewpoints",
    )
    if (
        viewpoints["native_wrist_control_id"] != NATIVE_WRIST_CONTROL_ID
        or viewpoints["front_home_id"] != FRONT_HOME_ID
        or viewpoints["per_scene_capture_order"] != list(PER_SCENE_CAPTURE_ORDER)
        or viewpoints["front_qualification_ids"]
        != [FRONT_HOME_ID, *FRONT_ALTERNATE_IDS]
        or viewpoints["g0b_shortlist_order"] != list(G0B_SHORTLIST_ORDER)
        or viewpoints["primary_tie_break_fields"] != list(PRIMARY_TIE_BREAK_FIELDS)
        or viewpoints["primary_selection_policy"]
        != "qualified-then-shortlist-tier-coverage-p90-max-cov95-recall-frozen-order/v1"
    ):
        raise ValueError("G2A viewpoint order/PRIMARY selection rule 漂移")

    adapter = _require_keys(
        config["adapter"],
        {
            "version",
            "config_sha256",
            "role_substitution_semantics",
            "frame_convention",
            "target_camera",
            "actual_pose_source",
            "maximum_rgb_pose_skew_s",
            "maximum_rotation_projection_error_frobenius",
            "geometric_motion_provider_id",
            "geometric_motion_semantics",
            "geometric_motion_value",
            "qualification_only",
        },
        "adapter",
    )
    adapter_config = ActiveFrontProviderAdapterConfig(
        maximum_rgb_pose_skew_s=float(adapter["maximum_rgb_pose_skew_s"]),
    )
    if (
        adapter["version"] != ACTIVE_FRONT_PROVIDER_ADAPTER_VERSION
        or adapter["config_sha256"] != adapter_config.sha256
        or adapter["role_substitution_semantics"]
        != FRONT_PROVIDER_ROLE_SUBSTITUTION_SEMANTICS
        or adapter["frame_convention"] != FRONT_PROVIDER_FRAME_CONVENTION
        or adapter["target_camera"] != "base_camera"
        or adapter["actual_pose_source"] != ACTUAL_EXTERNAL_POSE_SOURCE
        or float(adapter["maximum_rotation_projection_error_frobenius"]) != 1e-6
        or adapter["geometric_motion_provider_id"]
        != "safe-hold-commanded-tcp-target-delta/simulator-static-zero-measured/v1"
        or adapter["geometric_motion_semantics"]
        != "commanded-tcp-target-delta/base-frame/m-rad/v1"
        or adapter["geometric_motion_value"] != [0.0, 0.0, 0.0, 0.0]
        or adapter["qualification_only"] is not True
    ):
        raise ValueError("G2A front adapter/geometry provider identity 漂移")

    geometry = _require_keys(
        config["geometry"],
        {
            "object_center_plane_base_z_m",
            "object_center_plane_tolerance_m",
            "plane_source",
            "provider",
            "covariance_policy",
            "covariance_chi_square_threshold",
            "maximum_position_std_m",
        },
        "geometry",
    )
    if (
        float(geometry["object_center_plane_base_z_m"]) != 0.02
        or float(geometry["object_center_plane_tolerance_m"]) != 1e-5
        or geometry["plane_source"] != "pick-cube-task-contract-not-runtime-gt/v1"
        or geometry["provider"]
        != "same-frame-actual-camera-pose-intrinsic-plane-intersection/v1"
        or geometry["covariance_policy"]
        != "xy-jacobian-sigma-95pct-chi-square-5.991/v1"
        or float(geometry["covariance_chi_square_threshold"]) != 5.991
        or float(geometry["maximum_position_std_m"]) != 0.02
    ):
        raise ValueError("G2A geometry/covariance rule 漂移")

    observability = _require_keys(
        config["observability"],
        {
            "support_radius_px",
            "visibility_threshold",
            "projection_threshold",
            "min_object_mask_probability",
            "max_goal_mask_probability",
            "prediction_rule",
            "write_score_diagnostic_threshold",
            "write_threshold_source",
        },
        "observability",
    )
    for name in (
        "visibility_threshold",
        "projection_threshold",
        "min_object_mask_probability",
        "max_goal_mask_probability",
        "write_score_diagnostic_threshold",
    ):
        _probability(observability[name], f"observability.{name}")
    if (
        observability["support_radius_px"] != 2
        or float(observability["visibility_threshold"]) != 0.5
        or float(observability["projection_threshold"]) != 0.5
        or float(observability["min_object_mask_probability"]) != 0.5
        or float(observability["max_goal_mask_probability"]) != 0.5
        or observability["prediction_rule"]
        != "object-visibility-head-threshold/v1"
        or float(observability["write_score_diagnostic_threshold"])
        != 0.6194030046463013
        or observability["write_threshold_source"]
        != "e018-p0-development-wrist-calibration-diagnostic-only/v1"
    ):
        raise ValueError("G2A observability/write diagnostic rule 漂移")

    qualification = _require_keys(
        config["qualification"],
        {
            "native_wrist_control_required",
            "front_results_if_native_fails",
            "minimum_visibility_precision",
            "minimum_visibility_recall",
            "maximum_observable_world_xyz_p90_m",
            "maximum_observable_world_xyz_max_m",
            "safe_world_xyz_error_m",
            "catastrophic_world_xyz_error_m",
            "minimum_gripper_opening_ratio",
            "maximum_finger_contact_n",
            "maximum_unsafe_accepted_count",
            "maximum_catastrophic_accepted_count",
            "minimum_accepted_safe_coverage",
            "minimum_covariance_95_coverage",
            "minimum_covariance_evaluable_count",
            "wrist_pose_ood_is_diagnostic_only",
            "prediction_before_gt_required",
            "zero_qualified_front_policy",
        },
        "qualification",
    )
    if (
        qualification["native_wrist_control_required"] is not True
        or qualification["front_results_if_native_fails"]
        != "inconclusive-native-wrist-control-failed"
        or float(qualification["minimum_visibility_precision"]) != 0.95
        or float(qualification["minimum_visibility_recall"]) != 0.90
        or float(qualification["maximum_observable_world_xyz_p90_m"]) != 0.005
        or float(qualification["maximum_observable_world_xyz_max_m"]) != 0.02
        or float(qualification["safe_world_xyz_error_m"]) != 0.005
        or float(qualification["catastrophic_world_xyz_error_m"]) != 0.02
        or float(qualification["minimum_gripper_opening_ratio"]) != 0.95
        or float(qualification["maximum_finger_contact_n"]) != 0.01
        or qualification["maximum_unsafe_accepted_count"] != 0
        or qualification["maximum_catastrophic_accepted_count"] != 0
        or float(qualification["minimum_accepted_safe_coverage"]) != 0.10
        or float(qualification["minimum_covariance_95_coverage"]) != 0.90
        or qualification["minimum_covariance_evaluable_count"] != 30
        or qualification["wrist_pose_ood_is_diagnostic_only"] is not True
        or qualification["prediction_before_gt_required"] is not True
        or qualification["zero_qualified_front_policy"]
        != "freeze-negative-receipt-no-training/v1"
    ):
        raise ValueError("G2A qualification 数值门槛或失败规则漂移")

    execution = _require_keys(
        config["execution"],
        {
            "device",
            "use_bf16",
            "batch_size",
            "num_workers",
            "save_rgb",
            "static_qualification_pose_configuration_allowed",
            "environment_step_allowed",
            "runtime_camera_actuation_allowed",
            "provider_training_allowed",
            "memory_read_allowed",
            "memory_write_allowed",
            "manipulation_progression_allowed",
            "require_clean_worktree",
        },
        "execution",
    )
    if (
        execution["device"] != "cuda"
        or execution["use_bf16"] is not True
        or _positive_int(execution["batch_size"], "execution.batch_size") != 32
        or execution["num_workers"] != 0
        or execution["save_rgb"] is not False
        or execution["static_qualification_pose_configuration_allowed"] is not True
        or execution["environment_step_allowed"] is not False
        or execution["runtime_camera_actuation_allowed"] is not False
        or execution["provider_training_allowed"] is not False
        or execution["memory_read_allowed"] is not False
        or execution["memory_write_allowed"] is not False
        or execution["manipulation_progression_allowed"] is not False
        or execution["require_clean_worktree"] is not True
    ):
        raise ValueError("G2A execution 必须为 CUDA BF16 static/no-step/no-training/no-memory")
    return config


def audit_qualification_seed_sets(
    *,
    candidate_seeds: list[int],
    manifest_seed_groups: dict[str, list[int]],
    known_development_seed_groups: dict[str, list[int]],
) -> dict[str, Any]:
    """对已解析 metadata 做纯 seed 交集审计；任何交集都 fail closed。"""

    if (
        not candidate_seeds
        or len(candidate_seeds) != len(set(candidate_seeds))
        or any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            for seed in candidate_seeds
        )
    ):
        raise ValueError("qualification candidate seeds 必须是非空唯一非负整数")
    candidate = set(candidate_seeds)
    overlaps: dict[str, list[int]] = {}
    group_summaries: dict[str, dict[str, Any]] = {}
    for prefix, groups in (
        ("manifest", manifest_seed_groups),
        ("development", known_development_seed_groups),
    ):
        if not isinstance(groups, dict) or not groups:
            raise ValueError(f"{prefix} seed groups 不能为空")
        for name, values in sorted(groups.items()):
            if not isinstance(values, list) or any(
                not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
                for seed in values
            ):
                raise ValueError(f"{prefix}.{name} seeds 无效")
            overlap = sorted(candidate.intersection(values))
            key = f"{prefix}:{name}"
            overlaps[key] = overlap
            group_summaries[key] = {
                "seed_count": len(values),
                "unique_seed_count": len(set(values)),
                "candidate_overlap": overlap,
            }
    passed = not any(overlaps.values())
    result = {
        "candidate_seed_count": len(candidate_seeds),
        "candidate_seed_min": min(candidate_seeds),
        "candidate_seed_max": max(candidate_seeds),
        "groups": group_summaries,
        "overlaps": overlaps,
        "passed": passed,
    }
    result["audit_sha256"] = canonical_sha256(result)
    return result


def audit_g2a_seed_disjointness(
    *,
    config: dict[str, Any],
    e013_deployable_root: str | Path,
    e016_fresh_deployable_root: str | Path,
) -> dict[str, Any]:
    """只读两份 manifest metadata；不解析任何 trajectory/label array。"""

    e013_root = Path(e013_deployable_root)
    e016_root = Path(e016_fresh_deployable_root)
    paths = {
        "e013_all_splits": e013_root / "manifest.jsonl",
        "e016_fresh_all_splits": e016_root / "manifest.jsonl",
    }
    expected_hashes = {
        "e013_all_splits": config["data_identity"]["e013_manifest_sha256"],
        "e016_fresh_all_splits": config["data_identity"][
            "e016_fresh_manifest_sha256"
        ],
    }
    for name, path in paths.items():
        if not path.is_file() or file_sha256(path) != expected_hashes[name]:
            raise RuntimeError(f"G2A {name} manifest identity 漂移")
    entries = {
        "e013_all_splits": load_manifest(e013_root),
        "e016_fresh_all_splits": load_manifest(e016_root),
    }
    seeds_by_group: dict[str, list[int]] = {}
    manifests: dict[str, Any] = {}
    for name, rows in entries.items():
        seeds: list[int] = []
        split_counts = Counter()
        for row in rows:
            seed = row.randomization.get("seed")
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                raise RuntimeError(f"G2A {name} randomization.seed 无效")
            seeds.append(seed)
            split_counts[row.split] += 1
        seeds_by_group[name] = seeds
        manifests[name] = {
            "manifest_sha256": expected_hashes[name],
            "trajectory_metadata_count": len(rows),
            "split_counts": dict(sorted(split_counts.items())),
            "seed_min": min(seeds),
            "seed_max": max(seeds),
            "unique_seed_count": len(set(seeds)),
        }
    audit = audit_qualification_seed_sets(
        candidate_seeds=list(config["sampling"]["seeds"]),
        manifest_seed_groups=seeds_by_group,
        known_development_seed_groups=config["sampling"][
            "known_development_seeds"
        ],
    )
    if not audit["passed"]:
        raise RuntimeError(f"G2A qualification seeds 与既有数据重叠: {audit['overlaps']}")
    result = {
        "version": E018_P1_G2A_RESULT_VERSION,
        "passed": True,
        "manifest_metadata_reads": 2,
        "manifests": manifests,
        "intersection_audit": audit,
        "test_manifest_metadata_read_allowed": True,
        "test_trajectory_array_read_count": 0,
        "test_label_array_read_count": 0,
    }
    result["receipt_sha256"] = canonical_sha256(result)
    return result


def build_e013_wrist_pose_envelope(
    *,
    config: dict[str, Any],
    e013_deployable_root: str | Path,
) -> dict[str, Any]:
    """仅从 E013 train+val 指定 pose arrays 建立 componentwise envelope。"""

    root = Path(e013_deployable_root)
    allowed_splits = tuple(config["data_identity"]["wrist_pose_envelope_source_splits"])
    if allowed_splits != ("train", "val"):
        raise ValueError("G2A pose envelope 只允许 E013 train+val")
    vectors: list[np.ndarray] = []
    array_file_reads = 0
    trajectory_counts: Counter[str] = Counter()
    for split in allowed_splits:
        for entry in load_manifest(root, split=split):
            if entry.split == "test":
                raise RuntimeError("G2A pose envelope 禁止读取 test trajectory")
            path = resolve_trajectory_path(root, entry.file)
            with np.load(path, allow_pickle=False) as payload:
                required = {
                    "wrist_camera_position_base_m",
                    "wrist_camera_rotation_6d_base",
                    "camera_pose_valid",
                }
                if not required.issubset(payload.files):
                    raise RuntimeError("E013 train/val 缺少 wrist pose envelope arrays")
                position = np.asarray(payload["wrist_camera_position_base_m"])
                rotation = np.asarray(payload["wrist_camera_rotation_6d_base"])
                valid = np.asarray(payload["camera_pose_valid"])
            array_file_reads += 1
            trajectory_counts[split] += 1
            if (
                position.shape != (entry.num_steps, 3)
                or rotation.shape != (entry.num_steps, 6)
                or valid.shape != (entry.num_steps,)
                or valid.dtype != np.bool_
                or not np.isfinite(position[valid]).all()
                or not np.isfinite(rotation[valid]).all()
            ):
                raise RuntimeError("E013 wrist pose envelope array schema 漂移")
            if np.any(valid):
                vectors.append(
                    np.concatenate((position[valid], rotation[valid]), axis=1).astype(
                        np.float64,
                        copy=False,
                    )
                )
    if not vectors:
        raise RuntimeError("E013 train+val 没有有效 wrist camera pose")
    stacked = np.concatenate(vectors, axis=0)
    result = {
        "version": E018_P1_G2A_RESULT_VERSION,
        "policy": config["data_identity"]["wrist_pose_envelope_policy"],
        "feature_order": [
            "base_x_m",
            "base_y_m",
            "base_z_m",
            "rotation6d_0",
            "rotation6d_1",
            "rotation6d_2",
            "rotation6d_3",
            "rotation6d_4",
            "rotation6d_5",
        ],
        "minimum": stacked.min(axis=0).astype(float).tolist(),
        "maximum": stacked.max(axis=0).astype(float).tolist(),
        "valid_pose_count": int(stacked.shape[0]),
        "trajectory_array_file_reads": array_file_reads,
        "trajectory_counts": dict(sorted(trajectory_counts.items())),
        "array_names_read": [
            "camera_pose_valid",
            "wrist_camera_position_base_m",
            "wrist_camera_rotation_6d_base",
        ],
        "test_trajectory_array_read_count": 0,
        "test_label_array_read_count": 0,
    }
    result["envelope_sha256"] = canonical_sha256(result)
    return result


def camera_pose_ood_diagnostic(
    base_from_camera_cv: np.ndarray,
    envelope: dict[str, Any],
) -> dict[str, Any]:
    transform = validate_se3(base_from_camera_cv, "base_from_camera_cv")
    vector = np.concatenate(
        (transform[:3, 3], rotation_matrix_to_6d(transform[:3, :3]))
    ).astype(np.float64)
    minimum = np.asarray(envelope["minimum"], dtype=np.float64)
    maximum = np.asarray(envelope["maximum"], dtype=np.float64)
    if minimum.shape != (9,) or maximum.shape != (9,) or np.any(minimum > maximum):
        raise ValueError("wrist pose envelope 无效")
    low_excess = np.maximum(minimum - vector, 0.0)
    high_excess = np.maximum(vector - maximum, 0.0)
    excess = np.maximum(low_excess, high_excess)
    outside = excess > 0.0
    return {
        "outside_dimension_count": int(np.count_nonzero(outside)),
        "outside_envelope": bool(np.any(outside)),
        "maximum_component_excess": float(excess.max()),
        "component_excess": excess.astype(float).tolist(),
        "pose_vector": vector.astype(float).tolist(),
        "envelope_sha256": envelope["envelope_sha256"],
        "diagnostic_only": True,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _covariance_quality(
    covariance: np.ndarray,
    *,
    maximum_position_std_m: float,
) -> dict[str, Any]:
    value = np.asarray(covariance, dtype=np.float64)
    finite = bool(value.shape == (3, 3) and np.isfinite(value).all())
    symmetric = bool(finite and np.allclose(value, value.T, rtol=0.0, atol=1e-12))
    eigenvalues = (
        np.linalg.eigvalsh((value + value.T) * 0.5)
        if finite and symmetric
        else np.asarray([], dtype=np.float64)
    )
    psd = bool(eigenvalues.size == 3 and float(eigenvalues.min()) >= -1e-12)
    maximum_std = (
        float(np.sqrt(max(float(eigenvalues.max()), 0.0))) if psd else None
    )
    within_std = bool(
        maximum_std is not None and maximum_std <= maximum_position_std_m
    )
    return {
        "finite": finite,
        "symmetric": symmetric,
        "positive_semidefinite": psd,
        "maximum_position_std_m": maximum_std,
        "within_maximum_position_std": within_std,
        "valid": bool(finite and symmetric and psd and within_std),
    }


def _mahalanobis_squared_psd(error_xy: np.ndarray, covariance_xy: np.ndarray) -> float:
    """在 PSD（可奇异）协方差上计算距离；零方差方向的非零误差视为无限大。"""

    error = np.asarray(error_xy, dtype=np.float64)
    covariance = np.asarray(covariance_xy, dtype=np.float64)
    if error.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError("Mahalanobis 输入 shape 无效")
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    components = eigenvectors.T @ error
    tolerance = 1e-15
    zero = eigenvalues <= tolerance
    if np.any(np.abs(components[zero]) > 1e-12):
        return math.inf
    positive = ~zero
    if not np.any(positive):
        return 0.0
    return float(np.sum(np.square(components[positive]) / eigenvalues[positive]))


def summarize_qualification_rows(
    rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """按一个 camera/viewpoint 汇总冻结指标；不可定义指标直接 fail。"""

    if not rows:
        raise ValueError("qualification rows 不能为空")
    primitive_ids = {str(row["primitive_id"]) for row in rows}
    if len(primitive_ids) != 1:
        raise ValueError("qualification summary 必须只含一个 primitive")
    q = config["qualification"]
    expected_seed_count = int(config["sampling"]["required_seed_count"])
    expected_seeds = {int(seed) for seed in config["sampling"]["seeds"]}
    unique_seeds = {int(row["seed"]) for row in rows}
    calibration_identities = {
        str(row["calibration_identity_sha256"]) for row in rows
    }
    provider_identities = {str(row["provider_identity_sha256"]) for row in rows}
    predicted = [bool(row["predicted_observable"]) for row in rows]
    target = [bool(row["gt_observable"]) for row in rows]
    true_positive = sum(a and b for a, b in zip(predicted, target))
    false_positive = sum(a and not b for a, b in zip(predicted, target))
    true_negative = sum(not a and not b for a, b in zip(predicted, target))
    false_negative = sum(not a and b for a, b in zip(predicted, target))
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)

    localization_errors = [
        float(row["world_xyz_error_m"])
        for row in rows
        if bool(row["gt_observable"])
        and bool(row["geometry_valid"])
        and row["world_xyz_error_m"] is not None
    ]
    p50 = (
        None
        if not localization_errors
        else float(np.quantile(np.asarray(localization_errors), 0.50))
    )
    p90 = (
        None
        if not localization_errors
        else float(np.quantile(np.asarray(localization_errors), 0.90))
    )
    maximum = None if not localization_errors else float(max(localization_errors))

    accepted = [row for row in rows if bool(row["write_accepted"])]
    unsafe_accepted = sum(not bool(row["oracle_safe_measurement"]) for row in accepted)
    catastrophic_accepted = sum(
        row["world_xyz_error_m"] is not None
        and float(row["world_xyz_error_m"])
        > float(q["catastrophic_world_xyz_error_m"])
        for row in accepted
    )
    oracle_safe_evaluable = [
        row
        for row in rows
        if bool(row["structurally_evaluable"])
        and bool(row["oracle_safe_measurement"])
    ]
    accepted_safe = sum(
        bool(row["write_accepted"]) for row in oracle_safe_evaluable
    )
    accepted_safe_coverage = _ratio(accepted_safe, len(oracle_safe_evaluable))

    covariance_rows = [
        row
        for row in rows
        if bool(row["gt_observable"])
        and bool(row["geometry_valid"])
        and row["measurement_covariance_base_m2"] is not None
        and row["world_xy_error_vector_m"] is not None
    ]
    covariance_inside = 0
    covariance_all_valid = True
    maximum_covariance_std = 0.0
    for row in covariance_rows:
        covariance = np.asarray(row["measurement_covariance_base_m2"], dtype=np.float64)
        quality = _covariance_quality(
            covariance,
            maximum_position_std_m=float(config["geometry"]["maximum_position_std_m"]),
        )
        covariance_all_valid = covariance_all_valid and bool(quality["valid"])
        if quality["maximum_position_std_m"] is not None:
            maximum_covariance_std = max(
                maximum_covariance_std,
                float(quality["maximum_position_std_m"]),
            )
        if not quality["valid"]:
            continue
        error_xy = np.asarray(row["world_xy_error_vector_m"], dtype=np.float64)
        covariance_xy = covariance[:2, :2]
        mahalanobis_sq = _mahalanobis_squared_psd(error_xy, covariance_xy)
        covariance_inside += int(
            mahalanobis_sq
            <= float(config["geometry"]["covariance_chi_square_threshold"])
        )
    covariance_coverage = _ratio(covariance_inside, len(covariance_rows))

    checks = {
        "complete_seed_coverage": (
            len(rows) == expected_seed_count and unique_seeds == expected_seeds
        ),
        "capture_integrity": all(
            bool(row["capture_integrity_passed"]) for row in rows
        ),
        "physical_safety": all(bool(row["physical_safety_passed"]) for row in rows),
        "calibration_identity_stable": len(calibration_identities) == 1,
        "provider_identity_stable": len(provider_identities) == 1,
        "visibility_precision_defined": precision is not None,
        "visibility_precision": bool(
            precision is not None
            and precision >= float(q["minimum_visibility_precision"])
        ),
        "visibility_recall_defined": recall is not None,
        "visibility_recall": bool(
            recall is not None and recall >= float(q["minimum_visibility_recall"])
        ),
        "localization_defined": p90 is not None and maximum is not None,
        "observable_world_xyz_p90": bool(
            p90 is not None
            and p90 <= float(q["maximum_observable_world_xyz_p90_m"])
        ),
        "observable_world_xyz_max": bool(
            maximum is not None
            and maximum <= float(q["maximum_observable_world_xyz_max_m"])
        ),
        "unsafe_accepted": unsafe_accepted
        <= int(q["maximum_unsafe_accepted_count"]),
        "catastrophic_accepted": catastrophic_accepted
        <= int(q["maximum_catastrophic_accepted_count"]),
        "accepted_safe_coverage_defined": accepted_safe_coverage is not None,
        "accepted_safe_coverage": bool(
            accepted_safe_coverage is not None
            and accepted_safe_coverage >= float(q["minimum_accepted_safe_coverage"])
        ),
        "covariance_support": len(covariance_rows)
        >= int(q["minimum_covariance_evaluable_count"]),
        "covariance_all_finite_symmetric_psd_and_bounded": covariance_all_valid,
        "covariance_95_coverage_defined": covariance_coverage is not None,
        "covariance_95_coverage": bool(
            covariance_coverage is not None
            and covariance_coverage >= float(q["minimum_covariance_95_coverage"])
        ),
    }
    passed = all(checks.values())
    return {
        "version": E018_P1_G2A_RESULT_VERSION,
        "primitive_id": next(iter(primitive_ids)),
        "sample_count": len(rows),
        "unique_seed_count": len(unique_seeds),
        "calibration_identity_count": len(calibration_identities),
        "provider_identity_count": len(provider_identities),
        "gt_observable_count": sum(target),
        "gt_unobservable_count": len(rows) - sum(target),
        "visibility_true_positive": true_positive,
        "visibility_false_positive": false_positive,
        "visibility_true_negative": true_negative,
        "visibility_false_negative": false_negative,
        "visibility_precision": precision,
        "visibility_recall": recall,
        "observable_valid_geometry_count": len(localization_errors),
        "observable_world_xyz_p50_m": p50,
        "observable_world_xyz_p90_m": p90,
        "observable_world_xyz_max_m": maximum,
        "write_accepted_count": len(accepted),
        "unsafe_accepted_count": unsafe_accepted,
        "catastrophic_accepted_count": catastrophic_accepted,
        "oracle_safe_structurally_evaluable_count": len(oracle_safe_evaluable),
        "accepted_safe_count": accepted_safe,
        "accepted_safe_coverage": accepted_safe_coverage,
        "covariance_evaluable_count": len(covariance_rows),
        "covariance_inside_95_count": covariance_inside,
        "covariance_95_coverage": covariance_coverage,
        "maximum_covariance_position_std_m": maximum_covariance_std,
        "camera_pose_outside_envelope_count": sum(
            bool(row["camera_pose_ood"]["outside_envelope"]) for row in rows
        ),
        "camera_pose_outside_dimension_count": sum(
            int(row["camera_pose_ood"]["outside_dimension_count"]) for row in rows
        ),
        "camera_pose_maximum_component_excess": max(
            float(row["camera_pose_ood"]["maximum_component_excess"])
            for row in rows
        ),
        "qualification_checks": checks,
        "absolute_gate_passed": passed,
        "failure_reasons": [name for name, value in checks.items() if not value],
    }


def select_primary_front_viewpoint(
    summaries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """仅在绝对合格 alternate 中应用冻结 PRIMARY tie-break。"""

    shortlist_index = {
        primitive_id: index for index, primitive_id in enumerate(G0B_SHORTLIST_ORDER)
    }
    eligible = [
        summary
        for summary in summaries
        if summary["primitive_id"] in FRONT_ALTERNATE_IDS
        and bool(summary["absolute_gate_passed"])
        and summary.get("status") == "pass"
    ]
    if not eligible:
        return None

    def key(summary: dict[str, Any]) -> tuple[Any, ...]:
        primitive_id = str(summary["primitive_id"])
        is_shortlist = primitive_id in shortlist_index
        final_order: Any = (
            shortlist_index[primitive_id] if is_shortlist else primitive_id
        )
        return (
            0 if is_shortlist else 1,
            -float(summary["accepted_safe_coverage"]),
            float(summary["observable_world_xyz_p90_m"]),
            float(summary["observable_world_xyz_max_m"]),
            abs(float(summary["covariance_95_coverage"]) - 0.95),
            -float(summary["visibility_recall"]),
            final_order,
        )

    selected = min(eligible, key=key)
    return {
        "primitive_id": selected["primitive_id"],
        "selection_policy": (
            "qualified-then-shortlist-tier-coverage-p90-max-cov95-recall-"
            "frozen-order/v1"
        ),
        "eligible_primitive_ids": sorted(
            str(summary["primitive_id"]) for summary in eligible
        ),
        "selection_key": list(key(selected)),
    }


def finalize_qualification_summaries(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """应用 native-control 因果门禁，再选择 PRIMARY；不改写绝对指标。"""

    by_id = {str(summary["primitive_id"]): dict(summary) for summary in summaries}
    if (
        len(summaries) != len(PER_SCENE_CAPTURE_ORDER)
        or len(by_id) != len(summaries)
        or set(by_id) != set(PER_SCENE_CAPTURE_ORDER)
    ):
        raise ValueError("G2A summary viewpoint 集合不完整")
    native = by_id[NATIVE_WRIST_CONTROL_ID]
    native_passed = bool(native["absolute_gate_passed"])
    native["status"] = "pass" if native_passed else "fail-parent-health"
    front: list[dict[str, Any]] = []
    for primitive_id in (FRONT_HOME_ID, *FRONT_ALTERNATE_IDS):
        summary = by_id[primitive_id]
        if not native_passed:
            summary["status"] = "inconclusive-native-wrist-control-failed"
        else:
            summary["status"] = (
                "pass" if bool(summary["absolute_gate_passed"]) else "fail"
            )
        front.append(summary)
    ordered = [native, *front]
    if not native_passed:
        return {
            "status": "inconclusive_parent_health",
            "native_wrist_control_passed": False,
            "qualified_front_alternate_ids": [],
            "primary": None,
            "summaries": ordered,
        }
    qualified = [
        str(summary["primitive_id"])
        for summary in front
        if summary["primitive_id"] in FRONT_ALTERNATE_IDS
        and summary["status"] == "pass"
    ]
    primary = select_primary_front_viewpoint(front)
    status = "pass" if qualified else "fail-no-qualified-front-alternate"
    return {
        "status": status,
        "native_wrist_control_passed": True,
        "qualified_front_alternate_ids": qualified,
        "primary": primary,
        "summaries": ordered,
    }


@dataclass(frozen=True)
class _DeployableCapture:
    seed: int
    scene_id: str
    primitive_id: str
    source_camera: str
    rgb: np.ndarray
    structured_state: np.ndarray
    geometric_motion: np.ndarray
    base_from_camera_cv: np.ndarray
    intrinsic_cv: np.ndarray
    input_digest: str
    calibration_identity_sha256: str
    provider_identity: dict[str, Any]
    camera_pose_ood: dict[str, Any]
    rotation_projection_audit: dict[str, Any]
    physical_safety: dict[str, Any]


@dataclass(frozen=True)
class _ModelContext:
    spec: RobotSpec
    model: Any
    torch: Any
    keypoint_temperature: float
    checkpoint_sha256: str
    checkpoint_parameter_sha256: str
    checkpoint_provenance_sha256: str
    model_config_sha256: str
    predictor_identity: dict[str, Any]
    proprio_normalizer: ProprioNormalizer
    finger_force_normalizer: FingerForceNormalizer
    proprio_stats_sha256: str
    proprio_normalizer_sha256: str
    finger_force_stats_sha256: str
    finger_force_normalizer_sha256: str
    adapter_config: ActiveFrontProviderAdapterConfig


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} 不存在: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是 JSON object")
    return value


def _source_identity(repository_root: Path) -> dict[str, Any]:
    safe_repository = str(repository_root.resolve())
    git = ("git", "-c", f"safe.directory={safe_repository}")
    missing = [relative for relative in _SOURCE_FILES if not (repository_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"G2A source files 缺失: {missing}")
    identity = {
        "git_commit": subprocess.run(
            [*git, "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "git_status": subprocess.run(
            [*git, "status", "--porcelain", "--untracked-files=all"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines(),
        "source_file_sha256": {
            relative: file_sha256(repository_root / relative)
            for relative in _SOURCE_FILES
        },
    }
    identity["worktree_clean"] = not identity["git_status"]
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def _verify_g0c_receipt(
    path: Path,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    parents = config["parents"]
    if file_sha256(path) != parents["g0c_receipt_raw_sha256"]:
        raise RuntimeError("G2A G0C receipt raw SHA-256 漂移")
    receipt = _read_json(path, "G0C receipt")
    if (
        receipt.get("version") != "e018-p1-g0c-rotated-motion-result/v1"
        or receipt.get("status") != "complete-development-only"
        or receipt.get("gate_passed") is not True
        or receipt.get("test_split_status") != "prohibited-unread"
        or receipt.get("formal_claim_allowed") is not False
        or receipt.get("config_sha256") != parents["g0c_config_sha256"]
        or receipt.get("receipt_sha256") != parents["g0c_receipt_internal_sha256"]
    ):
        raise RuntimeError("G2A G0C receipt 未通过或 identity 漂移")
    return receipt


def _normalizer_identities(
    spec: RobotSpec,
    proprio: ProprioNormalizer,
    force: FingerForceNormalizer,
) -> tuple[str, str]:
    proprio_sha = canonical_sha256(
        {
            "mean": proprio.mean.tolist(),
            "std": proprio.std.tolist(),
            "clip": proprio.clip,
            "robot_spec": spec.to_dict(),
        }
    )
    force_sha = canonical_sha256(
        {
            "stats": asdict(force.stats),
            "scale": force.scale.tolist(),
            "clip": force.clip,
            "robot_spec": spec.to_dict(),
        }
    )
    return proprio_sha, force_sha


def _load_model_context(
    *,
    config: dict[str, Any],
    e016_config_path: Path,
    training_output: Path,
    stats_root: Path,
) -> _ModelContext:
    import torch

    from robot_vla.precision.checkpoint import (
        PrecisionCheckpointRole,
        load_precision_checkpoint,
    )
    from robot_vla.precision.e016_training import load_e016_p1_config
    from robot_vla.precision.provider import (
        TorchPrecisionFramePredictor,
        TorchPrecisionFramePredictorConfig,
    )

    parents = config["parents"]
    data = config["data_identity"]
    e016_config = load_e016_p1_config(e016_config_path)
    if e016_config.sha256 != parents["e016_config_sha256"]:
        raise RuntimeError("G2A E016 config identity 漂移")
    checkpoint_receipt_path = training_output / "checkpoint_receipt.json"
    if file_sha256(checkpoint_receipt_path) != parents[
        "checkpoint_receipt_raw_sha256"
    ]:
        raise RuntimeError("G2A checkpoint receipt raw identity 漂移")
    checkpoint_receipt = _read_json(checkpoint_receipt_path, "E016 checkpoint receipt")
    checkpoint_meta = checkpoint_receipt.get("checkpoint")
    if (
        not isinstance(checkpoint_meta, dict)
        or checkpoint_receipt.get("passed") is not True
        or checkpoint_receipt.get("selected_epoch") != parents["selected_epoch"]
        or checkpoint_receipt.get("training_config_sha256") != e016_config.sha256
        or checkpoint_meta.get("checkpoint_sha256") != parents["checkpoint_sha256"]
        or checkpoint_meta.get("parameter_state_sha256")
        != parents["checkpoint_parameter_sha256"]
        or checkpoint_meta.get("provenance_sha256")
        != parents["checkpoint_provenance_sha256"]
    ):
        raise RuntimeError("G2A checkpoint receipt 内容 identity 漂移")
    loaded = load_precision_checkpoint(
        training_output / "precision-formal.pt",
        expected_checkpoint_sha256=parents["checkpoint_sha256"],
        expected_provenance_sha256=parents["checkpoint_provenance_sha256"],
        expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
    )
    if loaded.receipt.parameter_state_sha256 != parents["checkpoint_parameter_sha256"]:
        raise RuntimeError("G2A loaded parameter identity 漂移")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("G2A 要求支持 BF16 的 CUDA GPU")

    proprio_path = stats_root / "proprio_stats.json"
    force_path = stats_root / "finger_force_stats.json"
    if (
        file_sha256(proprio_path) != data["proprio_stats_sha256"]
        or file_sha256(force_path) != data["finger_force_stats_sha256"]
    ):
        raise RuntimeError("G2A stats raw identity 漂移")
    spec = RobotSpec()
    proprio_normalizer = ProprioNormalizer(ProprioStats.from_json(proprio_path), spec)
    force_normalizer = FingerForceNormalizer(FingerForceStats.from_json(force_path), spec)
    proprio_normalizer_sha, force_normalizer_sha = _normalizer_identities(
        spec,
        proprio_normalizer,
        force_normalizer,
    )
    if (
        proprio_normalizer_sha != data["proprio_normalizer_sha256"]
        or force_normalizer_sha != data["finger_force_normalizer_sha256"]
    ):
        raise RuntimeError("G2A normalizer semantic identity 漂移")
    predictor = TorchPrecisionFramePredictor(
        loaded.model,
        checkpoint_sha256=loaded.receipt.checkpoint_sha256,
        config=TorchPrecisionFramePredictorConfig(
            device="cuda",
            use_bf16=True,
            temperature=float(e016_config.loss.keypoint_temperature),
            synchronize_cuda_for_latency=True,
        ),
    )
    if (
        predictor.identity.parameter_state_sha256
        != parents["checkpoint_parameter_sha256"]
        or predictor.identity.model_config_sha256 != parents["model_config_sha256"]
    ):
        raise RuntimeError("G2A predictor parameter/model-config identity 漂移")
    if tuple(predictor.model.config.keypoint_names) != (
        "object_center",
        "goal_center",
    ) or tuple(predictor.model.config.mask_names) != ("object", "goal"):
        raise RuntimeError("G2A model object/goal channel identity 漂移")
    predictor.verify_identity()
    adapter_config = ActiveFrontProviderAdapterConfig(
        maximum_rgb_pose_skew_s=float(config["adapter"]["maximum_rgb_pose_skew_s"])
    )
    return _ModelContext(
        spec=spec,
        model=predictor.model,
        torch=torch,
        keypoint_temperature=float(e016_config.loss.keypoint_temperature),
        checkpoint_sha256=loaded.receipt.checkpoint_sha256,
        checkpoint_parameter_sha256=loaded.receipt.parameter_state_sha256,
        checkpoint_provenance_sha256=loaded.receipt.provenance_sha256,
        model_config_sha256=predictor.identity.model_config_sha256,
        predictor_identity=predictor.identity.to_dict(),
        proprio_normalizer=proprio_normalizer,
        finger_force_normalizer=force_normalizer,
        proprio_stats_sha256=data["proprio_stats_sha256"],
        proprio_normalizer_sha256=proprio_normalizer_sha,
        finger_force_stats_sha256=data["finger_force_stats_sha256"],
        finger_force_normalizer_sha256=force_normalizer_sha,
        adapter_config=adapter_config,
    )


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _single_matrix(value: Any, name: str) -> np.ndarray:
    if callable(getattr(value, "to_transformation_matrix", None)):
        value = value.to_transformation_matrix()
    matrix = _numpy(value)
    if matrix.shape == (1, 4, 4):
        matrix = matrix[0]
    return validate_se3(matrix, name)


def _single_vector(value: Any, size: int, name: str) -> np.ndarray:
    vector = _numpy(value)
    if vector.shape == (1, size):
        vector = vector[0]
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise RuntimeError(f"{name} 必须是有限 [{size}]，实际 {vector.shape}")
    return vector


def _canonical_rigid(
    value: Any,
    name: str,
    *,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = _single_matrix(value, name)
    canonical, audit = _closest_rigid_transform(
        raw,
        name,
        maximum_rotation_projection_error_frobenius=float(
            config["adapter"]["maximum_rotation_projection_error_frobenius"]
        ),
    )
    return canonical, {
        "raw_sha256": _array_sha256(np.asarray(raw, dtype=np.float64)),
        "canonical_sha256": _array_sha256(canonical),
        "projection": audit.ledger_record(),
    }


def _intrinsic(value: Any) -> np.ndarray:
    intrinsic = _numpy(value)
    if intrinsic.shape == (1, 3, 3):
        intrinsic = intrinsic[0]
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
        or not np.allclose(intrinsic[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-8)
    ):
        raise RuntimeError("G2A intrinsic_cv schema 漂移")
    return intrinsic


def _base_transforms(
    base_env: Any,
    *,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    world_from_base, base_audit = _canonical_rigid(
        base_env.agent.robot.pose,
        "world_from_robot_base",
        config=config,
    )
    world_from_tcp, tcp_audit = _canonical_rigid(
        base_env.agent.tcp_pose,
        "world_from_tcp",
        config=config,
    )
    base_from_world = invert_se3(world_from_base, "world_from_robot_base")
    base_from_tcp = validate_se3(
        base_from_world @ world_from_tcp,
        "base_from_tcp",
    )
    return world_from_base, base_from_tcp, {
        "world_from_base": base_audit,
        "world_from_tcp": tcp_audit,
    }


def _finger_force_n(base_env: Any) -> np.ndarray:
    scene = base_env.scene
    agent = base_env.agent
    cube = base_env.cube
    left = _numpy(scene.get_pairwise_contact_forces(agent.finger1_link, cube))[0]
    right = _numpy(scene.get_pairwise_contact_forces(agent.finger2_link, cube))[0]
    force = np.asarray(
        (np.linalg.norm(left), np.linalg.norm(right)),
        dtype=np.float32,
    )
    if force.shape != (2,) or not np.isfinite(force).all() or np.any(force < 0.0):
        raise RuntimeError("G2A pairwise finger-cube force schema 漂移")
    return force


def _physical_state(
    base_env: Any,
    *,
    spec: RobotSpec,
    config: dict[str, Any],
    anchor_arm_q: np.ndarray,
    anchor_base_from_tcp: np.ndarray,
    parent_g0c: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    robot = base_env.agent.robot
    qpos = _numpy(robot.get_qpos())
    qvel = _numpy(robot.get_qvel())
    if qpos.shape != (1, 9) or qvel.shape != (1, 9):
        raise RuntimeError("G2A 只支持 Panda num_envs=1 的 9-DoF state")
    joint_names = tuple(joint.name for joint in robot.active_joints)
    proprio = FrankaObservationAdapter(spec).from_maniskill(
        qpos[0],
        qvel[0],
        joint_names,
    )
    _, base_from_tcp, transform_audit = _base_transforms(base_env, config=config)
    force = _finger_force_n(base_env)
    safety = parent_g0c["safety"]
    qualification = config["qualification"]
    arm_drift = float(np.max(np.abs(qpos[0, :7] - anchor_arm_q)))
    tcp_position_drift = float(
        np.linalg.norm(base_from_tcp[:3, 3] - anchor_base_from_tcp[:3, 3])
    )
    from robot_vla.precision.active_front_camera import rotation_angular_distance_rad

    tcp_rotation_drift = rotation_angular_distance_rad(
        anchor_base_from_tcp[:3, :3],
        base_from_tcp[:3, :3],
    )
    physical_safety = {
        "gripper_opening": float(proprio[-1]),
        "gripper_open": bool(
            float(proprio[-1])
            >= float(qualification["minimum_gripper_opening_ratio"])
        ),
        "left_finger_force_n": float(force[0]),
        "right_finger_force_n": float(force[1]),
        "contact_free": bool(
            float(np.max(force)) <= float(qualification["maximum_finger_contact_n"])
        ),
        "arm_joint_drift_rad": arm_drift,
        "tcp_position_drift_m": tcp_position_drift,
        "tcp_orientation_drift_rad": tcp_rotation_drift,
        "arm_hold_valid": arm_drift <= float(safety["arm_joint_drift_max_rad"]),
        "tcp_hold_valid": bool(
            tcp_position_drift <= float(safety["tcp_position_drift_max_m"])
            and tcp_rotation_drift <= float(safety["tcp_orientation_drift_max_rad"])
        ),
        "pairwise_force_source": "maniskill-finger-cube-pairwise-force/v1",
    }
    physical_safety["passed"] = all(
        bool(physical_safety[name])
        for name in ("gripper_open", "contact_free", "arm_hold_valid", "tcp_hold_valid")
    )
    return proprio, base_from_tcp, force, physical_safety, transform_audit


def _camera_data(
    observation: dict[str, Any],
    *,
    camera_uid: str,
    primitive_id: str,
    base_env: Any,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict[str, Any]]:
    sensor = observation["sensor_data"][camera_uid]
    params = observation["sensor_param"][camera_uid]
    rgb = _numpy(sensor["rgb"])
    expected_shape = tuple(config["environment"]["image_shape_hwc"])
    if rgb.shape != (1, *expected_shape) or rgb.dtype != np.uint8:
        raise RuntimeError(f"G2A {camera_uid} RGB schema 漂移: {rgb.shape}/{rgb.dtype}")
    rgb = np.ascontiguousarray(rgb[0])
    intrinsic = _intrinsic(params["intrinsic_cv"])
    world_from_camera_gl, camera_audit = _canonical_rigid(
        params["cam2world_gl"],
        f"world_from_{camera_uid}_gl",
        config=config,
    )
    world_from_base, _, base_audit = _base_transforms(base_env, config=config)
    world_from_camera_cv = opengl_camera_to_opencv(world_from_camera_gl)
    base_from_camera_cv = validate_se3(
        invert_se3(world_from_base, "world_from_robot_base") @ world_from_camera_cv,
        f"base_from_{camera_uid}_cv",
    )
    calibration = {
        "version": E018_P1_G2A_RESULT_VERSION,
        "camera_uid": camera_uid,
        "primitive_id": primitive_id,
        "frame_convention": FRONT_PROVIDER_FRAME_CONVENTION,
        "actual_base_from_camera_cv_sha256": _array_sha256(base_from_camera_cv),
        "intrinsic_cv_sha256": _array_sha256(intrinsic),
    }
    calibration_sha = canonical_sha256(calibration)
    projection_audit = {
        "camera": camera_audit,
        "robot_base": base_audit["world_from_base"],
    }
    return rgb, base_from_camera_cv, intrinsic, calibration_sha, projection_audit


def _wrist_input_digest(
    *,
    seed: int,
    rgb: np.ndarray,
    structured_state: np.ndarray,
    geometric_motion: np.ndarray,
    base_from_camera_cv: np.ndarray,
    intrinsic_cv: np.ndarray,
) -> str:
    return canonical_sha256(
        {
            "version": E018_P1_G2A_RESULT_VERSION,
            "seed": seed,
            "primitive_id": NATIVE_WRIST_CONTROL_ID,
            "source_camera": "hand_camera",
            "timestamp_s": 0.0,
            "rgb_sha256": _array_sha256(rgb),
            "structured_state_sha256": _array_sha256(structured_state),
            "geometric_motion_sha256": _array_sha256(geometric_motion),
            "base_from_camera_cv_sha256": _array_sha256(base_from_camera_cv),
            "intrinsic_cv_sha256": _array_sha256(intrinsic_cv),
        }
    )


def _capture_one_deployable_view(
    *,
    seed: int,
    primitive_id: str,
    camera_uid: str,
    observation: dict[str, Any],
    base_env: Any,
    context: _ModelContext,
    config: dict[str, Any],
    parent_g0c: dict[str, Any],
    pose_envelope: dict[str, Any],
    anchor_arm_q: np.ndarray,
    anchor_base_from_tcp: np.ndarray,
) -> _DeployableCapture:
    proprio, base_from_tcp, force, physical_safety, state_transform_audit = (
        _physical_state(
            base_env,
            spec=context.spec,
            config=config,
            anchor_arm_q=anchor_arm_q,
            anchor_base_from_tcp=anchor_base_from_tcp,
            parent_g0c=parent_g0c,
        )
    )
    rgb, base_from_camera, intrinsic, calibration_sha, camera_projection_audit = (
        _camera_data(
            observation,
            camera_uid=camera_uid,
            primitive_id=primitive_id,
            base_env=base_env,
            config=config,
        )
    )
    motion = PrecisionGeometricMotionInput(
        timestamp_s=0.0,
        motion=tuple(float(item) for item in config["adapter"]["geometric_motion_value"]),
    )
    if primitive_id == NATIVE_WRIST_CONTROL_ID:
        if camera_uid != "hand_camera":
            raise RuntimeError("G2A native control 必须使用 hand_camera")
        structured_state = build_precision_camera_role_state(
            spec=context.spec,
            proprio_normalizer=context.proprio_normalizer,
            finger_force_normalizer=context.finger_force_normalizer,
            physical_proprio=proprio,
            base_from_tcp=base_from_tcp,
            base_from_camera_cv=base_from_camera,
            finger_force_n=force,
        )
        input_digest = _wrist_input_digest(
            seed=seed,
            rgb=rgb,
            structured_state=structured_state,
            geometric_motion=motion.as_array(),
            base_from_camera_cv=base_from_camera,
            intrinsic_cv=intrinsic,
        )
        provider_identity = {
            "version": "e018-p1-g2a-native-wrist-control-provider/v1",
            "source_training_camera": "hand_camera",
            "target_camera": "hand_camera",
            "checkpoint_sha256": context.checkpoint_sha256,
            "checkpoint_parameter_sha256": context.checkpoint_parameter_sha256,
            "checkpoint_provenance_sha256": context.checkpoint_provenance_sha256,
            "model_config_sha256": context.model_config_sha256,
            "proprio_stats_sha256": context.proprio_stats_sha256,
            "proprio_normalizer_sha256": context.proprio_normalizer_sha256,
            "finger_force_stats_sha256": context.finger_force_stats_sha256,
            "finger_force_normalizer_sha256": (
                context.finger_force_normalizer_sha256
            ),
            "calibration_identity_sha256": calibration_sha,
            "geometric_motion_provider_id": config["adapter"][
                "geometric_motion_provider_id"
            ],
            "frame_convention": FRONT_PROVIDER_FRAME_CONVENTION,
            "execution_mode": "qualification-control/no-memory/no-actuation/v1",
        }
    else:
        if camera_uid != "base_camera":
            raise RuntimeError("G2A front qualification 必须使用 base_camera")
        model_input = build_active_front_model_input(
            spec=context.spec,
            proprio_normalizer=context.proprio_normalizer,
            finger_force_normalizer=context.finger_force_normalizer,
            config=context.adapter_config,
            episode_id=f"g2a-seed-{seed:06d}",
            request_id=f"g2a-qualification-seed-{seed:06d}",
            observation_sequence_id=(
                f"g2a-seed-{seed:06d}-{primitive_id.lower().replace('_', '-')}"
            ),
            primitive_id=primitive_id,
            rgb_external=rgb,
            physical_proprio=proprio,
            base_from_tcp=base_from_tcp,
            base_from_external_camera_cv=base_from_camera,
            finger_force_n=force,
            intrinsic_cv=intrinsic,
            control_timestamp_s=0.0,
            rgb_timestamp_s=0.0,
            camera_pose_timestamp_s=0.0,
            tcp_pose_timestamp_s=0.0,
            geometric_motion=motion,
            geometric_motion_provider_id=config["adapter"][
                "geometric_motion_provider_id"
            ],
            camera_motion_state=ExternalCameraMotionState.COLLECT,
            settled=True,
        )
        structured_state = model_input.structured_state
        input_digest = model_input.input_digest
        provider = ActiveFrontProviderIdentity(
            checkpoint_sha256=context.checkpoint_sha256,
            checkpoint_parameter_sha256=context.checkpoint_parameter_sha256,
            checkpoint_provenance_sha256=context.checkpoint_provenance_sha256,
            model_config_sha256=context.model_config_sha256,
            proprio_stats_sha256=context.proprio_stats_sha256,
            proprio_normalizer_sha256=context.proprio_normalizer_sha256,
            finger_force_stats_sha256=context.finger_force_stats_sha256,
            finger_force_normalizer_sha256=context.finger_force_normalizer_sha256,
            adapter_config_sha256=context.adapter_config.sha256,
            primitive_id=primitive_id,
            calibration_identity_sha256=calibration_sha,
            geometric_motion_provider_id=config["adapter"][
                "geometric_motion_provider_id"
            ],
            source_training_camera="hand_camera",
            target_camera="base_camera",
        )
        provider_identity = provider.to_dict()
        provider_identity["identity_sha256"] = provider.sha256
    return _DeployableCapture(
        seed=seed,
        scene_id=f"g2a-seed-{seed:06d}",
        primitive_id=primitive_id,
        source_camera=camera_uid,
        rgb=rgb,
        structured_state=structured_state,
        geometric_motion=motion.as_array(),
        base_from_camera_cv=base_from_camera,
        intrinsic_cv=intrinsic,
        input_digest=input_digest,
        calibration_identity_sha256=calibration_sha,
        provider_identity=provider_identity,
        camera_pose_ood=camera_pose_ood_diagnostic(base_from_camera, pose_envelope),
        rotation_projection_audit={
            "camera": camera_projection_audit,
            "state": state_transform_audit,
        },
        physical_safety=physical_safety,
    )


def _viewpoint_map(
    parent_g0c: dict[str, Any],
) -> dict[str, tuple[FrontCameraViewpoint, FrontCameraOrientationMode]]:
    from robot_vla.precision.e018_p1_g0c import _expand_primitives, _parse_library

    home, anchors, orientations = _parse_library(parent_g0c)
    center = next(item for item in orientations if item.orientation_id == "CENTER")
    result = {FRONT_HOME_ID: (home, center)}
    result.update(
        {
            primitive.viewpoint_id: (primitive, orientation)
            for primitive, orientation in _expand_primitives(anchors, orientations)
        }
    )
    if set(result) != {FRONT_HOME_ID, *FRONT_ALTERNATE_IDS}:
        raise RuntimeError("G2A G0C viewpoint library expansion 漂移")
    return result


def _capture_deployable_phase(
    *,
    config: dict[str, Any],
    parent_g0c: dict[str, Any],
    context: _ModelContext,
    pose_envelope: dict[str, Any],
    seeds: list[int],
) -> tuple[list[_DeployableCapture], dict[str, Any]]:
    """Phase A：不读取 segmentation/object pose，只捕获可部署输入。"""

    import gymnasium as gym
    import mani_skill
    import sapien
    import torch
    from mani_skill.utils import sapien_utils

    from robot_vla.precision.e018_p1_viewpoint_screen import (
        _capture_sensor_observation,
        _set_static_camera_pose,
    )
    from robot_vla.sim import register_robot_vla_maniskill_envs

    if mani_skill.__version__ != config["software"]["expected_mani_skill_version"]:
        raise RuntimeError("G2A ManiSkill version 漂移")
    if sapien.__version__ != config["software"]["expected_sapien_version"]:
        raise RuntimeError("G2A SAPIEN version 漂移")
    if not torch.cuda.is_available():
        raise RuntimeError("G2A Phase A 要求 CUDA")
    environment = config["environment"]
    viewpoint_by_id = _viewpoint_map(parent_g0c)
    register_robot_vla_maniskill_envs()
    env = gym.make(
        environment["environment_id"],
        obs_mode=environment["obs_mode"],
        control_mode=environment["control_mode"],
        num_envs=environment["num_envs"],
        robot_uids=environment["robot_uid"],
    )
    captures: list[_DeployableCapture] = []
    static_pose_configurations = 0
    try:
        base_env = env.unwrapped
        external_sensor = base_env._sensors.get(environment["external_camera_uid"])
        wrist_sensor = base_env._sensors.get(environment["wrist_camera_uid"])
        if external_sensor is None or wrist_sensor is None:
            raise RuntimeError("G2A 需要 base_camera 与 hand_camera")
        camera = external_sensor.camera
        if external_sensor.entity is not None or not callable(
            getattr(camera, "set_local_pose", None)
        ):
            raise RuntimeError("G2A 需要独立可设位姿的 unmounted external camera")
        for seed in seeds:
            env.reset(seed=seed)
            qpos = _numpy(base_env.agent.robot.get_qpos())
            if qpos.shape != (1, 9):
                raise RuntimeError("G2A reset Panda qpos schema 漂移")
            anchor_arm_q = np.asarray(qpos[0, :7], dtype=np.float64).copy()
            _, anchor_tcp, _ = _base_transforms(base_env, config=config)
            native_observation = _capture_sensor_observation(base_env)
            captures.append(
                _capture_one_deployable_view(
                    seed=seed,
                    primitive_id=NATIVE_WRIST_CONTROL_ID,
                    camera_uid=environment["wrist_camera_uid"],
                    observation=native_observation,
                    base_env=base_env,
                    context=context,
                    config=config,
                    parent_g0c=parent_g0c,
                    pose_envelope=pose_envelope,
                    anchor_arm_q=anchor_arm_q,
                    anchor_base_from_tcp=anchor_tcp,
                )
            )
            for primitive_id in (FRONT_HOME_ID, *FRONT_ALTERNATE_IDS):
                viewpoint, orientation = viewpoint_by_id[primitive_id]
                _set_static_camera_pose(
                    camera,
                    viewpoint,
                    orientation,
                    sapien_module=sapien,
                    sapien_utils_module=sapien_utils,
                )
                static_pose_configurations += 1
                observation = _capture_sensor_observation(base_env)
                captures.append(
                    _capture_one_deployable_view(
                        seed=seed,
                        primitive_id=primitive_id,
                        camera_uid=environment["external_camera_uid"],
                        observation=observation,
                        base_env=base_env,
                        context=context,
                        config=config,
                        parent_g0c=parent_g0c,
                        pose_envelope=pose_envelope,
                        anchor_arm_q=anchor_arm_q,
                        anchor_base_from_tcp=anchor_tcp,
                    )
                )
    finally:
        env.close()
    expected = len(seeds) * len(PER_SCENE_CAPTURE_ORDER)
    if len(captures) != expected:
        raise RuntimeError(f"G2A Phase A capture count 漂移: {len(captures)} != {expected}")
    audit = {
        "version": E018_P1_G2A_RESULT_VERSION,
        "phase": "deployable-capture-before-gt/v1",
        "seed_count": len(seeds),
        "capture_count": len(captures),
        "per_scene_capture_order": list(PER_SCENE_CAPTURE_ORDER),
        "static_qualification_pose_configuration_count": static_pose_configurations,
        "environment_step_count": 0,
        "runtime_dynamic_camera_actuation_count": 0,
        "arm_actuation_count": 0,
        "manipulation_progression_count": 0,
        "segmentation_array_read_count": 0,
        "object_pose_read_count": 0,
        "goal_pose_read_count": 0,
        "live_memory_read_count": 0,
        "live_memory_write_count": 0,
        "test_trajectory_array_read_count": 0,
        "test_label_array_read_count": 0,
        "physical_safety_passed": all(
            bool(capture.physical_safety["passed"]) for capture in captures
        ),
        "environment_identity": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(torch.device("cuda")),
            "mani_skill": mani_skill.__version__,
            "sapien": sapien.__version__,
            "external_sensor_class": (
                type(external_sensor).__module__ + "." + type(external_sensor).__name__
            ),
            "wrist_sensor_class": (
                type(wrist_sensor).__module__ + "." + type(wrist_sensor).__name__
            ),
        },
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return captures, audit


def _measurement_covariance(
    local_jacobian_xy_m_per_px: list[list[float]],
    sigma_xy_px: np.ndarray,
) -> np.ndarray:
    jacobian = np.asarray(local_jacobian_xy_m_per_px, dtype=np.float64)
    sigma = np.asarray(sigma_xy_px, dtype=np.float64)
    if (
        jacobian.shape != (2, 2)
        or sigma.shape != (2,)
        or not np.isfinite(jacobian).all()
        or not np.isfinite(sigma).all()
        or np.any(sigma < 0.0)
    ):
        raise ValueError("G2A measurement covariance 输入无效")
    covariance = np.zeros((3, 3), dtype=np.float64)
    covariance[:2, :2] = jacobian @ np.diag(np.square(sigma)) @ jacobian.T
    if not np.isfinite(covariance).all():
        raise ValueError("G2A measurement covariance 非有限")
    return covariance


_PREDICTION_FORBIDDEN_KEYS = {
    "gt_observable",
    "gt_object_position_base_m",
    "gt_projected_normalized_uv",
    "object_mask",
    "goal_mask",
    "oracle_safe_measurement",
    "world_xyz_error_m",
    "world_xy_error_vector_m",
    "segmentation",
}


def assert_prediction_ledger_deployable_only(rows: list[dict[str, Any]]) -> None:
    """在写盘前拒绝任何 privileged GT/segmentation 字段。"""

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in _PREDICTION_FORBIDDEN_KEYS or key.startswith("gt_"):
                    raise ValueError(f"prediction ledger 含 privileged field: {path}.{key}")
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    for index, row in enumerate(rows):
        walk(row, f"rows[{index}]")


def _predict_captures(
    captures: list[_DeployableCapture],
    *,
    context: _ModelContext,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """只对 Phase-A deployable captures 推理；结果不含任何 GT。"""

    torch = context.torch
    model = context.model
    batch_size = int(config["execution"]["batch_size"])
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(captures), batch_size):
            batch_captures = captures[start : start + batch_size]
            image_numpy = np.stack(
                [capture.rgb.transpose(2, 0, 1) for capture in batch_captures]
            ).astype(np.float32)
            image_numpy /= np.float32(255.0)
            state_numpy = np.stack(
                [capture.structured_state for capture in batch_captures]
            )
            motion_numpy = np.stack(
                [capture.geometric_motion for capture in batch_captures]
            )
            image = torch.from_numpy(image_numpy).to(torch.device("cuda"))
            state = torch.from_numpy(state_numpy).to(torch.device("cuda"))
            motion = torch.from_numpy(motion_numpy).to(torch.device("cuda"))
            torch.cuda.synchronize()
            batch_started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                output = model(image, state, motion)
            decoded = output.decode_for_control(
                temperature=context.keypoint_temperature
            )
            torch.cuda.synchronize()
            batch_latency = time.perf_counter() - batch_started
            predicted_uv = decoded.keypoints.normalized_uv.detach().float().cpu().numpy()
            visibility = decoded.visibility_probability.detach().float().cpu().numpy()
            projection = (
                decoded.projection_validity_probability.detach().float().cpu().numpy()
            )
            entropy = decoded.keypoints.normalized_entropy.detach().float().cpu().numpy()
            sigma = decoded.keypoint_sigma_px.detach().float().cpu().numpy()
            mask_probability = torch.sigmoid(output.mask_logits.detach().float()).cpu().numpy()
            if (
                predicted_uv.shape != (len(batch_captures), 2, 2)
                or visibility.shape != (len(batch_captures), 2)
                or projection.shape != (len(batch_captures),)
                or entropy.shape != (len(batch_captures), 2)
                or sigma.shape != (len(batch_captures), 2, 2)
                or mask_probability.shape[:2] != (len(batch_captures), 2)
            ):
                raise RuntimeError("G2A model output channel/shape identity 漂移")
            for index, capture in enumerate(batch_captures):
                object_uv = predicted_uv[index, 0]
                object_mask_probability = mask_probability_at_normalized_uv(
                    mask_probability[index, 0],
                    object_uv,
                )
                goal_mask_probability = mask_probability_at_normalized_uv(
                    mask_probability[index, 1],
                    object_uv,
                )
                try:
                    geometry = geometry_conditioning(
                        normalized_uv=object_uv,
                        intrinsic_cv=capture.intrinsic_cv,
                        base_from_camera_cv=capture.base_from_camera_cv,
                        image_size_hw=tuple(capture.rgb.shape[:2]),
                        plane_base_z_m=float(
                            config["geometry"]["object_center_plane_base_z_m"]
                        ),
                    )
                    predicted_world = np.asarray(
                        geometry["predicted_world_point_base_m"],
                        dtype=np.float64,
                    )
                    covariance = _measurement_covariance(
                        geometry["local_jacobian_xy_m_per_px"],
                        sigma[index, 0],
                    )
                    geometry_valid = True
                    geometry_payload: dict[str, Any] = {
                        "valid": True,
                        "predicted_object_position_base_m": predicted_world.tolist(),
                        "measurement_covariance_base_m2": covariance.tolist(),
                        "abs_n_dot_unit_ray": geometry["abs_n_dot_unit_ray"],
                        "jacobian_sigma_max_mm_per_px": geometry[
                            "jacobian_sigma_max_mm_per_px"
                        ],
                    }
                except ValueError as error:
                    predicted_world = None
                    covariance = None
                    geometry_valid = False
                    geometry_payload = {
                        "valid": False,
                        "predicted_object_position_base_m": None,
                        "measurement_covariance_base_m2": None,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                evidence = ObjectWriteEvidence(
                    visibility_probability=float(visibility[index, 0]),
                    projection_validity_probability=float(projection[index]),
                    object_mask_probability=float(object_mask_probability),
                    goal_mask_probability=float(goal_mask_probability),
                    normalized_entropy=float(entropy[index, 0]),
                    radial_sigma_px=float(np.linalg.norm(sigma[index, 0])),
                    geometry_valid=geometry_valid,
                    min_object_mask_probability=float(
                        config["observability"]["min_object_mask_probability"]
                    ),
                    max_goal_mask_probability=float(
                        config["observability"]["max_goal_mask_probability"]
                    ),
                )
                provider_identity_sha = str(
                    capture.provider_identity.get("identity_sha256")
                    or canonical_sha256(capture.provider_identity)
                )
                rows.append(
                    {
                        "version": E018_P1_G2A_RESULT_VERSION,
                        "phase": "prediction-before-gt/v1",
                        "seed": capture.seed,
                        "scene_id": capture.scene_id,
                        "primitive_id": capture.primitive_id,
                        "source_camera": capture.source_camera,
                        "input_digest": capture.input_digest,
                        "rgb_sha256": _array_sha256(capture.rgb),
                        "structured_state_sha256": _array_sha256(
                            capture.structured_state
                        ),
                        "geometric_motion_sha256": _array_sha256(
                            capture.geometric_motion
                        ),
                        "actual_base_from_camera_cv": (
                            capture.base_from_camera_cv.tolist()
                        ),
                        "actual_base_from_camera_cv_sha256": _array_sha256(
                            capture.base_from_camera_cv
                        ),
                        "intrinsic_cv": capture.intrinsic_cv.tolist(),
                        "intrinsic_cv_sha256": _array_sha256(capture.intrinsic_cv),
                        "calibration_identity_sha256": (
                            capture.calibration_identity_sha256
                        ),
                        "provider_identity": capture.provider_identity,
                        "provider_identity_sha256": provider_identity_sha,
                        "camera_pose_ood": capture.camera_pose_ood,
                        "rotation_projection_audit": (
                            capture.rotation_projection_audit
                        ),
                        "physical_safety": capture.physical_safety,
                        "predicted_object_normalized_uv": object_uv.astype(
                            float
                        ).tolist(),
                        "predicted_goal_normalized_uv": predicted_uv[
                            index, 1
                        ].astype(float).tolist(),
                        "object_visibility_probability": float(
                            visibility[index, 0]
                        ),
                        "goal_visibility_probability": float(visibility[index, 1]),
                        "projection_validity_probability": float(projection[index]),
                        "object_normalized_entropy": float(entropy[index, 0]),
                        "object_sigma_xy_px": sigma[index, 0].astype(float).tolist(),
                        "object_mask_probability_at_predicted_object": float(
                            object_mask_probability
                        ),
                        "goal_mask_probability_at_predicted_object": float(
                            goal_mask_probability
                        ),
                        "predicted_observable": bool(
                            float(visibility[index, 0])
                            >= float(config["observability"]["visibility_threshold"])
                        ),
                        "geometry": geometry_payload,
                        "write_evidence": evidence.to_dict(),
                        "write_accepted": evidence.accepted(
                            threshold=float(
                                config["observability"][
                                    "write_score_diagnostic_threshold"
                                ]
                            )
                        ),
                        "batch_latency_s": batch_latency,
                        "batch_size": len(batch_captures),
                        "qualification_only": True,
                        "memory_write_allowed": False,
                        "actuation_allowed": False,
                    }
                )
    assert_prediction_ledger_deployable_only(rows)
    audit = {
        "version": E018_P1_G2A_RESULT_VERSION,
        "prediction_count": len(rows),
        "model_forward_batch_count": math.ceil(len(captures) / batch_size),
        "model_forward_sample_count": len(captures),
        "elapsed_s": time.perf_counter() - started,
        "privileged_field_scan_passed": True,
        "gt_read_count_before_prediction_freeze": 0,
        "live_memory_read_count": 0,
        "live_memory_write_count": 0,
        "actuation_count": 0,
        "test_trajectory_array_read_count": 0,
        "test_label_array_read_count": 0,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return rows, audit


def freeze_prediction_ledger(
    output_root: Path,
    *,
    rows: list[dict[str, Any]],
    config_sha256: str,
) -> dict[str, Any]:
    """原子写入/fsync prediction ledger，并以 marker 锁定其 hash。"""

    assert_prediction_ledger_deployable_only(rows)
    ledger_path = output_root / "prediction_ledger.jsonl"
    _atomic_jsonl(ledger_path, rows)
    ledger_sha = file_sha256(ledger_path)
    marker = {
        "version": E018_P1_G2A_RESULT_VERSION,
        "status": "frozen-before-privileged-gt-read",
        "config_sha256": config_sha256,
        "prediction_ledger_sha256": ledger_sha,
        "prediction_count": len(rows),
        "privileged_field_scan_passed": True,
        "gt_read_count_before_freeze": 0,
    }
    marker["freeze_marker_sha256"] = canonical_sha256(marker)
    _atomic_json(output_root / "prediction_freeze.json", marker)
    if file_sha256(ledger_path) != ledger_sha:
        raise RuntimeError("G2A prediction ledger freeze 后发生漂移")
    return marker


def load_frozen_prediction_ledger(
    output_root: Path,
    *,
    config_sha256: str,
    expected_prediction_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """只从已落盘冻结件重载 Phase-B 输入，并在读取前后复核 hash。"""

    _require_sha256(config_sha256, "config_sha256")
    _positive_int(expected_prediction_count, "expected_prediction_count")
    marker = _require_keys(
        _read_json(output_root / "prediction_freeze.json", "prediction freeze marker"),
        {
            "version",
            "status",
            "config_sha256",
            "prediction_ledger_sha256",
            "prediction_count",
            "privileged_field_scan_passed",
            "gt_read_count_before_freeze",
            "freeze_marker_sha256",
        },
        "prediction freeze marker",
    )
    marker_sha = _require_sha256(
        marker["freeze_marker_sha256"],
        "prediction_freeze.freeze_marker_sha256",
    )
    marker_payload = dict(marker)
    del marker_payload["freeze_marker_sha256"]
    if canonical_sha256(marker_payload) != marker_sha:
        raise RuntimeError("G2A prediction freeze marker internal SHA-256 漂移")
    if (
        marker["version"] != E018_P1_G2A_RESULT_VERSION
        or marker["status"] != "frozen-before-privileged-gt-read"
        or marker["config_sha256"] != config_sha256
        or marker["prediction_count"] != expected_prediction_count
        or marker["privileged_field_scan_passed"] is not True
        or marker["gt_read_count_before_freeze"] != 0
    ):
        raise RuntimeError("G2A prediction freeze marker 内容漂移")
    ledger_sha = _require_sha256(
        marker["prediction_ledger_sha256"],
        "prediction_freeze.prediction_ledger_sha256",
    )
    ledger_path = output_root / "prediction_ledger.jsonl"
    if not ledger_path.is_file() or file_sha256(ledger_path) != ledger_sha:
        raise RuntimeError("G2A frozen prediction ledger SHA-256 漂移")

    rows: list[dict[str, Any]] = []
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(
                    f"G2A frozen prediction ledger 第 {line_number} 行为空"
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"G2A frozen prediction ledger 第 {line_number} 行 JSON 无效"
                ) from error
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"G2A frozen prediction ledger 第 {line_number} 行不是 object"
                )
            rows.append(row)
    if len(rows) != expected_prediction_count:
        raise RuntimeError(
            "G2A frozen prediction ledger count 漂移: "
            f"{len(rows)} != {expected_prediction_count}"
        )
    assert_prediction_ledger_deployable_only(rows)
    if file_sha256(ledger_path) != ledger_sha:
        raise RuntimeError("G2A frozen prediction ledger 读取期间发生漂移")
    return rows, marker


def _base_point(base_env: Any, actor: Any, *, config: dict[str, Any]) -> np.ndarray:
    world_from_base, _, _ = _base_transforms(base_env, config=config)
    base_from_world = invert_se3(world_from_base, "world_from_robot_base")
    point_world = _single_vector(actor.pose.p, 3, "privileged actor position world")
    return (
        base_from_world
        @ np.concatenate((point_world, np.ones(1, dtype=np.float64)))
    )[:3]


def _segmentation_masks(
    observation: dict[str, Any],
    *,
    camera_uid: str,
    object_actor_id: int,
    goal_actor_id: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    segmentation = _numpy(
        observation["sensor_data"][camera_uid]["segmentation"]
    )
    if segmentation.ndim != 4 or segmentation.shape[:3] != (1, 128, 128):
        raise RuntimeError(f"G2A {camera_uid} segmentation schema 漂移")
    actor_ids = np.asarray(segmentation[0, ..., 0])
    if not np.issubdtype(actor_ids.dtype, np.integer):
        raise RuntimeError("G2A segmentation actor-id channel 必须是整数")
    object_mask = np.asarray(actor_ids == object_actor_id, dtype=np.bool_)
    goal_mask = np.asarray(actor_ids == goal_actor_id, dtype=np.bool_)
    return object_mask, goal_mask, _array_sha256(actor_ids)


def _score_prediction(
    prediction: dict[str, Any],
    *,
    observation: dict[str, Any],
    camera_uid: str,
    base_env: Any,
    object_actor_id: int,
    goal_actor_id: int,
    object_position_base_m: np.ndarray,
    spec: RobotSpec,
    config: dict[str, Any],
    parent_g0c: dict[str, Any],
    pose_envelope: dict[str, Any],
    anchor_arm_q: np.ndarray,
    anchor_base_from_tcp: np.ndarray,
    prediction_ledger_sha256: str,
) -> dict[str, Any]:
    primitive_id = str(prediction["primitive_id"])
    rgb, base_from_camera, intrinsic, calibration_sha, _ = _camera_data(
        observation,
        camera_uid=camera_uid,
        primitive_id=primitive_id,
        base_env=base_env,
        config=config,
    )
    _, _, _, physical_safety, _ = _physical_state(
        base_env,
        spec=spec,
        config=config,
        anchor_arm_q=anchor_arm_q,
        anchor_base_from_tcp=anchor_base_from_tcp,
        parent_g0c=parent_g0c,
    )
    capture_integrity = {
        "rgb_match": _array_sha256(rgb) == prediction["rgb_sha256"],
        "base_from_camera_match": _array_sha256(base_from_camera)
        == prediction["actual_base_from_camera_cv_sha256"],
        "intrinsic_match": _array_sha256(intrinsic)
        == prediction["intrinsic_cv_sha256"],
        "calibration_identity_match": calibration_sha
        == prediction["calibration_identity_sha256"],
        "pose_ood_match": camera_pose_ood_diagnostic(base_from_camera, pose_envelope)
        == prediction["camera_pose_ood"],
    }
    capture_integrity["passed"] = all(capture_integrity.values())
    object_mask, goal_mask, segmentation_sha = _segmentation_masks(
        observation,
        camera_uid=camera_uid,
        object_actor_id=object_actor_id,
        goal_actor_id=goal_actor_id,
    )
    try:
        gt_uv = project_base_point_to_normalized_uv(
            object_position_base_m,
            intrinsic,
            base_from_camera,
            object_mask.shape,
        )
        projection_valid = True
    except ValueError:
        gt_uv = None
        projection_valid = False
    observability = derive_object_observability(
        object_exists=True,
        projection_valid=projection_valid,
        projected_normalized_uv=gt_uv,
        object_mask=object_mask,
        goal_mask=goal_mask,
        legacy_visible=bool(np.any(object_mask)),
        support_radius_px=int(config["observability"]["support_radius_px"]),
    )
    predicted_position_value = prediction["geometry"][
        "predicted_object_position_base_m"
    ]
    predicted_position = (
        None
        if predicted_position_value is None
        else np.asarray(predicted_position_value, dtype=np.float64)
    )
    if predicted_position is not None and predicted_position.shape != (3,):
        raise RuntimeError("G2A predicted object position shape 漂移")
    error_vector = (
        None if predicted_position is None else predicted_position - object_position_base_m
    )
    world_error = (
        None if error_vector is None else float(np.linalg.norm(error_vector))
    )
    geometry_valid = bool(prediction["geometry"]["valid"])
    structurally_evaluable = bool(
        physical_safety["passed"]
        and observability.observable
        and geometry_valid
        and world_error is not None
    )
    oracle_safe = bool(
        structurally_evaluable
        and world_error <= float(config["qualification"]["safe_world_xyz_error_m"])
    )
    covariance = prediction["geometry"]["measurement_covariance_base_m2"]
    covariance_quality = (
        None
        if covariance is None
        else _covariance_quality(
            np.asarray(covariance, dtype=np.float64),
            maximum_position_std_m=float(
                config["geometry"]["maximum_position_std_m"]
            ),
        )
    )
    predicted_uv = np.asarray(
        prediction["predicted_object_normalized_uv"],
        dtype=np.float64,
    )
    pixel_error = (
        None
        if gt_uv is None
        else float(
            np.linalg.norm(
                (predicted_uv - gt_uv)
                * np.asarray((object_mask.shape[1], object_mask.shape[0]))
            )
        )
    )
    return {
        "version": E018_P1_G2A_RESULT_VERSION,
        "phase": "offline-privileged-scoring-after-prediction-freeze/v1",
        "prediction_ledger_sha256": prediction_ledger_sha256,
        "seed": int(prediction["seed"]),
        "scene_id": prediction["scene_id"],
        "primitive_id": primitive_id,
        "source_camera": camera_uid,
        "provider_identity_sha256": prediction["provider_identity_sha256"],
        "calibration_identity_sha256": prediction["calibration_identity_sha256"],
        "capture_integrity": capture_integrity,
        "capture_integrity_passed": bool(capture_integrity["passed"]),
        "physical_safety": physical_safety,
        "physical_safety_passed": bool(physical_safety["passed"]),
        "camera_pose_ood": prediction["camera_pose_ood"],
        "gt_object_position_base_m": object_position_base_m.astype(float).tolist(),
        "gt_projected_normalized_uv": (
            None if gt_uv is None else gt_uv.astype(float).tolist()
        ),
        "gt_observable": bool(observability.observable),
        "observability": observability.to_dict(),
        "object_mask_visible_pixel_count": int(np.count_nonzero(object_mask)),
        "goal_mask_visible_pixel_count": int(np.count_nonzero(goal_mask)),
        "segmentation_actor_sha256": segmentation_sha,
        "predicted_observable": bool(prediction["predicted_observable"]),
        "object_visibility_probability": prediction[
            "object_visibility_probability"
        ],
        "projection_validity_probability": prediction[
            "projection_validity_probability"
        ],
        "predicted_object_normalized_uv": prediction[
            "predicted_object_normalized_uv"
        ],
        "object_pixel_error": pixel_error,
        "geometry_valid": geometry_valid,
        "predicted_object_position_base_m": (
            None if predicted_position is None else predicted_position.tolist()
        ),
        "measurement_covariance_base_m2": covariance,
        "covariance_quality": covariance_quality,
        "world_xyz_error_m": world_error,
        "world_xy_error_vector_m": (
            None if error_vector is None else error_vector[:2].astype(float).tolist()
        ),
        "write_evidence": prediction["write_evidence"],
        "write_accepted": bool(prediction["write_accepted"]),
        "structurally_evaluable": structurally_evaluable,
        "oracle_safe_measurement": oracle_safe,
        "used_by_runtime_control": False,
        "memory_read_executed": False,
        "memory_write_executed": False,
        "test_data_read": False,
    }


def _score_after_prediction_freeze(
    *,
    config: dict[str, Any],
    parent_g0c: dict[str, Any],
    spec: RobotSpec,
    pose_envelope: dict[str, Any],
    seeds: list[int],
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Phase B：只重载冻结预测，再读取 simulator GT 并离线评分。"""

    import gymnasium as gym
    import mani_skill
    import sapien
    from mani_skill.utils import sapien_utils

    from robot_vla.precision.e018_p1_viewpoint_screen import (
        _capture_sensor_observation,
        _set_static_camera_pose,
    )
    from robot_vla.sim import register_robot_vla_maniskill_envs

    expected_prediction_count = len(seeds) * len(PER_SCENE_CAPTURE_ORDER)
    predictions, prediction_freeze = load_frozen_prediction_ledger(
        output_root,
        config_sha256=canonical_sha256(config),
        expected_prediction_count=expected_prediction_count,
    )
    ledger_path = output_root / "prediction_ledger.jsonl"
    expected_ledger_sha = prediction_freeze["prediction_ledger_sha256"]
    by_key = {
        (int(row["seed"]), str(row["primitive_id"])): row for row in predictions
    }
    if len(by_key) != len(predictions):
        raise RuntimeError("G2A prediction identity 重复")
    environment = config["environment"]
    viewpoint_by_id = _viewpoint_map(parent_g0c)
    register_robot_vla_maniskill_envs()
    env = gym.make(
        environment["environment_id"],
        obs_mode=environment["obs_mode"],
        control_mode=environment["control_mode"],
        num_envs=environment["num_envs"],
        robot_uids=environment["robot_uid"],
    )
    rows: list[dict[str, Any]] = []
    static_pose_configurations = 0
    try:
        base_env = env.unwrapped
        sensor = base_env._sensors.get(environment["external_camera_uid"])
        if sensor is None:
            raise RuntimeError("G2A Phase B external camera 缺失")
        camera = sensor.camera
        for seed in seeds:
            env.reset(seed=seed)
            qpos = _numpy(base_env.agent.robot.get_qpos())
            anchor_arm_q = np.asarray(qpos[0, :7], dtype=np.float64).copy()
            _, anchor_tcp, _ = _base_transforms(base_env, config=config)
            # 以下 privileged actor identity/pose 读取发生在 prediction freeze 之后。
            object_actor_id = int(
                _numpy(base_env.cube.per_scene_id).reshape(-1)[0]
            )
            goal_actor_id = int(
                _numpy(base_env.goal_site.per_scene_id).reshape(-1)[0]
            )
            object_position = _base_point(base_env, base_env.cube, config=config)
            if abs(
                float(object_position[2])
                - float(config["geometry"]["object_center_plane_base_z_m"])
            ) > float(config["geometry"]["object_center_plane_tolerance_m"]):
                raise RuntimeError("G2A object center 不符合冻结 task plane contract")
            native_observation = _capture_sensor_observation(base_env)
            rows.append(
                _score_prediction(
                    by_key[(seed, NATIVE_WRIST_CONTROL_ID)],
                    observation=native_observation,
                    camera_uid=environment["wrist_camera_uid"],
                    base_env=base_env,
                    object_actor_id=object_actor_id,
                    goal_actor_id=goal_actor_id,
                    object_position_base_m=object_position,
                    spec=spec,
                    config=config,
                    parent_g0c=parent_g0c,
                    pose_envelope=pose_envelope,
                    anchor_arm_q=anchor_arm_q,
                    anchor_base_from_tcp=anchor_tcp,
                    prediction_ledger_sha256=expected_ledger_sha,
                )
            )
            for primitive_id in (FRONT_HOME_ID, *FRONT_ALTERNATE_IDS):
                viewpoint, orientation = viewpoint_by_id[primitive_id]
                _set_static_camera_pose(
                    camera,
                    viewpoint,
                    orientation,
                    sapien_module=sapien,
                    sapien_utils_module=sapien_utils,
                )
                static_pose_configurations += 1
                observation = _capture_sensor_observation(base_env)
                rows.append(
                    _score_prediction(
                        by_key[(seed, primitive_id)],
                        observation=observation,
                        camera_uid=environment["external_camera_uid"],
                        base_env=base_env,
                        object_actor_id=object_actor_id,
                        goal_actor_id=goal_actor_id,
                        object_position_base_m=object_position,
                        spec=spec,
                        config=config,
                        parent_g0c=parent_g0c,
                        pose_envelope=pose_envelope,
                        anchor_arm_q=anchor_arm_q,
                        anchor_base_from_tcp=anchor_tcp,
                        prediction_ledger_sha256=expected_ledger_sha,
                    )
                )
    finally:
        env.close()
    expected = len(seeds) * len(PER_SCENE_CAPTURE_ORDER)
    if len(rows) != expected:
        raise RuntimeError(f"G2A Phase B scoring count 漂移: {len(rows)} != {expected}")
    if file_sha256(ledger_path) != expected_ledger_sha:
        raise RuntimeError("G2A privileged scoring 后 prediction ledger hash 漂移")
    audit = {
        "version": E018_P1_G2A_RESULT_VERSION,
        "phase": "privileged-scoring-after-prediction-freeze/v1",
        "prediction_source": "fsynced-frozen-ledger-reload/v1",
        "prediction_freeze_marker_sha256": prediction_freeze[
            "freeze_marker_sha256"
        ],
        "prediction_ledger_sha256_before": expected_ledger_sha,
        "prediction_ledger_sha256_after": file_sha256(ledger_path),
        "seed_count": len(seeds),
        "scored_row_count": len(rows),
        "static_qualification_pose_configuration_count": static_pose_configurations,
        "environment_step_count": 0,
        "runtime_dynamic_camera_actuation_count": 0,
        "arm_actuation_count": 0,
        "manipulation_progression_count": 0,
        "segmentation_array_read_count": len(rows),
        "object_pose_read_count": len(seeds),
        "runtime_gt_control_read_count": 0,
        "model_context_received": False,
        "provider_forward_count": 0,
        "live_memory_read_count": 0,
        "live_memory_write_count": 0,
        "test_trajectory_array_read_count": 0,
        "test_label_array_read_count": 0,
        "capture_integrity_passed": all(
            bool(row["capture_integrity_passed"]) for row in rows
        ),
        "physical_safety_passed": all(
            bool(row["physical_safety_passed"]) for row in rows
        ),
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return rows, audit


def _report_markdown(summary: dict[str, Any]) -> str:
    primary_id = (
        None if summary.get("primary") is None else summary["primary"]["primitive_id"]
    )
    lines = [
        "# E018-P1 G2A front provider qualification",
        "",
        f"- status: `{summary['status']}`",
        f"- config SHA-256: `{summary['config_sha256']}`",
        f"- prediction ledger SHA-256: `{summary['prediction_ledger_sha256']}`",
        f"- native wrist control passed: `{summary.get('native_wrist_control_passed')}`",
        f"- PRIMARY: `{primary_id}`",
        "- test trajectory/label array reads: `0 / 0`",
        "- live Memory reads/writes: `0 / 0`",
        "- runtime camera/arm/manipulation actuation: `0 / 0 / 0`",
        "",
        "| viewpoint | status | P | R | XYZ p90 mm | XYZ max mm | accepted | "
        "unsafe | safe coverage | cov95 | cov n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary.get("viewpoint_summaries", []):
        def number(name: str, scale: float = 1.0) -> str:
            value = item.get(name)
            return "n/a" if value is None else f"{float(value) * scale:.6f}"

        lines.append(
            "| {primitive} | {status} | {precision} | {recall} | {p90} | "
            "{maximum} | {accepted} | {unsafe} | {coverage} | {covariance} | "
            "{covariance_count} |".format(
                primitive=item["primitive_id"],
                status=item.get("status", "n/a"),
                precision=number("visibility_precision"),
                recall=number("visibility_recall"),
                p90=number("observable_world_xyz_p90_m", 1000.0),
                maximum=number("observable_world_xyz_max_m", 1000.0),
                accepted=item["write_accepted_count"],
                unsafe=item["unsafe_accepted_count"],
                coverage=number("accepted_safe_coverage"),
                covariance=number("covariance_95_coverage"),
                covariance_count=item["covariance_evaluable_count"],
            )
        )
    lines.extend(
        [
            "",
            "本结果仅是 simulator development qualification；不授权 Memory write、",
            "manipulation resume、actuator promotion 或 test-once。",
            "",
        ]
    )
    return "\n".join(lines)


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
        _fsync_parent_directory(path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_hashes(output_root: Path, names: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "path": name,
            "sha256": file_sha256(output_root / name),
            "size_bytes": (output_root / name).stat().st_size,
        }
        for name in names
    ]


def verify_g2a_receipt(output_root: str | Path) -> dict[str, Any]:
    """复核完成 receipt 的内部 hash、固定文件集合及每个 artifact 身份。"""

    root = Path(output_root)
    receipt = _require_keys(
        _read_json(root / "receipt.json", "G2A receipt"),
        {
            "version",
            "status",
            "gate",
            "gate_evaluated",
            "gate_passed",
            "config_sha256",
            "source_identity_sha256",
            "prediction_ledger_sha256",
            "prediction_frozen_before_gt",
            "test_split_status",
            "test_trajectory_array_read_count",
            "test_label_array_read_count",
            "live_memory_read_count",
            "live_memory_write_count",
            "runtime_camera_actuation_count",
            "arm_actuation_count",
            "manipulation_progression_count",
            "provider_training_count",
            "files",
            "receipt_sha256",
        },
        "G2A receipt",
    )
    receipt_sha = _require_sha256(receipt["receipt_sha256"], "receipt_sha256")
    receipt_payload = dict(receipt)
    del receipt_payload["receipt_sha256"]
    if canonical_sha256(receipt_payload) != receipt_sha:
        raise RuntimeError("G2A receipt internal SHA-256 漂移")
    for name in (
        "config_sha256",
        "source_identity_sha256",
        "prediction_ledger_sha256",
    ):
        _require_sha256(receipt[name], f"receipt.{name}")
    if (
        receipt["version"] != E018_P1_G2A_RESULT_VERSION
        or receipt["gate"] != E018_P1_G2A_GATE
        or receipt["prediction_frozen_before_gt"] is not True
        or receipt["test_split_status"]
        != "manifest-metadata-read-arrays-prohibited-unread"
        or any(
            receipt[name] != 0
            for name in (
                "test_trajectory_array_read_count",
                "test_label_array_read_count",
                "live_memory_read_count",
                "live_memory_write_count",
                "runtime_camera_actuation_count",
                "arm_actuation_count",
                "manipulation_progression_count",
                "provider_training_count",
            )
        )
    ):
        raise RuntimeError("G2A receipt scope/counter 漂移")
    if receipt["status"] == "complete-preflight-no-qualification-claim":
        if receipt["gate_evaluated"] is not False or receipt["gate_passed"] is not None:
            raise RuntimeError("G2A preflight receipt gate 语义漂移")
    elif receipt["status"] == "complete-development-only":
        if receipt["gate_evaluated"] is not True or not isinstance(
            receipt["gate_passed"], bool
        ):
            raise RuntimeError("G2A development receipt gate 语义漂移")
    else:
        raise RuntimeError("G2A receipt completion status 漂移")

    file_records = receipt["files"]
    if not isinstance(file_records, list):
        raise RuntimeError("G2A receipt files 必须是 list")
    by_path: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(file_records):
        record = _require_keys(value, {"path", "sha256", "size_bytes"}, f"files[{index}]")
        relative = record["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in by_path
        ):
            raise RuntimeError("G2A receipt artifact path 无效或重复")
        _require_sha256(record["sha256"], f"files[{index}].sha256")
        if (
            not isinstance(record["size_bytes"], int)
            or isinstance(record["size_bytes"], bool)
            or record["size_bytes"] < 0
        ):
            raise RuntimeError("G2A receipt artifact size 无效")
        by_path[relative] = record
    if set(by_path) != set(_ARTIFACT_NAMES):
        raise RuntimeError("G2A receipt artifact 集合漂移")
    for relative, record in by_path.items():
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or file_sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"G2A receipt artifact identity 漂移: {relative}")
    config_snapshot = _read_json(root / "config_snapshot.json", "config snapshot")
    source_identity = _read_json(root / "source_identity.json", "source identity")
    prediction_freeze = _read_json(root / "prediction_freeze.json", "prediction freeze")
    if (
        canonical_sha256(config_snapshot) != receipt["config_sha256"]
        or source_identity.get("identity_sha256")
        != receipt["source_identity_sha256"]
        or prediction_freeze.get("prediction_ledger_sha256")
        != receipt["prediction_ledger_sha256"]
    ):
        raise RuntimeError("G2A receipt config/source/prediction 绑定漂移")
    return receipt


def run_e018_p1_g2a(
    *,
    config_path: str | Path,
    parent_g0c_config_path: str | Path,
    parent_g0c_receipt_path: str | Path,
    e016_config_path: str | Path,
    e013_deployable_root: str | Path,
    e016_fresh_deployable_root: str | Path,
    training_output: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """运行 G2A；preflight 只验证同一链路，不形成资格结论。"""

    if not isinstance(preflight_only, bool):
        raise TypeError("preflight_only 必须是 bool")
    config_file = Path(config_path)
    parent_file = Path(parent_g0c_config_path)
    repository = Path(repository_root)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"E018-P1 G2A output 已存在: {output}")
    config = load_e018_p1_g2a_config(
        config_file,
        parent_g0c_config_path=parent_file,
    )
    from robot_vla.precision.e018_p1_g0c import load_e018_p1_g0c_config

    parent_g0c = load_e018_p1_g0c_config(parent_file)
    config_sha = canonical_sha256(config)
    source_identity = _source_identity(repository)
    if config["execution"]["require_clean_worktree"] and not source_identity[
        "worktree_clean"
    ]:
        raise RuntimeError("G2A 要求 clean worktree 与精确 commit identity")
    seeds = (
        [int(config["sampling"]["seeds"][0])]
        if preflight_only
        else [int(seed) for seed in config["sampling"]["seeds"]]
    )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _atomic_json(
        output / "run_state.json",
        {
            "version": E018_P1_G2A_RESULT_VERSION,
            "status": "in-progress-preflight" if preflight_only else "in-progress",
            "gate": E018_P1_G2A_GATE,
            "config_sha256": config_sha,
            "prediction_before_gt_required": True,
            "test_trajectory_array_read_count": 0,
            "test_label_array_read_count": 0,
        },
    )
    try:
        _atomic_json(output / "config_snapshot.json", config)
        _atomic_json(output / "source_identity.json", source_identity)
        _verify_g0c_receipt(Path(parent_g0c_receipt_path), config=config)
        seed_audit = audit_g2a_seed_disjointness(
            config=config,
            e013_deployable_root=e013_deployable_root,
            e016_fresh_deployable_root=e016_fresh_deployable_root,
        )
        _atomic_json(output / "seed_audit.json", seed_audit)
        pose_envelope = build_e013_wrist_pose_envelope(
            config=config,
            e013_deployable_root=e013_deployable_root,
        )
        _atomic_json(output / "wrist_pose_envelope.json", pose_envelope)
        context = _load_model_context(
            config=config,
            e016_config_path=Path(e016_config_path),
            training_output=Path(training_output),
            stats_root=Path(e013_deployable_root),
        )
        captures, capture_audit = _capture_deployable_phase(
            config=config,
            parent_g0c=parent_g0c,
            context=context,
            pose_envelope=pose_envelope,
            seeds=seeds,
        )
        _atomic_json(output / "deployable_capture_audit.json", capture_audit)
        predictions, inference_audit = _predict_captures(
            captures,
            context=context,
            config=config,
        )
        _atomic_json(output / "inference_audit.json", inference_audit)
        prediction_freeze = freeze_prediction_ledger(
            output,
            rows=predictions,
            config_sha256=config_sha,
        )
        # Phase B 的 API 不接收这些可变/可执行对象；唯一预测输入来自冻结 ledger。
        del captures
        del predictions
        scored_rows, scoring_audit = _score_after_prediction_freeze(
            config=config,
            parent_g0c=parent_g0c,
            spec=context.spec,
            pose_envelope=pose_envelope,
            seeds=seeds,
            output_root=output,
        )
        _atomic_jsonl(output / "offline_scoring_ledger.jsonl", scored_rows)
        _atomic_json(output / "offline_scoring_audit.json", scoring_audit)
        if preflight_only:
            if not (
                scoring_audit["capture_integrity_passed"]
                and scoring_audit["physical_safety_passed"]
            ):
                raise RuntimeError("G2A preflight capture integrity/physical safety 未通过")
            summary = {
                "version": E018_P1_G2A_RESULT_VERSION,
                "status": "preflight-pass-no-qualification-claim",
                "gate": E018_P1_G2A_GATE,
                "gate_evaluated": False,
                "config_sha256": config_sha,
                "source_identity_sha256": source_identity["identity_sha256"],
                "seed_count": len(seeds),
                "sample_count": len(scored_rows),
                "prediction_ledger_sha256": prediction_freeze[
                    "prediction_ledger_sha256"
                ],
                "capture_integrity_passed": scoring_audit[
                    "capture_integrity_passed"
                ],
                "physical_safety_passed": scoring_audit["physical_safety_passed"],
                "native_wrist_control_passed": None,
                "primary": None,
                "viewpoint_summaries": [],
                "allowed_conclusion": (
                    "G2A runtime/schema two-phase ordering works for one frozen seed; "
                    "no provider qualification claim"
                ),
            }
        else:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in scored_rows:
                grouped[str(row["primitive_id"])].append(row)
            viewpoint_summaries = [
                summarize_qualification_rows(grouped[primitive_id], config=config)
                for primitive_id in PER_SCENE_CAPTURE_ORDER
            ]
            final = finalize_qualification_summaries(viewpoint_summaries)
            if final["status"] == "pass":
                allowed_conclusion = (
                    "at least one G0C front alternate passed development-only "
                    "qualification for the frozen wrist checkpoint plus explicit "
                    "camera-role adapter"
                )
            elif final["status"] == "inconclusive_parent_health":
                allowed_conclusion = (
                    "native wrist control failed the same gates; front results are "
                    "inconclusive and cannot be attributed to camera-role shift"
                )
            else:
                allowed_conclusion = (
                    "no G0C front alternate passed for the current frozen wrist "
                    "checkpoint plus camera-role adapter; this does not show that "
                    "active vision itself is ineffective"
                )
            summary = {
                "version": E018_P1_G2A_RESULT_VERSION,
                "status": final["status"],
                "gate": E018_P1_G2A_GATE,
                "gate_evaluated": True,
                "gate_passed": final["status"] == "pass",
                "config_sha256": config_sha,
                "source_identity_sha256": source_identity["identity_sha256"],
                "seed_count": len(seeds),
                "sample_count": len(scored_rows),
                "prediction_ledger_sha256": prediction_freeze[
                    "prediction_ledger_sha256"
                ],
                "native_wrist_control_passed": final[
                    "native_wrist_control_passed"
                ],
                "qualified_front_alternate_ids": final[
                    "qualified_front_alternate_ids"
                ],
                "primary": final["primary"],
                "viewpoint_summaries": final["summaries"],
                "allowed_conclusion": allowed_conclusion,
                "test_manifest_metadata_read_count": 2,
                "test_trajectory_array_read_count": 0,
                "test_label_array_read_count": 0,
                "live_memory_read_count": 0,
                "live_memory_write_count": 0,
                "runtime_camera_actuation_count": 0,
                "arm_actuation_count": 0,
                "manipulation_progression_count": 0,
                "provider_training_count": 0,
                "static_qualification_pose_configuration_count": (
                    capture_audit[
                        "static_qualification_pose_configuration_count"
                    ]
                    + scoring_audit[
                        "static_qualification_pose_configuration_count"
                    ]
                ),
                "environment_step_count": 0,
            }
        _atomic_json(output / "summary.json", summary)
        _atomic_text(output / "report.md", _report_markdown(summary))
        receipt = {
            "version": E018_P1_G2A_RESULT_VERSION,
            "status": (
                "complete-preflight-no-qualification-claim"
                if preflight_only
                else "complete-development-only"
            ),
            "gate": E018_P1_G2A_GATE,
            "gate_evaluated": not preflight_only,
            "gate_passed": (
                None if preflight_only else bool(summary["gate_passed"])
            ),
            "config_sha256": config_sha,
            "source_identity_sha256": source_identity["identity_sha256"],
            "prediction_ledger_sha256": prediction_freeze[
                "prediction_ledger_sha256"
            ],
            "prediction_frozen_before_gt": True,
            "test_split_status": "manifest-metadata-read-arrays-prohibited-unread",
            "test_trajectory_array_read_count": 0,
            "test_label_array_read_count": 0,
            "live_memory_read_count": 0,
            "live_memory_write_count": 0,
            "runtime_camera_actuation_count": 0,
            "arm_actuation_count": 0,
            "manipulation_progression_count": 0,
            "provider_training_count": 0,
            "files": _artifact_hashes(output, list(_ARTIFACT_NAMES)),
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        _atomic_json(output / "receipt.json", receipt)
        receipt = verify_g2a_receipt(output)
        receipt_raw_sha256 = file_sha256(output / "receipt.json")
        _atomic_json(
            output / "run_state.json",
            {
                "version": E018_P1_G2A_RESULT_VERSION,
                "status": "complete",
                "receipt_sha256": receipt["receipt_sha256"],
                "receipt_raw_sha256": receipt_raw_sha256,
                "summary_status": summary["status"],
            },
        )
        return {"summary": summary, "receipt": receipt}
    except Exception as error:
        _atomic_json(
            output / "failure.json",
            {
                "version": E018_P1_G2A_RESULT_VERSION,
                "status": "failed-preserved",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "prediction_ledger_exists": (output / "prediction_ledger.jsonl").is_file(),
                "prediction_ledger_sha256": (
                    file_sha256(output / "prediction_ledger.jsonl")
                    if (output / "prediction_ledger.jsonl").is_file()
                    else None
                ),
                "test_trajectory_array_read_count": 0,
                "test_label_array_read_count": 0,
                "live_memory_read_count": 0,
                "live_memory_write_count": 0,
                "runtime_camera_actuation_count": 0,
                "arm_actuation_count": 0,
                "manipulation_progression_count": 0,
            },
        )
        _atomic_json(
            output / "run_state.json",
            {
                "version": E018_P1_G2A_RESULT_VERSION,
                "status": "failed-preserved",
                "error_type": type(error).__name__,
            },
        )
        raise


__all__ = [
    "E018_P1_G2A_CONFIG_VERSION",
    "E018_P1_G2A_GATE",
    "E018_P1_G2A_RESULT_VERSION",
    "FRONT_ALTERNATE_IDS",
    "FRONT_HOME_ID",
    "G0B_SHORTLIST_ORDER",
    "NATIVE_WRIST_CONTROL_ID",
    "PER_SCENE_CAPTURE_ORDER",
    "PRIMARY_TIE_BREAK_FIELDS",
    "audit_g2a_seed_disjointness",
    "audit_qualification_seed_sets",
    "assert_prediction_ledger_deployable_only",
    "build_e013_wrist_pose_envelope",
    "camera_pose_ood_diagnostic",
    "canonical_sha256",
    "file_sha256",
    "finalize_qualification_summaries",
    "freeze_prediction_ledger",
    "load_e018_p1_g2a_config",
    "load_frozen_prediction_ledger",
    "run_e018_p1_g2a",
    "select_primary_front_viewpoint",
    "summarize_qualification_rows",
    "verify_g2a_receipt",
]
