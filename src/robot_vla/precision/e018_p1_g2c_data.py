"""E018-P1 G2C：front-provider 专用 free-static 数据合同与采集器。

本模块只创建隔离的 development 数据。模型输入与 privileged label 分开写入；
reset 返回帧只进入 diagnostic，固定五个 SafeHold-open step 之后的帧才可能 eligible。
它不读取 test array，不接 Memory，也不给 canonical camera/controller 任何权限。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
from robot_vla.data.trajectory import load_manifest
from robot_vla.observation import (
    OBSERVATION_V2_FRAME_STATE_DIM,
    invert_se3,
    opengl_camera_to_opencv,
    validate_se3,
)
from robot_vla.precision.active_external_observation import _closest_rigid_transform
from robot_vla.precision.active_front_camera import rotation_angular_distance_rad
from robot_vla.precision.active_front_provider import build_precision_camera_role_state
from robot_vla.precision.e018_p1_g2a import (
    FRONT_ALTERNATE_IDS,
    FRONT_HOME_ID,
    _normalizer_identities,
    _viewpoint_map,
    canonical_sha256,
    file_sha256,
)
from robot_vla.precision.geometry import (
    normalized_uv_to_base_z_plane,
    project_base_point_to_normalized_uv,
)
from robot_vla.precision.object_observability import derive_object_observability
from robot_vla.precision.observability import derive_goal_observability

E018_P1_G2C_DATA_CONFIG_VERSION = (
    "e018-p1-g2c-front-provider-data-development/v1"
)
E018_P1_G2C_DATA_RESULT_VERSION = (
    "e018-p1-g2c-front-provider-data-result/v1"
)
E018_P1_G2C_DATA_GATE = "G2C_FRONT_PROVIDER_ADAPTATION_DATA"
G2C_DEPLOYABLE_SCHEMA_VERSION = (
    "e018-p1-g2c-front-deployable-seed-bundle/v1"
)
G2C_LABEL_SCHEMA_VERSION = (
    "e018-p1-g2c-front-privileged-label-seed-bundle/v1"
)
G2C_MANIFEST_SCHEMA_VERSION = "e018-p1-g2c-front-manifest/v1"
G2C_VIEW_ORDER = (FRONT_HOME_ID, *FRONT_ALTERNATE_IDS)
G2C_STATIC_SPLITS = ("train", "model_val", "calibration")
G2C_ALL_SPLITS = (*G2C_STATIC_SPLITS, "qualification")
G2C_SMOKE_SPLIT = "engineering_smoke"
G2C_STATIC_CAPTURE_SIMULATION_CONTROL_TIME_S = 0.25
G2C_RAW_RESET_DIAGNOSTIC_PHASE = "raw-reset-return-before-warmup/v1"
G2C_POST_WARMUP_DIAGNOSTIC_PHASE = "post-five-safe-hold-open-warmup/v1"
G2C_RESET_DIAGNOSTIC_PHASES = (
    G2C_RAW_RESET_DIAGNOSTIC_PHASE,
    G2C_POST_WARMUP_DIAGNOSTIC_PHASE,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SOURCE_FILES = (
    "configs/e018_p1_g2c_front_provider_data_development_v1.json",
    "src/robot_vla/precision/e018_p1_g2c_data.py",
    "src/robot_vla/precision/e018_p1_g2c.py",
    "src/robot_vla/cli/run_e018_p1_g2c.py",
)


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
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} 必须是 64 位小写 SHA-256")
    return value


def _require_finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} 必须是满足下界的有限数值")
    return result


def _expand_seed_spec(value: Mapping[str, Any], name: str) -> list[int]:
    keys = set(value)
    if keys == {"values"}:
        seeds = value["values"]
        if not isinstance(seeds, list):
            raise TypeError(f"{name}.values 必须是 list")
    elif keys in (
        {"start_seed", "end_seed"},
        {"start_seed", "end_seed", "seed_count", "capture_mode"},
    ):
        start = value["start_seed"]
        end = value["end_seed"]
        if any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in (start, end)
        ) or start > end:
            raise ValueError(f"{name} seed range 无效")
        seeds = list(range(start, end + 1))
        if "seed_count" in value and value["seed_count"] != len(seeds):
            raise ValueError(f"{name}.seed_count 与闭区间不一致")
    else:
        raise ValueError(f"{name} seed spec keys 漂移")
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed <= 0
            for seed in seeds
        )
    ):
        raise ValueError(f"{name} 必须展开为唯一正整数 seed")
    return list(seeds)


def g2c_split_seeds(config: Mapping[str, Any]) -> dict[str, list[int]]:
    formal = config["sampling"]["formal_splits"]
    return {
        name: _expand_seed_spec(formal[name], f"sampling.formal_splits.{name}")
        for name in G2C_ALL_SPLITS
    }


def _validate_pairwise_seed_disjoint(groups: Mapping[str, Sequence[int]]) -> None:
    names = tuple(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(set(groups[left]).intersection(groups[right]))
            if overlap:
                raise ValueError(f"G2C seed groups 重叠: {left}/{right}={overlap}")


def load_e018_p1_g2c_data_config(
    path: str | Path,
    *,
    parent_g0c_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """严格读取 D036 冻结的 DATA/v1；不接受运行后补写的研究参数。"""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"G2C DATA config 不存在: {config_path}")
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
            "capture",
            "dataset",
            "engineering_smoke",
            "execution",
        },
        "G2C DATA config",
    )
    if config["version"] != E018_P1_G2C_DATA_CONFIG_VERSION:
        raise ValueError("G2C DATA config version 漂移")
    if config["status"] != "development-only-data-protocol-no-formal-claim":
        raise ValueError("G2C DATA 只能是 development-only")

    scope = _require_keys(
        config["scope"],
        {
            "gate",
            "test_manifest_metadata_read_allowed",
            "test_trajectory_array_read_allowed",
            "test_label_array_read_allowed",
            "formal_claim_allowed",
            "runtime_gt_control_allowed",
            "runtime_camera_actuation_allowed",
            "physical_camera_actuation_allowed",
            "arm_actuation_allowed",
            "memory_read_allowed",
            "memory_write_allowed",
            "manipulation_progression_allowed",
        },
        "scope",
    )
    expected_scope = {
        "gate": E018_P1_G2C_DATA_GATE,
        "test_manifest_metadata_read_allowed": True,
        "test_trajectory_array_read_allowed": False,
        "test_label_array_read_allowed": False,
        "formal_claim_allowed": False,
        "runtime_gt_control_allowed": False,
        "runtime_camera_actuation_allowed": False,
        "physical_camera_actuation_allowed": False,
        "arm_actuation_allowed": False,
        "memory_read_allowed": False,
        "memory_write_allowed": False,
        "manipulation_progression_allowed": False,
    }
    if scope != expected_scope:
        raise ValueError("G2C DATA scope 必须保持 metadata-only/no-test/no-actuation")

    parents = _require_keys(
        config["parents"],
        {
            "decision_id",
            "g0c_config_version",
            "g0c_config_sha256",
            "g0c_receipt_internal_sha256",
            "e016_config_sha256",
            "e016_checkpoint_sha256",
            "e016_checkpoint_parameter_sha256",
            "e016_checkpoint_provenance_sha256",
            "e016_checkpoint_model_config_sha256",
            "source_training_camera",
            "target_training_camera",
        },
        "parents",
    )
    for name in (
        "g0c_config_sha256",
        "g0c_receipt_internal_sha256",
        "e016_config_sha256",
        "e016_checkpoint_sha256",
        "e016_checkpoint_parameter_sha256",
        "e016_checkpoint_provenance_sha256",
        "e016_checkpoint_model_config_sha256",
    ):
        _require_sha256(parents[name], f"parents.{name}")
    if (
        parents["decision_id"] != "D037"
        or parents["g0c_config_version"]
        != "e018-p1-g0c-rotated-motion-development/v1"
        or parents["source_training_camera"] != "hand_camera"
        or parents["target_training_camera"] != "base_camera"
    ):
        raise ValueError("G2C DATA parent identity/camera role 漂移")
    if parent_g0c_config_path is not None:
        from robot_vla.precision.e018_p1_g0c import load_e018_p1_g0c_config

        parent = load_e018_p1_g0c_config(parent_g0c_config_path)
        if canonical_sha256(parent) != parents["g0c_config_sha256"]:
            raise ValueError("G2C DATA parent G0C canonical SHA-256 漂移")

    data_identity = _require_keys(
        config["data_identity"],
        {
            "e013_manifest_sha256",
            "e016_fresh_manifest_sha256",
            "proprio_stats_sha256",
            "proprio_normalizer_sha256",
            "finger_force_stats_sha256",
            "finger_force_normalizer_sha256",
        },
        "data_identity",
    )
    for name, value in data_identity.items():
        _require_sha256(value, f"data_identity.{name}")

    if config["software"] != {
        "expected_mani_skill_version": "3.0.1",
        "expected_sapien_version": "3.0.3",
    }:
        raise ValueError("G2C DATA software identity 漂移")
    environment = _require_keys(
        config["environment"],
        {
            "environment_id",
            "robot_uid",
            "external_camera_uid",
            "obs_mode",
            "control_mode",
            "num_envs",
            "control_hz",
            "image_shape_hwc",
        },
        "environment",
    )
    if environment != {
        "environment_id": "RobotVLAPickCubeToRegion-v1",
        "robot_uid": "panda_wristcam",
        "external_camera_uid": "base_camera",
        "obs_mode": "rgb+segmentation",
        "control_mode": "pd_joint_delta_pos",
        "num_envs": 1,
        "control_hz": 20,
        "image_shape_hwc": [128, 128, 3],
    }:
        raise ValueError("G2C DATA environment identity 漂移")

    sampling = _require_keys(
        config["sampling"],
        {
            "seed_policy",
            "formal_splits",
            "test_split",
            "smoke_only_seeds",
            "known_development_seed_groups",
            "require_pairwise_split_disjoint",
            "require_disjoint_from_manifest_metadata",
            "require_disjoint_from_inventory_registry",
            "seed_replacement_allowed",
            "row_deletion_allowed",
        },
        "sampling",
    )
    if (
        sampling["seed_policy"] != "four-disjoint-frozen-development-splits/v1"
        or sampling["test_split"] is not None
        or sampling["require_pairwise_split_disjoint"] is not True
        or sampling["require_disjoint_from_manifest_metadata"] is not True
        or sampling["require_disjoint_from_inventory_registry"] is not True
        or sampling["seed_replacement_allowed"] is not False
        or sampling["row_deletion_allowed"] is not False
    ):
        raise ValueError("G2C DATA seed/test fail-closed 规则漂移")
    formal = _require_keys(
        sampling["formal_splits"], set(G2C_ALL_SPLITS), "sampling.formal_splits"
    )
    expected_ranges = {
        "train": (76001, 76400, 400, "static-render"),
        "model_val": (76501, 76600, 100, "static-render"),
        "calibration": (76601, 76650, 50, "static-render"),
        "qualification": (76701, 76750, 50, "dynamic-roundtrip"),
    }
    for name, expected in expected_ranges.items():
        spec = _require_keys(
            formal[name],
            {"start_seed", "end_seed", "seed_count", "capture_mode"},
            f"sampling.formal_splits.{name}",
        )
        if (
            spec["start_seed"],
            spec["end_seed"],
            spec["seed_count"],
            spec["capture_mode"],
        ) != expected:
            raise ValueError(f"G2C DATA {name} split identity 漂移")
        _expand_seed_spec(spec, f"sampling.formal_splits.{name}")
    smoke = sampling["smoke_only_seeds"]
    if smoke != [76801, 76802, 76803, 76804]:
        raise ValueError("G2C DATA smoke-only seed identity 漂移")
    known = _require_keys(
        sampling["known_development_seed_groups"],
        {"g0", "g0b", "g0c", "g1a", "g2a"},
        "sampling.known_development_seed_groups",
    )
    expanded = {
        name: _expand_seed_spec(spec, f"known development {name}")
        for name, spec in known.items()
    }
    expected_known = {
        "g0": [71001, 71013, 71027, 71039],
        "g0b": list(range(72001, 72051)),
        "g0c": [73001, 73013, 73027, 73039],
        "g1a": [74101],
        "g2a": list(range(75001, 75051)),
    }
    if expanded != expected_known:
        raise ValueError("G2C DATA known development seed registry 漂移")
    split_groups = {**g2c_split_seeds(config), G2C_SMOKE_SPLIT: list(smoke)}
    _validate_pairwise_seed_disjoint(split_groups)
    if any(set(values).intersection(smoke) for values in expanded.values()):
        raise ValueError("G2C DATA smoke seed 与既有 development seed 重叠")

    viewpoints = _require_keys(
        config["viewpoints"],
        {
            "library_source",
            "home_id",
            "capture_order",
            "expected_view_count",
            "non_home_view_count",
        },
        "viewpoints",
    )
    if viewpoints != {
        "library_source": "frozen-g0c-parent-config/v1",
        "home_id": FRONT_HOME_ID,
        "capture_order": list(G2C_VIEW_ORDER),
        "expected_view_count": 11,
        "non_home_view_count": 10,
    }:
        raise ValueError("G2C DATA viewpoint order/count 漂移")

    capture = _require_keys(
        config["capture"],
        {
            "raw_reset_observation_role",
            "reset_warmup_command",
            "reset_warmup_ticks",
            "reset_warmup_frequency_hz",
            "static_capture_policy",
            "actual_pose_source",
            "frame_convention",
            "timestamp_source",
            "maximum_rgb_pose_skew_s",
            "maximum_rotation_projection_error_frobenius",
            "maximum_camera_position_tracking_error_m",
            "maximum_camera_orientation_tracking_error_rad",
            "maximum_geometry_roundtrip_error_m",
            "support_radius_px",
            "geometric_motion_value",
            "eligible_lifecycle_invariants",
        },
        "capture",
    )
    if (
        capture["raw_reset_observation_role"]
        != "reset-diagnostic-only-never-eligible/v1"
        or capture["reset_warmup_command"] != "safe-hold-open"
        or capture["reset_warmup_ticks"] != 5
        or capture["reset_warmup_frequency_hz"] != 20
        or (
            float(capture["reset_warmup_ticks"])
            / float(capture["reset_warmup_frequency_hz"])
            != G2C_STATIC_CAPTURE_SIMULATION_CONTROL_TIME_S
        )
        or capture["static_capture_policy"]
        != "set-pose-render-refresh-one-eligible-frame-no-environment-step/v1"
        or capture["actual_pose_source"]
        != "same-observation.sensor_param.base_camera.cam2world_gl/v1"
        or capture["frame_convention"]
        != "robot-base-from-opencv-optical-camera/v1"
        or capture["timestamp_source"]
        != "post-warmup-simulation-control-time-shared-no-step-static/v1"
        or float(capture["maximum_rgb_pose_skew_s"]) != 0.01
        or float(capture["maximum_rotation_projection_error_frobenius"]) != 1e-6
        or float(capture["maximum_camera_position_tracking_error_m"]) != 1e-5
        or float(capture["maximum_camera_orientation_tracking_error_rad"])
        != 1e-4
        or float(capture["maximum_geometry_roundtrip_error_m"]) != 1e-5
        or capture["support_radius_px"] != 2
        or capture["geometric_motion_value"] != [0.0, 0.0, 0.0, 0.0]
    ):
        raise ValueError("G2C DATA capture/timestamp/frame contract 漂移")
    lifecycle = _require_keys(
        capture["eligible_lifecycle_invariants"],
        {
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
            "fail_whole_split",
        },
        "capture.eligible_lifecycle_invariants",
    )
    expected_lifecycle = {
        "object_center_base_z_m": 0.02,
        "object_center_base_z_tolerance_m": 1e-5,
        "require_not_grasped": True,
        "require_finger_force_valid": True,
        "maximum_finger_force_n": 0.01,
        "minimum_raw_gripper_opening_ratio": 0.95,
        "maximum_arm_joint_drift_rad": 1e-5,
        "maximum_tcp_position_drift_m": 1e-5,
        "maximum_tcp_orientation_drift_rad": 1e-4,
        "maximum_robot_object_contact_force_n": 0.01,
        "fail_whole_split": True,
    }
    if lifecycle != expected_lifecycle:
        raise ValueError("G2C DATA lifecycle invariant 漂移")

    dataset = _require_keys(
        config["dataset"],
        {
            "deployable_schema_version",
            "privileged_label_schema_version",
            "manifest_schema_version",
            "privileged_sidecar_required",
            "prediction_before_label_required_for",
            "data_receipt_must_precede_train_config",
            "train_config_must_bind_data_receipt_sha256",
            "data_and_train_identity_must_be_distinct",
        },
        "dataset",
    )
    if dataset != {
        "deployable_schema_version": G2C_DEPLOYABLE_SCHEMA_VERSION,
        "privileged_label_schema_version": G2C_LABEL_SCHEMA_VERSION,
        "manifest_schema_version": G2C_MANIFEST_SCHEMA_VERSION,
        "privileged_sidecar_required": True,
        "prediction_before_label_required_for": [
            "model_val",
            "calibration",
            "qualification",
        ],
        "data_receipt_must_precede_train_config": True,
        "train_config_must_bind_data_receipt_sha256": True,
        "data_and_train_identity_must_be_distinct": True,
    }:
        raise ValueError("G2C DATA/label/TRAIN identity contract 漂移")

    smoke_config = _require_keys(
        config["engineering_smoke"],
        {
            "mode",
            "seed_count",
            "training_seeds",
            "prediction_freeze_seed",
            "expected_eligible_capture_count",
            "training_optimizer_steps_per_candidate",
            "candidate_ids",
            "checkpoint_persistence_allowed",
            "formal_split_consumed",
            "canonical_data_receipt_allowed",
        },
        "engineering_smoke",
    )
    if smoke_config != {
        "mode": "non-canonical-four-seed-no-checkpoint/v1",
        "seed_count": 4,
        "training_seeds": [76801, 76802, 76803],
        "prediction_freeze_seed": 76804,
        "expected_eligible_capture_count": 44,
        "training_optimizer_steps_per_candidate": 2,
        "candidate_ids": ["W-KV0", "S"],
        "checkpoint_persistence_allowed": False,
        "formal_split_consumed": False,
        "canonical_data_receipt_allowed": False,
    }:
        raise ValueError("G2C engineering smoke identity 漂移")

    execution = _require_keys(
        config["execution"],
        {
            "device",
            "use_bf16",
            "batch_size",
            "num_workers",
            "full_data_collection_requires_r2_go",
            "formal_training_allowed",
            "formal_model_selection_allowed",
            "formal_calibration_allowed",
            "formal_qualification_allowed",
            "runtime_camera_actuation_allowed",
            "physical_camera_actuation_allowed",
            "arm_motion_command_allowed",
            "gripper_command_mode",
            "memory_read_allowed",
            "memory_write_allowed",
            "manipulation_progression_allowed",
            "output_overwrite_allowed",
        },
        "execution",
    )
    if execution != {
        "device": "cuda",
        "use_bf16": True,
        "batch_size": 32,
        "num_workers": 0,
        "full_data_collection_requires_r2_go": True,
        "formal_training_allowed": False,
        "formal_model_selection_allowed": False,
        "formal_calibration_allowed": False,
        "formal_qualification_allowed": False,
        "runtime_camera_actuation_allowed": False,
        "physical_camera_actuation_allowed": False,
        "arm_motion_command_allowed": False,
        "gripper_command_mode": "safe-hold-open",
        "memory_read_allowed": False,
        "memory_write_allowed": False,
        "manipulation_progression_allowed": False,
        "output_overwrite_allowed": False,
    }:
        raise ValueError("G2C DATA execution 权限或 smoke-only 状态漂移")
    return config


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
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_parent(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} 不存在: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{name} 第 {line_number} 行为空")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{name} 第 {line_number} 行必须是 object")
            rows.append(value)
    return rows


def _source_identity(repository_root: Path) -> dict[str, Any]:
    missing = [name for name in _SOURCE_FILES if not (repository_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"G2C source files 缺失: {missing}")
    safe_repository = str(repository_root.resolve())
    git = ("git", "-c", f"safe.directory={safe_repository}")
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
            name: file_sha256(repository_root / name) for name in _SOURCE_FILES
        },
    }
    identity["worktree_clean"] = not identity["git_status"]
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def _inventory_seed_values(value: Any) -> set[int]:
    """只识别显式 seed 字段；普通数字（计数、时间、hash）不被误报。"""

    found: set[int] = set()
    if isinstance(value, dict):
        if isinstance(value.get("seed"), int) and not isinstance(value["seed"], bool):
            found.add(int(value["seed"]))
        seeds = value.get("seeds")
        if isinstance(seeds, list):
            found.update(
                int(item)
                for item in seeds
                if isinstance(item, int) and not isinstance(item, bool)
            )
        start = value.get("start_seed")
        end = value.get("end_seed")
        if (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 < start <= end
            and end - start <= 100_000
        ):
            found.update(range(start, end + 1))
        for child in value.values():
            found.update(_inventory_seed_values(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_inventory_seed_values(child))
    return found


def audit_g2c_seed_disjointness(
    *,
    config: Mapping[str, Any],
    e013_deployable_root: str | Path,
    e016_fresh_deployable_root: str | Path,
    inventory_path: str | Path,
) -> dict[str, Any]:
    """只读 manifest/inventory 元数据，绝不打开 trajectory 或 label arrays。"""

    roots = {
        "e013_all_splits": Path(e013_deployable_root),
        "e016_fresh_all_splits": Path(e016_fresh_deployable_root),
    }
    expected_hashes = {
        "e013_all_splits": config["data_identity"]["e013_manifest_sha256"],
        "e016_fresh_all_splits": config["data_identity"][
            "e016_fresh_manifest_sha256"
        ],
    }
    manifest_seeds: dict[str, set[int]] = {}
    manifest_summary: dict[str, Any] = {}
    for name, root in roots.items():
        manifest = root / "manifest.jsonl"
        if not manifest.is_file() or file_sha256(manifest) != expected_hashes[name]:
            raise RuntimeError(f"G2C {name} manifest identity 漂移")
        entries = load_manifest(root)
        values: list[int] = []
        split_counts: Counter[str] = Counter()
        for entry in entries:
            seed = entry.randomization.get("seed")
            if not isinstance(seed, int) or isinstance(seed, bool) or seed <= 0:
                raise RuntimeError(f"G2C {name} randomization.seed 无效")
            values.append(seed)
            split_counts[entry.split] += 1
        manifest_seeds[name] = set(values)
        manifest_summary[name] = {
            "manifest_sha256": expected_hashes[name],
            "metadata_row_count": len(entries),
            "unique_seed_count": len(set(values)),
            "split_counts": dict(sorted(split_counts.items())),
            "seed_min": min(values),
            "seed_max": max(values),
        }

    inventory = Path(inventory_path)
    if not inventory.is_file():
        raise FileNotFoundError(f"G2C inventory 不存在: {inventory}")
    inventory_value = json.loads(inventory.read_text(encoding="utf-8"))
    inventory_seeds = _inventory_seed_values(inventory_value)
    groups = {
        **g2c_split_seeds(config),
        G2C_SMOKE_SPLIT: list(config["sampling"]["smoke_only_seeds"]),
    }
    _validate_pairwise_seed_disjoint(groups)
    known = {
        name: set(_expand_seed_spec(spec, f"known development {name}"))
        for name, spec in config["sampling"]["known_development_seed_groups"].items()
    }
    overlaps: dict[str, dict[str, list[int]]] = {}
    for group_name, seeds in groups.items():
        candidate = set(seeds)
        values = {
            **{
                f"manifest:{name}": sorted(candidate.intersection(items))
                for name, items in manifest_seeds.items()
            },
            **{
                f"development:{name}": sorted(candidate.intersection(items))
                for name, items in known.items()
            },
            "inventory:explicit_seed_fields": sorted(
                candidate.intersection(inventory_seeds)
            ),
        }
        overlaps[group_name] = values
    passed = not any(
        values for group in overlaps.values() for values in group.values()
    )
    result = {
        "version": E018_P1_G2C_DATA_RESULT_VERSION,
        "phase": "seed-disjoint-metadata-only-audit/v1",
        "passed": passed,
        "candidate_groups": {
            name: {
                "seed_count": len(values),
                "seed_min": min(values),
                "seed_max": max(values),
            }
            for name, values in groups.items()
        },
        "pairwise_candidate_groups_disjoint": True,
        "manifests": manifest_summary,
        "manifest_metadata_read_count": len(roots),
        "inventory": {
            "sha256": file_sha256(inventory),
            "explicit_seed_count": len(inventory_seeds),
        },
        "overlaps": overlaps,
        "test_trajectory_array_read_count": 0,
        "test_label_array_read_count": 0,
    }
    result["audit_sha256"] = canonical_sha256(result)
    if not passed:
        raise RuntimeError(f"G2C seed overlap: {overlaps}")
    return result


def _load_normalizers(
    *,
    stats_root: Path,
    config: Mapping[str, Any],
) -> tuple[RobotSpec, ProprioNormalizer, FingerForceNormalizer, dict[str, str]]:
    proprio_path = stats_root / "proprio_stats.json"
    force_path = stats_root / "finger_force_stats.json"
    expected = config["data_identity"]
    if file_sha256(proprio_path) != expected["proprio_stats_sha256"]:
        raise RuntimeError("G2C proprio stats raw SHA-256 漂移")
    if file_sha256(force_path) != expected["finger_force_stats_sha256"]:
        raise RuntimeError("G2C finger-force stats raw SHA-256 漂移")
    spec = RobotSpec()
    proprio = ProprioNormalizer(ProprioStats.from_json(proprio_path), spec)
    force = FingerForceNormalizer(FingerForceStats.from_json(force_path), spec)
    proprio_sha, force_sha = _normalizer_identities(spec, proprio, force)
    if proprio_sha != expected["proprio_normalizer_sha256"]:
        raise RuntimeError("G2C proprio normalizer semantic identity 漂移")
    if force_sha != expected["finger_force_normalizer_sha256"]:
        raise RuntimeError("G2C finger-force normalizer semantic identity 漂移")
    return spec, proprio, force, {
        "proprio_stats_sha256": expected["proprio_stats_sha256"],
        "proprio_normalizer_sha256": proprio_sha,
        "finger_force_stats_sha256": expected["finger_force_stats_sha256"],
        "finger_force_normalizer_sha256": force_sha,
    }


def _verify_g0c_receipt(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"G2C G0C receipt 不存在: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    parent = config["parents"]
    if (
        receipt.get("version") != "e018-p1-g0c-rotated-motion-result/v1"
        or receipt.get("status") != "complete-development-only"
        or receipt.get("gate_passed") is not True
        or receipt.get("config_sha256") != parent["g0c_config_sha256"]
        or receipt.get("receipt_sha256")
        != parent["g0c_receipt_internal_sha256"]
        or receipt.get("test_split_status") != "prohibited-unread"
    ):
        raise RuntimeError("G2C G0C parent receipt identity/gate 漂移")
    return receipt


_DEPLOYABLE_ARRAYS = (
    "sample_index",
    "seed",
    "viewpoint_id",
    "eligible_capture",
    "rgb_external",
    "physical_proprio",
    "structured_state",
    "geometric_motion",
    "base_from_tcp",
    "base_from_external_camera_cv",
    "actual_world_from_external_camera_gl",
    "external_intrinsic_cv",
    "rgb_timestamp_s",
    "pose_timestamp_s",
    "finger_force_n",
    "finger_force_valid",
    "raw_gripper_opening_ratio",
    "arm_joint_drift_rad",
    "tcp_position_drift_m",
    "tcp_orientation_drift_rad",
    "camera_position_tracking_error_m",
    "camera_orientation_tracking_error_rad",
    "rotation_projection_error_frobenius",
)
_LABEL_ARRAYS = (
    "source_sample_index",
    "seed",
    "viewpoint_id",
    "object_position_base_m",
    "goal_position_base_m",
    "object_mask",
    "goal_mask",
    "normalized_uv",
    "keypoint_projection_valid",
    "keypoint_observable",
    "object_exists",
    "goal_exists",
    "is_grasped",
    "robot_object_contact_force_n",
    "geometry_roundtrip_error_m",
)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _single_bool(value: Any, name: str) -> bool:
    array = _numpy(value)
    if array.size != 1:
        raise RuntimeError(f"{name} 必须是单环境 bool")
    return bool(array.reshape(-1)[0])


def _single_vector(value: Any, size: int, name: str) -> np.ndarray:
    vector = _numpy(value)
    if vector.shape == (1, size):
        vector = vector[0]
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise RuntimeError(f"{name} 必须是有限 [{size}]")
    return vector


def _single_rigid(
    value: Any,
    name: str,
    *,
    maximum_projection_error: float,
) -> tuple[np.ndarray, float]:
    if callable(getattr(value, "to_transformation_matrix", None)):
        value = value.to_transformation_matrix()
    matrix = _numpy(value)
    if matrix.shape == (1, 4, 4):
        matrix = matrix[0]
    canonical, audit = _closest_rigid_transform(
        matrix,
        name,
        maximum_rotation_projection_error_frobenius=maximum_projection_error,
    )
    return canonical, float(audit.correction_frobenius)


def _finger_force_n(base_env: Any) -> np.ndarray:
    scene = base_env.scene
    agent = base_env.agent
    cube = base_env.cube
    values = []
    for link in (agent.finger1_link, agent.finger2_link):
        force = _numpy(scene.get_pairwise_contact_forces(link, cube))
        if force.shape != (1, 3) or not np.isfinite(force).all():
            raise RuntimeError("G2C finger-cube force schema 漂移")
        values.append(float(np.linalg.norm(force[0])))
    result = np.asarray(values, dtype=np.float32)
    if np.any(result < 0.0):
        raise RuntimeError("G2C finger force 不能为负")
    return result


def _robot_object_contact_force_n(base_env: Any) -> float:
    maximum = 0.0
    for link in base_env.agent.robot.links:
        force = _numpy(base_env.scene.get_pairwise_contact_forces(link, base_env.cube))
        if force.shape != (1, 3) or not np.isfinite(force).all():
            raise RuntimeError("G2C robot-object contact force schema 漂移")
        maximum = max(maximum, float(np.linalg.norm(force[0])))
    return maximum


def _base_point(
    base_env: Any,
    actor: Any,
    *,
    maximum_projection_error: float,
) -> np.ndarray:
    world_from_base, _ = _single_rigid(
        base_env.agent.robot.pose,
        "world_from_robot_base",
        maximum_projection_error=maximum_projection_error,
    )
    point_world = _single_vector(actor.pose.p, 3, "privileged actor position world")
    point_base = invert_se3(world_from_base, "world_from_robot_base") @ np.concatenate(
        (point_world, np.ones(1, dtype=np.float64))
    )
    return np.asarray(point_base[:3], dtype=np.float32)


def _physical_state(
    *,
    base_env: Any,
    spec: RobotSpec,
    maximum_projection_error: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    robot = base_env.agent.robot
    qpos = _numpy(robot.get_qpos())
    qvel = _numpy(robot.get_qvel())
    if qpos.shape != (1, 9) or qvel.shape != (1, 9):
        raise RuntimeError("G2C 只支持单环境 Panda 9-DoF state")
    joint_names = tuple(joint.name for joint in robot.active_joints)
    proprio = FrankaObservationAdapter(spec).from_maniskill(
        qpos[0], qvel[0], joint_names
    )
    world_from_base, _ = _single_rigid(
        robot.pose,
        "world_from_robot_base",
        maximum_projection_error=maximum_projection_error,
    )
    world_from_tcp, _ = _single_rigid(
        base_env.agent.tcp_pose,
        "world_from_tcp",
        maximum_projection_error=maximum_projection_error,
    )
    base_from_tcp = validate_se3(
        invert_se3(world_from_base, "world_from_robot_base") @ world_from_tcp,
        "base_from_tcp",
    )
    finger_force = _finger_force_n(base_env)
    contact = _robot_object_contact_force_n(base_env)
    return proprio, base_from_tcp, finger_force, contact


def _reset_diagnostic(
    *,
    seed: int,
    observation: Mapping[str, Any],
    base_env: Any,
    spec: RobotSpec,
    maximum_projection_error: float,
    phase: str,
) -> dict[str, Any]:
    proprio, base_from_tcp, force, contact = _physical_state(
        base_env=base_env,
        spec=spec,
        maximum_projection_error=maximum_projection_error,
    )
    camera_uid = "base_camera"
    rgb = _numpy(observation["sensor_data"][camera_uid]["rgb"])
    if rgb.shape != (1, 128, 128, 3) or rgb.dtype != np.uint8:
        raise RuntimeError("G2C reset diagnostic RGB schema 漂移")
    object_position = _base_point(
        base_env,
        base_env.cube,
        maximum_projection_error=maximum_projection_error,
    )
    return {
        "version": E018_P1_G2C_DATA_RESULT_VERSION,
        "role": "reset-diagnostic-only-never-eligible/v1",
        "phase": phase,
        "seed": seed,
        "eligible_capture": False,
        "rgb_sha256": hashlib.sha256(np.ascontiguousarray(rgb[0]).tobytes()).hexdigest(),
        "raw_gripper_opening_ratio": float(proprio[-1]),
        "left_finger_force_n": float(force[0]),
        "right_finger_force_n": float(force[1]),
        "robot_object_contact_force_n": contact,
        "is_grasped": _single_bool(
            base_env.agent.is_grasping(base_env.cube), "is_grasped"
        ),
        "object_position_base_m": object_position.astype(float).tolist(),
        "base_from_tcp_sha256": hashlib.sha256(
            np.ascontiguousarray(base_from_tcp).tobytes()
        ).hexdigest(),
        "used_for_training": False,
        "used_for_selection": False,
        "used_for_calibration": False,
        "used_for_qualification": False,
    }


def _validate_reset_diagnostic_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int],
) -> dict[str, int]:
    """验证每个 seed 的 raw/post-warmup diagnostic 恰好各一条。"""

    seeds = tuple(expected_seeds)
    if (
        not seeds
        or any(
            not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds
        )
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("G2C reset diagnostic expected seeds 必须非空且唯一")
    expected_seed_set = set(seeds)
    pairs: Counter[tuple[int, str]] = Counter()
    usage_fields = (
        "used_for_training",
        "used_for_selection",
        "used_for_calibration",
        "used_for_qualification",
    )
    for row in rows:
        seed = row.get("seed")
        phase = row.get("phase")
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or seed not in expected_seed_set
            or phase not in G2C_RESET_DIAGNOSTIC_PHASES
            or row.get("version") != E018_P1_G2C_DATA_RESULT_VERSION
            or row.get("role") != "reset-diagnostic-only-never-eligible/v1"
            or row.get("eligible_capture") is not False
            or any(row.get(name) is not False for name in usage_fields)
        ):
            raise RuntimeError("G2C reset diagnostic phase/role/usage 漂移")
        pairs[(seed, str(phase))] += 1
    expected_pairs = Counter(
        (seed, phase)
        for seed in seeds
        for phase in G2C_RESET_DIAGNOSTIC_PHASES
    )
    if pairs != expected_pairs:
        raise RuntimeError("G2C reset diagnostic 缺 seed、缺 phase 或存在重复")
    raw_count = sum(
        count
        for (_, phase), count in pairs.items()
        if phase == G2C_RAW_RESET_DIAGNOSTIC_PHASE
    )
    post_count = sum(
        count
        for (_, phase), count in pairs.items()
        if phase == G2C_POST_WARMUP_DIAGNOSTIC_PHASE
    )
    return {
        "raw_reset_diagnostic_count": raw_count,
        "post_warmup_diagnostic_count": post_count,
        "reset_diagnostic_count": len(rows),
    }


def _project_labels(
    *,
    object_position: np.ndarray,
    goal_position: np.ndarray,
    intrinsic: np.ndarray,
    base_from_camera: np.ndarray,
    object_mask: np.ndarray,
    goal_mask: np.ndarray,
    support_radius_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = (object_position, goal_position)
    uv = np.zeros((2, 2), dtype=np.float32)
    projection_valid = np.zeros(2, dtype=np.bool_)
    roundtrip = np.zeros(2, dtype=np.float64)
    for index, position in enumerate(positions):
        try:
            projected = project_base_point_to_normalized_uv(
                position, intrinsic, base_from_camera, object_mask.shape
            )
        except ValueError:
            continue
        if np.all((projected >= 0.0) & (projected <= 1.0)):
            uv[index] = projected
            projection_valid[index] = True
            recovered = normalized_uv_to_base_z_plane(
                projected,
                intrinsic,
                base_from_camera,
                object_mask.shape,
                plane_base_z_m=float(position[2]),
            ).point_base_m
            roundtrip[index] = float(np.linalg.norm(recovered - position))
    object_observability = derive_object_observability(
        object_exists=True,
        projection_valid=bool(projection_valid[0]),
        projected_normalized_uv=uv[0] if projection_valid[0] else None,
        object_mask=object_mask,
        goal_mask=goal_mask,
        legacy_visible=bool(object_mask.any()),
        support_radius_px=support_radius_px,
    )
    goal_observability = derive_goal_observability(
        goal_exists=True,
        projection_valid=bool(projection_valid[1]),
        projected_normalized_uv=uv[1] if projection_valid[1] else None,
        goal_mask=goal_mask,
        object_mask=object_mask,
        legacy_visible=bool(goal_mask.any()),
        support_radius_px=support_radius_px,
    )
    observable = np.asarray(
        (object_observability.observable, goal_observability.observable),
        dtype=np.bool_,
    )
    return uv, projection_valid, observable, roundtrip


@dataclass(frozen=True)
class _CaptureRow:
    deployable: dict[str, Any]
    privileged: dict[str, Any]


def _capture_static_view(
    *,
    seed: int,
    sample_index: int,
    viewpoint_id: str,
    observation: Mapping[str, Any],
    commanded_world_from_gl: np.ndarray,
    base_env: Any,
    spec: RobotSpec,
    proprio_normalizer: ProprioNormalizer,
    force_normalizer: FingerForceNormalizer,
    anchor_arm_q: np.ndarray,
    anchor_base_from_tcp: np.ndarray,
    config: Mapping[str, Any],
) -> _CaptureRow:
    camera_uid = config["environment"]["external_camera_uid"]
    sensor = observation["sensor_data"][camera_uid]
    params = observation["sensor_param"][camera_uid]
    rgb = _numpy(sensor["rgb"])
    segmentation = _numpy(sensor["segmentation"])
    expected_rgb = (1, *config["environment"]["image_shape_hwc"])
    if rgb.shape != expected_rgb or rgb.dtype != np.uint8:
        raise RuntimeError(f"G2C external RGB schema 漂移: {rgb.shape}/{rgb.dtype}")
    if segmentation.ndim != 4 or segmentation.shape[:3] != (1, 128, 128):
        raise RuntimeError("G2C external segmentation schema 漂移")
    actor_ids = np.asarray(segmentation[0, ..., 0])
    if not np.issubdtype(actor_ids.dtype, np.integer):
        raise RuntimeError("G2C segmentation actor-id channel 必须为整数")
    intrinsic = _numpy(params["intrinsic_cv"])
    if intrinsic.shape == (1, 3, 3):
        intrinsic = intrinsic[0]
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
    ):
        raise RuntimeError("G2C external intrinsic_cv schema 漂移")
    maximum_projection = float(
        config["capture"]["maximum_rotation_projection_error_frobenius"]
    )
    actual_world_from_gl, camera_projection_error = _single_rigid(
        params["cam2world_gl"],
        "actual_world_from_external_camera_gl",
        maximum_projection_error=maximum_projection,
    )
    world_from_base, base_projection_error = _single_rigid(
        base_env.agent.robot.pose,
        "world_from_robot_base",
        maximum_projection_error=maximum_projection,
    )
    base_from_camera = validate_se3(
        invert_se3(world_from_base, "world_from_robot_base")
        @ opengl_camera_to_opencv(actual_world_from_gl),
        "base_from_external_camera_cv",
    )
    proprio, base_from_tcp, finger_force, robot_contact = _physical_state(
        base_env=base_env,
        spec=spec,
        maximum_projection_error=maximum_projection,
    )
    structured_state = build_precision_camera_role_state(
        spec=spec,
        proprio_normalizer=proprio_normalizer,
        finger_force_normalizer=force_normalizer,
        physical_proprio=proprio,
        base_from_tcp=base_from_tcp,
        base_from_camera_cv=base_from_camera,
        finger_force_n=finger_force,
    )
    qpos = _numpy(base_env.agent.robot.get_qpos())[0]
    arm_drift = float(np.max(np.abs(qpos[:7] - anchor_arm_q)))
    tcp_position_drift = float(
        np.linalg.norm(base_from_tcp[:3, 3] - anchor_base_from_tcp[:3, 3])
    )
    tcp_orientation_drift = rotation_angular_distance_rad(
        anchor_base_from_tcp[:3, :3], base_from_tcp[:3, :3]
    )
    commanded_world_from_gl = validate_se3(
        commanded_world_from_gl, "commanded_world_from_external_camera_gl"
    )
    camera_position_error = float(
        np.linalg.norm(
            actual_world_from_gl[:3, 3] - commanded_world_from_gl[:3, 3]
        )
    )
    camera_orientation_error = rotation_angular_distance_rad(
        commanded_world_from_gl[:3, :3], actual_world_from_gl[:3, :3]
    )
    object_actor_id = int(_numpy(base_env.cube.per_scene_id).reshape(-1)[0])
    goal_actor_id = int(_numpy(base_env.goal_site.per_scene_id).reshape(-1)[0])
    object_mask = np.asarray(actor_ids == object_actor_id, dtype=np.bool_)
    goal_mask = np.asarray(actor_ids == goal_actor_id, dtype=np.bool_)
    object_position = _base_point(
        base_env,
        base_env.cube,
        maximum_projection_error=maximum_projection,
    )
    goal_position = _base_point(
        base_env,
        base_env.goal_site,
        maximum_projection_error=maximum_projection,
    )
    uv, projected, observable, roundtrip = _project_labels(
        object_position=object_position,
        goal_position=goal_position,
        intrinsic=intrinsic,
        base_from_camera=base_from_camera,
        object_mask=object_mask,
        goal_mask=goal_mask,
        support_radius_px=int(config["capture"]["support_radius_px"]),
    )
    # 11 个 static-render view 之间没有 environment step；它们共享 warmup
    # 完成后的真实 simulation-control-time，采集顺序只由 sample_index 表达。
    timestamp = G2C_STATIC_CAPTURE_SIMULATION_CONTROL_TIME_S
    deployable = {
        "sample_index": np.int64(sample_index),
        "seed": np.int64(seed),
        "viewpoint_id": viewpoint_id,
        "eligible_capture": np.bool_(True),
        "rgb_external": np.ascontiguousarray(rgb[0]),
        "physical_proprio": proprio.astype(np.float32, copy=False),
        "structured_state": structured_state.astype(np.float32, copy=False),
        "geometric_motion": np.asarray(
            config["capture"]["geometric_motion_value"], dtype=np.float32
        ),
        "base_from_tcp": base_from_tcp.astype(np.float64, copy=False),
        "base_from_external_camera_cv": base_from_camera.astype(
            np.float64, copy=False
        ),
        "actual_world_from_external_camera_gl": actual_world_from_gl.astype(
            np.float64, copy=False
        ),
        "external_intrinsic_cv": intrinsic,
        "rgb_timestamp_s": np.float64(timestamp),
        "pose_timestamp_s": np.float64(timestamp),
        "finger_force_n": finger_force,
        "finger_force_valid": np.bool_(True),
        "raw_gripper_opening_ratio": np.float32(proprio[-1]),
        "arm_joint_drift_rad": np.float64(arm_drift),
        "tcp_position_drift_m": np.float64(tcp_position_drift),
        "tcp_orientation_drift_rad": np.float64(tcp_orientation_drift),
        "camera_position_tracking_error_m": np.float64(camera_position_error),
        "camera_orientation_tracking_error_rad": np.float64(
            camera_orientation_error
        ),
        "rotation_projection_error_frobenius": np.float64(
            max(camera_projection_error, base_projection_error)
        ),
    }
    privileged = {
        "source_sample_index": np.int64(sample_index),
        "seed": np.int64(seed),
        "viewpoint_id": viewpoint_id,
        "object_position_base_m": object_position,
        "goal_position_base_m": goal_position,
        "object_mask": object_mask,
        "goal_mask": goal_mask,
        "normalized_uv": uv,
        "keypoint_projection_valid": projected,
        "keypoint_observable": observable,
        "object_exists": np.bool_(True),
        "goal_exists": np.bool_(True),
        "is_grasped": np.bool_(
            _single_bool(base_env.agent.is_grasping(base_env.cube), "is_grasped")
        ),
        "robot_object_contact_force_n": np.float32(robot_contact),
        "geometry_roundtrip_error_m": roundtrip,
    }
    return _CaptureRow(deployable=deployable, privileged=privileged)


def _stack_rows(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("G2C seed bundle 不能为空")
    if any(set(row) != set(names) for row in rows):
        raise ValueError("G2C seed bundle row keys 漂移")
    return {name: np.stack([np.asarray(row[name]) for row in rows]) for name in names}


def _require_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type[np.generic],
) -> np.ndarray:
    value = arrays[name]
    if value.shape != shape or value.dtype != np.dtype(dtype):
        raise ValueError(
            f"G2C {name} shape/dtype 漂移: {value.shape}/{value.dtype} != "
            f"{shape}/{np.dtype(dtype)}"
        )
    return value


def validate_g2c_seed_bundle(
    deployable: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    *,
    expected_seed: int,
    expected_view_order: Sequence[str] = G2C_VIEW_ORDER,
) -> None:
    """验证单 seed 的 model-input/privileged sidecar 边界与逐行 identity。"""

    if set(deployable) != set(_DEPLOYABLE_ARRAYS):
        raise ValueError("G2C deployable array keys 漂移")
    if set(labels) != set(_LABEL_ARRAYS):
        raise ValueError("G2C privileged label array keys 漂移")
    count = len(expected_view_order)
    _require_array(deployable, "sample_index", (count,), np.int64)
    _require_array(deployable, "seed", (count,), np.int64)
    view = deployable["viewpoint_id"]
    if view.shape != (count,) or view.dtype.kind != "U":
        raise ValueError("G2C viewpoint_id 必须是 unicode [V]")
    _require_array(deployable, "eligible_capture", (count,), np.bool_)
    _require_array(deployable, "rgb_external", (count, 128, 128, 3), np.uint8)
    _require_array(deployable, "physical_proprio", (count, 15), np.float32)
    _require_array(
        deployable,
        "structured_state",
        (count, OBSERVATION_V2_FRAME_STATE_DIM),
        np.float32,
    )
    _require_array(deployable, "geometric_motion", (count, 4), np.float32)
    for name in (
        "base_from_tcp",
        "base_from_external_camera_cv",
        "actual_world_from_external_camera_gl",
    ):
        _require_array(deployable, name, (count, 4, 4), np.float64)
    _require_array(deployable, "external_intrinsic_cv", (count, 3, 3), np.float64)
    for name in (
        "rgb_timestamp_s",
        "pose_timestamp_s",
        "arm_joint_drift_rad",
        "tcp_position_drift_m",
        "tcp_orientation_drift_rad",
        "camera_position_tracking_error_m",
        "camera_orientation_tracking_error_rad",
        "rotation_projection_error_frobenius",
    ):
        _require_array(deployable, name, (count,), np.float64)
    _require_array(deployable, "finger_force_n", (count, 2), np.float32)
    _require_array(deployable, "finger_force_valid", (count,), np.bool_)
    _require_array(
        deployable, "raw_gripper_opening_ratio", (count,), np.float32
    )

    _require_array(labels, "source_sample_index", (count,), np.int64)
    _require_array(labels, "seed", (count,), np.int64)
    label_view = labels["viewpoint_id"]
    if label_view.shape != (count,) or label_view.dtype.kind != "U":
        raise ValueError("G2C label viewpoint_id 必须是 unicode [V]")
    for name in ("object_position_base_m", "goal_position_base_m"):
        _require_array(labels, name, (count, 3), np.float32)
    for name in ("object_mask", "goal_mask"):
        _require_array(labels, name, (count, 128, 128), np.bool_)
    _require_array(labels, "normalized_uv", (count, 2, 2), np.float32)
    for name in (
        "keypoint_projection_valid",
        "keypoint_observable",
    ):
        _require_array(labels, name, (count, 2), np.bool_)
    for name in ("object_exists", "goal_exists", "is_grasped"):
        _require_array(labels, name, (count,), np.bool_)
    _require_array(
        labels, "robot_object_contact_force_n", (count,), np.float32
    )
    _require_array(labels, "geometry_roundtrip_error_m", (count, 2), np.float64)

    expected_index = np.arange(count, dtype=np.int64)
    if not np.array_equal(deployable["sample_index"], expected_index):
        raise ValueError("G2C sample_index 必须完整覆盖 0..V-1")
    expected_timestamp = np.full(
        count,
        G2C_STATIC_CAPTURE_SIMULATION_CONTROL_TIME_S,
        dtype=np.float64,
    )
    if not np.array_equal(
        deployable["rgb_timestamp_s"], expected_timestamp
    ) or not np.array_equal(deployable["pose_timestamp_s"], expected_timestamp):
        raise ValueError(
            "G2C static view 必须共享 post-warmup simulation-control-time"
        )
    if not np.array_equal(labels["source_sample_index"], expected_index):
        raise ValueError("G2C source_sample_index 与 deployable 不对齐")
    if not np.all(deployable["seed"] == expected_seed) or not np.array_equal(
        labels["seed"], deployable["seed"]
    ):
        raise ValueError("G2C seed identity 漂移")
    expected_views = np.asarray(tuple(expected_view_order), dtype=view.dtype)
    if not np.array_equal(view, expected_views) or not np.array_equal(
        label_view.astype(view.dtype), view
    ):
        raise ValueError("G2C capture order 或 sidecar view identity 漂移")
    if not bool(deployable["eligible_capture"].all()):
        raise ValueError("G2C seed bundle 只能包含 warmup 后 eligible capture")
    if not bool(deployable["finger_force_valid"].all()):
        raise ValueError("G2C finger force validity 不能缺失")
    if np.any(deployable["finger_force_n"] < 0.0):
        raise ValueError("G2C finger force 不能为负")
    if np.any(deployable["raw_gripper_opening_ratio"] < 0.0) or np.any(
        deployable["raw_gripper_opening_ratio"] > 1.0
    ):
        raise ValueError("G2C raw gripper ratio 必须位于 [0,1]")
    if not bool(labels["object_exists"].all() and labels["goal_exists"].all()):
        raise ValueError("G2C PickCube static scene 要求 object/goal 存在")
    if np.any(labels["keypoint_observable"] & ~labels["keypoint_projection_valid"]):
        raise ValueError("G2C observable keypoint 必须具有有效投影")
    valid_uv = labels["normalized_uv"][labels["keypoint_projection_valid"]]
    if np.any(valid_uv < 0.0) or np.any(valid_uv > 1.0):
        raise ValueError("G2C 有效 normalized_uv 必须位于 [0,1]")
    floating = [
        value
        for value in (*deployable.values(), *labels.values())
        if value.dtype.kind in {"f", "c"}
    ]
    if any(not np.isfinite(value).all() for value in floating):
        raise ValueError("G2C seed bundle 不能包含 NaN/Inf")
    for name in (
        "base_from_tcp",
        "base_from_external_camera_cv",
        "actual_world_from_external_camera_gl",
    ):
        for index, matrix in enumerate(deployable[name]):
            validate_se3(matrix, f"G2C {name}[{index}]")
    if not np.allclose(
        deployable["external_intrinsic_cv"][:, 2],
        np.asarray((0.0, 0.0, 1.0)),
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError("G2C intrinsic 最后一行漂移")


def audit_g2c_lifecycle(
    deployable: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """对全部 eligible row 运行 fail-whole invariant，不删行也不补 seed。"""

    seed = int(np.asarray(deployable["seed"])[0])
    validate_g2c_seed_bundle(deployable, labels, expected_seed=seed)
    lifecycle = config["capture"]["eligible_lifecycle_invariants"]
    skew = np.abs(
        deployable["rgb_timestamp_s"] - deployable["pose_timestamp_s"]
    )
    projected = labels["keypoint_projection_valid"]
    roundtrip_violation = (
        labels["geometry_roundtrip_error_m"]
        > float(config["capture"]["maximum_geometry_roundtrip_error_m"])
    ) & projected
    checks = {
        "object_position_finite": ~np.isfinite(
            labels["object_position_base_m"]
        ).all(axis=1),
        "object_center_plane": np.abs(
            labels["object_position_base_m"][:, 2]
            - float(lifecycle["object_center_base_z_m"])
        )
        > float(lifecycle["object_center_base_z_tolerance_m"]),
        "is_grasped": labels["is_grasped"],
        "finger_force_valid": ~deployable["finger_force_valid"],
        "left_finger_force": deployable["finger_force_n"][:, 0]
        > float(lifecycle["maximum_finger_force_n"]),
        "right_finger_force": deployable["finger_force_n"][:, 1]
        > float(lifecycle["maximum_finger_force_n"]),
        "raw_gripper_opening_ratio": deployable["raw_gripper_opening_ratio"]
        < float(lifecycle["minimum_raw_gripper_opening_ratio"]),
        "arm_hold": deployable["arm_joint_drift_rad"]
        > float(lifecycle["maximum_arm_joint_drift_rad"]),
        "tcp_position_hold": deployable["tcp_position_drift_m"]
        > float(lifecycle["maximum_tcp_position_drift_m"]),
        "tcp_orientation_hold": deployable["tcp_orientation_drift_rad"]
        > float(lifecycle["maximum_tcp_orientation_drift_rad"]),
        "robot_object_contact": labels["robot_object_contact_force_n"]
        > float(lifecycle["maximum_robot_object_contact_force_n"]),
        "rgb_pose_skew": skew
        > float(config["capture"]["maximum_rgb_pose_skew_s"]),
        "camera_position_tracking": deployable["camera_position_tracking_error_m"]
        > float(config["capture"]["maximum_camera_position_tracking_error_m"]),
        "camera_orientation_tracking": deployable[
            "camera_orientation_tracking_error_rad"
        ]
        > float(config["capture"]["maximum_camera_orientation_tracking_error_rad"]),
        "rotation_projection": deployable["rotation_projection_error_frobenius"]
        > float(config["capture"]["maximum_rotation_projection_error_frobenius"]),
        "geometry_roundtrip": np.any(roundtrip_violation, axis=1),
    }
    violations = {
        name: int(np.count_nonzero(mask)) for name, mask in checks.items()
    }
    examples = []
    for name, mask in checks.items():
        for index in np.flatnonzero(mask)[:4]:
            examples.append(
                {
                    "invariant": name,
                    "seed": int(deployable["seed"][index]),
                    "sample_index": int(deployable["sample_index"][index]),
                    "viewpoint_id": str(deployable["viewpoint_id"][index]),
                }
            )
    passed = not any(violations.values())
    result = {
        "version": E018_P1_G2C_DATA_RESULT_VERSION,
        "eligible_capture_count": len(deployable["seed"]),
        "passed": passed,
        "fail_whole_split_required": bool(lifecycle["fail_whole_split"]),
        "violation_counts": violations,
        "violation_examples": examples,
    }
    result["audit_sha256"] = canonical_sha256(result)
    return result


def _load_npz(path: Path, expected_names: Sequence[str]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"G2C NPZ 不存在: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != set(expected_names):
            raise ValueError(f"G2C NPZ keys 漂移: {path}")
        return {name: payload[name] for name in expected_names}


def _collect_one_seed(
    *,
    env: Any,
    base_env: Any,
    camera: Any,
    seed: int,
    viewpoint_by_id: Mapping[str, tuple[Any, Any]],
    config: Mapping[str, Any],
    spec: RobotSpec,
    proprio_normalizer: ProprioNormalizer,
    force_normalizer: FingerForceNormalizer,
    sapien_module: Any,
    sapien_utils_module: Any,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[dict[str, Any]],
    dict[str, int],
]:
    from robot_vla.precision.e018_p1_g0 import _step_hold_open
    from robot_vla.precision.e018_p1_viewpoint_screen import (
        _capture_sensor_observation,
        _set_static_camera_pose,
    )

    maximum_projection = float(
        config["capture"]["maximum_rotation_projection_error_frobenius"]
    )
    raw_observation, _ = env.reset(seed=seed)
    diagnostics = [
        _reset_diagnostic(
            seed=seed,
            observation=raw_observation,
            base_env=base_env,
            spec=spec,
            maximum_projection_error=maximum_projection,
            phase=G2C_RAW_RESET_DIAGNOSTIC_PHASE,
        )
    ]
    home, home_orientation = viewpoint_by_id[FRONT_HOME_ID]
    observation = raw_observation
    pose_set_count = 0
    safe_hold_steps = 0
    for _ in range(config["capture"]["reset_warmup_ticks"]):
        _set_static_camera_pose(
            camera,
            home,
            home_orientation,
            sapien_module=sapien_module,
            sapien_utils_module=sapien_utils_module,
        )
        pose_set_count += 1
        observation, terminated, truncated = _step_hold_open(
            env, env.action_space.shape
        )
        safe_hold_steps += 1
        if terminated or truncated:
            raise RuntimeError(f"G2C seed {seed} reset warmup 期间环境结束")
    diagnostics.append(
        _reset_diagnostic(
            seed=seed,
            observation=observation,
            base_env=base_env,
            spec=spec,
            maximum_projection_error=maximum_projection,
            phase=G2C_POST_WARMUP_DIAGNOSTIC_PHASE,
        )
    )
    qpos = _numpy(base_env.agent.robot.get_qpos())
    if qpos.shape != (1, 9):
        raise RuntimeError("G2C post-warmup qpos schema 漂移")
    anchor_arm_q = np.asarray(qpos[0, :7], dtype=np.float64).copy()
    _, anchor_base_from_tcp, _, _ = _physical_state(
        base_env=base_env,
        spec=spec,
        maximum_projection_error=maximum_projection,
    )
    deployable_rows: list[dict[str, Any]] = []
    privileged_rows: list[dict[str, Any]] = []
    for sample_index, viewpoint_id in enumerate(G2C_VIEW_ORDER):
        viewpoint, orientation = viewpoint_by_id[viewpoint_id]
        _, commanded_world_from_gl = _set_static_camera_pose(
            camera,
            viewpoint,
            orientation,
            sapien_module=sapien_module,
            sapien_utils_module=sapien_utils_module,
        )
        pose_set_count += 1
        observation = _capture_sensor_observation(base_env)
        capture = _capture_static_view(
            seed=seed,
            sample_index=sample_index,
            viewpoint_id=viewpoint_id,
            observation=observation,
            commanded_world_from_gl=commanded_world_from_gl,
            base_env=base_env,
            spec=spec,
            proprio_normalizer=proprio_normalizer,
            force_normalizer=force_normalizer,
            anchor_arm_q=anchor_arm_q,
            anchor_base_from_tcp=anchor_base_from_tcp,
            config=config,
        )
        deployable_rows.append(capture.deployable)
        privileged_rows.append(capture.privileged)
    deployable = _stack_rows(deployable_rows, _DEPLOYABLE_ARRAYS)
    privileged = _stack_rows(privileged_rows, _LABEL_ARRAYS)
    validate_g2c_seed_bundle(deployable, privileged, expected_seed=seed)
    return deployable, privileged, diagnostics, {
        "simulator_camera_pose_set_count": pose_set_count,
        "simulator_safe_hold_open_step_count": safe_hold_steps,
        "static_render_refresh_count": len(G2C_VIEW_ORDER),
    }


def _collect_static_data(
    *,
    config: Mapping[str, Any],
    parent_g0c: Mapping[str, Any],
    split_seeds: Mapping[str, Sequence[int]],
    spec: RobotSpec,
    proprio_normalizer: ProprioNormalizer,
    force_normalizer: FingerForceNormalizer,
) -> tuple[
    list[tuple[str, int, dict[str, np.ndarray], dict[str, np.ndarray]]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, int],
]:
    import gymnasium as gym
    import mani_skill
    import sapien
    import torch
    from mani_skill.utils import sapien_utils

    from robot_vla.sim import register_robot_vla_maniskill_envs

    if mani_skill.__version__ != config["software"]["expected_mani_skill_version"]:
        raise RuntimeError("G2C ManiSkill version 漂移")
    if sapien.__version__ != config["software"]["expected_sapien_version"]:
        raise RuntimeError("G2C SAPIEN version 漂移")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("G2C DATA 要求支持 BF16 的 CUDA GPU")
    environment = config["environment"]
    viewpoint_by_id = _viewpoint_map(dict(parent_g0c))
    if tuple(viewpoint_by_id) != G2C_VIEW_ORDER:
        raise RuntimeError("G2C parent G0C viewpoint expansion/order 漂移")
    register_robot_vla_maniskill_envs()
    env = gym.make(
        environment["environment_id"],
        obs_mode=environment["obs_mode"],
        control_mode=environment["control_mode"],
        num_envs=environment["num_envs"],
        robot_uids=environment["robot_uid"],
    )
    bundles: list[
        tuple[str, int, dict[str, np.ndarray], dict[str, np.ndarray]]
    ] = []
    diagnostics: list[dict[str, Any]] = []
    counts = {
        "simulator_camera_pose_set_count": 0,
        "simulator_safe_hold_open_step_count": 0,
        "static_render_refresh_count": 0,
    }
    try:
        base_env = env.unwrapped
        if base_env.control_freq != environment["control_hz"]:
            raise RuntimeError("G2C simulator control_hz 漂移")
        external_sensor = base_env._sensors.get(environment["external_camera_uid"])
        if external_sensor is None:
            raise RuntimeError("G2C external camera 不存在")
        camera = external_sensor.camera
        if external_sensor.entity is not None or not callable(
            getattr(camera, "set_local_pose", None)
        ):
            raise RuntimeError("G2C 要求 unmounted 可设位姿 RenderCamera")
        for split, seeds in split_seeds.items():
            for seed in seeds:
                deployable, privileged, rows, seed_counts = _collect_one_seed(
                    env=env,
                    base_env=base_env,
                    camera=camera,
                    seed=int(seed),
                    viewpoint_by_id=viewpoint_by_id,
                    config=config,
                    spec=spec,
                    proprio_normalizer=proprio_normalizer,
                    force_normalizer=force_normalizer,
                    sapien_module=sapien,
                    sapien_utils_module=sapien_utils,
                )
                bundles.append((split, int(seed), deployable, privileged))
                diagnostics.extend(rows)
                for name, value in seed_counts.items():
                    counts[name] += value
    finally:
        env.close()
    environment_identity = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(torch.device("cuda")),
        "mani_skill": mani_skill.__version__,
        "sapien": sapien.__version__,
        "external_camera_uid": environment["external_camera_uid"],
        "external_camera_unmounted": external_sensor.entity is None,
        "external_camera_class": (
            type(camera).__module__ + "." + type(camera).__name__
        ),
    }
    return bundles, diagnostics, environment_identity, counts


def _write_seed_bundles(
    *,
    output_root: Path,
    bundles: Sequence[
        tuple[str, int, Mapping[str, np.ndarray], Mapping[str, np.ndarray]]
    ],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    deployable_manifest: list[dict[str, Any]] = []
    label_manifest: list[dict[str, Any]] = []
    split_audits: dict[str, list[dict[str, Any]]] = {}
    for split, seed, deployable, labels in bundles:
        audit = audit_g2c_lifecycle(deployable, labels, config=config)
        split_audits.setdefault(split, []).append(audit)
        relative_deployable = Path("bundles") / split / f"seed-{seed:06d}.npz"
        relative_label = Path("bundles") / split / f"seed-{seed:06d}.npz"
        deployable_path = output_root / "deployable" / relative_deployable
        label_path = output_root / "privileged_labels" / relative_label
        _atomic_npz(deployable_path, deployable)
        _atomic_npz(label_path, labels)
        deployable_sha = file_sha256(deployable_path)
        label_sha = file_sha256(label_path)
        common = {
            "manifest_schema_version": G2C_MANIFEST_SCHEMA_VERSION,
            "split": split,
            "seed": seed,
            "sample_count": len(G2C_VIEW_ORDER),
            "view_order": list(G2C_VIEW_ORDER),
        }
        deployable_manifest.append(
            {
                **common,
                "schema_version": G2C_DEPLOYABLE_SCHEMA_VERSION,
                "file": relative_deployable.as_posix(),
                "sha256": deployable_sha,
                "contains_privileged_labels": False,
            }
        )
        label_manifest.append(
            {
                **common,
                "schema_version": G2C_LABEL_SCHEMA_VERSION,
                "file": relative_label.as_posix(),
                "sha256": label_sha,
                "source_deployable_file": relative_deployable.as_posix(),
                "source_deployable_sha256": deployable_sha,
                "contains_model_input_rgb": False,
            }
        )
    split_summaries: dict[str, Any] = {}
    for split, audits in sorted(split_audits.items()):
        violations = Counter()
        for audit in audits:
            violations.update(audit["violation_counts"])
        split_summaries[split] = {
            "seed_count": len(audits),
            "eligible_capture_count": sum(
                audit["eligible_capture_count"] for audit in audits
            ),
            "passed": bool(audits) and all(audit["passed"] for audit in audits),
            "fail_whole_split": True,
            "violation_counts": dict(sorted(violations.items())),
            "seed_audit_sha256": [audit["audit_sha256"] for audit in audits],
        }
    audit = {
        "version": E018_P1_G2C_DATA_RESULT_VERSION,
        "split_summaries": split_summaries,
        "all_collected_splits_passed": bool(split_summaries)
        and all(value["passed"] for value in split_summaries.values()),
        "row_deletion_count": 0,
        "seed_replacement_count": 0,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return deployable_manifest, label_manifest, audit


def _artifact_hashes(root: Path, names: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"G2C artifact 缺失: {name}")
        result[name] = file_sha256(path)
    return result


def _resolve_artifact_file(root: Path, relative: str) -> Path:
    base = root.resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base) or path.suffix != ".npz":
        raise ValueError("G2C manifest file 必须是 artifact root 内的 .npz")
    return path


def _validate_g2c_manifest_seed_identity(
    deployable_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """按 receipt mode 冻结 manifest split/seed、canonical 与 diagnostic 数量。"""

    mode = receipt.get("mode")
    if mode == "smoke":
        expected_pairs = tuple(
            (G2C_SMOKE_SPLIT, seed)
            for seed in config["sampling"]["smoke_only_seeds"]
        )
        expected_canonical = False
        expected_statuses = {
            "complete-engineering-smoke-pass",
            "complete-engineering-smoke-protocol-invalid",
        }
    elif mode == "full":
        formal = g2c_split_seeds(config)
        expected_pairs = tuple(
            (split, seed)
            for split in G2C_STATIC_SPLITS
            for seed in formal[split]
        )
        expected_canonical = True
        expected_statuses = {
            "complete-data-pass",
            "complete-data-protocol-invalid",
        }
    else:
        raise RuntimeError("G2C data receipt mode 漂移")

    def manifest_pairs(
        rows: Sequence[Mapping[str, Any]], name: str
    ) -> tuple[tuple[str, int], ...]:
        pairs = []
        for row in rows:
            split = row.get("split")
            seed = row.get("seed")
            if (
                not isinstance(split, str)
                or not isinstance(seed, int)
                or isinstance(seed, bool)
            ):
                raise TypeError(f"G2C {name} manifest split/seed 类型漂移")
            pairs.append((split, seed))
        return tuple(pairs)

    if (
        manifest_pairs(deployable_rows, "deployable") != expected_pairs
        or manifest_pairs(label_rows, "label") != expected_pairs
        or receipt.get("seed_count") != len(expected_pairs)
        or receipt.get("canonical_data_receipt") is not expected_canonical
        or receipt.get("status") not in expected_statuses
    ):
        raise RuntimeError(
            "G2C manifest 缺 seed、重复 seed、split、mode 或 canonical identity 漂移"
        )
    expected_seed_count = len(expected_pairs)
    return {
        "mode": mode,
        "canonical_data_receipt": expected_canonical,
        "formal_split_consumed": mode == "full",
        "seed_count": expected_seed_count,
        "expected_pairs": expected_pairs,
        "raw_reset_diagnostic_count": expected_seed_count,
        "post_warmup_diagnostic_count": expected_seed_count,
        "reset_diagnostic_count": expected_seed_count * 2,
    }


def verify_g2c_data_receipt(
    output_root: str | Path,
    *,
    config_path: str | Path | None = None,
    parent_g0c_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """重载所有小型 bundle/hash 并验证 receipt；不会访问外部 test 数据。"""

    root = Path(output_root)
    receipt_path = root / "data_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"G2C data receipt 不存在: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_receipt_sha = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if expected_receipt_sha != canonical_sha256(unsigned):
        raise RuntimeError("G2C data receipt internal SHA-256 漂移")
    if (
        receipt.get("version") != E018_P1_G2C_DATA_RESULT_VERSION
        or receipt.get("status")
        not in {
            "complete-engineering-smoke-pass",
            "complete-engineering-smoke-protocol-invalid",
            "complete-data-pass",
            "complete-data-protocol-invalid",
        }
        or receipt.get("test_trajectory_array_read_count") != 0
        or receipt.get("test_label_array_read_count") != 0
        or receipt.get("memory_read_count") != 0
        or receipt.get("memory_write_count") != 0
        or receipt.get("runtime_camera_actuation_count") != 0
        or receipt.get("physical_camera_actuation_count") != 0
        or receipt.get("arm_motion_command_count") != 0
        or receipt.get("gripper_close_command_count") != 0
        or receipt.get("manipulation_progression_count") != 0
        or receipt.get("checkpoint_write_count") != 0
    ):
        raise RuntimeError("G2C data receipt status/permission counter 漂移")
    artifacts = receipt.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("G2C data receipt artifact hash 集合缺失")
    for name, expected in artifacts.items():
        _require_sha256(expected, f"receipt artifact {name}")
        if file_sha256(root / name) != expected:
            raise RuntimeError(f"G2C artifact SHA-256 漂移: {name}")
    snapshot = json.loads((root / "config_snapshot.json").read_text(encoding="utf-8"))
    if canonical_sha256(snapshot) != receipt.get("config_sha256"):
        raise RuntimeError("G2C config snapshot canonical SHA-256 漂移")
    if config_path is not None:
        config = load_e018_p1_g2c_data_config(
            config_path, parent_g0c_config_path=parent_g0c_config_path
        )
        if config != snapshot:
            raise RuntimeError("G2C supplied config 与 snapshot 不一致")
    else:
        config = snapshot
    deployable_rows = _read_jsonl(
        root / "deployable" / "manifest.jsonl", "G2C deployable manifest"
    )
    label_rows = _read_jsonl(
        root / "privileged_labels" / "manifest.jsonl", "G2C label manifest"
    )
    if len(deployable_rows) != len(label_rows) or not deployable_rows:
        raise RuntimeError("G2C deployable/label manifest 数量不一致")
    expected_identity = _validate_g2c_manifest_seed_identity(
        deployable_rows,
        label_rows,
        config=config,
        receipt=receipt,
    )
    total_samples = 0
    split_counts: Counter[str] = Counter()
    lifecycle_passed = True
    manifest_seeds: list[int] = []
    for deployable_meta, label_meta in zip(
        deployable_rows, label_rows, strict=True
    ):
        seed = int(deployable_meta["seed"])
        manifest_seeds.append(seed)
        split = str(deployable_meta["split"])
        if (
            label_meta["seed"] != seed
            or label_meta["split"] != split
            or deployable_meta["schema_version"] != G2C_DEPLOYABLE_SCHEMA_VERSION
            or label_meta["schema_version"] != G2C_LABEL_SCHEMA_VERSION
            or deployable_meta["view_order"] != list(G2C_VIEW_ORDER)
            or label_meta["view_order"] != list(G2C_VIEW_ORDER)
            or deployable_meta["sample_count"] != len(G2C_VIEW_ORDER)
            or label_meta["sample_count"] != len(G2C_VIEW_ORDER)
            or deployable_meta["contains_privileged_labels"] is not False
            or label_meta["contains_model_input_rgb"] is not False
        ):
            raise RuntimeError("G2C manifest row identity/schema 漂移")
        deployable_path = _resolve_artifact_file(
            root / "deployable", str(deployable_meta["file"])
        )
        label_path = _resolve_artifact_file(
            root / "privileged_labels", str(label_meta["file"])
        )
        if (
            file_sha256(deployable_path) != deployable_meta["sha256"]
            or file_sha256(label_path) != label_meta["sha256"]
            or label_meta["source_deployable_sha256"]
            != deployable_meta["sha256"]
            or label_meta["source_deployable_file"] != deployable_meta["file"]
        ):
            raise RuntimeError("G2C bundle file/hash lineage 漂移")
        deployable = _load_npz(deployable_path, _DEPLOYABLE_ARRAYS)
        labels = _load_npz(label_path, _LABEL_ARRAYS)
        validate_g2c_seed_bundle(deployable, labels, expected_seed=seed)
        audit = audit_g2c_lifecycle(deployable, labels, config=config)
        lifecycle_passed = lifecycle_passed and bool(audit["passed"])
        total_samples += len(deployable["seed"])
        split_counts[split] += len(deployable["seed"])
    expected_status_pass = receipt["status"].endswith("-pass")
    if lifecycle_passed != expected_status_pass:
        raise RuntimeError("G2C receipt status 与重算 lifecycle gate 不一致")
    if total_samples != receipt.get("eligible_capture_count"):
        raise RuntimeError("G2C eligible capture count 漂移")
    diagnostics = _read_jsonl(
        root / "reset_diagnostic.jsonl", "G2C reset diagnostics"
    )
    diagnostic_counts = _validate_reset_diagnostic_rows(
        diagnostics,
        expected_seeds=manifest_seeds,
    )
    data_summary = json.loads(
        (root / "data_summary.json").read_text(encoding="utf-8")
    )
    timestamp_identity = {
        "static_capture_timestamp_source": (
            "post-warmup-simulation-control-time-shared-no-step-static/v1"
        ),
        "static_capture_simulation_control_time_s": (
            G2C_STATIC_CAPTURE_SIMULATION_CONTROL_TIME_S
        ),
        "static_capture_sequence_field": "sample_index",
        "static_views_share_timestamp_without_environment_step": True,
    }
    if any(
        receipt.get(name) != count or data_summary.get(name) != count
        for name, count in diagnostic_counts.items()
    ) or any(
        diagnostic_counts[name] != expected_identity[name]
        for name in (
            "raw_reset_diagnostic_count",
            "post_warmup_diagnostic_count",
            "reset_diagnostic_count",
        )
    ) or any(
        receipt.get(name) != value or data_summary.get(name) != value
        for name, value in timestamp_identity.items()
    ) or (
        data_summary.get("mode") != expected_identity["mode"]
        or data_summary.get("canonical_data_receipt")
        is not expected_identity["canonical_data_receipt"]
        or data_summary.get("formal_split_consumed")
        is not expected_identity["formal_split_consumed"]
        or data_summary.get("seed_count") != expected_identity["seed_count"]
    ):
        raise RuntimeError("G2C diagnostic/timestamp summary/receipt identity 漂移")
    result = {
        "version": E018_P1_G2C_DATA_RESULT_VERSION,
        "status": receipt["status"],
        "verified": True,
        "receipt_raw_sha256": file_sha256(receipt_path),
        "receipt_sha256": expected_receipt_sha,
        "config_sha256": receipt["config_sha256"],
        "data_identity_sha256": receipt["data_identity_sha256"],
        "seed_bundle_count": len(deployable_rows),
        "eligible_capture_count": total_samples,
        **diagnostic_counts,
        **timestamp_identity,
        "split_capture_counts": dict(sorted(split_counts.items())),
        "lifecycle_gate_passed": lifecycle_passed,
        "test_trajectory_array_read_count": 0,
        "test_label_array_read_count": 0,
    }
    result["verification_sha256"] = canonical_sha256(result)
    return result


def run_e018_p1_g2c_data(
    *,
    config_path: str | Path,
    parent_g0c_config_path: str | Path,
    parent_g0c_receipt_path: str | Path,
    e013_deployable_root: str | Path,
    e016_fresh_deployable_root: str | Path,
    stats_root: str | Path,
    inventory_path: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
    mode: str = "smoke",
    decision_exit_go: bool = False,
) -> dict[str, Any]:
    """采集四 seed smoke 或经新 R2 GO 授权的完整 static DATA。"""

    if mode not in {"smoke", "full"}:
        raise ValueError("G2C DATA mode 只能是 smoke/full")
    if mode == "full" and decision_exit_go is not True:
        raise PermissionError("G2C full DATA 必须有 full-data 前 R2 exit GO")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"G2C DATA output 已存在: {output}")
    config = load_e018_p1_g2c_data_config(
        config_path, parent_g0c_config_path=parent_g0c_config_path
    )
    parent_g0c = json.loads(Path(parent_g0c_config_path).read_text(encoding="utf-8"))
    parent_receipt = _verify_g0c_receipt(Path(parent_g0c_receipt_path), config)
    seed_audit = audit_g2c_seed_disjointness(
        config=config,
        e013_deployable_root=e013_deployable_root,
        e016_fresh_deployable_root=e016_fresh_deployable_root,
        inventory_path=inventory_path,
    )
    source_identity = _source_identity(Path(repository_root))
    if not source_identity["worktree_clean"]:
        raise RuntimeError("G2C DATA 执行要求 exact clean worktree")
    spec, proprio, force, normalizer_identity = _load_normalizers(
        stats_root=Path(stats_root), config=config
    )
    if mode == "smoke":
        split_seeds: dict[str, Sequence[int]] = {
            G2C_SMOKE_SPLIT: list(config["sampling"]["smoke_only_seeds"])
        }
        canonical_data_receipt = False
        formal_split_consumed = False
    else:
        formal = g2c_split_seeds(config)
        split_seeds = {name: formal[name] for name in G2C_STATIC_SPLITS}
        canonical_data_receipt = True
        formal_split_consumed = True
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    config_sha = canonical_sha256(config)
    _atomic_json(output / "config_snapshot.json", config)
    _atomic_json(output / "source_identity.json", source_identity)
    _atomic_json(output / "seed_audit.json", seed_audit)
    _atomic_json(
        output / "run_state.json",
        {
            "version": E018_P1_G2C_DATA_RESULT_VERSION,
            "status": "in-progress",
            "mode": mode,
            "config_sha256": config_sha,
            "test_trajectory_array_read_count": 0,
            "test_label_array_read_count": 0,
        },
    )
    bundles, diagnostics, environment_identity, simulator_counts = (
        _collect_static_data(
            config=config,
            parent_g0c=parent_g0c,
            split_seeds=split_seeds,
            spec=spec,
            proprio_normalizer=proprio,
            force_normalizer=force,
        )
    )
    diagnostic_counts = _validate_reset_diagnostic_rows(
        diagnostics,
        expected_seeds=[seed for _, seed, _, _ in bundles],
    )
    timestamp_identity = {
        "static_capture_timestamp_source": config["capture"]["timestamp_source"],
        "static_capture_simulation_control_time_s": (
            G2C_STATIC_CAPTURE_SIMULATION_CONTROL_TIME_S
        ),
        "static_capture_sequence_field": "sample_index",
        "static_views_share_timestamp_without_environment_step": True,
    }
    deployable_manifest, label_manifest, lifecycle_audit = _write_seed_bundles(
        output_root=output,
        bundles=bundles,
        config=config,
    )
    _atomic_jsonl(output / "deployable" / "manifest.jsonl", deployable_manifest)
    _atomic_jsonl(
        output / "privileged_labels" / "manifest.jsonl", label_manifest
    )
    _atomic_jsonl(output / "reset_diagnostic.jsonl", diagnostics)
    _atomic_json(output / "lifecycle_audit.json", lifecycle_audit)
    _atomic_json(output / "environment_identity.json", environment_identity)
    _atomic_json(output / "normalizer_identity.json", normalizer_identity)
    passed = bool(lifecycle_audit["all_collected_splits_passed"])
    status = (
        f"complete-{'engineering-smoke' if mode == 'smoke' else 'data'}-"
        f"{'pass' if passed else 'protocol-invalid'}"
    )
    capture_count = len(bundles) * len(G2C_VIEW_ORDER)
    expected = (
        int(config["engineering_smoke"]["expected_eligible_capture_count"])
        if mode == "smoke"
        else sum(len(values) for values in split_seeds.values())
        * len(G2C_VIEW_ORDER)
    )
    if capture_count != expected:
        raise RuntimeError(f"G2C capture count 漂移: {capture_count} != {expected}")
    permissions = {
        **simulator_counts,
        "simulator_safe_hold_open_action_count": simulator_counts[
            "simulator_safe_hold_open_step_count"
        ],
        "privileged_label_capture_count": capture_count,
        "runtime_camera_actuation_count": 0,
        "physical_camera_actuation_count": 0,
        "arm_motion_command_count": 0,
        "gripper_close_command_count": 0,
        "manipulation_progression_count": 0,
        "memory_read_count": 0,
        "memory_write_count": 0,
        "test_trajectory_array_read_count": 0,
        "test_label_array_read_count": 0,
        "checkpoint_write_count": 0,
    }
    manifest_hashes = {
        "deployable_manifest_sha256": file_sha256(
            output / "deployable" / "manifest.jsonl"
        ),
        "privileged_label_manifest_sha256": file_sha256(
            output / "privileged_labels" / "manifest.jsonl"
        ),
        "reset_diagnostic_sha256": file_sha256(output / "reset_diagnostic.jsonl"),
    }
    data_identity = canonical_sha256(
        {
            "version": E018_P1_G2C_DATA_RESULT_VERSION,
            "config_sha256": config_sha,
            "source_identity_sha256": source_identity["identity_sha256"],
            "mode": mode,
            "split_seeds": split_seeds,
            "view_order": G2C_VIEW_ORDER,
            **manifest_hashes,
        }
    )
    summary = {
        "version": E018_P1_G2C_DATA_RESULT_VERSION,
        "status": status,
        "mode": mode,
        "gate_passed": passed,
        "canonical_data_receipt": canonical_data_receipt,
        "formal_split_consumed": formal_split_consumed,
        "config_sha256": config_sha,
        "source_identity_sha256": source_identity["identity_sha256"],
        "data_identity_sha256": data_identity,
        "parent_g0c_receipt_sha256": parent_receipt["receipt_sha256"],
        "seed_audit_sha256": seed_audit["audit_sha256"],
        "seed_count": len(bundles),
        "view_count": len(G2C_VIEW_ORDER),
        "eligible_capture_count": capture_count,
        "expected_eligible_capture_count": expected,
        **diagnostic_counts,
        **timestamp_identity,
        "lifecycle_audit": lifecycle_audit,
        "permissions": permissions,
        "normalizer_identity": normalizer_identity,
        **manifest_hashes,
        "prediction_before_label_required_for": config["dataset"][
            "prediction_before_label_required_for"
        ],
        "prediction_before_label_evaluated_in_data_phase": False,
        "checkpoint_persisted": False,
        "test_split": None,
    }
    _atomic_json(output / "data_summary.json", summary)
    _atomic_json(
        output / "run_state.json",
        {
            "version": E018_P1_G2C_DATA_RESULT_VERSION,
            "status": status,
            "mode": mode,
            "complete": True,
            "config_sha256": config_sha,
            **permissions,
        },
    )
    artifact_names = (
        "config_snapshot.json",
        "source_identity.json",
        "seed_audit.json",
        "run_state.json",
        "reset_diagnostic.jsonl",
        "lifecycle_audit.json",
        "environment_identity.json",
        "normalizer_identity.json",
        "data_summary.json",
        "deployable/manifest.jsonl",
        "privileged_labels/manifest.jsonl",
    )
    receipt = {
        "version": E018_P1_G2C_DATA_RESULT_VERSION,
        "status": status,
        "mode": mode,
        "gate_passed": passed,
        "canonical_data_receipt": canonical_data_receipt,
        "config_sha256": config_sha,
        "source_identity_sha256": source_identity["identity_sha256"],
        "data_identity_sha256": data_identity,
        "seed_count": len(bundles),
        "eligible_capture_count": capture_count,
        **diagnostic_counts,
        **timestamp_identity,
        "artifact_sha256": _artifact_hashes(output, artifact_names),
        **permissions,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_json(output / "data_receipt.json", receipt)
    verified = verify_g2c_data_receipt(
        output,
        config_path=config_path,
        parent_g0c_config_path=parent_g0c_config_path,
    )
    return {**summary, "receipt": receipt, "verification": verified}


@dataclass(frozen=True)
class G2CBundleMeta:
    split: str
    seed: int
    file: str
    sha256: str
    sample_count: int
    view_order: tuple[str, ...]
    schema_version: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, privileged: bool) -> G2CBundleMeta:
        required = {
            "manifest_schema_version",
            "split",
            "seed",
            "sample_count",
            "view_order",
            "schema_version",
            "file",
            "sha256",
        }
        required.add("contains_model_input_rgb" if privileged else "contains_privileged_labels")
        if privileged:
            required.update({"source_deployable_file", "source_deployable_sha256"})
        if set(value) != required:
            raise ValueError("G2C manifest metadata keys 漂移")
        expected_schema = (
            G2C_LABEL_SCHEMA_VERSION if privileged else G2C_DEPLOYABLE_SCHEMA_VERSION
        )
        if (
            value["manifest_schema_version"] != G2C_MANIFEST_SCHEMA_VERSION
            or value["schema_version"] != expected_schema
            or value["sample_count"] != len(G2C_VIEW_ORDER)
            or tuple(value["view_order"]) != G2C_VIEW_ORDER
            or (privileged and value["contains_model_input_rgb"] is not False)
            or (not privileged and value["contains_privileged_labels"] is not False)
        ):
            raise ValueError("G2C manifest metadata schema/role 漂移")
        _require_sha256(value["sha256"], "G2C bundle sha256")
        return cls(
            split=str(value["split"]),
            seed=int(value["seed"]),
            file=str(value["file"]),
            sha256=str(value["sha256"]),
            sample_count=int(value["sample_count"]),
            view_order=tuple(str(item) for item in value["view_order"]),
            schema_version=str(value["schema_version"]),
        )


class G2CDeployableDataset:
    """只加载 front RGB/状态/pose；不解析任何 privileged sidecar。"""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        *,
        seeds: Sequence[int] | None = None,
    ) -> None:
        if split not in {*G2C_STATIC_SPLITS, G2C_SMOKE_SPLIT}:
            raise ValueError("G2C deployable Dataset split 不允许 test/qualification")
        self.root = Path(data_root)
        self.deployable_root = self.root / "deployable"
        rows = _read_jsonl(
            self.deployable_root / "manifest.jsonl", "G2C deployable manifest"
        )
        seed_filter = None if seeds is None else set(seeds)
        if seed_filter is not None and (
            not seed_filter
            or len(seed_filter) != len(seeds)
            or any(
                not isinstance(seed, int) or isinstance(seed, bool) or seed <= 0
                for seed in seeds
            )
        ):
            raise ValueError("G2C deployable seed filter 必须是唯一正整数")
        self.entries = [
            G2CBundleMeta.from_dict(row, privileged=False)
            for row in rows
            if row.get("split") == split
            and (seed_filter is None or int(row.get("seed", -1)) in seed_filter)
        ]
        if not self.entries:
            raise ValueError(f"G2C deployable split 为空: {split}")
        self.split = split
        if seed_filter is not None and {entry.seed for entry in self.entries} != seed_filter:
            raise ValueError("G2C deployable seed filter 未完整命中 manifest")
        self.index = [
            (entry_index, sample_index)
            for entry_index, entry in enumerate(self.entries)
            for sample_index in range(entry.sample_count)
        ]
        self._cache_index: int | None = None
        self._cache: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.index)

    def _bundle(self, entry_index: int) -> dict[str, np.ndarray]:
        if self._cache_index == entry_index and self._cache is not None:
            return self._cache
        meta = self.entries[entry_index]
        path = _resolve_artifact_file(self.deployable_root, meta.file)
        if file_sha256(path) != meta.sha256:
            raise RuntimeError("G2C deployable Dataset bundle SHA-256 漂移")
        arrays = _load_npz(path, _DEPLOYABLE_ARRAYS)
        self._cache_index = entry_index
        self._cache = arrays
        return arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry_index, sample_index = self.index[index]
        meta = self.entries[entry_index]
        arrays = self._bundle(entry_index)
        if int(arrays["seed"][sample_index]) != meta.seed:
            raise RuntimeError("G2C deployable Dataset seed lineage 漂移")
        return {
            "model_inputs": {
                "rgb_external": arrays["rgb_external"][sample_index].copy(),
                "structured_state": arrays["structured_state"][sample_index].copy(),
                "geometric_motion": arrays["geometric_motion"][sample_index].copy(),
            },
            "capture": {
                "seed": meta.seed,
                "split": meta.split,
                "sample_index": int(arrays["sample_index"][sample_index]),
                "viewpoint_id": str(arrays["viewpoint_id"][sample_index]),
                "base_from_external_camera_cv": arrays[
                    "base_from_external_camera_cv"
                ][sample_index].copy(),
                "external_intrinsic_cv": arrays["external_intrinsic_cv"][
                    sample_index
                ].copy(),
                "rgb_timestamp_s": float(arrays["rgb_timestamp_s"][sample_index]),
                "pose_timestamp_s": float(arrays["pose_timestamp_s"][sample_index]),
                "input_sha256": canonical_sha256(
                    {
                        "seed": meta.seed,
                        "viewpoint_id": str(arrays["viewpoint_id"][sample_index]),
                        "rgb_sha256": hashlib.sha256(
                            np.ascontiguousarray(
                                arrays["rgb_external"][sample_index]
                            ).tobytes()
                        ).hexdigest(),
                        "structured_state_sha256": hashlib.sha256(
                            np.ascontiguousarray(
                                arrays["structured_state"][sample_index]
                            ).tobytes()
                        ).hexdigest(),
                        "base_from_camera_sha256": hashlib.sha256(
                            np.ascontiguousarray(
                                arrays["base_from_external_camera_cv"][sample_index]
                            ).tobytes()
                        ).hexdigest(),
                    }
                ),
            },
        }


class G2CFrontTrainingDataset:
    """仅供 train/smoke 拟合；model-val/calibration 必须走 prediction freeze。"""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        *,
        seeds: Sequence[int] | None = None,
    ) -> None:
        if split not in {"train", G2C_SMOKE_SPLIT}:
            raise ValueError("G2C training Dataset 禁止读取 model-val/calibration/test")
        self.deployable = G2CDeployableDataset(data_root, split, seeds=seeds)
        self.label_root = Path(data_root) / "privileged_labels"
        label_rows = _read_jsonl(
            self.label_root / "manifest.jsonl", "G2C privileged label manifest"
        )
        seed_filter = None if seeds is None else set(seeds)
        entries = [
            G2CBundleMeta.from_dict(row, privileged=True)
            for row in label_rows
            if row.get("split") == split
            and (seed_filter is None or int(row.get("seed", -1)) in seed_filter)
        ]
        if len(entries) != len(self.deployable.entries):
            raise ValueError("G2C training source/label bundle 数量不一致")
        self.label_entries = entries
        for deployable, label in zip(
            self.deployable.entries, self.label_entries, strict=True
        ):
            if (deployable.split, deployable.seed, deployable.view_order) != (
                label.split,
                label.seed,
                label.view_order,
            ):
                raise ValueError("G2C training source/label manifest identity 不一致")
        self._label_cache_index: int | None = None
        self._label_cache: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.deployable)

    def _labels(self, entry_index: int) -> dict[str, np.ndarray]:
        if self._label_cache_index == entry_index and self._label_cache is not None:
            return self._label_cache
        meta = self.label_entries[entry_index]
        path = _resolve_artifact_file(self.label_root, meta.file)
        if file_sha256(path) != meta.sha256:
            raise RuntimeError("G2C training label bundle SHA-256 漂移")
        arrays = _load_npz(path, _LABEL_ARRAYS)
        self._label_cache_index = entry_index
        self._label_cache = arrays
        return arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.deployable[index]
        entry_index, sample_index = self.deployable.index[index]
        labels = self._labels(entry_index)
        observable = labels["keypoint_observable"][sample_index].copy()
        normalized_uv = labels["normalized_uv"][sample_index].copy()
        normalized_uv[~observable] = 0.0
        return {
            "model_inputs": source["model_inputs"],
            "supervision": {
                "mask_targets": np.stack(
                    (
                        labels["object_mask"][sample_index],
                        labels["goal_mask"][sample_index],
                    )
                ).astype(np.float32),
                "normalized_uv_targets": normalized_uv,
                "keypoint_valid": observable,
                "keypoint_observable": observable.copy(),
                "motion_residual_targets": np.zeros(4, dtype=np.float32),
                "motion_valid": np.zeros(4, dtype=np.bool_),
                "projection_valid": np.asarray(
                    bool(labels["keypoint_projection_valid"][sample_index].all()),
                    dtype=np.bool_,
                ),
            },
            "audit": {
                **source["capture"],
                "object_position_base_m": labels["object_position_base_m"][
                    sample_index
                ].copy(),
                "goal_position_base_m": labels["goal_position_base_m"][
                    sample_index
                ].copy(),
                "keypoint_projection_valid": labels[
                    "keypoint_projection_valid"
                ][sample_index].copy(),
                "keypoint_observable": observable.copy(),
            },
        }


__all__ = [
    "E018_P1_G2C_DATA_CONFIG_VERSION",
    "E018_P1_G2C_DATA_GATE",
    "E018_P1_G2C_DATA_RESULT_VERSION",
    "G2C_ALL_SPLITS",
    "G2C_DEPLOYABLE_SCHEMA_VERSION",
    "G2C_LABEL_SCHEMA_VERSION",
    "G2C_MANIFEST_SCHEMA_VERSION",
    "G2C_SMOKE_SPLIT",
    "G2C_STATIC_SPLITS",
    "G2C_VIEW_ORDER",
    "G2CBundleMeta",
    "G2CDeployableDataset",
    "G2CFrontTrainingDataset",
    "audit_g2c_lifecycle",
    "audit_g2c_seed_disjointness",
    "g2c_split_seeds",
    "load_e018_p1_g2c_data_config",
    "run_e018_p1_g2c_data",
    "validate_g2c_seed_bundle",
    "verify_g2c_data_receipt",
]
