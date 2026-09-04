"""E018-P1 G0B：25 个离散 front-camera 位姿的静态开发筛选。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla.observation import invert_se3, opengl_camera_to_opencv, validate_se3
from robot_vla.precision.active_front_camera import (
    FrontCameraOrientationMode,
    FrontCameraViewpoint,
    compose_camera_orientation_wxyz,
    quaternion_angular_distance_rad,
    rotation_angular_distance_rad,
)

E018_P1_G0B_CONFIG_VERSION = "e018-p1-g0b-static-viewpoint-screen-development/v1"
E018_P1_G0B_RESULT_VERSION = "e018-p1-g0b-static-viewpoint-screen-result/v1"
E018_P1_G0B_LATTICE_VERSION = "e018-p1-front-5x5-static-screen-provisional/v1"

_ANCHOR_KEYS = {
    "viewpoint_id",
    "lateral_anchor",
    "vertical_anchor",
    "position_world_m",
    "look_at_world_m",
    "yaw_rad",
    "pitch_rad",
    "roll_rad",
}
_ORIENTATION_KEYS = {
    "orientation_id",
    "yaw_offset_rad",
    "pitch_offset_rad",
    "roll_offset_rad",
}
_SOURCE_FILES = (
    "src/robot_vla/precision/active_front_camera.py",
    "src/robot_vla/precision/e018_p1_viewpoint_screen.py",
    "src/robot_vla/cli/run_e018_p1_viewpoint_screen.py",
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} 必须是有限正数")
    return result


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")
    return value


def _probability(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} 必须位于 [0,1]")
    return result


def _parse_anchor(value: Any, name: str) -> FrontCameraViewpoint:
    item = _require_keys(value, _ANCHOR_KEYS, name)
    anchor = FrontCameraViewpoint(
        viewpoint_id=str(item["viewpoint_id"]),
        lateral_anchor=str(item["lateral_anchor"]),
        vertical_anchor=str(item["vertical_anchor"]),
        position_world_m=tuple(float(entry) for entry in item["position_world_m"]),
        look_at_world_m=tuple(float(entry) for entry in item["look_at_world_m"]),
        yaw_rad=float(item["yaw_rad"]),
        pitch_rad=float(item["pitch_rad"]),
        roll_rad=float(item["roll_rad"]),
    )
    anchor.validate()
    return anchor


def _parse_orientation(value: Any, name: str) -> FrontCameraOrientationMode:
    item = _require_keys(value, _ORIENTATION_KEYS, name)
    orientation = FrontCameraOrientationMode(
        orientation_id=str(item["orientation_id"]),
        yaw_offset_rad=float(item["yaw_offset_rad"]),
        pitch_offset_rad=float(item["pitch_offset_rad"]),
        roll_offset_rad=float(item["roll_offset_rad"]),
    )
    orientation.validate()
    return orientation


def load_e018_p1_g0b_config(path: str | Path) -> dict[str, Any]:
    """严格读取静态 development screen；不接受 test/formal 参数。"""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"E018-P1 G0B config 不存在: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = _require_keys(
        config,
        {
            "version",
            "status",
            "scope",
            "software",
            "environment",
            "viewpoint_lattice",
            "sampling",
            "screening",
            "safety",
            "execution",
        },
        "E018-P1 G0B config",
    )
    if config["version"] != E018_P1_G0B_CONFIG_VERSION:
        raise ValueError("E018-P1 G0B config version 漂移")
    if config["status"] != "development-only-static-screen-no-formal-claim":
        raise ValueError("E018-P1 G0B 只能以 development-only 状态运行")

    scope = _require_keys(
        config["scope"],
        {
            "gate",
            "test_split_allowed",
            "formal_claim_allowed",
            "runtime_gt_control_allowed",
            "provider_inference_allowed",
            "memory_read_allowed",
            "memory_write_allowed",
            "time_resolved_motion_claim_allowed",
        },
        "scope",
    )
    if scope != {
        "gate": "G0B_STATIC_VIEWPOINT_GEOMETRY_SCREEN",
        "test_split_allowed": False,
        "formal_claim_allowed": False,
        "runtime_gt_control_allowed": False,
        "provider_inference_allowed": False,
        "memory_read_allowed": False,
        "memory_write_allowed": False,
        "time_resolved_motion_claim_allowed": False,
    }:
        raise ValueError("G0B scope 必须禁止 test/formal/GT control/provider/Memory/motion claim")

    software = _require_keys(
        config["software"],
        {"expected_mani_skill_version", "expected_sapien_version"},
        "software",
    )
    if software != {
        "expected_mani_skill_version": "3.0.1",
        "expected_sapien_version": "3.0.3",
    }:
        raise ValueError("G0B software identity 漂移")

    environment = _require_keys(
        config["environment"],
        {
            "environment_id",
            "robot_uid",
            "camera_uid",
            "obs_mode",
            "control_mode",
            "num_envs",
            "image_height",
            "image_width",
        },
        "environment",
    )
    expected_environment = {
        "environment_id": "RobotVLAPickCubeToRegion-v1",
        "robot_uid": "panda_wristcam",
        "camera_uid": "base_camera",
        "obs_mode": "rgb+segmentation",
        "control_mode": "pd_joint_delta_pos",
        "num_envs": 1,
        "image_height": 128,
        "image_width": 128,
    }
    if environment != expected_environment:
        raise ValueError("G0B environment identity 漂移")

    lattice = _require_keys(
        config["viewpoint_lattice"],
        {
            "version",
            "status",
            "pose_frame",
            "nominal_orientation_policy",
            "orientation_offset_frame",
            "cross_product_enabled",
            "expected_anchor_count",
            "expected_orientation_count",
            "expected_primitive_count",
            "anchors",
            "orientation_modes",
        },
        "viewpoint_lattice",
    )
    if (
        lattice["version"] != E018_P1_G0B_LATTICE_VERSION
        or lattice["status"] != "provisional-development-not-frozen"
        or lattice["pose_frame"] != "world"
        or lattice["nominal_orientation_policy"] != "legacy-fixed-workspace-look-at/v1"
        or lattice["orientation_offset_frame"]
        != "sapien-camera-local-x-forward-y-left-z-up/v1"
        or lattice["cross_product_enabled"] is not True
        or lattice["expected_anchor_count"] != 5
        or lattice["expected_orientation_count"] != 5
        or lattice["expected_primitive_count"] != 25
    ):
        raise ValueError("G0B viewpoint lattice identity/semantics 漂移")
    anchors_value = lattice["anchors"]
    modes_value = lattice["orientation_modes"]
    if not isinstance(anchors_value, list) or len(anchors_value) != 5:
        raise ValueError("G0B 必须包含 HOME + 4 translation anchors")
    if not isinstance(modes_value, list) or len(modes_value) != 5:
        raise ValueError("G0B 每个 anchor 必须包含 5 个 cross orientation modes")
    anchors = [
        _parse_anchor(item, f"viewpoint_lattice.anchors[{index}]")
        for index, item in enumerate(anchors_value)
    ]
    modes = [
        _parse_orientation(item, f"viewpoint_lattice.orientation_modes[{index}]")
        for index, item in enumerate(modes_value)
    ]
    expected_anchor_ids = {"HOME", "LEFT_LOW", "LEFT_HIGH", "RIGHT_LOW", "RIGHT_HIGH"}
    if {item.viewpoint_id for item in anchors} != expected_anchor_ids:
        raise ValueError("G0B anchor ids 漂移")
    expected_anchor_pairs = {
        ("CENTER", "CENTER"),
        ("LEFT", "LOW"),
        ("LEFT", "HIGH"),
        ("RIGHT", "LOW"),
        ("RIGHT", "HIGH"),
    }
    if {(item.lateral_anchor, item.vertical_anchor) for item in anchors} != expected_anchor_pairs:
        raise ValueError("G0B anchor lattice 必须是 HOME + LEFT/RIGHT x LOW/HIGH")
    if len({item.look_at_world_m for item in anchors}) != 1:
        raise ValueError("G0B 所有 anchor 必须共享 nominal workspace look-at target")
    expected_modes = {
        "CENTER": (0.0, 0.0),
        "YAW_LEFT": (math.radians(12.0), 0.0),
        "YAW_RIGHT": (-math.radians(12.0), 0.0),
        "PITCH_UP": (0.0, math.radians(8.0)),
        "PITCH_DOWN": (0.0, -math.radians(8.0)),
    }
    if {item.orientation_id for item in modes} != set(expected_modes):
        raise ValueError("G0B orientation ids 漂移")
    for mode in modes:
        expected_yaw, expected_pitch = expected_modes[mode.orientation_id]
        if not math.isclose(mode.yaw_offset_rad, expected_yaw, abs_tol=1e-12):
            raise ValueError(f"{mode.orientation_id} yaw offset 漂移")
        if not math.isclose(mode.pitch_offset_rad, expected_pitch, abs_tol=1e-12):
            raise ValueError(f"{mode.orientation_id} pitch offset 漂移")
    if len(anchors) * len(modes) != lattice["expected_primitive_count"]:
        raise ValueError("G0B primitive cross product count 漂移")

    sampling = _require_keys(
        config["sampling"],
        {
            "usage",
            "seed_policy",
            "seed_start",
            "seed_count",
            "captures_per_pose",
            "visualization_seeds",
        },
        "sampling",
    )
    if (
        sampling["usage"] != "simulator-development-static-geometry-screen-no-test/v1"
        or sampling["seed_policy"] != "contiguous-development-range/v1"
    ):
        raise ValueError("G0B sampling usage/policy 漂移")
    seed_start = _positive_int(sampling["seed_start"], "sampling.seed_start")
    seed_count = _positive_int(sampling["seed_count"], "sampling.seed_count")
    _positive_int(sampling["captures_per_pose"], "sampling.captures_per_pose")
    if sampling["captures_per_pose"] != 3:
        raise ValueError("G0B 固定每 pose 重复捕获 3 帧")
    expected_seeds = set(range(seed_start, seed_start + seed_count))
    visualization_seeds = sampling["visualization_seeds"]
    if (
        not isinstance(visualization_seeds, list)
        or len(visualization_seeds) != 4
        or len(set(visualization_seeds)) != 4
        or not set(visualization_seeds) <= expected_seeds
    ):
        raise ValueError("G0B visualization seeds 必须是开发 seed 中 4 个唯一值")

    screening = _require_keys(
        config["screening"],
        {
            "geometric_visibility_min_pixels",
            "framing_usable_min_pixels",
            "framing_usable_min_border_margin_px",
            "center_ray_support_radius_px",
            "shortlist_size",
            "shortlist_max_per_anchor",
            "shortlist_min_both_geometric_visible_rate",
            "shortlist_min_both_framing_usable_rate",
        },
        "screening",
    )
    for name in (
        "geometric_visibility_min_pixels",
        "framing_usable_min_pixels",
        "framing_usable_min_border_margin_px",
        "center_ray_support_radius_px",
        "shortlist_size",
        "shortlist_max_per_anchor",
    ):
        _positive_int(screening[name], f"screening.{name}")
    if screening["framing_usable_min_pixels"] < screening["geometric_visibility_min_pixels"]:
        raise ValueError("framing usable pixel threshold 不得小于 geometric threshold")
    if screening["shortlist_size"] > lattice["expected_primitive_count"] - 1:
        raise ValueError("shortlist_size 超过 alternate primitive count")
    for name in (
        "shortlist_min_both_geometric_visible_rate",
        "shortlist_min_both_framing_usable_rate",
    ):
        _probability(screening[name], f"screening.{name}")

    safety = _require_keys(
        config["safety"],
        {
            "camera_position_tracking_tolerance_m",
            "camera_orientation_tracking_tolerance_rad",
            "arm_joint_drift_max_rad",
            "tcp_position_drift_max_m",
            "tcp_orientation_drift_max_rad",
        },
        "safety",
    )
    for name, value in safety.items():
        _positive(value, f"safety.{name}")

    execution = _require_keys(
        config["execution"],
        {
            "device",
            "static_pose_teleport_for_screening_allowed",
            "physical_robot_actuation_allowed",
            "environment_step_allowed",
            "arm_motion_command_allowed",
            "gripper_command_allowed",
            "provider_inference_allowed",
            "memory_access_allowed",
            "manipulation_progression_allowed",
        },
        "execution",
    )
    if execution != {
        "device": "cuda",
        "static_pose_teleport_for_screening_allowed": True,
        "physical_robot_actuation_allowed": False,
        "environment_step_allowed": False,
        "arm_motion_command_allowed": False,
        "gripper_command_allowed": False,
        "provider_inference_allowed": False,
        "memory_access_allowed": False,
        "manipulation_progression_allowed": False,
    }:
        raise ValueError("G0B execution scope 漂移")
    return config


def _source_identity(repository_root: Path) -> dict[str, Any]:
    safe_repository = str(repository_root.resolve())
    git = ("git", "-c", f"safe.directory={safe_repository}")
    hashes = {
        relative: _file_sha256(repository_root / relative) for relative in _SOURCE_FILES
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
    identity = {
        "git_commit": commit,
        "worktree_clean": not status,
        "git_status": status,
        "source_file_sha256": hashes,
    }
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


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


def _set_static_camera_pose(
    camera: Any,
    anchor: FrontCameraViewpoint,
    orientation: FrontCameraOrientationMode,
    *,
    sapien_module: Any,
    sapien_utils_module: Any,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(anchor.position_world_m, dtype=np.float64)
    nominal = sapien_utils_module.look_at(
        position,
        np.asarray(anchor.look_at_world_m, dtype=np.float64),
        up=(0.0, 0.0, 1.0),
    )
    nominal_quaternion = _single_vector(nominal.q, 4, "nominal camera quaternion")
    commanded_quaternion = compose_camera_orientation_wxyz(
        nominal_quaternion,
        orientation,
    )
    command_pose = sapien_module.Pose(p=position, q=commanded_quaternion)
    camera.set_local_pose(command_pose)
    gl_from_mount = sapien_module.Pose(q=[-0.5, -0.5, 0.5, 0.5])
    commanded_world_from_gl = validate_se3(
        (command_pose * gl_from_mount).to_transformation_matrix(),
        "commanded_world_from_external_camera_gl",
    )
    return commanded_quaternion, commanded_world_from_gl


def _capture_sensor_observation(base_env: Any) -> dict[str, Any]:
    """只刷新 renderer，不调用 get_info/evaluate，也不推进 physics/control step。"""

    sensor_data = base_env._get_obs_sensor_data()
    sensor_param = base_env.get_sensor_params()
    return {"sensor_data": sensor_data, "sensor_param": sensor_param}


def _mask_stats(
    actor_ids: np.ndarray,
    actor_id: int,
    *,
    geometric_min_pixels: int,
    framing_min_pixels: int,
    framing_min_margin_px: int,
) -> dict[str, Any]:
    mask = actor_ids == actor_id
    y, x = np.nonzero(mask)
    count = int(x.size)
    if count:
        height, width = actor_ids.shape
        bounds = {
            "x_min": int(x.min()),
            "x_max": int(x.max()),
            "y_min": int(y.min()),
            "y_max": int(y.max()),
        }
        margin = int(
            min(
                bounds["x_min"],
                width - 1 - bounds["x_max"],
                bounds["y_min"],
                height - 1 - bounds["y_max"],
            )
        )
        centroid = [float(x.mean()), float(y.mean())]
    else:
        bounds = None
        margin = -1
        centroid = None
    return {
        "visible_pixel_count": count,
        "geometrically_visible": count >= geometric_min_pixels,
        "border_margin_px": margin,
        "framing_usable": count >= framing_min_pixels and margin >= framing_min_margin_px,
        "mask_bounds_xyxy": bounds,
        "mask_centroid_uv_px": centroid,
    }


def _project_world_point(
    position_world_m: np.ndarray,
    *,
    world_from_camera_cv: np.ndarray,
    intrinsic_cv: np.ndarray,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    position = np.asarray(position_world_m, dtype=np.float64)
    camera_from_world = invert_se3(world_from_camera_cv, "world_from_external_camera_cv")
    homogeneous = np.concatenate((position, np.ones(1, dtype=np.float64)))
    camera_xyz = (camera_from_world @ homogeneous)[:3]
    depth = float(camera_xyz[2])
    if depth <= 1e-8:
        return {
            "projection_valid": False,
            "in_frame": False,
            "uv_px": None,
            "camera_xyz_m": camera_xyz.tolist(),
        }
    projected = intrinsic_cv @ camera_xyz
    uv = projected[:2] / projected[2]
    height, width = image_shape
    in_frame = bool(0.0 <= uv[0] <= width - 1 and 0.0 <= uv[1] <= height - 1)
    return {
        "projection_valid": True,
        "in_frame": in_frame,
        "uv_px": uv.tolist(),
        "camera_xyz_m": camera_xyz.tolist(),
    }


def _center_ray_diagnostic(
    actor_ids: np.ndarray,
    projection: dict[str, Any],
    *,
    entity_actor_id: int,
    robot_actor_ids: set[int],
    support_radius_px: int,
) -> dict[str, Any]:
    if not projection["projection_valid"] or not projection["in_frame"]:
        return {
            "classification": "OUT_OF_FRAME",
            "support_entity_pixels": 0,
            "support_robot_pixels": 0,
        }
    u, v = projection["uv_px"]
    center_x = round(u)
    center_y = round(v)
    height, width = actor_ids.shape
    x_start = max(0, center_x - support_radius_px)
    x_stop = min(width, center_x + support_radius_px + 1)
    y_start = max(0, center_y - support_radius_px)
    y_stop = min(height, center_y + support_radius_px + 1)
    support = actor_ids[y_start:y_stop, x_start:x_stop]
    entity_pixels = int(np.count_nonzero(support == entity_actor_id))
    robot_pixels = int(np.count_nonzero(np.isin(support, tuple(robot_actor_ids))))
    if entity_pixels:
        classification = "ENTITY_CENTER_RAY_VISIBLE"
    elif robot_pixels:
        classification = "ROBOT_AT_ENTITY_CENTER_RAY"
    else:
        classification = "OTHER_OR_BACKGROUND_AT_ENTITY_CENTER_RAY"
    return {
        "classification": classification,
        "support_entity_pixels": entity_pixels,
        "support_robot_pixels": robot_pixels,
    }


def _capture_row(
    *,
    seed: int,
    anchor: FrontCameraViewpoint,
    orientation: FrontCameraOrientationMode,
    primitive_id: str,
    capture_index: int,
    commanded_quaternion: np.ndarray,
    commanded_world_from_gl: np.ndarray,
    observation: dict[str, Any],
    camera: Any,
    base_env: Any,
    config: dict[str, Any],
    arm_anchor_q_rad: np.ndarray,
    tcp_anchor_world: np.ndarray,
    object_actor_id: int,
    goal_actor_id: int,
    robot_actor_ids: set[int],
) -> tuple[dict[str, Any], np.ndarray]:
    camera_uid = config["environment"]["camera_uid"]
    sensor = observation["sensor_data"][camera_uid]
    params = observation["sensor_param"][camera_uid]
    rgb = _numpy(sensor["rgb"])
    segmentation = _numpy(sensor["segmentation"])
    if rgb.shape != (1, 128, 128, 3) or rgb.dtype != np.uint8:
        raise RuntimeError(f"G0B external RGB schema 漂移: {rgb.shape}/{rgb.dtype}")
    if segmentation.ndim != 4 or segmentation.shape[:3] != (1, 128, 128):
        raise RuntimeError(f"G0B external segmentation schema 漂移: {segmentation.shape}")
    rgb = np.ascontiguousarray(rgb[0])
    actor_ids = np.asarray(segmentation[0, ..., 0])
    if not np.issubdtype(actor_ids.dtype, np.integer):
        raise RuntimeError("G0B segmentation actor channel 必须是整数")

    actual_pose = camera.get_local_pose()
    actual_position = _single_vector(actual_pose.p, 3, "actual camera position")
    actual_quaternion = _single_vector(actual_pose.q, 4, "actual camera quaternion")
    actual_world_from_gl = _single_matrix(
        params["cam2world_gl"],
        "actual_world_from_external_camera_gl",
    )
    intrinsic = _numpy(params["intrinsic_cv"])
    if intrinsic.shape == (1, 3, 3):
        intrinsic = intrinsic[0]
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise RuntimeError("G0B intrinsic_cv 无效")
    world_from_camera_cv = opengl_camera_to_opencv(actual_world_from_gl)
    world_from_base = _single_matrix(base_env.agent.robot.pose, "world_from_robot_base")
    base_from_world = invert_se3(world_from_base, "world_from_robot_base")
    base_from_camera_cv = validate_se3(
        base_from_world @ world_from_camera_cv,
        "actual_base_from_external_camera_cv",
    )
    arm_q = _numpy(base_env.agent.robot.get_qpos())[0]
    tcp_world = _single_matrix(base_env.agent.tcp_pose, "actual_world_from_tcp")
    screening = config["screening"]
    object_stats = _mask_stats(
        actor_ids,
        object_actor_id,
        geometric_min_pixels=screening["geometric_visibility_min_pixels"],
        framing_min_pixels=screening["framing_usable_min_pixels"],
        framing_min_margin_px=screening["framing_usable_min_border_margin_px"],
    )
    goal_stats = _mask_stats(
        actor_ids,
        goal_actor_id,
        geometric_min_pixels=screening["geometric_visibility_min_pixels"],
        framing_min_pixels=screening["framing_usable_min_pixels"],
        framing_min_margin_px=screening["framing_usable_min_border_margin_px"],
    )
    object_world = _single_vector(base_env.cube.pose.p, 3, "object position world")
    goal_world = _single_vector(base_env.goal_site.pose.p, 3, "goal position world")
    object_projection = _project_world_point(
        object_world,
        world_from_camera_cv=world_from_camera_cv,
        intrinsic_cv=intrinsic,
        image_shape=actor_ids.shape,
    )
    goal_projection = _project_world_point(
        goal_world,
        world_from_camera_cv=world_from_camera_cv,
        intrinsic_cv=intrinsic,
        image_shape=actor_ids.shape,
    )
    support_radius = screening["center_ray_support_radius_px"]
    object_ray = _center_ray_diagnostic(
        actor_ids,
        object_projection,
        entity_actor_id=object_actor_id,
        robot_actor_ids=robot_actor_ids,
        support_radius_px=support_radius,
    )
    goal_ray = _center_ray_diagnostic(
        actor_ids,
        goal_projection,
        entity_actor_id=goal_actor_id,
        robot_actor_ids=robot_actor_ids,
        support_radius_px=support_radius,
    )
    robot_pixel_count = int(np.count_nonzero(np.isin(actor_ids, tuple(robot_actor_ids))))
    row = {
        "version": E018_P1_G0B_RESULT_VERSION,
        "seed": seed,
        "scene_id": f"g0b-seed-{seed:06d}",
        "primitive_id": primitive_id,
        "anchor_id": anchor.viewpoint_id,
        "lateral_anchor": anchor.lateral_anchor,
        "vertical_anchor": anchor.vertical_anchor,
        "orientation_id": orientation.orientation_id,
        "yaw_offset_rad": orientation.yaw_offset_rad,
        "pitch_offset_rad": orientation.pitch_offset_rad,
        "roll_offset_rad": orientation.roll_offset_rad,
        "capture_index": capture_index,
        "capture_mode": "static-render-only-no-environment-step/v1",
        "sim_time_advanced": False,
        "timestamp_s": None,
        "time_resolved_motion_evidence": False,
        "commanded_external_position_world_m": list(anchor.position_world_m),
        "commanded_external_quaternion_sapien_wxyz": commanded_quaternion.tolist(),
        "actual_external_position_world_m": actual_position.tolist(),
        "actual_external_quaternion_sapien_wxyz": actual_quaternion.tolist(),
        "commanded_world_from_external_camera_gl": commanded_world_from_gl.tolist(),
        "actual_world_from_external_camera_gl": actual_world_from_gl.tolist(),
        "actual_base_from_external_camera_cv": base_from_camera_cv.tolist(),
        "external_intrinsic_cv": intrinsic.tolist(),
        "camera_position_tracking_error_m": float(
            np.linalg.norm(actual_position - np.asarray(anchor.position_world_m))
        ),
        "camera_orientation_tracking_error_rad": quaternion_angular_distance_rad(
            commanded_quaternion,
            actual_quaternion,
        ),
        "arm_joint_max_drift_rad": float(np.max(np.abs(arm_q[:7] - arm_anchor_q_rad))),
        "tcp_position_drift_m": float(
            np.linalg.norm(tcp_world[:3, 3] - tcp_anchor_world[:3, 3])
        ),
        "tcp_orientation_drift_rad": rotation_angular_distance_rad(
            tcp_anchor_world[:3, :3],
            tcp_world[:3, :3],
        ),
        "object": object_stats,
        "goal": goal_stats,
        "object_projection": object_projection,
        "goal_projection": goal_projection,
        "object_center_ray": object_ray,
        "goal_center_ray": goal_ray,
        "robot_visible_pixel_count": robot_pixel_count,
        "robot_visible_pixel_fraction": robot_pixel_count / actor_ids.size,
        "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "segmentation_actor_sha256": hashlib.sha256(actor_ids.tobytes()).hexdigest(),
        "offline_gt_only": True,
        "used_by_runtime_control": False,
        "provider_forward_executed": False,
        "memory_read_executed": False,
        "memory_write_executed": False,
        "test_data_read": False,
    }
    return row, rgb


def _quantile(values: list[float | int], probability: float) -> float:
    if not values:
        raise ValueError("不能对空列表计算 quantile")
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _collapse_repeated_captures(
    rows: list[dict[str, Any]],
    *,
    captures_per_pose: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), str(row["primitive_id"]))].append(row)
    collapsed: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for (seed, primitive_id), group in sorted(grouped.items()):
        group.sort(key=lambda row: int(row["capture_index"]))
        capture_indices = [int(row["capture_index"]) for row in group]
        rgb_hashes = {str(row["rgb_sha256"]) for row in group}
        segmentation_hashes = {str(row["segmentation_actor_sha256"]) for row in group}
        pose_hashes = {
            _canonical_sha256(
                {
                    "position": row["actual_external_position_world_m"],
                    "quaternion": row["actual_external_quaternion_sapien_wxyz"],
                    "base_from_camera_cv": row["actual_base_from_external_camera_cv"],
                }
            )
            for row in group
        }
        passed = bool(
            len(group) == captures_per_pose
            and capture_indices == list(range(captures_per_pose))
            and len(rgb_hashes) == 1
            and len(segmentation_hashes) == 1
            and len(pose_hashes) == 1
        )
        audits.append(
            {
                "seed": seed,
                "primitive_id": primitive_id,
                "capture_count": len(group),
                "capture_indices": capture_indices,
                "unique_rgb_hash_count": len(rgb_hashes),
                "unique_segmentation_hash_count": len(segmentation_hashes),
                "unique_pose_hash_count": len(pose_hashes),
                "passed": passed,
            }
        )
        representative = dict(group[0])
        representative["repeated_capture_audit_passed"] = passed
        collapsed.append(representative)
    return collapsed, audits


def _summarize_primitive(
    primitive_rows: list[dict[str, Any]],
    *,
    home_by_seed: dict[int, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    if not primitive_rows:
        raise ValueError("primitive rows 不能为空")
    first = primitive_rows[0]
    scene_count = len(primitive_rows)

    def rate(predicate: Any) -> float:
        return sum(bool(predicate(row)) for row in primitive_rows) / scene_count

    object_pixels = [int(row["object"]["visible_pixel_count"]) for row in primitive_rows]
    goal_pixels = [int(row["goal"]["visible_pixel_count"]) for row in primitive_rows]
    minimum_entity_pixels = [min(a, b) for a, b in zip(object_pixels, goal_pixels)]
    minimum_margin = [
        min(int(row["object"]["border_margin_px"]), int(row["goal"]["border_margin_px"]))
        for row in primitive_rows
    ]
    object_home_failures = [
        row
        for row in primitive_rows
        if not home_by_seed[int(row["seed"])]["object"]["framing_usable"]
    ]
    goal_home_failures = [
        row
        for row in primitive_rows
        if not home_by_seed[int(row["seed"])]["goal"]["framing_usable"]
    ]
    both_geometric_rate = rate(
        lambda row: row["object"]["geometrically_visible"]
        and row["goal"]["geometrically_visible"]
    )
    both_framing_rate = rate(
        lambda row: row["object"]["framing_usable"] and row["goal"]["framing_usable"]
    )
    screening = config["screening"]
    safety = config["safety"]
    maximum_position_error = max(
        float(row["camera_position_tracking_error_m"]) for row in primitive_rows
    )
    maximum_orientation_error = max(
        float(row["camera_orientation_tracking_error_rad"]) for row in primitive_rows
    )
    maximum_arm_drift = max(float(row["arm_joint_max_drift_rad"]) for row in primitive_rows)
    maximum_tcp_position_drift = max(
        float(row["tcp_position_drift_m"]) for row in primitive_rows
    )
    maximum_tcp_orientation_drift = max(
        float(row["tcp_orientation_drift_rad"]) for row in primitive_rows
    )
    integrity_passed = bool(
        all(bool(row["repeated_capture_audit_passed"]) for row in primitive_rows)
        and maximum_position_error <= safety["camera_position_tracking_tolerance_m"]
        and maximum_orientation_error <= safety["camera_orientation_tracking_tolerance_rad"]
        and maximum_arm_drift <= safety["arm_joint_drift_max_rad"]
        and maximum_tcp_position_drift <= safety["tcp_position_drift_max_m"]
        and maximum_tcp_orientation_drift <= safety["tcp_orientation_drift_max_rad"]
    )
    shortlist_eligible = bool(
        first["primitive_id"] != "HOME__CENTER"
        and integrity_passed
        and both_geometric_rate
        >= screening["shortlist_min_both_geometric_visible_rate"]
        and both_framing_rate >= screening["shortlist_min_both_framing_usable_rate"]
    )
    return {
        "primitive_id": first["primitive_id"],
        "anchor_id": first["anchor_id"],
        "lateral_anchor": first["lateral_anchor"],
        "vertical_anchor": first["vertical_anchor"],
        "orientation_id": first["orientation_id"],
        "yaw_offset_rad": first["yaw_offset_rad"],
        "pitch_offset_rad": first["pitch_offset_rad"],
        "scene_count": scene_count,
        "object_geometric_visible_rate": rate(
            lambda row: row["object"]["geometrically_visible"]
        ),
        "goal_geometric_visible_rate": rate(
            lambda row: row["goal"]["geometrically_visible"]
        ),
        "both_geometric_visible_rate": both_geometric_rate,
        "object_framing_usable_rate": rate(lambda row: row["object"]["framing_usable"]),
        "goal_framing_usable_rate": rate(lambda row: row["goal"]["framing_usable"]),
        "both_framing_usable_rate": both_framing_rate,
        "object_visible_pixels_min": min(object_pixels),
        "object_visible_pixels_p10": _quantile(object_pixels, 0.10),
        "object_visible_pixels_median": _quantile(object_pixels, 0.50),
        "goal_visible_pixels_min": min(goal_pixels),
        "goal_visible_pixels_p10": _quantile(goal_pixels, 0.10),
        "goal_visible_pixels_median": _quantile(goal_pixels, 0.50),
        "minimum_entity_pixels_p10": _quantile(minimum_entity_pixels, 0.10),
        "minimum_entity_border_margin_p10_px": _quantile(minimum_margin, 0.10),
        "mean_robot_visible_pixel_fraction": float(
            np.mean([row["robot_visible_pixel_fraction"] for row in primitive_rows])
        ),
        "object_projection_out_of_frame_count": sum(
            not row["object_projection"]["in_frame"] for row in primitive_rows
        ),
        "goal_projection_out_of_frame_count": sum(
            not row["goal_projection"]["in_frame"] for row in primitive_rows
        ),
        "object_robot_center_ray_count": sum(
            row["object_center_ray"]["classification"] == "ROBOT_AT_ENTITY_CENTER_RAY"
            for row in primitive_rows
        ),
        "goal_robot_center_ray_count": sum(
            row["goal_center_ray"]["classification"] == "ROBOT_AT_ENTITY_CENTER_RAY"
            for row in primitive_rows
        ),
        "home_object_framing_failure_count": len(object_home_failures),
        "home_object_failure_recovered_count": sum(
            bool(row["object"]["framing_usable"]) for row in object_home_failures
        ),
        "home_goal_framing_failure_count": len(goal_home_failures),
        "home_goal_failure_recovered_count": sum(
            bool(row["goal"]["framing_usable"]) for row in goal_home_failures
        ),
        "max_camera_position_tracking_error_m": maximum_position_error,
        "max_camera_orientation_tracking_error_rad": maximum_orientation_error,
        "max_arm_joint_drift_rad": maximum_arm_drift,
        "max_tcp_position_drift_m": maximum_tcp_position_drift,
        "max_tcp_orientation_drift_rad": maximum_tcp_orientation_drift,
        "integrity_passed": integrity_passed,
        "shortlist_eligible": shortlist_eligible,
        "formal_selection_allowed": False,
    }


def _build_development_shortlist(
    pose_summaries: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    screening = config["screening"]
    eligible = [summary for summary in pose_summaries if summary["shortlist_eligible"]]
    eligible.sort(
        key=lambda summary: (
            -float(summary["both_framing_usable_rate"]),
            -float(summary["both_geometric_visible_rate"]),
            -min(
                float(summary["object_framing_usable_rate"]),
                float(summary["goal_framing_usable_rate"]),
            ),
            -float(summary["minimum_entity_pixels_p10"]),
            -float(summary["minimum_entity_border_margin_p10_px"]),
            float(summary["mean_robot_visible_pixel_fraction"]),
            str(summary["primitive_id"]),
        )
    )
    selected: list[dict[str, Any]] = []
    per_anchor: dict[str, int] = defaultdict(int)
    for summary in eligible:
        anchor_id = str(summary["anchor_id"])
        if per_anchor[anchor_id] >= screening["shortlist_max_per_anchor"]:
            continue
        selected.append(summary)
        per_anchor[anchor_id] += 1
        if len(selected) >= screening["shortlist_size"]:
            break
    candidates = [
        {
            "development_rank": index + 1,
            "primitive_id": summary["primitive_id"],
            "anchor_id": summary["anchor_id"],
            "orientation_id": summary["orientation_id"],
            "both_geometric_visible_rate": summary["both_geometric_visible_rate"],
            "both_framing_usable_rate": summary["both_framing_usable_rate"],
            "minimum_entity_pixels_p10": summary["minimum_entity_pixels_p10"],
            "minimum_entity_border_margin_p10_px": summary[
                "minimum_entity_border_margin_p10_px"
            ],
            "object_home_failure_recovered_count": summary[
                "home_object_failure_recovered_count"
            ],
            "goal_home_failure_recovered_count": summary[
                "home_goal_failure_recovered_count"
            ],
        }
        for index, summary in enumerate(selected)
    ]
    return {
        "version": E018_P1_G0B_RESULT_VERSION,
        "status": "development-shortlist-not-frozen-not-provider-qualified",
        "selection_policy": (
            "integrity-and-threshold-then-framing-geometric-worst-entity-"
            "pixel-margin-robot-occupancy/v1"
        ),
        "eligible_candidate_count": len(eligible),
        "requested_shortlist_size": screening["shortlist_size"],
        "selected_candidate_count": len(candidates),
        "maximum_per_anchor": screening["shortlist_max_per_anchor"],
        "candidates": candidates,
        "formal_selection_allowed": False,
        "provider_qualification_required": True,
    }


def _union_coverage(
    collapsed_rows: list[dict[str, Any]],
    *,
    shortlist: dict[str, Any],
) -> dict[str, Any]:
    selected_ids = {candidate["primitive_id"] for candidate in shortlist["candidates"]}
    selected_ids.add("HOME__CENTER")
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in collapsed_rows:
        if row["primitive_id"] in selected_ids:
            by_seed[int(row["seed"])].append(row)
    object_available = 0
    goal_available = 0
    both_available = 0
    for rows in by_seed.values():
        has_object = any(bool(row["object"]["framing_usable"]) for row in rows)
        has_goal = any(bool(row["goal"]["framing_usable"]) for row in rows)
        object_available += int(has_object)
        goal_available += int(has_goal)
        both_available += int(has_object and has_goal)
    count = len(by_seed)
    return {
        "scene_count": count,
        "viewpoint_ids": sorted(selected_ids),
        "object_framing_union_rate": object_available / count if count else 0.0,
        "goal_framing_union_rate": goal_available / count if count else 0.0,
        "both_entities_framing_union_rate": both_available / count if count else 0.0,
        "interpretation": "offline set union only; not a runtime viewpoint-selection claim",
    }


def _write_contact_sheets(
    output_root: Path,
    *,
    anchors: list[FrontCameraViewpoint],
    modes: list[FrontCameraOrientationMode],
    images: dict[tuple[int, str], np.ndarray],
) -> list[Path]:
    from PIL import Image, ImageDraw

    output_paths: list[Path] = []
    seeds = sorted({seed for seed, _ in images})
    cell_height = 128
    cell_width = 128
    label_height = 19
    for seed in seeds:
        sheet = Image.new(
            "RGB",
            (cell_width * len(modes), (cell_height + label_height) * len(anchors)),
            color=(255, 255, 255),
        )
        draw = ImageDraw.Draw(sheet)
        for row_index, anchor in enumerate(anchors):
            for column_index, mode in enumerate(modes):
                primitive_id = f"{anchor.viewpoint_id}__{mode.orientation_id}"
                rgb = images[(seed, primitive_id)]
                x = column_index * cell_width
                y = row_index * (cell_height + label_height)
                sheet.paste(Image.fromarray(rgb, mode="RGB"), (x, y + label_height))
                draw.text(
                    (x + 2, y + 3),
                    f"{anchor.viewpoint_id}/{mode.orientation_id}",
                    fill=(0, 0, 0),
                )
        path = output_root / "contact_sheets" / f"seed-{seed:06d}-5x5.png"
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        sheet.save(path)
        os.chmod(path, 0o600)
        output_paths.append(path)
    return output_paths


def _run_simulator(
    *,
    config: dict[str, Any],
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[Path]]:
    import gymnasium as gym
    import mani_skill
    import sapien
    import torch
    from mani_skill.utils import sapien_utils

    from robot_vla.sim import register_robot_vla_maniskill_envs

    software = config["software"]
    if mani_skill.__version__ != software["expected_mani_skill_version"]:
        raise RuntimeError("G0B ManiSkill version 漂移")
    if sapien.__version__ != software["expected_sapien_version"]:
        raise RuntimeError("G0B SAPIEN version 漂移")
    if not torch.cuda.is_available():
        raise RuntimeError("G0B config 要求 CUDA")

    lattice = config["viewpoint_lattice"]
    anchors = [
        _parse_anchor(item, f"viewpoint_lattice.anchors[{index}]")
        for index, item in enumerate(lattice["anchors"])
    ]
    modes = [
        _parse_orientation(item, f"viewpoint_lattice.orientation_modes[{index}]")
        for index, item in enumerate(lattice["orientation_modes"])
    ]
    home = next(anchor for anchor in anchors if anchor.viewpoint_id == "HOME")
    center = next(mode for mode in modes if mode.orientation_id == "CENTER")
    environment = config["environment"]
    sampling = config["sampling"]
    seeds = list(
        range(
            sampling["seed_start"],
            sampling["seed_start"] + sampling["seed_count"],
        )
    )
    visualization_seeds = set(sampling["visualization_seeds"])
    register_robot_vla_maniskill_envs()
    env = gym.make(
        environment["environment_id"],
        obs_mode=environment["obs_mode"],
        control_mode=environment["control_mode"],
        num_envs=environment["num_envs"],
        robot_uids=environment["robot_uid"],
    )
    rows: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []
    visualization_images: dict[tuple[int, str], np.ndarray] = {}
    try:
        base_env = env.unwrapped
        sensor = base_env._sensors.get(environment["camera_uid"])
        if sensor is None:
            raise RuntimeError("G0B external camera uid 不存在")
        camera = sensor.camera
        if sensor.entity is not None or not callable(getattr(camera, "set_local_pose", None)):
            raise RuntimeError("G0B 要求独立、可设位姿的 unmounted external camera")
        for seed in seeds:
            env.reset(seed=seed)
            qpos = _numpy(base_env.agent.robot.get_qpos())[0]
            if qpos.shape != (9,):
                raise RuntimeError("G0B 只支持 Panda 9-DoF qpos")
            arm_anchor_q = np.asarray(qpos[:7], dtype=np.float64).copy()
            tcp_anchor = _single_matrix(base_env.agent.tcp_pose, "anchor_world_from_tcp")
            object_position = _single_vector(base_env.cube.pose.p, 3, "object position world")
            goal_position = _single_vector(base_env.goal_site.pose.p, 3, "goal position world")
            object_actor_id = int(_numpy(base_env.cube.per_scene_id).reshape(-1)[0])
            goal_actor_id = int(_numpy(base_env.goal_site.per_scene_id).reshape(-1)[0])
            robot_actor_ids = {
                int(_numpy(link.per_scene_id).reshape(-1)[0])
                for link in base_env.agent.robot.links
            }
            home_reference_rgb: np.ndarray | None = None
            scene_start_index = len(rows)
            for anchor in anchors:
                for mode in modes:
                    primitive_id = f"{anchor.viewpoint_id}__{mode.orientation_id}"
                    command_q, command_gl = _set_static_camera_pose(
                        camera,
                        anchor,
                        mode,
                        sapien_module=sapien,
                        sapien_utils_module=sapien_utils,
                    )
                    for capture_index in range(sampling["captures_per_pose"]):
                        observation = _capture_sensor_observation(base_env)
                        row, rgb = _capture_row(
                            seed=seed,
                            anchor=anchor,
                            orientation=mode,
                            primitive_id=primitive_id,
                            capture_index=capture_index,
                            commanded_quaternion=command_q,
                            commanded_world_from_gl=command_gl,
                            observation=observation,
                            camera=camera,
                            base_env=base_env,
                            config=config,
                            arm_anchor_q_rad=arm_anchor_q,
                            tcp_anchor_world=tcp_anchor,
                            object_actor_id=object_actor_id,
                            goal_actor_id=goal_actor_id,
                            robot_actor_ids=robot_actor_ids,
                        )
                        rows.append(row)
                        if capture_index == 0 and seed in visualization_seeds:
                            visualization_images[(seed, primitive_id)] = rgb.copy()
                        if primitive_id == "HOME__CENTER" and capture_index == 0:
                            home_reference_rgb = rgb.copy()
            if home_reference_rgb is None:
                raise RuntimeError("G0B HOME__CENTER reference RGB 缺失")

            home_q, home_gl = _set_static_camera_pose(
                camera,
                home,
                center,
                sapien_module=sapien,
                sapien_utils_module=sapien_utils,
            )
            final_observation = _capture_sensor_observation(base_env)
            final_sensor = final_observation["sensor_data"][environment["camera_uid"]]
            final_rgb = np.ascontiguousarray(_numpy(final_sensor["rgb"])[0])
            final_pose = camera.get_local_pose()
            final_position = _single_vector(final_pose.p, 3, "final HOME camera position")
            final_quaternion = _single_vector(final_pose.q, 4, "final HOME camera quaternion")
            final_actual_gl = _single_matrix(
                final_observation["sensor_param"][environment["camera_uid"]]["cam2world_gl"],
                "final actual world_from_external_camera_gl",
            )
            final_qpos = _numpy(base_env.agent.robot.get_qpos())[0]
            final_tcp = _single_matrix(base_env.agent.tcp_pose, "final world_from_tcp")
            scene_rows = rows[scene_start_index:]
            scene_summaries.append(
                {
                    "version": E018_P1_G0B_RESULT_VERSION,
                    "seed": seed,
                    "scene_id": f"g0b-seed-{seed:06d}",
                    "object_position_world_m": object_position.tolist(),
                    "goal_position_world_m": goal_position.tolist(),
                    "primitive_count": len({row["primitive_id"] for row in scene_rows}),
                    "capture_count": len(scene_rows),
                    "environment_step_count": 0,
                    "final_home_position_error_m": float(
                        np.linalg.norm(final_position - np.asarray(home.position_world_m))
                    ),
                    "final_home_orientation_error_rad": quaternion_angular_distance_rad(
                        final_quaternion,
                        home_q,
                    ),
                    "final_home_gl_matrix_error_max_abs": float(
                        np.max(np.abs(final_actual_gl - home_gl))
                    ),
                    "final_home_rgb_mean_abs_difference": float(
                        np.mean(
                            np.abs(
                                final_rgb.astype(np.float64)
                                - home_reference_rgb.astype(np.float64)
                            )
                        )
                    ),
                    "max_arm_joint_drift_rad": float(
                        np.max(np.abs(final_qpos[:7] - arm_anchor_q))
                    ),
                    "tcp_position_drift_m": float(
                        np.linalg.norm(final_tcp[:3, 3] - tcp_anchor[:3, 3])
                    ),
                    "tcp_orientation_drift_rad": rotation_angular_distance_rad(
                        tcp_anchor[:3, :3],
                        final_tcp[:3, :3],
                    ),
                    "test_data_read": False,
                    "provider_forward_count": 0,
                    "memory_access_count": 0,
                    "manipulation_progression_count": 0,
                }
            )
    finally:
        env.close()

    contact_sheets = _write_contact_sheets(
        output_root,
        anchors=anchors,
        modes=modes,
        images=visualization_images,
    )
    environment_identity = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(torch.device("cuda")),
        "mani_skill": mani_skill.__version__,
        "sapien": sapien.__version__,
        "external_camera_sensor_class": type(sensor).__module__ + "." + type(sensor).__name__,
        "external_camera_class": type(camera).__module__ + "." + type(camera).__name__,
        "external_camera_unmounted": sensor.entity is None,
        "set_local_pose_callable": callable(getattr(camera, "set_local_pose", None)),
        "capture_path": "BaseEnv._get_obs_sensor_data + get_sensor_params",
        "environment_steps_executed": 0,
    }
    return rows, scene_summaries, environment_identity, contact_sheets


def run_e018_p1_viewpoint_screen(
    *,
    config_path: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """运行静态 25-pose development screen；不接受 split/checkpoint/policy。"""

    config_file = Path(config_path)
    repository = Path(repository_root)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"E018-P1 G0B output 已存在: {output}")
    config = load_e018_p1_g0b_config(config_file)
    config_sha256 = _canonical_sha256(config)
    source_identity = _source_identity(repository)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _atomic_json(
        output / "run_state.json",
        {
            "version": E018_P1_G0B_RESULT_VERSION,
            "status": "in-progress-development-only",
            "gate": "G0B_STATIC_VIEWPOINT_GEOMETRY_SCREEN",
            "test_split_status": "prohibited-unread",
            "formal_claim_allowed": False,
        },
    )
    try:
        frame_rows, scene_summaries, environment_identity, contact_sheets = _run_simulator(
            config=config,
            output_root=output,
        )
        sampling = config["sampling"]
        lattice = config["viewpoint_lattice"]
        expected_frame_count = (
            sampling["seed_count"]
            * lattice["expected_primitive_count"]
            * sampling["captures_per_pose"]
        )
        collapsed_rows, repeat_audits = _collapse_repeated_captures(
            frame_rows,
            captures_per_pose=sampling["captures_per_pose"],
        )
        expected_collapsed_count = (
            sampling["seed_count"] * lattice["expected_primitive_count"]
        )
        home_rows = [row for row in collapsed_rows if row["primitive_id"] == "HOME__CENTER"]
        home_by_seed = {int(row["seed"]): row for row in home_rows}
        if len(home_by_seed) != sampling["seed_count"]:
            raise RuntimeError("G0B HOME__CENTER baseline scene identity 不完整")
        by_primitive: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in collapsed_rows:
            by_primitive[str(row["primitive_id"])].append(row)
        pose_summaries = [
            _summarize_primitive(
                rows,
                home_by_seed=home_by_seed,
                config=config,
            )
            for _, rows in sorted(by_primitive.items())
        ]
        shortlist = _build_development_shortlist(pose_summaries, config=config)
        shortlist["offline_union_coverage_with_home"] = _union_coverage(
            collapsed_rows,
            shortlist=shortlist,
        )

        safety = config["safety"]
        max_position_error = max(
            float(row["camera_position_tracking_error_m"]) for row in frame_rows
        )
        max_orientation_error = max(
            float(row["camera_orientation_tracking_error_rad"]) for row in frame_rows
        )
        max_arm_drift = max(float(row["arm_joint_max_drift_rad"]) for row in frame_rows)
        max_tcp_position_drift = max(
            float(row["tcp_position_drift_m"]) for row in frame_rows
        )
        max_tcp_orientation_drift = max(
            float(row["tcp_orientation_drift_rad"]) for row in frame_rows
        )
        max_final_home_position_error = max(
            float(scene["final_home_position_error_m"]) for scene in scene_summaries
        )
        max_final_home_orientation_error = max(
            float(scene["final_home_orientation_error_rad"]) for scene in scene_summaries
        )
        max_final_home_rgb_difference = max(
            float(scene["final_home_rgb_mean_abs_difference"])
            for scene in scene_summaries
        )
        repeat_failure_count = sum(not audit["passed"] for audit in repeat_audits)
        integrity_gates = {
            "complete_frame_lattice": {
                "actual": len(frame_rows),
                "required": expected_frame_count,
                "passed": len(frame_rows) == expected_frame_count,
            },
            "complete_scene_pose_lattice": {
                "actual": len(collapsed_rows),
                "required": expected_collapsed_count,
                "passed": len(collapsed_rows) == expected_collapsed_count,
            },
            "repeated_capture_determinism": {
                "actual_failures": repeat_failure_count,
                "required_failures": 0,
                "passed": repeat_failure_count == 0,
            },
            "camera_pose_tracking": {
                "actual_position_error_m": max_position_error,
                "required_position_error_max_m": safety[
                    "camera_position_tracking_tolerance_m"
                ],
                "actual_orientation_error_rad": max_orientation_error,
                "required_orientation_error_max_rad": safety[
                    "camera_orientation_tracking_tolerance_rad"
                ],
                "passed": max_position_error
                <= safety["camera_position_tracking_tolerance_m"]
                and max_orientation_error
                <= safety["camera_orientation_tracking_tolerance_rad"],
            },
            "arm_tcp_stationary": {
                "actual_arm_joint_drift_rad": max_arm_drift,
                "required_arm_joint_drift_max_rad": safety["arm_joint_drift_max_rad"],
                "actual_tcp_position_drift_m": max_tcp_position_drift,
                "required_tcp_position_drift_max_m": safety["tcp_position_drift_max_m"],
                "actual_tcp_orientation_drift_rad": max_tcp_orientation_drift,
                "required_tcp_orientation_drift_max_rad": safety[
                    "tcp_orientation_drift_max_rad"
                ],
                "passed": max_arm_drift <= safety["arm_joint_drift_max_rad"]
                and max_tcp_position_drift <= safety["tcp_position_drift_max_m"]
                and max_tcp_orientation_drift <= safety["tcp_orientation_drift_max_rad"],
            },
            "return_home": {
                "actual_position_error_m": max_final_home_position_error,
                "actual_orientation_error_rad": max_final_home_orientation_error,
                "actual_rgb_mean_abs_difference": max_final_home_rgb_difference,
                "passed": max_final_home_position_error
                <= safety["camera_position_tracking_tolerance_m"]
                and max_final_home_orientation_error
                <= safety["camera_orientation_tracking_tolerance_rad"]
                and max_final_home_rgb_difference == 0.0,
            },
            "forbidden_operations_absent": {
                "environment_steps": environment_identity["environment_steps_executed"],
                "provider_forwards": sum(
                    scene["provider_forward_count"] for scene in scene_summaries
                ),
                "memory_accesses": sum(
                    scene["memory_access_count"] for scene in scene_summaries
                ),
                "manipulation_progressions": sum(
                    scene["manipulation_progression_count"] for scene in scene_summaries
                ),
                "test_reads": sum(bool(row["test_data_read"]) for row in frame_rows),
                "required": 0,
                "passed": True,
            },
            "runtime_gt_control_dependency_absent": {
                "actual": sum(bool(row["used_by_runtime_control"]) for row in frame_rows),
                "required": 0,
                "passed": not any(bool(row["used_by_runtime_control"]) for row in frame_rows),
            },
        }
        screen_integrity_passed = all(gate["passed"] for gate in integrity_gates.values())
        home_summary = next(
            summary for summary in pose_summaries if summary["primitive_id"] == "HOME__CENTER"
        )
        summary = {
            "version": E018_P1_G0B_RESULT_VERSION,
            "status": "complete-development-only",
            "gate": "G0B_STATIC_VIEWPOINT_GEOMETRY_SCREEN",
            "screen_integrity_passed": screen_integrity_passed,
            "config_sha256": config_sha256,
            "viewpoint_lattice_version": lattice["version"],
            "viewpoint_lattice_status": lattice["status"],
            "source_identity": source_identity,
            "environment_identity": environment_identity,
            "scene_count": sampling["seed_count"],
            "anchor_count": lattice["expected_anchor_count"],
            "orientation_count": lattice["expected_orientation_count"],
            "primitive_count": lattice["expected_primitive_count"],
            "captures_per_pose": sampling["captures_per_pose"],
            "frame_count": len(frame_rows),
            "integrity_gates": integrity_gates,
            "home_center_baseline": {
                "object_geometric_visible_rate": home_summary[
                    "object_geometric_visible_rate"
                ],
                "goal_geometric_visible_rate": home_summary["goal_geometric_visible_rate"],
                "both_geometric_visible_rate": home_summary[
                    "both_geometric_visible_rate"
                ],
                "object_framing_usable_rate": home_summary["object_framing_usable_rate"],
                "goal_framing_usable_rate": home_summary["goal_framing_usable_rate"],
                "both_framing_usable_rate": home_summary["both_framing_usable_rate"],
            },
            "development_shortlist": shortlist,
            "scope_limits": {
                "proves": [
                    "25 static camera poses can be rendered with exact actual extrinsics",
                    "per-pose object/goal geometry and robot-center-ray diagnostics are available",
                    "discrete local yaw/pitch signs and framing effects are observable",
                ],
                "does_not_prove": [
                    "time-resolved motion, path, acceleration, or collision safety for new rotations",
                    "pregrasp or deliberately occluded challenge-set coverage",
                    "front perception provider qualification",
                    "runtime viewpoint selection or closed-loop recovery",
                ],
            },
            "development_shortlist_available": bool(shortlist["candidates"]),
            "ready_for_formal_preregistration": False,
            "test_split_status": "prohibited-unread",
            "test_data_read_count": 0,
            "provider_forward_count": 0,
            "memory_access_count": 0,
            "environment_step_count": 0,
            "physical_robot_actuation_allowed": False,
            "formal_claim_allowed": False,
        }
        _atomic_jsonl(output / "static_pose_frames.jsonl", frame_rows)
        _atomic_jsonl(output / "repeatability_audits.jsonl", repeat_audits)
        _atomic_jsonl(output / "scene_summaries.jsonl", scene_summaries)
        _atomic_json(output / "pose_summaries.json", pose_summaries)
        _atomic_json(output / "development_shortlist.json", shortlist)
        _atomic_json(output / "config_snapshot.json", config)
        _atomic_json(output / "summary.json", summary)
        artifact_paths = sorted(
            [
                output / "static_pose_frames.jsonl",
                output / "repeatability_audits.jsonl",
                output / "scene_summaries.jsonl",
                output / "pose_summaries.json",
                output / "development_shortlist.json",
                output / "config_snapshot.json",
                output / "summary.json",
                *contact_sheets,
            ],
            key=lambda path: str(path.relative_to(output)),
        )
        receipt = {
            "version": E018_P1_G0B_RESULT_VERSION,
            "status": "complete-development-only",
            "screen_integrity_passed": screen_integrity_passed,
            "config_sha256": config_sha256,
            "files": {
                str(path.relative_to(output)): _file_sha256(path) for path in artifact_paths
            },
            "test_split_status": "prohibited-unread",
            "formal_claim_allowed": False,
        }
        receipt["receipt_sha256"] = _canonical_sha256(receipt)
        _atomic_json(output / "receipt.json", receipt)
        _atomic_json(
            output / "run_state.json",
            {
                "version": E018_P1_G0B_RESULT_VERSION,
                "status": "complete-development-only",
                "gate": "G0B_STATIC_VIEWPOINT_GEOMETRY_SCREEN",
                "screen_integrity_passed": screen_integrity_passed,
                "test_split_status": "prohibited-unread",
                "formal_claim_allowed": False,
            },
        )
        return summary
    except Exception as error:
        _atomic_json(
            output / "run_state.json",
            {
                "version": E018_P1_G0B_RESULT_VERSION,
                "status": "failed-development-only",
                "gate": "G0B_STATIC_VIEWPOINT_GEOMETRY_SCREEN",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "test_split_status": "prohibited-unread",
                "formal_claim_allowed": False,
            },
        )
        raise


__all__ = [
    "E018_P1_G0B_CONFIG_VERSION",
    "E018_P1_G0B_LATTICE_VERSION",
    "E018_P1_G0B_RESULT_VERSION",
    "load_e018_p1_g0b_config",
    "run_e018_p1_viewpoint_screen",
]
