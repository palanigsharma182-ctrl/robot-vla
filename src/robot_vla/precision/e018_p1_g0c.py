"""E018-P1 G0C：低位离散旋转视角的有时延运动可行性门禁。"""

from __future__ import annotations

import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla.precision import e018_p1_g0 as _g0
from robot_vla.precision.active_front_camera import (
    ExternalCameraMotionState,
    FrontCameraOrientationMode,
    FrontCameraViewpoint,
)

E018_P1_G0C_CONFIG_VERSION = "e018-p1-g0c-rotated-motion-development/v1"
E018_P1_G0C_RESULT_VERSION = "e018-p1-g0c-rotated-motion-result/v1"
E018_P1_G0C_LIBRARY_VERSION = (
    "e018-p1-front-low-2x5-rotated-motion-provisional/v1"
)
E018_P1_G0C_GATE = "G0C_ROTATED_POSE_MOTION_FEASIBILITY"

_G0B_CONFIG_SHA256 = "b413487cc35a8ffb8bbaeba8ec6401bbe76eb885b6a3774d1e53b16e18b058a3"
_G0B_RECEIPT_SHA256 = "faf4609954cb1be33e16198ee7b305c7e8f5fabd58a8a48710fd30035f5309ea"
_ELIGIBLE_PRIMITIVE_IDS = (
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
_SOURCE_FILES = (
    "src/robot_vla/precision/active_front_camera.py",
    "src/robot_vla/precision/e018_p1_g0.py",
    "src/robot_vla/precision/e018_p1_g0c.py",
    "src/robot_vla/cli/run_e018_p1_g0c.py",
)


def _parse_orientation(value: Any, name: str) -> FrontCameraOrientationMode:
    item = _g0._require_keys(
        value,
        {
            "orientation_id",
            "yaw_offset_rad",
            "pitch_offset_rad",
            "roll_offset_rad",
        },
        name,
    )
    orientation = FrontCameraOrientationMode(
        orientation_id=str(item["orientation_id"]),
        yaw_offset_rad=float(item["yaw_offset_rad"]),
        pitch_offset_rad=float(item["pitch_offset_rad"]),
        roll_offset_rad=float(item["roll_offset_rad"]),
    )
    orientation.validate()
    return orientation


def _parse_library(
    config: dict[str, Any],
) -> tuple[
    FrontCameraViewpoint,
    list[FrontCameraViewpoint],
    list[FrontCameraOrientationMode],
]:
    library = config["viewpoint_library"]
    home = _g0._parse_viewpoint(library["home"], "viewpoint_library.home")
    anchors = [
        _g0._parse_viewpoint(item, f"viewpoint_library.anchors[{index}]")
        for index, item in enumerate(library["anchors"])
    ]
    orientations = [
        _parse_orientation(item, f"viewpoint_library.orientation_modes[{index}]")
        for index, item in enumerate(library["orientation_modes"])
    ]
    return home, anchors, orientations


def _expand_primitives(
    anchors: list[FrontCameraViewpoint],
    orientations: list[FrontCameraOrientationMode],
) -> list[tuple[FrontCameraViewpoint, FrontCameraOrientationMode]]:
    result: list[tuple[FrontCameraViewpoint, FrontCameraOrientationMode]] = []
    for anchor in anchors:
        for orientation in orientations:
            primitive = FrontCameraViewpoint(
                viewpoint_id=f"{anchor.viewpoint_id}__{orientation.orientation_id}",
                lateral_anchor=anchor.lateral_anchor,
                vertical_anchor=anchor.vertical_anchor,
                position_world_m=anchor.position_world_m,
                look_at_world_m=anchor.look_at_world_m,
                yaw_rad=anchor.yaw_rad,
                pitch_rad=anchor.pitch_rad,
                roll_rad=anchor.roll_rad,
            )
            primitive.validate()
            result.append((primitive, orientation))
    return result


def load_e018_p1_g0c_config(path: str | Path) -> dict[str, Any]:
    """严格读取 G0C development config；接口不接受 split/checkpoint/policy。"""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"E018-P1 G0C config 不存在: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = _g0._require_keys(
        config,
        {
            "version",
            "status",
            "scope",
            "parent_screen",
            "software",
            "environment",
            "viewpoint_library",
            "motion",
            "safety",
            "experiment",
            "execution",
        },
        "E018-P1 G0C config",
    )
    if config["version"] != E018_P1_G0C_CONFIG_VERSION:
        raise ValueError("E018-P1 G0C config version 漂移")
    if config["status"] != "development-only-g0c-no-formal-claim":
        raise ValueError("E018-P1 G0C 只能以 development-only 状态运行")

    scope = _g0._require_keys(
        config["scope"],
        {
            "gate",
            "test_split_allowed",
            "formal_claim_allowed",
            "runtime_gt_control_allowed",
            "p0_rules_consumed",
            "provider_inference_allowed",
            "memory_read_allowed",
            "memory_write_allowed",
        },
        "scope",
    )
    if scope != {
        "gate": E018_P1_G0C_GATE,
        "test_split_allowed": False,
        "formal_claim_allowed": False,
        "runtime_gt_control_allowed": False,
        "p0_rules_consumed": False,
        "provider_inference_allowed": False,
        "memory_read_allowed": False,
        "memory_write_allowed": False,
    }:
        raise ValueError("G0C scope 必须禁止 test/formal/GT/provider/Memory/P0 runtime")

    parent = _g0._require_keys(
        config["parent_screen"],
        {
            "gate",
            "config_sha256",
            "receipt_sha256",
            "screen_integrity_required",
            "selection_status",
            "eligible_primitive_ids",
        },
        "parent_screen",
    )
    if (
        parent["gate"] != "G0B_STATIC_VIEWPOINT_GEOMETRY_SCREEN"
        or parent["config_sha256"] != _G0B_CONFIG_SHA256
        or parent["receipt_sha256"] != _G0B_RECEIPT_SHA256
        or parent["screen_integrity_required"] is not True
        or parent["selection_status"] != "static-eligible-development-not-frozen"
        or tuple(parent["eligible_primitive_ids"]) != _ELIGIBLE_PRIMITIVE_IDS
    ):
        raise ValueError("G0C parent G0B identity/eligible pool 漂移")

    software = _g0._require_keys(
        config["software"],
        {"expected_mani_skill_version", "expected_sapien_version"},
        "software",
    )
    if software != {
        "expected_mani_skill_version": "3.0.1",
        "expected_sapien_version": "3.0.3",
    }:
        raise ValueError("G0C software identity 漂移")

    environment = _g0._require_keys(
        config["environment"],
        {
            "environment_id",
            "robot_uid",
            "camera_uid",
            "obs_mode",
            "control_mode",
            "num_envs",
            "control_hz",
        },
        "environment",
    )
    if environment != {
        "environment_id": "RobotVLAPickCubeToRegion-v1",
        "robot_uid": "panda_wristcam",
        "camera_uid": "base_camera",
        "obs_mode": "rgb+segmentation",
        "control_mode": "pd_joint_delta_pos",
        "num_envs": 1,
        "control_hz": 20,
    }:
        raise ValueError("G0C environment identity 漂移")

    library = _g0._require_keys(
        config["viewpoint_library"],
        {
            "version",
            "status",
            "pose_frame",
            "nominal_orientation_policy",
            "orientation_offset_frame",
            "interpolation",
            "route_policy",
            "expected_anchor_count",
            "expected_orientation_count",
            "expected_primitive_count",
            "home",
            "anchors",
            "orientation_modes",
        },
        "viewpoint_library",
    )
    expected_library_identity = {
        "version": E018_P1_G0C_LIBRARY_VERSION,
        "status": "provisional-development-not-frozen",
        "pose_frame": "world",
        "nominal_orientation_policy": "fixed-workspace-look-at/v1",
        "orientation_offset_frame": (
            "sapien-camera-local-x-forward-y-left-z-up/v1"
        ),
        "interpolation": "shared-quintic-position-and-local-orientation-offset/v1",
        "route_policy": "HOME_TO_ONE_FULL_POSE_TO_HOME/v1",
        "expected_anchor_count": 2,
        "expected_orientation_count": 5,
        "expected_primitive_count": 10,
    }
    for name, expected in expected_library_identity.items():
        if library[name] != expected:
            raise ValueError(f"G0C viewpoint_library.{name} 漂移")

    expected_home = {
        "viewpoint_id": "HOME",
        "lateral_anchor": "CENTER",
        "vertical_anchor": "CENTER",
        "position_world_m": [0.3, 0.0, 0.6],
        "look_at_world_m": [-0.1, 0.0, 0.1],
        "yaw_rad": math.pi,
        "pitch_rad": -0.8960553845713439,
        "roll_rad": 0.0,
    }
    expected_anchors = [
        {
            "viewpoint_id": "LEFT_LOW",
            "lateral_anchor": "LEFT",
            "vertical_anchor": "LOW",
            "position_world_m": [0.3, -0.16, 0.48],
            "look_at_world_m": [-0.1, 0.0, 0.1],
            "yaw_rad": 2.761086276477428,
            "pitch_rad": -0.7228106035383705,
            "roll_rad": 0.0,
        },
        {
            "viewpoint_id": "RIGHT_LOW",
            "lateral_anchor": "RIGHT",
            "vertical_anchor": "LOW",
            "position_world_m": [0.3, 0.16, 0.48],
            "look_at_world_m": [-0.1, 0.0, 0.1],
            "yaw_rad": -2.761086276477428,
            "pitch_rad": -0.7228106035383705,
            "roll_rad": 0.0,
        },
    ]
    expected_orientations = [
        {
            "orientation_id": "CENTER",
            "yaw_offset_rad": 0.0,
            "pitch_offset_rad": 0.0,
            "roll_offset_rad": 0.0,
        },
        {
            "orientation_id": "YAW_LEFT",
            "yaw_offset_rad": 0.20943951023931953,
            "pitch_offset_rad": 0.0,
            "roll_offset_rad": 0.0,
        },
        {
            "orientation_id": "YAW_RIGHT",
            "yaw_offset_rad": -0.20943951023931953,
            "pitch_offset_rad": 0.0,
            "roll_offset_rad": 0.0,
        },
        {
            "orientation_id": "PITCH_UP",
            "yaw_offset_rad": 0.0,
            "pitch_offset_rad": 0.13962634015954636,
            "roll_offset_rad": 0.0,
        },
        {
            "orientation_id": "PITCH_DOWN",
            "yaw_offset_rad": 0.0,
            "pitch_offset_rad": -0.13962634015954636,
            "roll_offset_rad": 0.0,
        },
    ]
    if library["home"] != expected_home:
        raise ValueError("G0C registered HOME pose 数值漂移")
    if library["anchors"] != expected_anchors:
        raise ValueError("G0C registered low-anchor pose 数值或顺序漂移")
    if library["orientation_modes"] != expected_orientations:
        raise ValueError("G0C registered orientation offset 数值或顺序漂移")

    home, anchors, orientations = _parse_library(config)
    if (
        home.viewpoint_id != "HOME"
        or home.lateral_anchor != "CENTER"
        or home.vertical_anchor != "CENTER"
    ):
        raise ValueError("G0C HOME anchor 漂移")
    if len(anchors) != library["expected_anchor_count"]:
        raise ValueError("G0C anchor count 漂移")
    if {(item.lateral_anchor, item.vertical_anchor) for item in anchors} != {
        ("LEFT", "LOW"),
        ("RIGHT", "LOW"),
    }:
        raise ValueError("G0C 只允许 LEFT_LOW 与 RIGHT_LOW")
    if len({item.viewpoint_id for item in anchors}) != len(anchors):
        raise ValueError("G0C anchor viewpoint_id 必须唯一")
    if any(item.look_at_world_m != home.look_at_world_m for item in anchors):
        raise ValueError("G0C anchors 必须共享冻结 workspace look-at target")
    if len(orientations) != library["expected_orientation_count"]:
        raise ValueError("G0C orientation count 漂移")
    if [item.orientation_id for item in orientations] != [
        "CENTER",
        "YAW_LEFT",
        "YAW_RIGHT",
        "PITCH_UP",
        "PITCH_DOWN",
    ]:
        raise ValueError("G0C orientation mode/order 漂移")
    primitives = _expand_primitives(anchors, orientations)
    if len(primitives) != library["expected_primitive_count"]:
        raise ValueError("G0C primitive count 漂移")
    if tuple(item.viewpoint_id for item, _ in primitives) != _ELIGIBLE_PRIMITIVE_IDS:
        raise ValueError("G0C expanded primitive ids 与 G0B eligible pool 不一致")

    motion = _g0._require_keys(
        config["motion"],
        {
            "command_mode",
            "interpolation",
            "warmup_ticks",
            "move_duration_s",
            "settle_ticks",
            "collect_ticks",
            "max_linear_velocity_m_s",
            "max_linear_acceleration_m_s2",
            "max_angular_velocity_rad_s",
            "max_angular_acceleration_rad_s2",
        },
        "motion",
    )
    if motion["command_mode"] != "time-indexed-kinematic-render-camera/v1":
        raise ValueError("G0C camera command mode 漂移")
    if (
        motion["interpolation"]
        != "shared-quintic-position-and-local-orientation-offset/v1"
    ):
        raise ValueError("G0C interpolation 漂移")
    for name in ("warmup_ticks", "settle_ticks", "collect_ticks"):
        _g0._positive_int(motion[name], f"motion.{name}")
    move_duration = _g0._positive(
        motion["move_duration_s"],
        "motion.move_duration_s",
    )
    move_steps = move_duration * environment["control_hz"]
    if not float(move_steps).is_integer() or int(move_steps) <= 1:
        raise ValueError("G0C move_duration_s * control_hz 必须是大于 1 的整数")
    for name in (
        "max_linear_velocity_m_s",
        "max_linear_acceleration_m_s2",
        "max_angular_velocity_rad_s",
        "max_angular_acceleration_rad_s2",
    ):
        _g0._positive(motion[name], f"motion.{name}")

    safety = _g0._require_keys(
        config["safety"],
        {
            "camera_position_tracking_tolerance_m",
            "camera_orientation_tracking_tolerance_rad",
            "settled_linear_velocity_max_m_s",
            "settled_angular_velocity_max_rad_s",
            "required_consecutive_settled_ticks",
            "home_position_tolerance_m",
            "home_orientation_tolerance_rad",
            "arm_joint_drift_max_rad",
            "tcp_position_drift_max_m",
            "tcp_orientation_drift_max_rad",
            "minimum_open_finger_joint_position_m",
            "unexpected_finger_contact_max_n",
            "minimum_alternate_rgb_mean_abs_difference",
            "maximum_return_home_rgb_mean_abs_difference",
        },
        "safety",
    )
    _g0._positive_int(
        safety["required_consecutive_settled_ticks"],
        "safety.required_consecutive_settled_ticks",
    )
    if safety["required_consecutive_settled_ticks"] > motion["settle_ticks"]:
        raise ValueError("G0C required settled ticks 超过 settle window")
    for name, value in safety.items():
        if name == "required_consecutive_settled_ticks":
            continue
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"safety.{name} 必须是有限非负数")

    experiment = _g0._require_keys(
        config["experiment"],
        {
            "usage",
            "seeds",
            "execute_all_primitives",
            "save_settled_rgb",
            "offline_segmentation_diagnostics",
        },
        "experiment",
    )
    if (
        experiment["usage"]
        != "simulator-development-only-no-test-no-provider-no-memory/v1"
        or experiment["execute_all_primitives"] is not True
        or experiment["save_settled_rgb"] is not True
        or experiment["offline_segmentation_diagnostics"] is not True
    ):
        raise ValueError("G0C experiment scope 漂移")
    seeds = experiment["seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != 4
        or len(set(seeds)) != len(seeds)
        or any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed <= 0
            for seed in seeds
        )
    ):
        raise ValueError("G0C seeds 必须是四个唯一正整数 development seeds")

    execution = _g0._require_keys(
        config["execution"],
        {
            "device",
            "physical_robot_actuation_allowed",
            "arm_motion_command_allowed",
            "gripper_command_mode",
            "simulated_external_camera_motion_allowed",
            "provider_inference_allowed",
            "memory_read_allowed",
            "memory_write_allowed",
            "manipulation_progression_allowed",
        },
        "execution",
    )
    if execution != {
        "device": "cuda",
        "physical_robot_actuation_allowed": False,
        "arm_motion_command_allowed": False,
        "gripper_command_mode": "safe-hold-open",
        "simulated_external_camera_motion_allowed": True,
        "provider_inference_allowed": False,
        "memory_read_allowed": False,
        "memory_write_allowed": False,
        "manipulation_progression_allowed": False,
    }:
        raise ValueError("G0C execution safety scope 漂移")
    return config


def _source_identity(repository_root: Path) -> dict[str, Any]:
    safe_repository = str(repository_root.resolve())
    git = ("git", "-c", f"safe.directory={safe_repository}")
    hashes = {
        relative: _g0._file_sha256(repository_root / relative)
        for relative in _SOURCE_FILES
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
    identity["identity_sha256"] = _g0._canonical_sha256(identity)
    return identity


def _run_simulator(
    *,
    config: dict[str, Any],
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import gymnasium as gym
    import mani_skill
    import sapien
    import torch
    from mani_skill.utils import sapien_utils

    from robot_vla.sim import register_robot_vla_maniskill_envs

    software = config["software"]
    if mani_skill.__version__ != software["expected_mani_skill_version"]:
        raise RuntimeError(
            f"ManiSkill version 漂移: {mani_skill.__version__} != "
            f"{software['expected_mani_skill_version']}"
        )
    if sapien.__version__ != software["expected_sapien_version"]:
        raise RuntimeError(
            f"SAPIEN version 漂移: {sapien.__version__} != "
            f"{software['expected_sapien_version']}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("E018-P1 G0C config 要求 CUDA")

    home, anchors, orientations = _parse_library(config)
    primitives = _expand_primitives(anchors, orientations)
    environment = config["environment"]
    register_robot_vla_maniskill_envs()
    env = gym.make(
        environment["environment_id"],
        obs_mode=environment["obs_mode"],
        control_mode=environment["control_mode"],
        num_envs=environment["num_envs"],
        robot_uids=environment["robot_uid"],
    )
    all_rows: list[dict[str, Any]] = []
    route_summaries: list[dict[str, Any]] = []
    contact_sheet_images: dict[tuple[int, str, str], np.ndarray] = {}
    try:
        base_env = env.unwrapped
        if base_env.control_freq != environment["control_hz"]:
            raise RuntimeError(
                f"ManiSkill control_hz 漂移: {base_env.control_freq} != "
                f"{environment['control_hz']}"
            )
        if environment["camera_uid"] not in base_env._sensors:
            raise RuntimeError("G0C external camera uid 不存在")
        sensor = base_env._sensors[environment["camera_uid"]]
        camera = sensor.camera
        if sensor.entity is not None or not callable(
            getattr(camera, "set_local_pose", None)
        ):
            raise RuntimeError("G0C 要求独立、可设位姿的 unmounted external camera")
        for seed in config["experiment"]["seeds"]:
            for primitive, orientation in primitives:
                rows, summary, route_images = _g0._run_route(
                    env=env,
                    base_env=base_env,
                    camera=camera,
                    config=config,
                    seed=seed,
                    home=home,
                    alternate=primitive,
                    output_root=output_root,
                    sapien_module=sapien,
                    sapien_utils_module=sapien_utils,
                    alternate_orientation=orientation,
                    result_version=E018_P1_G0C_RESULT_VERSION,
                    episode_prefix="g0c",
                    source_phase="G0C_ROTATED_MOTION_NO_EXECUTIVE_PHASE",
                    camera_owner="ACTIVE_REOBSERVE_G0C_PROBE",
                )
                all_rows.extend(rows)
                route_summaries.append(summary)
                contact_sheet_images[
                    (seed, primitive.viewpoint_id, "alternate")
                ] = route_images["alternate"]
                contact_sheet_images.setdefault(
                    (seed, "HOME", "home_before"),
                    route_images["home_before"],
                )
    finally:
        env.close()

    primitive_ids = [item.viewpoint_id for item, _ in primitives]
    _g0._write_contact_sheet(
        output_root / "viewpoint_contact_sheet.png",
        seeds=list(config["experiment"]["seeds"]),
        alternate_ids=primitive_ids,
        images=contact_sheet_images,
    )
    environment_identity = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(torch.device("cuda")),
        "mani_skill": mani_skill.__version__,
        "sapien": sapien.__version__,
        "external_camera_sensor_class": (
            type(sensor).__module__ + "." + type(sensor).__name__
        ),
        "external_camera_class": (
            type(camera).__module__ + "." + type(camera).__name__
        ),
        "external_camera_unmounted": sensor.entity is None,
        "set_local_pose_callable": callable(getattr(camera, "set_local_pose", None)),
        "actual_pose_source": "observation.sensor_param.base_camera.cam2world_gl",
    }
    return all_rows, route_summaries, environment_identity


def run_e018_p1_g0c(
    *,
    config_path: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """运行 G0C development motion gate；不接受 test/provider/Memory 参数。"""

    config_file = Path(config_path)
    repository = Path(repository_root)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"E018-P1 G0C output 已存在: {output}")
    config = load_e018_p1_g0c_config(config_file)
    config_sha256 = _g0._canonical_sha256(config)
    source_identity = _source_identity(repository)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _g0._atomic_json(
        output / "run_state.json",
        {
            "version": E018_P1_G0C_RESULT_VERSION,
            "status": "in-progress-development-only",
            "gate": E018_P1_G0C_GATE,
            "test_split_status": "prohibited-unread",
            "formal_claim_allowed": False,
        },
    )
    try:
        frame_rows, route_summaries, environment_identity = _run_simulator(
            config=config,
            output_root=output,
        )
        expected_route_count = (
            len(config["experiment"]["seeds"])
            * config["viewpoint_library"]["expected_primitive_count"]
        )
        route_count = len(route_summaries)
        all_routes_passed = bool(route_summaries) and all(
            bool(route["passed"]) for route in route_summaries
        )
        gate_passed = all_routes_passed and route_count == expected_route_count
        aggregate = {
            "max_camera_position_tracking_error_m": max(
                row["external_position_tracking_error_m"] for row in frame_rows
            ),
            "max_camera_orientation_tracking_error_rad": max(
                row["external_orientation_tracking_error_rad"] for row in frame_rows
            ),
            "max_camera_linear_velocity_m_s": max(
                row["external_linear_speed_m_s"] for row in frame_rows
            ),
            "max_camera_linear_acceleration_m_s2": max(
                row["external_linear_acceleration_m_s2"] for row in frame_rows
            ),
            "max_camera_angular_velocity_rad_s": max(
                row["external_angular_speed_rad_s"] for row in frame_rows
            ),
            "max_camera_angular_acceleration_rad_s2": max(
                row["external_angular_acceleration_rad_s2"] for row in frame_rows
            ),
            "max_arm_joint_drift_rad": max(
                row["arm_joint_max_drift_rad"] for row in frame_rows
            ),
            "max_tcp_position_drift_m": max(
                row["tcp_position_drift_m"] for row in frame_rows
            ),
            "max_tcp_orientation_drift_rad": max(
                row["tcp_orientation_drift_rad"] for row in frame_rows
            ),
            "max_finger_object_contact_force_n": max(
                row["finger_object_contact_force_n"] for row in frame_rows
            ),
            "motion_or_unsettled_write_eligible_count": sum(
                bool(row["measurement_write_eligible"])
                for row in frame_rows
                if row["camera_motion_state"]
                != ExternalCameraMotionState.COLLECT.value
            ),
            "memory_write_count": sum(
                bool(row["memory_write_executed"]) for row in frame_rows
            ),
            "terminated_or_truncated_count": sum(
                bool(row["terminated"] or row["truncated"]) for row in frame_rows
            ),
        }
        summary = {
            "version": E018_P1_G0C_RESULT_VERSION,
            "status": "complete-development-only",
            "gate": E018_P1_G0C_GATE,
            "gate_passed": gate_passed,
            "config_sha256": config_sha256,
            "parent_screen_identity": config["parent_screen"],
            "viewpoint_library_version": config["viewpoint_library"]["version"],
            "viewpoint_library_status": config["viewpoint_library"]["status"],
            "source_identity": source_identity,
            "environment_identity": environment_identity,
            "seed_count": len(config["experiment"]["seeds"]),
            "translation_anchor_count": config["viewpoint_library"][
                "expected_anchor_count"
            ],
            "orientation_count": config["viewpoint_library"][
                "expected_orientation_count"
            ],
            "primitive_count": config["viewpoint_library"][
                "expected_primitive_count"
            ],
            "route_count": route_count,
            "expected_route_count": expected_route_count,
            "passed_route_count": sum(
                bool(route["passed"]) for route in route_summaries
            ),
            "frame_count": len(frame_rows),
            "failed_routes": [
                route["episode_id"]
                for route in route_summaries
                if not route["passed"]
            ],
            "aggregate": aggregate,
            "scope_limits": {
                "proves": [
                    (
                        "ten low-anchor discrete position/orientation poses can be "
                        "reached over nonzero simulator control ticks"
                    ),
                    (
                        "position and local yaw/pitch offsets share a smooth quintic "
                        "command schedule"
                    ),
                    (
                        "actual per-frame camera pose, velocity, acceleration, arm/TCP "
                        "hold, contact, settle, and HOME return are auditable"
                    ),
                ],
                "does_not_prove": [
                    "physical camera actuator dynamics, backlash, calibration, or safety",
                    "camera collision-envelope safety",
                    "front perception provider qualification",
                    "runtime viewpoint selection or closed-loop recovery",
                    "active versus passive information-recovery benefit",
                ],
            },
            "ready_for_g1_observation_schema": gate_passed,
            "ready_for_formal_preregistration": False,
            "test_split_status": "prohibited-unread",
            "test_episode_count": 0,
            "provider_forward_count": 0,
            "memory_read_count": 0,
            "memory_write_count": 0,
            "physical_robot_actuation_allowed": False,
            "manipulation_progression_allowed": False,
            "formal_claim_allowed": False,
        }
        _g0._atomic_jsonl(output / "camera_pose_ledger.jsonl", frame_rows)
        _g0._atomic_jsonl(output / "route_summaries.jsonl", route_summaries)
        _g0._atomic_json(output / "config_snapshot.json", config)
        _g0._atomic_json(output / "summary.json", summary)
        artifact_paths = sorted(
            [
                output / "camera_pose_ledger.jsonl",
                output / "route_summaries.jsonl",
                output / "config_snapshot.json",
                output / "summary.json",
                output / "viewpoint_contact_sheet.png",
                *(output / "images").glob("*.png"),
            ],
            key=lambda path: str(path.relative_to(output)),
        )
        receipt = {
            "version": E018_P1_G0C_RESULT_VERSION,
            "status": "complete-development-only",
            "gate_passed": gate_passed,
            "config_sha256": config_sha256,
            "files": {
                str(path.relative_to(output)): _g0._file_sha256(path)
                for path in artifact_paths
            },
            "test_split_status": "prohibited-unread",
            "formal_claim_allowed": False,
        }
        receipt["receipt_sha256"] = _g0._canonical_sha256(receipt)
        _g0._atomic_json(output / "receipt.json", receipt)
        _g0._atomic_json(
            output / "run_state.json",
            {
                "version": E018_P1_G0C_RESULT_VERSION,
                "status": "complete-development-only",
                "gate": E018_P1_G0C_GATE,
                "gate_passed": gate_passed,
                "test_split_status": "prohibited-unread",
                "formal_claim_allowed": False,
            },
        )
        return summary
    except Exception as error:
        _g0._atomic_json(
            output / "run_state.json",
            {
                "version": E018_P1_G0C_RESULT_VERSION,
                "status": "failed-development-only",
                "gate": E018_P1_G0C_GATE,
                "error_type": type(error).__name__,
                "error": str(error),
                "test_split_status": "prohibited-unread",
                "formal_claim_allowed": False,
            },
        )
        raise


__all__ = [
    "E018_P1_G0C_CONFIG_VERSION",
    "E018_P1_G0C_GATE",
    "E018_P1_G0C_LIBRARY_VERSION",
    "E018_P1_G0C_RESULT_VERSION",
    "load_e018_p1_g0c_config",
    "run_e018_p1_g0c",
]
