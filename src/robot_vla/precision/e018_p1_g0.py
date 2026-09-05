"""E018-P1 G0：独立 front camera 的仿真 API/时序运动可行性实验。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from robot_vla.observation import invert_se3, opengl_camera_to_opencv, validate_se3
from robot_vla.precision.active_front_camera import (
    ExternalCameraMotionState,
    FrontCameraOrientationMode,
    FrontCameraViewpoint,
    compose_camera_orientation_wxyz,
    measurement_write_eligible,
    quaternion_angular_distance_rad,
    rotation_angular_distance_rad,
    sample_translation_path,
    smootherstep,
)

E018_P1_G0_CONFIG_VERSION = "e018-p1-g0-camera-feasibility-development/v1"
E018_P1_G0_RESULT_VERSION = "e018-p1-g0-camera-feasibility-result/v1"
E018_P1_G0_VIEWPOINT_LIBRARY_VERSION = "e018-p1-front-home-2x2-provisional/v1"

_VIEWPOINT_KEYS = {
    "viewpoint_id",
    "lateral_anchor",
    "vertical_anchor",
    "position_world_m",
    "look_at_world_m",
    "yaw_rad",
    "pitch_rad",
    "roll_rad",
}
_SOURCE_FILES = (
    "src/robot_vla/precision/active_front_camera.py",
    "src/robot_vla/precision/e018_p1_g0.py",
    "src/robot_vla/cli/run_e018_p1_g0.py",
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


def _parse_viewpoint(value: Any, name: str) -> FrontCameraViewpoint:
    item = _require_keys(value, _VIEWPOINT_KEYS, name)
    viewpoint = FrontCameraViewpoint(
        viewpoint_id=str(item["viewpoint_id"]),
        lateral_anchor=str(item["lateral_anchor"]),
        vertical_anchor=str(item["vertical_anchor"]),
        position_world_m=tuple(float(entry) for entry in item["position_world_m"]),
        look_at_world_m=tuple(float(entry) for entry in item["look_at_world_m"]),
        yaw_rad=float(item["yaw_rad"]),
        pitch_rad=float(item["pitch_rad"]),
        roll_rad=float(item["roll_rad"]),
    )
    viewpoint.validate()
    return viewpoint


def load_e018_p1_g0_config(path: str | Path) -> dict[str, Any]:
    """严格读取 development-only G0 config；函数没有 test/formal 开关。"""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"E018-P1 G0 config 不存在: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = _require_keys(
        config,
        {
            "version",
            "status",
            "scope",
            "software",
            "environment",
            "viewpoint_library",
            "motion",
            "safety",
            "experiment",
            "execution",
        },
        "E018-P1 G0 config",
    )
    if config["version"] != E018_P1_G0_CONFIG_VERSION:
        raise ValueError("E018-P1 G0 config version 漂移")
    if config["status"] != "development-only-g0-no-formal-claim":
        raise ValueError("E018-P1 G0 只能以 development-only 状态运行")

    scope = _require_keys(
        config["scope"],
        {
            "gate",
            "test_split_allowed",
            "formal_claim_allowed",
            "runtime_gt_control_allowed",
            "p0_rules_consumed",
            "provider_inference_allowed",
        },
        "scope",
    )
    if scope != {
        "gate": "G0_SIMULATOR_API_FEASIBILITY",
        "test_split_allowed": False,
        "formal_claim_allowed": False,
        "runtime_gt_control_allowed": False,
        "p0_rules_consumed": False,
        "provider_inference_allowed": False,
    }:
        raise ValueError("G0 scope 必须禁止 test/formal/GT control/provider/P0 runtime")

    software = _require_keys(
        config["software"],
        {"expected_mani_skill_version", "expected_sapien_version"},
        "software",
    )
    if software["expected_mani_skill_version"] != "3.0.1":
        raise ValueError("G0 当前只验证 ManiSkill 3.0.1")
    if software["expected_sapien_version"] != "3.0.3":
        raise ValueError("G0 当前只验证 SAPIEN 3.0.3")

    environment = _require_keys(
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
    expected_environment = {
        "environment_id": "RobotVLAPickCubeToRegion-v1",
        "robot_uid": "panda_wristcam",
        "camera_uid": "base_camera",
        "obs_mode": "rgb+segmentation",
        "control_mode": "pd_joint_delta_pos",
        "num_envs": 1,
        "control_hz": 20,
    }
    if environment != expected_environment:
        raise ValueError("G0 environment identity 漂移")

    library = _require_keys(
        config["viewpoint_library"],
        {
            "version",
            "status",
            "pose_frame",
            "orientation_policy",
            "route_policy",
            "look_at_up_world",
            "home",
            "alternates",
        },
        "viewpoint_library",
    )
    if (
        library["version"] != E018_P1_G0_VIEWPOINT_LIBRARY_VERSION
        or library["status"] != "provisional-development-not-frozen"
        or library["pose_frame"] != "world"
        or library["orientation_policy"] != "fixed-workspace-look-at-yaw-pitch-roll-zero/v1"
        or library["route_policy"] != "HOME_TO_ONE_ALTERNATE_TO_HOME/v1"
        or library["look_at_up_world"] != [0.0, 0.0, 1.0]
    ):
        raise ValueError("G0 viewpoint library identity/semantics 漂移")
    home = _parse_viewpoint(library["home"], "viewpoint_library.home")
    if (
        home.viewpoint_id != "HOME"
        or home.lateral_anchor != "CENTER"
        or home.vertical_anchor != "CENTER"
    ):
        raise ValueError("G0 HOME anchor 漂移")
    alternates_value = library["alternates"]
    if not isinstance(alternates_value, list) or len(alternates_value) != 4:
        raise ValueError("G0 必须有 2x2 共四个 alternate anchors")
    alternates = [
        _parse_viewpoint(item, f"viewpoint_library.alternates[{index}]")
        for index, item in enumerate(alternates_value)
    ]
    expected_anchors = {
        ("LEFT", "LOW"),
        ("LEFT", "HIGH"),
        ("RIGHT", "LOW"),
        ("RIGHT", "HIGH"),
    }
    if {(item.lateral_anchor, item.vertical_anchor) for item in alternates} != expected_anchors:
        raise ValueError("G0 alternate anchors 必须恰好是 LEFT/RIGHT x LOW/HIGH")
    if len({item.viewpoint_id for item in alternates}) != len(alternates):
        raise ValueError("G0 alternate viewpoint_id 必须唯一")
    if any(item.look_at_world_m != home.look_at_world_m for item in alternates):
        raise ValueError("G0 所有视角必须使用同一个冻结 workspace look-at target")

    motion = _require_keys(
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
        raise ValueError("G0 camera command mode 漂移")
    if motion["interpolation"] != "quintic-smootherstep-position-plus-fixed-look-at/v1":
        raise ValueError("G0 interpolation 漂移")
    for name in ("warmup_ticks", "settle_ticks", "collect_ticks"):
        _positive_int(motion[name], f"motion.{name}")
    move_duration = _positive(motion["move_duration_s"], "motion.move_duration_s")
    move_steps = move_duration * environment["control_hz"]
    if not math.isclose(move_steps, round(move_steps), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("move_duration_s * control_hz 必须是整数 Tick")
    if round(move_steps) <= 1:
        raise ValueError("G0 camera motion 必须跨越多个 Tick")
    for name in (
        "max_linear_velocity_m_s",
        "max_linear_acceleration_m_s2",
        "max_angular_velocity_rad_s",
        "max_angular_acceleration_rad_s2",
    ):
        _positive(motion[name], f"motion.{name}")

    safety = _require_keys(
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
    _positive_int(
        safety["required_consecutive_settled_ticks"],
        "safety.required_consecutive_settled_ticks",
    )
    if safety["required_consecutive_settled_ticks"] > motion["settle_ticks"]:
        raise ValueError("settle_ticks 不足以形成要求的连续 settled evidence")
    for name in set(safety) - {"required_consecutive_settled_ticks"}:
        _positive(safety[name], f"safety.{name}")

    experiment = _require_keys(
        config["experiment"],
        {
            "usage",
            "seeds",
            "execute_all_alternates",
            "save_settled_rgb",
            "offline_segmentation_diagnostics",
        },
        "experiment",
    )
    if experiment["usage"] != "simulator-development-only-no-test-no-provider-no-memory/v1":
        raise ValueError("G0 experiment usage 漂移")
    seeds = experiment["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("G0 seeds 必须是非空唯一非负整数列表")
    for name in (
        "execute_all_alternates",
        "save_settled_rgb",
        "offline_segmentation_diagnostics",
    ):
        if experiment[name] is not True:
            raise ValueError(f"experiment.{name} 必须为 true")

    execution = _require_keys(
        config["execution"],
        {
            "device",
            "physical_robot_actuation_allowed",
            "arm_motion_command_allowed",
            "gripper_command_mode",
            "simulated_external_camera_motion_allowed",
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
        "memory_write_allowed": False,
        "manipulation_progression_allowed": False,
    }:
        raise ValueError("G0 execution safety scope 漂移")
    return config


def _source_identity(repository_root: Path) -> dict[str, Any]:
    safe_repository = str(repository_root.resolve())
    git = ("git", "-c", f"safe.directory={safe_repository}")
    hashes = {relative: _file_sha256(repository_root / relative) for relative in _SOURCE_FILES}
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


def _single_bool(value: Any) -> bool:
    array = _numpy(value)
    if array.size != 1:
        raise RuntimeError(f"G0 只支持单环境 bool，实际 shape={array.shape}")
    return bool(array.reshape(-1)[0])


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


def _gate(actual: Any, required: str, passed: bool) -> dict[str, Any]:
    return {"actual": actual, "required": required, "passed": bool(passed)}


def recompute_route_gates(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    home_position_world_m: Sequence[float],
    home_quaternion_sapien: Sequence[float],
    alternate_orientation_id: str,
    requested_orientation_offset_rad: float,
    actual_orientation_offset_rad: float,
    alternate_target_orientation_error_rad: float,
    alternate_rgb_mean_abs_difference: float,
    return_home_rgb_mean_abs_difference: float,
) -> dict[str, dict[str, Any]]:
    """从 route ledger 与显式 RGB 证据重算 G0/G0C 的完整门禁。

    RGB 均值差仍来自运行期原始像素；其余运动、安全与状态门禁只依赖逐帧
    ledger 和冻结 config。函数不读取仿真器，也不修改输入，供 runner 与公开
    verifier 共用。
    """

    if not rows:
        raise ValueError("G0 route rows 不能为空")
    environment = config["environment"]
    motion = config["motion"]
    safety = config["safety"]
    control_period = 1.0 / float(environment["control_hz"])
    move_steps = round(float(motion["move_duration_s"]) * environment["control_hz"])
    collect_rows = [
        row for row in rows if row["camera_motion_state"] == ExternalCameraMotionState.COLLECT.value
    ]
    non_collect_rows = [
        row for row in rows if row["camera_motion_state"] != ExternalCameraMotionState.COLLECT.value
    ]
    if not collect_rows:
        raise ValueError("G0 route 缺少 COLLECT rows")

    home_position = np.asarray(home_position_world_m, dtype=np.float64)
    home_quaternion = np.asarray(home_quaternion_sapien, dtype=np.float64)
    initial_position = np.asarray(rows[0]["actual_external_position_world_m"], dtype=np.float64)
    alternate_actual_position = np.asarray(
        collect_rows[-1]["actual_external_position_world_m"], dtype=np.float64
    )
    final_position = np.asarray(rows[-1]["actual_external_position_world_m"], dtype=np.float64)
    actual_displacement = float(np.linalg.norm(alternate_actual_position - initial_position))
    final_home_position_error = float(np.linalg.norm(final_position - home_position))
    final_home_orientation_error = quaternion_angular_distance_rad(
        rows[-1]["actual_external_quaternion_sapien"],
        home_quaternion,
    )
    max_position_tracking = max(float(row["external_position_tracking_error_m"]) for row in rows)
    max_orientation_tracking = max(
        float(row["external_orientation_tracking_error_rad"]) for row in rows
    )
    max_linear_speed = max(float(row["external_linear_speed_m_s"]) for row in rows)
    max_linear_acceleration = max(float(row["external_linear_acceleration_m_s2"]) for row in rows)
    max_angular_speed = max(float(row["external_angular_speed_rad_s"]) for row in rows)
    max_angular_acceleration = max(
        float(row["external_angular_acceleration_rad_s2"]) for row in rows
    )
    max_arm_drift = max(float(row["arm_joint_max_drift_rad"]) for row in rows)
    max_tcp_position_drift = max(float(row["tcp_position_drift_m"]) for row in rows)
    max_tcp_orientation_drift = max(float(row["tcp_orientation_drift_rad"]) for row in rows)
    minimum_finger_position = min(float(row["minimum_finger_joint_position_m"]) for row in rows)
    max_finger_contact = max(float(row["finger_object_contact_force_n"]) for row in rows)
    motion_or_unsettled_write_eligible_count = sum(
        bool(row["measurement_write_eligible"]) for row in non_collect_rows
    )
    memory_write_count = sum(bool(row["memory_write_executed"]) for row in rows)
    terminated_or_truncated_count = sum(bool(row["terminated"] or row["truncated"]) for row in rows)
    collect_all_settled = all(
        bool(row["settled"] and row["measurement_write_eligible"]) for row in collect_rows
    )
    dynamic_pose_count = len(
        {
            tuple(round(float(value), 8) for value in row["actual_external_position_world_m"])
            for row in rows
        }
    )

    return {
        "nonzero_time_camera_motion": _gate(
            {
                "move_ticks_each_leg": move_steps,
                "move_duration_s_each_leg": move_steps * control_period,
            },
            "move_ticks_each_leg > 1 and duration > 0",
            move_steps > 1 and move_steps * control_period > 0.0,
        ),
        "actual_dynamic_pose_observed": _gate(
            {
                "unique_actual_positions": dynamic_pose_count,
                "alternate_displacement_m": actual_displacement,
            },
            "unique_actual_positions > 2 and displacement > 0.10 m",
            dynamic_pose_count > 2 and actual_displacement > 0.10,
        ),
        "alternate_orientation_target_reached": _gate(
            {
                "orientation_id": alternate_orientation_id,
                "requested_offset_rad": float(requested_orientation_offset_rad),
                "actual_offset_rad": float(actual_orientation_offset_rad),
                "target_error_rad": float(alternate_target_orientation_error_rad),
            },
            (
                "actual endpoint matches nominal look-at plus registered local offset; "
                f"error <= {safety['camera_orientation_tracking_tolerance_rad']} rad"
            ),
            float(alternate_target_orientation_error_rad)
            <= safety["camera_orientation_tracking_tolerance_rad"],
        ),
        "commanded_actual_tracking": _gate(
            {
                "max_position_error_m": max_position_tracking,
                "max_orientation_error_rad": max_orientation_tracking,
            },
            (
                f"position <= {safety['camera_position_tracking_tolerance_m']} m; "
                f"orientation <= {safety['camera_orientation_tracking_tolerance_rad']} rad"
            ),
            max_position_tracking <= safety["camera_position_tracking_tolerance_m"]
            and max_orientation_tracking <= safety["camera_orientation_tracking_tolerance_rad"],
        ),
        "camera_velocity_limits": _gate(
            {
                "max_linear_velocity_m_s": max_linear_speed,
                "max_angular_velocity_rad_s": max_angular_speed,
            },
            (
                f"linear <= {motion['max_linear_velocity_m_s']} m/s; "
                f"angular <= {motion['max_angular_velocity_rad_s']} rad/s"
            ),
            max_linear_speed <= motion["max_linear_velocity_m_s"]
            and max_angular_speed <= motion["max_angular_velocity_rad_s"],
        ),
        "camera_acceleration_limits": _gate(
            {
                "max_linear_acceleration_m_s2": max_linear_acceleration,
                "max_angular_acceleration_rad_s2": max_angular_acceleration,
            },
            (
                f"linear <= {motion['max_linear_acceleration_m_s2']} m/s^2; "
                f"angular <= {motion['max_angular_acceleration_rad_s2']} rad/s^2"
            ),
            max_linear_acceleration <= motion["max_linear_acceleration_m_s2"]
            and max_angular_acceleration <= motion["max_angular_acceleration_rad_s2"],
        ),
        "settled_collection_window": _gate(
            {
                "collect_frames": len(collect_rows),
                "eligible_collect_frames": sum(
                    bool(row["measurement_write_eligible"]) for row in collect_rows
                ),
            },
            f"all {motion['collect_ticks']} collect frames settled and eligible",
            len(collect_rows) == motion["collect_ticks"] and collect_all_settled,
        ),
        "return_home_pose": _gate(
            {
                "position_error_m": final_home_position_error,
                "orientation_error_rad": final_home_orientation_error,
                "final_settled": rows[-1]["settled"],
            },
            (
                f"position <= {safety['home_position_tolerance_m']} m; orientation <= "
                f"{safety['home_orientation_tolerance_rad']} rad; settled"
            ),
            final_home_position_error <= safety["home_position_tolerance_m"]
            and final_home_orientation_error <= safety["home_orientation_tolerance_rad"]
            and bool(rows[-1]["settled"]),
        ),
        "rendered_view_changed": _gate(
            float(alternate_rgb_mean_abs_difference),
            f">= {safety['minimum_alternate_rgb_mean_abs_difference']} mean |RGB diff|",
            float(alternate_rgb_mean_abs_difference)
            >= safety["minimum_alternate_rgb_mean_abs_difference"],
        ),
        "return_home_render_recovered": _gate(
            float(return_home_rgb_mean_abs_difference),
            f"<= {safety['maximum_return_home_rgb_mean_abs_difference']} mean |RGB diff|",
            float(return_home_rgb_mean_abs_difference)
            <= safety["maximum_return_home_rgb_mean_abs_difference"],
        ),
        "arm_joint_hold": _gate(
            max_arm_drift,
            f"<= {safety['arm_joint_drift_max_rad']} rad",
            max_arm_drift <= safety["arm_joint_drift_max_rad"],
        ),
        "tcp_hold": _gate(
            {
                "max_position_drift_m": max_tcp_position_drift,
                "max_orientation_drift_rad": max_tcp_orientation_drift,
            },
            (
                f"position <= {safety['tcp_position_drift_max_m']} m; orientation <= "
                f"{safety['tcp_orientation_drift_max_rad']} rad"
            ),
            max_tcp_position_drift <= safety["tcp_position_drift_max_m"]
            and max_tcp_orientation_drift <= safety["tcp_orientation_drift_max_rad"],
        ),
        "gripper_safe_hold_open": _gate(
            minimum_finger_position,
            f">= {safety['minimum_open_finger_joint_position_m']} m",
            minimum_finger_position >= safety["minimum_open_finger_joint_position_m"],
        ),
        "unexpected_contact_absent": _gate(
            max_finger_contact,
            f"<= {safety['unexpected_finger_contact_max_n']} N",
            max_finger_contact <= safety["unexpected_finger_contact_max_n"],
        ),
        "non_collect_frame_write_invalidation": _gate(
            motion_or_unsettled_write_eligible_count,
            "0 non-COLLECT frames eligible",
            motion_or_unsettled_write_eligible_count == 0,
        ),
        "memory_write_disabled": _gate(
            memory_write_count,
            "0 writes",
            memory_write_count == 0,
        ),
        "episode_remained_nonterminal": _gate(
            terminated_or_truncated_count,
            "0 terminated/truncated frames",
            terminated_or_truncated_count == 0,
        ),
        "runtime_gt_control_dependency_absent": _gate(
            0,
            "0; segmentation is post-command offline diagnostic only",
            True,
        ),
    }


def _set_camera_pose(
    camera: Any,
    viewpoint: FrontCameraViewpoint,
    position_world_m: np.ndarray,
    *,
    orientation: FrontCameraOrientationMode | None = None,
    sapien_module: Any,
    sapien_utils_module: Any,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(position_world_m, dtype=np.float64)
    target = np.asarray(viewpoint.look_at_world_m, dtype=np.float64)
    look_at_pose = sapien_utils_module.look_at(
        position,
        target,
        up=(0.0, 0.0, 1.0),
    )
    quaternion = _single_vector(look_at_pose.q, 4, "commanded camera quaternion")
    if orientation is not None:
        quaternion = compose_camera_orientation_wxyz(quaternion, orientation)
    command_pose = sapien_module.Pose(p=position, q=quaternion)
    camera.set_local_pose(command_pose)
    # ManiSkill RenderCamera.get_model_matrix 使用同一固定 GL->ROS 轴变换。
    gl_from_mount = sapien_module.Pose(q=[-0.5, -0.5, 0.5, 0.5])
    world_from_camera_gl = validate_se3(
        (command_pose * gl_from_mount).to_transformation_matrix(),
        "commanded_world_from_external_camera_gl",
    )
    return quaternion, world_from_camera_gl


def _step_hold_open(env: Any, action_shape: tuple[int, ...]) -> tuple[dict[str, Any], bool, bool]:
    if action_shape != (8,):
        raise RuntimeError(f"G0 期望 Panda action shape=(8,)，实际 {action_shape}")
    action = np.zeros(action_shape, dtype=np.float32)
    action[-1] = 1.0
    observation, _, terminated, truncated, _ = env.step(action)
    return observation, _single_bool(terminated), _single_bool(truncated)


def _read_finger_contact_force_pair_n(base_env: Any) -> tuple[float, float]:
    scene = base_env.scene
    agent = base_env.agent
    cube = base_env.cube
    left = _numpy(scene.get_pairwise_contact_forces(agent.finger1_link, cube))[0]
    right = _numpy(scene.get_pairwise_contact_forces(agent.finger2_link, cube))[0]
    return float(np.linalg.norm(left)), float(np.linalg.norm(right))


def _read_finger_contact_force_n(base_env: Any) -> float:
    return max(_read_finger_contact_force_pair_n(base_env))


def _read_robot_object_contact_witness(
    base_env: Any,
) -> tuple[float, list[dict[str, Any]]]:
    """读取全 robot-link 对 object 的接触力；仅供显式 qualification witness。"""

    witnesses: list[dict[str, Any]] = []
    maximum = 0.0
    for link in base_env.agent.robot.links:
        force = _numpy(base_env.scene.get_pairwise_contact_forces(link, base_env.cube))
        if force.shape != (1, 3) or not np.isfinite(force).all():
            raise RuntimeError("G0 robot-object contact force schema 漂移")
        vector = np.asarray(force[0], dtype=np.float64)
        magnitude = float(np.linalg.norm(vector))
        maximum = max(maximum, magnitude)
        witnesses.append(
            {
                "link_name": str(link.name),
                "force_xyz_n": vector.tolist(),
                "force_magnitude_n": magnitude,
            }
        )
    return maximum, witnesses


def _offline_actor_ids(
    base_env: Any,
    *,
    enabled: bool,
) -> tuple[int | None, int | None]:
    """只在显式 offline diagnostic 路径读取 privileged actor id。"""

    if not enabled:
        return None, None
    return (
        int(_numpy(base_env.cube.per_scene_id).reshape(-1)[0]),
        int(_numpy(base_env.goal_site.per_scene_id).reshape(-1)[0]),
    )


def _offline_segmentation_diagnostics(
    sensor: Any,
    *,
    enabled: bool,
    object_actor_id: int | None,
    goal_actor_id: int | None,
) -> dict[str, Any] | None:
    """隔离 oracle-only segmentation；disabled 分支不索引 sensor。"""

    if not enabled:
        return None
    if object_actor_id is None or goal_actor_id is None:
        raise RuntimeError("G0 offline segmentation diagnostics 缺 actor id")
    segmentation = _numpy(sensor["segmentation"])
    if segmentation.ndim != 4 or segmentation.shape[0] != 1:
        raise RuntimeError("G0 external segmentation 必须是 [1,H,W,C]")
    actor_ids = segmentation[0, ..., 0]
    return {
        "oracle_only": True,
        "used_by_runtime_control": False,
        "object_visible_pixel_count": int(np.count_nonzero(actor_ids == object_actor_id)),
        "goal_visible_pixel_count": int(np.count_nonzero(actor_ids == goal_actor_id)),
    }


def _raw_safety_witness_fields(
    *,
    enabled: bool,
    arm_anchor_q_rad: np.ndarray,
    arm_current_q_rad: np.ndarray,
    tcp_anchor_world: np.ndarray,
    tcp_current_world: np.ndarray,
    world_from_robot_base: np.ndarray,
    finger_joint_positions_m: np.ndarray,
) -> dict[str, Any]:
    """qualification-only raw witness；默认关闭时返回严格空字典。"""

    if not enabled:
        return {}
    return {
        "arm_anchor_q_rad": np.asarray(arm_anchor_q_rad, dtype=np.float64).tolist(),
        "arm_current_q_rad": np.asarray(arm_current_q_rad, dtype=np.float64).tolist(),
        "tcp_anchor_world": np.asarray(tcp_anchor_world, dtype=np.float64).tolist(),
        "tcp_current_world": np.asarray(tcp_current_world, dtype=np.float64).tolist(),
        "world_from_robot_base": np.asarray(world_from_robot_base, dtype=np.float64).tolist(),
        "finger_joint_positions_m": np.asarray(finger_joint_positions_m, dtype=np.float64).tolist(),
    }


def _record_frame(
    *,
    episode_id: str,
    request_id: str,
    command_sequence_id: str,
    frame_index: int,
    control_tick: int,
    timestamp_s: float,
    state: ExternalCameraMotionState,
    viewpoint: FrontCameraViewpoint,
    orientation: FrontCameraOrientationMode,
    orientation_progress: float,
    commanded_position_world_m: np.ndarray,
    commanded_quaternion: np.ndarray,
    commanded_world_from_camera_gl: np.ndarray,
    observation: dict[str, Any],
    camera: Any,
    base_env: Any,
    arm_anchor_q_rad: np.ndarray,
    tcp_anchor_world: np.ndarray,
    previous_camera_position_world_m: np.ndarray | None,
    previous_camera_quaternion: np.ndarray | None,
    previous_linear_velocity_m_s: np.ndarray | None,
    previous_angular_speed_rad_s: float | None,
    control_period_s: float,
    config: dict[str, Any],
    object_actor_id: int | None,
    goal_actor_id: int | None,
    result_version: str,
    source_phase: str,
    camera_owner: str,
    include_raw_safety_witnesses: bool,
    include_privileged_object_state_witnesses: bool,
    include_robot_object_contact_witnesses: bool,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    sensor = observation["sensor_data"][config["environment"]["camera_uid"]]
    params = observation["sensor_param"][config["environment"]["camera_uid"]]
    rgb = _numpy(sensor["rgb"])
    if rgb.shape[0] != 1 or rgb.dtype != np.uint8:
        raise RuntimeError(f"G0 external RGB 必须是单环境 uint8，实际 {rgb.shape}/{rgb.dtype}")
    rgb = np.ascontiguousarray(rgb[0])

    actual_pose = camera.get_local_pose()
    actual_position = _single_vector(actual_pose.p, 3, "actual external camera position")
    actual_quaternion = _single_vector(actual_pose.q, 4, "actual external camera quaternion")
    actual_world_from_gl = _single_matrix(
        params["cam2world_gl"],
        "actual_world_from_external_camera_gl",
    )
    world_from_base = _single_matrix(base_env.agent.robot.pose, "world_from_robot_base")
    base_from_world = invert_se3(world_from_base, "world_from_robot_base")
    actual_base_from_cv = validate_se3(
        base_from_world @ opengl_camera_to_opencv(actual_world_from_gl),
        "actual_base_from_external_camera_cv",
    )
    commanded_base_from_cv = validate_se3(
        base_from_world @ opengl_camera_to_opencv(commanded_world_from_camera_gl),
        "commanded_base_from_external_camera_cv",
    )
    intrinsic = _numpy(params["intrinsic_cv"])
    if intrinsic.shape == (1, 3, 3):
        intrinsic = intrinsic[0]
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise RuntimeError("G0 external intrinsic_cv 无效")

    if previous_camera_position_world_m is None:
        linear_velocity = np.zeros(3, dtype=np.float64)
        angular_speed = 0.0
    else:
        linear_velocity = (actual_position - previous_camera_position_world_m) / control_period_s
        if previous_camera_quaternion is None:
            raise RuntimeError("G0 previous camera quaternion 缺失")
        angular_speed = (
            quaternion_angular_distance_rad(
                previous_camera_quaternion,
                actual_quaternion,
            )
            / control_period_s
        )
    if previous_linear_velocity_m_s is None:
        linear_acceleration = 0.0
    else:
        linear_acceleration = float(
            np.linalg.norm(linear_velocity - previous_linear_velocity_m_s) / control_period_s
        )
    angular_acceleration = (
        0.0
        if previous_angular_speed_rad_s is None
        else abs(angular_speed - previous_angular_speed_rad_s) / control_period_s
    )

    position_tracking_error = float(
        np.linalg.norm(actual_position - np.asarray(commanded_position_world_m))
    )
    orientation_tracking_error = quaternion_angular_distance_rad(
        commanded_quaternion,
        actual_quaternion,
    )
    arm_q = _numpy(base_env.agent.robot.get_qpos())[0]
    if arm_q.shape != (9,):
        raise RuntimeError(f"G0 Panda qpos 必须是 [9]，实际 {arm_q.shape}")
    tcp_world = _single_matrix(base_env.agent.tcp_pose, "actual_world_from_tcp")
    tcp_position_drift = float(np.linalg.norm(tcp_world[:3, 3] - tcp_anchor_world[:3, 3]))
    tcp_orientation_drift = rotation_angular_distance_rad(
        tcp_anchor_world[:3, :3],
        tcp_world[:3, :3],
    )
    finger_force_left, finger_force_right = _read_finger_contact_force_pair_n(base_env)
    finger_contact_force = max(finger_force_left, finger_force_right)
    robot_object_contact_force = None
    robot_object_contact_witness = None
    if include_raw_safety_witnesses and include_robot_object_contact_witnesses:
        (
            robot_object_contact_force,
            robot_object_contact_witness,
        ) = _read_robot_object_contact_witness(base_env)
    is_grasping = (
        _single_bool(base_env.agent.is_grasping(base_env.cube))
        if include_privileged_object_state_witnesses
        else None
    )

    offline_diagnostics = _offline_segmentation_diagnostics(
        sensor,
        enabled=config["experiment"]["offline_segmentation_diagnostics"],
        object_actor_id=object_actor_id,
        goal_actor_id=goal_actor_id,
    )

    linear_speed = float(np.linalg.norm(linear_velocity))
    safety = config["safety"]
    settle_evidence = bool(
        position_tracking_error <= safety["camera_position_tracking_tolerance_m"]
        and orientation_tracking_error <= safety["camera_orientation_tracking_tolerance_rad"]
        and linear_speed <= safety["settled_linear_velocity_max_m_s"]
        and angular_speed <= safety["settled_angular_velocity_max_rad_s"]
    )
    row = {
        "version": result_version,
        "episode_id": episode_id,
        "request_id": request_id,
        "camera_command_sequence_id": command_sequence_id,
        "frame_index": frame_index,
        "control_tick": control_tick,
        "timestamp_s": timestamp_s,
        "external_rgb_timestamp_s": timestamp_s,
        "external_pose_timestamp_s": timestamp_s,
        "timestamp_source": "synchronous-simulator-control-tick-derived/v1",
        "external_rgb_pose_skew_s": 0.0,
        "source_phase": source_phase,
        "camera_motion_state": state.value,
        "viewpoint_primitive_id": viewpoint.viewpoint_id,
        "target_orientation_id": orientation.orientation_id,
        "orientation_progress": float(orientation_progress),
        "commanded_yaw_offset_rad": orientation.yaw_offset_rad,
        "commanded_pitch_offset_rad": orientation.pitch_offset_rad,
        "commanded_roll_offset_rad": orientation.roll_offset_rad,
        "arm_owner": "SAFE_HOLD",
        "gripper_owner": "SAFE_HOLD_OPEN",
        "external_camera_owner": camera_owner,
        "arm_motion_command_max_abs": 0.0,
        "gripper_hold_open_command": 1.0,
        "commanded_external_position_world_m": commanded_position_world_m.tolist(),
        "commanded_external_quaternion_sapien": commanded_quaternion.tolist(),
        "actual_external_position_world_m": actual_position.tolist(),
        "actual_external_quaternion_sapien": actual_quaternion.tolist(),
        "commanded_world_from_external_camera_gl": commanded_world_from_camera_gl.tolist(),
        "actual_world_from_external_camera_gl": actual_world_from_gl.tolist(),
        "commanded_base_from_external_camera_cv": commanded_base_from_cv.tolist(),
        "actual_base_from_external_camera_cv": actual_base_from_cv.tolist(),
        "external_intrinsic_cv": np.asarray(intrinsic, dtype=np.float64).tolist(),
        "external_pose_valid": True,
        "external_position_tracking_error_m": position_tracking_error,
        "external_orientation_tracking_error_rad": orientation_tracking_error,
        "external_linear_velocity_m_s": linear_velocity.tolist(),
        "external_linear_speed_m_s": linear_speed,
        "external_linear_acceleration_m_s2": linear_acceleration,
        "external_angular_speed_rad_s": angular_speed,
        "external_angular_acceleration_rad_s2": angular_acceleration,
        "settle_evidence_passed": settle_evidence,
        "settle_streak": 0,
        "settled": False,
        "measurement_write_eligible": False,
        "memory_write_executed": False,
        "arm_joint_max_drift_rad": float(np.max(np.abs(arm_q[:7] - arm_anchor_q_rad))),
        "tcp_position_drift_m": tcp_position_drift,
        "tcp_orientation_drift_rad": tcp_orientation_drift,
        "minimum_finger_joint_position_m": float(np.min(arm_q[7:])),
        "finger_object_contact_force_n": finger_contact_force,
        "is_grasping": is_grasping,
        "terminated": False,
        "truncated": False,
        "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "offline_segmentation_diagnostics": offline_diagnostics,
    }
    row.update(
        _raw_safety_witness_fields(
            enabled=include_raw_safety_witnesses,
            arm_anchor_q_rad=arm_anchor_q_rad,
            arm_current_q_rad=arm_q[:7],
            tcp_anchor_world=tcp_anchor_world,
            tcp_current_world=tcp_world,
            world_from_robot_base=world_from_base,
            finger_joint_positions_m=arm_q[7:],
        )
    )
    if include_raw_safety_witnesses:
        row.update(
            {
                "finger_force_left_n": finger_force_left,
                "finger_force_right_n": finger_force_right,
                "robot_object_contact_force_n": robot_object_contact_force,
                "robot_object_contact_by_link": robot_object_contact_witness,
            }
        )
    return (
        row,
        rgb,
        actual_position,
        actual_quaternion,
        linear_velocity,
        angular_speed,
    )


def _save_png(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)
    os.chmod(path, 0o600)


def _run_route(
    *,
    env: Any,
    base_env: Any,
    camera: Any,
    config: dict[str, Any],
    seed: int,
    home: FrontCameraViewpoint,
    alternate: FrontCameraViewpoint,
    output_root: Path,
    sapien_module: Any,
    sapien_utils_module: Any,
    alternate_orientation: FrontCameraOrientationMode | None = None,
    result_version: str = E018_P1_G0_RESULT_VERSION,
    episode_prefix: str = "g0",
    source_phase: str = "G0_FEASIBILITY_NO_EXECUTIVE_PHASE",
    camera_owner: str = "ACTIVE_REOBSERVE_G0_PROBE",
    frame_hook: Callable[[dict[str, Any], np.ndarray, dict[str, Any]], None] | None = None,
    warmup_hook: Callable[[int, dict[str, Any]], None] | None = None,
    pre_command_hook: (
        Callable[[ExternalCameraMotionState, int, str], None] | None
    ) = None,
    episode_id_override: str | None = None,
    request_id_override: str | None = None,
    command_sequence_id_override: str | None = None,
    include_raw_safety_witnesses: bool = False,
    include_privileged_object_state_witnesses: bool = True,
    include_robot_object_contact_witnesses: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    environment = config["environment"]
    motion = config["motion"]
    safety = config["safety"]
    control_period = 1.0 / environment["control_hz"]
    move_steps = round(motion["move_duration_s"] * environment["control_hz"])
    center_orientation = FrontCameraOrientationMode(
        orientation_id="CENTER",
        yaw_offset_rad=0.0,
        pitch_offset_rad=0.0,
        roll_offset_rad=0.0,
    )
    target_orientation = alternate_orientation or center_orientation
    target_orientation.validate()

    def scaled_target_orientation(scale: float) -> FrontCameraOrientationMode:
        if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
            raise ValueError("orientation interpolation scale 必须位于 [0,1]")
        return FrontCameraOrientationMode(
            orientation_id=target_orientation.orientation_id,
            yaw_offset_rad=target_orientation.yaw_offset_rad * scale,
            pitch_offset_rad=target_orientation.pitch_offset_rad * scale,
            roll_offset_rad=target_orientation.roll_offset_rad * scale,
        )

    default_episode_id = (
        f"{episode_prefix}-seed-{seed:06d}-"
        f"{alternate.viewpoint_id.lower().replace('_', '-')}"
    )
    episode_id = episode_id_override or default_episode_id
    request_id = request_id_override or f"{episode_id}-request-00"
    command_sequence_id = (
        command_sequence_id_override or f"{episode_id}-camera-sequence-00"
    )
    if any(
        not isinstance(value, str) or not value
        for value in (episode_id, request_id, command_sequence_id)
    ):
        raise ValueError("G0 route identity override 必须是非空字符串")

    _, _ = env.reset(seed=seed)
    home_position = np.asarray(home.position_world_m, dtype=np.float64)
    home_quaternion, home_world_from_gl = _set_camera_pose(
        camera,
        home,
        home_position,
        orientation=center_orientation,
        sapien_module=sapien_module,
        sapien_utils_module=sapien_utils_module,
    )
    observation: dict[str, Any] | None = None
    terminated = False
    truncated = False
    for warmup_index in range(motion["warmup_ticks"]):
        home_quaternion, home_world_from_gl = _set_camera_pose(
            camera,
            home,
            home_position,
            orientation=center_orientation,
            sapien_module=sapien_module,
            sapien_utils_module=sapien_utils_module,
        )
        observation, terminated, truncated = _step_hold_open(env, env.action_space.shape)
        if warmup_hook is not None:
            warmup_hook(warmup_index, observation)
        if terminated or truncated:
            raise RuntimeError(f"{episode_id} warmup 期间环境提前结束")
    if observation is None:
        raise RuntimeError("G0 warmup 没有产生 observation")

    arm_q = _numpy(base_env.agent.robot.get_qpos())[0]
    arm_anchor_q = np.asarray(arm_q[:7], dtype=np.float64).copy()
    tcp_anchor = _single_matrix(base_env.agent.tcp_pose, "anchor_world_from_tcp")
    object_actor_id, goal_actor_id = _offline_actor_ids(
        base_env,
        enabled=config["experiment"]["offline_segmentation_diagnostics"],
    )

    rows: list[dict[str, Any]] = []
    images: dict[str, np.ndarray] = {}
    previous_position: np.ndarray | None = None
    previous_quaternion: np.ndarray | None = None
    previous_linear_velocity: np.ndarray | None = None
    previous_angular_speed: float | None = None
    control_tick = 0
    settle_streak = motion["warmup_ticks"]

    def append_observation(
        *,
        state: ExternalCameraMotionState,
        viewpoint: FrontCameraViewpoint,
        orientation: FrontCameraOrientationMode,
        orientation_progress: float,
        commanded_position: np.ndarray,
        commanded_quaternion: np.ndarray,
        commanded_world_from_gl: np.ndarray,
        current_observation: dict[str, Any],
        reset_settle_streak: bool = False,
    ) -> np.ndarray:
        nonlocal previous_position
        nonlocal previous_quaternion
        nonlocal previous_linear_velocity
        nonlocal previous_angular_speed
        nonlocal settle_streak
        if reset_settle_streak:
            settle_streak = 0
        result = _record_frame(
            episode_id=episode_id,
            request_id=request_id,
            command_sequence_id=command_sequence_id,
            frame_index=len(rows),
            control_tick=control_tick,
            timestamp_s=control_tick * control_period,
            state=state,
            viewpoint=viewpoint,
            orientation=orientation,
            orientation_progress=orientation_progress,
            commanded_position_world_m=commanded_position,
            commanded_quaternion=commanded_quaternion,
            commanded_world_from_camera_gl=commanded_world_from_gl,
            observation=current_observation,
            camera=camera,
            base_env=base_env,
            arm_anchor_q_rad=arm_anchor_q,
            tcp_anchor_world=tcp_anchor,
            previous_camera_position_world_m=previous_position,
            previous_camera_quaternion=previous_quaternion,
            previous_linear_velocity_m_s=previous_linear_velocity,
            previous_angular_speed_rad_s=previous_angular_speed,
            control_period_s=control_period,
            config=config,
            object_actor_id=object_actor_id,
            goal_actor_id=goal_actor_id,
            result_version=result_version,
            source_phase=source_phase,
            camera_owner=camera_owner,
            include_raw_safety_witnesses=include_raw_safety_witnesses,
            include_privileged_object_state_witnesses=(
                include_privileged_object_state_witnesses
            ),
            include_robot_object_contact_witnesses=(
                include_robot_object_contact_witnesses
            ),
        )
        row, rgb, position, quaternion, linear_velocity, angular_speed = result
        stationary_state = state in {
            ExternalCameraMotionState.HOME_ANCHOR,
            ExternalCameraMotionState.SETTLE_AT_VIEW,
            ExternalCameraMotionState.COLLECT,
            ExternalCameraMotionState.VERIFY_HOME_AND_ARM_HOLD,
        }
        if stationary_state and row["settle_evidence_passed"]:
            settle_streak += 1
        elif stationary_state:
            settle_streak = 0
        else:
            settle_streak = 0
        row["settle_streak"] = settle_streak
        row["settled"] = bool(settle_streak >= safety["required_consecutive_settled_ticks"])
        row["measurement_write_eligible"] = measurement_write_eligible(
            state,
            settled=row["settled"],
        )
        rows.append(row)
        if frame_hook is not None:
            frame_hook(row, rgb, current_observation)
        previous_position = position
        previous_quaternion = quaternion
        previous_linear_velocity = linear_velocity
        previous_angular_speed = angular_speed
        return rgb

    def authorize_camera_command(
        state: ExternalCameraMotionState,
        viewpoint: FrontCameraViewpoint,
    ) -> None:
        """实验专用 supervisor gate；默认 ``None`` 时保持旧 G0/G0C 行为。"""

        if pre_command_hook is not None:
            pre_command_hook(state, len(rows), viewpoint.viewpoint_id)

    home_before_rgb = append_observation(
        state=ExternalCameraMotionState.HOME_ANCHOR,
        viewpoint=home,
        orientation=center_orientation,
        orientation_progress=1.0,
        commanded_position=home_position,
        commanded_quaternion=home_quaternion,
        commanded_world_from_gl=home_world_from_gl,
        current_observation=observation,
    )
    images["home_before"] = home_before_rgb.copy()

    alternate_position = np.asarray(alternate.position_world_m, dtype=np.float64)
    outward_path = sample_translation_path(home_position, alternate_position, steps=move_steps)
    for index, position in enumerate(outward_path):
        control_tick += 1
        orientation_progress = smootherstep((index + 1) / move_steps)
        command_orientation = scaled_target_orientation(orientation_progress)
        authorize_camera_command(ExternalCameraMotionState.MOVE_TO_VIEW, alternate)
        command_q, command_gl = _set_camera_pose(
            camera,
            alternate,
            position,
            orientation=command_orientation,
            sapien_module=sapien_module,
            sapien_utils_module=sapien_utils_module,
        )
        observation, terminated, truncated = _step_hold_open(env, env.action_space.shape)
        rgb = append_observation(
            state=ExternalCameraMotionState.MOVE_TO_VIEW,
            viewpoint=alternate,
            orientation=command_orientation,
            orientation_progress=orientation_progress,
            commanded_position=position,
            commanded_quaternion=command_q,
            commanded_world_from_gl=command_gl,
            current_observation=observation,
            reset_settle_streak=index == 0,
        )
        del rgb
        rows[-1]["terminated"] = terminated
        rows[-1]["truncated"] = truncated
        if terminated or truncated:
            raise RuntimeError(f"{episode_id} outbound motion 期间环境提前结束")

    settle_streak = 0
    for _ in range(motion["settle_ticks"]):
        control_tick += 1
        authorize_camera_command(ExternalCameraMotionState.SETTLE_AT_VIEW, alternate)
        command_q, command_gl = _set_camera_pose(
            camera,
            alternate,
            alternate_position,
            orientation=target_orientation,
            sapien_module=sapien_module,
            sapien_utils_module=sapien_utils_module,
        )
        observation, terminated, truncated = _step_hold_open(env, env.action_space.shape)
        append_observation(
            state=ExternalCameraMotionState.SETTLE_AT_VIEW,
            viewpoint=alternate,
            orientation=target_orientation,
            orientation_progress=1.0,
            commanded_position=alternate_position,
            commanded_quaternion=command_q,
            commanded_world_from_gl=command_gl,
            current_observation=observation,
        )
        rows[-1]["terminated"] = terminated
        rows[-1]["truncated"] = truncated
        if terminated or truncated:
            raise RuntimeError(f"{episode_id} alternate settle 期间环境提前结束")

    alternate_rgb: np.ndarray | None = None
    for _ in range(motion["collect_ticks"]):
        control_tick += 1
        authorize_camera_command(ExternalCameraMotionState.COLLECT, alternate)
        command_q, command_gl = _set_camera_pose(
            camera,
            alternate,
            alternate_position,
            orientation=target_orientation,
            sapien_module=sapien_module,
            sapien_utils_module=sapien_utils_module,
        )
        observation, terminated, truncated = _step_hold_open(env, env.action_space.shape)
        alternate_rgb = append_observation(
            state=ExternalCameraMotionState.COLLECT,
            viewpoint=alternate,
            orientation=target_orientation,
            orientation_progress=1.0,
            commanded_position=alternate_position,
            commanded_quaternion=command_q,
            commanded_world_from_gl=command_gl,
            current_observation=observation,
        )
        rows[-1]["terminated"] = terminated
        rows[-1]["truncated"] = truncated
        if terminated or truncated:
            raise RuntimeError(f"{episode_id} collect 期间环境提前结束")
    if alternate_rgb is None:
        raise RuntimeError("G0 collect 没有产生 RGB")
    images["alternate"] = alternate_rgb.copy()

    return_path = sample_translation_path(alternate_position, home_position, steps=move_steps)
    for index, position in enumerate(return_path):
        control_tick += 1
        orientation_progress = 1.0 - smootherstep((index + 1) / move_steps)
        command_orientation = scaled_target_orientation(orientation_progress)
        authorize_camera_command(ExternalCameraMotionState.RETURN_HOME, home)
        command_q, command_gl = _set_camera_pose(
            camera,
            home,
            position,
            orientation=command_orientation,
            sapien_module=sapien_module,
            sapien_utils_module=sapien_utils_module,
        )
        observation, terminated, truncated = _step_hold_open(env, env.action_space.shape)
        append_observation(
            state=ExternalCameraMotionState.RETURN_HOME,
            viewpoint=home,
            orientation=command_orientation,
            orientation_progress=orientation_progress,
            commanded_position=position,
            commanded_quaternion=command_q,
            commanded_world_from_gl=command_gl,
            current_observation=observation,
            reset_settle_streak=index == 0,
        )
        rows[-1]["terminated"] = terminated
        rows[-1]["truncated"] = truncated
        if terminated or truncated:
            raise RuntimeError(f"{episode_id} return motion 期间环境提前结束")

    settle_streak = 0
    home_after_rgb: np.ndarray | None = None
    for _ in range(motion["settle_ticks"]):
        control_tick += 1
        authorize_camera_command(
            ExternalCameraMotionState.VERIFY_HOME_AND_ARM_HOLD,
            home,
        )
        command_q, command_gl = _set_camera_pose(
            camera,
            home,
            home_position,
            orientation=center_orientation,
            sapien_module=sapien_module,
            sapien_utils_module=sapien_utils_module,
        )
        observation, terminated, truncated = _step_hold_open(env, env.action_space.shape)
        home_after_rgb = append_observation(
            state=ExternalCameraMotionState.VERIFY_HOME_AND_ARM_HOLD,
            viewpoint=home,
            orientation=center_orientation,
            orientation_progress=1.0,
            commanded_position=home_position,
            commanded_quaternion=command_q,
            commanded_world_from_gl=command_gl,
            current_observation=observation,
        )
        rows[-1]["terminated"] = terminated
        rows[-1]["truncated"] = truncated
        if terminated or truncated:
            raise RuntimeError(f"{episode_id} HOME verify 期间环境提前结束")
    if home_after_rgb is None:
        raise RuntimeError("G0 HOME verify 没有产生 RGB")
    images["home_after"] = home_after_rgb.copy()

    image_root = output_root / "images"
    if config["experiment"]["save_settled_rgb"]:
        for role, rgb in images.items():
            _save_png(image_root / f"{episode_id}__{role}.png", rgb)

    collect_rows = [
        row for row in rows if row["camera_motion_state"] == ExternalCameraMotionState.COLLECT.value
    ]
    alternate_diff = float(
        np.mean(
            np.abs(
                images["alternate"].astype(np.float64) - images["home_before"].astype(np.float64)
            )
        )
    )
    return_diff = float(
        np.mean(
            np.abs(
                images["home_after"].astype(np.float64) - images["home_before"].astype(np.float64)
            )
        )
    )
    alternate_nominal_pose = sapien_utils_module.look_at(
        alternate_position,
        np.asarray(alternate.look_at_world_m, dtype=np.float64),
        up=(0.0, 0.0, 1.0),
    )
    alternate_nominal_quaternion = _single_vector(
        alternate_nominal_pose.q,
        4,
        "alternate nominal camera quaternion",
    )
    alternate_target_quaternion = compose_camera_orientation_wxyz(
        alternate_nominal_quaternion,
        target_orientation,
    )
    alternate_actual_quaternion = np.asarray(
        collect_rows[-1]["actual_external_quaternion_sapien"],
        dtype=np.float64,
    )
    requested_orientation_offset = quaternion_angular_distance_rad(
        alternate_nominal_quaternion,
        alternate_target_quaternion,
    )
    actual_orientation_offset = quaternion_angular_distance_rad(
        alternate_nominal_quaternion,
        alternate_actual_quaternion,
    )
    alternate_target_orientation_error = quaternion_angular_distance_rad(
        alternate_target_quaternion,
        alternate_actual_quaternion,
    )
    gates = recompute_route_gates(
        rows,
        config=config,
        home_position_world_m=home_position,
        home_quaternion_sapien=home_quaternion,
        alternate_orientation_id=target_orientation.orientation_id,
        requested_orientation_offset_rad=requested_orientation_offset,
        actual_orientation_offset_rad=actual_orientation_offset,
        alternate_target_orientation_error_rad=alternate_target_orientation_error,
        alternate_rgb_mean_abs_difference=alternate_diff,
        return_home_rgb_mean_abs_difference=return_diff,
    )
    actual_displacement = gates["actual_dynamic_pose_observed"]["actual"][
        "alternate_displacement_m"
    ]
    route_passed = all(item["passed"] for item in gates.values())
    object_pixels = [
        row["offline_segmentation_diagnostics"]["object_visible_pixel_count"]
        for row in collect_rows
        if row["offline_segmentation_diagnostics"] is not None
    ]
    goal_pixels = [
        row["offline_segmentation_diagnostics"]["goal_visible_pixel_count"]
        for row in collect_rows
        if row["offline_segmentation_diagnostics"] is not None
    ]
    route_summary = {
        "version": result_version,
        "episode_id": episode_id,
        "seed": seed,
        "alternate_viewpoint_id": alternate.viewpoint_id,
        "alternate_orientation_id": target_orientation.orientation_id,
        "yaw_offset_rad": target_orientation.yaw_offset_rad,
        "pitch_offset_rad": target_orientation.pitch_offset_rad,
        "roll_offset_rad": target_orientation.roll_offset_rad,
        "status": "passed" if route_passed else "failed",
        "passed": route_passed,
        "frame_count": len(rows),
        "control_hz": environment["control_hz"],
        "motion_ticks_each_leg": move_steps,
        "route_simulated_duration_s": control_tick * control_period,
        "gates": gates,
        "diagnostics": {
            "alternate_rgb_mean_abs_difference": alternate_diff,
            "return_home_rgb_mean_abs_difference": return_diff,
            "alternate_displacement_m": actual_displacement,
            "requested_orientation_offset_rad": requested_orientation_offset,
            "actual_orientation_offset_rad": actual_orientation_offset,
            "alternate_target_orientation_error_rad": alternate_target_orientation_error,
            "object_visible_pixels_collect_min": min(object_pixels) if object_pixels else None,
            "goal_visible_pixels_collect_min": min(goal_pixels) if goal_pixels else None,
        },
        "test_split_status": "prohibited-unread",
        "provider_forward_count": 0,
        "memory_write_count": 0,
        "formal_claim_allowed": False,
    }
    return rows, route_summary, images


def _write_contact_sheet(
    path: Path,
    *,
    seeds: list[int],
    alternate_ids: list[str],
    images: dict[tuple[int, str, str], np.ndarray],
) -> None:
    from PIL import Image, ImageDraw

    roles = ["HOME", *alternate_ids]
    first = next(iter(images.values()))
    height, width = first.shape[:2]
    label_height = 20
    sheet = Image.new(
        "RGB",
        (width * len(roles), (height + label_height) * len(seeds)),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(sheet)
    for row_index, seed in enumerate(seeds):
        for column_index, role in enumerate(roles):
            key = (seed, role, "home_before" if role == "HOME" else "alternate")
            image = Image.fromarray(images[key], mode="RGB")
            x = column_index * width
            y = row_index * (height + label_height)
            sheet.paste(image, (x, y + label_height))
            draw.text((x + 3, y + 3), f"{seed} {role}", fill=(0, 0, 0))
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    sheet.save(path)
    os.chmod(path, 0o600)


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
        raise RuntimeError("E018-P1 G0 config 要求 CUDA")

    library = config["viewpoint_library"]
    home = _parse_viewpoint(library["home"], "viewpoint_library.home")
    alternates = [
        _parse_viewpoint(item, f"viewpoint_library.alternates[{index}]")
        for index, item in enumerate(library["alternates"])
    ]
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
            raise RuntimeError("G0 external camera uid 不存在")
        sensor = base_env._sensors[environment["camera_uid"]]
        camera = sensor.camera
        if sensor.entity is not None or not callable(getattr(camera, "set_local_pose", None)):
            raise RuntimeError("G0 要求独立、可设位姿的 unmounted external camera")
        for seed in config["experiment"]["seeds"]:
            for alternate in alternates:
                rows, summary, route_images = _run_route(
                    env=env,
                    base_env=base_env,
                    camera=camera,
                    config=config,
                    seed=seed,
                    home=home,
                    alternate=alternate,
                    output_root=output_root,
                    sapien_module=sapien,
                    sapien_utils_module=sapien_utils,
                )
                all_rows.extend(rows)
                route_summaries.append(summary)
                contact_sheet_images[(seed, alternate.viewpoint_id, "alternate")] = route_images[
                    "alternate"
                ]
                contact_sheet_images.setdefault(
                    (seed, "HOME", "home_before"),
                    route_images["home_before"],
                )
    finally:
        env.close()

    _write_contact_sheet(
        output_root / "viewpoint_contact_sheet.png",
        seeds=list(config["experiment"]["seeds"]),
        alternate_ids=[item.viewpoint_id for item in alternates],
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
        "external_camera_sensor_class": type(sensor).__module__ + "." + type(sensor).__name__,
        "external_camera_class": type(camera).__module__ + "." + type(camera).__name__,
        "external_camera_unmounted": sensor.entity is None,
        "set_local_pose_callable": callable(getattr(camera, "set_local_pose", None)),
        "actual_pose_source": "observation.sensor_param.base_camera.cam2world_gl",
    }
    return all_rows, route_summaries, environment_identity


def run_e018_p1_g0(
    *,
    config_path: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """运行 G0 development probe；签名故意不接受 split、checkpoint 或 policy。"""

    config_file = Path(config_path)
    repository = Path(repository_root)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"E018-P1 G0 output 已存在: {output}")
    config = load_e018_p1_g0_config(config_file)
    config_sha256 = _canonical_sha256(config)
    source_identity = _source_identity(repository)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _atomic_json(
        output / "run_state.json",
        {
            "version": E018_P1_G0_RESULT_VERSION,
            "status": "in-progress-development-only",
            "gate": "G0_SIMULATOR_API_FEASIBILITY",
            "test_split_status": "prohibited-unread",
            "formal_claim_allowed": False,
        },
    )
    try:
        frame_rows, route_summaries, environment_identity = _run_simulator(
            config=config,
            output_root=output,
        )
        route_count = len(route_summaries)
        expected_route_count = len(config["experiment"]["seeds"]) * len(
            config["viewpoint_library"]["alternates"]
        )
        all_routes_passed = bool(route_summaries) and all(
            bool(route["passed"]) for route in route_summaries
        )
        gate_passed = all_routes_passed and route_count == expected_route_count
        summary = {
            "version": E018_P1_G0_RESULT_VERSION,
            "status": "complete-development-only",
            "gate": "G0_SIMULATOR_API_FEASIBILITY",
            "gate_passed": gate_passed,
            "config_sha256": config_sha256,
            "viewpoint_library_version": config["viewpoint_library"]["version"],
            "viewpoint_library_status": config["viewpoint_library"]["status"],
            "source_identity": source_identity,
            "environment_identity": environment_identity,
            "seed_count": len(config["experiment"]["seeds"]),
            "alternate_count": len(config["viewpoint_library"]["alternates"]),
            "route_count": route_count,
            "expected_route_count": expected_route_count,
            "passed_route_count": sum(bool(route["passed"]) for route in route_summaries),
            "frame_count": len(frame_rows),
            "failed_routes": [
                route["episode_id"] for route in route_summaries if not route["passed"]
            ],
            "aggregate": {
                "max_camera_position_tracking_error_m": max(
                    row["external_position_tracking_error_m"] for row in frame_rows
                ),
                "max_camera_orientation_tracking_error_rad": max(
                    row["external_orientation_tracking_error_rad"] for row in frame_rows
                ),
                "max_arm_joint_drift_rad": max(
                    row["arm_joint_max_drift_rad"] for row in frame_rows
                ),
                "max_tcp_position_drift_m": max(row["tcp_position_drift_m"] for row in frame_rows),
                "max_tcp_orientation_drift_rad": max(
                    row["tcp_orientation_drift_rad"] for row in frame_rows
                ),
                "max_finger_object_contact_force_n": max(
                    row["finger_object_contact_force_n"] for row in frame_rows
                ),
                "motion_or_unsettled_write_eligible_count": sum(
                    bool(row["measurement_write_eligible"])
                    for row in frame_rows
                    if row["camera_motion_state"] != ExternalCameraMotionState.COLLECT.value
                ),
                "memory_write_count": sum(bool(row["memory_write_executed"]) for row in frame_rows),
                "terminated_or_truncated_count": sum(
                    bool(row["terminated"] or row["truncated"]) for row in frame_rows
                ),
            },
            "scope_limits": {
                "proves": [
                    "ManiSkill base_camera pose can be updated over nonzero control ticks",
                    "dynamic actual camera extrinsic is present in each recorded frame",
                    "HOME-to-one-alternate-to-HOME can preserve arm/TCP/gripper safe hold",
                    "motion/unsettled frames are structurally ineligible for Memory writes",
                ],
                "does_not_prove": [
                    "physical camera actuator dynamics or calibration",
                    "camera collision-envelope safety",
                    "front perception provider qualification",
                    "active versus passive information-recovery benefit",
                    "closed-loop manipulation safety or task success",
                ],
            },
            "ready_for_g1_observation_schema": gate_passed,
            "ready_for_formal_preregistration": False,
            "test_split_status": "prohibited-unread",
            "test_episode_count": 0,
            "provider_forward_count": 0,
            "memory_write_count": 0,
            "physical_robot_actuation_allowed": False,
            "manipulation_progression_allowed": False,
            "formal_claim_allowed": False,
        }
        _atomic_jsonl(output / "camera_pose_ledger.jsonl", frame_rows)
        _atomic_jsonl(output / "route_summaries.jsonl", route_summaries)
        _atomic_json(output / "config_snapshot.json", config)
        _atomic_json(output / "summary.json", summary)
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
            "version": E018_P1_G0_RESULT_VERSION,
            "status": "complete-development-only",
            "gate_passed": gate_passed,
            "config_sha256": config_sha256,
            "files": {str(path.relative_to(output)): _file_sha256(path) for path in artifact_paths},
            "test_split_status": "prohibited-unread",
            "formal_claim_allowed": False,
        }
        receipt["receipt_sha256"] = _canonical_sha256(receipt)
        _atomic_json(output / "receipt.json", receipt)
        _atomic_json(
            output / "run_state.json",
            {
                "version": E018_P1_G0_RESULT_VERSION,
                "status": "complete-development-only",
                "gate": "G0_SIMULATOR_API_FEASIBILITY",
                "gate_passed": gate_passed,
                "test_split_status": "prohibited-unread",
                "formal_claim_allowed": False,
            },
        )
        return summary
    except Exception as error:
        _atomic_json(
            output / "run_state.json",
            {
                "version": E018_P1_G0_RESULT_VERSION,
                "status": "failed-development-only",
                "gate": "G0_SIMULATOR_API_FEASIBILITY",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "test_split_status": "prohibited-unread",
                "formal_claim_allowed": False,
            },
        )
        raise


__all__ = [
    "E018_P1_G0_CONFIG_VERSION",
    "E018_P1_G0_RESULT_VERSION",
    "E018_P1_G0_VIEWPOINT_LIBRARY_VERSION",
    "load_e018_p1_g0_config",
    "run_e018_p1_g0",
]
