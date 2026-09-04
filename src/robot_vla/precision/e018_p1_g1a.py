"""E018-P1 G1A：动态 external observation 的最小仿真 capability probe。"""

from __future__ import annotations

import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla.observation import invert_se3
from robot_vla.precision import e018_p1_g0 as _g0
from robot_vla.precision.active_external_observation import (
    ACTIVE_EXTERNAL_OBSERVATION_VERSION,
    ACTUAL_EXTERNAL_POSE_SOURCE,
    base_camera_round_trip_error_m,
    extract_active_external_observation,
    project_base_point,
)
from robot_vla.precision.active_front_camera import (
    ExternalCameraMotionState,
    FrontCameraOrientationMode,
    rotation_angular_distance_rad,
    sample_translation_path,
    smootherstep,
)
from robot_vla.precision.e018_p1_g0c import (
    _expand_primitives,
    _parse_library,
    load_e018_p1_g0c_config,
)

E018_P1_G1A_CONFIG_VERSION = "e018-p1-g1a-dynamic-external-observation-probe/v2"
E018_P1_G1A_RESULT_VERSION = "e018-p1-g1a-dynamic-external-observation-result/v1"
E018_P1_G1A_GATE = "G1A_DYNAMIC_EXTERNAL_OBSERVATION_CAPABILITY"

_PARENT_CONFIG_SHA256 = "c93bbfd48b6d9bc2fc75b5b87e4ded7161efebd7eda50cd81cc2ded47810e965"
_PARENT_RECEIPT_SHA256 = "bf8232b620cd5ff8de8c0007391252b8829c3ebbac320a7d5a60507beaca258e"
_ALLOWED_PRIMITIVE_IDS = {
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
}
_SOURCE_FILES = (
    "src/robot_vla/precision/active_external_observation.py",
    "src/robot_vla/precision/active_front_camera.py",
    "src/robot_vla/precision/e018_p1_g0.py",
    "src/robot_vla/precision/e018_p1_g0c.py",
    "src/robot_vla/precision/e018_p1_g1a.py",
    "src/robot_vla/cli/run_e018_p1_g1a.py",
)


def load_e018_p1_g1a_config(
    path: str | Path,
    *,
    parent_g0c_config_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """严格加载 development-only G1A 和它绑定的 G0C parent。"""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"E018-P1 G1A config 不存在: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = _g0._require_keys(
        config,
        {
            "version",
            "status",
            "scope",
            "parent_motion",
            "software",
            "environment",
            "probe",
            "gates",
            "execution",
        },
        "E018-P1 G1A config",
    )
    if config["version"] != E018_P1_G1A_CONFIG_VERSION:
        raise ValueError("E018-P1 G1A config version 漂移")
    if config["status"] != "development-only-g1a-no-provider-no-memory-no-formal-claim":
        raise ValueError("E018-P1 G1A 只能以 development-only 状态运行")

    scope = _g0._require_keys(
        config["scope"],
        {
            "gate",
            "test_split_allowed",
            "formal_claim_allowed",
            "runtime_gt_control_allowed",
            "provider_inference_allowed",
            "memory_read_allowed",
            "memory_write_allowed",
            "executive_mutation_allowed",
        },
        "scope",
    )
    if scope != {
        "gate": E018_P1_G1A_GATE,
        "test_split_allowed": False,
        "formal_claim_allowed": False,
        "runtime_gt_control_allowed": False,
        "provider_inference_allowed": False,
        "memory_read_allowed": False,
        "memory_write_allowed": False,
        "executive_mutation_allowed": False,
    }:
        raise ValueError("G1A scope 必须禁止 test/formal/runtime GT/provider/Memory/Executive")

    parent = _g0._require_keys(
        config["parent_motion"],
        {
            "gate",
            "config_version",
            "config_sha256",
            "receipt_sha256",
            "gate_passed",
            "library_status",
        },
        "parent_motion",
    )
    if parent != {
        "gate": "G0C_ROTATED_POSE_MOTION_FEASIBILITY",
        "config_version": "e018-p1-g0c-rotated-motion-development/v1",
        "config_sha256": _PARENT_CONFIG_SHA256,
        "receipt_sha256": _PARENT_RECEIPT_SHA256,
        "gate_passed": True,
        "library_status": "provisional-development-not-frozen",
    }:
        raise ValueError("G1A parent G0C identity 漂移")
    parent_config = load_e018_p1_g0c_config(parent_g0c_config_path)
    if _g0._canonical_sha256(parent_config) != _PARENT_CONFIG_SHA256:
        raise ValueError("G1A parent G0C config SHA-256 漂移")

    software = _g0._require_keys(
        config["software"],
        {"expected_mani_skill_version", "expected_sapien_version"},
        "software",
    )
    if software != parent_config["software"]:
        raise ValueError("G1A software identity 必须继承 G0C")
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
    if environment != parent_config["environment"]:
        raise ValueError("G1A environment identity 必须继承 G0C")

    probe = _g0._require_keys(
        config["probe"],
        {
            "usage",
            "seed",
            "selected_primitive_id",
            "selection_policy",
            "warmup_ticks",
            "move_duration_s",
            "settle_ticks",
            "repeat_capture_ticks",
            "home_barrier_ticks",
            "center_ray_support_radius_px",
            "save_rgb",
            "offline_gt_diagnostics",
        },
        "probe",
    )
    if (
        probe["usage"] != "simulator-development-capability-probe-only/v1"
        or probe["selection_policy"] != "fixed-before-probe-from-g0b-shortlist/v1"
        or probe["selected_primitive_id"] not in _ALLOWED_PRIMITIVE_IDS
        or probe["save_rgb"] is not True
        or probe["offline_gt_diagnostics"] is not True
    ):
        raise ValueError("G1A probe scope/primitive 漂移")
    if not isinstance(probe["seed"], int) or isinstance(probe["seed"], bool) or probe["seed"] <= 0:
        raise ValueError("G1A seed 必须是正整数 development seed")
    for name in (
        "warmup_ticks",
        "settle_ticks",
        "repeat_capture_ticks",
        "home_barrier_ticks",
        "center_ray_support_radius_px",
    ):
        _g0._positive_int(probe[name], f"probe.{name}")
    move_duration_s = _g0._positive(probe["move_duration_s"], "probe.move_duration_s")
    if not float(move_duration_s * environment["control_hz"]).is_integer():
        raise ValueError("G1A move_duration_s * control_hz 必须是整数")
    if probe["home_barrier_ticks"] != 4:
        raise ValueError("G1A HOME barrier 必须固定为四个全新 frame")

    gates = _g0._require_keys(
        config["gates"],
        {
            "maximum_rgb_pose_skew_s",
            "minimum_alternate_rgb_mean_abs_difference",
            "maximum_intra_pose_rgb_mean_abs_difference",
            "maximum_return_home_rgb_mean_abs_difference",
            "camera_position_tracking_tolerance_m",
            "camera_orientation_tracking_tolerance_rad",
            "maximum_rotation_projection_error_frobenius",
            "maximum_base_camera_round_trip_error_m",
            "maximum_static_object_base_drift_m",
            "minimum_center_ray_entity_fraction",
            "arm_joint_drift_max_rad",
            "tcp_position_drift_max_m",
            "tcp_orientation_drift_max_rad",
            "minimum_open_finger_joint_position_m",
            "unexpected_finger_contact_max_n",
        },
        "gates",
    )
    for name, value in gates.items():
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"gates.{name} 必须是有限非负数")
    if not 0.0 <= gates["minimum_center_ray_entity_fraction"] <= 1.0:
        raise ValueError("minimum_center_ray_entity_fraction 必须位于 [0,1]")

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
            "test_data_read_allowed",
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
        "test_data_read_allowed": False,
    }:
        raise ValueError("G1A execution safety scope 漂移")
    return config, parent_config


def _source_identity(repository_root: Path) -> dict[str, Any]:
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
            path: _g0._file_sha256(repository_root / path) for path in _SOURCE_FILES
        },
    }
    identity["worktree_clean"] = not identity["git_status"]
    identity["identity_sha256"] = _g0._canonical_sha256(identity)
    return identity


def _point_base(base_env: Any, actor: Any) -> np.ndarray:
    world_from_base = _g0._single_matrix(base_env.agent.robot.pose, "world_from_robot_base")
    base_from_world = invert_se3(world_from_base, "world_from_robot_base")
    point_world = _g0._single_vector(actor.pose.p, 3, "object position world")
    return (base_from_world @ np.concatenate((point_world, np.ones(1))))[:3]


def _center_ray_entity_hit(
    observation: dict[str, Any],
    *,
    camera_uid: str,
    actor_id: int,
    projection: dict[str, Any],
    support_radius_px: int,
) -> bool:
    if not projection["projection_valid"] or not projection["in_frame"]:
        return False
    segmentation = _g0._numpy(observation["sensor_data"][camera_uid]["segmentation"])
    if segmentation.ndim != 4 or segmentation.shape[0] != 1:
        raise RuntimeError("G1A external segmentation 必须是 [1,H,W,C]")
    actor_ids = np.asarray(segmentation[0, ..., 0])
    u, v = projection["uv_px"]
    center_x, center_y = round(u), round(v)
    height, width = actor_ids.shape
    support = actor_ids[
        max(0, center_y - support_radius_px) : min(height, center_y + support_radius_px + 1),
        max(0, center_x - support_radius_px) : min(width, center_x + support_radius_px + 1),
    ]
    return bool(np.count_nonzero(support == actor_id))


def _run_simulator(
    *,
    config: dict[str, Any],
    parent_config: dict[str, Any],
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    import gymnasium as gym
    import mani_skill
    import sapien
    import torch
    from mani_skill.utils import sapien_utils

    from robot_vla.sim import register_robot_vla_maniskill_envs

    software = config["software"]
    if mani_skill.__version__ != software["expected_mani_skill_version"]:
        raise RuntimeError("G1A ManiSkill version 漂移")
    if sapien.__version__ != software["expected_sapien_version"]:
        raise RuntimeError("G1A SAPIEN version 漂移")
    if not torch.cuda.is_available():
        raise RuntimeError("G1A config 要求 CUDA")

    environment = config["environment"]
    probe = config["probe"]
    gates = config["gates"]
    home, anchors, orientations = _parse_library(parent_config)
    primitive_by_id = {
        primitive.viewpoint_id: (primitive, orientation)
        for primitive, orientation in _expand_primitives(anchors, orientations)
    }
    selected, selected_orientation = primitive_by_id[probe["selected_primitive_id"]]
    center_orientation = FrontCameraOrientationMode("CENTER", 0.0, 0.0, 0.0)
    episode_id = f"g1a-seed-{probe['seed']:06d}-{selected.viewpoint_id.lower().replace('_', '-')}"
    request_id = f"{episode_id}-request-00"
    command_sequence_id = f"{episode_id}-camera-command-00"
    control_period_s = 1.0 / environment["control_hz"]
    move_steps = round(probe["move_duration_s"] * environment["control_hz"])

    register_robot_vla_maniskill_envs()
    env = gym.make(
        environment["environment_id"],
        obs_mode=environment["obs_mode"],
        control_mode=environment["control_mode"],
        num_envs=environment["num_envs"],
        robot_uids=environment["robot_uid"],
    )
    sidecar_rows: list[dict[str, Any]] = []
    offline_rows: list[dict[str, Any]] = []
    captured_images: dict[str, list[np.ndarray]] = {
        "home_before": [],
        "alternate": [],
        "home_barrier": [],
    }
    try:
        base_env = env.unwrapped
        if base_env.control_freq != environment["control_hz"]:
            raise RuntimeError("G1A control_hz 漂移")
        sensor = base_env._sensors.get(environment["camera_uid"])
        if sensor is None or sensor.entity is not None:
            raise RuntimeError("G1A 要求独立 unmounted external camera")
        camera = sensor.camera
        if not callable(getattr(camera, "set_local_pose", None)):
            raise TypeError("G1A external camera 不支持 set_local_pose")

        env.reset(seed=probe["seed"])
        home_position = np.asarray(home.position_world_m, dtype=np.float64)
        alternate_position = np.asarray(selected.position_world_m, dtype=np.float64)
        observation: dict[str, Any] | None = None
        home_q: np.ndarray | None = None
        home_gl: np.ndarray | None = None
        for _ in range(probe["warmup_ticks"]):
            home_q, home_gl = _g0._set_camera_pose(
                camera,
                home,
                home_position,
                orientation=center_orientation,
                sapien_module=sapien,
                sapien_utils_module=sapien_utils,
            )
            observation, terminated, truncated = _g0._step_hold_open(env, env.action_space.shape)
            if terminated or truncated:
                raise RuntimeError("G1A warmup 期间环境提前结束")
        if observation is None or home_q is None or home_gl is None:
            raise RuntimeError("G1A warmup 没有产生 observation")

        arm_anchor = _g0._numpy(base_env.agent.robot.get_qpos())[0, :7].astype(np.float64)
        tcp_anchor = _g0._single_matrix(base_env.agent.tcp_pose, "anchor_world_from_tcp")
        object_base_anchor = _point_base(base_env, base_env.cube)
        object_actor_id = int(_g0._numpy(base_env.cube.per_scene_id).reshape(-1)[0])
        control_tick = 0
        previous_actual_pose: np.ndarray | None = None

        def capture(
            *,
            state: ExternalCameraMotionState,
            viewpoint_id: str,
            commanded_world_from_gl: np.ndarray,
            current_observation: dict[str, Any],
            settled: bool,
            image_group: str | None = None,
        ) -> None:
            nonlocal previous_actual_pose
            timestamp_s = control_tick * control_period_s
            world_from_base = _g0._single_matrix(
                base_env.agent.robot.pose,
                "world_from_robot_base",
            )
            sidecar = extract_active_external_observation(
                current_observation,
                camera_uid=environment["camera_uid"],
                world_from_robot_base=world_from_base,
                commanded_world_from_external_camera_gl=commanded_world_from_gl,
                episode_id=episode_id,
                request_id=request_id,
                observation_sequence_id=f"{episode_id}-observation-{control_tick:04d}",
                camera_command_sequence_id=command_sequence_id,
                control_tick=control_tick,
                control_timestamp_s=timestamp_s,
                rgb_timestamp_s=timestamp_s,
                camera_pose_timestamp_s=timestamp_s,
                camera_motion_state=state,
                viewpoint_primitive_id=viewpoint_id,
                settled=settled,
                maximum_rotation_projection_error_frobenius=(
                    gates["maximum_rotation_projection_error_frobenius"]
                ),
            )
            arm_q = _g0._numpy(base_env.agent.robot.get_qpos())[0]
            tcp = _g0._single_matrix(base_env.agent.tcp_pose, "actual_world_from_tcp")
            actual_pose = sidecar.actual_world_from_external_camera_gl
            position_tracking = float(
                np.linalg.norm(actual_pose[:3, 3] - commanded_world_from_gl[:3, 3])
            )
            orientation_tracking = rotation_angular_distance_rad(
                actual_pose[:3, :3],
                commanded_world_from_gl[:3, :3],
            )
            dynamic_pose_changed = previous_actual_pose is not None and not np.allclose(
                actual_pose,
                previous_actual_pose,
                rtol=0.0,
                atol=1e-10,
            )
            row = sidecar.ledger_record()
            row.update(
                {
                    "arm_owner": "SAFE_HOLD",
                    "gripper_owner": "SAFE_HOLD_OPEN",
                    "external_camera_owner": "ACTIVE_REOBSERVE_G1A_PROBE",
                    "arm_motion_command_max_abs": 0.0,
                    "arm_joint_max_drift_rad": float(
                        np.max(np.abs(arm_q[:7] - arm_anchor))
                    ),
                    "tcp_position_drift_m": float(
                        np.linalg.norm(tcp[:3, 3] - tcp_anchor[:3, 3])
                    ),
                    "tcp_orientation_drift_rad": rotation_angular_distance_rad(
                        tcp[:3, :3],
                        tcp_anchor[:3, :3],
                    ),
                    "minimum_finger_joint_position_m": float(np.min(arm_q[7:])),
                    "finger_object_contact_force_n": _g0._read_finger_contact_force_n(base_env),
                    "camera_position_tracking_error_m": position_tracking,
                    "camera_orientation_tracking_error_rad": orientation_tracking,
                    "actual_pose_changed_from_previous": dynamic_pose_changed,
                }
            )
            sidecar_rows.append(row)

            object_base = _point_base(base_env, base_env.cube)
            projection = project_base_point(sidecar, object_base)
            center_hit = _center_ray_entity_hit(
                current_observation,
                camera_uid=environment["camera_uid"],
                actor_id=object_actor_id,
                projection=projection,
                support_radius_px=probe["center_ray_support_radius_px"],
            )
            offline_rows.append(
                {
                    "version": E018_P1_G1A_RESULT_VERSION,
                    "observation_sequence_id": sidecar.observation_sequence_id,
                    "camera_motion_state": state.value,
                    "viewpoint_primitive_id": viewpoint_id,
                    "object_position_base_m": object_base.tolist(),
                    "object_base_drift_m": float(
                        np.linalg.norm(object_base - object_base_anchor)
                    ),
                    "base_camera_round_trip_error_m": base_camera_round_trip_error_m(
                        sidecar,
                        object_base,
                    ),
                    "object_projection": projection,
                    "object_center_ray_entity_hit": center_hit,
                    "oracle_only": True,
                    "used_by_runtime_control": False,
                }
            )
            if image_group is not None:
                captured_images[image_group].append(sidecar.rgb_external.copy())
            previous_actual_pose = actual_pose.copy()

        def command_and_step(
            *,
            viewpoint: Any,
            position: np.ndarray,
            orientation: FrontCameraOrientationMode,
        ) -> tuple[dict[str, Any], np.ndarray]:
            nonlocal control_tick
            _, commanded_gl = _g0._set_camera_pose(
                camera,
                viewpoint,
                position,
                orientation=orientation,
                sapien_module=sapien,
                sapien_utils_module=sapien_utils,
            )
            current, terminated, truncated = _g0._step_hold_open(env, env.action_space.shape)
            control_tick += 1
            if terminated or truncated:
                raise RuntimeError("G1A probe 期间环境提前结束")
            return current, commanded_gl

        for _ in range(probe["repeat_capture_ticks"]):
            observation, commanded_gl = command_and_step(
                viewpoint=home,
                position=home_position,
                orientation=center_orientation,
            )
            capture(
                state=ExternalCameraMotionState.HOME_ANCHOR,
                viewpoint_id=home.viewpoint_id,
                commanded_world_from_gl=commanded_gl,
                current_observation=observation,
                settled=True,
                image_group="home_before",
            )

        outward_path = sample_translation_path(home_position, alternate_position, steps=move_steps)
        for index, position in enumerate(outward_path):
            scale = smootherstep((index + 1) / move_steps)
            orientation = FrontCameraOrientationMode(
                selected_orientation.orientation_id,
                selected_orientation.yaw_offset_rad * scale,
                selected_orientation.pitch_offset_rad * scale,
                selected_orientation.roll_offset_rad * scale,
            )
            observation, commanded_gl = command_and_step(
                viewpoint=selected,
                position=position,
                orientation=orientation,
            )
            capture(
                state=ExternalCameraMotionState.MOVE_TO_VIEW,
                viewpoint_id=selected.viewpoint_id,
                commanded_world_from_gl=commanded_gl,
                current_observation=observation,
                settled=False,
            )

        for settle_index in range(probe["settle_ticks"]):
            observation, commanded_gl = command_and_step(
                viewpoint=selected,
                position=alternate_position,
                orientation=selected_orientation,
            )
            capture(
                state=ExternalCameraMotionState.SETTLE_AT_VIEW,
                viewpoint_id=selected.viewpoint_id,
                commanded_world_from_gl=commanded_gl,
                current_observation=observation,
                settled=settle_index == probe["settle_ticks"] - 1,
            )

        for _ in range(probe["repeat_capture_ticks"]):
            observation, commanded_gl = command_and_step(
                viewpoint=selected,
                position=alternate_position,
                orientation=selected_orientation,
            )
            capture(
                state=ExternalCameraMotionState.COLLECT,
                viewpoint_id=selected.viewpoint_id,
                commanded_world_from_gl=commanded_gl,
                current_observation=observation,
                settled=True,
                image_group="alternate",
            )

        return_path = sample_translation_path(alternate_position, home_position, steps=move_steps)
        for index, position in enumerate(return_path):
            scale = 1.0 - smootherstep((index + 1) / move_steps)
            orientation = FrontCameraOrientationMode(
                selected_orientation.orientation_id,
                selected_orientation.yaw_offset_rad * scale,
                selected_orientation.pitch_offset_rad * scale,
                selected_orientation.roll_offset_rad * scale,
            )
            observation, commanded_gl = command_and_step(
                viewpoint=home,
                position=position,
                orientation=orientation,
            )
            capture(
                state=ExternalCameraMotionState.RETURN_HOME,
                viewpoint_id=home.viewpoint_id,
                commanded_world_from_gl=commanded_gl,
                current_observation=observation,
                settled=False,
            )

        for _ in range(probe["home_barrier_ticks"]):
            observation, commanded_gl = command_and_step(
                viewpoint=home,
                position=home_position,
                orientation=center_orientation,
            )
            capture(
                state=ExternalCameraMotionState.VERIFY_HOME_AND_ARM_HOLD,
                viewpoint_id=home.viewpoint_id,
                commanded_world_from_gl=commanded_gl,
                current_observation=observation,
                settled=True,
                image_group="home_barrier",
            )

        if probe["save_rgb"]:
            for group, images in captured_images.items():
                for index, image in enumerate(images):
                    _g0._save_png(
                        output_root / "images" / f"{group}-{index:02d}.png",
                        image,
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
            "actual_pose_source": ACTUAL_EXTERNAL_POSE_SOURCE,
            "simulator_timestamp_semantics": "same-post-step-observation-control-tick/v1",
        }
    finally:
        env.close()

    def maximum_mean_difference(images: list[np.ndarray]) -> float:
        reference = images[0].astype(np.float64)
        return max(float(np.mean(np.abs(image.astype(np.float64) - reference))) for image in images)

    home_before = captured_images["home_before"]
    alternate = captured_images["alternate"]
    home_barrier = captured_images["home_barrier"]
    if not home_before or not alternate or len(home_barrier) != probe["home_barrier_ticks"]:
        raise RuntimeError("G1A capture/barrier frame 数不足")
    alternate_difference = float(
        np.mean(np.abs(alternate[-1].astype(np.float64) - home_before[-1].astype(np.float64)))
    )
    return_home_difference = float(
        np.mean(np.abs(home_barrier[-1].astype(np.float64) - home_before[-1].astype(np.float64)))
    )
    audited_states = {
        ExternalCameraMotionState.HOME_ANCHOR.value,
        ExternalCameraMotionState.COLLECT.value,
        ExternalCameraMotionState.VERIFY_HOME_AND_ARM_HOLD.value,
    }
    audited_offline = [row for row in offline_rows if row["camera_motion_state"] in audited_states]
    center_hit_fraction = sum(row["object_center_ray_entity_hit"] for row in audited_offline) / len(
        audited_offline
    )
    actual_positions = {
        tuple(
            round(value, 8)
            for value in np.asarray(row["actual_world_from_external_camera_gl"])[
                :3, 3
            ]
        )
        for row in sidecar_rows
    }
    forbidden_runtime_keys = {
        "object_position_base_m",
        "object_projection",
        "object_center_ray_entity_hit",
        "goal_position_base_m",
        "gt_visibility",
    }
    runtime_gt_leak_count = sum(
        bool(forbidden_runtime_keys.intersection(row)) for row in sidecar_rows
    )
    metrics = {
        "frame_count": len(sidecar_rows),
        "unique_actual_camera_positions": len(actual_positions),
        "max_rgb_pose_skew_s": max(row["rgb_pose_skew_s"] for row in sidecar_rows),
        "alternate_rgb_mean_abs_difference": alternate_difference,
        "max_home_intra_pose_rgb_mean_abs_difference": maximum_mean_difference(home_before),
        "max_alternate_intra_pose_rgb_mean_abs_difference": maximum_mean_difference(alternate),
        "return_home_rgb_mean_abs_difference": return_home_difference,
        "max_camera_position_tracking_error_m": max(
            row["camera_position_tracking_error_m"] for row in sidecar_rows
        ),
        "max_camera_orientation_tracking_error_rad": max(
            row["camera_orientation_tracking_error_rad"] for row in sidecar_rows
        ),
        "max_actual_rotation_projection_error_frobenius": max(
            row["actual_rotation_projection_audit"]["correction_frobenius"]
            for row in sidecar_rows
        ),
        "max_base_rotation_projection_error_frobenius": max(
            row["base_rotation_projection_audit"]["correction_frobenius"]
            for row in sidecar_rows
        ),
        "max_actual_rotation_orthogonality_error_before_frobenius": max(
            row["actual_rotation_projection_audit"][
                "orthogonality_error_before_frobenius"
            ]
            for row in sidecar_rows
        ),
        "max_actual_rotation_orthogonality_error_after_frobenius": max(
            row["actual_rotation_projection_audit"][
                "orthogonality_error_after_frobenius"
            ]
            for row in sidecar_rows
        ),
        "max_actual_rotation_determinant_deviation_before": max(
            abs(row["actual_rotation_projection_audit"]["determinant_before"] - 1.0)
            for row in sidecar_rows
        ),
        "max_actual_rotation_determinant_deviation_after": max(
            abs(row["actual_rotation_projection_audit"]["determinant_after"] - 1.0)
            for row in sidecar_rows
        ),
        "max_base_camera_round_trip_error_m": max(
            row["base_camera_round_trip_error_m"] for row in offline_rows
        ),
        "max_static_object_base_drift_m": max(row["object_base_drift_m"] for row in offline_rows),
        "center_ray_entity_fraction": center_hit_fraction,
        "max_arm_joint_drift_rad": max(row["arm_joint_max_drift_rad"] for row in sidecar_rows),
        "max_tcp_position_drift_m": max(row["tcp_position_drift_m"] for row in sidecar_rows),
        "max_tcp_orientation_drift_rad": max(
            row["tcp_orientation_drift_rad"] for row in sidecar_rows
        ),
        "minimum_finger_joint_position_m": min(
            row["minimum_finger_joint_position_m"] for row in sidecar_rows
        ),
        "max_finger_object_contact_force_n": max(
            row["finger_object_contact_force_n"] for row in sidecar_rows
        ),
        "motion_or_unsettled_write_eligible_count": sum(
            row["memory_write_eligible"]
            for row in sidecar_rows
            if row["camera_motion_state"] != ExternalCameraMotionState.COLLECT.value
        ),
        "runtime_gt_leak_count": runtime_gt_leak_count,
        "provider_forward_count": 0,
        "memory_read_count": 0,
        "memory_write_count": 0,
        "test_read_count": 0,
    }
    gate_checks = {
        "dynamic_actual_pose": len(actual_positions) > 2,
        "same_observation_pose_source": all(
            row["actual_pose_source"] == ACTUAL_EXTERNAL_POSE_SOURCE for row in sidecar_rows
        ),
        "rgb_pose_time_relation": metrics["max_rgb_pose_skew_s"]
        <= gates["maximum_rgb_pose_skew_s"],
        "rendered_view_changed": alternate_difference
        >= gates["minimum_alternate_rgb_mean_abs_difference"],
        "repeat_capture_deterministic": max(
            metrics["max_home_intra_pose_rgb_mean_abs_difference"],
            metrics["max_alternate_intra_pose_rgb_mean_abs_difference"],
        )
        <= gates["maximum_intra_pose_rgb_mean_abs_difference"],
        "return_home_render_recovered": return_home_difference
        <= gates["maximum_return_home_rgb_mean_abs_difference"],
        "commanded_actual_tracking": (
            metrics["max_camera_position_tracking_error_m"]
            <= gates["camera_position_tracking_tolerance_m"]
            and metrics["max_camera_orientation_tracking_error_rad"]
            <= gates["camera_orientation_tracking_tolerance_rad"]
        ),
        "rotation_projection_within_frozen_tolerance": max(
            metrics["max_actual_rotation_projection_error_frobenius"],
            metrics["max_base_rotation_projection_error_frobenius"],
        )
        <= gates["maximum_rotation_projection_error_frobenius"],
        "base_camera_round_trip": metrics["max_base_camera_round_trip_error_m"]
        <= gates["maximum_base_camera_round_trip_error_m"],
        "static_object_cross_view": metrics["max_static_object_base_drift_m"]
        <= gates["maximum_static_object_base_drift_m"],
        "opencv_projection_matches_segmentation": center_hit_fraction
        >= gates["minimum_center_ray_entity_fraction"],
        "arm_tcp_gripper_hold": (
            metrics["max_arm_joint_drift_rad"] <= gates["arm_joint_drift_max_rad"]
            and metrics["max_tcp_position_drift_m"] <= gates["tcp_position_drift_max_m"]
            and metrics["max_tcp_orientation_drift_rad"] <= gates["tcp_orientation_drift_max_rad"]
            and metrics["minimum_finger_joint_position_m"]
            >= gates["minimum_open_finger_joint_position_m"]
            and metrics["max_finger_object_contact_force_n"]
            <= gates["unexpected_finger_contact_max_n"]
        ),
        "home_barrier_four_fresh_frames": len(
            {
                row["observation_sequence_id"]
                for row in sidecar_rows
                if row["camera_motion_state"]
                == ExternalCameraMotionState.VERIFY_HOME_AND_ARM_HOLD.value
            }
        )
        == 4,
        "no_runtime_gt_provider_memory_or_test": (
            metrics["runtime_gt_leak_count"] == 0
            and metrics["provider_forward_count"] == 0
            and metrics["memory_read_count"] == 0
            and metrics["memory_write_count"] == 0
            and metrics["test_read_count"] == 0
        ),
        "write_scope": metrics["motion_or_unsettled_write_eligible_count"] == 0,
    }
    return sidecar_rows, offline_rows, environment_identity, {
        "metrics": metrics,
        "gate_checks": gate_checks,
    }


def run_e018_p1_g1a(
    *,
    config_path: str | Path,
    parent_g0c_config_path: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """运行一个 seed/一个冻结 alternate 的 development-only G1A probe。"""

    repository = Path(repository_root)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"E018-P1 G1A output 已存在: {output}")
    config, parent_config = load_e018_p1_g1a_config(
        config_path,
        parent_g0c_config_path=parent_g0c_config_path,
    )
    source_identity = _source_identity(repository)
    config_sha256 = _g0._canonical_sha256(config)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _g0._atomic_json(
        output / "run_state.json",
        {
            "version": E018_P1_G1A_RESULT_VERSION,
            "status": "in-progress-development-only",
            "gate": E018_P1_G1A_GATE,
            "test_split_status": "prohibited-unread",
        },
    )
    try:
        sidecar_rows, offline_rows, environment_identity, result = _run_simulator(
            config=config,
            parent_config=parent_config,
            output_root=output,
        )
        gate_passed = all(result["gate_checks"].values())
        summary = {
            "version": E018_P1_G1A_RESULT_VERSION,
            "status": "complete-development-only",
            "gate": E018_P1_G1A_GATE,
            "gate_passed": gate_passed,
            "config_sha256": config_sha256,
            "parent_motion": config["parent_motion"],
            "active_external_observation_version": ACTIVE_EXTERNAL_OBSERVATION_VERSION,
            "source_identity": source_identity,
            "environment_identity": environment_identity,
            "seed": config["probe"]["seed"],
            "selected_primitive_id": config["probe"]["selected_primitive_id"],
            **result,
            "test_split_status": "prohibited-unread",
            "formal_claim_allowed": False,
            "provider_forward_count": 0,
            "memory_read_count": 0,
            "memory_write_count": 0,
            "executive_mutation_count": 0,
            "scope_limits": {
                "proves": [
                    (
                        "same-observation external RGB and actual cam2world_gl can form "
                        "an audited sidecar"
                    ),
                    "dynamic GL-to-CV/base transforms agree with offline projection diagnostics",
                    (
                        "one fixed G0C primitive changes the render and returns to HOME "
                        "under arm SafeHold"
                    ),
                ],
                "does_not_prove": [
                    "front provider qualification or object measurement quality",
                    "uncertainty-trigger correctness or runtime view selection",
                    (
                        "live Memory update, executive recovery, task benefit, or "
                        "physical-camera behavior"
                    ),
                ],
            },
        }
        _g0._atomic_jsonl(output / "active_external_observation_ledger.jsonl", sidecar_rows)
        _g0._atomic_jsonl(output / "offline_gt_diagnostics.jsonl", offline_rows)
        _g0._atomic_json(output / "config_snapshot.json", config)
        _g0._atomic_json(output / "summary.json", summary)
        artifact_paths = sorted(
            [
                output / "active_external_observation_ledger.jsonl",
                output / "offline_gt_diagnostics.jsonl",
                output / "config_snapshot.json",
                output / "summary.json",
                *(output / "images").glob("*.png"),
            ],
            key=lambda path: str(path.relative_to(output)),
        )
        receipt = {
            "version": E018_P1_G1A_RESULT_VERSION,
            "status": "complete-development-only",
            "gate_passed": gate_passed,
            "config_sha256": config_sha256,
            "files": {
                str(path.relative_to(output)): _g0._file_sha256(path) for path in artifact_paths
            },
            "test_split_status": "prohibited-unread",
            "formal_claim_allowed": False,
        }
        receipt["receipt_sha256"] = _g0._canonical_sha256(receipt)
        _g0._atomic_json(output / "receipt.json", receipt)
        _g0._atomic_json(
            output / "run_state.json",
            {
                "version": E018_P1_G1A_RESULT_VERSION,
                "status": "complete-development-only",
                "gate": E018_P1_G1A_GATE,
                "gate_passed": gate_passed,
                "test_split_status": "prohibited-unread",
            },
        )
        return summary
    except Exception as error:
        _g0._atomic_json(
            output / "run_state.json",
            {
                "version": E018_P1_G1A_RESULT_VERSION,
                "status": "failed-development-only",
                "gate": E018_P1_G1A_GATE,
                "error_type": type(error).__name__,
                "error": str(error),
                "test_split_status": "prohibited-unread",
            },
        )
        raise


__all__ = [
    "E018_P1_G1A_CONFIG_VERSION",
    "E018_P1_G1A_GATE",
    "E018_P1_G1A_RESULT_VERSION",
    "load_e018_p1_g1a_config",
    "run_e018_p1_g1a",
]
