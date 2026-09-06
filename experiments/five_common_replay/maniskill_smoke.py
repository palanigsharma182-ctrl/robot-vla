"""使用实际 ManiSkill RGB/相机外参验证提取接口；无模型与物体 GT。"""

from __future__ import annotations

import json
from importlib.metadata import version

import gymnasium as gym
import mani_skill.envs  # noqa: F401 — 注册官方环境。
import numpy as np

from robot_vla.precision.active_external_observation import (
    base_camera_round_trip_error_m, extract_active_external_observation, project_base_point,
)
from robot_vla.precision.active_front_camera import ExternalCameraMotionState


def numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def run_smoke() -> dict:
    env = gym.make(
        "PickCube-v1", num_envs=1, obs_mode="rgb", control_mode="pd_joint_delta_pos",
        sim_backend="cpu", render_backend="gpu", sensor_configs={"width": 64, "height": 64},
        max_episode_steps=8,
    )
    records = []
    try:
        for episode in range(2):
            observation, _ = env.reset(seed=episode)
            for tick in range(5):
                timestamp = tick / float(env.unwrapped.control_freq)
                actual_gl = numpy(observation["sensor_param"]["base_camera"]["cam2world_gl"])[0]
                world_from_base = numpy(env.unwrapped.agent.robot.pose.to_transformation_matrix())[0]
                sidecar = extract_active_external_observation(
                    observation, camera_uid="base_camera", world_from_robot_base=world_from_base,
                    commanded_world_from_external_camera_gl=actual_gl,
                    episode_id=f"smoke-{episode}", request_id=f"capture-{episode}",
                    observation_sequence_id=f"{episode}-{tick}", camera_command_sequence_id="stationary",
                    control_tick=tick, control_timestamp_s=timestamp, rgb_timestamp_s=timestamp,
                    camera_pose_timestamp_s=timestamp, camera_motion_state=ExternalCameraMotionState.COLLECT,
                    viewpoint_primitive_id="STATIONARY_INTERFACE_SMOKE", settled=True,
                    maximum_rotation_projection_error_frobenius=1e-6,
                )
                assert sidecar.camera_uid == "base_camera"
                assert sidecar.rgb_external.shape == (64, 64, 3)
                np.testing.assert_array_equal(sidecar.actual_world_from_external_camera_gl, actual_gl)
                # 人工几何探针位于光轴前一米，与任务中物体位置无关。
                probe = (sidecar.base_from_external_camera_cv @ np.array([0., 0., 1., 1.]))[:3]
                error = base_camera_round_trip_error_m(sidecar, probe)
                projection = project_base_point(sidecar, probe)
                assert error < 1e-6 and projection["projection_valid"] and projection["in_frame"]
                np.testing.assert_allclose(projection["uv_px"], sidecar.intrinsic_cv[:2, 2], atol=1e-5)
                records.append({"episode": episode, "tick": tick, "round_trip_error_m": error,
                                "rgb_sha256": sidecar.rgb_sha256, "audit_digest": sidecar.audit_digest()})
                if tick < 4:
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                    action[..., -1] = 1.0
                    observation, _, terminated, truncated, _ = env.step(action)
                    assert not bool(terminated) and not bool(truncated)
    finally:
        env.close()
    return {"evidence_level": "maniskill-rgb-interface-smoke", "mani_skill": version("mani-skill"),
            "sapien": version("sapien"), "episodes": 2, "env_steps": 8, "captures": records,
            "provider_inference_count": 0, "memory_write_count": 0}


if __name__ == "__main__":
    print(json.dumps(run_smoke(), ensure_ascii=False, allow_nan=False, indent=2))
