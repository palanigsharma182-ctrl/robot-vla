from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("mani_skill")

from robot_vla.sim import (
    PICK_CUBE_TO_REGION_ENV_ID,
    register_robot_vla_maniskill_envs,
)


def test_pick_cube_to_region_has_visible_dual_camera_goal() -> None:
    register_robot_vla_maniskill_envs()
    env = gym.make(
        PICK_CUBE_TO_REGION_ENV_ID,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        num_envs=1,
    )
    try:
        observation, info = env.reset(seed=7)
        base = env.unwrapped
        cube_position = base.cube.pose.p[0].detach().cpu().numpy()
        goal_position = base.goal_site.pose.p[0].detach().cpu().numpy()

        assert {"base_camera", "hand_camera"} <= set(observation["sensor_data"])
        assert base.goal_site not in base._hidden_objects
        assert np.isclose(goal_position[2], base.cube_half_size)
        assert np.linalg.norm(cube_position[:2] - goal_position[:2]) >= 0.10
        assert set(info) >= {
            "success",
            "is_obj_placed",
            "is_obj_static",
            "is_robot_static",
            "is_grasped",
            "stable_place_steps",
        }
        assert not bool(info["success"].item())
        assert env._max_episode_steps == 300
    finally:
        env.close()
