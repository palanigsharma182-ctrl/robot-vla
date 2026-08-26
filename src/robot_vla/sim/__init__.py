"""ManiSkill 可选集成入口。"""

from __future__ import annotations

from importlib import import_module

PICK_CUBE_TO_REGION_ENV_ID = "RobotVLAPickCubeToRegion-v1"


def register_robot_vla_maniskill_envs() -> None:
    """按需导入环境，避免基础数据/训练代码强依赖 ManiSkill。"""

    import_module("robot_vla.sim.pick_cube_to_region")


__all__ = ["PICK_CUBE_TO_REGION_ENV_ID", "register_robot_vla_maniskill_envs"]
