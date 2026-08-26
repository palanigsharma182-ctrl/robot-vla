"""目标可见、必须释放的桌面方块放置环境。"""

from __future__ import annotations

from typing import ClassVar

import mani_skill.envs.tasks  # noqa: F401  # 确保上游环境先完成注册
import torch
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from robot_vla.sim import PICK_CUBE_TO_REGION_ENV_ID


@register_env(PICK_CUBE_TO_REGION_ENV_ID, max_episode_steps=300)
class RobotVLAPickCubeToRegionEnv(PickCubeEnv):
    """D012 语义：把方块释放到传感器可见的桌面目标区域。"""

    SUPPORTED_ROBOTS: ClassVar[list[str]] = ["panda_wristcam"]
    minimum_spawn_separation_m = 0.10
    required_stable_place_steps = 4

    def __init__(self, *args, robot_uids: str = "panda_wristcam", **kwargs) -> None:
        if robot_uids != "panda_wristcam":
            raise ValueError("可信双相机数据只支持 panda_wristcam")
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options: dict) -> None:
        super()._load_scene(options)
        # 官方 PickCube 把目标只当作评测辅助物而隐藏；这里目标是策略输入的一部分。
        self._hidden_objects.remove(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        with torch.device(self.device):
            batch_size = len(env_idx)
            if not hasattr(self, "_stable_place_steps"):
                self._stable_place_steps = torch.zeros(
                    self.num_envs,
                    dtype=torch.int32,
                    device=self.device,
                )
            self._stable_place_steps[env_idx] = 0
            self.table_scene.initialize(env_idx)

            # 将方块和目标采样到桌面左右两侧，并随机交换两侧；x 距离保证至少 10 cm。
            cube_xy = torch.empty((batch_size, 2))
            goal_xy = torch.empty((batch_size, 2))
            cube_xy[:, 0] = torch.rand(batch_size) * 0.05 - 0.10
            goal_xy[:, 0] = torch.rand(batch_size) * 0.05 + 0.05
            cube_xy[:, 1] = torch.rand(batch_size) * 0.16 - 0.08
            goal_xy[:, 1] = torch.rand(batch_size) * 0.16 - 0.08
            swap_sides = torch.rand(batch_size) < 0.5
            swapped_cube_x = torch.where(swap_sides, goal_xy[:, 0], cube_xy[:, 0])
            swapped_goal_x = torch.where(swap_sides, cube_xy[:, 0], goal_xy[:, 0])
            cube_xy[:, 0] = swapped_cube_x
            goal_xy[:, 0] = swapped_goal_x

            cube_xyz = torch.zeros((batch_size, 3))
            cube_xyz[:, :2] = cube_xy
            cube_xyz[:, 2] = self.cube_half_size
            cube_q = randomization.random_quaternions(
                batch_size,
                lock_x=True,
                lock_y=True,
            )
            self.cube.set_pose(Pose.create_from_pq(cube_xyz, cube_q))

            goal_xyz = torch.zeros((batch_size, 3))
            goal_xyz[:, :2] = goal_xy
            # 目标表示最终方块中心，因此固定在桌面上的方块中心高度。
            goal_xyz[:, 2] = self.cube_half_size
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def evaluate(self) -> dict[str, torch.Tensor]:
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, dim=1)
            <= self.goal_thresh
        )
        is_obj_static = self.cube.is_static(lin_thresh=0.01, ang_thresh=0.5)
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        placed_now = is_obj_placed & is_obj_static & ~is_grasped
        self._stable_place_steps = torch.where(
            placed_now,
            self._stable_place_steps + 1,
            torch.zeros_like(self._stable_place_steps),
        )
        success = (
            self._stable_place_steps >= self.required_stable_place_steps
        ) & is_robot_static
        return {
            "success": success,
            "is_obj_placed": is_obj_placed,
            "is_obj_static": is_obj_static,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
            "stable_place_steps": self._stable_place_steps.clone(),
        }


__all__ = ["RobotVLAPickCubeToRegionEnv"]
