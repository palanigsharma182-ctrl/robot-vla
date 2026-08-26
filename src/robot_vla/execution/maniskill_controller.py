"""ManiSkill 单环境 Franka ``pd_joint_delta_pos`` 控制器适配。"""

from __future__ import annotations

from typing import Any

import numpy as np

from robot_vla.adapters import ActionAdapter, FrankaObservationAdapter
from robot_vla.contracts import RobotSpec
from robot_vla.execution.chunk_executor import FrankaControlState


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class ManiSkillFrankaController:
    """只支持 ``num_envs=1``，并将 hold 映射为零关节增量。"""

    def __init__(self, env: Any, spec: RobotSpec) -> None:
        self.env = env
        self.spec = spec
        self.observation_adapter = FrankaObservationAdapter(spec)
        self.action_adapter = ActionAdapter(spec)
        self._last_gripper_opening: float | None = None
        self.last_step_output: Any = None

    @property
    def _robot(self) -> Any:
        try:
            return self.env.unwrapped.agent.robot
        except AttributeError as error:
            raise RuntimeError("ManiSkill env 未暴露 env.unwrapped.agent.robot") from error

    def read_state(self) -> FrankaControlState:
        robot = self._robot
        joint_names = tuple(joint.name for joint in robot.active_joints)
        qpos = _as_numpy(robot.get_qpos())
        qvel = _as_numpy(robot.get_qvel())
        if qpos.shape != (1, len(self.spec.active_joint_names)) or qvel.shape != qpos.shape:
            raise RuntimeError("首版 ManiSkill Controller 只支持 num_envs=1 的 Franka 状态")
        proprio = self.observation_adapter.from_maniskill(
            qpos[0],
            qvel[0],
            joint_names,
        )
        gripper_opening = float(proprio[-1])
        self._last_gripper_opening = gripper_opening
        return FrankaControlState(
            joint_positions=proprio[: self.spec.arm_dof].copy(),
            gripper_opening=gripper_opening,
        )

    def send_action(self, controller_action: np.ndarray) -> None:
        action = np.asarray(controller_action, dtype=np.float32)
        if action.shape != (self.spec.action_dim,) or not np.isfinite(action).all():
            raise ValueError("ManiSkill controller_action shape/dtype 无效")
        if np.any(action < self.env.action_space.low) or np.any(action > self.env.action_space.high):
            raise ValueError("ManiSkill controller_action 超出环境 action_space")
        self._last_gripper_opening = float((action[-1] + 1.0) * 0.5)
        self.last_step_output = self.env.step(action)

    def hold_current(self) -> None:
        try:
            gripper_opening = self.read_state().gripper_opening
        except Exception:  # 状态故障时回退最后有效/命令开口；无回退值则重新抛出
            if self._last_gripper_opening is None:
                raise
            gripper_opening = self._last_gripper_opening
        physical_action = np.zeros(self.spec.action_dim, dtype=np.float32)
        physical_action[-1] = float(gripper_opening)
        self.send_action(self.action_adapter.to_maniskill(physical_action))
