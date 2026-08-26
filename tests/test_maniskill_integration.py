import numpy as np
import pytest

from robot_vla.adapters import ActionAdapter, FrankaObservationAdapter
from robot_vla.contracts import RobotSpec
from robot_vla.execution import ManiSkillFrankaController

gym = pytest.importorskip("gymnasium")
pytest.importorskip("mani_skill")
import mani_skill.envs  # noqa: F401 - 导入负责注册 ManiSkill Gym 环境


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def test_franka_contract_matches_maniskill_pickcube_runtime() -> None:
    spec = RobotSpec()
    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
    )
    try:
        env.reset(seed=7)
        robot = env.unwrapped.agent.robot
        joint_names = tuple(joint.name for joint in robot.active_joints)
        qpos = _numpy(robot.get_qpos())[0]
        qvel = _numpy(robot.get_qvel())[0]
        qlimits = _numpy(robot.get_qlimits())[0, : spec.arm_dof]

        assert joint_names == spec.active_joint_names
        np.testing.assert_allclose(qlimits, spec.joint_position_limits_rad, atol=1e-6)
        proprio = FrankaObservationAdapter(spec).from_maniskill(qpos, qvel, joint_names)
        assert proprio.shape == (spec.proprio_dim,)

        physical_action = np.zeros(spec.action_dim, dtype=np.float32)
        physical_action[-1] = proprio[-1]
        controller_action = ActionAdapter(spec).to_maniskill(physical_action)
        assert controller_action.shape == env.action_space.shape == (spec.action_dim,)
        assert np.all(controller_action >= env.action_space.low)
        assert np.all(controller_action <= env.action_space.high)
        env.step(controller_action)

        controller = ManiSkillFrankaController(env, spec)
        state = controller.read_state()
        assert state.joint_positions.shape == (spec.arm_dof,)
        opening_before_hold = state.gripper_opening
        controller.hold_current()
        assert controller.last_step_output is not None
        assert controller._last_gripper_opening == pytest.approx(opening_before_hold)
    finally:
        env.close()
