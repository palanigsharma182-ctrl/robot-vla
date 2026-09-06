"""比较官方两种关节增量接口；仅合成动作、CPU 物理、无渲染或模型。"""

from __future__ import annotations

import json
from importlib.metadata import version

import gymnasium as gym
import mani_skill.envs  # noqa: F401 - 注册官方环境
import numpy as np

from robot_vla.adapters import ActionAdapter
from robot_vla.contracts import RobotSpec
from robot_vla.execution import ManiSkillFrankaController, RecedingHorizonChunkExecutor


def numpy(value):
    return value.detach().cpu().numpy().copy()


def run_mode(mode: str) -> dict:
    spec = RobotSpec()
    env = gym.make(
        "PickCube-v1", num_envs=1, obs_mode="state", control_mode=mode,
        sim_backend="cpu", render_backend="none", max_episode_steps=32,
    )
    try:
        env.reset(seed=0)
        base = env.unwrapped
        arm = base.agent.controller.controllers["arm"]
        assert arm.config.use_delta and arm.config.normalize_action
        assert arm.config.use_target == (mode == "pd_joint_target_delta_pos")
        assert not arm.config.interpolate
        np.testing.assert_allclose(arm.config.lower, -spec.maniskill_arm_delta_range_rad)
        np.testing.assert_allclose(arm.config.upper, spec.maniskill_arm_delta_range_rad)
        robot = base.agent.robot
        assert tuple(j.name for j in robot.active_joints) == spec.active_joint_names
        np.testing.assert_allclose(
            numpy(robot.get_qlimits())[0, :7], spec.joint_position_limits_rad, atol=1e-6,
        )
        initial_q = numpy(robot.get_qpos())[0, :7]
        sent_targets = []
        original_set_targets = arm.set_drive_targets

        def record_targets(targets):
            # 只记录官方下发给 PD 的目标，继续调用原方法驱动物理仿真。
            sent_targets.append(numpy(targets)[0])
            original_set_targets(targets)

        arm.set_drive_targets = record_targets
        actual_q = []
        chunk = np.zeros((spec.action_horizon, spec.action_dim), dtype=np.float32)
        chunk[:, 0] = 0.005
        chunk[:, -1] = 1.0
        if mode == "pd_joint_delta_pos":
            controller = ManiSkillFrankaController(env, spec)
            original_send = controller.send_action

            def record_step(action):
                original_send(action)
                actual_q.append(numpy(robot.get_qpos())[0, :7])

            controller.send_action = record_step
            executor = RecedingHorizonChunkExecutor(spec)
            for _ in range(2):
                result = executor.execute(chunk, controller)
                assert result.success and result.executed_steps == 4
                assert not result.replan_required
            expected_last = initial_q.copy()
            expected_last[0] += 0.04
            np.testing.assert_allclose(executor.previous_command_q, expected_last, atol=1e-6)
            before_zero = numpy(robot.get_qpos())[0, :7]
            controller.hold_current()
            np.testing.assert_allclose(sent_targets[-1], before_zero, atol=1e-6)
        else:
            # 仅本比较使用目标增量模式，不接入当前 actual-relative 项目 controller。
            adapter = ActionAdapter(spec)
            for action in chunk[:8]:
                env.step(adapter.to_maniskill(action))
                actual_q.append(numpy(robot.get_qpos())[0, :7])
            before_zero = numpy(robot.get_qpos())[0, :7]
            target_before_zero = sent_targets[-1].copy()
            zero = np.zeros(spec.action_dim, dtype=np.float32)
            zero[-1] = 1.0
            env.step(adapter.to_maniskill(zero))
            np.testing.assert_allclose(sent_targets[-1], target_before_zero, atol=1e-6)

        expected_targets = np.repeat(initial_q[None], 8, axis=0)
        expected_targets[:, 0] += np.arange(1, 9) * 0.005
        np.testing.assert_allclose(sent_targets[:8], expected_targets, atol=1e-6)
        zero_target_actual_gap = float(np.max(np.abs(sent_targets[-1] - before_zero)))
        # 重置后第一个零增量应以新 Episode 的状态为参考。
        env.reset(seed=1)
        reset_q = numpy(robot.get_qpos())[0, :7]
        zero = np.zeros(spec.action_dim, dtype=np.float32)
        zero[-1] = 1.0
        env.step(zero)
        np.testing.assert_allclose(sent_targets[-1], reset_q, atol=1e-6)
        return {
            "mode": mode, "steps": 10, "targets": np.asarray(sent_targets[:8]).tolist(),
            "actual_q": np.asarray(actual_q[:8]).tolist(),
            "zero_action_target_actual_gap_rad": zero_target_actual_gap,
            "reset_reference_passed": True,
        }
    finally:
        env.close()


if __name__ == "__main__":
    results = [run_mode(mode) for mode in ("pd_joint_delta_pos", "pd_joint_target_delta_pos")]
    for key in ("targets", "actual_q"):
        np.testing.assert_allclose(results[0][key], results[1][key], atol=1e-5, rtol=0)
    print(json.dumps({
        "status": "passed", "mani_skill": version("mani-skill"), "sapien": version("sapien"),
        "total_control_steps": 20, "results": results,
        "claim": "仅小增量前缀的目标/轨迹一致及零动作/reset语义；未证明限幅或完整策略等价。",
    }, ensure_ascii=False, indent=2))
