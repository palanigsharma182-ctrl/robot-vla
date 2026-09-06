"""固定 development 场景的 HOME 快照与真实发送的抓取前动作标签。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from experiments.g2c_memory_integration.run import run, DevelopmentController
from experiments.memory_conditioning.conditioning import snapshot_memory
from robot_vla.adapters import ActionAdapter, FrankaObservationAdapter
from robot_vla.contracts import RobotSpec

SEEDS = tuple(range(1000100, 1000112))


def capture(*, env, frame, memory, safety, episode, tick, output):
    """GT 仅供教师规划；输入只包含真实 RGB/proprio 和已提交的 G2C Memory。"""
    import sapien
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
    from robot_vla.evaluation.maniskill import _read_observation_v2_frame

    snapshot = snapshot_memory(memory.state, memory.config, safety,
                               episode_id=episode, timestamp_s=frame.timestamp_s)
    if not snapshot.available:
        return dict(status="memory-unavailable", reasons=snapshot.reasons)
    base = env.unwrapped
    spec = RobotSpec()
    action_adapter = ActionAdapter(spec)
    observation_adapter = FrankaObservationAdapter(spec)
    controller = DevelopmentController(env, spec)
    planner = PandaArmMotionPlanningSolver(env, debug=False, vis=False,
        base_pose=base.agent.robot.pose, visualize_target_grasp_pose=False,
        print_env_info=False, joint_vel_limits=.4, joint_acc_limits=.4)
    labels, commands, actual, poses, times = [], [], [], [], []
    try:
        approaching = np.array([0., 0., -1.])
        closing = base.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        grasp = compute_grasp_info_by_obb(get_actor_obb(base.cube),
            approaching=approaching, target_closing=closing, depth=.025)
        pose = base.agent.build_grasp_pose(approaching, grasp["closing"], base.cube.pose.sp.p)
        plan = planner.move_to_pose_with_screw(pose * sapien.Pose([0, 0, -.05]), dry_run=True)
        if not isinstance(plan, dict) or plan.get("status") != "Success":
            return dict(status="teacher-plan-rejected")
        targets = np.asarray(plan["position"], dtype=np.float32)
        if targets.ndim != 2 or targets.shape[1] != 7 or len(targets) < 16:
            return dict(status="teacher-path-too-short", shape=list(targets.shape))
        # HOME 已清空旧 command reference；与执行器一致，以 actual q 初始化新序列。
        initial_reference = controller.read_state().joint_positions.copy()
        previous = initial_reference.copy()
        latest = frame
        for target in targets[:16]:
            # 只采实际执行的前 16 步，不裁剪目标或把 actual-q correction 当 label。
            if np.max(latest.finger_force_n) > .01 or controller.read_state().gripper_opening < .95:
                return dict(status="teacher-contact-or-gripper-stop", steps=len(labels))
            q = controller.read_state().joint_positions
            label = np.r_[target - previous, np.float32(1)].astype(np.float32)
            normalized = action_adapter.normalize(label, strict=True)
            correction = np.r_[target - q, np.float32(1)].astype(np.float32)
            command = action_adapter.to_maniskill(correction)
            actual.append(q.copy()); poses.append(latest.base_from_tcp.copy()); times.append(latest.timestamp_s)
            controller.send_action(command)
            labels.append(normalized); commands.append(target.copy())
            previous = target.copy()
            if controller.episode_done:
                return dict(status="teacher-terminated", steps=len(labels))
            latest = _read_observation_v2_frame(controller.last_step_output[0], base,
                observation_adapter, spec, control_step=tick + len(labels))
        if np.max(latest.finger_force_n) > .01:
            return dict(status="teacher-contact-stop", steps=len(labels))
        np.savez_compressed(output / "sample.npz", rgb_external=frame.rgb_external,
            rgb_wrist=frame.rgb_wrist, physical_proprio=frame.physical_proprio,
            memory_features=np.asarray(snapshot.features, np.float32),
            normalized_action=np.stack(labels), commanded_joint_target_rad=np.stack(commands),
            initial_previous_command_q_rad=initial_reference,
            actual_joint_position_rad=np.stack(actual), base_from_tcp=np.stack(poses),
            timestamp_s=np.asarray(times), initial_base_from_wrist_camera=frame.base_from_wrist_camera)
        (output / "sample.json").write_text(json.dumps(dict(snapshot=asdict(snapshot), memory_config=asdict(memory.config),
            steps=16, sample_schema="memory-conditioned-v1-home-snapshot/v1",
            labels="normalized target delta; first step relative to explicit initial command reference, subsequent steps relative to previous commanded target",
            reference_reset="new sequence after HOME action history reset; initialize from actual q",
            teacher_uses_privileged_object_pose=True, model_inputs_use_privileged_pose=False), indent=2)+"\n")
        return dict(status="captured", steps=16, sample="sample.npz")
    finally:
        # 保留失败场景已发送动作；部分轨迹不进入 16 步训练样本。
        (output / "teacher-attempt.json").write_text(json.dumps(dict(
            sent_steps=len(labels), normalized_labels=[v.tolist() for v in labels],
            commanded_targets=[v.tolist() for v in commands],
            actual_q_before_send=[v.tolist() for v in actual], timestamp_s=times), indent=2)+"\n")
        planner.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(seeds=SEEDS, train_seeds=SEEDS[:8], development_seeds=SEEDS[8:],
        maximum_collection_seconds=600, gpu_total_seconds=1800, disk_limit_bytes=4_000_000_000,
        train_steps_per_arm=32, batch_size=1, learning_rate=1e-5, sampling_seed=42,
        primary_metric="paired development normalized action flow MSE; fixed noise and time",
        train_scope="frozen Qwen and adapter; train Expert, candidate also MemoryEncoder",
        selection="last step only; no seed replacement or threshold adjustment",
        conclusion="M0 offline development signal only; no task success claim")
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    started = time.monotonic()
    records = []
    for seed in SEEDS:
        if time.monotonic() - started > 550:
            records.append(dict(seed=seed, status="budget-not-run")); continue
        try:
            result = run(args.bundle, args.output / str(seed), development_seed=seed, after_commit=capture)
            records.append(dict(seed=seed, status="completed", result=result))
        except Exception as error:
            records.append(dict(seed=seed, status="error", error_type=type(error).__name__, error=str(error)))
        (args.output / "collection.json").write_text(json.dumps(dict(records=records,
            elapsed_s=time.monotonic()-started), indent=2)+"\n")
    print(json.dumps(dict(attempted=len(records), captured=sum(
        r.get("result", {}).get("capture", {}).get("status") == "captured"
        for r in records if r.get("result", {}).get("capture") is not None))))


if __name__ == "__main__":
    main()
