"""固定新 train/development 场景：真实 HOME 后教师轨迹与逐帧 Memory 快照。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from experiments.memory_conditioning.conditioning import snapshot_memory
from experiments.memory_reobserve.session import frame_safety
from robot_vla.adapters import ActionAdapter, FrankaObservationAdapter
from robot_vla.contracts import RobotSpec

from experiments.memory_reobserve.protocol import PROTOCOL, SEEDS, Acquisition


def capture_sequence(*, seed, env, frame, memory, safety, episode, tick, output):
    """教师可读物体 GT；保存的模型输入只有传感器和合格测量产生的快照。"""
    import sapien
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
    from experiments.g2c_memory_integration.run import DevelopmentController
    from robot_vla.evaluation.maniskill import _read_observation_v2_frame

    base, spec = env.unwrapped, RobotSpec()
    adapter, observation_adapter = ActionAdapter(spec), FrankaObservationAdapter(spec)
    controller = DevelopmentController(env, spec)
    records, observations, labels, targets_sent = [], [], [], []
    hold_records=[]
    hold_tracking=bool(safety.controller_tracking_valid)
    hold_status='completed'
    try:
        for _ in range(PROTOCOL['initial_teacher_hold_steps'][str(seed)]):
            record=dict(tick_before=tick,send_returned=False,status='sending')
            hold_records.append(record)
            hold_target=controller.read_state().joint_positions.copy()
            controller.hold_current()
            tick+=1
            _,_,terminated,truncated,_=controller.last_step_output
            record.update(send_returned=True,timestamp_s=tick/spec.control_hz,
                terminated=bool(terminated.item()),truncated=bool(truncated.item()))
            frame=_read_observation_v2_frame(controller.last_step_output[0],base,
                observation_adapter,spec,control_step=tick)
            hold_tracking=bool(np.isfinite(frame.physical_proprio).all() and np.all(
                np.abs(frame.physical_proprio[:7]-hold_target)<=np.asarray(spec.effective_joint_delta_limits_rad)))
            contact=bool(np.max(frame.finger_force_n)>.01)
            gripper_open=bool(frame.physical_proprio[-1]>=.975)
            hold_status=('initial-hold-terminated' if record['terminated'] else
                'initial-hold-truncated' if record['truncated'] else
                'initial-hold-contact' if contact else
                'initial-hold-gripper-closed' if not gripper_open else
                'initial-hold-tracking-failed' if not hold_tracking else 'completed')
            record.update(proprio=frame.physical_proprio.tolist(),force=frame.finger_force_n.tolist(),
                contact_detected=contact,gripper_open=gripper_open,tracking_valid=hold_tracking,status=hold_status)
            if hold_status!='completed':
                return dict(status=hold_status,samples=0,hold_steps=len(hold_records))
    except Exception:
        hold_status='initial-hold-implementation-error'
        raise
    finally:
        (output/'teacher-holds.json').write_text(json.dumps(dict(ending_status=hold_status,
            records=hold_records),indent=2)+'\n')
    safety=frame_safety(frame,tracking_valid=hold_tracking)
    initial = controller.read_state().joint_positions.copy()
    previous, latest = initial.copy(), frame
    planner = PandaArmMotionPlanningSolver(env, debug=False, vis=False,
        base_pose=base.agent.robot.pose, visualize_target_grasp_pose=False,
        print_env_info=False, joint_vel_limits=.4, joint_acc_limits=.4)
    status = 'teacher-plan-rejected'
    rejected = None
    try:
        approaching = np.array([0.,0.,-1.])
        closing = base.agent.tcp.pose.to_transformation_matrix()[0,:3,1].cpu().numpy()
        grasp = compute_grasp_info_by_obb(get_actor_obb(base.cube), approaching=approaching,
                                        target_closing=closing, depth=.025)
        pose = base.agent.build_grasp_pose(approaching, grasp['closing'], base.cube.pose.sp.p)
        plan = planner.move_to_pose_with_screw(pose * sapien.Pose([0,0,-.05]), dry_run=True)
        if not isinstance(plan, dict) or plan.get('status') != 'Success':
            return dict(status=status, samples=0)
        targets = np.asarray(plan['position'], np.float32)
        if targets.ndim != 2 or targets.shape[1] != 7 or not np.isfinite(targets).all():
            status = 'teacher-invalid-path'
            raise ValueError('教师路径必须为 [T,7]')
        status = 'captured'
        for target in targets[:PROTOCOL['maximum_teacher_steps']]:
            if np.max(latest.finger_force_n) > .01 or controller.read_state().gripper_opening < .975:
                status = 'teacher-contact-stop'; break
            # 使用实际上一命令的跟踪误差；初帧沿用已验证的 HOME safety。
            actual = controller.read_state().joint_positions.copy()
            tracking = bool(np.isfinite(actual).all() and np.all(
                np.abs(actual-previous) <= np.asarray(spec.effective_joint_delta_limits_rad)))
            safe = safety if not labels else frame_safety(latest, tracking_valid=tracking)
            snapshot = snapshot_memory(memory.state, memory.config, safe,
                episode_id=episode, timestamp_s=latest.timestamp_s)
            label_delta, correction = target-previous, target-actual
            limits = np.asarray(spec.effective_joint_delta_limits_rad)
            bounds = np.asarray(spec.joint_position_limits_rad)
            failure = ('teacher-invalid-state' if not np.isfinite(actual).all() else
                'teacher-target-stop' if np.any(target < bounds[:,0]) or np.any(target > bounds[:,1]) else
                'teacher-label-stop' if np.any(np.abs(label_delta)>limits) else
                'teacher-tracking-stop' if np.any(np.abs(correction)>limits) else None)
            if failure:
                status = failure
                rejected = dict(target=target.tolist(),label_delta=label_delta.tolist(),
                    actual_correction=correction.tolist())
                break
            label = adapter.normalize(np.r_[label_delta, np.float32(1)].astype(np.float32), strict=True)
            command = adapter.to_maniskill(np.r_[correction, np.float32(1)].astype(np.float32))
            controller.send_action(command)
            observations.append(latest)
            records.append(dict(snapshot=asdict(snapshot), actual_q_before=actual.tolist(),
                previous_command_q=previous.tolist(), tracking_valid=tracking))
            labels.append(label); targets_sent.append(target.copy()); previous=target.copy()
            latest = _read_observation_v2_frame(controller.last_step_output[0], base,
                observation_adapter, spec, control_step=tick+len(labels))
            if controller.episode_done:
                status='teacher-terminal'; break
            if np.max(latest.finger_force_n)>.01 or controller.read_state().gripper_opening<.975:
                status='teacher-contact-stop'; break
        if len(labels)<16 and status=='captured':
            status='insufficient-horizon'
        anchors = list(range(0, len(labels)-PROTOCOL['horizon']+1, PROTOCOL['anchor_stride']))
        if anchors:
            np.savez_compressed(output/'sequence.npz',
                rgb_external=np.stack([f.rgb_external for f in observations]),
                rgb_wrist=np.stack([f.rgb_wrist for f in observations]),
                physical_proprio=np.stack([f.physical_proprio for f in observations]),
                base_from_tcp=np.stack([f.base_from_tcp for f in observations]),
                base_from_wrist_camera=np.stack([f.base_from_wrist_camera for f in observations]),
                timestamp_s=np.array([f.timestamp_s for f in observations]),
                memory_features=np.array([r['snapshot']['features'] for r in records],np.float32),
                memory_available=np.array([r['snapshot']['available'] for r in records],bool),
                normalized_action=np.stack(labels), commanded_joint_target_rad=np.stack(targets_sent),
                initial_previous_command_q_rad=initial, anchors=np.array(anchors,np.int64))
            (output/'sequence.json').write_text(json.dumps(dict(schema='memory-reobserve-sequence/v1',
                seed=seed,episode_id=episode,initial_hold_steps=len(hold_records),
                ending_status=status, includes_executed_contact_ending_action=status=='teacher-contact-stop',
                memory_config=asdict(memory.config), records=records,
                teacher_uses_privileged_object_pose=True, model_inputs_use_privileged_pose=False,
                camera_pose_source='actual settled HOME; provider audit in camera.jsonl',
                label='commanded-target delta; only executed targets, no hypothetical future labels'),indent=2)+'\n')
        return dict(status=status, samples=len(anchors), sent_steps=len(labels),
            available_anchors=sum(records[i]['snapshot']['available'] for i in anchors))
    except Exception:
        if status=='captured':status='implementation-error'
        raise
    finally:
        (output/'teacher-attempt.json').write_text(json.dumps(dict(status=status,
            rejected=rejected, sent_steps=len(labels), records=records, normalized_labels=[x.tolist() for x in labels],
            commanded_targets=[x.tolist() for x in targets_sent]),indent=2)+'\n')
        planner.close()


def main():
    from experiments.g2c_memory_integration.run import run
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2)+'\n')
    started=time.monotonic(); records=[]
    for seed in SEEDS:
        if time.monotonic()-started > PROTOCOL['maximum_collection_seconds']-30:
            record=dict(seed=seed,status='budget-not-run')
        else:
            try:
                result=run(args.bundle,args.output/str(seed),acquisition=Acquisition(seed))
                record=dict(seed=seed,status='completed',result=result)
            except Exception as error:
                record=dict(seed=seed,status='error',error_type=type(error).__name__,error=str(error))
        records.append(record)
        (args.output/'collection.json').write_text(json.dumps(dict(records=records,
            elapsed_s=time.monotonic()-started),indent=2)+'\n')
    errors=sum(r['status']=='error' for r in records)
    not_run=sum(r['status']=='budget-not-run' for r in records)
    print(json.dumps(dict(planned=len(SEEDS),started=len(records)-not_run,not_run=not_run,errors=errors,
        anchors=sum((r.get('result',{}).get('capture') or {}).get('samples',0) for r in records))))
    if errors or not_run:raise SystemExit(2)


if __name__=='__main__':main()
