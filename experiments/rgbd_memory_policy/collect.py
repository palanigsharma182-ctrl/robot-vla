"""真实执行教师轨迹，保存动作前Memory；不从未来帧补齐当前输入。"""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from experiments.oracle_reach_control.runner import position_step, SETTINGS
from experiments.rgbd_memory_policy.protocol import PROTOCOL, identity
from experiments.rgbd_memory_policy.stream import make_env, setup_scene, RGBDController
from robot_vla.adapters import ActionAdapter
from robot_vla.contracts import RobotSpec
from robot_vla.diagnostics.oracle_reach import FrankaTCPForwardKinematics, find_maniskill_panda_urdf


def collect(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2)+'\n')
    ledger=[dict(seed=s,status='not_run',split='train' if s in PROTOCOL['train_seeds'] else 'development') for s in PROTOCOL['seeds']]
    def save():
        (output/'collection.json').write_text(json.dumps(dict(protocol_sha256=identity(PROTOCOL),records=ledger),indent=2)+'\n')
    save();started=time.monotonic();env=make_env();spec=RobotSpec();adapter=ActionAdapter(spec)
    fk=FrankaTCPForwardKinematics(find_maniskill_panda_urdf(),spec)
    try:
        for entry in ledger:
            seed=entry['seed'];folder=output/str(seed);folder.mkdir();(folder/'frames').mkdir()
            entry['status']='running';save()
            inputs=[];labels=[];targets=[];snapshots=[];previous_records=[]
            try:
                position=setup_scene(env,seed)
                controller=RGBDController(env,f'rgbd-policy-train-{seed}',folder)
                controller.warmup(position)
                # 教师使用GT目标位置；学生只得到双图、proprio、预测Memory。
                target_world=position+np.array([0,0,PROTOCOL['approach_offset_m']])
                previous=controller.last_target.copy();initial=previous.copy()
                for _ in range(PROTOCOL['teacher_steps']):
                    if controller.episode_done or controller.closed or not controller.rows[-1]['tracking_valid']:break
                    current=controller.online();snapshot=controller.snapshot
                    q=controller.read_state().joint_positions.copy()
                    if np.linalg.norm(fk(q)-controller.env.unwrapped.agent.tcp_pose.p.cpu().numpy()[0])>.001:
                        raise ValueError('教师FK与实际TCP frame不一致')
                    desired=q+position_step(fk,q,target_world,limits=adapter.delta_limits,**SETTINGS)
                    bounds=np.asarray(spec.joint_position_limits_rad)
                    desired=np.clip(desired,bounds[:,0]+1e-5,bounds[:,1]-1e-5)
                    # 先落到执行器的float32目标，再计算和保存label，避免存盘量化改变参考。
                    target=(previous+np.clip(desired-previous,-adapter.delta_limits,adapter.delta_limits)).astype(np.float32)
                    correction=target-q
                    if np.any(np.abs(correction)>adapter.delta_limits+1e-7):
                        raise ValueError('教师跟踪误差超过命令执行范围')
                    label=adapter.normalize(np.r_[target-previous,1.].astype(np.float32),strict=True)
                    controller.send_action(adapter.to_maniskill(np.r_[correction,1.].astype(np.float32)))
                    # 只有实际发出的命令才构成监督；snapshot来自发送前。
                    inputs.append(current);snapshots.append(asdict(snapshot));labels.append(label)
                    targets.append(target.copy());previous_records.append(previous.copy());previous=target.copy()
                n=len(labels)
                if n>=PROTOCOL['horizon']:
                    np.savez_compressed(folder/'sequence.npz',
                        rgb_external=np.stack([x.rgb_external for x in inputs]),
                        rgb_wrist=np.stack([x.rgb_wrist for x in inputs]),
                        physical_proprio=np.stack([x.physical_proprio for x in inputs]),
                        normalized_action=np.asarray(labels,np.float32),commanded_joint_target_rad=np.asarray(targets,np.float32),
                        previous_command_q_rad=np.asarray(previous_records,np.float32),initial_previous_command_q_rad=initial,
                        memory_features=np.asarray([s['features'] for s in snapshots],np.float32),
                        memory_available=np.asarray([s['available'] for s in snapshots],bool),
                        timestamp_s=np.asarray([s['timestamp_s'] for s in snapshots]),
                        anchors=np.arange(0,n-15,PROTOCOL['anchor_stride']))
                (folder/'sequence.json').write_text(json.dumps(dict(episode_id=controller.episode,
                    protocol_sha256=identity(PROTOCOL),snapshots=snapshots,memory_config=asdict(controller.replay.config),
                    teacher_uses_gt=True,student_uses_gt=False),indent=2)+'\n')
                safely_completed=n==PROTOCOL['teacher_steps'] and controller.stop_reason is None
                entry.update(status='completed' if safely_completed else 'stopped',ending_reason=controller.stop_reason,
                    steps=n,samples=len(range(0,n-15,PROTOCOL['anchor_stride'])),
                    available_anchors=sum(snapshots[i]['available'] for i in range(0,n-15,PROTOCOL['anchor_stride'])),
                    final_teacher_distance_m=float(np.linalg.norm(fk(controller.read_state().joint_positions)-target_world)),
                    commits=sum(r['memory']['accepted'] for r in controller.rows))
            except Exception as e:
                entry.update(status='error',error_type=type(e).__name__,error=str(e))
                raise
            finally:save()
            print(json.dumps(entry),flush=True)
    finally:env.close()
    return dict(elapsed_s=time.monotonic()-started,records=ledger)
