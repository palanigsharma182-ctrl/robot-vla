"""共享七技能学生的完整任务 development 闭环；真实初态、无教师接管。"""
import argparse
import json
from pathlib import Path
import signal
import time

import torch

from experiments.skill_hierarchy.collect import DEV_SEEDS
from experiments.skill_hierarchy.probe import HierarchyTeacher
from experiments.skill_hierarchy.replay import ReplayController
from experiments.skill_hierarchy.train import FORMAT
from experiments.tcp_atomic_skills.protocol import save,sha,identity,verify_source
from experiments.tcp_atomic_skills.runtime import load_parent,predict,AtomicExecutor,execute_plan
from experiments.tcp_memory_control.kinematics import TCPKinematics
from experiments.tcp_memory_control.protocol import sampling_seed
from robot_vla.sim.collector import AtomicPreparation
from robot_vla.tasks.pick_place import build_pick_place_task

SEEDS=DEV_SEEDS[:4]
MAX_STEPS=400


def load_student(policy, path, result, config):
    if (result['status']!='completed' or not result['strict_reload'] or config['smoke']
            or result['configuration_identity']!=identity(config)
            or result['steps']!=config['steps'] or sha(path)!=result['checkpoint_sha256']):
        raise ValueError('完整任务评估必须使用已验证的完整训练阶段')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if (payload['format']!=FORMAT or payload['configuration_identity']!=identity(config)
            or not payload['completed'] or payload['step']!=config['steps']):
        raise ValueError('七技能 checkpoint 身份不符')
    policy.expert.load_state_dict(payload['expert'],strict=True)
    policy.memory_encoder.load_state_dict(payload['memory_encoder'],strict=True)
    policy.eval()


def summarize(rows):
    return dict(episodes=len(rows),successes=sum(bool(r.get('success')) for r in rows),
        canonical_successes=sum(bool(r.get('canonical_success')) for r in rows),
        failures_by_boundary={str(i):sum(not r.get('success',False) and r.get('seven_completed')==i
                                       for r in rows) for i in range(8)},
        total_policy_steps=sum(r.get('policy_steps',0) for r in rows))


def main():
    p=argparse.ArgumentParser()
    for key in ('training','checkpoint','model-cache','continued','parent-training','output','source-manifest'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();source=verify_source(args.source_manifest)
    config=json.loads((args.training/'config.json').read_text())
    trained=json.loads((args.training/'result.json').read_text())
    args.output.mkdir(exist_ok=False)
    rows=[dict(seed=s,status='not_run',success=False) for s in SEEDS]
    result=dict(status='running',data_use='tuning development; four fixed known scenes; no final test',
        source_sha256=source,training_result_sha256=sha(args.training/'result.json'),
        checkpoint_sha256=trained['checkpoint_sha256'],max_steps_per_episode=MAX_STEPS,
        scope='complete Pick-Carry-Place from true initial state; student actions only',
        records=rows)
    started=time.monotonic()
    def persist():
        result.update(elapsed_s=time.monotonic()-started,summary=summarize(rows))
        save(args.output/'result.json',result)
    def stop(*_):raise InterruptedError('停止完整任务开发检查，保留全部分母')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);persist()
    try:
        base,policy,upstream=load_parent(args)
        if upstream!=config['upstream']:raise ValueError('完整任务评估上游身份改变')
        load_student(policy,args.training/'latest.pt',trained,config);fk=TCPKinematics()
        for row in rows:
            folder=args.output/str(row['seed']);folder.mkdir();ctrl=None;plans=[]
            row['status']='running';persist()
            # 每个场景重新建仿真环境，不复用 reset 后的旧接触缓存。
            with HierarchyTeacher(None,max_episode_steps=MAX_STEPS) as teacher:
                try:
                    teacher.initialize(row['seed'])
                    session=teacher.session
                    preparation=AtomicPreparation(session.observation,session.tracker,session.progress,0)
                    ctrl=ReplayController(teacher,preparation,5,MAX_STEPS,
                        build_pick_place_task(row['seed']%3).instruction,folder)
                    save(folder/'initial.json',ctrl.audit());executor=AtomicExecutor(fk)
                    while ctrl.stop_reason is None:
                        anchor=fk.pose_base(ctrl.read_state().joint_positions)
                        seed=sampling_seed(row['seed'],len(plans))
                        physical=predict(base,policy,ctrl.online(),seed)
                        plan=dict(step=ctrl.steps,sampling_seed=seed,physical=physical.tolist(),
                                  base_from_tcp=anchor.tolist(),memory_available=False)
                        plans.append(plan)
                        plan['execution']=execute_plan(executor,ctrl,physical,anchor)
                    row.update(status='completed',**ctrl.result())
                except torch.cuda.OutOfMemoryError:
                    raise
                except (RuntimeError,ValueError) as exc:
                    if ctrl is not None:
                        ctrl.stop_reason='execution-or-boundary-failure';row.update(**ctrl.result())
                    row.update(status='failed',success=False,error=f'{type(exc).__name__}: {exc}')
                finally:
                    boundaries=getattr(teacher,'boundaries',None)
                    row.update(seven_completed=getattr(boundaries,'active',0),
                               boundary_events=getattr(boundaries,'events',[]),
                               canonical_success=bool(ctrl is not None and ctrl.progress.completed_skill_count>=5))
                    row['success']=bool(row.get('success') and row['seven_completed']==7
                                        and row['canonical_success'])
                    save(folder/'plans.json',plans);persist()
        result['status']='completed'
    except BaseException as exc:
        result.update(status='error',error=f'{type(exc).__name__}: {exc}');raise
    finally:persist()


if __name__=='__main__':main()
