"""独立技能策略的教师检查、开发评估和纠偏采集。"""
import argparse
import json
from pathlib import Path
import shutil
import signal

import numpy as np
import torch

from experiments.seven_skill_dagger.runtime import (
    SkillTeacher, Budget, rollout, corrective_labels, slice_arrays, summarize,
)
from experiments.skill_hierarchy.contract import SKILLS, ENTRY, EXIT, violations, contract_document
from experiments.skill_hierarchy.full_task import verify_upstream
from experiments.skill_dagger.data import training_seeds, save_arrays
from experiments.tcp_atomic_skills.runtime import load_parent
from experiments.tcp_atomic_skills.protocol import save, sha, identity, verify_source
from experiments.tcp_memory_control.kinematics import TCPKinematics
from robot_vla.sim.collector import EpisodeRejected
from robot_vla.tasks.pick_place import build_pick_place_task

PARENT_SHA='c0059309aa4a2dc90f1c498f8716227adc04f888193c72900fb0cb8428a3efea'
DEV=tuple(range(1811000,1811008))


def load_student(args):
    old=json.loads((args.metric_run/'config.json').read_text())
    digest=sha(args.student)
    if args.expected_sha and digest!=args.expected_sha:
        raise ValueError('学生checkpoint SHA不符')
    payload=torch.load(args.student,map_location='cpu',weights_only=True)
    if not payload.get('completed') or payload.get('format') not in (
            'skill-dagger-r1-checkpoint/v1','independent-skill-checkpoint/v1'):
        raise ValueError('必须使用完整且受支持的学生checkpoint')
    if payload['format']=='independent-skill-checkpoint/v1' and payload['skill_id'] != args.skill:
        raise ValueError('checkpoint与目标技能不符')
    base,policy,upstream=load_parent(args); verify_upstream(upstream,old['upstream'])
    policy.expert.load_state_dict(payload['expert'],strict=True)
    policy.memory_encoder.load_state_dict(payload['memory_encoder'],strict=True)
    for module in (policy.context_encoder,policy.adapter,policy.memory_encoder):module.requires_grad_(False)
    policy.eval()
    return base,policy,payload,dict(old,parent_sha256=digest)


def execute(args):
    args.output.mkdir(parents=True,exist_ok=False)
    source=verify_source(args.source_manifest)
    budget=Budget(args.output,args.control_limit,args.wall_seconds)
    def stop(*_):budget.stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    seeds=training_seeds(args.collection,args.seed_start+args.seed_count)[args.seed_start:] if args.mode=='collect' else list(DEV[:args.seed_count])
    fractions=((.25,.75,None) if args.late_takeover else (.25,.5,.75)) if args.mode=='collect' else (None,)
    rows=[dict(seed=s,skill=i,case=('until-stop' if args.mode=='collect' else 'standard') if f is None else f'progress-{f}',fraction=f,
               split='train' if args.mode=='collect' else 'development',status='not_run',success=False)
          for s in seeds for i in (range(7) if args.skill is None else [args.skill]) for f in fractions]
    rows=[r for n,r in enumerate(rows) if n%args.shards==args.shard]
    result=dict(schema='seven-skill-dagger/v1',mode=args.mode,status='running',source_sha256=source,
        checkpoint_sha256=None if args.mode=='teacher' else sha(args.student),shard=args.shard,shards=args.shards,
        contract=contract_document(),records=rows,student_limit=160,recovery_limit=160,
        acceptance='diagnostic-eight-scenes; not full 32-scene or perturbed acceptance',
        collection_guard='approach/align opening command below .8: takeover before sending; collection only',
        seed_start=args.seed_start,late_takeover=args.late_takeover)
    filename='collection.json' if args.mode=='collect' else 'result.json'
    def persist():
        result['skills']=summarize(rows);save(args.output/filename,result);budget.save()
    persist()
    try:
        base=policy=None
        if args.mode!='teacher':base,policy,_,_=load_student(args)
        fk=TCPKinematics()
        for row in rows:
            budget.check();budget.episodes+=1
            if args.mode=='collect' and shutil.disk_usage(args.output).free<6*1024**3:
                raise RuntimeError('纠偏数据磁盘余量不足6GiB')
            folder=args.output/f"{row['seed']}-{row['skill']}-{row['case']}";folder.mkdir()
            row['status']='preparing';persist();skill=row['skill']
            teacher=None;start=takeover=None;controller=None
            try:
                with SkillTeacher(None,max_episode_steps=900) as teacher:
                    teacher.budget=budget;teacher.initialize(row['seed']);teacher.run_until(skill)
                    row.update(entry=teacher.metrics.copy(),preparation_steps=len(teacher.rows))
                    if violations(ENTRY[SKILLS[skill]],teacher.metrics):
                        row['status']='entry_invalid';continue
                    start=len(teacher.session.recorder.action)
                    row['instruction']=build_pick_place_task(row['seed']%3).instruction
                    if args.mode=='teacher':
                        success=teacher.recover_skill(skill)
                        row.update(status='completed',success=success,teacher_steps=len(teacher.rows)-start,
                                   final=teacher.metrics.copy(),fine_completed=teacher.boundaries.active)
                    else:
                        controller=rollout(base,policy,teacher,fk,skill,160,row['instruction'],folder,
                                           record=args.mode=='collect',fraction=row['fraction'])
                        row.update(status='completed',**controller.result(),fine_completed=teacher.boundaries.active,
                                   final=teacher.metrics.copy(),exit_violations=violations(EXIT[SKILLS[skill]],teacher.metrics))
                        if args.mode=='collect':
                            row['student_result']=controller.result();row['success']=False
                            if controller.stop_reason=='success' or teacher.session.done:
                                row['status']='no_candidate'
                            elif controller.stop_reason not in ('step-budget-exhausted','collection-progress-reached','collection-before-close'):
                                row['status']='prefix_failed'
                            else:
                                takeover=len(teacher.session.recorder.action)
                                obs=controller.online();handoff=(obs.rgb_external.copy(),obs.rgb_wrist.copy(),obs.physical_proprio.copy())
                                row['takeover_state']=teacher.metrics.copy()
                                recovered=teacher.recover_skill(skill)
                                row.update(status='recovered' if recovered else 'teacher_failed',success=recovered)
                                if recovered:
                                    arrays=teacher.session.recorder.build()
                                    if not (np.array_equal(arrays.rgb_external[takeover],handoff[0])
                                            and np.array_equal(arrays.rgb_wrist[takeover],handoff[1])
                                            and np.allclose(arrays.proprio[takeover],handoff[2],atol=1e-7,rtol=0)):
                                        raise ValueError('交接观测不一致')
                                    row['handoff_parity']=True
                    row['final']=teacher.metrics.copy()
            except EpisodeRejected as exc:
                row.update(status='teacher_failed' if takeover is not None else 'preparation_failed',error=str(exc),success=False)
            except RuntimeError as exc:
                # 合同失败是已知负样本；OOM及其他工程错误不能被吞作技能失败。
                if isinstance(exc,torch.cuda.OutOfMemoryError) or not any(
                    marker in str(exc) for marker in ('入口不合法','教师路径及等待后未完成',': grasp_lost',': held',': opening')):
                    raise
                row.update(status='teacher_failed' if takeover is not None else 'preparation_failed',error=str(exc),success=False)
            finally:
                if args.mode=='collect' and teacher is not None and start is not None:
                    recorder=teacher.session.recorder
                    if len(recorder.action)>start and not recorder._pending_transition:
                        arrays=slice_arrays(recorder.build(),start)
                        path=folder/'trajectory.npz';save_arrays(path,arrays)
                        row.update(trajectory=str(path.relative_to(args.output)),trajectory_sha256=sha(path),
                                   recording_start_tick=start)
                        if row['status']=='recovered':
                            row.update(takeover=takeover-start,teacher_end=arrays.num_steps)
                            ids=[r['fine_skill_id'] for r in teacher.rows[start:]]
                            try:
                                labels=corrective_labels(arrays,row['takeover'],row['teacher_end'],ids,fk)
                                path=folder/'labels.npz';np.savez_compressed(path,**labels)
                                row.update(labels=str(path.relative_to(args.output)),labels_sha256=sha(path),windows=len(labels['anchor']))
                            except ValueError as exc:
                                row.update(status='label_invalid',error=str(exc),success=False)
                    boundary_path=folder/'boundaries.json'
                    save(boundary_path,dict(rows=teacher.rows,events=teacher.boundaries.events))
                    row.update(boundaries=str(boundary_path.relative_to(args.output)),boundaries_sha256=sha(boundary_path))
                persist()
            print(json.dumps({k:row.get(k) for k in ('seed','skill','case','status','success','policy_steps','windows')}),flush=True)
        result['status']='completed';persist()
    except BaseException as exc:
        result.update(status='stopped' if isinstance(exc,InterruptedError) else 'error',error=f'{type(exc).__name__}: {exc}')
        persist();raise


def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','model-cache','continued','parent-training','collection','metric-run','student','output','source-manifest'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-sha',default=PARENT_SHA)
    p.add_argument('--skill',type=int,choices=range(7))
    p.add_argument('--mode',choices=('teacher','evaluate','collect'),required=True)
    p.add_argument('--seed-count',type=int,default=8)
    p.add_argument('--seed-start',type=int,default=0)
    p.add_argument('--late-takeover',action='store_true')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--control-limit',type=int,default=150000);p.add_argument('--wall-seconds',type=int,default=14400)
    args=p.parse_args()
    if not 1<=args.seed_count<=8 or not 0<=args.shard<args.shards:
        raise ValueError('无效场景数或分片')
    if args.seed_start<0 or args.seed_start+args.seed_count>8 or (args.mode!='collect' and (args.seed_start or args.late_takeover)):
        raise ValueError('仅采集允许前8个train场景内偏移和late接管')
    if args.mode != 'teacher' and args.skill is None:
        raise ValueError('独立策略必须明确指定skill')
    execute(args)


if __name__=='__main__':main()
