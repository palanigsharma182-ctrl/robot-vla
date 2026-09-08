"""metric-bc-v2 定长 A/B；每块评估驱动下一块分布，不恢复旧 supervisor。"""
from __future__ import annotations

import argparse
import os
import copy
import hashlib
import json
from pathlib import Path
import signal
import shutil
import time

import numpy as np
import torch

from experiments.skill_hierarchy.data import SevenSkillWindows
from experiments.skill_hierarchy.full_task import load_student, verify_upstream
from experiments.skill_hierarchy.metric_rules import VERSION, CHANNELS, FeedbackSampler, CommandEvents, group_index, aggregate_errors
from experiments.skill_hierarchy.metric_rollout import RunBudget, evaluate_closed
from experiments.skill_hierarchy.train import FeatureCache, balanced_schedule, restore_continuation, SEED
from experiments.tcp_atomic_skills.train import assess, loss_for
from experiments.tcp_atomic_skills.runtime import load_parent
from experiments.tcp_atomic_skills.protocol import save, sha, identity, verify_source
from experiments.tcp_memory_control.kinematics import TCPKinematics
from experiments.tcp_memory_control.protocol import sampling_seed

FEEDBACK=tuple(range(1811000,1811008))
RESERVED=tuple(range(1811008,1811032))
FORMAT='metric-bc-checkpoint/v2'


def model_digest(policy):
    h=hashlib.sha256()
    for name,t in sorted(policy.expert.state_dict().items()):
        h.update(name.encode()); h.update(t.detach().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def training_support(dataset,collection,fk,budget):
    """从 train 实际 FK 与 sidecar 重建小型状态邻域，不读取开发 RGB。"""
    manifest=json.loads((collection/'collection.json').read_text())
    records={r['trajectory_id']:r for r in manifest['records'] if 'trajectory_id' in r}
    support=[]
    for entry in dataset.entries:
        budget.check(); record=records[entry.trajectory_id]
        sidecar_path=collection/'sidecars'/f'{entry.trajectory_id}.json'
        if not sidecar_path.exists():
            # 采集 manifest 可能提供显式相对字段；不能猜另一个轨迹。
            field=next((record[k] for k in ('sidecar_file','sidecar') if k in record),None)
            if field is None: raise ValueError('缺少 train sidecar')
            sidecar_path=collection/field
        side=json.loads(sidecar_path.read_text())
        if side['trajectory_id']!=entry.trajectory_id: raise ValueError('sidecar 轨迹身份不符')
        arrays=dataset.store.get(entry)
        target=np.array(side['world_from_pregrasp']); pre=np.linalg.inv(fk.world_from_base)@target
        cmds=CommandEvents(); previous=None
        for t,row in enumerate(side['rows']):
            pose=fk.pose_base(arrays.proprio[t,:7]).astype(float)
            direction=pre[:3,3]-pose[:3,3]; length=np.linalg.norm(direction)
            speed=None if previous is None or length<.001 else float((pose[:3,3]-previous[:3,3])@direction/(.05*length))
            if row['fine_skill_id']<=3 and speed is not None:
                support.append(dict(seed=entry.randomization['seed'],held=row['before']['held'],
                    command_state=cmds.state,relative_tcp=np.linalg.inv(pre)@pose,radial_velocity=speed))
            cmds.observe(row['gripper_command'],t)
            previous=pose
    return support


def offline(policy,cache,dataset,groups,seeds,budget,output):
    records=[]
    selected=set(seeds)
    for i,(entry,tick) in enumerate(dataset.index):
        if dataset.entries[entry].randomization['seed'] not in selected: continue
        budget.check(); x=cache.get('val',i); valid=x['mask'].clone()
        y=dict(x); y['mask']=torch.ones_like(valid)
        pred=assess(policy,[y])[0]
        # assess 的全 mask 对 episode 尾部含无效标签，重新仅在真实有效前缀计分。
        from experiments.tcp_atomic_skills.data import action_spec
        actual=action_spec().denormalize(x['action'][0].cpu().numpy())
        predicted=np.array(pred['predicted_first4']); mask=valid[0,:4].cpu().numpy()
        error=(predicted-actual[:4])[mask]
        pred.update(index=i,translation_error_mm=float(np.linalg.norm(error[:,:3],axis=1).mean()*1000),
            rotation_delta_error_deg=float(np.linalg.norm(error[:,3:6],axis=1).mean()*180/np.pi),
            gripper_mae=float(np.abs(error[:,6]).mean()),
            gripper_binary_accuracy=float(((predicted[:,6][mask]>=.5)==(actual[:4,6][mask]>=.5)).mean()))
        records.append(pred)
    result=aggregate_errors(records,groups)
    save(output,dict(groups=result,records=records,seeds=list(seeds),mask='online all valid input; valid labels only'))
    return result


def verify_reuse_identity(previous, current):
    """按 JSON 身份比较实际配置与落盘配置，保留 tuple/list 等价性。"""
    for key in ('parent_sha256','data','upstream','feedback','reserved','deadline_unix','training_sdpa','deterministic_algorithms'):
        if identity(previous[key]) != identity(current[key]):
            raise ValueError(f'复用基线身份改变: {key}')


def verify_resume_config(previous,current,complete_planned_run=False):
    """只有显式完成计划模式允许移除旧墙钟限制，训练参数仍冻结。"""
    excluded={'source'}
    if complete_planned_run:
        if current['deadline_unix'] is not None: raise ValueError('完成计划模式仍有墙钟截止')
        excluded.add('deadline_unix')
    old={k:v for k,v in previous.items() if k not in excluded}
    new={k:v for k,v in current.items() if k not in excluded}
    if identity(old)!=identity(new): raise ValueError('恢复实验配置改变')


def verify_resume_checkpoint(payload,previous):
    if (payload['format']!=FORMAT or payload['configuration_identity']!=identity(previous)
            or payload['branch']!='A' or payload['step'] not in (512,1024) or payload['completed']):
        raise ValueError('此次恢复仅接受 A512/A1024 评估前恢复点')


def main():
    p=argparse.ArgumentParser()
    for name in ('collection','audit','teacher-replay','training','checkpoint','model-cache','continued','parent-training','output','source-manifest'):
        p.add_argument('--'+name,type=Path,required=True)
    stop_mode=p.add_mutually_exclusive_group(required=True)
    stop_mode.add_argument('--deadline-unix',type=float)
    stop_mode.add_argument('--complete-planned-run',action='store_true')
    p.add_argument('--reuse-baseline',type=Path)
    p.add_argument('--resume-run',type=Path)
    args=p.parse_args(); args.output.mkdir(exist_ok=False)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise ValueError('确定性运行要求启动前设置 CUBLAS_WORKSPACE_CONFIG=:4096:8')
    torch.use_deterministic_algorithms(True)
    budget=RunBudget(args.deadline_unix,args.output)
    if args.resume_run:
        old_budget=json.loads((args.resume_run/'budget.json').read_text())
        if old_budget['deadline_unix']!=args.deadline_unix and not args.complete_planned_run:
            raise ValueError('恢复截止时间改变')
        budget.student=old_budget['student_step_reservations']
        budget.teacher=old_budget['teacher_step_reservations']; budget.episodes=old_budget['episodes']
    def stop(*_): budget.stopped=True
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    started=time.time(); step=0; branch='preflight'; sampler=None; optimizer=None; policy=None; config_id=None; current_stage='preflight'
    def status(stage,**extra):
        nonlocal current_stage
        current_stage=stage
        save(args.output/'status.json',dict(stage=stage,branch=branch,step=step,updated_unix=time.time(),
            elapsed_s=time.time()-started,deadline_unix=args.deadline_unix,**extra))
        budget.save(); print(json.dumps(dict(stage=stage,branch=branch,step=step,**extra)),flush=True)
    def persist(path, completed=False):
        if policy is None or optimizer is None or sampler is None: return
        budget.save()
        payload=dict(format=FORMAT,configuration_identity=config_id,branch=branch,step=step,completed=completed,
            expert={k:v.detach().cpu() for k,v in policy.expert.state_dict().items()},
            memory_encoder={k:v.detach().cpu() for k,v in policy.memory_encoder.state_dict().items()},
            optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            sampler=sampler.state())
        temporary=path.with_suffix('.tmp'); torch.save(payload,temporary); temporary.replace(path)
    try:
        status('preflight'); budget.check(); source=verify_source(args.source_manifest)
        prior=json.loads((args.training/'config.json').read_text()); result=json.loads((args.training/'result.json').read_text())
        audit=json.loads((args.audit/'result.json').read_text()); replay=json.loads((args.teacher_replay/'result.json').read_text())
        if audit['status']!='passed' or replay['status']!='passed' or replay['collection_sha256']!=sha(args.collection/'collection.json'):
            raise ValueError('原数据或教师审计不匹配')
        datasets={s:SevenSkillWindows(args.collection,s) for s in ('train','val')}
        for split,d in datasets.items():
            if d.hashes!=audit['splits'][split]['files'] or d.hashes!=prior['data'][split]: raise ValueError('数据身份改变')
        base,policy,upstream=load_parent(args); verify_upstream(upstream,prior['upstream'])
        load_student(policy,args.training/'latest.pt',result,prior)
        policy.context_encoder.requires_grad_(False); policy.adapter.requires_grad_(False); policy.memory_encoder.requires_grad_(False)
        params=list(policy.expert.parameters()); optimizer=torch.optim.AdamW(params,lr=1e-5)
        parent=torch.load(args.training/'latest.pt',map_location='cpu',weights_only=True)
        offset=prior.get('training_offset',0)+prior['steps']
        groups,scenes=group_index(datasets['train']); valgroups,_=group_index(datasets['val'])
        schedule=balanced_schedule(datasets['train'].buckets,1024,SEED+offset)
        config=dict(version=VERSION,source=source,parent_sha256=result['checkpoint_sha256'],upstream=upstream,
            data=prior['data'],audit_sha256=sha(args.audit/'result.json'),feedback=list(FEEDBACK),reserved=list(RESERVED),
            steps_per_branch=1024,block=256,offset=offset,seed=SEED,learning_rate=1e-5,
            deterministic_algorithms=True,training_sdpa='math for both A and B; BF16 unchanged',
            selection='fixed final; no intermediate checkpoint selection',deadline_unix=args.deadline_unix,
            cache_identity=prior.get('cache_identity',identity(prior)),cache_root=prior.get('cache_root',str(args.training/'feature-cache')),
            group_counts=np.bincount(groups,minlength=28).tolist(),group_scene_counts=[len(np.unique(scenes[groups==g])) for g in range(28)])
        config_id=identity(config); save(args.output/'config.json',config)
        cache=FeatureCache(Path(config['cache_root']),base,datasets,config['cache_identity'],
            lambda:budget.stopped or (budget.deadline is not None and time.time()>=budget.deadline))
        fk=TCPKinematics(); status('train_support_index')
        support=training_support(datasets['train'],args.collection,fk,budget)
        save(args.output/'support-index.json',dict(count=len(support),scenes=len({r['seed'] for r in support})))
        def reset():
            nonlocal sampler,step
            step=0; restore_continuation(policy,optimizer,dict(parent,optimizer=copy.deepcopy(parent['optimizer'])),prior)
            if set(int(v['step']) for v in optimizer.state.values() if 'step' in v)!={offset}: raise ValueError('优化器恢复步数错误')
            sampler=FeedbackSampler(groups,scenes,schedule)
        def update(adaptive):
            nonlocal step
            budget.check(); policy.expert.train(); optimizer.zero_grad(set_to_none=True)
            cursor=sampler.cursor; rng=copy.deepcopy(sampler.rng.bit_generator.state)
            cpu_rng=torch.get_rng_state(); cuda_rng=torch.cuda.get_rng_state_all()
            rows=sampler.draw(step,adaptive); total=0.
            try:
                for k,(index,extra) in enumerate(rows):
                    x=cache.get('train',index)
                    # 两组统一数学 SDPA，避免反向原子累加导致配对暖身漂移。
                    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                        value=loss_for(policy,x,sampling_seed(SEED,(offset+step)*7+k))
                    if not torch.isfinite(value): raise ValueError('非有限 loss')
                    (value/7).backward(); total+=float(value.detach())/7
            except BaseException:
                # 未提交更新时回滚队列与随机流；保留的恢复点必须可再次执行同一更新。
                sampler.cursor=cursor; sampler.rng.bit_generator.state=rng
                torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
                optimizer.zero_grad(set_to_none=True)
                raise
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True); optimizer.step()
            sampler.account(rows); step+=1
            for module in (policy.context_encoder,policy.adapter,policy.memory_encoder):
                if any(q.grad is not None for q in module.parameters()): raise ValueError('冻结模块出现梯度')
            return total
        if args.resume_run:
            previous=args.resume_run
            old_config=json.loads((previous/'config.json').read_text())
            verify_resume_config(old_config,config,args.complete_planned_run)
            common=json.loads((previous/'baseline-feedback.json').read_text())
            if not common['valid'] or common['status']!='completed' or len(common['records'])!=16:
                raise ValueError('恢复基线不完整')
            # 复制小型证据到新运行；原目录、失败记录和权重均不覆盖。
            shutil.copytree(previous/'A',args.output/'A',ignore=shutil.ignore_patterns('*.pt','*.tmp'))
            for name in ('smoke.json','baseline-feedback.json','baseline-reuse.json'):
                shutil.copy2(previous/name,args.output/name)
            save(args.output/'resume.json',dict(source=str(previous),checkpoint='A/latest.pt',
                checkpoint_sha256=sha(previous/'A/latest.pt'),previous_config_identity=identity(old_config),
                retained_budget=old_budget,reason='Resume pending block feedback; preserve attempts and training state',
                complete_planned_run=args.complete_planned_run,
                deadline_change=dict(previous=old_budget['deadline_unix'],current=args.deadline_unix),
                incomplete_episodes='See archived -interrupted folders; all original attempts remain charged',
                missing_losses='Any unsaved per-step losses are represented by null'))
        elif args.reuse_baseline:
            import ast
            previous_run=args.reuse_baseline
            old_config=json.loads((previous_run/'config.json').read_text())
            verify_reuse_identity(old_config,config)
            old_source=previous_run.parent/'code-v4'
            new_source=Path.cwd()
            # 只复用未改变的模型推理、离线计分、事件与闭环协议；不凭目录名认定同源。
            for file in ('experiments/skill_hierarchy/metric_rollout.py','experiments/tcp_atomic_skills/train.py',
                         'experiments/tcp_atomic_skills/runtime.py','src/robot_vla/model/policy.py'):
                if sha(old_source/file)!=sha(new_source/file): raise ValueError('基线消费者源码改变，禁止复用')
            for file,names in [('experiments/skill_hierarchy/metric_train.py',('offline',)),
                               ('experiments/skill_hierarchy/metric_rules.py',('CommandEvents','event_groups','group_index','aggregate_errors'))]:
                def selected(root):
                    tree=ast.parse((root/file).read_text())
                    return {n.name:ast.dump(n,include_attributes=False) for n in tree.body
                            if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names}
                if selected(old_source)!=selected(new_source): raise ValueError('基线指标定义改变，禁止复用')
            smoke=json.loads((previous_run/'smoke.json').read_text())
            if smoke['status']!='passed' or smoke['records'][0]['digest']!=smoke['records'][1]['digest']:
                raise ValueError('复用 smoke 未通过')
            common=json.loads((previous_run/'baseline-feedback.json').read_text())
            if not common['valid'] or common['status']!='completed' or len(common['records'])!=16:
                raise ValueError('复用基线不完整')
            if [(r['seed'],r['start']) for r in common['records']]!=[(s,t) for s in FEEDBACK for t in ('F','S')]:
                raise ValueError('复用基线场景分母改变')
            old_budget=json.loads((previous_run/'budget.json').read_text())
            if old_budget['episodes']!=16: raise ValueError('复用范围不是完整的起点基线')
            budget.student=old_budget['student_step_reservations'];budget.teacher=old_budget['teacher_step_reservations'];budget.episodes=16
            save(args.output/'baseline-reuse.json',dict(source=str(previous_run),
                feedback_sha256=sha(previous_run/'baseline-feedback.json'),smoke_sha256=sha(previous_run/'smoke.json'),
                policy_steps=sum(r['policy_steps'] for r in common['records']),additional_updates=0,additional_episodes=0))
            save(args.output/'smoke.json',smoke);save(args.output/'baseline-feedback.json',common)
        else:
            smoke=[]
            for branch in ('smoke-A','smoke-B'):
                reset(); status('smoke')
                losses=[update(branch=='smoke-B') for _ in range(2)]
                path=args.output/f'{branch}.pt'; persist(path)
                digest=model_digest(policy); payload=torch.load(path,map_location='cpu',weights_only=True)
                policy.expert.load_state_dict(payload['expert'],strict=True)
                if model_digest(policy)!=digest: raise ValueError('smoke 权重重载不一致')
                # 固定真实输入的预测和完整采样器状态也须重载一致。
                policy.eval(); probe=cache.get('train',0); prediction=assess(policy,[probe])
                policy.expert.load_state_dict(payload['expert'],strict=True)
                if assess(policy,[probe])!=prediction: raise ValueError('smoke 重载预测不一致')
                cloned=FeedbackSampler(groups,scenes,schedule); cloned.restore(payload['sampler'])
                if cloned.draw(step,True)!=sampler.draw(step,True): raise ValueError('采样器恢复不一致')
                smoke.append(dict(branch=branch,losses=losses,digest=digest,strict_reload=True)); del payload
            if smoke[0]['digest']!=smoke[1]['digest'] or smoke[0]['losses']!=smoke[1]['losses']:
                raise ValueError('A/B 暖身更新不一致')
            save(args.output/'smoke.json',dict(status='passed',records=smoke,updates=4))
            branch='baseline'; reset(); policy.eval(); status('baseline_offline')
            baseline_m7=offline(policy,cache,datasets['val'],valgroups,FEEDBACK,budget,args.output/'baseline-m7.json')
            status('baseline_closed_loop')
            common=evaluate_closed(base,policy,fk,FEEDBACK,args.output/'baseline-feedback',budget,support)
            common['m7']=baseline_m7; save(args.output/'baseline-feedback.json',common)
        warmup=None; results={}; resume_step=None
        if args.resume_run:
            warmup=json.loads((args.output/'A/warmup.json').read_text())['digest']
        for branch in ('A','B'):
            reset(); folder=args.output/branch; folder.mkdir(exist_ok=bool(args.resume_run and branch=='A')); sampler.history=[copy.deepcopy(common)]
            losses=[]
            start_block=0
            if args.resume_run and branch=='A':
                payload=torch.load(args.resume_run/'A/latest.pt',map_location='cpu',weights_only=True)
                verify_resume_checkpoint(payload,old_config)
                policy.expert.load_state_dict(payload['expert'],strict=True)
                policy.memory_encoder.load_state_dict(payload['memory_encoder'],strict=True)
                optimizer.load_state_dict(payload['optimizer']); sampler.restore(payload['sampler'])
                step=payload['step']; torch.set_rng_state(payload['torch_rng']); torch.cuda.set_rng_state_all(payload['cuda_rng'])
                resume_step=step
                if set(int(v['step']) for v in optimizer.state.values() if 'step' in v)!={offset+step}:
                    raise ValueError('恢复优化器步数不一致')
                losses=json.loads((folder/'losses.json').read_text())
                losses += [None]*(step-len(losses))
                start_block=(step-1)//256
                persist(folder/'latest.pt'); status('resumed_before_feedback',checkpoint_step=step)
                del payload
            for block in range(start_block,4):
                status('training',block=block+1)
                while step<(block+1)*256:
                    loss=update(branch=='B'); losses.append(loss)
                    if step%16==0: status('training',block=block+1,loss_mean_last16=float(np.mean(losses[-16:])))
                    if step%128==0: persist(folder/'latest.pt')
                if step==256:
                    digest=model_digest(policy)
                    if branch=='A': warmup=digest
                    elif digest!=warmup: raise ValueError('A/B 前 256 更新权重不一致')
                    save(folder/'warmup.json',dict(digest=digest,paired_equal=branch=='B'))
                policy.eval(); state_cpu=torch.get_rng_state(); state_cuda=torch.cuda.get_rng_state_all()
                status('evaluating_offline')
                offline_path=folder/f'm7-{step}.json'
                resuming_feedback=bool(args.resume_run and branch=='A' and step==resume_step)
                if resuming_feedback and offline_path.exists():
                    m7=json.loads(offline_path.read_text())['groups']
                else:
                    m7=offline(policy,cache,datasets['val'],valgroups,FEEDBACK,budget,offline_path)
                status('evaluating_closed_loop')
                feedback=evaluate_closed(base,policy,fk,FEEDBACK,folder/f'feedback-{step}',budget,support,resume=resuming_feedback)
                feedback['m7']=m7
                if branch=='B': notes=sampler.update(feedback,step)
                else: sampler.history.append(feedback); notes=[]
                save(folder/f'feedback-{step}.json',feedback)
                save(folder/'sampling.json',dict(state=sampler.state(),notes=notes))
                torch.set_rng_state(state_cpu); torch.cuda.set_rng_state_all(state_cuda)
                persist(folder/'latest.pt',completed=step==1024)
                save(folder/'losses.json',losses)
            results[branch]=dict(checkpoint_sha256=sha(folder/'latest.pt'),steps=step,
                feedback=feedback['tasks'],exposure=int(sampler.exposure.sum()))
            save(folder/'result.json',dict(status='completed',**results[branch]))
        save(args.output/'candidates-frozen.json',results)
        for branch in ('baseline','A','B'):
            budget.check(); status('reserved_evaluation')
            payload=parent if branch=='baseline' else torch.load(args.output/branch/'latest.pt',map_location='cpu',weights_only=True)
            policy.expert.load_state_dict(payload['expert'],strict=True); policy.eval()
            out=evaluate_closed(base,policy,fk,RESERVED,args.output/f'reserved-{branch}',budget,support)
            offline(policy,cache,datasets['val'],valgroups,RESERVED,budget,args.output/f'reserved-m7-{branch}.json')
            if branch!='baseline': del payload
        status('completed',results=results)
        save(args.output/'result.json',dict(status='completed',config_identity=config_id,results=results,
            total_updates=2048,smoke_updates=4,training_converged=False))
    except BaseException as exc:
        if policy is not None and sampler is not None and current_stage!='reserved_evaluation':
            persist(args.output/'interrupted.pt')
        status('stopped' if isinstance(exc,InterruptedError) else 'error',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__': main()
