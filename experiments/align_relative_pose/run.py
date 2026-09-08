"""Align 显式相对位姿误差的隔离 BC 对照；只做离线评估，不接控制器。"""
import argparse
import copy
import gc
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch import nn

from experiments.seven_skill_dagger.run import load_student
from experiments.seven_skill_dagger.data import skill_targets
from experiments.skill_hierarchy.data import SevenSkillWindows
from experiments.skill_hierarchy.train import FeatureCache
from experiments.tcp_atomic_skills.train import loss_for
from experiments.tcp_atomic_skills.data import action_spec
from experiments.tcp_atomic_skills.protocol import save, sha, verify_source
from experiments.tcp_memory_control.geometry import pose_delta, apply_delta
from experiments.tcp_memory_control.kinematics import TCPKinematics
from experiments.tcp_memory_control.protocol import sampling_seed
from experiments.memory_conditioning.conditioning import MemoryBatch
from robot_vla.training.flow_matching import euler_integrate_actions

SEED = 1950042
SCALE = np.array([.05]*3+[.5]*3)


class RelativeProjection(nn.Module):
    """等价于在 proprio 第一层追加 6 列；零初始化保持原策略起点。"""
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.relative = nn.Linear(6, original.out_features, bias=False).to(original.weight)
        nn.init.zeros_(self.relative.weight)
        self.error = None

    def forward(self, proprio):
        if self.error is None or self.error.shape != (proprio.shape[0], 6):
            raise ValueError('显式误差必须与当前样本同批次')
        return self.original(proprio)+self.relative(self.error)


def selected(dataset, scenes, anchors):
    chosen = sorted({dataset.index[i][0] for i in dataset.buckets[1]})[:scenes]
    indices=[]
    for entry in chosen:
        values=[i for i in dataset.buckets[1] if dataset.index[i][0]==entry]
        # 在场景内均匀覆盖，先固定索引，不按任何模型表现选择。
        positions=np.unique(np.linspace(0,len(values)-1,anchors).astype(int))
        indices.extend(values[int(i)] for i in positions)
    if len(chosen)!=scenes:raise ValueError('合格场景不足')
    return indices


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--launch-spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-manifest',type=Path,required=True)
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--wall-seconds',type=int,default=900)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();stop=False
    result=dict(status='running',evidence='privileged-input offline development diagnostic',arms={})
    def halt(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    def check():
        if stop or time.monotonic()-started>=args.wall_seconds:raise InterruptedError('达到总时间上限或收到停止请求')
    def status(stage,**extra):
        check();save(args.output/'status.json',dict(stage=stage,elapsed_s=time.monotonic()-started,**extra))
        print(json.dumps(dict(stage=stage,**extra)),flush=True)
    try:
        status('preflight')
        source=verify_source(args.source_manifest)
        spec=json.loads(args.launch_spec.read_text())
        common=argparse.Namespace(**{k[2:].replace('-','_'):Path(v) for k,v in zip(spec['common'][::2],spec['common'][1::2])})
        common.student=args.launch_spec.parent/'bc-1/latest.pt'
        common.expected_sha=sha(common.student);common.skill=1
        datasets={s:SevenSkillWindows(common.collection,s) for s in ('train','val')}
        indices={'train':selected(datasets['train'],8,8),'val':selected(datasets['val'],8,3)}
        train_seeds={datasets['train'].entries[datasets['train'].index[i][0]].randomization['seed'] for i in indices['train']}
        val_seeds={datasets['val'].entries[datasets['val'].index[i][0]].randomization['seed'] for i in indices['val']}
        assert not train_seeds & val_seeds
        config=dict(schema='align-relative-pose-offline/v1',source_sha256=source,runner_sha256=sha(Path(__file__)),
            parent_sha256=common.expected_sha,steps_per_arm=args.steps,accumulation=4,learning_rate=1e-5,
            seed=SEED,train_indices=indices['train'],val_indices=indices['val'],train_seeds=sorted(train_seeds),
            val_seeds=sorted(val_seeds),feature='pose_delta(actual_tcp, fixed_teacher_pregrasp, actual_tcp)',
            feature_scale=SCALE.tolist(),feature_source='privileged teacher sidecar; no future action labels',
            input_A='Qwen context + proprio + six zeros',input_B='Qwen context + proprio + six relative pose errors',
            added_projection='6 to existing first state hidden layer; bias false; zero initialization in both arms',
            frozen='Qwen, adapter, memory encoder; original memory unavailable',training='nominal BC only; no DAgger',
            primary='mean teacher translation error of valid first four actions (mm), equal scene weighting',
            secondary=['first action command distance change to goal','rotation action error','gripper MAE'],
            inference='16-step mask, ten Euler steps; score only skill-valid first four labels',
            selection='fixed last update; no best-checkpoint selection',wall_seconds=args.wall_seconds,
            limitation='eight development teacher trajectories; no student rollout or convergence claim')
        save(args.output/'config.json',config)
        base,policy,_,old=load_student(common)
        if any(datasets[s].hashes!=old['data'][s] for s in datasets):raise ValueError('缓存来源数据不一致')
        cache=FeatureCache(Path(old['cache_root']),base,datasets,old['cache_identity'],lambda:stop)
        for s in indices:
            if any(not (cache.root/f'{s}-{i:06d}.pt').exists() for i in indices[s]):
                raise ValueError('本轮只读现有缓存，所选样本有缺失')
        fk=TCPKinematics();collection=json.loads((common.collection/'collection.json').read_text())
        records={r['trajectory_id']:r for r in collection['records'] if 'trajectory_id' in r}
        metadata={};side_hashes={}
        for split in indices:
            ds=datasets[split];metadata[split]={}
            for i in indices[split]:
                check();e,t=ds.index[i];entry=ds.entries[e];row=records[entry.trajectory_id]
                side=common.collection/row['sidecar'];h=sha(side);assert h==row['sidecar_sha256']
                side_hashes[str(side.relative_to(common.collection))]=h
                goal=np.linalg.inv(fk.world_from_base)@np.asarray(json.loads(side.read_text())['world_from_pregrasp'])
                actual=fk.pose_base(ds.store.get(entry).proprio[t,:7])
                delta=pose_delta(actual,goal,actual)
                assert np.max(np.abs(apply_delta(actual,delta,actual)-goal))<1e-5
                metadata[split][i]=dict(error=(delta/SCALE).astype(np.float32),actual=actual,goal=goal,seed=entry.randomization['seed'],anchor=t)
        save(args.output/'feature-provenance.json',dict(sidecar_sha256=side_hashes,
            rows={s:{str(i):dict(seed=m['seed'],anchor=m['anchor'],error=m['error'].tolist()) for i,m in rows.items()} for s,rows in metadata.items()}))
        original=policy.expert.state_encoder.projection[0]
        initial=copy.deepcopy(policy.expert.state_dict())
        projection=RelativeProjection(original)
        policy.expert.state_encoder.projection[0]=projection
        def example(split,i,enabled):
            check();x=cache.get(split,int(i));e,t=datasets[split].index[int(i)]
            assert x['anchor']==t and x['skill_id']==1 and int(x['seed'])==metadata[split][int(i)]['seed']
            assert not bool(x['available'].any()) and not bool(x['features'].any())
            a,m=skill_targets(datasets[split].labels[e],t,1)
            x['action']=torch.tensor(a[None],device='cuda');x['mask']=torch.tensor(m[None],device='cuda')
            feature=metadata[split][int(i)]['error'] if enabled else np.zeros(6,np.float32)
            projection.error=torch.tensor(feature[None],device='cuda')
            return x
        # 零初始化时额外真实误差与六个零必须产生完全相同的第一层输出。
        x=example('train',indices['train'][0],True)
        with torch.no_grad():
            assert torch.equal(projection(x['proprio']),original(x['proprio']))
        result['initial_projection_parity']=True
        rng=np.random.default_rng(SEED);schedule=[]
        while len(schedule)<args.steps*4:schedule.extend(rng.permutation(indices['train']).tolist())
        schedule=np.asarray(schedule[:args.steps*4]).reshape(args.steps,4)
        save(args.output/'schedule.json',schedule.tolist())
        def predict(x):
            context=policy.condition_context(x['context'],MemoryBatch(x['features'],x['available']))
            kv=policy.expert.prepare_context_kv(context);mask=torch.ones_like(x['mask'])
            noise=torch.randn(x['action'].shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(sampling_seed(x['seed'],x['anchor'])))
            return euler_integrate_actions(lambda a,t:policy.expert(context,x['proprio'],a,t,mask,context_kv=kv),noise,mask,num_steps=10)
        def evaluate(enabled,name):
            status('evaluating',evaluation=name);policy.eval();rows=[]
            for i in indices['val']:
                x=example('val',i,enabled);meta=metadata['val'][i]
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16),torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    normalized=predict(x)
                a=action_spec().denormalize(normalized[0].float().cpu().numpy())
                truth=action_spec().denormalize(x['action'][0].cpu().numpy());mask=x['mask'][0,:4].cpu().numpy()
                d=a[:4][mask]-truth[:4][mask]
                end=apply_delta(meta['actual'],a[0,:6],meta['actual'])
                before=np.linalg.norm(meta['actual'][:3,3]-meta['goal'][:3,3])*1000
                after=np.linalg.norm(end[:3,3]-meta['goal'][:3,3])*1000
                rows.append(dict(index=i,seed=meta['seed'],anchor=meta['anchor'],translation_error_mm=float(np.linalg.norm(d[:,:3],axis=1).mean()*1000),
                    rotation_error_deg=float(np.linalg.norm(d[:,3:6],axis=1).mean()*180/np.pi),gripper_mae=float(np.abs(d[:,6]).mean()),
                    first_command_distance_change_mm=float(after-before),initial_distance_mm=float(before),
                    first_command_away=bool(before>8 and after-before>.05),first_action=a[0].tolist()))
            keys=('translation_error_mm','rotation_error_deg','gripper_mae','first_command_distance_change_mm')
            summary={k:float(np.mean([r[k] for r in rows])) for k in keys}
            summary.update(away_count=sum(r['first_command_away'] for r in rows),away_eligible=sum(r['initial_distance_mm']>8 for r in rows),windows=len(rows),scenes=len(val_seeds))
            save(args.output/(name+'.json'),dict(summary=summary,rows=rows));return summary
        # 仅评估一次共同起点，不用它选择样本或改变预算。
        result['before']=evaluate(False,'before')
        for name,enabled in [('A',False),('B',True)]:
            status('training_start',arm=name)
            policy.expert.state_encoder.projection[0]=original
            policy.expert.load_state_dict(initial,strict=True)
            projection=RelativeProjection(original);policy.expert.state_encoder.projection[0]=projection
            params=list(policy.expert.parameters());optimizer=torch.optim.AdamW(params,lr=1e-5)
            torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.use_deterministic_algorithms(True)
            losses=[];policy.expert.train()
            for step,chosen in enumerate(schedule,1):
                optimizer.zero_grad(set_to_none=True);total=0.
                for slot,i in enumerate(chosen):
                    x=example('train',i,enabled)
                    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                        loss=loss_for(policy,x,sampling_seed(SEED,(step-1)*4+slot))
                    if not torch.isfinite(loss):raise ValueError('非有限 loss')
                    (loss/4).backward();total+=float(loss.detach())/4
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();losses.append(total)
                if step%16==0:status('training',arm=name,step=step,loss_last16=float(np.mean(losses[-16:])))
            if not enabled:assert torch.count_nonzero(projection.relative.weight)==0
            weight=projection.relative.weight.detach().float().norm().item()
            assert not enabled or weight>0
            # 最小探索点只保存模型；不伪装可精确续训的完整优化器恢复点。
            path=args.output/(name+'.pt')
            payload=dict(format='align-relative-pose-probe/v1',arm=name,steps=args.steps,completed=True,
                expert={k:v.detach().cpu() for k,v in policy.expert.state_dict().items()},config_sha256=sha(args.output/'config.json'))
            torch.save(payload,path);loaded=torch.load(path,map_location='cpu',weights_only=True)
            policy.expert.load_state_dict(loaded['expert'],strict=True);del loaded,payload
            summary=evaluate(enabled,name+'-after')
            result['arms'][name]=dict(summary=summary,losses=losses,checkpoint_sha256=sha(path),relative_projection_norm=weight,strict_reload=True)
            if enabled:result['B_zeroed']=evaluate(False,'B-zeroed')
            save(args.output/'result.json',result);del optimizer;gc.collect()
        result.update(status='completed',elapsed_s=time.monotonic()-started)
        result['primary_B_minus_A_mm']=result['arms']['B']['summary']['translation_error_mm']-result['arms']['A']['summary']['translation_error_mm']
        save(args.output/'result.json',result);status('completed',primary_B_minus_A_mm=result['primary_B_minus_A_mm'])
    except BaseException as exc:
        result.update(status='stopped' if isinstance(exc,InterruptedError) else 'error',error=f'{type(exc).__name__}: {exc}',elapsed_s=time.monotonic()-started)
        save(args.output/'result.json',result);save(args.output/'status.json',dict(stage=result['status'],error=result['error']));raise


if __name__=='__main__':main()
