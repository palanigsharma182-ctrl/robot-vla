"""同数据与更新预算的联合对照；保存新动作身份，拒绝套用旧joint checkpoint。"""
from dataclasses import asdict
from pathlib import Path
import gc
import json
import time
import numpy as np
import torch
from experiments.g2c_memory_integration.vla import load_runtime,sha256
from experiments.memory_conditioning.conditioning import MemoryBatch
from experiments.rgbd_memory_policy.train import save_json
from experiments.tcp_memory_control.data import prepare_examples
from experiments.tcp_memory_control.kinematics import TCPKinematics
from experiments.tcp_memory_control.policy import build_policy, ARMS, RESET_PREFIXES
from experiments.tcp_memory_control.protocol import PROTOCOL, identity, sampling_seed
from experiments.rgbd_memory_policy.stream import INSTRUCTION
from robot_vla.runtime.policy_runtime import _move_model_inputs
from robot_vla.training.flow_matching import sample_flow_training_target,masked_flow_mse,euler_integrate_actions

FORMAT='tcp-memory-combined-checkpoint/v1'
REQUIRED_SOURCE_PATHS=tuple(sorted(
    ['experiments/tcp_memory_control/'+name+'.py' for name in
     ('data','geometry','kinematics','policy','protocol','train','evaluate','run')]
    +['experiments/memory_conditioning/conditioning.py','experiments/memory_reobserve/runtime.py',
      'experiments/rgbd_memory_policy/data.py','experiments/rgbd_memory_policy/stream.py',
      'experiments/rgbd_memory_policy/protocol.py','experiments/g2c_memory_integration/vla.py',
      'experiments/front_rgbd_memory/geometry.py','experiments/front_rgbd_memory/memory.py',
      'src/robot_vla/training/flow_matching.py','src/robot_vla/model/expert.py',
      'src/robot_vla/model/policy.py','src/robot_vla/model/qwen_context.py',
      'src/robot_vla/adapters.py','src/robot_vla/contracts.py',
      'src/robot_vla/diagnostics/oracle_reach.py','src/robot_vla/execution/chunk_executor.py']))


def verify_source(path):
    entries=json.loads(Path(path).read_text())
    if not isinstance(entries,dict) or not set(REQUIRED_SOURCE_PATHS).issubset(entries):
        raise ValueError('源码manifest缺少联合实验必需依赖')
    if any(sha256(name)!=digest for name,digest in entries.items()):
        raise ValueError('实际源码与冻结manifest不一致')
    return sha256(path)


def train(args):
    source_hash=verify_source(args.source_manifest)
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    fk=TCPKinematics();raw,hashes,denominator=prepare_examples(args.data,fk)
    if len(denominator)!=24 or any(not raw[s] for s in raw):
        raise ValueError('必须使用完整共同数据')
    base,upstream=load_runtime(args.checkpoint,args.model_cache)
    torch.cuda.reset_peak_memory_stats()
    examples={s:[] for s in raw}
    for split,rows in raw.items():
        for x in rows:
            encoded=base.processor_adapter.encode(x['rgb_external'],x['rgb_wrist'],INSTRUCTION)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                context=base.policy.encode_context(_move_model_inputs(encoded.model_inputs,base.device))
            examples[split].append(dict(seed=x['seed'],anchor=x['anchor'],available=x['snapshot']['available'],context=context,
                proprio=torch.tensor(base.proprio_normalizer.normalize(x['physical_proprio'])[None],device='cuda'),
                joint_action=torch.tensor(x['action'][None],device='cuda'),tcp_action=torch.tensor(x['tcp_action'][None],device='cuda'),
                world_features=torch.tensor([x['snapshot']['features']],dtype=torch.float32,device='cuda'),
                tcp_features=torch.tensor(x['tcp_features'][None],device='cuda')))
    del raw;gc.collect()
    rng=np.random.default_rng(PROTOCOL['seed']);schedule=rng.integers(len(examples['train']),size=(PROTOCOL['steps'],PROTOCOL['accumulation']))
    dropout=rng.random(schedule.shape)<PROTOCOL['memory_dropout']
    ident=dict(protocol=PROTOCOL,source_manifest_sha256=source_hash,required_source_paths=list(REQUIRED_SOURCE_PATHS),data_sha256=hashes,denominator=denominator,
        upstream=upstream,urdf_sha256=sha256(fk.urdf_path),instruction=INSTRUCTION,
        schedule=schedule.tolist(),dropout=dropout.tolist(),reset_prefixes=RESET_PREFIXES)
    ident=json.loads(json.dumps(ident));ident_hash=identity(ident);save_json(args.output/'identity.json',ident)
    results={};mask=torch.ones((1,16),dtype=torch.bool,device='cuda')
    for arm in ARMS:
        policy=build_policy(base.policy,arm).to('cuda');policy.eval()
        initial={k:v.detach().cpu().clone() for k,v in policy.memory_encoder.state_dict().items()}
        params=list(policy.expert.parameters())+list(policy.memory_encoder.parameters())
        optimizer=torch.optim.AdamW(params,lr=PROTOCOL['learning_rate'])
        def condition(x,enabled=True):
            f=x['world_features' if arm=='joint-world' else 'tcp_features']
            return policy.condition_context(x['context'],MemoryBatch(f,torch.tensor([[enabled and x['available']]],device='cuda')))
        def action(x):return x['joint_action' if arm=='joint-world' else 'tcp_action']
        def loss(x,enabled,g):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                target=sample_flow_training_target(action(x),mask,generator=g)
                return masked_flow_mse(policy.expert(condition(x,enabled),x['proprio'],target.noisy_action,target.flow_time,mask),target.target_velocity,mask)
        def assess():
            policy.eval();rows=[]
            with torch.no_grad():
                for x in examples['development']:
                    seed=sampling_seed(x['seed'],x['anchor']);g=torch.Generator(device='cuda').manual_seed(seed)
                    flow=float(loss(x,True,g))
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        context=condition(x);kv=policy.expert.prepare_context_kv(context)
                        noise=torch.randn(action(x).shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
                        prediction=euler_integrate_actions(lambda a,t:policy.expert(context,x['proprio'],a,t,mask,context_kv=kv),noise,mask,num_steps=10)
                    rows.append(dict(seed=x['seed'],anchor=x['anchor'],available=x['available'],flow_mse=flow,
                        sampled_first4_mae_normalized=float((prediction[:,:4]-action(x)[:,:4]).abs().mean())))
            return rows
        policy.expert.train();policy.memory_encoder.train();losses=[];exposure=0;grad_steps=0
        completed=False
        try:
            for step,indices in enumerate(schedule):
                optimizer.zero_grad(set_to_none=True);total=0.
                for k,index in enumerate(indices):
                    x=examples['train'][int(index)];enabled=not dropout[step,k]
                    # 每样本重新派生seed，使7D/8D不同噪声长度不改变后续flow time抽样。
                    generator=torch.Generator(device='cuda').manual_seed(sampling_seed(PROTOCOL['seed'],step*PROTOCOL['accumulation']+k))
                    exposure+=int(enabled and x['available']);value=loss(x,enabled,generator)
                    if not torch.isfinite(value):raise ValueError('非有限loss')
                    (value/PROTOCOL['accumulation']).backward();total+=float(value.detach())/PROTOCOL['accumulation']
                grad_steps+=int(any(p.grad is not None and bool((p.grad!=0).any()) for p in policy.memory_encoder.parameters()))
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();losses.append(total)
                if (step+1)%64==0:print(json.dumps(dict(arm=arm,step=step+1,loss=total)),flush=True)
            completed=True
        finally:
            delta=float(sum((v.detach().cpu()-initial[k]).square().sum() for k,v in policy.memory_encoder.state_dict().items()))**.5
            valid=completed and exposure>0 and grad_steps>0 and delta>0
            payload=dict(format=FORMAT,arm=arm,identity=ident,identity_sha256=ident_hash,completed=valid,steps=len(losses),
                expert={k:v.detach().cpu() for k,v in policy.expert.state_dict().items()},
                memory_encoder={k:v.detach().cpu() for k,v in policy.memory_encoder.state_dict().items()},losses=losses)
            torch.save(payload,args.output/(arm+'.pt'));del payload
        if not valid:raise ValueError('训练未完成或Memory未参与')
        if any(p.grad is not None for p in policy.context_encoder.parameters()) or any(p.grad is not None for p in policy.adapter.parameters()):
            raise ValueError('冻结上游出现梯度')
        metrics=assess();digest=sha256(args.output/(arm+'.pt'))
        restore(policy,args.output/(arm+'.pt'),digest,ident_hash,arm,source_hash)
        if metrics!=assess():raise ValueError('严格重载预测不一致')
        results[arm]=dict(steps=len(losses),exposure=exposure,masked=512-exposure,memory_gradient_steps=grad_steps,
            memory_parameter_change_l2=delta,checkpoint_sha256=digest,development=metrics,strict_reload=True,
            first_loss=losses[0],last_loss=losses[-1],gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save_json(args.output/'result.json',dict(status='completed' if len(results)==2 else 'partial',identity_sha256=ident_hash,
            results=results,elapsed_s=time.monotonic()-started))
        del optimizer,policy,params;gc.collect();torch.cuda.empty_cache()


def restore(policy,path,expected_sha,expected_identity,arm,source_hash):
    if sha256(path)!=expected_sha:raise ValueError('checkpoint文件身份不匹配')
    p=torch.load(path,map_location='cpu',weights_only=True)
    if (p['format']!=FORMAT or p['arm']!=arm or not p['completed'] or p['steps']!=PROTOCOL['steps']
        or p['identity_sha256']!=expected_identity or identity(p['identity'])!=expected_identity
        or p['identity']['protocol']!=PROTOCOL or p['identity']['source_manifest_sha256']!=source_hash
        or p['identity'].get('required_source_paths')!=list(REQUIRED_SOURCE_PATHS)):
        raise ValueError('不接受旧动作合同或不同训练身份')
    policy.expert.load_state_dict(p['expert'],strict=True);policy.memory_encoder.load_state_dict(p['memory_encoder'],strict=True)
