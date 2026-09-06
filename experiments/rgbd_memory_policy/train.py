"""冻结Qwen/Adapter，两臂等预算训练；训练和执行共用MemoryConditionedPolicy。"""
from collections import Counter
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.front_rgbd_memory.memory import candidate_config
from experiments.g2c_memory_integration.vla import load_runtime, CHECKPOINT_SHA256, sha256
from experiments.memory_conditioning.conditioning import MemoryConditionedPolicy, MemoryBatch, MEMORY_SCHEMA
from experiments.rgbd_memory_policy.data import load_examples
from experiments.rgbd_memory_policy.protocol import PROTOCOL, identity
from experiments.rgbd_memory_policy.stream import INSTRUCTION
from robot_vla.runtime.policy_runtime import _move_model_inputs
from robot_vla.training.flow_matching import sample_flow_training_target, masked_flow_mse

FORMAT='rgbd-pregrasp-memory-policy/v1'


def implementation_identity():
    paths=['experiments/rgbd_memory_policy/'+x+'.py' for x in ('data','train','protocol','stream')]
    paths+=['experiments/memory_conditioning/conditioning.py']
    return dict(instruction=INSTRUCTION,source_sha256={p:sha256(p) for p in paths})


def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def train(data,checkpoint,model_cache,output,source_manifest):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    raw,hashes,denominator=load_examples(data)
    coverage={split:dict(samples=len(rows),available=sum(x['snapshot']['available'] for x in rows),
        available_scenes=len({x['seed'] for x in rows if x['snapshot']['available']}),
        reasons=dict(Counter(reason for x in rows for reason in x['snapshot']['reasons']))) for split,rows in raw.items()}
    save_json(output/'coverage.json',dict(coverage=coverage,denominator=denominator))
    if coverage['train']['available_scenes']<4 or coverage['development']['available_scenes']<1:
        raise ValueError('不足以验证真实Memory训练消费；不挑选替代seed')
    runtime,upstream=load_runtime(Path(checkpoint),Path(model_cache))
    upstream=json.loads(json.dumps(upstream))
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(PROTOCOL['seed'])
    policy=MemoryConditionedPolicy(runtime.policy.context_encoder,runtime.policy.expert,runtime.policy.adapter).to('cuda')
    policy.context_encoder.requires_grad_(False);policy.adapter.requires_grad_(False);policy.eval()
    initial={k:v.detach().cpu().clone() for k,v in policy.expert.state_dict().items()}
    encoder_initial={k:v.detach().cpu().clone() for k,v in policy.memory_encoder.state_dict().items()}
    examples={'train':[],'development':[]}
    for split,rows in raw.items():
        for index,x in enumerate(rows):
            processed=runtime.processor_adapter.encode(x['rgb_external'],x['rgb_wrist'],INSTRUCTION)
            with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                context=policy.encode_context(_move_model_inputs(processed.model_inputs,torch.device('cuda')))
            snap=x['snapshot']
            examples[split].append(dict(seed=x['seed'],anchor=x['anchor'],snapshot=snap,context=context,
                proprio=torch.tensor(runtime.proprio_normalizer.normalize(x['physical_proprio'])[None],device='cuda'),
                action=torch.tensor(x['action'][None],device='cuda'),
                features=torch.tensor([snap['features']],device='cuda'),
                mask=torch.ones((1,16),dtype=torch.bool,device='cuda')))
            if (index+1)%32==0:print(json.dumps(dict(stage='cache',split=split,done=index+1)),flush=True)
    del raw;gc.collect()
    rng=np.random.default_rng(PROTOCOL['seed'])
    schedule=rng.integers(0,len(examples['train']),size=(PROTOCOL['train_steps_per_arm'],PROTOCOL['accumulation']))
    dropout=rng.random(schedule.shape)<PROTOCOL['memory_dropout']
    training_identity=dict(protocol=PROTOCOL,source_manifest_sha256=sha256(source_manifest),data_sha256=hashes,
        implementation=implementation_identity(),upstream_identity=upstream,
        denominator=denominator,coverage=coverage,schedule=schedule.tolist(),dropout=dropout.tolist(),
        upstream_sha256=CHECKPOINT_SHA256,memory_schema=MEMORY_SCHEMA,memory_config=asdict(candidate_config()),
        proprio_normalization=dict(mean=runtime.proprio_normalizer.mean.tolist(),std=runtime.proprio_normalizer.std.tolist(),clip=runtime.proprio_normalizer.clip))
    save_json(output/'identity.json',training_identity)
    identity_hash=identity(training_identity)

    def loss_for(x,enabled,generator):
        available=bool(enabled and x['snapshot']['available'])
        memory=MemoryBatch(x['features'],torch.tensor([[available]],dtype=torch.bool,device='cuda'))
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            context=policy.condition_context(x['context'],memory)
            target=sample_flow_training_target(x['action'],x['mask'],generator=generator)
            prediction=policy.expert(context,x['proprio'],target.noisy_action,target.flow_time,x['mask'])
            return masked_flow_mse(prediction,target.target_velocity,x['mask'])

    def evaluate(enabled):
        policy.eval();rows=[]
        with torch.no_grad():
            for x in examples['development']:
                generator=torch.Generator(device='cuda').manual_seed(9000000+x['seed']*100+x['anchor'])
                values=[float(loss_for(x,enabled,generator)) for _ in range(2)]
                rows.append(dict(seed=x['seed'],anchor=x['anchor'],available=x['snapshot']['available'],
                    reasons=x['snapshot']['reasons'],flow_mse=float(np.mean(values))))
        return rows

    results={}
    for arm in PROTOCOL['arms']:
        policy.expert.load_state_dict(initial,strict=True);policy.memory_encoder.load_state_dict(encoder_initial,strict=True)
        torch.manual_seed(PROTOCOL['seed'])
        params=list(policy.expert.parameters())+([] if arm=='visual' else list(policy.memory_encoder.parameters()))
        optimizer=torch.optim.AdamW(params,lr=PROTOCOL['learning_rate'])
        generator=torch.Generator(device='cuda').manual_seed(PROTOCOL['seed'])
        before=evaluate(arm=='memory');policy.expert.train();policy.memory_encoder.train()
        losses=[];exposure=Counter();gradient_steps=0;max_gradient=0.;exposed_samples=set();change=0.
        try:
            for step,indices in enumerate(schedule):
                optimizer.zero_grad(set_to_none=True);total=0.
                for k,index in enumerate(indices):
                    x=examples['train'][int(index)]
                    enabled=arm=='memory' and not dropout[step,k]
                    exposure['token-on' if enabled and x['snapshot']['available'] else 'masked']+=1
                    if enabled and x['snapshot']['available']:exposed_samples.add((x['seed'],x['anchor']))
                    loss=loss_for(x,enabled,generator)
                    if not torch.isfinite(loss):raise ValueError('非有限Flow loss')
                    (loss/PROTOCOL['accumulation']).backward();total+=float(loss.detach())/PROTOCOL['accumulation']
                grad=float(sum(p.grad.detach().float().square().sum() for p in policy.memory_encoder.parameters() if p.grad is not None))**.5
                gradient_steps+=int(grad>0);max_gradient=max(max_gradient,grad)
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();losses.append(total)
                if (step+1)%64==0:print(json.dumps(dict(stage='train',arm=arm,step=step+1,loss=total)),flush=True)
        finally:
            change=float(sum((p.detach().cpu()-encoder_initial[k]).square().sum() for k,p in policy.memory_encoder.state_dict().items()))**.5
            consumption=arm=='memory' and exposure['token-on']>0 and gradient_steps>0 and max_gradient>0 and change>0
            arm_validated=arm=='visual' or consumption
            # 仅保存推理权重和训练身份；不保存Adam状态，不宣称可无损恢复优化器。
            payload=dict(format=FORMAT,arm=arm,steps=len(losses),completed=len(losses)==len(schedule) and arm_validated,
                memory_consumption_validated=consumption,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                training_identity=training_identity,training_identity_sha256=identity_hash,
                expert={k:v.detach().cpu() for k,v in policy.expert.state_dict().items()},
                memory_encoder={k:v.detach().cpu() for k,v in policy.memory_encoder.state_dict().items()},losses=losses)
            tensor_bytes=sum(v.numel()*v.element_size() for part in ('expert','memory_encoder') for v in payload[part].values())
            current_bytes=sum(p.stat().st_size for p in output.parent.rglob('*') if p.is_file())
            if 2*(current_bytes+tensor_bytes+16_000_000)+40_000_000>=4_000_000_000:
                raise RuntimeError('保存权重前的双副本磁盘预算检查失败')
            torch.save(payload,output/(arm+'.pt'));del payload
        if not arm_validated:raise ValueError('Memory未实际参与训练，保留未完成权重并停止评估')
        if any(p.grad is not None for p in policy.context_encoder.parameters()) or any(p.grad is not None for p in policy.adapter.parameters()):
            raise ValueError('冻结的Qwen或Adapter出现梯度')
        after=evaluate(arm=='memory');masked=evaluate(False) if arm=='memory' else after
        results[arm]=dict(steps=len(losses),checkpoint_sha256=sha256(output/(arm+'.pt')),
            exposure=dict(exposure),memory_gradient_steps=gradient_steps,max_memory_gradient_l2=max_gradient,
            exposed_unique_samples=[list(x) for x in sorted(exposed_samples)],
            memory_parameter_change_l2=change,first_loss=losses[0],last_loss=losses[-1],
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            development_before=before,development=after,masked_development=masked)
        # 显式重载保存的最后一步，不按开发loss选择checkpoint。
        payload=torch.load(output/(arm+'.pt'),map_location='cpu',weights_only=True)
        policy.expert.load_state_dict(payload['expert'],strict=True)
        policy.memory_encoder.load_state_dict(payload['memory_encoder'],strict=True);del payload
        if evaluate(arm=='memory')!=after:raise ValueError('保存/重载后的固定噪声预测不一致')
        results[arm]['strict_reload']=True
        save_json(output/'result.json',dict(status='completed' if len(results)==2 else 'partial',
            results=results,training_identity_sha256=identity_hash,coverage=coverage,denominator=denominator,
            elapsed_s=time.monotonic()-started))
        del optimizer;gc.collect();torch.cuda.empty_cache()
    return results


def restore(policy,path,expected_sha,expected_identity,expected_arm,normalizer,upstream_identity,expected_source_manifest_sha):
    """调用方给出本次训练产物身份；不放宽历史候选loader。"""
    if sha256(path)!=expected_sha:raise ValueError('checkpoint SHA不匹配')
    p=torch.load(path,map_location='cpu',weights_only=True)
    ident=p['training_identity']
    if (p['format']!=FORMAT or p['arm']!=expected_arm or not p['completed']
        or p['steps']!=PROTOCOL['train_steps_per_arm'] or identity(ident)!=expected_identity
        or p['training_identity_sha256']!=expected_identity or ident['protocol']!=PROTOCOL
        or ident['upstream_sha256']!=CHECKPOINT_SHA256 or ident['memory_config']!=asdict(candidate_config())
        or ident['implementation']!=implementation_identity()
        or p.get('memory_consumption_validated')!=(expected_arm=='memory')
        or ident['upstream_identity']!=json.loads(json.dumps(upstream_identity))
        or ident['source_manifest_sha256']!=expected_source_manifest_sha
        or ident['memory_schema']!=MEMORY_SCHEMA):raise ValueError('训练协议、Memory或上游身份不一致')
    if ident['proprio_normalization']!=dict(mean=normalizer.mean.tolist(),std=normalizer.std.tolist(),clip=normalizer.clip):
        raise ValueError('proprio归一化不一致')
    policy.expert.load_state_dict(p['expert'],strict=True);policy.memory_encoder.load_state_dict(p['memory_encoder'],strict=True)
