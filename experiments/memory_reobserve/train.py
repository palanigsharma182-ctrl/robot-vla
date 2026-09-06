"""冻结真实 Qwen/Adapter context 的两臂五技能续训；开发集不参与选择。"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from experiments.g2c_memory_integration.vla import load_runtime, CHECKPOINT_SHA256
from experiments.memory_conditioning.conditioning import MemoryConditionedPolicy, MemoryBatch, MEMORY_SCHEMA
from experiments.memory_reobserve.protocol import PROTOCOL
from experiments.memory_reobserve.data import new_examples, replay_examples, digest
from robot_vla.precision.active_front_memory_provider import build_stage2_object_memory_config
from robot_vla.runtime.policy_runtime import _move_model_inputs
from robot_vla.training.flow_matching import sample_flow_training_target, masked_flow_mse

FORMAT='memory-reobserve-five-skills/v1'


def identity_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def condition_coverage(examples):
    result={}
    for split,rows in examples.items():
        seen=set();transitions=set();reasons=Counter()
        for item in sorted(rows,key=lambda x:(x['seed'],x['anchor'])):
            if item['available']:seen.add(item['seed'])
            else:
                reasons.update(item['invalidation_reasons'])
                if item['seed'] in seen and 'memory_stale' in item['invalidation_reasons']:
                    transitions.add(item['seed'])
        result[split]=dict(available=sum(x['available'] for x in rows),unavailable=sum(not x['available'] for x in rows),
            transition_seeds=sorted(transitions),invalidation_reasons=dict(reasons))
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ('data','d0','checkpoint','model-cache','output','source-manifest'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    new,hashes,denominator=new_examples(args.data)
    qualified={split:sorted(set(x['seed'] for x in values if x['available'])) for split,values in new.items()}
    coverage=condition_coverage(new)
    if len(qualified['train'])<4 or len(qualified['development'])<1 or any(not v['transition_seeds'] for v in coverage.values()):
        (args.output/'result.json').write_text(json.dumps(dict(status='insufficient-qualified-data',
            optimizer_steps=0,qualified_scenes=qualified,coverage=coverage,denominator=denominator),indent=2)+'\n')
        raise SystemExit(2)
    runtime,identity=load_runtime(args.checkpoint,args.model_cache)
    torch.manual_seed(42)
    policy=MemoryConditionedPolicy(runtime.policy.context_encoder,runtime.policy.expert,runtime.policy.adapter).to('cuda')
    policy.context_encoder.requires_grad_(False);policy.adapter.requires_grad_(False);policy.eval()
    replay,replay_identity=replay_examples(args.d0,runtime.proprio_normalizer)
    train_raw=replay+new['train'];dev_raw=new['development']
    initial_expert={k:v.detach().cpu().clone() for k,v in policy.expert.state_dict().items()}
    initial_encoder={k:v.detach().cpu().clone() for k,v in policy.memory_encoder.state_dict().items()}
    freeze=dict(protocol=PROTOCOL,data_sha256=hashes,d0_identity=replay_identity,qualified_scenes=qualified,
        source_manifest_sha256=digest(args.source_manifest),
        collection_sha256=digest(args.data/'collection.json'),coverage=coverage,
        denominator=denominator,upstream=identity,maximum_process_seconds=3000,
        maximum_cache_seconds=600,maximum_training_seconds_per_arm=1000,evaluation_and_save_reserve_seconds=400,
        microbatch_size=1,accumulation_steps=4,replay_samples_per_skill=128,
        minimum_available_train_scenes=4,minimum_available_development_scenes=1,
        evaluation='four fixed noise/time draws per new development anchor; last checkpoint only',
        meaning='bounded development training, not convergence or final-test evidence')
    (args.output/'protocol.json').write_text(json.dumps(freeze,indent=2)+'\n')
    def cache(raw):
        encoded=runtime.processor_adapter.encode(raw['rgb_external'],raw['rgb_wrist'],raw['instruction'])
        with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            context=runtime.policy.encode_context(_move_model_inputs(encoded.model_inputs,torch.device('cuda')))
        proprio=raw.get('proprio')
        if proprio is None:proprio=runtime.proprio_normalizer.normalize(raw['physical_proprio'])
        item={k:raw[k] for k in ('seed','anchor','source','skill_id','available')}
        item.update(invalidation_reasons=raw.get('invalidation_reasons',['d0-no-memory-history']),age_s=raw.get('age_s'))
        item.update(context=context,
            proprio=torch.tensor(proprio[None],dtype=torch.float32,device='cuda'),
            action=torch.tensor(raw['action'][None],dtype=torch.float32,device='cuda'),
            mask=torch.tensor(raw['action_mask'][None],dtype=torch.bool,device='cuda'),
            features=torch.tensor(raw['features'][None],dtype=torch.float32,device='cuda'))
        if item['action'].shape!=(1,16,8) or not all(torch.isfinite(item[k]).all() for k in ('proprio','action','features')):
            raise ValueError('cache shape/finite 失败')
        return item
    train=[cache(x) for x in train_raw];development=[cache(x) for x in dev_raw]
    cache_seconds=time.monotonic()-started
    if cache_seconds>600:raise TimeoutError('缓存超预算，保留两臂相同训练时长，不启动部分条件')
    del train_raw,dev_raw,replay,new
    buckets={i:[j for j,x in enumerate(train) if x['source']=='d0' and x['skill_id']==i] for i in range(5)}
    fresh=[j for j,x in enumerate(train) if x['source']=='g2c-sequence']
    rng=np.random.default_rng(42)
    # 每个 update 恰好3个D0 + 1个新序列；D0技能轮换，两个训练条件共享同一 schedule。
    schedule=[[int(rng.choice(buckets[(step*3+k)%5])) for k in range(3)] + [int(rng.choice(fresh))]
              for step in range(PROTOCOL['train_steps_per_arm'])]
    dropout=np.random.default_rng(43).random((len(schedule),4))<PROTOCOL['memory_dropout']
    schedule_record=dict(indices=schedule,dropout=dropout.tolist(),
        exposure=dict(Counter(f"{train[i]['source']}/skill-{train[i]['skill_id']}" for row in schedule for i in row)),
        memory_exposure=dict(Counter('unavailable' if not train[i]['available'] else
            'dropout' if dropout[step,k] else 'token-on' for step,row in enumerate(schedule) for k,i in enumerate(row))))
    schedule_record['condition_exposure']=dict(Counter(
        f"{train[i]['source']}/"+('unavailable:'+','.join(train[i]['invalidation_reasons']) if not train[i]['available']
            else 'dropout' if dropout[step,k] else 'token-on')
        for step,row in enumerate(schedule) for k,i in enumerate(row)))
    schedule_record['per_skill_microbatch_counts']=dict(Counter(str(train[i]['skill_id']) for row in schedule for i in row))
    schedule_record['five_skill_uniform']=False
    schedule_record['pregrasp_weighted_fraction']=schedule_record['per_skill_microbatch_counts']['0']/(len(schedule)*4)
    (args.output/'schedule.json').write_text(json.dumps(schedule_record,indent=2)+'\n')
    freeze.update(schedule=schedule_record,cache_seconds=cache_seconds)
    training_identity=json.loads(json.dumps(freeze))
    training_identity_sha256=identity_digest(training_identity)
    (args.output/'protocol.json').write_text(json.dumps(training_identity,indent=2)+'\n')
    def loss_for(item,enabled,generator):
        memory=MemoryBatch(item['features'] if enabled else torch.zeros_like(item['features']),
            torch.tensor([[bool(enabled and item['available'])]],dtype=torch.bool,device='cuda'))
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            context=policy.condition_context(item['context'],memory)
            target=sample_flow_training_target(item['action'],item['mask'],generator=generator)
            prediction=policy.expert(context,item['proprio'],target.noisy_action,target.flow_time,item['mask'])
            return masked_flow_mse(prediction,target.target_velocity,item['mask'])
    def evaluate(enabled):
        policy.expert.eval();policy.memory_encoder.eval();rows=[]
        with torch.no_grad():
            for item in development:
                generator=torch.Generator(device='cuda').manual_seed(9000000+item['seed']*100+item['anchor'])
                value=float(torch.stack([loss_for(item,enabled,generator) for _ in range(4)]).mean())
                rows.append(dict(seed=item['seed'],anchor=item['anchor'],available=item['available'],flow_mse=value,
                    invalidation_reasons=item['invalidation_reasons'],age_s=item['age_s']))
        return rows
    results={}
    for arm in ('visual','memory'):
        policy.expert.load_state_dict(initial_expert,strict=True)
        policy.memory_encoder.load_state_dict(initial_encoder,strict=True)
        torch.manual_seed(42)
        params=list(policy.expert.parameters())+([] if arm=='visual' else list(policy.memory_encoder.parameters()))
        assert not any(p.requires_grad for p in policy.context_encoder.parameters())
        assert not any(p.requires_grad for p in policy.adapter.parameters())
        optimizer=torch.optim.AdamW(params,lr=PROTOCOL['learning_rate'])
        generator=torch.Generator(device='cuda').manual_seed(42)
        policy.expert.train();policy.memory_encoder.train();losses=[]
        arm_started=time.monotonic()
        step=-1
        try:
            for step,indices in enumerate(schedule):
                if time.monotonic()-arm_started > 1000:
                    raise TimeoutError('达到训练保守墙钟上限，保存已有恢复状态，不作等预算效果结论')
                optimizer.zero_grad(set_to_none=True); total=0.
                for k,index in enumerate(indices):
                    loss=loss_for(train[index],arm=='memory' and not dropout[step,k],generator)
                    if not torch.isfinite(loss):raise ValueError('loss 非有限')
                    (loss/4).backward();total+=float(loss.detach())/4
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                optimizer.step();losses.append(total)
                if (step+1)%128==0:
                    print(json.dumps(dict(arm=arm,step=step+1,loss_mean_128=float(np.mean(losses[-128:])),
                        elapsed_s=time.monotonic()-started)),flush=True)
        finally:
            # 唯一输出目录；即使有界失败也保存实际已完成optimizer步，不能冒充1024步。
            payload=dict(format=FORMAT,arm=arm,steps=len(losses),upstream_sha256=CHECKPOINT_SHA256,
                memory_schema=MEMORY_SCHEMA,memory_config=asdict(build_stage2_object_memory_config()),
                protocol=PROTOCOL,protocol_sha256=digest(args.output/'protocol.json'),
                schedule_sha256=digest(args.output/'schedule.json'),data_sha256=hashes,
                training_identity=training_identity,training_identity_sha256=training_identity_sha256,
                proprio_normalization=dict(mean=runtime.proprio_normalizer.mean.tolist(),
                    std=runtime.proprio_normalizer.std.tolist(),clip=runtime.proprio_normalizer.clip),
                expert={k:v.detach().cpu() for k,v in policy.expert.state_dict().items()},
                memory_encoder={k:v.detach().cpu() for k,v in policy.memory_encoder.state_dict().items()},
                optimizer=optimizer.state_dict(),flow_rng_state=generator.get_state(),
                torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(),
                losses=losses,completed=len(losses)==len(schedule))
            torch.save(payload,args.output/(arm+'.pt'))
        on=evaluate(arm=='memory');off=evaluate(False) if arm=='memory' else on
        results[arm]=dict(steps=len(losses),checkpoint_sha256=digest(args.output/(arm+'.pt')),
            first_loss=losses[0],last_loss=losses[-1],development=on,masked_development=off)
        (args.output/'result.json').write_text(json.dumps(dict(status='training-completed' if len(results)==2 else 'partial',
            arms=results,qualified_scenes=qualified,coverage=coverage,training_identity_sha256=training_identity_sha256,
            elapsed_s=time.monotonic()-started,
            task_success_evaluated=False),indent=2)+'\n')
    print(json.dumps(dict(status='training-completed',elapsed_s=time.monotonic()-started)))


if __name__=='__main__':main()
