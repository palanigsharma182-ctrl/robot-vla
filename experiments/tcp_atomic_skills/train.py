"""五技能均衡学习与旧Reach回放，保存可恢复状态并使用预定最后checkpoint。"""
import argparse
import gc
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from experiments.tcp_atomic_skills.data import load_examples, action_spec
from experiments.tcp_atomic_skills.protocol import PROTOCOL, save, sha, identity, verify_source
from experiments.tcp_atomic_skills.runtime import load_parent
from experiments.tcp_memory_control.data import prepare_examples
from experiments.tcp_memory_control.geometry import TCPActionSpec
from experiments.tcp_memory_control.kinematics import TCPKinematics
from experiments.tcp_memory_control.protocol import sampling_seed
from experiments.rgbd_memory_policy.stream import INSTRUCTION
from experiments.memory_conditioning.conditioning import MemoryBatch
from robot_vla.runtime.policy_runtime import _move_model_inputs
from robot_vla.training.flow_matching import sample_flow_training_target, masked_flow_mse, euler_integrate_actions

FORMAT = 'tcp-five-skills-checkpoint/v1'


def loss_for(policy, x, seed):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        context = policy.condition_context(x['context'], MemoryBatch(x['features'], x['available']))
        target = sample_flow_training_target(x['action'], x['mask'],
            generator=torch.Generator(device='cuda').manual_seed(seed))
        prediction = policy.expert(context, x['proprio'], target.noisy_action, target.flow_time, x['mask'])
        return masked_flow_mse(prediction, target.target_velocity, x['mask'])


@torch.no_grad()
def assess(policy, examples, *, parent=False):
    policy.eval(); rows = []
    for x in examples:
        seed = sampling_seed(x['seed'], x['anchor'])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            context = policy.condition_context(x['context'], MemoryBatch(x['features'], x['available']))
            kv = policy.expert.prepare_context_kv(context)
            noise = torch.randn(x['action'].shape, device='cuda',
                generator=torch.Generator(device='cuda').manual_seed(seed))
            prediction = euler_integrate_actions(lambda a,t:policy.expert(
                context,x['proprio'],a,t,x['mask'],context_kv=kv),noise,x['mask'],num_steps=10)
        raw = prediction[0].float().cpu().numpy()
        predicted = (TCPActionSpec() if parent else action_spec()).denormalize(raw)
        target = action_spec().denormalize(x['action'][0].cpu().numpy())
        valid = x['mask'][0,:4].cpu().numpy()
        error = predicted[:4][valid]-target[:4][valid]
        rows.append(dict(seed=x['seed'],anchor=x['anchor'],skill_id=x['skill_id'],
            translation_error_mm=float(np.linalg.norm(error[:,:3],axis=1).mean()*1000),
            rotation_delta_error_deg=float(np.linalg.norm(error[:,3:6],axis=1).mean()*180/np.pi),
            gripper_mae=float(abs(error[:,6]).mean()),
            gripper_binary_accuracy=float(((predicted[:4,6][valid]>=.5)==(target[:4,6][valid]>=.5)).mean()),
            predicted_first4=predicted[:4].tolist()))
    return rows


def encode(base, rows):
    output = []
    for i,x in enumerate(rows):
        processed = base.processor_adapter.encode(x['rgb_external'],x['rgb_wrist'],x['instruction'])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            context = base.policy.encode_context(_move_model_inputs(processed.model_inputs,base.device))
        output.append(dict(seed=x['seed'],anchor=x['anchor'],skill_id=x['skill_id'],context=context,
            proprio=torch.tensor(base.proprio_normalizer.normalize(x['physical_proprio'])[None],device='cuda'),
            action=torch.tensor(x['action'][None],device='cuda'),mask=torch.tensor(x['action_mask'][None],device='cuda'),
            features=torch.tensor(x['features'][None],device='cuda'),
            available=torch.tensor([[x['available']]],dtype=torch.bool,device='cuda')))
        if (i+1)%128==0:
            print(json.dumps(dict(stage='encoding',done=i+1,total=len(rows))),flush=True)
    return output


def restore_final(policy, path, result, config):
    if sha(path)!=result['checkpoint_sha256']:
        raise ValueError('五技能checkpoint SHA不符')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if (payload['format']!=FORMAT or payload['configuration_identity']!=identity(config)
        or payload['step']!=PROTOCOL['steps'] or not payload['completed']
        or config['protocol']!=PROTOCOL):
        raise ValueError('五技能checkpoint未完成或训练身份错误')
    policy.expert.load_state_dict(payload['expert'],strict=True)
    policy.memory_encoder.load_state_dict(payload['memory_encoder'],strict=True)


def main():
    p=argparse.ArgumentParser()
    for key in ('collection','reach-data','audit','teacher-replay','checkpoint','model-cache',
                'continued','parent-training','output','source-manifest'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args(); source=verify_source(args.source_manifest)
    audit=json.loads((args.audit/'result.json').read_text())
    replay=json.loads((args.teacher_replay/'result.json').read_text())
    if audit['status']!='passed' or replay['status']!='passed':
        raise ValueError('数据审计与教师执行前置检查未通过')
    args.output.mkdir(exist_ok=False); start=time.monotonic(); step=0
    def stop(*_):
        raise InterruptedError('收到停止请求，保存本次训练恢复点')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    fk=TCPKinematics(); raw,data_identity=load_examples(args.collection,fk)
    if data_identity!=audit['data']:
        raise ValueError('训练数据与审计时不一致')
    reach,reach_hashes,reach_denominator=prepare_examples(args.reach_data,fk)
    base,policy,parent_identity=load_parent(args)
    config=dict(protocol=PROTOCOL,source_sha256=source,parent=parent_identity,data=data_identity,
        reach_data=reach_hashes,reach_denominator=reach_denominator,
        urdf_sha256=sha(fk.urdf_path),audit_sha256=sha(args.audit/'result.json'),
        teacher_replay_sha256=sha(args.teacher_replay/'result.json'))
    config=json.loads(json.dumps(config));config_id=identity(config);save(args.output/'config.json',config)
    save(args.output/'status.json',dict(stage='encoding',step=0,configuration_identity=config_id))
    for split in ('train','development'):
        for x in reach[split]:
            raw[split].append(dict(seed=x['seed'],anchor=x['anchor'],skill_id=5,
                rgb_external=x['rgb_external'],rgb_wrist=x['rgb_wrist'],physical_proprio=x['physical_proprio'],
                instruction=INSTRUCTION,action=action_spec().normalize(TCPActionSpec().denormalize(x['tcp_action'])),
                action_mask=np.ones(16,bool),features=x['tcp_features'],available=x['snapshot']['available']))
    examples={s:encode(base,rows) for s,rows in raw.items()}
    del raw,reach;gc.collect()
    before=assess(policy,examples['development'],parent=True);save(args.output/'development-before.json',before)
    params=list(policy.expert.parameters())+list(policy.memory_encoder.parameters())
    optimizer=torch.optim.AdamW(params,lr=PROTOCOL['learning_rate'])
    rng=np.random.default_rng(PROTOCOL['seed'])
    buckets=[[i for i,x in enumerate(examples['train']) if x['skill_id']==s] for s in range(6)]
    if any(not b for b in buckets):
        raise ValueError('六个采样桶必须非空')
    schedule=np.array([[int(rng.choice(b)) for b in buckets] for _ in range(PROTOCOL['steps'])])
    save(args.output/'schedule.json',dict(indices=schedule.tolist(),bucket_sizes=[len(b) for b in buckets]))
    torch.manual_seed(PROTOCOL['seed']);torch.cuda.manual_seed_all(PROTOCOL['seed'])
    losses=[];exposures=np.zeros(len(examples['train']),dtype=np.int64);completed=False
    def persist():
        payload=dict(format=FORMAT,configuration_identity=config_id,step=step,completed=completed,
            expert={k:v.detach().cpu() for k,v in policy.expert.state_dict().items()},
            memory_encoder={k:v.detach().cpu() for k,v in policy.memory_encoder.state_dict().items()},
            optimizer=optimizer.state_dict(),losses=losses,exposures=exposures.tolist(),
            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        temporary=args.output/'latest.tmp.pt';torch.save(payload,temporary);temporary.replace(args.output/'latest.pt')
        save(args.output/'status.json',dict(stage='trained' if completed else 'training',step=step,
            elapsed_s=time.monotonic()-start,configuration_identity=config_id,
            last_loss=None if not losses else losses[-1]))
    try:
        policy.expert.train();policy.memory_encoder.train()
        for index,indices in enumerate(schedule):
            optimizer.zero_grad(set_to_none=True);total=0.
            for k,i in enumerate(indices):
                x=examples['train'][int(i)]
                value=loss_for(policy,x,sampling_seed(PROTOCOL['seed'],index*6+k))
                if not torch.isfinite(value):raise ValueError('非有限训练loss')
                (value/6).backward();total+=float(value.detach())/6;exposures[int(i)]+=1
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step();step=index+1;losses.append(total)
            if step%128==0:
                progress=dict(stage='training',step=step,total_steps=PROTOCOL['steps'],loss_mean_last128=float(np.mean(losses[-128:])),elapsed_s=time.monotonic()-start)
                save(args.output/'status.json',progress);print(json.dumps(progress),flush=True)
            if step%512==0:persist()
        completed=True
    finally:
        persist()
    if any(p.grad is not None for p in policy.context_encoder.parameters()) or any(p.grad is not None for p in policy.adapter.parameters()):
        raise ValueError('冻结上游出现梯度')
    after=assess(policy,examples['development']);save(args.output/'development-after.json',after)
    result=dict(status='completed',steps=step,configuration_identity=config_id,
        checkpoint_sha256=sha(args.output/'latest.pt'),elapsed_s=time.monotonic()-start,
        skill_exposures=[int(sum(exposures[i] for i in b)) for b in buckets],
        unique_exposed_windows=int((exposures>0).sum()),selection=PROTOCOL['selection'])
    restore_final(policy,args.output/'latest.pt',result,config)
    if assess(policy,examples['development'][:2])!=after[:2]:
        raise ValueError('checkpoint重载预测不一致')
    result['strict_reload']=True;save(args.output/'result.json',result)


if __name__=='__main__':
    main()
