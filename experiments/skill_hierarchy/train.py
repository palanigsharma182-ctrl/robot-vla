"""七技能均衡 BC；按需缓存冻结特征，保存可恢复训练状态。"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from experiments.skill_hierarchy.data import SevenSkillWindows
from experiments.tcp_atomic_skills.runtime import load_parent
from experiments.tcp_atomic_skills.train import restore_final, loss_for, assess, encode
from experiments.tcp_atomic_skills.protocol import save, sha, identity, verify_source
from experiments.tcp_memory_control.protocol import sampling_seed
from robot_vla.model.qwen_context import QwenContext

FORMAT = 'seven-skill-bc-checkpoint/v1'
SEED = 1820042


def restore_continuation(policy, optimizer, payload, previous):
    """继承完整优化状态；与从旧权重重新初始化 AdamW 区分。"""
    if (payload['format']!=FORMAT or payload['configuration_identity']!=identity(previous)
            or not payload['completed'] or payload['step']!=previous['steps']):
        raise ValueError('续训恢复点身份或进度不一致')
    policy.expert.load_state_dict(payload['expert'],strict=True)
    policy.memory_encoder.load_state_dict(payload['memory_encoder'],strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state_all(payload['cuda_rng'])


def balanced_schedule(buckets, steps, seed=SEED):
    """每桶无放回走完再洗牌；完整阶段覆盖最长桶，因此覆盖所有训练窗口。"""
    if len(buckets) != 7 or any(not b for b in buckets):
        raise ValueError('七技能桶必须非空')
    if len(set(i for b in buckets for i in b)) != sum(map(len,buckets)):
        raise ValueError('同一窗口不能属于多个采样桶')
    rng=np.random.default_rng(seed); columns=[]
    for bucket in buckets:
        values=[]
        while len(values)<steps: values.extend(rng.permutation(bucket).tolist())
        columns.append(values[:steps])
    return np.array(columns,dtype=np.int64).T


def probe_indices(dataset, *, train=False, smoke=False):
    """每个选定场景每技能固定一个锚点；开发集覆盖全部 32 个场景。"""
    rng=np.random.default_rng(SEED+1);selected=[]
    entries = {e for e,_ in dataset.index}
    chosen = {min(entries)} if smoke else ({e for e in entries if e%8==0} if train else entries)
    for e in sorted(chosen):
        for skill in range(7):
            candidates=[i for i in dataset.buckets[skill] if dataset.index[i][0]==e]
            if not candidates: raise ValueError('评估场景缺少技能')
            selected.append(int(rng.choice(candidates)))
    return selected


def summarize(rows):
    keys=('translation_error_mm','rotation_delta_error_deg','gripper_mae','gripper_binary_accuracy')
    per_skill={str(s):{k:float(np.mean([r[k] for r in rows if r['skill_id']==s])) for k in keys}
               for s in range(7)}
    if any(not np.isfinite(v) for row in per_skill.values() for v in row.values()):
        raise ValueError('评估必须包含七技能有限指标')
    return dict(windows=len(rows), scenes=len({r['seed'] for r in rows}),per_skill=per_skill,
                macro={k:float(np.mean([r[k] for r in per_skill.values()])) for k in keys})


class FeatureCache:
    """磁盘持久缓存 + 有界 CPU LRU；显存只保存当前参与计算的样本。"""
    def __init__(self, root, base, datasets, config_id, stopped):
        self.root=root;root.mkdir(exist_ok=True);self.base=base;self.datasets=datasets
        self.config_id=config_id;self.stopped=stopped;self.ram=OrderedDict();self.ram_bytes=0
        self.encoded=0;self.disk_hits=0;self.disk_bytes=0

    def get(self,split,index):
        if self.stopped(): raise InterruptedError('停止请求：保留已完成更新')
        key=f'{split}-{index:06d}';path=self.root/(key+'.pt')
        if key in self.ram:
            payload,size=self.ram.pop(key);self.ram[key]=(payload,size)
        else:
            if path.exists():
                payload=torch.load(path,map_location='cpu',weights_only=True);self.disk_hits+=1
                if payload['configuration_identity']!=self.config_id:raise ValueError('缓存身份错误')
            else:
                raw=self.datasets[split][index]
                encoded=encode(self.base,[raw])[0]
                context=encoded.pop('context')
                payload={k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in encoded.items()}
                payload['context']={k:None if v is None else v.detach().cpu() for k,v in
                    dict(tokens=context.tokens,mask=context.mask,image_time_indices=context.image_time_indices).items()}
                payload['configuration_identity']=self.config_id
                if shutil.disk_usage(self.root).free < 3*1024**3:raise RuntimeError('缓存磁盘余量低于 3GiB')
                temporary=path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(path)
                self.encoded+=1;self.disk_bytes+=path.stat().st_size
            tensors=[v for v in payload.values() if isinstance(v,torch.Tensor)]+[
                v for v in payload['context'].values() if isinstance(v,torch.Tensor)]
            size=sum(t.numel()*t.element_size() for t in tensors)
            self.ram[key]=(payload,size);self.ram_bytes+=size
            while self.ram_bytes>6*1024**3 and len(self.ram)>1:
                _,(_,old_size)=self.ram.popitem(last=False);self.ram_bytes-=old_size
        output={k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in payload.items()
                if k not in ('context','configuration_identity')}
        output['context']=QwenContext(**{k:None if v is None else v.to('cuda') for k,v in payload['context'].items()})
        return output


def main():
    parser=argparse.ArgumentParser()
    for name in ('collection','audit','teacher-replay','five-training','checkpoint','model-cache',
                 'continued','parent-training','output','source-manifest'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--continue-from',type=Path)
    args=parser.parse_args();source=verify_source(args.source_manifest)
    if args.resume:
        if not (args.output/'latest.pt').exists():raise ValueError('恢复点不存在')
    else:args.output.mkdir(exist_ok=False)
    started=time.monotonic();stopped=False;step=0;stage='preflight';losses=[];history=[];completed=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def publish(**fields):
        value=dict(stage=stage,step=step,elapsed_s=time.monotonic()-started,updated_unix=time.time(),**fields)
        save(args.output/'status.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)
    publish()
    audit=json.loads((args.audit/'result.json').read_text());replay=json.loads((args.teacher_replay/'result.json').read_text())
    if audit['status']!='passed' or replay['status']!='passed':raise ValueError('教师数据及执行验证未通过')
    if replay['collection_sha256']!=sha(args.collection/'collection.json'):raise ValueError('教师验证绑定了其他数据')
    datasets={s:SevenSkillWindows(args.collection,s) for s in ('train','val')}
    for split,dataset in datasets.items():
        if dataset.hashes!=audit['splits'][split]['files']:raise ValueError('训练数据与审计不一致')
    steps=2 if args.smoke else max(map(len,datasets['train'].buckets))
    previous=None;training_offset=0
    if args.continue_from:
        previous=json.loads((args.continue_from/'config.json').read_text())
        previous_result=json.loads((args.continue_from/'result.json').read_text())
        if (previous_result['status']!='completed' or not previous_result['strict_reload']
                or previous_result['configuration_identity']!=identity(previous)
                or previous_result['checkpoint_sha256']!=sha(args.continue_from/'latest.pt')
                or previous_result['steps']!=previous['steps'] or previous['smoke']!=args.smoke):
            raise ValueError('续训必须使用完整且校验通过的同类阶段')
        training_offset=previous.get('training_offset',0)+previous['steps']
    schedule=balanced_schedule(datasets['train'].buckets,steps,SEED+training_offset)
    probes={s:probe_indices(d,train=s=='train',smoke=args.smoke) for s,d in datasets.items()}
    base,policy,upstream=load_parent(args)
    parent_config=json.loads((args.five_training/'config.json').read_text())
    parent_result=json.loads((args.five_training/'result.json').read_text())
    restore_final(policy,args.five_training/'latest.pt',parent_result,parent_config)
    policy.eval();policy.context_encoder.requires_grad_(False);policy.adapter.requires_grad_(False)
    policy.memory_encoder.requires_grad_(False)
    config=dict(format=FORMAT,source_sha256=source,parent_checkpoint_sha256=parent_result['checkpoint_sha256'],
        parent_configuration_identity=identity(parent_config),upstream=upstream,steps=steps,seed=SEED,
        learning_rate=1e-5,accumulation=7,optimizer='AdamW reset for new dataset; expert only',
        frozen='Qwen, Adapter, Memory encoder',dtype='BF16',eval_every=512,
        collection_sha256=sha(args.collection/'collection.json'),audit_sha256=sha(args.audit/'result.json'),
        replay_sha256=sha(args.teacher_replay/'result.json'),data={s:d.hashes for s,d in datasets.items()},
        bucket_sizes=[len(b) for b in datasets['train'].buckets],probes=probes,
        sampling='one sample per skill per update; shuffled without replacement within each bucket',
        selection='last update of first full-coverage stage; no convergence claim',
        action='existing 25mm/axis, 0.1rad/axis TCP; horizon16 execute4',smoke=args.smoke)
    config=json.loads(json.dumps(config))
    if previous is not None:
        # 只允许阶段身份改变；数据、冻结模块、优化设置及固定探针均保持一致。
        for key in config:
            if key not in ('source_sha256','selection') and config[key]!=previous[key]:
                raise ValueError(f'续训合同改变：{key}')
        config.update(training_offset=training_offset,
            continuation_checkpoint_sha256=previous_result['checkpoint_sha256'],
            continuation_configuration_identity=identity(previous),
            continuation_optimizer_state='restored without reset',
            cache_identity=previous.get('cache_identity',identity(previous)),
            cache_root=previous.get('cache_root',str((args.continue_from/'feature-cache').resolve())),
            selection='last update of continued full-coverage stage; no convergence claim')
    config_id=identity(config)
    if args.resume:
        if json.loads((args.output/'config.json').read_text())!=config:raise ValueError('恢复配置不匹配')
    else:save(args.output/'config.json',config)
    save(args.output/'schedule.json',dict(indices=schedule.tolist(),bucket_sizes=config['bucket_sizes']))
    cache=FeatureCache(Path(config.get('cache_root',args.output/'feature-cache')),base,datasets,
                       config.get('cache_identity',config_id),lambda:stopped)
    params=list(policy.expert.parameters());optimizer=torch.optim.AdamW(params,lr=1e-5)
    exposures=np.zeros(len(datasets['train']),np.int64)
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    if previous is not None and not args.resume:
        payload=torch.load(args.continue_from/'latest.pt',map_location='cpu',weights_only=True)
        restore_continuation(policy,optimizer,payload,previous)
        optimizer_steps=[int(v['step']) for v in optimizer.state.values() if 'step' in v]
        if not optimizer_steps or set(optimizer_steps)!={training_offset}:
            raise ValueError('AdamW 累计步数不匹配')
        save(args.output/'continuation-restore.json',dict(training_offset=training_offset,
            parent_checkpoint_sha256=previous_result['checkpoint_sha256'],
            optimizer_step_min=min(optimizer_steps),optimizer_step_max=max(optimizer_steps),
            rng_restored=True,strict_state_load=True))
        del payload
    if args.resume:
        payload=torch.load(args.output/'latest.pt',map_location='cpu',weights_only=True)
        if payload['format']!=FORMAT or payload['configuration_identity']!=config_id:raise ValueError('恢复身份错误')
        policy.expert.load_state_dict(payload['expert'],strict=True)
        policy.memory_encoder.load_state_dict(payload['memory_encoder'],strict=True)
        optimizer.load_state_dict(payload['optimizer']);step=payload['step'];losses=payload['losses'];history=payload['history']
        exposures=np.asarray(payload['exposures'],np.int64)
        torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state_all(payload['cuda_rng'])

    def persist():
        payload=dict(format=FORMAT,configuration_identity=config_id,step=step,completed=completed,
            expert={k:v.detach().cpu() for k,v in policy.expert.state_dict().items()},
            memory_encoder={k:v.detach().cpu() for k,v in policy.memory_encoder.state_dict().items()},
            optimizer=optimizer.state_dict(),losses=losses,history=history,exposures=exposures.tolist(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        temp=args.output/'latest.tmp.pt';torch.save(payload,temp);temp.replace(args.output/'latest.pt')

    def evaluate():
        nonlocal stage
        stage='evaluating';publish(total_steps=steps)
        policy.eval();rows={}
        for split in datasets:
            rows[split]=[]
            for index in probes[split]:rows[split].extend(assess(policy,[cache.get(split,index)]))
        entry=dict(step=step,groups={s:summarize(r) for s,r in rows.items()})
        history.append(entry);save(args.output/f'evaluation-{step:06d}.json',dict(**entry,records=rows))
        save(args.output/'history.json',history)
        return rows

    try:
        if not history:evaluate();persist()
        stage='training';publish(total_steps=steps)
        policy.expert.train()
        for offset in range(step,steps):
            if stopped:break
            optimizer.zero_grad(set_to_none=True);total=0.
            for k,index in enumerate(schedule[offset]):
                x=cache.get('train',int(index));value=loss_for(policy,x,sampling_seed(SEED,(training_offset+offset)*7+k))
                if not torch.isfinite(value):raise ValueError('非有限训练 loss')
                (value/7).backward();total+=float(value.detach())/7
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
            for index in schedule[offset]:exposures[int(index)]+=1
            step=offset+1;losses.append(total)
            if step==1 or step%16==0:
                publish(total_steps=steps,loss_mean_last16=float(np.mean(losses[-16:])),
                        unique_exposed_windows=int((exposures>0).sum()),encoded_contexts=cache.encoded,
                        cpu_cache_bytes=cache.ram_bytes,gpu_peak_bytes=torch.cuda.max_memory_allocated())
            if step%128==0:persist()
            if step%512==0:
                evaluate();persist();stage='training';policy.expert.train()
        if not stopped and step==steps:
            final=evaluate()
            if any(p.grad is not None for module in (policy.context_encoder,policy.adapter,policy.memory_encoder) for p in module.parameters()):
                raise ValueError('冻结模块出现梯度')
            if not args.smoke and not (exposures>0).all():raise ValueError('完整阶段未覆盖全部训练窗口')
            completed=True;persist()
            payload=torch.load(args.output/'latest.pt',map_location='cpu',weights_only=True)
            policy.expert.load_state_dict(payload['expert'],strict=True);policy.memory_encoder.load_state_dict(payload['memory_encoder'],strict=True)
            if assess(policy,[cache.get('val',probes['val'][0])])!=final['val'][:1]:raise ValueError('checkpoint 重载预测不一致')
            stage='stage_completed'
            save(args.output/'result.json',dict(status='completed',configuration_identity=config_id,steps=step,
                checkpoint_sha256=sha(args.output/'latest.pt'),strict_reload=True,
                unique_exposed_windows=int((exposures>0).sum()),skill_exposures=[int(exposures[b].sum()) for b in datasets['train'].buckets],
                elapsed_s=time.monotonic()-started,offline_converged=False))
        else:stage='interrupted'
    except InterruptedError:
        stage='interrupted'
    except BaseException as exc:
        stage='error';publish(error=f'{type(exc).__name__}: {exc}');raise
    finally:
        persist();publish(total_steps=steps,unique_exposed_windows=int((exposures>0).sum()),
                          last_loss=None if not losses else losses[-1])


if __name__=='__main__':main()
