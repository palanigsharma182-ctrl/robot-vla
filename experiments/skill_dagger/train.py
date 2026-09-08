"""首轮接管DAgger与原示范BC的定长、等更新数对照。"""
import copy
from pathlib import Path
import numpy as np
import torch

from experiments.skill_hierarchy.train import FeatureCache,balanced_schedule
from experiments.skill_hierarchy.metric_rollout import evaluate_closed
from experiments.skill_hierarchy.metric_train import FEEDBACK
from experiments.tcp_atomic_skills.train import loss_for
from experiments.tcp_atomic_skills.protocol import save,sha,identity
from experiments.tcp_memory_control.protocol import sampling_seed
from experiments.skill_dagger.data import CorrectiveWindows

SEED=1920042
STEPS=512


def schedules(dataset,corrective_count):
    nominal=balanced_schedule(dataset.buckets,STEPS,SEED)
    if corrective_count<1:raise ValueError('纠偏窗口为空')
    rng=np.random.default_rng(SEED+1);correction=[]
    while len(correction)<STEPS*2:correction.extend(rng.permutation(corrective_count).tolist())
    return nominal,np.array(correction[:STEPS*2]).reshape(STEPS,2)


def batch_rows(step,branch,nominal,corrective):
    rows=[('train',int(i)) for i in nominal[step]]
    if branch=='DAgger':
        for slot,index in zip((step%7,(step+3)%7),corrective[step]):rows[slot]=('corrective',int(index))
    elif branch!='BC':raise ValueError('未知对照分支')
    return rows


def compare(base,policy,fk,dataset,parent,parent_config,collection,output,budget,status):
    output=Path(output);output.mkdir(exist_ok=False)
    correction=CorrectiveWindows(collection)
    nominal,corrective=schedules(dataset,len(correction))
    config=dict(schema='skill-dagger-r1',steps=STEPS,windows_per_update=7,corrective_slots=2,
        seed=SEED,learning_rate=1e-5,parent_identity=parent['configuration_identity'],
        collection_sha256=sha(Path(collection)/'collection.json'),selection='fixed-final-512',
        nominal_data=dataset.hashes,corrective_windows=len(correction),feedback=list(FEEDBACK))
    config_id=identity(config);save(output/'config.json',config)
    oldcache=FeatureCache(Path(parent_config['cache_root']),base,{'train':dataset},parent_config['cache_identity'],lambda:budget.stopped)
    newcache=FeatureCache(output/'feature-cache',base,{'corrective':correction},config_id,lambda:budget.stopped)
    params=list(policy.expert.parameters());optimizer=torch.optim.AdamW(params,lr=1e-5)
    results={}
    def reset():
        policy.expert.load_state_dict(parent['expert'],strict=True)
        policy.memory_encoder.load_state_dict(parent['memory_encoder'],strict=True)
        optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
        torch.set_rng_state(parent['torch_rng']);torch.cuda.set_rng_state_all(parent['cuda_rng'])
    def persist(folder,branch,step,exposure,complete=False):
        payload=dict(format='skill-dagger-r1-checkpoint/v1',configuration_identity=config_id,parent_identity=parent['configuration_identity'],
            branch=branch,step=step,completed=complete,expert={k:v.detach().cpu() for k,v in policy.expert.state_dict().items()},
            memory_encoder={k:v.detach().cpu() for k,v in policy.memory_encoder.state_dict().items()},
            optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            schedule=nominal.tolist(),corrective_schedule=corrective.tolist(),exposure=exposure)
        temporary=folder/'latest.tmp';torch.save(payload,temporary);temporary.replace(folder/'latest.pt')
    for branch in ('BC','DAgger'):
        reset();folder=output/branch;folder.mkdir();losses=[];exposure=dict(train=0,corrective=0)
        step=0
        try:
            while step<STEPS:
                budget.check();policy.expert.train();optimizer.zero_grad(set_to_none=True);total=0.
                rows=batch_rows(step,branch,nominal,corrective)
                for slot,(split,index) in enumerate(rows):
                    x=(oldcache if split=='train' else newcache).get(split,index)
                    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                        loss=loss_for(policy,x,sampling_seed(SEED,step*7+slot))
                    if not torch.isfinite(loss):raise ValueError('非有限训练loss')
                    (loss/7).backward();total+=float(loss.detach())/7
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
                step+=1;losses.append(total)
                for split,_ in rows:exposure[split]+=1
                if step%16==0:status('training',branch=branch,step=step,loss_mean_last16=float(np.mean(losses[-16:])))
                if step%128==0:persist(folder,branch,step,exposure)
            persist(folder,branch,step,exposure,True);save(folder/'losses.json',losses)
            # 使用保存后的权重严格重载，再进入独立开发闭环。
            payload=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
            policy.expert.load_state_dict(payload['expert'],strict=True)
            results[branch]=dict(steps=step,exposure=exposure.copy(),checkpoint_sha256=sha(folder/'latest.pt'))
            save(folder/'result.json',dict(status='training_completed',**results[branch]));del payload
        except BaseException:
            persist(folder,branch,step,exposure);save(folder/'losses.json',losses);raise
    save(output/'candidates-frozen.json',results)
    for branch in ('parent','BC','DAgger'):
        status('evaluating_closed_loop',branch=branch)
        payload=parent if branch=='parent' else torch.load(output/branch/'latest.pt',map_location='cpu',weights_only=True)
        policy.expert.load_state_dict(payload['expert'],strict=True);policy.eval()
        evaluation=evaluate_closed(base,policy,fk,FEEDBACK,output/f'eval-{branch}',budget,[])
        if branch=='parent':results[branch]={}
        results[branch]['feedback']=evaluation['tasks']
        if branch!='parent':del payload
        save(output/'results.json',dict(status='running',results=results))
    result=dict(status='completed',results=results,training_converged=False)
    save(output/'results.json',result);return result
