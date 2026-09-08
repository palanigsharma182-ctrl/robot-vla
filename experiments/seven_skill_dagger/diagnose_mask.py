"""只读离线诊断：同一真实train入口、同一初始噪声，比较推理mask的影响。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.seven_skill_dagger.run import load_student
from experiments.seven_skill_dagger.data import skill_targets
from experiments.skill_hierarchy.data import SevenSkillWindows
from experiments.skill_hierarchy.train import FeatureCache
from experiments.skill_dagger.data import training_seeds
from experiments.tcp_atomic_skills.data import action_spec
from experiments.tcp_atomic_skills.protocol import sha, save, verify_source
from experiments.tcp_memory_control.protocol import sampling_seed
from experiments.memory_conditioning.conditioning import MemoryBatch
from robot_vla.training.flow_matching import euler_integrate_actions


def main():
    p=argparse.ArgumentParser()
    for name in ('launch-spec','output','candidate','source-manifest'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    source=verify_source(args.source_manifest)
    spec=json.loads(args.launch_spec.read_text())
    common=argparse.Namespace(**{k[2:].replace('-','_'):Path(v) for k,v in zip(spec['common'][::2],spec['common'][1::2])})
    common.student=args.candidate.parent.parent/'bc-1/latest.pt'
    common.expected_sha=sha(common.student);common.skill=1
    base,policy,_,old=load_student(common)
    dataset=SevenSkillWindows(common.collection,'train')
    if dataset.hashes!=old['data']['train']:raise ValueError('缓存数据来源改变')
    allowed=training_seeds(common.collection,8)
    indices=[]
    for seed in allowed:
        indices.append(next(i for i in dataset.buckets[1] if dataset.entries[dataset.index[i][0]].randomization['seed']==seed))
    cache=FeatureCache(Path(old['cache_root']),base,{'train':dataset},old['cache_identity'],lambda:False)
    if any(not (cache.root/f'train-{i:06d}.pt').exists() for i in indices):raise ValueError('只读诊断不生成缺失缓存')
    result=dict(status='running',source_sha256=source,diagnostic_sha256=sha(__file__),data_use='train-only offline diagnostic',
                masks=[1,4,16],paired_noise=True,first_action_only=True,rows=[],models={},indices=indices)
    for name,path in (('BC512',common.student),('R2-DAgger',args.candidate)):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload['skill_id']!=1 or not payload['completed']:raise ValueError('策略身份非法')
        result['models'][name]=sha(path)
        policy.expert.load_state_dict(payload['expert'],strict=True)
        policy.memory_encoder.load_state_dict(payload['memory_encoder'],strict=True);policy.eval()
        for index in indices:
            x=cache.get('train',index);e,anchor=dataset.index[index]
            target,_=skill_targets(dataset.labels[e],anchor,1)
            truth=action_spec().denormalize(target)[0]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                context=policy.condition_context(x['context'],MemoryBatch(x['features'],x['available']))
                kv=policy.expert.prepare_context_kv(context)
                for horizon in (1,4,16):
                    mask=(torch.arange(16,device='cuda')<horizon)[None]
                    noise=torch.randn((1,16,7),device='cuda',generator=torch.Generator(device='cuda').manual_seed(sampling_seed(x['seed'],anchor)))
                    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                        prediction=euler_integrate_actions(lambda a,t:policy.expert(context,x['proprio'],a,t,mask,context_kv=kv),noise,mask,num_steps=10)
                    action=action_spec().denormalize(prediction[0].float().cpu().numpy())[0]
                    if not np.isfinite(action).all():raise ValueError('非有限预测')
                    result['rows'].append(dict(model=name,seed=x['seed'],anchor=anchor,mask_steps=horizon,
                        translation_error_mm=float(np.linalg.norm(action[:3]-truth[:3])*1000),
                        rotation_error_deg=float(np.linalg.norm(action[3:6]-truth[3:6])*180/np.pi),
                        gripper_mae=float(abs(action[6]-truth[6])),predicted=action.tolist(),target=truth.tolist()))
            save(args.output/'result.json',result)
    result['summary']={name:{str(h):{k:float(np.mean([r[k] for r in result['rows'] if r['model']==name and r['mask_steps']==h]))
        for k in ('translation_error_mm','rotation_error_deg','gripper_mae')} for h in (1,4,16)} for name in result['models']}
    result['status']='completed';save(args.output/'result.json',result);print(json.dumps(result['summary']))


if __name__=='__main__':main()
