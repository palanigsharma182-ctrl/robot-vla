"""保留完整场景分母，单独比较配对共同执行前缀。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def summarize(result):
    rows=result['records'];seeds=list(range(1600200,1600216))
    expected={(arm,seed) for arm in ('before','after') for seed in seeds}
    keys=[(r['arm'],r['seed']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:
        raise ValueError('场景分母重复或缺失')
    valid=[r for r in rows if r['status'] in ('completed','stopped')]
    if result['status']=='completed' and len(valid)!=32:
        raise ValueError('未完成全部32个episode，不能声明完成')
    groups={}
    for arm in ('before','after'):
        selected=[r for r in valid if r['arm']==arm]
        groups[arm]=dict(planned=16,evaluated=len(selected),
            reached=sum(r['reached'] for r in selected),
            completed_88_steps=sum(r['status']=='completed' and r['policy_steps']==88 for r in selected),
            policy_steps_median=statistics.median([r['policy_steps'] for r in selected]) if selected else None,
            policy_steps_total=sum(r['policy_steps'] for r in selected),
            stopping_reasons=dict(Counter(r.get('ending_reason') for r in selected if r['status']=='stopped')))
    paired=[]
    for seed in seeds:
        pair={r['arm']:r for r in valid if r['seed']==seed}
        if len(pair)!=2:continue
        common=min(r['policy_steps'] for r in pair.values())
        distances={a:r['distance_by_policy_step'][str(common)] if common else r['initial_distance_m']
                   for a,r in pair.items()}
        paired.append(dict(seed=seed,common_steps=common,common_distance_m=distances,
            after_minus_before_common_distance_mm=1000*(distances['after']-distances['before']),
            reached={a:r['reached'] for a,r in pair.items()},steps={a:r['policy_steps'] for a,r in pair.items()}))
    return dict(schema='tcp-continuation-closed-loop-summary/v1',status=result['status'],groups=groups,
        paired=paired,elapsed_s=result['elapsed_s'],
        limitation='既有development事后比较；不同比较终点时长不作公平排名；不支持物理部署或抓取。')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();result=summarize(json.loads(args.input.read_text()))
    with args.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result['groups'],ensure_ascii=False))
