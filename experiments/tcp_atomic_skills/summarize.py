"""原子、组合与完整任务的明确分母，不丢弃准备失败或未运行。"""
from collections import Counter
import argparse
import json
from pathlib import Path
from experiments.tcp_atomic_skills.protocol import CASES,EVAL_SEEDS,save


def summarize(result):
    rows=result['records']
    keys=[(r['seed'],r['case'],r['arm']) for r in rows]
    expected={(s,c[0],a) for s in EVAL_SEEDS for c in CASES for a in ('before','after')}
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('评估分母重复或缺失')
    if result['status']=='completed' and any(r['status'] not in ('completed','preparation_failed') for r in rows):
        raise ValueError('不能把未完成单元计为完整实验')
    groups=[]
    for case,start,target,steps in CASES:
        for arm in ('before','after'):
            selected=[r for r in rows if r['case']==case and r['arm']==arm]
            executed=[r for r in selected if r['status']=='completed']
            # 邻接组合交接由第一技能完成定义；末阶段本身不计入条件分母。
            reached_next=[r for r in executed if r['final_completed']>=start+1] if target>start+1 else []
            success=sum(bool(r.get('success')) for r in selected)
            groups.append(dict(case=case,arm=arm,planned=len(selected),executed=len(executed),success=success,
                preparation_failed=sum(r['status']=='preparation_failed' for r in selected),
                statuses=dict(Counter(r['status'] for r in selected)),
                stop_reasons=dict(Counter(r['stop_reason'] for r in executed)),
                total_policy_steps=sum(r['policy_steps'] for r in executed),
                reached_next_stage=len(reached_next) if target>start+1 else None,
                success_given_first_stage=success/len(reached_next) if reached_next else None))
    return dict(schema='tcp-five-skills-summary/v1',status=result['status'],groups=groups,
        elapsed_s=result['elapsed_s'],
        limitation='四个新development场景重复用于技能条件；80单元不是80独立场景；前后是五技能扩展整体比较。')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();save(args.output,summarize(json.loads(args.input.read_text())))
