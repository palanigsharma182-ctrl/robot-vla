"""只读结果汇总：场景等权 loss、完整评估分母、组合交接与 Memory 实际曝光。"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean_by_scene(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[row['seed']].append(row['flow_mse'])
    return {str(k):statistics.mean(v) for k,v in sorted(grouped.items())}


def failure_category(value):
    # 公开聚合只保留异常类型；原始异常（可能含私有路径）留在运行目录。
    return value.split(':', 1)[0] if value else None


def acquisition_summary(evaluation, rows):
    outcomes=[]
    for row in rows:
        if not row.get('request'):
            continue
        path=evaluation/f"{row['condition']}-{row['case']}-{row['seed']}"/'result.json'
        detail=json.loads(path.read_text())
        candidate=detail.get('candidate') or {}
        outcomes.append(dict(case=row['case'],seed=row['seed'],condition=row['condition'],
            memory_write_count=detail['memory_write_count'],
            rejection_reasons=detail.get('terminal_reasons',[]),
            information_gain=candidate.get('information_gain'),
            minimum_score=candidate.get('minimum_score')))
    return outcomes


def summarize(training,evaluation):
    trained=json.loads((training/'result.json').read_text())
    schedule=json.loads((training/'schedule.json').read_text())
    evaluated=json.loads((evaluation/'result.json').read_text())
    if trained['status']!='training-completed' or any(a['steps']!=1024 for a in trained['arms'].values()):
        raise ValueError('两臂没有完成预定训练，不能生成完整比较')
    rows=evaluated['records'];protocol=evaluated['protocol']
    expected={(c[0],seed,condition) for c in protocol['cases'] for seed in protocol['seeds'] for condition in protocol['conditions']}
    actual=[(r['case'],r['seed'],r['condition']) for r in rows]
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('评估分母存在遗漏或重复')
    groups={};overview={}
    for condition in protocol['conditions']:
        for case,start,target,_ in protocol['cases']:
            selected=[r for r in rows if r['condition']==condition and r['case']==case]
            completed=[r for r in selected if r['status']=='completed']
            success=sum(r['result']['success'] for r in completed)
            first_reached=sum(r['result']['final_completed']>=start+1 for r in completed)
            groups[condition+'/'+case]=dict(planned=len(selected),completed=len(completed),success=success,
                states=dict(Counter(r['status'] for r in selected)),
                failures=dict(Counter(failure_category(r['result']['failure']) for r in completed if r['result']['failure'])),
                error_types=dict(Counter(r.get('error_type','unknown') for r in selected if r['status']=='error')),
                initial_skill_count=start,target_skill_count=target,
                first_skill_reached=first_reached,
                conditional_success_denominator=first_reached if target==start+2 else None,
                memory_inferences=sum(r['result']['memory_inferences'] for r in completed),
                visual_inferences=sum(r['result']['visual_inferences'] for r in completed),
                requests=sum(r.get('request',False) for r in selected),
                memory_commits=sum(r.get('memory_write_count',0) for r in selected),
                policy_steps=sum(r['result']['policy_steps'] for r in completed),
                observation_prefix_steps=sum(r['result']['observation_prefix_steps'] for r in completed),
                final_skill_counts=dict(Counter(r['result']['final_completed'] for r in completed)))
        selected=[r for r in rows if r['condition']==condition and r['status']=='completed']
        overview[condition]=dict(completed=len(selected),requests=sum(r.get('request',False) for r in selected),
            memory_commits=sum(r.get('memory_write_count',0) for r in selected),
            memory_inferences=sum(r['result']['memory_inferences'] for r in selected),
            visual_inferences=sum(r['result']['visual_inferences'] for r in selected),
            cleanup_reasons=dict(Counter(e['reason'] for r in selected for e in r['result']['cleanup'])))
    loss={}
    for arm,data in trained['arms'].items():
        scene=mean_by_scene(data['development']);masked=mean_by_scene(data['masked_development'])
        loss[arm]=dict(scene_mean_flow_mse=statistics.mean(scene.values()),scene_flow_mse=scene,
            masked_scene_mean_flow_mse=statistics.mean(masked.values()),
            available_anchor_flow_mse=statistics.mean(r['flow_mse'] for r in data['development'] if r['available']),
            checkpoint_sha256=data['checkpoint_sha256'],steps=data['steps'])
    execution_errors=[dict(case=r['case'],seed=r['seed'],condition=r['condition'],
        error_type=r.get('error_type') or failure_category(r.get('result',{}).get('failure')))
        for r in rows if r['status']=='error' or ':' in (r.get('result',{}).get('failure') or '')]
    memory_inferences=sum(o['memory_inferences'] for o in overview.values())
    result=dict(schema='memory-reobserve-development-summary/v1',
        status='complete' if not execution_errors and all(r['status']=='completed' for r in rows) else 'incomplete-or-errors',
        source_results={'training_sha256':sha(training/'result.json'),'evaluation_sha256':sha(evaluation/'result.json')},
        data_coverage=trained['coverage'],training_identity_sha256=trained['training_identity_sha256'],
        train_conditions=loss,training_exposure=schedule['condition_exposure'],
        effective_memory_fraction=schedule['memory_exposure']['token-on']/sum(schedule['memory_exposure'].values()),
        evaluation_protocol=protocol,evaluation_states=dict(Counter(r['status'] for r in rows)),
        unique_evaluation_scenes=len(set(protocol['seeds'])),execution_errors=execution_errors,
        memory_effect_evidence='inconclusive-no-memory-consumption' if memory_inferences==0 else 'development-comparison-only',
        observation_outcomes=acquisition_summary(evaluation,rows),
        condition_overview=overview,groups=groups,
        scope='development only; four scenes per task/condition, no final-test/physical claim')
    return result


def markdown(result):
    loss=result['train_conditions']
    lines=['# Memory 与一次观察的开发结果','',
        '以下为固定预算内的开发实验，不是历史 final test 或实机验证。训练使用预定1024次更新，不代表全量数据收敛。','',
        '| 训练条件 | 每场景先平均的 Flow MSE | 更新次数 |','|---|---:|---:|']
    for arm in ('visual','memory'):
        lines.append(f"| {arm} | {loss[arm]['scene_mean_flow_mse']:.9f} | {loss[arm]['steps']} |")
    delta=100*(loss['memory']['scene_mean_flow_mse']/loss['visual']['scene_mean_flow_mse']-1)
    lines+=['',f'Memory 相对视觉的开发 loss 变化为 {delta:+.3f}%（越低越好）。有效 Memory 实际占训练曝光的 {100*result["effective_memory_fraction"]:.2f}%。',
        '开发 loss 先在各场景内平均，再对8个场景等权平均；有合法 Memory 的仅3个场景，不能把多个锚点或噪声抽样当作独立场景。','',
        '| 任务 | visual 成功/完成 | fixed 成功/完成 | evidence 成功/完成 |','|---|---:|---:|---:|']
    for case,*_ in result['evaluation_protocol']['cases']:
        vals=[result['groups'][c+'/'+case] for c in ('visual','fixed','evidence')]
        lines.append('| '+case+' | '+' | '.join(f"{x['success']}/{x['completed']}" for x in vals)+' |')
    lines+=['','每格计划4个场景，所有任务和条件复用这4个新 development seeds，不能把120个单元视为120个独立场景。完整执行不等于任务成功。原子技能由专家准备起点，组合只在起点准备，交接不重置或补教师。',
        'fixed/evidence 共用同一 Memory checkpoint。evidence 仅在起始三个真实 HOME tick 使用未校准的 score 请求候选；HOME 分数不被冒充合格三维测量。','',
        '| 条件 | 完成单元 | 请求 | 合格 Memory 提交 | Memory 推理 | 纯视觉推理 |','|---|---:|---:|---:|---:|---:|']
    for c,o in result['condition_overview'].items():
        lines.append(f"| {c} | {o['completed']} | {o['requests']} | {o['memory_commits']} | {o['memory_inferences']} | {o['visual_inferences']} |")
    if result['memory_effect_evidence']=='inconclusive-no-memory-consumption':
        lines+=['','**本次闭环没有实际消费有效 Memory，Memory 动作收益不可估计。** 这能诊断当前观察/提交链的覆盖问题，不能作为 Memory 本身无效的结论。']
    lines+=['','| 相邻组合 | visual 成功/已完成前一技能 | fixed | evidence |',
        '|---|---:|---:|---:|']
    for case,start,target,_ in result['evaluation_protocol']['cases']:
        if target!=start+2:
            continue
        values=[]
        for condition in ('visual','fixed','evidence'):
            group=result['groups'][condition+'/'+case]
            denominator=group['conditional_success_denominator']
            values.append(f"{group['success']}/{denominator}" if denominator else '不可估计（分母0）')
        lines.append('| '+case+' | '+' | '.join(values)+' |')
    lines+=['','抓后原子技能没有伪造的 Memory；抓前 Memory 失效后清理旧动作并用实时视觉继续。未运行、实现错误、任务失败和完整分母保留在配套 JSON 中。','',
        '训练、数据和评估协议见 [README.md](README.md)；配套聚合结果见 [summary.json](summary.json)。','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser()
    for name in ('training','evaluation','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=summarize(a.training,a.evaluation)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'results.md').write_text(markdown(result))
    print(json.dumps(dict(status=result['status'],evaluation_states=result['evaluation_states'])))


if __name__=='__main__':main()
