"""Align既有闭环轨迹离线定位；只用FK与记录的固定GT目标，不调用学生或环境。"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from experiments.tcp_memory_control.kinematics import TCPKinematics
from experiments.tcp_memory_control.geometry import apply_delta
from experiments.skill_hierarchy.contract import cube_orientation_error_deg


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def error(pose, goal):
    return float(np.linalg.norm(pose[:3, 3]-goal[:3, 3])*1000), cube_orientation_error_deg(pose[:3, :3], goal[:3, :3])


def analyze_episode(folder, record, goal_world, fk):
    controls = [json.loads(s) for s in (folder/'control.jsonl').read_text().splitlines()]
    metrics = [json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
    plans = read(folder/'plans.json')
    assert len(controls) == len(metrics) == record['policy_steps']
    goal = np.linalg.inv(fk.world_from_base) @ goal_world
    checks = Counter()
    poses = []
    for j, (c, m) in enumerate(zip(controls, metrics), 1):
        assert c['step'] == m['step'] == j
        before, command, after = [fk.pose_base(np.array(c[k])) for k in ('q_before', 'command_q', 'q_after')]
        distance, angle = error(after, goal)
        errors = {
            'metric_position_mm': abs(distance-m['metrics']['pregrasp_distance_m']*1000),
            'metric_angle_deg': abs(angle-m['metrics']['orientation_error_deg']),
            'logged_tcp_mm': float(np.linalg.norm((fk.world_from_base@after)[:3, 3]-c['state']['tcp_position'])*1000),
        }
        for k, v in errors.items():
            checks[k] = max(checks[k], v)
        poses.append((before, command, after))
    assert checks['metric_position_mm'] < .1 and checks['logged_tcp_mm'] < .1, checks
    assert checks['metric_angle_deg'] < .1, checks
    initial = error(poses[0][0], goal)
    assert abs(initial[0]-record['entry']['pregrasp_distance_m']*1000)<.1
    assert abs(initial[1]-record['entry']['orientation_error_deg'])<.1
    steps, chunks = [], []
    for pi, plan in enumerate(plans):
        begin = plan['step']
        count = plan['execution']['executed_steps']
        assert 0 <= count <= 4
        if not count:
            continue
        anchor = poses[begin][0]
        target = anchor.copy()
        local = []
        for slot in range(count):
            j = begin+slot
            assert controls[j]['step'] == j+1
            target = apply_delta(target, np.asarray(plan['physical'][slot][:6]), anchor)
            before, command, after = poses[j]
            position_difference = float(np.linalg.norm(target[:3, 3]-command[:3, 3])*1000)
            orientation_difference = cube_orientation_error_deg(target[:3, :3], command[:3, :3])
            checks['plan_command_mm'] = max(checks['plan_command_mm'], position_difference)
            checks['plan_command_deg'] = max(checks['plan_command_deg'], orientation_difference)
            assert position_difference < .15 and orientation_difference < .15
            bdist, bang = error(before, goal)
            cdist, cang = error(command, goal)
            adist, aang = error(after, goal)
            row = dict(step=j+1, chunk=pi, slot=slot+1, before_mm=bdist, command_mm=cdist, actual_mm=adist,
                       before_deg=bang, command_deg=cang, actual_deg=aang,
                       command_away_position=bdist>8 and cdist-bdist>.05,
                       command_away_angle=bang>5 and cang-bang>.1,
                       tracking_mm=float(np.linalg.norm(command[:3, 3]-after[:3, 3])*1000),
                       gripper_target=controls[j]['gripper_target'])
            steps.append(row)
            local.append(row)
        start_d, start_a = error(anchor, goal)
        final = local[-1]
        chunks.append(dict(chunk=pi, start_step=begin, executed=count,
                           start_mm=start_d, end_command_mm=final['command_mm'], end_actual_mm=final['actual_mm'],
                           start_deg=start_a, end_command_deg=final['command_deg'], end_actual_deg=final['actual_deg'],
                           first_command_position_delta_mm=local[0]['command_mm']-start_d,
                           command_position_delta_mm=final['command_mm']-start_d,
                           command_angle_delta_deg=final['command_deg']-start_a,
                           actual_position_delta_mm=final['actual_mm']-start_d,
                           actual_angle_delta_deg=final['actual_deg']-start_a))
    assert len(steps) == len(controls)
    chunks4 = [x for x in chunks if x['executed']==4]
    summary = dict(seed=record['seed'], success=record['success'], stop_reason=record['stop_reason'],
                   steps=len(steps), chunks=len(chunks), initial_mm=initial[0], initial_deg=initial[1],
                   final_mm=steps[-1]['actual_mm'], final_deg=steps[-1]['actual_deg'],
                   first_away_position=next((x['step'] for x in steps if x['command_away_position']), None),
                   first_away_angle=next((x['step'] for x in steps if x['command_away_angle']), None),
                   first_chunk_position_regression=next((x['start_step'] for x in chunks4 if x['start_mm']>8 and x['command_position_delta_mm']>1), None),
                   first_chunk_angle_regression=next((x['start_step'] for x in chunks4 if x['start_deg']>5 and x['command_angle_delta_deg']>2), None),
                   first_step_good_suffix_bad=sum(x['start_mm']>8 and x['first_command_position_delta_mm']<-.05 and x['command_position_delta_mm']>1 for x in chunks4),
                   command_progress_actual_regress=sum(x['command_position_delta_mm']<-1 and x['actual_position_delta_mm']>1 for x in chunks4),
                   position_regressing_chunks=sum(x['start_mm']>8 and x['command_position_delta_mm']>1 for x in chunks4),
                   angle_regressing_chunks=sum(x['start_deg']>5 and x['command_angle_delta_deg']>2 for x in chunks4),
                   tracking_mm_median=float(np.median([x['tracking_mm'] for x in steps])),
                   tracking_mm_max=max(x['tracking_mm'] for x in steps),
                   by_slot={str(i):dict(count=sum(x['slot']==i for x in steps),
                                       position_away=sum(x['slot']==i and x['command_away_position'] for x in steps),
                                       angle_away=sum(x['slot']==i and x['command_away_angle'] for x in steps)) for i in range(1,5)})
    return dict(summary=summary, checks=dict(checks), steps=steps, chunks=chunks)


def main():
    p=argparse.ArgumentParser()
    for name in ('run-root','collection','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    collection=read(args.collection/'collection.json')
    source_rows={r['seed']:r for r in collection['records']}
    fk=TCPKinematics()
    result=dict(status='running',scope='post-hoc recorded development; FK only; oracle geometry is not a teacher action label',
                script_sha256=digest(Path(__file__)),thresholds=dict(position_gate_mm=8,angle_gate_deg=5,
                step_position_increase_mm=.05,step_angle_increase_deg=.1,chunk_position_increase_mm=1,chunk_angle_increase_deg=2),
                episodes=[],inputs={},limitations=['No saved evaluation RGB or full simulator snapshots.',
                'Cannot query original BC or teacher on exact diverged states from these records alone.',
                'Same-seed trajectories share initial conditions but later states differ.',
                'Diagnostic increase thresholds are post-hoc descriptions, not acceptance changes.'])
    names=('eval-bc-1','eval-r1-v3-DAgger-1','eval-r2-v4-DAgger-1')
    for name in names:
        manifest=args.run_root/name/'result.json';ev=read(manifest)
        assert ev['status']=='completed' and len(ev['records'])==8
        result['inputs'][str(manifest)]=digest(manifest)
        for record in ev['records']:
            assert record['skill']==1 and record['split']=='development'
            sr=source_rows[record['seed']]
            side=args.collection/sr['sidecar']
            assert digest(side)==sr['sidecar_sha256']
            result['inputs'][str(side)]=digest(side)
            goal=np.asarray(read(side)['world_from_pregrasp'])
            folder=args.run_root/name/f"{record['seed']}-1-standard"
            for f in ('plans.json','metrics.jsonl','control.jsonl'):
                result['inputs'][str(folder/f)]=digest(folder/f)
            episode=analyze_episode(folder,record,goal,fk)
            episode['run']=name;result['episodes'].append(episode)
    result['status']='completed'
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps([dict(run=e['run'],**e['summary'],checks=e['checks']) for e in result['episodes']],indent=2))


if __name__=='__main__':
    main()
