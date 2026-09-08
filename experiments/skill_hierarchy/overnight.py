"""云端串行续训；最低训练时长内不因开发集平台提前停止。"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def plateau(history):
    """开发监控的平台启发式，不能替代独立验证或闭环成功。"""
    if len(history)<6:return {'stable':False,'reason':'insufficient_evaluations'}
    recent=history[-5:]
    if recent[-1]['step']-recent[0]['step']<2048:
        return {'stable':False,'reason':'insufficient_span'}
    # 宏平均及每个技能都须稳定，避免收益与退化互相抵消。
    for skill in [None]+[str(i) for i in range(7)]:
        rows=[e['groups']['val']['macro'] if skill is None else
              e['groups']['val']['per_skill'][skill] for e in recent]
        for metric in ('translation_error_mm','rotation_delta_error_deg','gripper_mae','gripper_binary_accuracy'):
            values=[r[metric] for r in rows]
            if not all(math.isfinite(x) for x in values):
                raise ValueError('非有限开发评估指标')
            mean=sum(values)/len(values)
            if metric=='gripper_mae':tolerance=.001 if skill is None else .005
            elif metric=='gripper_binary_accuracy':tolerance=.005 if skill is None else .01
            else:tolerance=max(abs(mean),1e-9)*(.02 if skill is None else .05)
            if max(values)-min(values)>tolerance:
                return {'stable':False,'reason':'metrics_still_changing'}
    baseline=history[0]['groups']['val'];last=recent[-1]['groups']['val'];regressions=[]
    for skill in [str(i) for i in range(7)]:
        for metric in ('translation_error_mm','rotation_delta_error_deg','gripper_mae','gripper_binary_accuracy'):
            before=baseline['per_skill'][skill][metric];after=last['per_skill'][skill][metric]
            bad=(after<before-.01) if metric=='gripper_binary_accuracy' else (
                after>before+.005 if metric=='gripper_mae' else after>before*1.05)
            if bad:regressions.append(f'{skill}:{metric}')
    return {'stable':True,'reason':'plateau_with_regression' if regressions else 'offline_plateau',
            'regressions':regressions,'first_step':recent[0]['step'],'last_step':recent[-1]['step'],
            'closed_loop_verified':False}


def may_finish(active_seconds, minimum_seconds, assessment):
    return active_seconds>=minimum_seconds and assessment['stable']


def command_for(spec, output, source, previous):
    args=list(spec['command'])
    args[args.index('--output')+1]=str(output)
    args[args.index('--source-manifest')+1]=str(source)
    args.extend(['--continue-from',str(previous)])
    return args


def full_task_command(spec, training, output, source):
    original=spec['command']
    args=[original[0],'-m','experiments.skill_hierarchy.full_task']
    for key in ('checkpoint','model-cache','continued','parent-training'):
        args.extend(['--'+key,original[original.index('--'+key)+1]])
    args.extend(['--training',str(training),'--output',str(output),'--source-manifest',str(source)])
    return args


def process_matches(pid, output):
    path=Path(f'/proc/{pid}/cmdline')
    try:parts=path.read_bytes().split(b'\0')
    except FileNotFoundError:return False
    return str(output).encode() in parts and b'experiments.skill_hierarchy.train' in parts


def completed(root, name):
    output=root/name;result=read(output/'result.json');config=read(output/'config.json')
    if ((root/(name+'.exitcode')).read_text().strip()!='0' or result['status']!='completed'
            or not result['strict_reload'] or result['steps']!=config['steps']
            or result['unique_exposed_windows']!=sum(config['bucket_sizes'])):
        raise ValueError(f'{name} 完成验证失败')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--launch-spec',type=Path,required=True)
    parser.add_argument('--minimum-hours',type=float,default=7)
    parser.add_argument('--adopt-current-run',action='store_true')
    args=parser.parse_args();root=args.root;spec=read(args.launch_spec)
    if not math.isfinite(args.minimum_hours) or args.minimum_hours<7:
        raise ValueError('本次授权要求至少 7 小时')
    lock=(root/'overnight.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # 不自动恢复失败的 supervisor，避免重复消费阶段；由监控保留证据并报告。
    state_path=root/'overnight-status.json'
    if state_path.exists():
        if not args.adopt_current_run:raise ValueError('已有夜间任务状态，请核对后人工恢复')
        state=read(state_path)
        # 本次仅接管仍由原 launcher 管理的 v1；不推断或重启已有子训练。
        if (state['current_run']!='training-v1' or state['stage']!='training'
                or state['minimum_active_seconds']!=args.minimum_hours*3600
                or not process_matches(state['training_pid'],root/'training-v1')):
            raise ValueError('仅支持对当前仍运行的第一阶段接管调度')
        state.update(previous_supervisor_pid=state['supervisor_pid'],supervisor_pid=os.getpid(),
                     adopted_unix=time.time())
    else:
        state=dict(stage='waiting_current_training',started_unix=time.time(),active_seconds=0.,
                   minimum_active_seconds=args.minimum_hours*3600,current_run='training-v1',
                   supervisor_pid=os.getpid(),offline_converged=False,minimum_reached=False)
    state.update(evaluation_scope='atomic_offline',source_version='v3')
    write(root/'overnight-policy.json',dict(minimum_hours=args.minimum_hours,
        stopping='after minimum duration, assess five fixed evaluations spanning >=2048 updates at full-stage boundaries',
        macro_translation_rotation_range=.02,per_skill_translation_rotation_range=.05,
        macro_gripper_mae_range=.001,per_skill_gripper_mae_range=.005,
        macro_gripper_accuracy_range=.005,per_skill_gripper_accuracy_range=.01,
        early_transition='after full coverage and a stable development plateau without per-skill regression, run complete-task development and continue shared BC until minimum duration',
        full_task_development='fixed four development seeds; each <=400 actions; no teacher takeover; repeat at subsequent full-stage boundaries',
        full_task_training='same shared BC and complete teacher trajectories with cross-skill chunks; no new model or loss',
        meaning='development plateau heuristic; no closed-loop or formal convergence claim',
        on_failure='preserve checkpoints and stop; no automatic restart'))
    env=dict(os.environ,**spec['environment']);child=None;history=[];cycle=1
    def publish():
        state['updated_unix']=time.time();state['minimum_reached']=state['active_seconds']>=state['minimum_active_seconds']
        write(state_path,state)
    try:
        while True:
            name=f'training-v{cycle}';output=root/name
            pid=int((root/(name+'.pid')).read_text());state.update(current_run=name,training_pid=pid)
            last=time.monotonic()
            while process_matches(pid,output):
                now=time.monotonic();state['active_seconds']+=now-last;last=now
                state['stage']='training';publish();time.sleep(20)
            if child is not None:
                code=child.wait();(root/(name+'.exitcode')).write_text(str(code)+'\n');child=None
            else:
                for _ in range(10):
                    if (root/(name+'.exitcode')).exists():break
                    time.sleep(1)
            completed(root,name)
            config=read(output/'config.json');offset=config.get('training_offset',0)
            for entry in read(output/'history.json'):
                entry=dict(entry,step=offset+entry['step'])
                if not history or entry['step']>history[-1]['step']:history.append(entry)
            assessment=plateau(history);state.update(assessment=assessment,total_updates=history[-1]['step'])
            write(root/'overnight-history.json',history)
            if (assessment['stable'] and not assessment.get('regressions')) or state['evaluation_scope']=='full_task':
                state.update(stage='full_task_development',evaluation_scope='full_task');publish()
                evaluation=root/f'full-task-after-v{cycle}'
                cmd=full_task_command(spec,output,evaluation,root/'source-v3.json')
                with (root/f'full-task-after-v{cycle}.log').open('x') as log:
                    code=subprocess.call(cmd,cwd=root/'code-v3',env=env,stdout=log,stderr=subprocess.STDOUT)
                (root/f'full-task-after-v{cycle}.exitcode').write_text(str(code)+'\n')
                evaluated=read(evaluation/'result.json')
                if code or evaluated['status']!='completed':raise RuntimeError('完整任务开发检查执行异常')
                state['full_task_evaluation']=dict(run=str(evaluation),**evaluated['summary'])
                publish()
            if may_finish(state['active_seconds'],state['minimum_active_seconds'],assessment):
                state['stage']=assessment['reason']
                full=state.get('full_task_evaluation')
                if full and full['successes']<full['episodes']:
                    state['stage']='plateau_with_closed_loop_failures'
                publish();break
            if (root/'STOP_AFTER_STAGE').exists():
                state['stage']='user_stopped';publish();break
            if shutil.disk_usage(root).free<5*1024**3:raise RuntimeError('续训磁盘余量低于 5GiB')
            # GPU smoke 在当前训练退出后串行执行，不计入最低训练时长。
            if cycle==1:
                state['stage']='continuation_smoke';publish()
                smoke=root/'smoke-continuation-v2'
                cmd=command_for(spec,smoke,root/'source-v3.json',root/'smoke-v1')+['--smoke']
                with (root/'smoke-continuation-v2.log').open('x') as log:
                    code=subprocess.call(cmd,cwd=root/'code-v3',env=env,stdout=log,stderr=subprocess.STDOUT)
                (root/'smoke-continuation-v2.exitcode').write_text(str(code)+'\n')
                if code or not read(smoke/'result.json')['strict_reload']:
                    raise RuntimeError('续训 GPU smoke 未通过')
            cycle+=1;name=f'training-v{cycle}';next_output=root/name
            cmd=command_for(spec,next_output,root/'source-v3.json',output)
            with (root/(name+'.log')).open('x') as log:
                child=subprocess.Popen(cmd,cwd=root/'code-v3',env=env,stdout=log,stderr=subprocess.STDOUT,
                                       start_new_session=True)
            (root/(name+'.pid')).write_text(str(child.pid)+'\n')
    except BaseException as exc:
        state.update(stage='error',error=f'{type(exc).__name__}: {exc}');publish();raise


if __name__=='__main__':main()
