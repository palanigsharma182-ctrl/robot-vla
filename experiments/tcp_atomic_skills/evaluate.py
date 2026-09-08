"""全部五原子、四相邻组合与整任务；保留训练前/后完整分母。"""
import argparse
import copy
import json
from pathlib import Path
import signal
import time

from experiments.tcp_atomic_skills.protocol import PROTOCOL, CASES, EVAL_SEEDS, SKILLS, save, sha, verify_source
from experiments.tcp_atomic_skills.runtime import AtomicController, AtomicExecutor, predict, execute_plan, load_parent
from experiments.tcp_atomic_skills.train import restore_final
from experiments.tcp_memory_control.policy import build_policy
from experiments.tcp_memory_control.kinematics import TCPKinematics
from experiments.tcp_memory_control.protocol import sampling_seed


def main():
    from robot_vla.sim.collector import TrustedPickPlaceCollector, EpisodeRejected
    from robot_vla.evaluation.maniskill import _reset_atomic_time_limit
    from robot_vla.tasks.pick_place import build_pick_place_task
    p=argparse.ArgumentParser()
    for key in ('training','checkpoint','model-cache','continued','parent-training','output','source-manifest'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();source=verify_source(args.source_manifest)
    training=json.loads((args.training/'result.json').read_text());config=json.loads((args.training/'config.json').read_text())
    if training['status']!='completed':raise ValueError('训练尚未完成')
    args.output.mkdir(exist_ok=False);started=time.monotonic()
    base,before,parent_identity=load_parent(args);after=build_policy(base.policy,'tcp-relative').to('cuda')
    restore_final(after,args.training/'latest.pt',training,config);after.eval();fk=TCPKinematics()
    policies={'before':before,'after':after}
    rows=[dict(seed=s,case=name,start_skill=start,target_completed=target,max_steps=steps,arm=arm,status='not_run')
          for s in EVAL_SEEDS for name,start,target,steps in CASES for arm in policies]
    result=dict(status='running',protocol=PROTOCOL,records=rows,source_sha256=source,
                training_result_sha256=sha(args.training/'result.json'),parent=parent_identity)
    def persist():
        result['elapsed_s']=time.monotonic()-started;save(args.output/'result.json',result)
    def stop(*_):raise InterruptedError('收到停止请求，保留已经完成的评估单元')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);persist()
    try:
        with TrustedPickPlaceCollector(None,max_episode_steps=1000) as collector:
            for seed in EVAL_SEEDS:
                for name,start,target,steps in CASES:
                    pair=[r for r in rows if r['seed']==seed and r['case']==name]
                    try:
                        # 每一对只准备一次；同一真实起点快照分别交给两个学生。
                        prep=copy.deepcopy(collector.prepare_atomic(seed=seed,skill_name=SKILLS[start]))
                        state=copy.deepcopy(collector.env.unwrapped.get_state_dict())
                        stable_place=collector.env.unwrapped._stable_place_steps.clone()
                    except EpisodeRejected as error:
                        for row in pair:row.update(status='preparation_failed',error=str(error),success=False)
                        persist();continue
                    initial=None
                    for row in pair:
                        collector.env.reset(seed=seed)
                        collector.env.unwrapped.set_state_dict(copy.deepcopy(state))
                        collector.env.unwrapped._stable_place_steps.copy_(stable_place)
                        _reset_atomic_time_limit(collector.env)
                        folder=args.output/f'{row["arm"]}-{seed}-{name}';folder.mkdir()
                        controller=AtomicController(collector.env,copy.deepcopy(prep),target,steps,
                            build_pick_place_task(seed%3).instruction,folder)
                        audit=controller.audit();save(folder/'initial.json',audit)
                        if initial is None:initial=audit
                        elif audit!=initial:raise ValueError('恢复后的前后两臂初态不一致')
                        row.update(status='running',initial_state=audit);persist()
                        executor=AtomicExecutor(fk);plans=[]
                        try:
                            while controller.stop_reason is None:
                                anchor=fk.pose_base(controller.read_state().joint_positions)
                                noise_seed=sampling_seed(seed,start*100000+target*10000+len(plans))
                                physical=predict(base,policies[row['arm']],controller.online(),noise_seed,
                                                 parent=row['arm']=='before')
                                plan=dict(step=controller.steps,sampling_seed=noise_seed,physical=physical.tolist(),
                                          base_from_tcp=anchor.tolist(),memory_available=False)
                                plans.append(plan)
                                try:
                                    plan['execution']=execute_plan(executor,controller,physical,anchor)
                                    plan['ik_targets']=executor.last_targets
                                except ValueError as error:
                                    controller.stop_reason='plan-rejected';plan['rejection']=str(error)
                                    break
                            row.update(status='completed',**controller.result())
                        finally:
                            save(folder/'plans.json',plans);persist()
                        print(json.dumps({k:v for k,v in row.items() if k!='initial_state'}),flush=True)
        result['status']='completed'
    except BaseException as error:
        result.update(status='error',error_type=type(error).__name__,error=str(error));raise
    finally:
        persist()


if __name__=='__main__':main()
