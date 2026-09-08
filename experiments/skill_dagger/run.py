"""有界三步执行：教师接管pilot，数据聚合，首轮BC/DAgger对照。"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import time
import torch

from experiments.tcp_atomic_skills.runtime import load_parent
from experiments.tcp_atomic_skills.protocol import save,sha,identity,verify_source
from experiments.skill_hierarchy.full_task import verify_upstream
from experiments.skill_hierarchy.data import SevenSkillWindows
from experiments.skill_hierarchy.metric_rollout import RunBudget
from experiments.skill_hierarchy.metric_train import FORMAT
from experiments.tcp_memory_control.kinematics import TCPKinematics
from experiments.skill_dagger.collect import collect
from experiments.skill_dagger.train import compare

PARENT_SHA='7cf5c95ba7dbe31843d615d7a550f9d1422167f3cc6e056dfefea8ee0e68b957'


class Budget(RunBudget):
    def consume(self,kind):
        if self.student+self.teacher>=37440:raise InterruptedError('本批37440控制步上限')
        super().consume(kind)


def verify_parent(payload,config,digest):
    if (digest!=PARENT_SHA or payload['format']!=FORMAT or payload['configuration_identity']!=identity(config)
            or not payload['completed'] or payload['branch']!='A' or payload['step']!=1024):
        raise ValueError('DAgger固定父学生身份不符')


def main():
    p=argparse.ArgumentParser()
    for name in ('collection','metric-run','checkpoint','model-cache','continued','parent-training','output','source-manifest'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--wait-for-evaluation',type=Path)
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('缺少确定性CUBLAS配置')
    torch.use_deterministic_algorithms(True)
    budget=Budget(None,args.output)
    def stop(*_):budget.stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def status(stage,**extra):
        record=dict(stage=stage,updated_unix=time.time(),**extra);save(args.output/'status.json',record)
        budget.save();print(json.dumps(record,ensure_ascii=False),flush=True)
    try:
        status('preflight');source=verify_source(args.source_manifest)
        old=json.loads((args.metric_run/'config.json').read_text())
        parent_path=args.metric_run/'A/latest.pt';digest=sha(parent_path)
        parent=torch.load(parent_path,map_location='cpu',weights_only=True);verify_parent(parent,old,digest)
        dataset=SevenSkillWindows(args.collection,'train')
        if dataset.hashes!=old['data']['train']:raise ValueError('训练数据发生变化')
        base,policy,upstream=load_parent(args);verify_upstream(upstream,old['upstream'])
        policy.expert.load_state_dict(parent['expert'],strict=True)
        policy.memory_encoder.load_state_dict(parent['memory_encoder'],strict=True)
        for module in (policy.context_encoder,policy.adapter,policy.memory_encoder):module.requires_grad_(False)
        policy.eval();fk=TCPKinematics()
        save(args.output/'config.json',dict(schema='skill-dagger-r1',source_manifest_sha256=source,
            parent_sha256=digest,parent_configuration_identity=identity(old),data=dataset.hashes,
            pilot_units=16,max_collection_units=64,training_steps_per_branch=512,control_step_limit=37440,
            stage_scope='Pick recovery pilot; full-task student evaluation',selection='fixed-final'))
        status('collecting_pilot_then_corrective')
        collected=collect(base,policy,fk,args.collection,args.output/'collection',budget,digest)
        if not collected['pilot_passed']:
            status('pilot_failed',recovered=sum(r['status']=='recovered' for r in collected['records'][:16]),planned=16)
            save(args.output/'result.json',dict(status='pilot_failed',training_started=False));return
        if args.wait_for_evaluation and not args.wait_for_evaluation.exists():
            status('waiting_for_previous_evaluation')
            while not args.wait_for_evaluation.exists():
                budget.check();time.sleep(30)
        status('training_preflight')
        result=compare(base,policy,fk,dataset,parent,old,args.output/'collection',args.output/'training',budget,status)
        save(args.output/'result.json',result);status('completed')
    except BaseException as exc:
        status('error' if not isinstance(exc,InterruptedError) else 'stopped',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':main()
