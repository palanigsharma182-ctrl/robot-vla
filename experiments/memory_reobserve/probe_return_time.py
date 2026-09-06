"""已见开发场景的返回时序事后诊断；不充当新的独立效果评估。"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from experiments.g2c_memory_integration.run import run
from experiments.memory_reobserve.runtime import load_memory_runtime, MemoryConditionedRuntime, M0_CANDIDATE_SHA256
from experiments.memory_reobserve.session import MemoryRouteSession


def main():
    p=argparse.ArgumentParser()
    for name in ('bundle','upstream','model-cache','candidate','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(seed=1001202, reason='post-hoc timing diagnosis of an already seen committed scene',
        return_steps=[40,10], primary_move_steps=40, home_barrier_steps=4,
        maximum_post_replans=12, starting_sample_index=2, maximum_process_seconds=240,
        max_memory_age_s=2.5, selection='report both; no threshold changes; no new training',
        claim='simulation timing only; camera has no physical motion dynamics model')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    started=time.monotonic()
    base,identity=load_memory_runtime(a.upstream,a.model_cache,a.candidate,M0_CANDIDATE_SHA256)
    (a.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    records=[]
    for steps in protocol['return_steps']:
        runtime=MemoryConditionedRuntime(base.policy,base.processor_adapter,base.proprio_normalizer,
            base.spec,base.device,replace(base.config,starting_sample_index=2))
        session=MemoryRouteSession(1001202,runtime,return_steps=steps,post_replans=12)
        output=a.output/str(steps)
        try:
            result=run(a.bundle,output,memory_session=session)
            status='completed'
        except Exception as e:
            result=dict(error_type=type(e).__name__,error=str(e));status='error'
        audit=json.loads((output/'memory-runtime.json').read_text())
        records.append(dict(return_steps=steps,status=status,result=result,
            available_sends=sum(r['available'] for r in audit['sends']),
            read_ages_s=[r['snapshot']['timestamp_s']-r['snapshot']['last_observed_timestamp_s']
                for r in audit['reads'] if r['snapshot']['available']],cleanup=audit['cleanup']))
        (a.output/'result.json').write_text(json.dumps(dict(protocol=protocol,records=records,
            elapsed_s=time.monotonic()-started),indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ('return_steps','status','available_sends','read_ages_s')} for r in records]))
    if any(r['status']=='error' for r in records):raise SystemExit(2)


if __name__=='__main__':main()
