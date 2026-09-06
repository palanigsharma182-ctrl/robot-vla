"""真实冻结 Qwen/候选权重的无 actuator Runtime 对照，不作任务效果结论。"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np

from experiments.memory_reobserve.runtime import load_memory_runtime, MemoryConditionedRuntime
from experiments.memory_conditioning.conditioning import MemorySnapshot
from robot_vla.runtime.policy_runtime import OnlineObservation


def main():
    p=argparse.ArgumentParser()
    for n in ('upstream','model-cache','candidate','sample','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--candidate-sha256',required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    runtime,identity=load_memory_runtime(a.upstream,a.model_cache,a.candidate,a.candidate_sha256)
    metadata=json.loads(a.sample.with_suffix('.json').read_text())
    raw=metadata['snapshot']; raw['features']=tuple(raw['features']); raw['reasons']=tuple(raw['reasons'])
    snap=MemorySnapshot(**raw)
    with np.load(a.sample,allow_pickle=False) as d:
        obs=OnlineObservation(d['rgb_external'],d['rgb_wrist'],d['physical_proprio'],
            'pick the cube and place it in the target region')
    runtime.reset_memory_episode(snap.episode_id)
    runtime._bind_snapshot(obs,snap,frame_timestamp_s=snap.timestamp_s)
    present=runtime.infer_action_chunk(obs)
    # 两个独立 Runtime 共享只读模型，同一真实帧/时间和采样seed作反事实诊断。
    comparison=MemoryConditionedRuntime(runtime.policy, runtime.processor_adapter, runtime.proprio_normalizer,
        runtime.spec, runtime.device, runtime.config)
    comparison.reset_memory_episode(snap.episode_id)
    comparison._bind_snapshot(obs,replace(snap,available=False,features=(0.,)*12,
        reasons=('diagnostic-memory-masked',)),frame_timestamp_s=snap.timestamp_s)
    masked=comparison.infer_action_chunk(obs)
    result=dict(status='real-runtime-no-actuator',identity=identity,shape=list(present.normalized_action.shape),
        finite=bool(np.isfinite(present.normalized_action).all()),
        memory_token_count=present.context_length-masked.context_length,
        same_sampling_seed=present.sampling.seed==masked.sampling.seed,
        action_difference_l2=float(np.linalg.norm(present.normalized_action-masked.normalized_action)),
        reads=runtime.memory_reads, masked_reads=comparison.memory_reads,actuator_steps=0,new_training_steps=0,elapsed_s=time.monotonic()-started)
    if not(result['finite'] and result['same_sampling_seed'] and result['memory_token_count']==1):
        raise RuntimeError('真实 Runtime 输入或对照无效')
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','shape','finite','memory_token_count','action_difference_l2','elapsed_s')}))


if __name__=='__main__':main()
