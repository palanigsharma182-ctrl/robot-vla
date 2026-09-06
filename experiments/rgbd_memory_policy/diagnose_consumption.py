"""事后只读诊断：分离Memory token存在与XYZ内容对动作的影响，不发送动作。"""
import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.g2c_memory_integration.vla import load_runtime,sha256
from experiments.memory_conditioning.conditioning import MEMORY_INPUT_KEY,MemoryBatch,MemoryConditionedPolicy
from experiments.rgbd_memory_policy.data import load_examples
from experiments.rgbd_memory_policy.protocol import PROTOCOL,identity
from experiments.rgbd_memory_policy.stream import INSTRUCTION
from experiments.rgbd_memory_policy.train import restore,save_json
from robot_vla.runtime.policy_runtime import _move_model_inputs


def main():
    import json
    p=argparse.ArgumentParser()
    for name in ('data','checkpoint','model-cache','training','source-manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.source_manifest.read_text())
    if any(sha256(path)!=digest for path,digest in manifest.items()):raise ValueError('原运行源码身份不匹配')
    raw,data_hashes,denominator=load_examples(a.data);selected={}
    result=json.loads((a.training/'result.json').read_text())
    training_identity=json.loads((a.training/'identity.json').read_text())
    if (result['status']!='completed' or identity(training_identity)!=result['training_identity_sha256']
        or training_identity['data_sha256']!=data_hashes
        or training_identity['source_manifest_sha256']!=sha256(a.source_manifest)):
        raise ValueError('诊断数据并非所选训练checkpoint使用的同一collection')
    if {x['seed'] for x in raw['development']}!=set(PROTOCOL['development_seeds']):
        raise ValueError('缺少原来8个development场景')
    for x in sorted(raw['development'],key=lambda x:(x['seed'],x['anchor'])):
        if x['snapshot']['available'] and x['seed'] not in selected:selected[x['seed']]=x
    if len(selected)!=5:raise ValueError('必须复用已核验collection-v1的5个可用development场景')
    protocol=dict(status='post-hoc-read-only-diagnostic',training=False,actuator_steps=0,
        selection='first available anchor in each development scene; no prediction-based selection',
        selected=[dict(seed=x['seed'],anchor=x['anchor']) for x in selected.values()],
        development_denominator=[dict(seed=s,status='selected' if s in selected else 'no-available-memory') for s in PROTOCOL['development_seeds']],
        training_identity_sha256=result['training_identity_sha256'],data_sha256=data_hashes,
        variants=['original','repeat-original','mask','x-plus-10cm','z-plus-10cm','zero-xyz'],
        seed=73,flow_steps=10,precision='BF16, unchanged real runtime path',
        script_sha256=sha256(__file__),source_manifest_sha256=sha256(a.source_manifest))
    save_json(a.output/'protocol.json',protocol)
    runtime,upstream=load_runtime(a.checkpoint,a.model_cache)
    policy=MemoryConditionedPolicy(runtime.policy.context_encoder,runtime.policy.expert,runtime.policy.adapter).to('cuda')
    restore(policy,a.training/'memory.pt',result['results']['memory']['checkpoint_sha256'],
        result['training_identity_sha256'],'memory',runtime.proprio_normalizer,upstream,sha256(a.source_manifest))
    policy.eval();rows=[]
    for x in selected.values():
        encoded=runtime.processor_adapter.encode(x['rgb_external'],x['rgb_wrist'],INSTRUCTION)
        inputs=_move_model_inputs(encoded.model_inputs,runtime.device)
        proprio=torch.tensor(runtime.proprio_normalizer.normalize(x['physical_proprio'])[None],device=runtime.device)
        outputs={}
        for variant in protocol['variants']:
            features=np.array(x['snapshot']['features'],np.float32)
            if variant=='x-plus-10cm':features[0]+=.10
            if variant=='z-plus-10cm':features[2]+=.10
            if variant=='zero-xyz':features[:3]=0.
            payload={**inputs,MEMORY_INPUT_KEY:MemoryBatch(torch.tensor(features[None],device=runtime.device),
                torch.tensor([[variant!='mask']],dtype=torch.bool,device=runtime.device))}
            with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                prediction=policy.sample_actions(payload,proprio,num_steps=10,
                    generator=torch.Generator(device=runtime.device).manual_seed(protocol['seed']))
            outputs[variant]=prediction[0].float().cpu().numpy()
        row=dict(seed=x['seed'],anchor=x['anchor'],repeat_bitwise_equal=bool(np.array_equal(outputs['original'],outputs['repeat-original'])),effects={})
        for variant in protocol['variants'][2:]:
            delta=outputs[variant][:,:7]-outputs['original'][:,:7]
            physical_delta=delta*np.asarray(runtime.action_adapter.delta_limits)[None]
            row['effects'][variant]=dict(normalized_joint_chunk_l2=float(np.linalg.norm(delta)),
                first4_joint_rms_rad=float(np.sqrt(np.mean(physical_delta[:4]**2))),
                first4_max_abs_rad=float(np.abs(physical_delta[:4]).max()))
        rows.append(row)
        save_json(a.output/'result.json',dict(protocol=protocol,rows=rows,
            checkpoint_sha256=result['results']['memory']['checkpoint_sha256'],
            limitation='Counterfactual numeric sensitivity, not accuracy, directional correctness or task benefit. Mask changes context length; XYZ variants retain token and mask.'))
    print(json.dumps(dict(samples=len(rows),repeat_equal=all(r['repeat_bitwise_equal'] for r in rows),actuator_steps=0)))


if __name__=='__main__':main()
