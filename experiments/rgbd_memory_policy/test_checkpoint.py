"""小型合成权重验证新身份拒绝；不加载真实Qwen或使用GPU。"""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.front_rgbd_memory.memory import candidate_config
from experiments.g2c_memory_integration.vla import CHECKPOINT_SHA256,sha256
from experiments.memory_conditioning.conditioning import MEMORY_SCHEMA
from experiments.rgbd_memory_policy.protocol import PROTOCOL,identity
from experiments.rgbd_memory_policy.train import FORMAT,implementation_identity,restore


@pytest.mark.parametrize('corruption',[None,'incomplete','no_consumption','instruction','upstream','arm','source_manifest'])
def test_checkpoint_rejects_unmatched_training_identity(tmp_path,corruption):
    policy=SimpleNamespace(expert=torch.nn.Linear(2,2),memory_encoder=torch.nn.Linear(12,2))
    normalizer=SimpleNamespace(mean=np.zeros(15),std=np.ones(15),clip=10.)
    upstream={'checkpoint_sha256':CHECKPOINT_SHA256,'test':'synthetic'}
    ident=dict(protocol=PROTOCOL,upstream_sha256=CHECKPOINT_SHA256,
        source_manifest_sha256='a'*64,
        memory_schema=MEMORY_SCHEMA,memory_config=asdict(candidate_config()),
        implementation=implementation_identity(),upstream_identity=upstream.copy(),
        proprio_normalization=dict(mean=normalizer.mean.tolist(),std=normalizer.std.tolist(),clip=10.))
    payload=dict(format=FORMAT,arm='memory',steps=PROTOCOL['train_steps_per_arm'],completed=True,
        memory_consumption_validated=True,training_identity=ident,expert=policy.expert.state_dict(),
        memory_encoder=policy.memory_encoder.state_dict())
    if corruption=='incomplete':payload['completed']=False
    if corruption=='no_consumption':payload['memory_consumption_validated']=False
    if corruption=='instruction':ident['implementation']['instruction']='different task'
    if corruption=='upstream':ident['upstream_identity']['test']='different'
    if corruption=='arm':payload['arm']='visual'
    if corruption=='source_manifest':ident['source_manifest_sha256']='b'*64
    digest=identity(ident);payload['training_identity_sha256']=digest
    path=tmp_path/'synthetic.pt';torch.save(payload,path)
    args=(policy,path,sha256(path),digest,'memory',normalizer,upstream,'a'*64)
    if corruption:
        with pytest.raises(ValueError):restore(*args)
    else:restore(*args)
