"""小型真实Expert验证7D forward/loss/gradient、共同初始化及旧身份拒绝。"""
from types import SimpleNamespace
import pytest
import torch
from experiments.memory_conditioning.probe import build_probe
from experiments.memory_conditioning.conditioning import MemoryBatch,MEMORY_INPUT_KEY
from experiments.tcp_memory_control.policy import build_policy,RESET_PREFIXES
from experiments.tcp_memory_control.train import restore,FORMAT,verify_source,REQUIRED_SOURCE_PATHS
from experiments.tcp_memory_control.protocol import PROTOCOL,identity,sampling_seed
from experiments.g2c_memory_integration.vla import sha256


def test_two_arms_share_trunk_and_memory_but_reinitialize_action_projections():
    base,_,inputs,proprio=build_probe()
    joint=build_policy(base,'joint-world');tcp=build_policy(base,'tcp-relative')
    assert joint.expert.config.action_dim==8 and tcp.expert.config.action_dim==7
    for name,value in joint.expert.state_dict().items():
        if not name.startswith(RESET_PREFIXES):assert torch.equal(value,tcp.expert.state_dict()[name])
    for name,value in joint.memory_encoder.state_dict().items():assert torch.equal(value,tcp.memory_encoder.state_dict()[name])
    assert not torch.equal(joint.expert.velocity_head.weight,base.expert.velocity_head.weight)
    memory=MemoryBatch(torch.ones(2,12)*.1,torch.ones(2,1,dtype=torch.bool))
    payload={**inputs,MEMORY_INPUT_KEY:memory};mask=torch.ones(2,16,dtype=torch.bool)
    loss=tcp.flow_matching_loss(payload,proprio,torch.zeros(2,16,7),mask,generator=torch.Generator().manual_seed(3)).loss
    loss.backward()
    assert any(p.grad is not None and bool((p.grad!=0).any()) for p in tcp.memory_encoder.parameters())
    assert all(p.grad is None for p in tcp.context_encoder.parameters())
    sample=tcp.sample_actions(payload,proprio,num_steps=2,generator=torch.Generator().manual_seed(4))
    assert sample.shape==(2,16,7) and torch.isfinite(sample).all()


def test_sampling_streams_do_not_slide_between_adjacent_scenes():
    first={sampling_seed(1600100,i) for i in range(22)}
    assert not first.intersection(sampling_seed(1600101,i) for i in range(22))


@pytest.mark.parametrize('corruption',[None,'old-format','other-arm','incomplete','source'])
def test_new_checkpoint_identity(tmp_path,corruption):
    model=SimpleNamespace(expert=torch.nn.Linear(2,2),memory_encoder=torch.nn.Linear(12,2))
    ident=dict(protocol=PROTOCOL,source_manifest_sha256='a'*64,required_source_paths=list(REQUIRED_SOURCE_PATHS))
    p=dict(format=FORMAT,arm='tcp-relative',completed=True,steps=256,identity=ident,identity_sha256=identity(ident),
        expert=model.expert.state_dict(),memory_encoder=model.memory_encoder.state_dict())
    if corruption=='old-format':p['format']='rgbd-pregrasp-memory-policy/v1'
    if corruption=='other-arm':p['arm']='joint-world'
    if corruption=='incomplete':p['completed']=False
    source='b'*64 if corruption=='source' else 'a'*64
    path=tmp_path/'model.pt';torch.save(p,path)
    args=(model,path,sha256(path),identity(ident),'tcp-relative',source)
    if corruption:
        with pytest.raises(ValueError):restore(*args)
    else:restore(*args)


def test_manifest_must_cover_required_source(tmp_path):
    import json
    manifest=tmp_path/'manifest.json';unrelated=tmp_path/'unrelated.txt';unrelated.write_text('valid hash, wrong coverage')
    manifest.write_text(json.dumps({str(unrelated):sha256(unrelated)}))
    with pytest.raises(ValueError,match='必需依赖'):verify_source(manifest)
