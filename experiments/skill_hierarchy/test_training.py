"""技能均衡覆盖与固定开发探针的反例。"""
from types import SimpleNamespace
import numpy as np
import pytest

from experiments.skill_hierarchy.train import balanced_schedule, probe_indices, summarize


def test_schedule_balanced_complete_and_repeatable():
    buckets=[];start=0
    for length in [2,3,4,5,6,7,8]:
        buckets.append(list(range(start,start+length)));start+=length
    schedule=balanced_schedule(buckets,8)
    assert schedule.shape==(8,7)
    assert set(schedule.ravel())==set(range(start))
    for skill,bucket in enumerate(buckets):
        assert set(schedule[:,skill])==set(bucket)
        assert set(schedule[:len(bucket),skill])==set(bucket)
    assert np.array_equal(schedule,balanced_schedule(buckets,8))


def test_duplicate_or_empty_buckets_rejected():
    with pytest.raises(ValueError):balanced_schedule([[0]]*7,2)
    with pytest.raises(ValueError):balanced_schedule([[]]+[[i] for i in range(6)],2)


def test_development_probe_keeps_every_scene_and_skill():
    d=SimpleNamespace(index=[],buckets=[[] for _ in range(7)])
    for scene in range(32):
        for skill in range(7):
            for t in range(3):
                d.buckets[skill].append(len(d.index));d.index.append((scene,skill*3+t))
    indices=probe_indices(d)
    assert len(indices)==224 and len(set(indices))==224
    assert {d.index[i][0] for i in indices}==set(range(32))
    for bucket in d.buckets:assert len(set(indices)&set(bucket))==32
    assert indices==probe_indices(d)


def test_summary_macro_weights_skills_equally():
    rows=[]
    for skill in range(7):
        for i in range(skill+1):
            rows.append(dict(seed=i,skill_id=skill,translation_error_mm=skill,
                rotation_delta_error_deg=skill,gripper_mae=.1,gripper_binary_accuracy=.9))
    assert summarize(rows)['macro']['translation_error_mm']==3


def test_continuation_matches_uninterrupted_adamw_update(monkeypatch):
    import copy
    import torch
    from experiments.skill_hierarchy.train import restore_continuation, FORMAT
    from experiments.tcp_atomic_skills.protocol import identity
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',lambda states:None)
    torch.manual_seed(7)
    policy=SimpleNamespace(expert=torch.nn.Linear(3,2),memory_encoder=torch.nn.Linear(1,1))
    optimizer=torch.optim.AdamW(policy.expert.parameters(),lr=1e-5)
    def update(model,opt):
        opt.zero_grad();model(torch.randn(2,3)).square().mean().backward();opt.step()
    update(policy.expert,optimizer)
    previous={'steps':1}
    payload=copy.deepcopy(dict(format=FORMAT,configuration_identity=identity(previous),completed=True,step=1,
        expert=policy.expert.state_dict(),memory_encoder=policy.memory_encoder.state_dict(),
        optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=[]))
    update(policy.expert,optimizer)
    restored=SimpleNamespace(expert=torch.nn.Linear(3,2),memory_encoder=torch.nn.Linear(1,1))
    next_optimizer=torch.optim.AdamW(restored.expert.parameters(),lr=.1)
    restore_continuation(restored,next_optimizer,payload,previous)
    update(restored.expert,next_optimizer)
    for a,b in zip(policy.expert.parameters(),restored.expert.parameters()):
        assert torch.equal(a,b)
    assert {int(v['step']) for v in next_optimizer.state.values()}=={2}
    payload['completed']=False
    with pytest.raises(ValueError):restore_continuation(restored,next_optimizer,payload,previous)
