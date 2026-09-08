"""整任务 checkpoint 与失败分母的检查。"""
from types import SimpleNamespace
import torch
import pytest

from experiments.skill_hierarchy.full_task import load_student,summarize
from experiments.skill_hierarchy.train import FORMAT
from experiments.tcp_atomic_skills.protocol import identity,sha


def test_seven_skill_checkpoint_load_rejects_incomplete_and_wrong_format(tmp_path):
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.expert=torch.nn.Linear(2,1);self.memory_encoder=torch.nn.Linear(1,1)
    p=Policy();config=dict(smoke=False,steps=5666)
    payload=dict(format=FORMAT,configuration_identity=identity(config),completed=True,step=5666,
                 expert=p.expert.state_dict(),memory_encoder=p.memory_encoder.state_dict())
    path=tmp_path/'latest.pt';torch.save(payload,path)
    result=dict(status='completed',strict_reload=True,configuration_identity=identity(config),
                steps=5666,checkpoint_sha256=sha(path))
    q=Policy();load_student(q,path,result,config)
    assert not q.training
    for a,b in zip(p.parameters(),q.parameters()):assert torch.equal(a,b)
    payload['format']='old-five-skills';torch.save(payload,path);result['checkpoint_sha256']=sha(path)
    with pytest.raises(ValueError):load_student(q,path,result,config)
    result['strict_reload']=False
    with pytest.raises(ValueError):load_student(q,path,result,config)


def test_failed_and_unrun_episodes_stay_in_denominator():
    r=summarize([dict(success=True,canonical_success=True,seven_completed=7,policy_steps=100),
                 dict(success=False,seven_completed=2,policy_steps=60),
                 dict(success=False,status='not_run'),dict(success=False,status='failed')])
    assert r['episodes']==4 and r['successes']==1 and r['total_policy_steps']==160
    assert r['failures_by_boundary']['2']==1
