"""验证失败保留、完整分母及不可估计的条件成功率。"""
import pytest
from experiments.tcp_atomic_skills.protocol import CASES,EVAL_SEEDS
from experiments.tcp_atomic_skills.summarize import summarize


def example():
    rows=[dict(seed=s,case=c[0],arm=a,status='completed',success=False,
               stop_reason='step-budget-exhausted',policy_steps=c[3],final_completed=c[1])
          for s in EVAL_SEEDS for c in CASES for a in ('before','after')]
    return dict(status='completed',elapsed_s=1.,records=rows)


def test_all_failures_remain_in_denominator():
    data=example();data['records'][0].update(status='preparation_failed')
    groups=summarize(data)['groups']
    assert groups[0]['planned']==4 and groups[0]['executed']==3
    assert groups[0]['preparation_failed']==1
    assert all(g['success_given_first_stage'] is None for g in groups)


def test_missing_or_duplicate_case_rejected():
    data=example();data['records'][-1]=data['records'][0]
    with pytest.raises(ValueError,match='分母'):summarize(data)


def test_incomplete_is_not_completed():
    data=example();data['records'][0]['status']='not_run'
    with pytest.raises(ValueError,match='未完成'):summarize(data)
