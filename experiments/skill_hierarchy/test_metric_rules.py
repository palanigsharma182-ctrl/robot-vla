"""验证分布约束、事件反例、反馈阈值与恢复的关键不变量。"""
import copy
import numpy as np
import pytest
from experiments.skill_hierarchy.metric_rules import CommandEvents,event_groups,FeedbackSampler,focus_distribution


def sampler():
    groups=np.repeat(np.arange(28),32)
    scenes=np.tile(np.repeat(np.arange(8),4),28)
    schedule=np.array([[s*128+(t%128) for s in range(7)] for t in range(1024)])
    return FeedbackSampler(groups,scenes,schedule)


def feedback(error=(2.,.3,.1),failure=True):
    return dict(valid=True,m7={str(g):dict(scenes=list(range(8)),median=list(error)) for g in range(28)},
        failures={k:[0,1] if failure else [] for k in ('approach','align','gripper')},
        tasks={s:dict(successes=0,stable=0) for s in ('F','S')},sparse={k:[0,1] for k in ('approach','align','gripper')})


def test_debounce_and_initial_close():
    c=CommandEvents()
    for t,u in enumerate([.49,.51]*10): assert c.observe(u,t) is None
    assert c.state=='UNKNOWN' and c.chatter()
    for t,u in enumerate([.2,.2,.5,.2,.2,.2],20): e=c.observe(u,t)
    assert e['state']=='CLOSED' and e['start']==23 and e['confirmed']==25
    assert c.observe(.1,26) is None
    for t in range(27,30): e=c.observe(.8,t)
    assert e['state']=='OPEN'


def test_groups_priority_and_tail():
    ids=np.array([0]*5+[1]*5+[2]*6)
    u=[1.]*8+[0.]*8
    g=event_groups(ids,u)
    assert len(g)==len(ids) and np.array_equal(g//4,ids)
    assert np.all(g%4==0)  # switch start=8，左右各8覆盖此短episode
    assert np.array_equal(event_groups([0]*4,[1]*4),[1]*4)


def test_warmup_and_resume_exact():
    a,b=sampler(),sampler()
    for t in range(256):
        x,y=a.draw(t,False),b.draw(t,True)
        assert x==y
        a.account(x);b.account(y)
    b.history=[feedback()]; b.update(feedback(),256)
    c=sampler(); c.restore(b.state())
    for t in range(256,280):
        x,y=b.draw(t,True),c.draw(t,True)
        assert x==y
        b.account(x);c.account(y)
    assert b.state()==c.state()


def test_small_absolute_error_does_not_trigger():
    s=sampler(); f=feedback((.09,.009,.001));s.history=[f];s.update(f,256)
    assert not s.focus.any()
    s.history=[feedback((4.,.3,.1))];s.update(feedback((2.,.3,.1)),512)
    assert s.focus.any()  # 改善50%但剩余误差仍重要


def test_single_feedback_scene_and_invalid_do_not_trigger():
    s=sampler(); f=feedback()
    f['failures']={k:[0] for k in f['failures']}
    s.history=[f];s.update(f,256)
    assert not s.focus.any()
    f['valid']=False
    with pytest.raises(ValueError):s.update(f,512)


def test_probability_caps_over_many_changes():
    s=sampler();w=np.zeros(28);old=s.base
    rng=np.random.default_rng(5)
    for _ in range(200):
        grades=rng.integers(0,3,28)
        w,p=focus_distribution(s.base,w,grades)
        assert np.isclose(p.sum(),1) and np.all(p>=.5*s.base-1e-12)
        assert max(p.reshape(7,4).sum(axis=1))<=.3+1e-9
        assert np.max(abs(p-old))<=.05+1e-9
        assert max(w)<=.1+1e-9 and w.sum()<=.5+1e-9
        old=p


def test_exhaustion_requires_actual_exposure():
    s=sampler(); f=feedback();s.history=[f];s.update(f,256)
    notes=s.update(f,768)
    assert any(n['reason']=='exposure_insufficient' for n in notes)
    assert not s.retired
    s.exposure[:]=4;s.extra[:]=2
    notes=s.update(f,1024)
    assert any(n['reason'].startswith('resampling_exhausted') for n in notes)


def test_corrective_requires_low_error_and_sparse_support():
    f=feedback((.1,.01,.001));s=sampler();s.exposure[:]=4;s.history=[f]
    f2=copy.deepcopy(f);f2['sparse']={k:[] for k in f2['sparse']}
    s.update(f2,512);assert not s.retired
    s.history=[f];s.update(f,768)
    assert s.retired and all(n['reason']=='corrective_data_candidate' for n in s.decisions[-1]['notes'])


def test_empty_related_bucket_does_not_block_corrective():
    s=sampler()
    # 删除 Align-switch，真实数据中该桶不存在；Grasp-switch 仍有充分覆盖。
    keep=s.groups!=4
    groups=s.groups[keep];scenes=s.scenes[keep]
    buckets=[np.flatnonzero(groups//4==i) for i in range(7)]
    baseline=np.array([[b[t%len(b)] for b in buckets] for t in range(1024)])
    s=FeedbackSampler(groups,scenes,baseline);s.exposure[:]=4
    f=feedback((.1,.01,.001));f['m7'].pop('4');s.history=[f]
    s.update(f,512)
    assert 8 in s.retired and 4 not in s.retired


def test_reuse_identity_json_roundtrip_and_real_change():
    import json
    from experiments.skill_hierarchy.metric_train import verify_reuse_identity
    current=dict(parent_sha256='fixed',data={'x':'sha'},upstream={'stats':(1.,2.)},feedback=(1,2),
                 reserved=(3,4),deadline_unix=12.,training_sdpa='math',deterministic_algorithms=True)
    stored=json.loads(json.dumps(current))
    verify_reuse_identity(stored,current)
    stored['upstream']['stats'][0]=9.
    with pytest.raises(ValueError,match='upstream'):verify_reuse_identity(stored,current)
