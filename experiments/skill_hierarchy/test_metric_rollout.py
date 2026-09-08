"""在合成几何上核验夹爪事件和机会分母，不运行物理仿真。"""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from experiments.skill_hierarchy.metric_rollout import Observer,support_count,feedback_records


class Pose:
    def __init__(self,batch=False):self.matrix=np.eye(4);self.batch=batch
    def to_transformation_matrix(self):return self.matrix[None] if self.batch else self.matrix


def observer():
    tcp=Pose(True); obj=Pose(True)
    metrics=dict(held=0,opening=1.,tcp_speed_m_s=0.,pregrasp_distance_m=0.,orientation_error_deg=0.)
    t=SimpleNamespace(base_env=SimpleNamespace(agent=SimpleNamespace(tcp_pose=tcp),cube=SimpleNamespace(pose=obj)),
        poses=[Pose()],pregrasp_world=np.eye(4),metrics=metrics,rows=[],measure=lambda **kw:dict(metrics))
    o=Observer(t);o.read(0)
    return o,t


def test_missing_close_requires_complete_opportunity():
    o,t=observer()
    for tick in range(1,24):o.read(tick,1.)
    assert [x['status'] for x in o.result()['opportunities']]==['missing_close']
    for tick in range(24,40):o.read(tick,1.)
    assert len(o.result()['opportunities'])==1


def test_far_from_grasp_is_not_missing_close():
    o,t=observer();t.base_env.agent.tcp_pose.matrix[0,3]=.1
    for tick in range(1,30):o.read(tick,1.)
    assert not o.result()['opportunities'] and o.result()['not_reached']


def test_timely_close_in_denominator_and_censored_failure():
    o,t=observer()
    for tick in range(1,9):o.read(tick,1. if tick<5 else 0.)
    r=o.result()
    assert r['opportunities'][0]['status']=='closed_or_grasped'
    assert r['close_events'][0]['start']==5 and r['close_events'][0]['confirmed']==7
    assert r['close_events'][0]['status']=='censored'
    for tick in range(9,46):o.read(tick,0.)
    assert o.result()['close_events'][0]['status']=='failed_close'


def test_opportunity_loss_not_missing_and_missing_velocity_unknown():
    o,t=observer()
    for tick in range(1,5):o.read(tick,1.)
    t.base_env.agent.tcp_pose.matrix[0,3]=.1;o.read(5,1.)
    assert o.result()['opportunities'][0]['status']=='lost_opportunity'
    assert support_count(o.rows[0],[]) is None


@pytest.mark.parametrize('moving',[False,True])
def test_held_geometry_always_serializes_native_bool(moving):
    o,t=observer(); t.metrics['held']=1
    for tick in range(1,8):
        t.base_env.cube.pose.matrix[0,3]=tick*.01 if moving else 0.
        o.read(tick,0.)
        assert type(o.ever_stable) is bool
        json.dumps(o.result(),allow_nan=False)
    assert o.ever_stable is (not moving)
    t.metrics['held']=0; o.read(8,0.)
    assert o.ever_stable is (not moving)
    json.dumps(o.result(),allow_nan=False)


def test_resume_preserves_completed_and_archives_interrupted(tmp_path):
    output=tmp_path/'feedback'
    records=feedback_records(output,[10])
    records[0].update(status='completed',ever_stable=False,execution_error=False,failures=[],support={})
    records[1]['status']='preparing'
    (output/'10-F').mkdir(); (output/'10-S').mkdir()
    (output/'10-S/plans.json').write_text('[]')
    (output/'result.json').write_text(json.dumps(dict(records=records)))
    loaded=feedback_records(output,[10],resume=True)
    assert loaded[0]==records[0] and (output/'10-F').exists()
    assert loaded[1]['status']=='not_run'
    assert (output/'10-S-interrupted/plans.json').exists()
    assert not (output/'10-S').exists()
    with pytest.raises(ValueError,match='场景身份'):
        feedback_records(output,[11],resume=True)


def test_resume_rejects_incomplete_completed_record(tmp_path):
    output=tmp_path/'feedback'; records=feedback_records(output,[10])
    records[0]['status']='completed'
    (output/'result.json').write_text(json.dumps(dict(records=records)))
    with pytest.raises(ValueError,match='记录不完整'):
        feedback_records(output,[10],resume=True)


def test_resume_freezes_config_and_checkpoint_identity():
    from experiments.skill_hierarchy.metric_train import verify_resume_config,verify_resume_checkpoint,FORMAT
    from experiments.tcp_atomic_skills.protocol import identity
    previous=dict(source='old',deadline_unix=123,steps_per_branch=1024)
    verify_resume_config(previous,dict(previous,source='new'))
    with pytest.raises(ValueError): verify_resume_config(previous,dict(previous,deadline_unix=124))
    payload=dict(format=FORMAT,configuration_identity=identity(previous),branch='A',step=512,completed=False)
    verify_resume_checkpoint(payload,previous)
    for change in (dict(branch='B'),dict(step=256),dict(completed=True),dict(configuration_identity='wrong')):
        with pytest.raises(ValueError): verify_resume_checkpoint(dict(payload,**change),previous)


def test_explicit_completion_mode_only_changes_wall_clock():
    from experiments.skill_hierarchy.metric_train import verify_resume_config,verify_resume_checkpoint,FORMAT
    from experiments.tcp_atomic_skills.protocol import identity
    previous=dict(source='old',deadline_unix=123,steps_per_branch=1024)
    current=dict(previous,source='new',deadline_unix=None)
    with pytest.raises(ValueError): verify_resume_config(previous,current)
    verify_resume_config(previous,current,complete_planned_run=True)
    with pytest.raises(ValueError):
        verify_resume_config(previous,dict(current,steps_per_branch=2048),complete_planned_run=True)
    with pytest.raises(ValueError):
        verify_resume_config(previous,dict(current,deadline_unix=456),complete_planned_run=True)
    verify_resume_checkpoint(dict(format=FORMAT,configuration_identity=identity(previous),branch='A',
        step=1024,completed=False),previous)


def test_completion_mode_preserves_explicit_stop_and_step_limits(tmp_path):
    from experiments.skill_hierarchy.metric_rollout import RunBudget
    timed=RunBudget(0,tmp_path)
    with pytest.raises(InterruptedError): timed.check()
    b=RunBudget(None,tmp_path); b.check(); b.consume('student'); b.save()
    assert json.loads((tmp_path/'budget.json').read_text())['deadline_unix'] is None
    b.stopped=True
    with pytest.raises(InterruptedError): b.check()
    b.stopped=False; b.student=115200
    with pytest.raises(InterruptedError): b.consume('student')
