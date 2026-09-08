"""针对接管监督边界、计划分母和配对采样的反例测试。"""
from types import SimpleNamespace
import copy
import json
import numpy as np
import pytest

from experiments.skill_dagger.data import teacher_windows,training_seeds
from experiments.skill_dagger.collect import pilot_gate,RecoveryTeacher,RecoveryComplete
from experiments.skill_dagger.train import schedules,batch_rows,STEPS
from experiments.skill_dagger.run import verify_parent,PARENT_SHA
from experiments.tcp_atomic_skills.protocol import identity
from experiments.skill_hierarchy.metric_train import FORMAT


class FK:
    def pose_base(self,q):
        p=np.eye(4);p[0,3]=q[0];return p


def arrays(n=10):
    target=np.zeros((n,7),np.float32);target[:,0]=np.arange(1,n+1)*.001
    previous=np.vstack([np.zeros((1,7)),target[:-1]]).astype(np.float32)
    label=np.column_stack([target-previous,np.zeros(n)]).astype(np.float32)
    a=SimpleNamespace(num_steps=n,commanded_joint_target_rad=target,previous_command_q_rad=previous,
        action=label,observation_valid=np.ones(n,bool),previous_command_valid=np.ones(n,bool),
        proprio=np.column_stack([previous,np.zeros((n,8))]),skill_id=np.zeros(n,np.int16))
    for k in ('timestamp_action','timestamp_external','timestamp_wrist','timestamp_proprio'):setattr(a,k,np.arange(n)*.05)
    return a


def test_teacher_only_anchors_and_tail_mask():
    a=arrays();x=teacher_windows(a,4,FK())
    assert x['anchor'].tolist()==[4,5,6]
    assert x['action'].shape==(3,16,7)
    assert x['action_mask'].sum(axis=1).tolist()==[6,5,4]
    assert x['action_mask'].dtype==np.bool_
    # 接管前的学生动作没有作为任何训练锚点，首动作由actual到command构造。
    assert np.all(x['anchor']>=4)
    with pytest.raises(ValueError,match='不足4'):teacher_windows(a,8,FK())
    a.previous_command_q_rad[5,0]+=.01
    with pytest.raises(ValueError):teacher_windows(a,4,FK())


def test_teacher_label_rejects_sensor_time_mismatch():
    a=arrays();a.timestamp_wrist=a.timestamp_wrist+.05
    with pytest.raises(ValueError,match='时间'):teacher_windows(a,4,FK())


def test_pilot_denominator_includes_no_candidate_and_failed():
    rows=[dict(seed=s,status='recovered') for s in range(8) for _ in range(2)]
    for i in range(4):rows[i]['status']='no_candidate'
    assert pilot_gate(rows)
    rows[4]['status']='teacher_failed';assert not pilot_gate(rows)
    assert not pilot_gate(rows[:15])
    clustered=[dict(seed=i%5,status='recovered') for i in range(16)]
    assert not pilot_gate(clustered)


def test_train_seed_selection_never_uses_development(tmp_path):
    rows=[dict(seed=20,split='val',status='completed'),dict(seed=3,split='train',status='failed'),
          dict(seed=9,split='train',status='completed'),dict(seed=5,split='train',status='completed')]
    (tmp_path/'collection.json').write_text(json.dumps(dict(records=rows)))
    assert training_seeds(tmp_path,2)==[5,9]
    with pytest.raises(ValueError):training_seeds(tmp_path,3)


def test_equal_update_counts_and_corrective_schedule_without_replacement():
    d=SimpleNamespace(buckets=[list(range(i*10,(i+1)*10)) for i in range(7)])
    a,c=schedules(d,64);b,e=schedules(d,64)
    assert np.array_equal(a,b) and np.array_equal(c,e)
    assert set(c.reshape(-1)[:64])==set(range(64))
    count=dict(BC=0,DAgger=0)
    for step in range(STEPS):
        bc=batch_rows(step,'BC',a,c);da=batch_rows(step,'DAgger',a,c)
        assert len(bc)==len(da)==7 and all(s=='train' for s,_ in bc)
        assert sum(s=='corrective' for s,_ in da)==2
        for slot,(s,i) in enumerate(da):
            if s=='train':assert bc[slot]==(s,i)
        count['BC']+=len(bc);count['DAgger']+=len(da)
    assert count==dict(BC=3584,DAgger=3584)


def test_parent_must_be_frozen_a_final():
    c={'source':'source'}
    p=dict(format=FORMAT,configuration_identity=identity(c),completed=True,branch='A',step=1024)
    verify_parent(p,c,PARENT_SHA)
    for change in (dict(branch='B'),dict(step=512),dict(completed=False)):
        with pytest.raises(ValueError):verify_parent(dict(p,**change),c,PARENT_SHA)
    with pytest.raises(ValueError):verify_parent(p,c,'wrong')


def test_recovery_replans_from_current_state_and_preserves_gripper_when_held():
    class Fake:
        def to_transformation_matrix(self):return np.eye(4)
    t=object.__new__(RecoveryTeacher);t.session=object();t.relative_history=[];calls=[]
    t._read_predicate_state=lambda:SimpleNamespace(is_grasped=False)
    t._hold=lambda session,**kw:calls.append(('hold',kw['gripper_opening']))
    t._phase_poses=lambda:(calls.append(('replan',None)) or (Fake(),Fake(),Fake(),Fake(),Fake()))
    t._move_to_pose=lambda session,pose,**kw:calls.append(('move',kw['gripper_opening']))
    assert t.recover() is False
    assert calls==[('hold',1.),('replan',None),('move',1.),('move',1.),('hold',0.)]
    calls.clear();t._read_predicate_state=lambda:SimpleNamespace(is_grasped=True)
    assert t.recover() is False
    assert calls==[('hold',0.)]
