"""教师动作与现有训练标签逐值相同，未执行尾部padding不改变实际前缀。"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from experiments.teacher_tcp_replay.runner import TeacherActions, action_digest
from experiments.tcp_memory_control import data as training_data
from experiments.teacher_tcp_replay import runner


class FK:
    def pose_base(self,q):
        result=np.eye(4);result[:3,3]=q[:3]
        result[:3,:3]=Rotation.from_rotvec(q[3:6]).as_matrix()
        return result


def sequence():
    previous=np.arange(96,dtype=np.float32)[:,None]*np.array([[.001,.0002,0.,.0001,.0002,0.,0.]],np.float32)
    target=previous+np.array([[.001,.0002,0.,.0001,.0002,0.,0.]],np.float32)
    actual=np.zeros((96,15),np.float32);actual[:,:7]=previous-.0001;actual[:,-1]=1.
    return {'physical_proprio':actual,'commanded_joint_target_rad':target,'previous_command_q_rad':previous,'timestamp_s':.45+np.arange(96)*.05}


def test_exact_training_label_parity(tmp_path,monkeypatch):
    d=sequence();folder=tmp_path/'1';folder.mkdir();np.savez(folder/'sequence.npz',**d)
    rows=[{'seed':1,'anchor':i,'action':np.zeros((16,8),np.float32),'snapshot':{'available':False,'features':np.zeros(12).tolist()}} for i in [0,8,80]]
    monkeypatch.setattr(training_data,'load_examples',lambda _:({'train':rows,'development':[]},{},[{'seed':1,'status':'completed'}]))
    original,_,_=training_data.prepare_examples(tmp_path,FK());teacher=TeacherActions(d,FK())
    for x in original['train']:
        normalized,real=teacher.normalized_chunk(x['anchor'])
        assert real==16 and np.array_equal(normalized,x['tcp_action'])


def test_actions_frozen_and_tail_only_unexecuted():
    teacher=TeacherActions(sequence(),FK());chunks=teacher.frozen_chunks()
    physical,real,digest=chunks[84]
    assert real==12 and action_digest(physical)==digest
    assert np.all(physical[12:,:6]==0) and np.all(physical[12:,6]==1)
    assert np.any(physical[:4,:6]!=0)
    with pytest.raises(ValueError):physical[0,0]=1
    assert action_digest(physical)==digest


def test_invalid_source_index_rejected():
    teacher=TeacherActions(sequence(),FK())
    for index in [-1,88,96]:
        with pytest.raises(ValueError):teacher.normalized_chunk(index)


def test_execution_source_cannot_be_omitted():
    with pytest.raises(ValueError,match='0.1执行模块'):
        runner.require_replay_sources({'experiments/teacher_tcp_replay/runner.py':'hash'})


@pytest.mark.parametrize('fault',['identity','split','historical-metric'])
def test_teacher_identity_and_denominator_are_bound(monkeypatch,fault):
    import copy
    records=[{'seed':seed,'status':'completed','split':'development' if seed%3==2 else 'train',
              'final_teacher_distance_m':.001} for seed in runner.PROTOCOL['seeds']]
    ident={'denominator':copy.deepcopy(records)}
    monkeypatch.setattr(runner,'APPROVED_TRAINING_IDENTITY',runner.semantic_identity(ident))
    runner.validate_training_identity(ident,records)
    if fault=='identity':ident['unapproved']=True
    elif fault=='split':records[0]['split']='development'
    else:records[0]['final_teacher_distance_m']=0.
    with pytest.raises(ValueError):runner.validate_training_identity(ident,records)
