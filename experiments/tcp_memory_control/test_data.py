"""联合两arm的首步标签使用actual参考，旧记录中的command参考不能漏入。"""
import numpy as np
from experiments.tcp_memory_control import data


def test_replan_first_label_uses_actual_for_both_arms(tmp_path,monkeypatch):
    folder=tmp_path/'1';folder.mkdir()
    actual=np.zeros((16,15),np.float32);targets=np.zeros((16,7),np.float32)
    targets[:,0]=.005+np.arange(16)*.001
    previous=targets.copy();previous[:,0]-=.001
    np.savez(folder/'sequence.npz',physical_proprio=actual,commanded_joint_target_rad=targets,previous_command_q_rad=previous)
    snapshot=dict(available=False,features=[0.]*12)
    old_labels=np.zeros((16,8),np.float32);old_labels[:,0]=.02;old_labels[:,7]=1.
    example=dict(seed=1,anchor=0,action=old_labels,snapshot=snapshot)
    monkeypatch.setattr(data,'load_examples',lambda _:({'train':[example],'development':[]},{},[dict(seed=1,status='completed')]))
    class FK:
        def pose_base(self,q):
            p=np.eye(4);p[:3,3]=q[:3];return p
    result,_,_=data.prepare_examples(tmp_path,FK());x=result['train'][0]
    np.testing.assert_allclose(x['action'][0,0],.1,atol=1e-6)
    np.testing.assert_allclose(x['action'][1:,0],.02,atol=1e-6)
    np.testing.assert_allclose(x['tcp_action'][0,0],.5,atol=1e-6)
    np.testing.assert_allclose(x['tcp_action'][1:,0],.1,atol=1e-6)
    assert not x['tcp_features'].any()
    assert old_labels[0,0]==np.float32(.02)
