"""控制失败必须在产生它的真实步后停止；不能继续warmup或晋升完成。"""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.front_rgbd_memory.memory import candidate_config
from experiments.memory_conditioning.conditioning import MemorySnapshot
from experiments.rgbd_memory_policy.stream import RGBDController
from robot_vla.runtime.policy_runtime import OnlineObservation


@pytest.mark.parametrize('failure',['tracking','close','terminal'])
def test_warmup_stops_immediately_after_failed_first_hold(failure):
    class Fake(RGBDController):
        def __init__(self):
            self.episode_done=False;self.closed=False;self.rows=[];self.holds=0;self.views=[]
        def set_view(self,index):self.views.append(index)
        def hold_current(self):
            self.holds+=1;self.rows.append({'tracking_valid':failure!='tracking'})
            self.closed=failure=='close';self.episode_done=failure=='terminal'
    c=Fake()
    with pytest.raises(RuntimeError,match='观察hold停止'):c.warmup(np.zeros(3))
    assert c.holds==1 and c.views==[0]


@pytest.mark.parametrize('failure',[None,'tracking','close','terminal'])
def test_last_teacher_action_failure_is_not_completed(tmp_path,monkeypatch,failure):
    import experiments.rgbd_memory_policy.collect as module
    protocol=dict(module.PROTOCOL,seeds=[1500000],train_seeds=[1500000])
    monkeypatch.setattr(module,'PROTOCOL',protocol)
    q=np.array([0.,0.,0.,-1.,0.,1.,0.],np.float32)
    env=SimpleNamespace(unwrapped=SimpleNamespace(agent=SimpleNamespace(tcp_pose=SimpleNamespace(p=torch.zeros((1,3))))),close=lambda:None)
    class Fake(RGBDController):
        def __init__(self,env,episode,output):
            self.env=env;self.episode=episode;self.episode_done=False;self.closed=False
            self.rows=[{'tracking_valid':True,'memory':{'accepted':False}}]
            self.last_target=q.copy();self.count=0;self.replay=SimpleNamespace(config=candidate_config())
            self.snapshot=MemorySnapshot(episode,.45,None,(0.,)*12,False,('uninitialized',),None,None)
        def warmup(self,position):pass
        def read_state(self):return SimpleNamespace(joint_positions=q.copy())
        def online(self):return OnlineObservation(np.zeros((4,4,3),np.uint8),np.zeros((4,4,3),np.uint8),np.r_[q,np.zeros(7),1.].astype(np.float32),'test')
        def send_action(self,action):
            self.count+=1;last=self.count==96
            self.rows.append({'tracking_valid':not(last and failure=='tracking'),'memory':{'accepted':False}})
            self.episode_done=last and failure=='terminal';self.closed=last and failure=='close'
    monkeypatch.setattr(module,'make_env',lambda:env)
    monkeypatch.setattr(module,'setup_scene',lambda *_:np.zeros(3))
    monkeypatch.setattr(module,'RGBDController',Fake)
    monkeypatch.setattr(module,'find_maniskill_panda_urdf',lambda:'unused')
    monkeypatch.setattr(module,'FrankaTCPForwardKinematics',lambda *_:lambda q:np.zeros(3))
    monkeypatch.setattr(module,'position_step',lambda *args,**kwargs:np.zeros(7))
    result=module.collect(tmp_path/'collection')
    assert result['records'][0]['steps']==96
    assert result['records'][0]['status']==('completed' if failure is None else 'stopped')
