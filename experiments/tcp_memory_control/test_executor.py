"""IK拒绝不发送动作；新规划从actual参考开始；无需运行物理仿真。"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from experiments.tcp_memory_control.kinematics import TCPChunkExecutor
from robot_vla.execution.chunk_executor import FrankaControlState


class Kinematics:
    def pose_base(self,q):
        p=np.eye(4);p[:3,3]=q[:3];return p
    def inverse(self,target,reference):
        q=reference.copy();q[:3]=target[:3,3];return q


class Controller:
    def __init__(self):self.q=np.array([0,0,0,-1.5,0,1.5,0],np.float32);self.sent=0
    def read_state(self):return FrankaControlState(self.q.copy(),1.)
    def send_action(self,a):self.q+=a[:7]*.1;self.sent+=1
    def hold_current(self):pass


def test_reference_restarts_from_actual_and_executes_only_four_steps():
    fk=Kinematics();executor=TCPChunkExecutor(fk);controller=Controller()
    a=np.zeros((16,7),np.float32);a[:,6]=1.;a[:4,0]=.001
    anchor=np.eye(4)
    first=executor.execute(a,controller,anchor)
    assert first.executed_steps==4 and controller.sent==4
    assert controller.q[0]==pytest.approx(.004)
    controller.q[0]=.020  # 实际位置与上次command不同，不能暗用上次参考。
    with pytest.raises(ValueError,match='anchor'):executor.execute(a,controller,anchor)
    executor.execute(a,controller,fk.pose_base(controller.q))
    assert controller.q[0]==pytest.approx(.024)


def test_failed_ik_later_in_prefix_sends_nothing():
    fk=Kinematics();original=fk.inverse;calls=0
    def inverse(*args):
        nonlocal calls
        calls+=1
        if calls==3:raise ValueError('synthetic IK rejection')
        return original(*args)
    fk.inverse=inverse;executor=TCPChunkExecutor(fk);controller=Controller()
    a=np.zeros((16,7),np.float32);a[:,6]=1.;a[:4,0]=.001
    with pytest.raises(ValueError):executor.execute(a,controller,np.eye(4))
    assert controller.sent==0
