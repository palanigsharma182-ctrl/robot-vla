"""误差分支的公平起点、梯度隔离和坐标往返检查。"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from experiments.align_relative_pose.run import RelativeProjection
from experiments.tcp_memory_control.geometry import pose_delta, apply_delta


def test_zero_initialized_projection_preserves_output_and_learns_feature():
    base=torch.nn.Linear(15,32)
    branch=RelativeProjection(base);x=torch.randn(2,15)
    branch.error=torch.ones(2,6)
    assert torch.equal(branch(x),base(x))
    branch(x).square().mean().backward()
    assert torch.isfinite(branch.relative.weight.grad).all()
    assert torch.count_nonzero(branch.relative.weight.grad)>0


def test_zero_feature_keeps_extra_parameters_inactive():
    branch=RelativeProjection(torch.nn.Linear(15,32))
    branch.error=torch.zeros(2,6)
    branch(torch.randn(2,15)).square().mean().backward()
    assert torch.count_nonzero(branch.relative.weight.grad)==0


def test_relative_pose_is_expressed_in_current_tcp_axes():
    current=np.eye(4);current[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix()
    goal=current.copy();goal[0,3]=.05
    goal[:3,:3]=Rotation.from_euler('z',100,degrees=True).as_matrix()
    delta=pose_delta(current,goal,current)
    np.testing.assert_allclose(delta[:3],[0,-.05,0],atol=1e-12)
    np.testing.assert_allclose(delta[3:],[0,0,np.deg2rad(10)],atol=1e-12)
    np.testing.assert_allclose(apply_delta(current,delta,current),goal,atol=1e-12)
