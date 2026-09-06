"""解析反例验证frame、旋转组合与mask；不依赖GPU或机器人仿真。"""
from copy import deepcopy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from experiments.tcp_memory_control.geometry import relative_features,pose_delta,apply_delta,TCPActionSpec


def transform(xyz=(0,0,0),rot=(0,0,0)):
    t=np.eye(4);t[:3,3]=xyz;t[:3,:3]=Rotation.from_rotvec(rot).as_matrix();return t


def test_world_to_tcp_rotates_position_and_covariance_without_mutating_memory():
    snap=dict(available=True,features=[1.,3.,3.,1.,0.,0.,2.,0.,3.,.2,.5,1.])
    original=deepcopy(snap);tcp=transform((1,2,3),(0,0,np.pi/2))
    value=relative_features(snap,tcp,offset_base=(0,0,.08))
    np.testing.assert_allclose(value[:3],[1,0,.08],atol=1e-7)
    np.testing.assert_allclose(value[3:9],[2,0,0,1,0,3],atol=1e-7)
    np.testing.assert_allclose(value[9:],snap['features'][9:]);assert snap==original


def test_invalid_memory_stays_zero_even_when_tcp_is_not_origin():
    assert not relative_features(dict(available=False,features=[0]*12),transform((1,2,3))).any()


def test_fixed_anchor_translation_and_rotation_have_exact_inverse():
    anchor=transform((.1,.2,.3),(.2,-.3,.6));previous=transform((.4,.5,.6),(-.1,.2,-.3))
    # 非交换旋转及非单位anchor防止把body/right-multiply误写成spatial/left-multiply。
    for delta in ([.001,-.002,.003,.03,-.01,.02],[-.001,.003,.001,-.02,.04,.01]):
        target=apply_delta(previous,delta,anchor)
        np.testing.assert_allclose(pose_delta(previous,target,anchor),delta,atol=1e-12)
        r=anchor[:3,:3]
        np.testing.assert_allclose(target[:3,3],previous[:3,3]+r@np.array(delta[:3]),atol=1e-12)
        previous=target


def test_tcp_axes_remain_anchor_axes_after_rotation():
    anchor=np.eye(4);turned=apply_delta(anchor,[0,0,0,0,0,np.pi/2],anchor)
    next_pose=apply_delta(turned,[.01,0,0,0,0,0],anchor)
    np.testing.assert_allclose(next_pose[:3,3],[.01,0,0],atol=1e-12)


@pytest.mark.parametrize('bad',[np.zeros((4,4)),np.full((4,4),np.nan),np.diag([2,1,1,1])])
def test_bad_pose_rejected(bad):
    with pytest.raises(ValueError):relative_features(dict(available=False),bad)


def test_action_units_gripper_and_range():
    spec=TCPActionSpec();physical=np.array([[.001,-.002,.003,.01,-.02,.03,.6]])
    np.testing.assert_allclose(spec.denormalize(spec.normalize(physical)),physical,atol=1e-7)
    with pytest.raises(ValueError):spec.normalize(np.array([[.02,0,0,0,0,0,1]]))
    with pytest.raises(ValueError):spec.denormalize(np.zeros((16,8)))
    with pytest.raises(ValueError):spec.normalize(np.array([[0,0,0,0,0,0,2]]))
