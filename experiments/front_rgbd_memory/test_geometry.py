"""合成几何及Memory不变量；不加载仿真，不使用历史评估数据。"""
import numpy as np
import pytest

from experiments.front_rgbd_memory.geometry import Estimate, backproject, estimate_center, measure, target_mask
from experiments.front_rgbd_memory.memory import MemoryReplay, fixture_safety, tracking_safety


def faces(center,angles=(.2,.3,.4),count=20):
    from scipy.spatial.transform import Rotation
    rotation=Rotation.from_euler('xyz',angles).as_matrix()
    origin=np.array([.7,.6,.8])
    grid=np.linspace(-.018,.018,count)
    a,b=np.meshgrid(grid,grid)
    points=[]
    for axis in range(3):
        other=[i for i in range(3) if i!=axis]
        local=np.zeros((count*count,3))
        sign=np.sign(rotation[:,axis]@(origin-center))
        local[:,axis]=.02*sign
        local[:,other[0]]=a.ravel();local[:,other[1]]=b.ravel()
        points.append(local@rotation.T+center)
    return points,origin


@pytest.mark.parametrize('z',[.05,.15,.30])
def test_arbitrary_height_and_rotation_center(z):
    center=np.array([.10,-.08,z])
    clouds,origin=faces(center)
    points=np.concatenate(clouds)+np.random.default_rng(3).normal(0,.00015,(1200,3))
    result=estimate_center(points,origin)
    assert result.valid,result.reason
    assert np.linalg.norm(result.position-center)<.0005
    assert np.all(np.linalg.eigvalsh(result.covariance)>0)


def test_one_face_cannot_pretend_to_be_object_center():
    clouds,origin=faces(np.array([.1,0,.2]))
    result=estimate_center(clouds[0],origin)
    assert not result.valid


def test_known_shape_rejects_wrong_size():
    center=np.array([.1,0,.2]);clouds,origin=faces(center)
    points=(np.concatenate(clouds)-center)*2+center
    assert not estimate_center(points,origin).valid


def test_depth_units_and_frame_conversion():
    depth=np.array([[500,0],[1000,1000]],dtype=np.int16)
    intrinsic=np.array([[100.,0,0],[0,100.,0],[0,0,1.]])
    transform=np.eye(4);transform[:3,3]=[.1,.2,.3]
    points=backproject(depth,np.ones((2,2),dtype=bool),intrinsic,transform)
    np.testing.assert_allclose(points,[[.1,.2,.8],[.1,.21,1.3],[.11,.21,1.3]])
    transform[0,0]=2
    with pytest.raises(ValueError,match='刚体'):
        backproject(depth,np.ones((2,2),bool),intrinsic,transform)


def test_red_mask_excludes_wood_and_rejects_two_objects():
    rgb=np.full((60,60,3),[180,95,45],dtype=np.uint8)
    assert not target_mask(rgb)[0].any()
    rgb[5:25,5:25]=[200,20,20]
    mask,reason=target_mask(rgb)
    assert reason=='ok' and mask[10,10] and not mask[40,40]
    rgb[35:55,35:55]=[200,20,20]
    assert target_mask(rgb)[1]=='target_ambiguous'


def test_depth_dropout_has_no_gt_fallback():
    rgb=np.full((40,40,3),[200,20,20],dtype=np.uint8)
    result,_=measure(rgb,np.zeros((40,40),np.int16),np.eye(3),np.eye(4))
    assert not result.valid and result.position is None


def test_memory_three_real_timestamps_hold_expiry_and_episode_reset():
    replay=MemoryReplay('synthetic-a');safety=fixture_safety(True,True)
    estimate=Estimate(np.array([.4,.1,.2]),np.eye(3)*4e-6,'accepted_candidate')
    results=[replay.update(estimate,timestamp=t,safety=safety) for t in (.05,.10,.15)]
    assert [r['accepted'] for r in results]==[False,False,True]
    held=replay.update(Estimate(reason='missing'),timestamp=.2,safety=safety)
    assert held['available'] and held['last_observed_timestamp_s']==.15
    stale=replay.update(Estimate(reason='missing'),timestamp=2.70,safety=safety)
    assert not stale['available'] and not any(stale['snapshot_features'])
    new=MemoryReplay('synthetic-b')
    assert not new.update(Estimate(reason='missing'),timestamp=.05,safety=safety)['available']


def test_new_camera_view_cannot_reuse_previous_two_candidate_frames():
    replay=MemoryReplay('views');safe=fixture_safety(True,True)
    estimate=Estimate(np.array([.4,.1,.2]),np.eye(3)*4e-6,'accepted_candidate')
    for t in (.05,.1,.15):replay.update(estimate,timestamp=t,safety=safe)
    replay.begin_view()
    rows=[replay.update(estimate,timestamp=t,safety=safe) for t in (.20,.25,.30)]
    assert [r['accepted'] for r in rows]==[False,False,True]
    assert rows[0]['available'] and rows[0]['last_observed_timestamp_s']==.15


def test_transient_tracking_drift_is_detected_on_that_tick():
    q=np.array([0.]*7+[.04,.04]);tcp=np.zeros(3)
    assert not tracking_safety(q,tcp,q[:7],tcp)[0].invalidation_reasons
    disturbed=q.copy();disturbed[0]=.04
    assert 'controller_tracking_invalid' in tracking_safety(disturbed,tcp,q[:7],tcp)[0].invalidation_reasons
    closed=q.copy();closed[-2:]=0
    assert 'gripper_not_open' in tracking_safety(closed,tcp,q[:7],tcp)[0].invalidation_reasons
