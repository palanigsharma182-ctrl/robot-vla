"""真实算法上的合成几何反例：坐标、单位、拒绝分母、时间与对称性。"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from experiments.align_precision_vision.geometry import (
    CUBE_POINTS, GeometrySpec, PoseMeasurement, measure_keypoints, align_condition,
    yaw_symmetries, corner_permutations, oracle_measurement,
)


def scene():
    intrinsic=np.array([[400.,0,160],[0,400.,120],[0,0,1]])
    transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('xyz',[.1,.2,.3]).as_matrix();transform[:3,3]=[.1,.04,.6]
    camera_points=CUBE_POINTS@transform[:3,:3].T+transform[:3,3]
    projected=camera_points@intrinsic.T;uv=projected[:,:2]/projected[:,2:]
    depth=np.zeros((240,320),np.float32)
    for point,(u,v) in zip(camera_points,uv):depth[int(np.floor(v+.5)),int(np.floor(u+.5))]=point[2]
    return intrinsic,transform,uv,depth


def measured(uv,depth,k,**kwargs):
    return measure_keypoints(uv,np.ones(8),depth,k,np.eye(4),episode='e',target='cube',timestamp_s=1.,**kwargs)


def test_rigid_pose_reconstruction_and_extrinsic():
    k,expected,uv,depth=scene();result=measured(uv,depth,k)
    assert result.reason=='valid'
    np.testing.assert_allclose(result.base_from_object,expected,atol=1e-6)
    camera=np.eye(4);camera[:3,:3]=Rotation.from_euler('z',.5).as_matrix();camera[:3,3]=[.3,-.2,.1]
    result=measure_keypoints(uv,np.ones(8),depth,k,camera,episode='e',target='cube',timestamp_s=1.)
    np.testing.assert_allclose(result.base_from_object,camera@expected,atol=1e-6)


def test_missing_or_wrong_units_cannot_become_pose():
    k,_,uv,depth=scene()
    assert measured(uv,depth*0,k).reason=='insufficient_visible_depth'
    assert measured(uv,depth*1000,k).reason=='insufficient_visible_depth'
    uv[:6]=np.nan
    assert measured(uv,depth,k).reason=='insufficient_visible_depth'


def test_inconsistent_depth_rejected():
    k,_,uv,depth=scene();u,v=uv[0];depth[int(np.floor(v+.5)),int(np.floor(u+.5))]+=.06
    assert measured(uv,depth,k).reason=='rigid_fit_rejected'


def test_collinear_model_cannot_determine_rotation():
    k,_,uv,depth=scene();model=np.c_[np.linspace(0,.04,8),np.zeros((8,2))]
    assert measured(uv,depth,k,object_points=model).reason=='degenerate_keypoints'


def test_relative_error_axes_and_symmetry():
    obj=np.eye(4);obj[:3,3]=[.05,0,0]
    m=PoseMeasurement('e','cube',1.,obj,'valid')
    tcp=np.eye(4);tcp[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix()
    c=align_condition(m,tcp,np.eye(4),episode='e',target='cube',now_s=1.,tcp_timestamp_s=1.,symmetries=yaw_symmetries())
    np.testing.assert_allclose(c.features[:6],[0,-1,0,0,0,0],atol=1e-7)
    assert c.features[6]==1 and c.symmetry_index==1
    assert len({tuple(x) for x in corner_permutations()})==4
    for p in corner_permutations():assert sorted(p.tolist())==list(range(8))


@pytest.mark.parametrize('changes,reason',[
    ({'episode':'other'},'episode_mismatch'),({'target':'other'},'target_mismatch'),
    ({'now_s':1.1},'stale_observation'),({'tcp_timestamp_s':.97},'sensor_skew'),
    ({'now_s':.99},'future_observation'),
])
def test_invalid_conditions_preserve_reason(changes,reason):
    m=PoseMeasurement('e','cube',1.,np.eye(4),'valid')
    args=dict(episode='e',target='cube',now_s=1.,tcp_timestamp_s=1.);args.update(changes)
    c=align_condition(m,np.eye(4),np.eye(4),**args)
    assert c.reason==reason and c.features[6]==0 and c.features[:6]==(0.,)*6


def test_oracle_origin_and_pose_copy():
    pose=np.eye(4);m=oracle_measurement(pose,episode='e',target='cube',timestamp_s=1.)
    pose[0,3]=1
    assert m.base_from_object[0,3]==0 and m.source=='oracle-diagnostic/v1'
    with pytest.raises(ValueError):m.base_from_object[0,3]=2
    with pytest.raises(ValueError):GeometrySpec(max_age_s=float('nan'))
