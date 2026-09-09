"""PnP/对称/轮廓深度隔离与三组分母的反例验证。"""
import json
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from experiments.align_precision_vision.geometry import CUBE_POINTS,corner_permutations,yaw_symmetries,measure_keypoints
from experiments.align_precision_vision.pose_recovery import recover_pose,project_cube,cube_ray_depth,interior_mask,PnPSpec
from experiments.align_precision_vision.compare_pose import compare
from experiments.align_precision_vision.test_geometry import scene
from experiments.align_precision_vision.test_labels import dataset
from experiments.align_precision_vision.evaluate import evaluate


def recover(uv,k,*,visibility=None,**kwargs):
    return recover_pose(uv,np.ones(8) if visibility is None else visibility,k,np.eye(4),
        image_shape=(240,320),episode='e',target='cube',timestamp_s=0.,**kwargs)


def test_pnp_ignores_depth_and_recovers_extrinsic():
    class ForbiddenDepth:
        def __array__(self,*args,**kwargs):raise AssertionError('PnP-only不应访问depth')
    k,truth,uv,_=scene();camera=np.eye(4);camera[:3,3]=[.1,-.2,.05]
    result=recover_pose(uv,np.ones(8),k,camera,image_shape=(240,320),episode='e',target='cube',timestamp_s=0.,depth_m=ForbiddenDepth())
    assert result.reason=='valid'
    np.testing.assert_allclose(result.base_from_object,camera@truth,atol=1e-6)


def test_visibility_minimum_and_planar_four_points():
    k,truth,uv,_=scene();vis=(CUBE_POINTS[:,2]<0).astype(float)
    result=recover(uv,k,visibility=vis)
    assert result.reason=='valid' and result.diagnostics['planar']
    np.testing.assert_allclose(result.base_from_object,truth,atol=1e-5)
    vis[np.flatnonzero(vis)[0]]=0
    assert recover(uv,k,visibility=vis).reason=='insufficient_visible_pnp'


def test_ransac_handles_one_outlier_and_symmetry_is_whole_object():
    k,truth,uv,_=scene();bad=uv.copy();bad[0]+=[20,-15]
    result=recover(bad,k)
    assert result.reason=='valid' and 0 not in result.diagnostics['pnp_inliers']
    np.testing.assert_allclose(result.base_from_object,truth,atol=1e-5)
    for permutation in corner_permutations():
        result=recover(uv[permutation],k)
        assert result.reason=='valid'
        angle=min(Rotation.from_matrix(result.base_from_object[:3,:3].T@(truth@s)[:3,:3]).magnitude() for s in yaw_symmetries())
        assert angle<1e-5


def rendered():
    k=np.array([[400.,0,160],[0,400.,120],[0,0,1]])
    truth=np.eye(4);truth[:3,:3]=Rotation.from_euler('xyz',[.2,-.15,.3]).as_matrix();truth[2,3]=.3
    yy,xx=np.indices((240,320));rays=np.c_[xx.ravel(),yy.ravel(),np.ones(xx.size)]@np.linalg.inv(k).T
    depth=cube_ray_depth(truth,rays).reshape(240,320);mask=np.isfinite(depth)
    rgb=np.zeros((240,320,3),np.uint8);rgb[mask]=[200,10,10]
    depth=np.where(mask,depth,.8).astype(np.float32)
    return k,truth,rgb,depth,mask


def test_refinement_never_uses_foreground_boundary_depth():
    k,truth,rgb,depth,mask=rendered();uv,_=project_cube(truth,k)
    interior=interior_mask(rgb,truth,k,PnPSpec())
    assert interior.sum()>32 and not np.any(interior&~mask)
    edge=mask&~cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    assert not (edge&interior).any()
    contaminated=depth.copy();contaminated[edge]=1.7
    a=recover(uv,k,method='pnp-depth-refine',rgb=rgb,depth_m=depth)
    b=recover(uv,k,method='pnp-depth-refine',rgb=rgb,depth_m=contaminated)
    assert a.reason==b.reason=='valid'
    np.testing.assert_allclose(a.base_from_object,b.base_from_object,atol=1e-10)


def test_interior_depth_refines_small_range_bias():
    k,truth,rgb,depth,_=rendered();biased=truth.copy();biased[2,3]+=.002
    uv,_=project_cube(biased,k)
    baseline=recover(uv,k);refined=recover(uv,k,method='pnp-depth-refine',rgb=rgb,depth_m=depth)
    assert refined.reason=='valid' and refined.diagnostics['refined']
    assert np.linalg.norm(refined.base_from_object[:3,3]-truth[:3,3])<np.linalg.norm(baseline.base_from_object[:3,3]-truth[:3,3])


def test_inconsistent_interior_rejects_and_missing_foreground_is_explicit():
    k,truth,rgb,depth,mask=rendered();uv,_=project_cube(truth,k)
    bad=depth.copy();bad[mask]+=.10
    assert recover(uv,k,method='pnp-depth-refine',rgb=rgb,depth_m=bad).reason=='depth_inconsistent'
    fallback=recover(uv,k,method='pnp-depth-refine',rgb=rgb*0,depth_m=depth)
    assert fallback.reason=='valid' and fallback.diagnostics['depth_status']=='insufficient_interior_fallback_pnp'


def test_refinement_failure_is_recorded_without_claiming_refinement(monkeypatch):
    import experiments.align_precision_vision.pose_recovery as module
    def fail(*args,**kwargs):raise ValueError('synthetic optimizer failure')
    monkeypatch.setattr(module,'least_squares',fail)
    k,truth,rgb,depth,_=rendered();uv,_=project_cube(truth,k)
    result=recover(uv,k,method='pnp-depth-refine',rgb=rgb,depth_m=depth)
    assert result.reason=='valid' and not result.diagnostics['refined']
    assert result.diagnostics['depth_status']=='optimizer_failed_consistent_fallback_pnp'


def test_old_arm_is_numerically_identical():
    k,_,uv,depth=scene()
    original=measure_keypoints(uv,np.ones(8),depth,k,np.eye(4),episode='e',target='cube',timestamp_s=0.)
    result=recover(uv,k,method='old-corner-depth',depth_m=depth)
    assert original.reason==result.reason
    np.testing.assert_array_equal(original.base_from_object,result.base_from_object)


def test_pnp_rejects_unrectified_intrinsics():
    k,_,uv,_=scene();k[0,1]=.1
    with pytest.raises(ValueError,match='内参'):recover(uv,k)


def test_three_arm_denominator_keeps_rejected_frame(tmp_path):
    path,data=dataset(tmp_path)
    # 第二张没有可见点的图仍保留在三组分母。
    from experiments.align_precision_vision.train import digest
    row=data['records'][0].copy();row.update(id='missing',scene='other',file='missing.npz')
    np.savez(tmp_path/row['file'],rgb=np.zeros((101,101,3),np.uint8),pixel_uv=np.full((8,2),np.nan,np.float32),visible=np.zeros(8,bool))
    row['sha256']=digest(tmp_path/row['file']);data['records'].append(row);path.write_text(json.dumps(data))
    result=compare(path,None,tmp_path/'compare.json',keypoints='gt-diagnostic',device='cpu')
    for method,groups in result['groups'].items():
        assert groups['all']['frames']==2 and groups['all']['valid']==1
        assert groups['all']['coverage']==.5
        assert groups['all']['position_error_mm']['p90'] is not None


def test_single_evaluator_defaults_to_pnp_without_depth(tmp_path):
    from experiments.align_precision_vision.train import digest
    path,data=dataset(tmp_path);row=data['records'][0];audit_path=tmp_path/row['audit_file']
    with np.load(audit_path) as x:audit={k:x[k] for k in x.files}
    audit['depth_m'][:]=np.nan;np.savez(audit_path,**audit)
    row['audit_sha256']=digest(audit_path);path.write_text(json.dumps(data))
    result=evaluate(path,tmp_path/'evaluation.json')
    assert result['method']=='pnp-only' and result['groups']['all']['valid']==1
