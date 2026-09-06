"""空间探针关键隔离约束的合成回归测试，不加载Qwen或仿真。"""
import numpy as np
import pytest
import torch

from experiments.qwen_spatial_readout.features import red_centroid
from experiments.qwen_spatial_readout.fit import (
    Readout, error_records, metrics, predicted_world, scene_derangement, train_statistics, validate_cache,
)
from robot_vla.diagnostics.v2_online_geometry_probe import online_geometry_probe_loss


def test_color_reference_and_missing_are_label_free():
    image = np.zeros((10,20,3),dtype=np.uint8)
    assert red_centroid(image) is None
    image[2:4,6:8,0] = 255
    assert red_centroid(image) == pytest.approx([7/20,3/10])


def test_derangement_retains_phase_without_reading_labels():
    rows = [dict(scene=scene,phase=phase) for scene in (30,10,20) for phase in range(4)]
    indices = scene_derangement(rows)
    assert sorted(indices)==list(range(12))
    assert all(rows[i]['scene']!=rows[j]['scene'] and rows[i]['phase']==rows[j]['phase'] for i,j in enumerate(indices))


def test_development_cannot_change_train_statistics():
    features = torch.randn(4,5,3)
    expected = train_statistics(features,torch.tensor([0,1]))
    features[2:] = 1e8
    actual = train_statistics(features,torch.tensor([0,1]))
    for a,b in zip(expected,actual):
        torch.testing.assert_close(a,b)


@pytest.mark.parametrize('nonlinear',[False,True])
def test_autonomous_forward_and_supervised_loss(nonlinear):
    torch.manual_seed(42)
    head = Readout(8,nonlinear)
    features = torch.randn(3,4,8)
    output = head(features)
    assert output.predicted_uv.shape==(3,1,2)
    centers = torch.tensor([[.25,.25],[.75,.25],[.25,.75],[.75,.75]]).expand(3,-1,-1)
    loss = online_geometry_probe_loss(output,torch.rand(3,1,2),centers,selector_loss_weight=.1)
    loss.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    torch.testing.assert_close(head(features).predicted_uv,output.predicted_uv)


def test_plane_conversion_ignores_gt_height_and_missing_counts_as_failure():
    transform = np.eye(4)
    transform[2,3]=1
    row = dict(sample_id='a',scene=1,phase=0,visibility='visible_center',
               uv=[.5,.5],image_size=[100,100],object_position_world_m=[0,0,.02],
               calibration=dict(intrinsic_external=[100,0,49.5,0,100,49.5,0,0,1],world_from_external=transform.ravel().tolist()))
    point = predicted_world([.5,.5],row)
    row['object_position_world_m'][2]=999
    np.testing.assert_allclose(predicted_world([.5,.5],row),point)
    np.testing.assert_allclose(point,[0,0,.02],atol=1e-6)
    summary = metrics(error_records([[.5,.5],None],[row,row]))
    assert summary['xy_error_m']['invalid']==1
    assert summary['within_1cm']==dict(passed=1,total=2,fraction=.5)


def test_cache_rejects_same_length_manifest_reordering_and_corruption():
    rows = [dict(sample_id='a'), dict(sample_id='b')]
    identity = dict(sample_ids=['a','b'],sample_manifest_sha256='sha')
    cache = dict(centers=torch.zeros(4,2), features={k:torch.zeros(2,4,h) for k,h in
                 dict(layer12=2048,layer24=2048,adapter12=720).items()}, **identity)
    rgb = dict(predictions=[None,None], **identity)
    validate_cache(cache,rgb,rows,'sha')
    with pytest.raises(ValueError,match='身份或顺序'):
        validate_cache(cache,rgb,rows[::-1],'sha')
    with pytest.raises(ValueError,match='身份或顺序'):
        validate_cache(cache,rgb,rows,'changed')
    cache['features']['adapter12'][0,0,0]=float('nan')
    with pytest.raises(ValueError,match='数值无效'):
        validate_cache(cache,rgb,rows,'sha')
