"""遮挡标签与完整分母评估的最小反例。"""
import json
import numpy as np
import pytest
import torch
from experiments.align_precision_vision.labels import corner_labels, project
from experiments.align_precision_vision.geometry import CUBE_POINTS
from experiments.align_precision_vision.evaluate import evaluate
from experiments.align_precision_vision.train import digest


def frontal():
    k=np.array([[200.,0,50],[0,200.,50],[0,0,1]])
    t=np.eye(4);t[2,3]=.5
    depth=np.zeros((101,101),np.float32);mask=np.zeros((101,101),bool)
    depth[42:59,42:59]=.48;mask[42:59,42:59]=True
    return k,t,depth,mask


def test_projection_does_not_imply_visibility():
    k,t,depth,mask=frontal();out=corner_labels(k,t,depth,mask)
    assert out['projected'].all()
    np.testing.assert_array_equal(out['visible'],CUBE_POINTS[:,2]<0)
    assert np.isnan(out['pixel_uv'][~out['visible']]).all()
    depth[mask]=.4
    assert not corner_labels(k,t,depth,mask)['visible'].any()


def test_mask_identity_and_behind_camera():
    k,t,depth,mask=frontal()
    assert not corner_labels(k,t,depth,np.zeros_like(mask))['visible'].any()
    t[2,3]=-.5;uv,z=project(CUBE_POINTS,k,t)
    assert np.isnan(uv).all() and (z<0).all()


def dataset(tmp_path):
    k,t,depth,mask=frontal();labels=corner_labels(k,t,depth,mask)
    sample=tmp_path/'sample.npz';audit=tmp_path/'audit.npz'
    np.savez(sample,rgb=np.zeros((101,101,3),np.uint8),pixel_uv=labels['pixel_uv'],visible=labels['visible'])
    np.savez(audit,depth_m=depth,intrinsic=k,base_from_camera_cv=np.eye(4),base_from_object=t,
             timestamp_s=0.,**labels)
    row=dict(id='one',scene='one',split='development',camera='base_camera',phase='align',
             file=sample.name,sha256=digest(sample),audit_file=audit.name,audit_sha256=digest(audit))
    data=dict(schema='align-precision-keypoints/v1',object_model='upright-cube-4cm',
              pixel_convention='zero-based-pixel-centers',records=[row])
    path=tmp_path/'manifest.json';path.write_text(json.dumps(data));return path,data


def test_gt_diagnostic_uses_complete_denominator_and_hash(tmp_path):
    path,data=dataset(tmp_path)
    result=evaluate(path,tmp_path/'result.json')
    assert result['mode']=='gt-keypoints-diagnostic'
    assert result['groups']['all']['frames']==1 and result['groups']['all']['valid']==1
    assert result['groups']['all']['within_2mm_5deg_fraction']==1
    data['records'][0]['audit_sha256']='bad';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='hash'):evaluate(path,tmp_path/'bad.json')


def test_reject_duplicate_evaluation_rows(tmp_path):
    path,data=dataset(tmp_path);data['records']*=2;path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='重复'):evaluate(path,tmp_path/'result.json')


def test_checkpoint_dataset_identity_must_match(tmp_path):
    path,_=dataset(tmp_path);checkpoint=tmp_path/'model.pt'
    torch.save(dict(format='align-precision-localizer/v1',completed=True,
                    config={'manifest_sha256':'different-data'}),checkpoint)
    with pytest.raises(ValueError,match='manifest身份'):evaluate(path,tmp_path/'result.json',checkpoint=checkpoint)
