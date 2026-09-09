"""同帧测量到真实 Expert 的接口验证；关键点替身只隔离网络准确率。"""
import numpy as np
import pytest
import torch
from torch import nn
from experiments.align_precision_vision.pipeline import PrecisionFrame,AlignPrecisionPipeline
from experiments.align_precision_vision.expert import PrecisionActionExpert
from experiments.align_precision_vision.test_geometry import scene
from experiments.align_precision_vision.test_model import setup_expert


class FixedKeypoints(nn.Module):
    def __init__(self,uv):
        super().__init__();self.anchor=nn.Parameter(torch.tensor(0.));self.uv=uv;self.calls=0

    def forward(self,image):
        self.calls+=1
        uv=torch.tensor(self.uv,dtype=image.dtype,device=image.device)[None]
        class Output:
            def decode(self):return uv,torch.ones(1,8,device=uv.device)
        return Output()


def inputs():
    k,pose,uv,depth=scene()
    frame=PrecisionFrame(np.zeros((240,320,3),np.uint8),depth,k,np.eye(4),'e','cube',1.,1.,1.)
    localizer=FixedKeypoints(uv);pipeline=AlignPrecisionPipeline(localizer,np.eye(4))
    return frame,pose,localizer,pipeline


def test_measurement_to_expert_chunk():
    frame,pose,net,pipeline=inputs();base,c,p,noise,mask=setup_expert();expert=PrecisionActionExpert(base)
    tcp=pose.copy();tcp[0,3]-=.02
    result=pipeline.predict(expert,c,p,noise,mask,frame,tcp,episode='e',target='cube',now_s=lambda:1.,tcp_timestamp_s=1.)
    assert result['condition'].reason=='valid' and result['measurement'].source=='precision-rgbd-keypoints/v1'
    assert result['action'].shape==(1,16,7) and torch.isfinite(result['action']).all()
    assert net.calls==1


@pytest.mark.parametrize('field,value,reason',[
    ('depth_timestamp_s',.9,'rgbd_calibration_skew'),
    ('calibration_timestamp_s',1.01,'future_observation'),
])
def test_bad_frame_time_rejected_before_network(field,value,reason):
    from dataclasses import replace
    frame,pose,net,pipeline=inputs();frame=replace(frame,**{field:value})
    measurement,condition=pipeline.condition(frame,pose,episode='e',target='cube',now_s=1.,tcp_timestamp_s=1.)
    assert condition.reason==reason and condition.features[6]==0 and net.calls==0


def test_incompatible_camera_and_depth_contract():
    from dataclasses import replace
    frame,_,_,_=inputs()
    with pytest.raises(ValueError):replace(frame,camera_convention='opengl')
    with pytest.raises(ValueError):replace(frame,depth_m=frame.depth_m.astype(np.uint16))


def test_offset_grasp_requires_a_different_symmetry_contract():
    _,_,net,_=inputs();grasp=np.eye(4);grasp[0,3]=.01
    with pytest.raises(ValueError):AlignPrecisionPipeline(net,grasp)


def test_inference_time_counts_toward_observation_age():
    frame,pose,_,pipeline=inputs();base,c,p,noise,mask=setup_expert()
    ticks=iter([1.,1.,1.2])
    with pytest.raises(TimeoutError):
        pipeline.predict(PrecisionActionExpert(base),c,p,noise,mask,frame,pose,
                         episode='e',target='cube',now_s=lambda:next(ticks),tcp_timestamp_s=1.)
