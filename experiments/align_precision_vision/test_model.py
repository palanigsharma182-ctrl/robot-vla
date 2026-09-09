"""图像监督、Expert 条件消费和序列隔离，不加载 Qwen 权重。"""
import numpy as np
import pytest
import torch
from experiments.align_precision_vision.vision import AlignLocalizer,localization_loss
from experiments.align_precision_vision.geometry import AlignCondition,corner_permutations
from experiments.align_precision_vision.expert import PrecisionActionExpert,condition_tensor,geometry_response
from robot_vla.model.expert import ExpertConfig,StandaloneActionExpert
from robot_vla.model.qwen_context import QwenContext


def setup_expert():
    torch.manual_seed(11)
    base=StandaloneActionExpert(ExpertConfig(action_dim=7,context_dim=16,hidden_size=16,
        state_hidden_size=16,num_layers=2,intermediate_size=32,num_attention_heads=2,num_key_value_heads=1,head_dim=8))
    context=QwenContext(torch.randn(1,3,16),torch.ones(1,3,dtype=torch.bool))
    return base,context,torch.randn(1,15),torch.randn(1,16,7),torch.ones(1,16,dtype=torch.bool)


def condition(x=.5,source='precision-rgbd-keypoints/v1'):
    return AlignCondition((x,0.,0.,0.,0.,0.,1.,0.),source,'valid')


def test_localizer_has_only_image_parameters_and_finite_gradients():
    net=AlignLocalizer(channels=(8,16));output=net(torch.rand(2,3,24,32))
    uv,vis=output.decode();assert uv.shape==(2,8,2) and vis.shape==(2,8)
    assert not any('motion' in name or 'state_encoder' in name for name,_ in net.named_parameters())
    target=torch.full((2,8,2),10.);visible=torch.ones(2,8,dtype=torch.bool)
    loss=localization_loss(output,target,visible);loss.backward()
    assert torch.isfinite(loss) and net.heatmap_head.weight.grad.abs().sum()>0
    assert net.visibility_head.weight.grad.abs().sum()>0


def test_localization_loss_uses_whole_object_symmetry_and_masks_occlusion():
    torch.manual_seed(5);net=AlignLocalizer(channels=(8,16));out=net(torch.rand(1,3,24,32))
    uv=torch.tensor([[[4.,5.],[7.,8.],[12.,6.],[15.,8.],[5.,16.],[8.,18.],[14.,17.],[17.,19.]]])
    visible=torch.tensor([[True,False,True,True,False,True,True,False]])
    uv[~visible]=float('nan');loss=localization_loss(out,uv,visible)
    perm=torch.tensor(corner_permutations()[1]);other=localization_loss(out,uv[:,perm],visible[:,perm])
    torch.testing.assert_close(loss,other)
    all_missing=localization_loss(out,torch.full_like(uv,float('nan')),torch.zeros_like(visible))
    assert torch.isfinite(all_missing)


def test_zero_projection_matches_parent_without_mutating_it():
    base,c,p,a,mask=setup_expert();model=PrecisionActionExpert(base);t=torch.tensor([.5])
    original=base(c,p,a,t,mask)
    actual=model(c,p,a,t,mask,conditions=[condition()])
    torch.testing.assert_close(original,actual,rtol=0,atol=0)
    assert base.config.proprio_dim==15 and model.expert.config.proprio_dim==23
    assert base.state_encoder.projection[0].weight.data_ptr()!=model.expert.state_encoder.projection[0].original.weight.data_ptr()


def test_source_rejection_and_invalid_measurement_mask():
    _,_,p,_,_=setup_expert()
    with pytest.raises(ValueError):condition_tensor([condition(source='oracle-diagnostic/v1')],p,'measured')
    with pytest.raises(ValueError):condition_tensor([condition()],p,'oracle')
    invalid=AlignCondition((999.,)*6+(0.,1.),'precision-rgbd-keypoints/v1','stale_observation')
    assert torch.count_nonzero(condition_tensor([invalid],p,'measured'))==0


def test_geometry_gradient_response_and_no_cross_call_state(tmp_path):
    base,c,p,a,mask=setup_expert();model=PrecisionActionExpert(base)
    loss=model.flow_loss(c,p,torch.zeros_like(a),mask,conditions=[condition()],generator=torch.Generator().manual_seed(42))
    loss.backward();assert model.geometry_projection.weight.grad.abs().sum()>0
    # 用确定性非零投影检验依赖；这不是训练效果或方向正确的证据。
    with torch.no_grad():model.geometry_projection.weight.copy_(torch.randn_like(model.geometry_projection.weight)*.1)
    t=torch.tensor([.5]);first=model(c,p,a,t,mask,conditions=[condition(.5)])
    other=model(c,p,a,t,mask,conditions=[condition(-.5)])
    repeat=model(c,p,a,t,mask,conditions=[condition(.5)])
    assert not torch.equal(first,other);torch.testing.assert_close(first,repeat,rtol=0,atol=0)
    path=tmp_path/'model.pt';torch.save(model.state_dict(),path)
    restored=PrecisionActionExpert(base);restored.load_state_dict(torch.load(path,weights_only=True),strict=True)
    torch.testing.assert_close(first,restored(c,p,a,t,mask,conditions=[condition(.5)]))
    report=geometry_response(model,c,p,a,mask,[condition()],steps=2)
    assert model.training
    assert report['normalized_action_rms']>0 and report['on'].shape==(1,16,7)
    optimizer=model.optimizer(expert_lr=1e-5,geometry_lr=1e-3)
    assert len(optimizer.param_groups)==2
