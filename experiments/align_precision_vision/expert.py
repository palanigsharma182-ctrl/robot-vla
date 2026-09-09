"""Align 的结构化几何条件 Expert；不修改已有策略对象或 canonical Action 合同。"""
import copy
import math
from dataclasses import replace
import torch
from torch import nn
from robot_vla.training.flow_matching import sample_flow_training_target, masked_flow_mse, euler_integrate_actions

CONDITION_DIM=8


class SplitStateProjection(nn.Module):
    def __init__(self,original):
        super().__init__();self.original=original
        self.geometry=nn.Linear(CONDITION_DIM,original.out_features,bias=False).to(original.weight)
        nn.init.zeros_(self.geometry.weight)

    def forward(self,value):
        return self.original(value[:,:15])+self.geometry(value[:,15:])


def condition_tensor(conditions,proprio,mode):
    if mode not in ('baseline','oracle','measured'):raise ValueError('未知对照模式')
    if mode=='baseline':return proprio.new_zeros((proprio.shape[0],CONDITION_DIM))
    if conditions is None or len(conditions)!=proprio.shape[0]:raise ValueError('每个样本须绑定一个几何条件')
    expected='oracle-diagnostic/v1' if mode=='oracle' else 'precision-rgbd-keypoints/v1'
    if any(c.source!=expected for c in conditions):raise ValueError('Oracle 与实测几何不能混用')
    value=proprio.new_tensor([c.features for c in conditions])
    if value.shape!=(proprio.shape[0],CONDITION_DIM) or not torch.isfinite(value).all():raise ValueError('几何条件必须有限[B,8]')
    if bool(((value[:,6]!=0)&(value[:,6]!=1)).any()) or bool(((value[:,7]<0)|(value[:,7]>1)).any()):
        raise ValueError('有效性与归一化年龄非法')
    # 无效测量严格退化为同一 Expert 无几何输入；不是默认认为已对齐。
    return torch.where(value[:,6:7].bool(),value,torch.zeros_like(value))


class PrecisionActionExpert(nn.Module):
    def __init__(self,original_expert):
        super().__init__()
        if original_expert.config.proprio_dim!=15 or original_expert.config.action_dim!=7:
            raise ValueError('仅接受现有15D proprio、7D TCP Action 的独立技能 Expert')
        self.expert=copy.deepcopy(original_expert)
        first=self.expert.state_encoder.projection[0]
        self.expert.state_encoder.projection[0]=SplitStateProjection(first)
        config=replace(self.expert.config,proprio_dim=15+CONDITION_DIM)
        self.expert.config=config;self.expert.state_encoder.config=config

    @property
    def geometry_projection(self):
        return self.expert.state_encoder.projection[0].geometry

    def forward(self,context,proprio,noisy_action,flow_time,mask,*,conditions=None,mode='measured',context_kv=None):
        if proprio.ndim!=2 or proprio.shape[-1]!=15:raise ValueError('proprio 必须为[B,15]')
        geometry=condition_tensor(conditions,proprio,mode)
        return self.expert(context,torch.cat((proprio,geometry),dim=1),noisy_action,flow_time,mask,context_kv=context_kv)

    def flow_loss(self,context,proprio,action,mask,*,conditions=None,mode='measured',generator=None):
        target=sample_flow_training_target(action,mask,generator=generator)
        predicted=self(context,proprio,target.noisy_action,target.flow_time,mask,conditions=conditions,mode=mode)
        return masked_flow_mse(predicted,target.target_velocity,mask)

    @torch.no_grad()
    def predict(self,context,proprio,noise,mask,*,conditions=None,mode='measured',steps=10):
        kv=self.expert.prepare_context_kv(context)
        return euler_integrate_actions(lambda a,t:self(context,proprio,a,t,mask,conditions=conditions,mode=mode,context_kv=kv),noise,mask,num_steps=steps)

    def optimizer(self,*,expert_lr,geometry_lr):
        """显式提供两组学习率；不把零初始化分支的有效学习当作既成事实。"""
        if not all(math.isfinite(v) and v>0 for v in (expert_lr,geometry_lr)):raise ValueError('学习率必须有限且为正')
        extra=list(self.geometry_projection.parameters());ids={id(p) for p in extra}
        rest=[p for p in self.parameters() if id(p) not in ids]
        return torch.optim.AdamW([{'params':rest,'lr':expert_lr},{'params':extra,'lr':geometry_lr}])


@torch.no_grad()
def geometry_response(model,context,proprio,noise,mask,conditions,*,mode='measured',steps=10):
    """同噪声比较实测输入与屏蔽输入；只记录动作差，不冒充闭环改善。"""
    was_training=model.training
    model.eval()
    try:
        on=model.predict(context,proprio,noise,mask,conditions=conditions,mode=mode,steps=steps)
        off=model.predict(context,proprio,noise,mask,mode='baseline',steps=steps)
    finally:
        model.train(was_training)
    diff=(on-off).float()
    return {'normalized_action_rms':float(diff.square().mean().sqrt()),'on':on,'off':off}
