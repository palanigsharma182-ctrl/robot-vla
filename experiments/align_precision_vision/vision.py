"""复用 Precision U-Net 图像分支；仅输出关键点及可见性，不输出动作。"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from robot_vla.precision.model import PrecisionThreeHeadUNet, PrecisionUNetConfig, decode_keypoints
from experiments.align_precision_vision.geometry import KEYPOINT_NAMES, corner_permutations


@dataclass
class LocalizationOutput:
    heatmap_logits: torch.Tensor
    offsets: torch.Tensor
    visibility_logits: torch.Tensor

    def decode(self):
        points=decode_keypoints(self.heatmap_logits,self.offsets)
        return points.pixel_uv,torch.sigmoid(self.visibility_logits.float())


class AlignLocalizer(nn.Module):
    """图像专用网络；未调用旧 Motion/State/Uncertainty 分支，不补造 V2 状态。"""
    def __init__(self,channels=(32,64,128,256)):
        super().__init__()
        existing=PrecisionThreeHeadUNet(PrecisionUNetConfig(
            encoder_channels=channels,keypoint_names=KEYPOINT_NAMES,mask_names=('object',)))
        self.encoder_blocks=existing.encoder_blocks;self.pool=existing.pool
        self.decoder_blocks=existing.decoder_blocks
        # 复用热图和格内偏移部分，不注册本轮没有监督的分割头。
        self.keypoint_shared=existing.localization_head.shared
        self.heatmap_head=existing.localization_head.heatmap
        self.offset_head=existing.localization_head.subpixel_offset
        self.visibility_head=nn.Linear(channels[-1],len(KEYPOINT_NAMES))
        self.minimum_size=2**(len(channels)-1)

    def forward(self,rgb):
        if (rgb.ndim!=4 or rgb.shape[1]!=3 or min(rgb.shape[-2:])<self.minimum_size
            or not rgb.is_floating_point() or not torch.isfinite(rgb).all()
            or bool(((rgb<0)|(rgb>1)).any())):raise ValueError('图像必须为[0,1]浮点[B,3,H,W]')
        skips=[];feature=rgb
        for i,block in enumerate(self.encoder_blocks):
            feature=block(self.pool(feature) if i else feature);skips.append(feature)
        pooled=F.adaptive_avg_pool2d(feature,1).flatten(1)
        for block,skip in zip(self.decoder_blocks,reversed(skips[:-1]),strict=True):
            feature=block(torch.cat((F.interpolate(feature,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip),dim=1))
        shared=self.keypoint_shared(feature);heatmap=self.heatmap_head(shared)
        b,k,h,w=heatmap.shape
        offsets=.5*torch.tanh(self.offset_head(shared)).reshape(b,k,2,h,w)
        return LocalizationOutput(heatmap,offsets,self.visibility_head(pooled))


def localization_loss(output,pixel_uv,visible,*,sigma_px=2.):
    """GT 仅在监督函数内使用；每个样本整组选择一个合法方块对称对应。"""
    logits=output.heatmap_logits;batch,count,height,width=logits.shape
    if (pixel_uv.shape!=(batch,count,2) or visible.shape!=(batch,count) or visible.dtype!=torch.bool
        or count!=len(KEYPOINT_NAMES) or not 0<sigma_px<min(height,width)):
        raise ValueError('监督 shape/可见性 dtype/热图宽度错误')
    selected=pixel_uv[visible]
    if not torch.isfinite(selected).all() or bool(((selected<0)|(selected>selected.new_tensor([width-1,height-1]))).any()):
        raise ValueError('可见关键点必须处于图像内且有限')
    # 不可见位置可以缺失；不对其回归，也不把 NaN 传播到 loss。
    safe=torch.where(visible[...,None],pixel_uv,torch.zeros_like(pixel_uv)).float()
    yy,xx=torch.meshgrid(torch.arange(height,device=logits.device),torch.arange(width,device=logits.device),indexing='ij')
    grid=torch.stack((xx,yy),dim=-1).float()
    predictions,_=output.decode();candidates=[]
    for permutation in corner_permutations():
        perm=torch.as_tensor(permutation,device=logits.device)
        truth=safe[:,perm];mask=visible[:,perm]
        heat=torch.exp(-((grid[None,None]-truth[:,:,None,None])**2).sum(-1)/(2*sigma_px**2))
        heat=heat/heat.sum((-1,-2),keepdim=True).clamp_min(1e-12)
        ce=-(heat*F.log_softmax(logits.float().flatten(2),dim=-1).reshape_as(logits)).sum((-1,-2))
        scale=predictions.new_tensor([width,height])
        coordinate=F.smooth_l1_loss(predictions/scale,truth/scale,reduction='none').sum(-1)
        localization=((ce+coordinate)*mask).sum(1)/mask.sum(1).clamp_min(1)
        visibility=F.binary_cross_entropy_with_logits(output.visibility_logits.float(),mask.float(),reduction='none').mean(1)
        candidates.append(localization+visibility)
    return torch.stack(candidates,dim=1).min(dim=1).values.mean()
