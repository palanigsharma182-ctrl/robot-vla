"""三头 U-Net 的可审计监督目标与异方差不确定性损失。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from robot_vla.precision.model import PrecisionUNetOutput


@dataclass(frozen=True)
class PrecisionLossConfig:
    heatmap_weight: float = 1.0
    mask_weight: float = 0.5
    coordinate_weight: float = 2.0
    motion_weight: float = 1.0
    uncertainty_weight: float = 0.1
    visibility_weight: float = 0.1
    projection_weight: float = 0.1
    keypoint_temperature: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("heatmap_weight", self.heatmap_weight),
            ("mask_weight", self.mask_weight),
            ("coordinate_weight", self.coordinate_weight),
            ("motion_weight", self.motion_weight),
            ("uncertainty_weight", self.uncertainty_weight),
            ("visibility_weight", self.visibility_weight),
            ("projection_weight", self.projection_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} 必须是有限非负数")
        if not math.isfinite(self.keypoint_temperature) or self.keypoint_temperature <= 0.0:
            raise ValueError("keypoint_temperature 必须是有限正数")


@dataclass(frozen=True)
class PrecisionSupervision:
    heatmap_targets: torch.Tensor
    mask_targets: torch.Tensor
    normalized_uv_targets: torch.Tensor
    keypoint_valid: torch.Tensor
    motion_residual_targets: torch.Tensor
    motion_valid: torch.Tensor
    projection_valid: torch.Tensor


@dataclass(frozen=True)
class PrecisionLoss:
    loss: torch.Tensor
    heatmap_loss: torch.Tensor
    mask_loss: torch.Tensor
    coordinate_loss: torch.Tensor
    motion_loss: torch.Tensor
    uncertainty_loss: torch.Tensor
    visibility_loss: torch.Tensor
    projection_loss: torch.Tensor


def build_gaussian_heatmaps(
    normalized_uv: torch.Tensor,
    keypoint_valid: torch.Tensor,
    image_size_hw: tuple[int, int],
    *,
    sigma_px: float = 1.5,
) -> torch.Tensor:
    """从项目像素中心坐标生成 `[B,K,H,W]` Gaussian heatmap。"""

    if normalized_uv.ndim != 3 or normalized_uv.shape[-1] != 2:
        raise ValueError("normalized_uv 必须是 [B,K,2]")
    if (
        keypoint_valid.shape != normalized_uv.shape[:2]
        or keypoint_valid.dtype != torch.bool
    ):
        raise ValueError("keypoint_valid 必须是对齐的 bool [B,K]")
    if not normalized_uv.is_floating_point() or not torch.isfinite(normalized_uv).all():
        raise ValueError("normalized_uv 必须是有限浮点 Tensor")
    valid_uv = normalized_uv[keypoint_valid]
    if torch.any(valid_uv < 0.0) or torch.any(valid_uv > 1.0):
        raise ValueError("有效 normalized_uv 必须位于 [0,1]")
    if len(image_size_hw) != 2 or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in image_size_hw
    ):
        raise ValueError("image_size_hw 必须是两个正整数")
    if not math.isfinite(sigma_px) or sigma_px <= 0.0:
        raise ValueError("sigma_px 必须是有限正数")
    height, width = image_size_hw
    dtype = torch.float32
    rows = torch.arange(height, dtype=dtype, device=normalized_uv.device)
    columns = torch.arange(width, dtype=dtype, device=normalized_uv.device)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    target_x = normalized_uv[..., 0].float() * float(width) - 0.5
    target_y = normalized_uv[..., 1].float() * float(height) - 0.5
    squared_distance = (
        (column_grid[None, None] - target_x[..., None, None]).square()
        + (row_grid[None, None] - target_y[..., None, None]).square()
    )
    heatmap = torch.exp(-0.5 * squared_distance / float(sigma_px**2))
    return heatmap * keypoint_valid[..., None, None].to(dtype=heatmap.dtype)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if value.shape != mask.shape:
        raise ValueError("masked mean 的 value/mask shape 必须一致")
    weight = mask.to(dtype=value.dtype)
    denominator = weight.sum()
    if int(denominator.detach().item()) == 0:
        return value.sum() * 0.0
    return torch.sum(value * weight) / denominator


def precision_unet_loss(
    output: PrecisionUNetOutput,
    target: PrecisionSupervision,
    config: PrecisionLossConfig | None = None,
) -> PrecisionLoss:
    """计算定位、受限 residual 与不确定性损失；不读取历史 Expert Action。"""

    loss_config = config or PrecisionLossConfig()
    batch_size, keypoint_count, height, width = output.heatmap_logits.shape
    expected_heatmap = (batch_size, keypoint_count, height, width)
    if tuple(target.heatmap_targets.shape) != expected_heatmap:
        raise ValueError(f"heatmap_targets 必须是 {expected_heatmap}")
    expected_mask = tuple(output.mask_logits.shape)
    if tuple(target.mask_targets.shape) != expected_mask:
        raise ValueError(f"mask_targets 必须是 {expected_mask}")
    expected_keypoint = (batch_size, keypoint_count)
    if tuple(target.normalized_uv_targets.shape) != (*expected_keypoint, 2):
        raise ValueError("normalized_uv_targets 必须是 [B,K,2]")
    if (
        target.keypoint_valid.shape != expected_keypoint
        or target.keypoint_valid.dtype != torch.bool
    ):
        raise ValueError("keypoint_valid 必须是 bool [B,K]")
    if tuple(target.motion_residual_targets.shape) != tuple(output.motion_residual.shape):
        raise ValueError("motion_residual_targets 与 motion_residual shape 必须一致")
    if (
        target.motion_valid.shape != output.motion_residual.shape
        or target.motion_valid.dtype != torch.bool
    ):
        raise ValueError("motion_valid 必须是对齐 motion residual 的 bool Tensor")
    if (
        target.projection_valid.shape != (batch_size,)
        or target.projection_valid.dtype != torch.bool
    ):
        raise ValueError("projection_valid 必须是 bool [B]")

    floating_targets = (
        target.heatmap_targets,
        target.mask_targets,
        target.normalized_uv_targets,
        target.motion_residual_targets,
    )
    if any(
        not value.is_floating_point() or not torch.isfinite(value).all()
        for value in floating_targets
    ):
        raise ValueError("Precision supervision 浮点 target 必须有限")
    if torch.any(target.heatmap_targets < 0.0) or torch.any(target.heatmap_targets > 1.0):
        raise ValueError("heatmap_targets 必须位于 [0,1]")
    if torch.any(target.mask_targets < 0.0) or torch.any(target.mask_targets > 1.0):
        raise ValueError("mask_targets 必须位于 [0,1]")

    flat_heatmap_target = target.heatmap_targets.float().reshape(
        batch_size,
        keypoint_count,
        -1,
    )
    target_mass = flat_heatmap_target.sum(dim=-1, keepdim=True)
    if torch.any(target.keypoint_valid & (target_mass.squeeze(-1) <= 0.0)):
        raise ValueError("有效 keypoint 的 heatmap target 质量不能为零")
    target_distribution = flat_heatmap_target / target_mass.clamp_min(1e-12)
    heatmap_log_probability = F.log_softmax(
        output.heatmap_logits.float().reshape(batch_size, keypoint_count, -1),
        dim=-1,
    )
    heatmap_per_keypoint = -torch.sum(
        target_distribution * heatmap_log_probability,
        dim=-1,
    )
    heatmap_loss = _masked_mean(heatmap_per_keypoint, target.keypoint_valid)
    mask_loss = F.binary_cross_entropy_with_logits(
        output.mask_logits.float(),
        target.mask_targets.float(),
    )

    decoded = output.decode_keypoints(temperature=loss_config.keypoint_temperature)
    coordinate_error = F.smooth_l1_loss(
        decoded.normalized_uv,
        target.normalized_uv_targets.float(),
        reduction="none",
    ).mean(dim=-1)
    coordinate_loss = _masked_mean(coordinate_error, target.keypoint_valid)

    motion_error = F.smooth_l1_loss(
        output.motion_residual,
        target.motion_residual_targets.float(),
        reduction="none",
    )
    motion_loss = _masked_mean(motion_error, target.motion_valid)

    keypoint_squared_error = (
        decoded.normalized_uv - target.normalized_uv_targets.float()
    ).square()
    keypoint_nll = 0.5 * (
        keypoint_squared_error * torch.exp(-output.keypoint_log_variance)
        + output.keypoint_log_variance
    )
    keypoint_uncertainty_mask = target.keypoint_valid[..., None].expand_as(keypoint_nll)
    keypoint_uncertainty = _masked_mean(keypoint_nll, keypoint_uncertainty_mask)
    motion_squared_error = (
        output.motion_residual - target.motion_residual_targets.float()
    ).square()
    motion_nll = 0.5 * (
        motion_squared_error * torch.exp(-output.motion_log_variance)
        + output.motion_log_variance
    )
    motion_uncertainty = _masked_mean(motion_nll, target.motion_valid)
    uncertainty_loss = keypoint_uncertainty + motion_uncertainty

    visibility_loss = F.binary_cross_entropy_with_logits(
        output.visibility_logits,
        target.keypoint_valid.to(dtype=torch.float32),
    )
    projection_loss = F.binary_cross_entropy_with_logits(
        output.projection_validity_logit,
        target.projection_valid.to(dtype=torch.float32),
    )
    total = (
        loss_config.heatmap_weight * heatmap_loss
        + loss_config.mask_weight * mask_loss
        + loss_config.coordinate_weight * coordinate_loss
        + loss_config.motion_weight * motion_loss
        + loss_config.uncertainty_weight * uncertainty_loss
        + loss_config.visibility_weight * visibility_loss
        + loss_config.projection_weight * projection_loss
    )
    return PrecisionLoss(
        loss=total,
        heatmap_loss=heatmap_loss,
        mask_loss=mask_loss,
        coordinate_loss=coordinate_loss,
        motion_loss=motion_loss,
        uncertainty_loss=uncertainty_loss,
        visibility_loss=visibility_loss,
        projection_loss=projection_loss,
    )


__all__ = [
    "PrecisionLoss",
    "PrecisionLossConfig",
    "PrecisionSupervision",
    "build_gaussian_heatmaps",
    "precision_unet_loss",
]
