"""当前腕部原图上的三头 U-Net：定位、度量残差与不确定性。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from robot_vla.observation import OBSERVATION_V2_FRAME_STATE_DIM
from robot_vla.precision.contracts import PRECISION_MODEL_ARCH, PrecisionMotionSpec


@dataclass(frozen=True)
class PrecisionUNetConfig:
    arch: str = PRECISION_MODEL_ARCH
    input_channels: int = 3
    encoder_channels: tuple[int, ...] = (32, 64, 128, 256)
    keypoint_names: tuple[str, ...] = ("object_center", "goal_center")
    mask_names: tuple[str, ...] = ("object", "goal")
    structured_state_dim: int = OBSERVATION_V2_FRAME_STATE_DIM
    state_hidden_size: int = 64
    head_hidden_size: int = 128
    motion_spec: PrecisionMotionSpec = field(default_factory=PrecisionMotionSpec)
    log_variance_min: float = -14.0
    log_variance_max: float = 4.0

    def __post_init__(self) -> None:
        if self.arch != PRECISION_MODEL_ARCH:
            raise ValueError(f"Precision U-Net arch 必须为 {PRECISION_MODEL_ARCH}")
        if self.input_channels <= 0:
            raise ValueError("input_channels 必须为正整数")
        if len(self.encoder_channels) < 2 or any(
            not isinstance(channel, int) or isinstance(channel, bool) or channel <= 0
            for channel in self.encoder_channels
        ):
            raise ValueError("encoder_channels 必须包含至少两个正整数")
        for name, values in (
            ("keypoint_names", self.keypoint_names),
            ("mask_names", self.mask_names),
        ):
            if not values or any(not value.strip() for value in values):
                raise ValueError(f"{name} 必须包含非空名称")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} 不能重复")
        if self.structured_state_dim <= 0:
            raise ValueError("structured_state_dim 必须为正整数")
        if self.state_hidden_size <= 0 or self.head_hidden_size <= 0:
            raise ValueError("state/head hidden size 必须为正整数")
        if (
            not math.isfinite(self.log_variance_min)
            or not math.isfinite(self.log_variance_max)
            or self.log_variance_min >= self.log_variance_max
        ):
            raise ValueError("log variance 截断范围无效")

    @property
    def keypoint_count(self) -> int:
        return len(self.keypoint_names)

    @property
    def mask_count(self) -> int:
        return len(self.mask_names)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = _group_count(output_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class _LocalizationHead(nn.Module):
    """稠密输出保持图像坐标，不读取 TCP/camera/force state。"""

    def __init__(self, channels: int, keypoint_count: int, mask_count: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 16)
        self.shared = _ConvBlock(channels, hidden)
        self.heatmap = nn.Conv2d(hidden, keypoint_count, 1)
        self.mask = nn.Conv2d(hidden, mask_count, 1)
        self.subpixel_offset = nn.Conv2d(hidden, keypoint_count * 2, 1)
        self.keypoint_count = keypoint_count

    def forward(
        self,
        feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.shared(feature)
        batch_size, _, height, width = shared.shape
        offsets = torch.tanh(self.subpixel_offset(shared)).reshape(
            batch_size,
            self.keypoint_count,
            2,
            height,
            width,
        )
        # 每个 heatmap cell 只允许在相邻半像素内 refinement。
        offsets = offsets * 0.5
        return self.heatmap(shared), self.mask(shared), offsets


@dataclass(frozen=True)
class DecodedKeypoints:
    pixel_uv: torch.Tensor
    normalized_uv: torch.Tensor
    peak_probability: torch.Tensor
    normalized_entropy: torch.Tensor


@dataclass(frozen=True)
class DecodedPrecisionPrediction:
    keypoints: DecodedKeypoints
    motion_residual: torch.Tensor
    visibility_probability: torch.Tensor
    projection_validity_probability: torch.Tensor
    keypoint_sigma_px: torch.Tensor
    motion_sigma: torch.Tensor
    mask_probability: torch.Tensor | None = None


@dataclass(frozen=True)
class PrecisionUNetOutput:
    heatmap_logits: torch.Tensor
    mask_logits: torch.Tensor
    subpixel_offsets: torch.Tensor
    motion_residual: torch.Tensor
    keypoint_log_variance: torch.Tensor
    motion_log_variance: torch.Tensor
    visibility_logits: torch.Tensor
    projection_validity_logit: torch.Tensor

    def decode_keypoints(self, *, temperature: float = 1.0) -> DecodedKeypoints:
        return decode_keypoints(
            self.heatmap_logits,
            self.subpixel_offsets,
            temperature=temperature,
        )

    def decode_for_control(
        self,
        *,
        temperature: float = 1.0,
        include_mask_probability: bool = False,
    ) -> DecodedPrecisionPrediction:
        if not isinstance(include_mask_probability, bool):
            raise TypeError("include_mask_probability 必须是 bool")
        keypoints = self.decode_keypoints(temperature=temperature)
        height, width = self.heatmap_logits.shape[-2:]
        pixel_scale = torch.tensor(
            (float(width), float(height)),
            dtype=torch.float32,
            device=self.heatmap_logits.device,
        )
        return DecodedPrecisionPrediction(
            keypoints=keypoints,
            motion_residual=self.motion_residual.float(),
            visibility_probability=torch.sigmoid(self.visibility_logits.float()),
            projection_validity_probability=torch.sigmoid(
                self.projection_validity_logit.float()
            ),
            # Head 3 在 normalized UV 上训练；进入控制门禁前显式换算成 pixel sigma。
            keypoint_sigma_px=torch.exp(0.5 * self.keypoint_log_variance.float())
            * pixel_scale,
            motion_sigma=torch.exp(0.5 * self.motion_log_variance.float()),
            # 证据消费者按需获取同次 forward 的 mask；默认控制输出不增加密集张量。
            mask_probability=(torch.sigmoid(self.mask_logits.float())
                              if include_mask_probability else None),
        )


def decode_keypoints(
    heatmap_logits: torch.Tensor,
    subpixel_offsets: torch.Tensor | None = None,
    *,
    temperature: float = 1.0,
) -> DecodedKeypoints:
    """用 soft-argmax 解码像素坐标，并保留峰值和归一化熵。"""

    if heatmap_logits.ndim != 4:
        raise ValueError("heatmap_logits 必须是 [B,K,H,W]")
    batch_size, keypoint_count, height, width = heatmap_logits.shape
    if height <= 0 or width <= 0:
        raise ValueError("heatmap spatial shape 必须为正")
    if not heatmap_logits.is_floating_point() or not torch.isfinite(heatmap_logits).all():
        raise ValueError("heatmap_logits 必须是有限浮点 Tensor")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature 必须是有限正数")
    if subpixel_offsets is None:
        offsets = torch.zeros(
            batch_size,
            keypoint_count,
            2,
            height,
            width,
            dtype=torch.float32,
            device=heatmap_logits.device,
        )
    else:
        expected = (batch_size, keypoint_count, 2, height, width)
        if tuple(subpixel_offsets.shape) != expected:
            raise ValueError(f"subpixel_offsets 必须为 {expected}")
        if not subpixel_offsets.is_floating_point() or not torch.isfinite(
            subpixel_offsets
        ).all():
            raise ValueError("subpixel_offsets 必须是有限浮点 Tensor")
        if torch.any(torch.abs(subpixel_offsets) > 0.5 + 1e-6):
            raise ValueError("subpixel_offsets 必须位于 [-0.5,0.5]")
        offsets = subpixel_offsets.float()

    flat_logits = heatmap_logits.float().reshape(batch_size, keypoint_count, -1)
    probability = torch.softmax(flat_logits / float(temperature), dim=-1)
    rows = torch.arange(height, dtype=torch.float32, device=heatmap_logits.device)
    columns = torch.arange(width, dtype=torch.float32, device=heatmap_logits.device)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    base_x = column_grid.reshape(1, 1, -1)
    base_y = row_grid.reshape(1, 1, -1)
    offset_x = offsets[:, :, 0].reshape(batch_size, keypoint_count, -1)
    offset_y = offsets[:, :, 1].reshape(batch_size, keypoint_count, -1)
    pixel_x = torch.sum(probability * (base_x + offset_x), dim=-1)
    pixel_y = torch.sum(probability * (base_y + offset_y), dim=-1)
    pixel_uv = torch.stack((pixel_x, pixel_y), dim=-1)
    normalized_uv = torch.stack(
        (
            (pixel_x + 0.5) / float(width),
            (pixel_y + 0.5) / float(height),
        ),
        dim=-1,
    )
    peak = probability.max(dim=-1).values
    support = height * width
    if support == 1:
        entropy = torch.zeros_like(peak)
    else:
        entropy = -torch.sum(
            probability * torch.log(probability.clamp_min(1e-12)),
            dim=-1,
        ) / math.log(support)
    return DecodedKeypoints(
        pixel_uv=pixel_uv,
        normalized_uv=normalized_uv,
        peak_probability=peak,
        normalized_entropy=entropy,
    )


class PrecisionThreeHeadUNet(nn.Module):
    """当前腕部 ROI 的 U-Net；四帧融合由外部状态估计器完成。"""

    def __init__(self, config: PrecisionUNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or PrecisionUNetConfig()
        channels = self.config.encoder_channels
        self.encoder_blocks = nn.ModuleList()
        self.encoder_blocks.append(_ConvBlock(self.config.input_channels, channels[0]))
        for input_channels, output_channels in zip(channels[:-1], channels[1:], strict=True):
            self.encoder_blocks.append(_ConvBlock(input_channels, output_channels))
        self.pool = nn.MaxPool2d(2)

        self.decoder_blocks = nn.ModuleList()
        current_channels = channels[-1]
        for skip_channels in reversed(channels[:-1]):
            self.decoder_blocks.append(
                _ConvBlock(current_channels + skip_channels, skip_channels)
            )
            current_channels = skip_channels

        self.localization_head = _LocalizationHead(
            channels[0],
            self.config.keypoint_count,
            self.config.mask_count,
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(self.config.structured_state_dim, self.config.state_hidden_size),
            nn.SiLU(),
            nn.Linear(self.config.state_hidden_size, self.config.state_hidden_size),
            nn.SiLU(),
        )
        fused_dim = (
            channels[-1]
            + self.config.state_hidden_size
            + self.config.motion_spec.motion_dim
        )
        self.motion_head = nn.Sequential(
            nn.Linear(fused_dim, self.config.head_hidden_size),
            nn.SiLU(),
            nn.Linear(self.config.head_hidden_size, self.config.motion_spec.motion_dim),
        )
        uncertainty_input_dim = fused_dim + self.config.motion_spec.motion_dim
        uncertainty_output_dim = (
            self.config.keypoint_count * 2
            + self.config.motion_spec.motion_dim
            + self.config.keypoint_count
            + 1
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(uncertainty_input_dim, self.config.head_hidden_size),
            nn.SiLU(),
            nn.Linear(self.config.head_hidden_size, uncertainty_output_dim),
        )
        self.register_buffer(
            "residual_limits",
            torch.tensor(self.config.motion_spec.residual_limits, dtype=torch.float32),
            persistent=True,
        )
        # 初始模型严格退化为 geometry-only；学习分支在 shadow 验证前不会注入偏移。
        final_motion_layer = self.motion_head[-1]
        assert isinstance(final_motion_layer, nn.Linear)
        nn.init.zeros_(final_motion_layer.weight)
        nn.init.zeros_(final_motion_layer.bias)

    def _validate_inputs(
        self,
        image: torch.Tensor,
        structured_state: torch.Tensor,
        geometric_motion: torch.Tensor,
    ) -> None:
        if image.ndim != 4 or image.shape[1] != self.config.input_channels:
            raise ValueError(
                f"image 必须是 [B,{self.config.input_channels},H,W]"
            )
        if min(image.shape[-2:]) < 2 ** (len(self.config.encoder_channels) - 1):
            raise ValueError("image spatial shape 小于 U-Net 下采样因子")
        if not image.is_floating_point() or not torch.isfinite(image).all():
            raise ValueError("image 必须是有限浮点 Tensor")
        expected_state = (image.shape[0], self.config.structured_state_dim)
        if tuple(structured_state.shape) != expected_state:
            raise ValueError(f"structured_state 必须是 {expected_state}")
        expected_motion = (image.shape[0], self.config.motion_spec.motion_dim)
        if tuple(geometric_motion.shape) != expected_motion:
            raise ValueError(f"geometric_motion 必须是 {expected_motion}")
        if not structured_state.is_floating_point() or not torch.isfinite(
            structured_state
        ).all():
            raise ValueError("structured_state 必须是有限浮点 Tensor")
        if not geometric_motion.is_floating_point() or not torch.isfinite(
            geometric_motion
        ).all():
            raise ValueError("geometric_motion 必须是有限浮点 Tensor")

    def forward(
        self,
        image: torch.Tensor,
        structured_state: torch.Tensor,
        geometric_motion: torch.Tensor,
    ) -> PrecisionUNetOutput:
        self._validate_inputs(image, structured_state, geometric_motion)
        skips: list[torch.Tensor] = []
        feature = image
        for index, block in enumerate(self.encoder_blocks):
            if index > 0:
                feature = self.pool(feature)
            feature = block(feature)
            skips.append(feature)
        bottleneck = skips[-1]

        decoded = bottleneck
        for block, skip in zip(self.decoder_blocks, reversed(skips[:-1]), strict=True):
            decoded = F.interpolate(
                decoded,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            decoded = block(torch.cat((decoded, skip), dim=1))
        heatmap_logits, mask_logits, offsets = self.localization_head(decoded)

        pooled = F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)
        model_dtype = pooled.dtype
        state_embedding = self.state_encoder(structured_state.to(dtype=model_dtype))
        geometry = geometric_motion.to(dtype=model_dtype)
        fused = torch.cat((pooled, state_embedding, geometry), dim=-1)
        residual_raw = self.motion_head(fused).float()
        motion_residual = torch.tanh(residual_raw) * self.residual_limits

        uncertainty_raw = self.uncertainty_head(
            torch.cat((fused, motion_residual.to(dtype=model_dtype)), dim=-1)
        ).float()
        keypoint_end = self.config.keypoint_count * 2
        motion_end = keypoint_end + self.config.motion_spec.motion_dim
        visibility_end = motion_end + self.config.keypoint_count
        keypoint_log_variance = uncertainty_raw[:, :keypoint_end].reshape(
            image.shape[0],
            self.config.keypoint_count,
            2,
        )
        motion_log_variance = uncertainty_raw[:, keypoint_end:motion_end]
        keypoint_log_variance = torch.clamp(
            keypoint_log_variance,
            self.config.log_variance_min,
            self.config.log_variance_max,
        )
        motion_log_variance = torch.clamp(
            motion_log_variance,
            self.config.log_variance_min,
            self.config.log_variance_max,
        )
        return PrecisionUNetOutput(
            heatmap_logits=heatmap_logits,
            mask_logits=mask_logits,
            subpixel_offsets=offsets,
            motion_residual=motion_residual,
            keypoint_log_variance=keypoint_log_variance,
            motion_log_variance=motion_log_variance,
            visibility_logits=uncertainty_raw[:, motion_end:visibility_end],
            projection_validity_logit=uncertainty_raw[:, visibility_end],
        )


__all__ = [
    "DecodedKeypoints",
    "DecodedPrecisionPrediction",
    "PrecisionThreeHeadUNet",
    "PrecisionUNetConfig",
    "PrecisionUNetOutput",
    "decode_keypoints",
]
