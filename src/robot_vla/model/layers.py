"""模型共享的数值稳定基础层。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FP32RMSNorm(nn.RMSNorm):
    """在 FP32 中计算 RMSNorm，并保持输出与输入 dtype 一致。"""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = None if self.weight is None else self.weight.float()
        normalized = F.rms_norm(
            value.float(),
            self.normalized_shape,
            weight,
            self.eps,
        )
        return normalized.to(dtype=value.dtype)
