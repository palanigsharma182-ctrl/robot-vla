"""厘米级闭环精调的感知、显式几何与受限控制边界。

本包刻意不在顶层导入 PyTorch 模型，使几何和控制契约可以在轻依赖环境中独立审计。
"""

from robot_vla.precision.contracts import (
    PRECISION_MODEL_ARCH,
    PRECISION_MOTION_SEMANTICS,
    PrecisionControlMode,
    PrecisionMotionSpec,
)
from robot_vla.precision.evaluation import (
    PrecisionEvaluationAssessment,
    PrecisionEvaluationMetrics,
    PrecisionEvaluationThresholds,
    PrecisionPerformanceTier,
    assess_precision_evaluation,
)

__all__ = [
    "PRECISION_MODEL_ARCH",
    "PRECISION_MOTION_SEMANTICS",
    "PrecisionControlMode",
    "PrecisionEvaluationAssessment",
    "PrecisionEvaluationMetrics",
    "PrecisionEvaluationThresholds",
    "PrecisionMotionSpec",
    "PrecisionPerformanceTier",
    "assess_precision_evaluation",
]
