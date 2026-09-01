"""原始 wrist RGB 到四时刻部署检测的 replay/shadow-only Provider。"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, Protocol

import numpy as np

from robot_vla.adapters import FingerForceNormalizer, ProprioNormalizer
from robot_vla.contracts import OBSERVATION_V2_VERSION, RobotSpec
from robot_vla.executive.contracts import PredicateSource
from robot_vla.executive.estimation import WristKeypointDetection
from robot_vla.observation import (
    OBSERVATION_MODALITIES,
    OBSERVATION_V2_FRAME_STATE_DIM,
    ObservationV2Window,
)
from robot_vla.precision.contracts import (
    PRECISION_MODEL_ARCH,
    PRECISION_MOTION_SEMANTICS,
)
from robot_vla.precision.detection import (
    PRECISION_TRACK_CONFIDENCE_SEMANTICS,
    PrecisionDetectionAdapterResult,
    precision_prediction_to_wrist_detection,
)

PRECISION_FRAME_PREDICTOR_VERSION = "precision-torch-frame-predictor/v1"
PRECISION_DETECTION_PROVIDER_VERSION = "precision-wrist-detection-provider/v1"
PRECISION_DETECTION_EXECUTION_MODE = "replay-shadow-only/no-actuation/v1"
PRECISION_IMAGE_INPUT_SEMANTICS = "rgb-uint8-hwc/to-float32-chw-unit-range/v1"
PRECISION_STRUCTURED_STATE_INPUT_SEMANTICS = (
    "observation-v2/frame-row/as-current-with-age-zero/v1"
)
PRECISION_FRAME_ORDER = "valid-history/oldest-to-newest/sequential-batch1/v1"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} 必须是 64 位小写 SHA-256")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TorchPrecisionFramePredictorConfig:
    """只负责冻结模型 forward；不包含 Action 或 controller 权限。"""

    device: str = "cpu"
    use_bf16: bool = False
    temperature: float = 1.0
    synchronize_cuda_for_latency: bool = True
    version: str = PRECISION_FRAME_PREDICTOR_VERSION

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Precision predictor device 只能是 cpu 或 cuda")
        if not isinstance(self.use_bf16, bool):
            raise TypeError("Precision predictor use_bf16 必须为 bool")
        if self.use_bf16 and self.device != "cuda":
            raise ValueError("Precision predictor BF16 只允许用于 CUDA")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("Precision predictor temperature 必须是有限正数")
        if not isinstance(self.synchronize_cuda_for_latency, bool):
            raise TypeError("synchronize_cuda_for_latency 必须为 bool")
        if self.version != PRECISION_FRAME_PREDICTOR_VERSION:
            raise ValueError(
                f"Precision predictor version 必须为 {PRECISION_FRAME_PREDICTOR_VERSION}"
            )


@dataclass(frozen=True)
class PrecisionPredictorIdentity:
    checkpoint_sha256: str
    parameter_state_sha256: str
    model_config_sha256: str
    keypoint_names: tuple[str, ...]
    structured_state_dim: int
    motion_dim: int
    device_type: str
    use_bf16: bool
    temperature: float
    synchronize_cuda_for_latency: bool = True
    model_arch: str = PRECISION_MODEL_ARCH
    motion_semantics: str = PRECISION_MOTION_SEMANTICS
    image_input_semantics: str = PRECISION_IMAGE_INPUT_SEMANTICS
    structured_state_input_semantics: str = (
        PRECISION_STRUCTURED_STATE_INPUT_SEMANTICS
    )
    version: str = PRECISION_FRAME_PREDICTOR_VERSION

    def __post_init__(self) -> None:
        for value, name in (
            (self.checkpoint_sha256, "checkpoint_sha256"),
            (self.parameter_state_sha256, "parameter_state_sha256"),
            (self.model_config_sha256, "model_config_sha256"),
        ):
            _require_sha256(value, name)
        if self.model_arch != PRECISION_MODEL_ARCH:
            raise ValueError(f"Precision model arch 必须为 {PRECISION_MODEL_ARCH}")
        if self.motion_semantics != PRECISION_MOTION_SEMANTICS:
            raise ValueError(
                f"Precision motion semantics 必须为 {PRECISION_MOTION_SEMANTICS}"
            )
        if self.image_input_semantics != PRECISION_IMAGE_INPUT_SEMANTICS:
            raise ValueError("Precision image input semantics 漂移")
        if (
            self.structured_state_input_semantics
            != PRECISION_STRUCTURED_STATE_INPUT_SEMANTICS
        ):
            raise ValueError("Precision structured state input semantics 漂移")
        if self.version != PRECISION_FRAME_PREDICTOR_VERSION:
            raise ValueError("Precision predictor identity version 漂移")
        if (
            not self.keypoint_names
            or any(
                not isinstance(name, str) or not name.strip()
                for name in self.keypoint_names
            )
            or len(set(self.keypoint_names)) != len(self.keypoint_names)
        ):
            raise ValueError("Precision predictor keypoint_names 无效")
        if not {"object_center", "goal_center"}.issubset(self.keypoint_names):
            raise ValueError("Precision predictor 必须包含 object_center/goal_center")
        if self.structured_state_dim <= 0 or self.motion_dim <= 0:
            raise ValueError("Precision predictor state/motion dim 必须为正数")
        if self.device_type not in {"cpu", "cuda"}:
            raise ValueError("Precision predictor device_type 无效")
        if not isinstance(self.use_bf16, bool):
            raise TypeError("Precision predictor use_bf16 必须为 bool")
        if not isinstance(self.synchronize_cuda_for_latency, bool):
            raise TypeError("Precision predictor synchronize_cuda_for_latency 必须为 bool")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("Precision predictor temperature 必须是有限正数")

    def to_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


class PrecisionFramePredictor(Protocol):
    @property
    def identity(self) -> PrecisionPredictorIdentity: ...

    def predict(
        self,
        rgb_wrist: np.ndarray,
        structured_state: np.ndarray,
        geometric_motion: np.ndarray,
    ) -> Any: ...


class TorchPrecisionFramePredictor:
    """惰性导入 Torch，顺序执行冻结的三头 U-Net 单帧 forward。"""

    def __init__(
        self,
        model: Any,
        *,
        checkpoint_sha256: str,
        config: TorchPrecisionFramePredictorConfig | None = None,
    ) -> None:
        _require_sha256(checkpoint_sha256, "checkpoint_sha256")
        self.config = config or TorchPrecisionFramePredictorConfig()
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as error:
            raise RuntimeError("TorchPrecisionFramePredictor 需要安装 PyTorch") from error
        self._torch = torch
        self.device = torch.device(self.config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Precision predictor 请求 CUDA，但当前不可用")
        if (
            self.config.use_bf16
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("当前 CUDA 不支持 Precision predictor BF16")

        model_config = getattr(model, "config", None)
        if model_config is None or not is_dataclass(model_config):
            raise TypeError("Precision model 必须携带 dataclass config")
        if getattr(model_config, "arch", None) != PRECISION_MODEL_ARCH:
            raise ValueError("Precision model arch 漂移")
        if getattr(model_config, "input_channels", None) != 3:
            raise ValueError("Precision provider 只接受三通道 RGB model")
        motion_spec = getattr(model_config, "motion_spec", None)
        if (
            motion_spec is None
            or getattr(motion_spec, "semantics", None) != PRECISION_MOTION_SEMANTICS
        ):
            raise ValueError("Precision model motion semantics 漂移")

        self.model = model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        if self.model.training or any(
            parameter.requires_grad for parameter in self.model.parameters()
        ):
            raise RuntimeError("Precision predictor 必须冻结且处于 eval 模式")
        parameter_sha256 = self._parameter_state_sha256()
        model_config_sha256 = _canonical_sha256(model_config)
        self._identity = PrecisionPredictorIdentity(
            checkpoint_sha256=checkpoint_sha256,
            parameter_state_sha256=parameter_sha256,
            model_config_sha256=model_config_sha256,
            keypoint_names=tuple(model_config.keypoint_names),
            structured_state_dim=int(model_config.structured_state_dim),
            motion_dim=int(motion_spec.motion_dim),
            device_type=self.device.type,
            use_bf16=self.config.use_bf16,
            temperature=float(self.config.temperature),
            synchronize_cuda_for_latency=self.config.synchronize_cuda_for_latency,
        )

    @property
    def identity(self) -> PrecisionPredictorIdentity:
        return self._identity

    def verify_identity(self) -> None:
        """Episode 起点复核冻结状态，避免 checkpoint/model tensor 静默漂移。"""

        if self.model.training or any(
            parameter.requires_grad for parameter in self.model.parameters()
        ):
            raise RuntimeError("Precision predictor 不再处于 frozen eval 状态")
        if self._parameter_state_sha256() != self.identity.parameter_state_sha256:
            raise RuntimeError("Precision predictor parameter state SHA-256 漂移")
        if _canonical_sha256(self.model.config) != self.identity.model_config_sha256:
            raise RuntimeError("Precision predictor model config SHA-256 漂移")

    def _parameter_state_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.model.state_dict().items()):
            if not isinstance(value, self._torch.Tensor):
                raise TypeError(f"Precision model state {name} 不是 Tensor")
            tensor = value.detach().cpu().contiguous()
            encoded_name = name.encode("utf-8")
            digest.update(len(encoded_name).to_bytes(4, "big"))
            digest.update(encoded_name)
            encoded_dtype = str(tensor.dtype).encode("ascii")
            digest.update(len(encoded_dtype).to_bytes(2, "big"))
            digest.update(encoded_dtype)
            shape = tuple(int(dimension) for dimension in tensor.shape)
            digest.update(len(shape).to_bytes(2, "big"))
            for dimension in shape:
                digest.update(dimension.to_bytes(8, "big", signed=False))
            raw = tensor.view(self._torch.uint8).numpy().tobytes()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        return digest.hexdigest()

    def predict(
        self,
        rgb_wrist: np.ndarray,
        structured_state: np.ndarray,
        geometric_motion: np.ndarray,
    ) -> Any:
        image = np.asarray(rgb_wrist)
        if (
            image.ndim != 3
            or image.shape[-1] != 3
            or min(image.shape[:2]) <= 0
            or image.dtype != np.uint8
        ):
            raise ValueError("rgb_wrist 必须是 uint8 [H,W,3]")
        state = np.asarray(structured_state)
        if (
            state.shape != (self.identity.structured_state_dim,)
            or state.dtype != np.float32
            or not np.isfinite(state).all()
        ):
            raise ValueError("structured_state 与冻结 Precision model 不一致")
        motion = np.asarray(geometric_motion)
        if (
            motion.shape != (self.identity.motion_dim,)
            or motion.dtype != np.float32
            or not np.isfinite(motion).all()
        ):
            raise ValueError("geometric_motion 与冻结 Precision model 不一致")

        image_float = np.ascontiguousarray(
            image.transpose(2, 0, 1),
            dtype=np.float32,
        ) / np.float32(255.0)
        image_tensor = self._torch.from_numpy(image_float).unsqueeze(0).to(self.device)
        state_tensor = self._torch.from_numpy(state.copy()).unsqueeze(0).to(self.device)
        motion_tensor = self._torch.from_numpy(motion.copy()).unsqueeze(0).to(self.device)
        if self.device.type == "cuda" and self.config.synchronize_cuda_for_latency:
            self._torch.cuda.synchronize(self.device)
        with self._torch.inference_mode():
            with self._torch.autocast(
                device_type=self.device.type,
                dtype=self._torch.bfloat16,
                enabled=self.config.use_bf16,
            ):
                output = self.model(image_tensor, state_tensor, motion_tensor)
            decoded = output.decode_for_control(temperature=self.config.temperature)
        if self.device.type == "cuda" and self.config.synchronize_cuda_for_latency:
            self._torch.cuda.synchronize(self.device)
        return decoded


@dataclass(frozen=True)
class PrecisionGeometricMotionInput:
    timestamp_s: float
    motion: tuple[float, ...]
    source: PredicateSource = PredicateSource.DEPLOYABLE_ESTIMATOR
    semantics: str = PRECISION_MOTION_SEMANTICS

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestamp_s, bool)
            or not math.isfinite(self.timestamp_s)
            or self.timestamp_s < 0.0
        ):
            raise ValueError("Precision geometry timestamp_s 必须是有限非负数")
        if len(self.motion) != 4 or any(
            isinstance(value, bool) or not math.isfinite(value)
            for value in self.motion
        ):
            raise ValueError("Precision geometric motion 必须是四个有限数值")
        if self.source != PredicateSource.DEPLOYABLE_ESTIMATOR:
            raise ValueError("Precision runtime geometry 禁止使用 evaluator GT")
        if self.semantics != PRECISION_MOTION_SEMANTICS:
            raise ValueError("Precision geometric motion semantics 漂移")

    def as_array(self) -> np.ndarray:
        return np.asarray(self.motion, dtype=np.float32)


PrecisionGeometricMotionProvider = Callable[
    [ObservationV2Window, int],
    PrecisionGeometricMotionInput,
]


@dataclass(frozen=True)
class PrecisionDetectionProviderConfig:
    enabled: bool = False
    max_geometry_timestamp_error_s: float = 1e-6
    execution_mode: str = PRECISION_DETECTION_EXECUTION_MODE
    frame_order: str = PRECISION_FRAME_ORDER
    version: str = PRECISION_DETECTION_PROVIDER_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("Precision detection provider enabled 必须为 bool")
        if (
            not math.isfinite(self.max_geometry_timestamp_error_s)
            or self.max_geometry_timestamp_error_s <= 0.0
        ):
            raise ValueError("max_geometry_timestamp_error_s 必须是有限正数")
        if self.execution_mode != PRECISION_DETECTION_EXECUTION_MODE:
            raise ValueError("Precision detection provider execution mode 漂移")
        if self.frame_order != PRECISION_FRAME_ORDER:
            raise ValueError("Precision detection provider frame order 漂移")
        if self.version != PRECISION_DETECTION_PROVIDER_VERSION:
            raise ValueError("Precision detection provider version 漂移")

    def to_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class PrecisionDetectionProviderIdentity:
    predictor: PrecisionPredictorIdentity
    robot_spec_sha256: str
    proprio_stats_sha256: str
    proprio_normalizer_sha256: str
    finger_force_stats_sha256: str
    finger_force_normalizer_sha256: str
    geometric_motion_provider_id: str
    provider_config_sha256: str
    observation_version: str = OBSERVATION_V2_VERSION
    confidence_semantics: str = PRECISION_TRACK_CONFIDENCE_SEMANTICS
    version: str = PRECISION_DETECTION_PROVIDER_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.predictor, PrecisionPredictorIdentity):
            raise TypeError("predictor 必须提供 PrecisionPredictorIdentity")
        for value, name in (
            (self.robot_spec_sha256, "robot_spec_sha256"),
            (self.proprio_stats_sha256, "proprio_stats_sha256"),
            (self.proprio_normalizer_sha256, "proprio_normalizer_sha256"),
            (self.finger_force_stats_sha256, "finger_force_stats_sha256"),
            (self.finger_force_normalizer_sha256, "finger_force_normalizer_sha256"),
            (self.provider_config_sha256, "provider_config_sha256"),
        ):
            _require_sha256(value, name)
        if (
            not isinstance(self.geometric_motion_provider_id, str)
            or not self.geometric_motion_provider_id.strip()
        ):
            raise ValueError("geometric_motion_provider_id 不能为空")
        if self.observation_version != OBSERVATION_V2_VERSION:
            raise ValueError("Precision provider observation version 漂移")
        if self.confidence_semantics != PRECISION_TRACK_CONFIDENCE_SEMANTICS:
            raise ValueError("Precision provider confidence semantics 漂移")
        if self.version != PRECISION_DETECTION_PROVIDER_VERSION:
            raise ValueError("Precision provider identity version 漂移")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "observation_version": self.observation_version,
            "confidence_semantics": self.confidence_semantics,
            "predictor": self.predictor.to_dict(),
            "robot_spec_sha256": self.robot_spec_sha256,
            "proprio_stats_sha256": self.proprio_stats_sha256,
            "proprio_normalizer_sha256": self.proprio_normalizer_sha256,
            "finger_force_stats_sha256": self.finger_force_stats_sha256,
            "finger_force_normalizer_sha256": self.finger_force_normalizer_sha256,
            "geometric_motion_provider_id": self.geometric_motion_provider_id,
            "provider_config_sha256": self.provider_config_sha256,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


class PrecisionFrameStatus(str, Enum):
    PADDING = "padding"
    WRIST_RGB_MISSING = "wrist_rgb_missing"
    PREDICTED = "predicted"
    ERROR = "error"
    NOT_RUN_AFTER_ERROR = "not_run_after_error"


@dataclass(frozen=True)
class PrecisionFrameInferenceRecord:
    frame_index: int
    status: PrecisionFrameStatus
    frame_timestamp_s: float | None
    wrist_timestamp_s: float | None
    geometry_timestamp_s: float | None
    predictor_latency_s: float
    total_latency_s: float
    evidence: PrecisionDetectionAdapterResult | None = None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.frame_index, int)
            or isinstance(self.frame_index, bool)
            or not 0 <= self.frame_index < 4
        ):
            raise ValueError("Precision frame_index 必须位于 [0,3]")
        for value, name in (
            (self.frame_timestamp_s, "frame_timestamp_s"),
            (self.wrist_timestamp_s, "wrist_timestamp_s"),
            (self.geometry_timestamp_s, "geometry_timestamp_s"),
        ):
            if value is not None and (
                not math.isfinite(value) or value < 0.0
            ):
                raise ValueError(f"{name} 必须是有限非负数或 None")
        for value, name in (
            (self.predictor_latency_s, "predictor_latency_s"),
            (self.total_latency_s, "total_latency_s"),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} 必须是有限非负数")
        if self.predictor_latency_s > self.total_latency_s + 1e-9:
            raise ValueError("predictor latency 不能大于 frame total latency")
        if self.status == PrecisionFrameStatus.PREDICTED:
            if (
                self.evidence is None
                or self.wrist_timestamp_s is None
                or self.geometry_timestamp_s is None
                or self.error_type is not None
                or self.error_message is not None
            ):
                raise ValueError("predicted frame record 字段不完整")
        elif self.evidence is not None:
            raise ValueError("非 predicted frame 不得携带 detection evidence")
        if self.status == PrecisionFrameStatus.ERROR and self.error_type is None:
            raise ValueError("error frame 必须记录 error_type")
        if self.status != PrecisionFrameStatus.ERROR and (
            self.error_type is not None or self.error_message is not None
        ):
            raise ValueError("非 error frame 不得携带错误")

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "status": self.status.value,
            "frame_timestamp_s": self.frame_timestamp_s,
            "wrist_timestamp_s": self.wrist_timestamp_s,
            "geometry_timestamp_s": self.geometry_timestamp_s,
            "predictor_latency_s": self.predictor_latency_s,
            "total_latency_s": self.total_latency_s,
            "evidence": None if self.evidence is None else self.evidence.to_dict(),
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class PrecisionDetectionProviderCall:
    call_index: int
    observation_timestamp_s: float | None
    success: bool
    detections_count: int
    frame_records: tuple[PrecisionFrameInferenceRecord, ...]
    provider_identity_sha256: str
    total_latency_s: float
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.call_index, int)
            or isinstance(self.call_index, bool)
            or self.call_index < 0
        ):
            raise ValueError("Precision provider call_index 必须是非负整数")
        if self.observation_timestamp_s is not None and (
            not math.isfinite(self.observation_timestamp_s)
            or self.observation_timestamp_s < 0.0
        ):
            raise ValueError("observation_timestamp_s 必须是有限非负数或 None")
        if not isinstance(self.success, bool):
            raise TypeError("Precision provider call success 必须为 bool")
        if (
            not isinstance(self.detections_count, int)
            or isinstance(self.detections_count, bool)
            or not 0 <= self.detections_count <= 4
        ):
            raise ValueError("detections_count 必须位于 [0,4]")
        if len(self.frame_records) != 4 or tuple(
            record.frame_index for record in self.frame_records
        ) != (0, 1, 2, 3):
            raise ValueError("Precision provider frame records 必须完整覆盖 0..3")
        _require_sha256(self.provider_identity_sha256, "provider_identity_sha256")
        if not math.isfinite(self.total_latency_s) or self.total_latency_s < 0.0:
            raise ValueError("Precision provider total_latency_s 必须是有限非负数")
        predicted = sum(
            record.status == PrecisionFrameStatus.PREDICTED
            for record in self.frame_records
        )
        if predicted != self.detections_count:
            raise ValueError("detections_count 与 predicted frame 数不一致")
        if self.success:
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("成功 provider call 不得携带错误")
            if any(
                record.status
                in {PrecisionFrameStatus.ERROR, PrecisionFrameStatus.NOT_RUN_AFTER_ERROR}
                for record in self.frame_records
            ):
                raise ValueError("成功 provider call 不得含 error/not-run frame")
        elif self.error_type is None:
            raise ValueError("失败 provider call 必须记录 error_type")

    def to_dict(self) -> dict[str, object]:
        return {
            "call_index": self.call_index,
            "observation_timestamp_s": self.observation_timestamp_s,
            "success": self.success,
            "detections_count": self.detections_count,
            "frame_records": [record.to_dict() for record in self.frame_records],
            "provider_identity_sha256": self.provider_identity_sha256,
            "total_latency_s": self.total_latency_s,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


class PrecisionDetectionProviderError(RuntimeError):
    def __init__(self, call: PrecisionDetectionProviderCall) -> None:
        self.call = call
        super().__init__(
            f"Precision detection provider call {call.call_index} 失败: "
            f"{call.error_type}: {call.error_message}"
        )


class PrecisionDetectionProvider:
    """顺序推理四时刻 wrist RGB；只返回检测，永不生成或执行 Action。"""

    def __init__(
        self,
        spec: RobotSpec,
        predictor: PrecisionFramePredictor,
        proprio_normalizer: ProprioNormalizer,
        finger_force_normalizer: FingerForceNormalizer,
        geometric_motion_provider: PrecisionGeometricMotionProvider,
        *,
        geometric_motion_provider_id: str,
        proprio_stats_sha256: str,
        finger_force_stats_sha256: str,
        config: PrecisionDetectionProviderConfig | None = None,
    ) -> None:
        self.spec = spec
        self.predictor = predictor
        self.proprio_normalizer = proprio_normalizer
        self.finger_force_normalizer = finger_force_normalizer
        self.geometric_motion_provider = geometric_motion_provider
        self.config = config or PrecisionDetectionProviderConfig()
        predictor_identity = predictor.identity
        if not isinstance(predictor_identity, PrecisionPredictorIdentity):
            raise TypeError("Precision predictor identity 无效")
        if predictor_identity.structured_state_dim != OBSERVATION_V2_FRAME_STATE_DIM:
            raise ValueError("Precision predictor state dim 与 Observation V2 不一致")
        if predictor_identity.motion_dim != 4:
            raise ValueError("Precision predictor motion dim 必须为 4")
        if proprio_normalizer.spec.to_dict() != spec.to_dict():
            raise ValueError("Precision proprio normalizer RobotSpec 漂移")
        if finger_force_normalizer.spec.to_dict() != spec.to_dict():
            raise ValueError("Precision force normalizer RobotSpec 漂移")
        if not callable(geometric_motion_provider):
            raise TypeError("geometric_motion_provider 必须可调用")
        if (
            not isinstance(geometric_motion_provider_id, str)
            or not geometric_motion_provider_id.strip()
        ):
            raise ValueError("geometric_motion_provider_id 不能为空")
        _require_sha256(proprio_stats_sha256, "proprio_stats_sha256")
        _require_sha256(finger_force_stats_sha256, "finger_force_stats_sha256")

        proprio_normalizer_sha256 = _canonical_sha256(
            {
                "mean": proprio_normalizer.mean.tolist(),
                "std": proprio_normalizer.std.tolist(),
                "clip": proprio_normalizer.clip,
                "robot_spec": spec.to_dict(),
            }
        )
        finger_force_normalizer_sha256 = _canonical_sha256(
            {
                "stats": asdict(finger_force_normalizer.stats),
                "scale": finger_force_normalizer.scale.tolist(),
                "clip": finger_force_normalizer.clip,
                "robot_spec": spec.to_dict(),
            }
        )
        self._identity = PrecisionDetectionProviderIdentity(
            predictor=predictor_identity,
            robot_spec_sha256=_canonical_sha256(spec.to_dict()),
            proprio_stats_sha256=proprio_stats_sha256,
            proprio_normalizer_sha256=proprio_normalizer_sha256,
            finger_force_stats_sha256=finger_force_stats_sha256,
            finger_force_normalizer_sha256=finger_force_normalizer_sha256,
            geometric_motion_provider_id=geometric_motion_provider_id,
            provider_config_sha256=_canonical_sha256(self.config.to_dict()),
        )
        self._records: list[PrecisionDetectionProviderCall] = []

    @property
    def identity(self) -> PrecisionDetectionProviderIdentity:
        return self._identity

    @property
    def records(self) -> tuple[PrecisionDetectionProviderCall, ...]:
        return tuple(self._records)

    @property
    def last_call(self) -> PrecisionDetectionProviderCall | None:
        return None if not self._records else self._records[-1]

    @property
    def records_jsonl(self) -> str:
        if not self._records:
            return ""
        return "\n".join(record.canonical_json() for record in self._records) + "\n"

    @property
    def records_sha256(self) -> str:
        return hashlib.sha256(self.records_jsonl.encode("utf-8")).hexdigest()

    def reset(self) -> None:
        verify_identity = getattr(self.predictor, "verify_identity", None)
        if callable(verify_identity):
            verify_identity()
        self._records.clear()

    def _state_history(self, window: ObservationV2Window) -> np.ndarray:
        normalized_proprio = np.zeros_like(window.physical_proprio)
        proprio_index = OBSERVATION_MODALITIES.index("proprio")
        valid_proprio = window.history_valid & window.modality_valid[:, proprio_index]
        normalized_proprio[valid_proprio] = self.proprio_normalizer.normalize(
            window.physical_proprio[valid_proprio]
        )
        normalized_force = np.zeros_like(window.finger_force_n)
        force_index = OBSERVATION_MODALITIES.index("finger_force")
        valid_force = window.history_valid & window.modality_valid[:, force_index]
        normalized_force[valid_force] = self.finger_force_normalizer.normalize(
            window.finger_force_n[valid_force]
        )
        state_history = window.frame_state(normalized_proprio, normalized_force)
        frame_age_index = self.spec.proprio_dim + 3 + 6 + 3 + 6 + 2
        expected_frame_age_index = (
            OBSERVATION_V2_FRAME_STATE_DIM - len(OBSERVATION_MODALITIES) - 1
        )
        if frame_age_index != expected_frame_age_index:
            raise RuntimeError("Observation V2 frame_age feature index 漂移")
        # U-Net 训练/部署都是单帧作为当前帧；真实 freshness 由外部 timestamp estimator 处理。
        state_history[window.history_valid, frame_age_index] = 0.0
        return state_history

    @staticmethod
    def _not_run_record(
        window: ObservationV2Window | None,
        frame_index: int,
    ) -> PrecisionFrameInferenceRecord:
        frame_timestamp = (
            float(window.frame_timestamp_s[frame_index])
            if window is not None and window.history_valid[frame_index]
            else None
        )
        return PrecisionFrameInferenceRecord(
            frame_index=frame_index,
            status=PrecisionFrameStatus.NOT_RUN_AFTER_ERROR,
            frame_timestamp_s=frame_timestamp,
            wrist_timestamp_s=None,
            geometry_timestamp_s=None,
            predictor_latency_s=0.0,
            total_latency_s=0.0,
        )

    def _failed_call(
        self,
        *,
        window: ObservationV2Window | None,
        started: float,
        frame_records: list[PrecisionFrameInferenceRecord],
        error: Exception,
    ) -> PrecisionDetectionProviderCall:
        completed_indices = {record.frame_index for record in frame_records}
        for frame_index in range(4):
            if frame_index not in completed_indices:
                frame_records.append(self._not_run_record(window, frame_index))
        frame_records.sort(key=lambda record: record.frame_index)
        call = PrecisionDetectionProviderCall(
            call_index=len(self._records),
            observation_timestamp_s=(
                float(window.timestamp_s)
                if window is not None
                else None
            ),
            success=False,
            detections_count=sum(
                record.status == PrecisionFrameStatus.PREDICTED
                for record in frame_records
            ),
            frame_records=tuple(frame_records),
            provider_identity_sha256=self.identity.sha256,
            total_latency_s=time.perf_counter() - started,
            error_type=type(error).__name__,
            error_message=str(error),
        )
        self._records.append(call)
        return call

    def __call__(
        self,
        window: ObservationV2Window,
    ) -> tuple[WristKeypointDetection | None, ...]:
        if not self.config.enabled:
            raise RuntimeError("Precision detection provider 默认关闭，必须显式 enabled")
        started = time.perf_counter()
        frame_records: list[PrecisionFrameInferenceRecord] = []
        try:
            if not isinstance(window, ObservationV2Window):
                raise TypeError("Precision detection provider 只接受 ObservationV2Window")
            window.validate(self.spec)
            if self.predictor.identity != self.identity.predictor:
                raise ValueError("Precision predictor identity 在 Provider 构造后漂移")
            state_history = self._state_history(window)
        except Exception as error:
            call = self._failed_call(
                window=(window if isinstance(window, ObservationV2Window) else None),
                started=started,
                frame_records=frame_records,
                error=error,
            )
            raise PrecisionDetectionProviderError(call) from error

        detections: list[WristKeypointDetection | None] = [None] * 4
        wrist_index = OBSERVATION_MODALITIES.index("rgb_wrist")
        for frame_index in range(4):
            frame_timestamp = (
                float(window.frame_timestamp_s[frame_index])
                if window.history_valid[frame_index]
                else None
            )
            if not window.history_valid[frame_index]:
                frame_records.append(
                    PrecisionFrameInferenceRecord(
                        frame_index=frame_index,
                        status=PrecisionFrameStatus.PADDING,
                        frame_timestamp_s=None,
                        wrist_timestamp_s=None,
                        geometry_timestamp_s=None,
                        predictor_latency_s=0.0,
                        total_latency_s=0.0,
                    )
                )
                continue
            if not window.modality_valid[frame_index, wrist_index]:
                frame_records.append(
                    PrecisionFrameInferenceRecord(
                        frame_index=frame_index,
                        status=PrecisionFrameStatus.WRIST_RGB_MISSING,
                        frame_timestamp_s=frame_timestamp,
                        wrist_timestamp_s=None,
                        geometry_timestamp_s=None,
                        predictor_latency_s=0.0,
                        total_latency_s=0.0,
                    )
                )
                continue

            frame_started = time.perf_counter()
            predictor_latency = 0.0
            wrist_timestamp = float(
                window.modality_timestamp_s[frame_index, wrist_index]
            )
            geometry_timestamp: float | None = None
            try:
                geometry = self.geometric_motion_provider(window, frame_index)
                if not isinstance(geometry, PrecisionGeometricMotionInput):
                    raise TypeError(
                        "geometric_motion_provider 必须返回 PrecisionGeometricMotionInput"
                    )
                geometry_timestamp = float(geometry.timestamp_s)
                if (
                    abs(geometry_timestamp - wrist_timestamp)
                    > self.config.max_geometry_timestamp_error_s
                ):
                    raise ValueError("Precision geometry 与 wrist RGB timestamp 不一致")
                predictor_started = time.perf_counter()
                try:
                    prediction = self.predictor.predict(
                        window.rgb_wrist[frame_index],
                        state_history[frame_index],
                        geometry.as_array(),
                    )
                finally:
                    predictor_latency = time.perf_counter() - predictor_started
                evidence = precision_prediction_to_wrist_detection(
                    prediction,
                    keypoint_names=self.identity.predictor.keypoint_names,
                    timestamp_s=wrist_timestamp,
                )
                detections[frame_index] = evidence.detection
                frame_records.append(
                    PrecisionFrameInferenceRecord(
                        frame_index=frame_index,
                        status=PrecisionFrameStatus.PREDICTED,
                        frame_timestamp_s=frame_timestamp,
                        wrist_timestamp_s=wrist_timestamp,
                        geometry_timestamp_s=geometry_timestamp,
                        predictor_latency_s=predictor_latency,
                        total_latency_s=time.perf_counter() - frame_started,
                        evidence=evidence,
                    )
                )
            except Exception as error:
                frame_records.append(
                    PrecisionFrameInferenceRecord(
                        frame_index=frame_index,
                        status=PrecisionFrameStatus.ERROR,
                        frame_timestamp_s=frame_timestamp,
                        wrist_timestamp_s=wrist_timestamp,
                        geometry_timestamp_s=geometry_timestamp,
                        predictor_latency_s=predictor_latency,
                        total_latency_s=time.perf_counter() - frame_started,
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                )
                call = self._failed_call(
                    window=window,
                    started=started,
                    frame_records=frame_records,
                    error=error,
                )
                raise PrecisionDetectionProviderError(call) from error

        call = PrecisionDetectionProviderCall(
            call_index=len(self._records),
            observation_timestamp_s=float(window.timestamp_s),
            success=True,
            detections_count=sum(detection is not None for detection in detections),
            frame_records=tuple(frame_records),
            provider_identity_sha256=self.identity.sha256,
            total_latency_s=time.perf_counter() - started,
        )
        self._records.append(call)
        return tuple(detections)


__all__ = [
    "PRECISION_DETECTION_EXECUTION_MODE",
    "PRECISION_DETECTION_PROVIDER_VERSION",
    "PRECISION_FRAME_ORDER",
    "PRECISION_FRAME_PREDICTOR_VERSION",
    "PRECISION_IMAGE_INPUT_SEMANTICS",
    "PRECISION_STRUCTURED_STATE_INPUT_SEMANTICS",
    "PrecisionDetectionProvider",
    "PrecisionDetectionProviderCall",
    "PrecisionDetectionProviderConfig",
    "PrecisionDetectionProviderError",
    "PrecisionDetectionProviderIdentity",
    "PrecisionFrameInferenceRecord",
    "PrecisionFramePredictor",
    "PrecisionFrameStatus",
    "PrecisionGeometricMotionInput",
    "PrecisionGeometricMotionProvider",
    "PrecisionPredictorIdentity",
    "TorchPrecisionFramePredictor",
    "TorchPrecisionFramePredictorConfig",
]
