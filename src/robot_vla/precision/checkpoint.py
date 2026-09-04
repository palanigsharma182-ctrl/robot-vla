"""Precision U-Net 的版本化、weights-only checkpoint 契约。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from robot_vla.precision.contracts import PRECISION_MODEL_ARCH, PrecisionMotionSpec
from robot_vla.precision.model import PrecisionThreeHeadUNet, PrecisionUNetConfig
from robot_vla.precision.provider import (
    TorchPrecisionFramePredictor,
    TorchPrecisionFramePredictorConfig,
)

PRECISION_CHECKPOINT_FORMAT = "precision-unet-weights-only/v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class PrecisionCheckpointRole(str, Enum):
    """区分仅验证训练链路的权重与可进入正式评估的权重。"""

    SYNTHETIC_DEBUG = "synthetic-debug"
    FORMAL_TRAINING = "formal-training"


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} 必须是 64 位小写 SHA-256")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical mapping 的 key 必须是字符串")
        return {key: _jsonable(item) for key, item in value.items()}
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} keys 漂移: missing={missing}, extra={extra}")


def precision_parameter_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """对名称、dtype、shape 和原始 tensor bytes 生成稳定摘要。"""

    if not isinstance(state, Mapping) or not state:
        raise TypeError("Precision model state 必须是非空 Mapping")
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not name:
            raise TypeError("Precision model state name 必须是非空字符串")
        if not isinstance(value, torch.Tensor):
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
        raw = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True)
class PrecisionCheckpointProvenance:
    """Checkpoint 训练来源的最小不可变身份，不包含路径或原始数据。"""

    role: PrecisionCheckpointRole
    data_identity_sha256: str
    training_config_sha256: str
    source_tree_sha256: str
    seed: int
    examples_seen: int
    optimizer_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.role, PrecisionCheckpointRole):
            raise TypeError("Precision checkpoint role 必须是 PrecisionCheckpointRole")
        for value, name in (
            (self.data_identity_sha256, "data_identity_sha256"),
            (self.training_config_sha256, "training_config_sha256"),
            (self.source_tree_sha256, "source_tree_sha256"),
        ):
            _require_sha256(value, name)
        for value, name in (
            (self.seed, "seed"),
            (self.examples_seen, "examples_seen"),
            (self.optimizer_steps, "optimizer_steps"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} 必须是非负整数")
        if self.examples_seen == 0 or self.optimizer_steps == 0:
            raise ValueError("Precision checkpoint 必须来自至少一个 example/optimizer step")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class PrecisionCheckpointReceipt:
    format_version: str
    checkpoint_sha256: str
    parameter_state_sha256: str
    model_config_sha256: str
    provenance_sha256: str

    def __post_init__(self) -> None:
        if self.format_version != PRECISION_CHECKPOINT_FORMAT:
            raise ValueError("Precision checkpoint format 漂移")
        for value, name in (
            (self.checkpoint_sha256, "checkpoint_sha256"),
            (self.parameter_state_sha256, "parameter_state_sha256"),
            (self.model_config_sha256, "model_config_sha256"),
            (self.provenance_sha256, "provenance_sha256"),
        ):
            _require_sha256(value, name)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class LoadedPrecisionCheckpoint:
    model: PrecisionThreeHeadUNet
    provenance: PrecisionCheckpointProvenance
    receipt: PrecisionCheckpointReceipt


@dataclass(frozen=True)
class LoadedPrecisionPredictor:
    predictor: TorchPrecisionFramePredictor
    provenance: PrecisionCheckpointProvenance
    receipt: PrecisionCheckpointReceipt


def _model_config_from_payload(value: Any) -> PrecisionUNetConfig:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError("Precision checkpoint model_config 必须是字符串 key Mapping")
    payload = dict(value)
    expected = {field.name for field in fields(PrecisionUNetConfig)}
    _require_exact_keys(payload, expected, "model_config")
    motion_value = payload.get("motion_spec")
    if not isinstance(motion_value, Mapping) or any(
        not isinstance(key, str) for key in motion_value
    ):
        raise TypeError("model_config.motion_spec 必须是字符串 key Mapping")
    motion_payload = dict(motion_value)
    motion_expected = {field.name for field in fields(PrecisionMotionSpec)}
    _require_exact_keys(motion_payload, motion_expected, "model_config.motion_spec")
    for name in ("components", "step_limits", "residual_limits"):
        value_for_tuple = motion_payload[name]
        if not isinstance(value_for_tuple, (tuple, list)):
            raise TypeError(f"model_config.motion_spec.{name} 必须是序列")
        motion_payload[name] = tuple(value_for_tuple)
    payload["motion_spec"] = PrecisionMotionSpec(**motion_payload)
    for name in ("encoder_channels", "keypoint_names", "mask_names"):
        value_for_tuple = payload[name]
        if not isinstance(value_for_tuple, (tuple, list)):
            raise TypeError(f"model_config.{name} 必须是序列")
        payload[name] = tuple(value_for_tuple)
    return PrecisionUNetConfig(**payload)


def _provenance_from_payload(value: Any) -> PrecisionCheckpointProvenance:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError("Precision checkpoint provenance 必须是字符串 key Mapping")
    payload = dict(value)
    expected = {field.name for field in fields(PrecisionCheckpointProvenance)}
    _require_exact_keys(payload, expected, "provenance")
    try:
        payload["role"] = PrecisionCheckpointRole(payload["role"])
    except (TypeError, ValueError) as error:
        raise ValueError("Precision checkpoint provenance role 无效") from error
    return PrecisionCheckpointProvenance(**payload)


def save_precision_checkpoint(
    path: str | Path,
    model: PrecisionThreeHeadUNet,
    provenance: PrecisionCheckpointProvenance,
) -> PrecisionCheckpointReceipt:
    """原子保存 weights-only checkpoint；已有目标一律拒绝覆盖。"""

    if not isinstance(model, PrecisionThreeHeadUNet):
        raise TypeError("Precision checkpoint 只接受 PrecisionThreeHeadUNet")
    if not isinstance(provenance, PrecisionCheckpointProvenance):
        raise TypeError("provenance 必须是 PrecisionCheckpointProvenance")
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Precision checkpoint 已存在，拒绝覆盖: {target}")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    model_config = _jsonable(model.config)
    model_config_sha256 = _canonical_sha256(model_config)
    model_state = {
        name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()
    }
    parameter_state_sha256 = precision_parameter_state_sha256(model_state)
    payload = {
        "format_version": PRECISION_CHECKPOINT_FORMAT,
        "model_arch": PRECISION_MODEL_ARCH,
        "model_config": model_config,
        "model_config_sha256": model_config_sha256,
        "parameter_state_sha256": parameter_state_sha256,
        "provenance": provenance.to_dict(),
        "provenance_sha256": provenance.sha256,
        "model_state": model_state,
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        torch.save(payload, temporary_path)
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return PrecisionCheckpointReceipt(
        format_version=PRECISION_CHECKPOINT_FORMAT,
        checkpoint_sha256=_file_sha256(target),
        parameter_state_sha256=parameter_state_sha256,
        model_config_sha256=model_config_sha256,
        provenance_sha256=provenance.sha256,
    )


def load_precision_checkpoint(
    path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    expected_provenance_sha256: str | None = None,
    expected_role: PrecisionCheckpointRole | None = None,
) -> LoadedPrecisionCheckpoint:
    """先验证文件身份，再以 ``weights_only=True`` 严格加载所有字段。"""

    _require_sha256(expected_checkpoint_sha256, "expected_checkpoint_sha256")
    if expected_provenance_sha256 is not None:
        _require_sha256(expected_provenance_sha256, "expected_provenance_sha256")
    if expected_role is not None and not isinstance(expected_role, PrecisionCheckpointRole):
        raise TypeError("expected_role 必须是 PrecisionCheckpointRole")
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Precision checkpoint 不存在: {checkpoint_path}")
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise RuntimeError("Precision checkpoint 文件 SHA-256 漂移")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or any(not isinstance(key, str) for key in payload):
        raise TypeError("Precision checkpoint payload 必须是字符串 key Mapping")
    payload = dict(payload)
    expected_keys = {
        "format_version",
        "model_arch",
        "model_config",
        "model_config_sha256",
        "parameter_state_sha256",
        "provenance",
        "provenance_sha256",
        "model_state",
    }
    _require_exact_keys(payload, expected_keys, "checkpoint")
    if payload["format_version"] != PRECISION_CHECKPOINT_FORMAT:
        raise ValueError("Precision checkpoint format 漂移")
    if payload["model_arch"] != PRECISION_MODEL_ARCH:
        raise ValueError("Precision checkpoint model arch 漂移")

    config = _model_config_from_payload(payload["model_config"])
    model_config_sha256 = _canonical_sha256(config)
    _require_sha256(payload["model_config_sha256"], "model_config_sha256")
    if payload["model_config_sha256"] != model_config_sha256:
        raise RuntimeError("Precision checkpoint model config SHA-256 漂移")

    provenance = _provenance_from_payload(payload["provenance"])
    _require_sha256(payload["provenance_sha256"], "provenance_sha256")
    if payload["provenance_sha256"] != provenance.sha256:
        raise RuntimeError("Precision checkpoint provenance SHA-256 漂移")
    if expected_provenance_sha256 is not None and provenance.sha256 != expected_provenance_sha256:
        raise RuntimeError("Precision checkpoint provenance 与预期不一致")
    if expected_role is not None and provenance.role != expected_role:
        raise RuntimeError("Precision checkpoint role 与预期不一致")

    state = payload["model_state"]
    if not isinstance(state, Mapping) or any(not isinstance(key, str) for key in state):
        raise TypeError("Precision checkpoint model_state 必须是字符串 key Mapping")
    _require_sha256(payload["parameter_state_sha256"], "parameter_state_sha256")
    parameter_state_sha256 = precision_parameter_state_sha256(state)
    if payload["parameter_state_sha256"] != parameter_state_sha256:
        raise RuntimeError("Precision checkpoint parameter state SHA-256 漂移")
    model = PrecisionThreeHeadUNet(config)
    model.load_state_dict(dict(state), strict=True)
    if precision_parameter_state_sha256(model.state_dict()) != parameter_state_sha256:
        raise RuntimeError("Precision checkpoint load 后 parameter state 漂移")
    receipt = PrecisionCheckpointReceipt(
        format_version=PRECISION_CHECKPOINT_FORMAT,
        checkpoint_sha256=checkpoint_sha256,
        parameter_state_sha256=parameter_state_sha256,
        model_config_sha256=model_config_sha256,
        provenance_sha256=provenance.sha256,
    )
    return LoadedPrecisionCheckpoint(model=model, provenance=provenance, receipt=receipt)


def load_torch_precision_frame_predictor(
    path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    expected_role: PrecisionCheckpointRole,
    predictor_config: TorchPrecisionFramePredictorConfig | None = None,
    expected_provenance_sha256: str | None = None,
) -> LoadedPrecisionPredictor:
    """从已验证 checkpoint 创建 frozen/eval Predictor，不接受裸 state dict。"""

    loaded = load_precision_checkpoint(
        path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_provenance_sha256=expected_provenance_sha256,
        expected_role=expected_role,
    )
    predictor = TorchPrecisionFramePredictor(
        loaded.model,
        checkpoint_sha256=loaded.receipt.checkpoint_sha256,
        config=predictor_config,
    )
    if predictor.identity.parameter_state_sha256 != loaded.receipt.parameter_state_sha256:
        raise RuntimeError("Precision predictor parameter identity 与 checkpoint 不一致")
    if predictor.identity.model_config_sha256 != loaded.receipt.model_config_sha256:
        raise RuntimeError("Precision predictor config identity 与 checkpoint 不一致")
    return LoadedPrecisionPredictor(
        predictor=predictor,
        provenance=loaded.provenance,
        receipt=loaded.receipt,
    )


__all__ = [
    "PRECISION_CHECKPOINT_FORMAT",
    "LoadedPrecisionCheckpoint",
    "LoadedPrecisionPredictor",
    "PrecisionCheckpointProvenance",
    "PrecisionCheckpointReceipt",
    "PrecisionCheckpointRole",
    "load_precision_checkpoint",
    "load_torch_precision_frame_predictor",
    "precision_parameter_state_sha256",
    "save_precision_checkpoint",
]
