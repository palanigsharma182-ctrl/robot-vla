"""E018-P1 G2B 的冻结标量 covariance 校准与稳定 provider identity。

该模块只做 qualification-only 后处理：不修改 checkpoint 权重、visibility/write
threshold，也不授予 Memory 或 actuator 权限。逐帧实际相机位姿单独进入 capture
record；它不属于稳定 calibration/provider provenance。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from robot_vla.observation import validate_se3
from robot_vla.precision.object_observability import ObjectWriteEvidence

SCALAR_COVARIANCE_CALIBRATION_METHOD = (
    "split-conformal-xy-mahalanobis-scalar/alpha-0.05-chi2-5.991/v1"
)
STABLE_CAMERA_CALIBRATION_IDENTITY_VERSION = "e018-p1-g2b-stable-camera-calibration-identity/v1"
CALIBRATED_PROVIDER_IDENTITY_VERSION = "e018-p1-g2b-calibrated-provider-identity/v1"
CALIBRATED_PROVIDER_EXECUTION_MODE = "qualification-only/no-memory/no-actuation/v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    dtype = str(array.dtype).encode("ascii")
    digest.update(len(dtype).to_bytes(2, "big"))
    digest.update(dtype)
    digest.update(len(array.shape).to_bytes(2, "big"))
    for dimension in array.shape:
        digest.update(int(dimension).to_bytes(8, "big", signed=False))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} 必须是 64 位小写 SHA-256")


def _intrinsic(value: np.ndarray) -> np.ndarray:
    intrinsic = np.asarray(value, dtype=np.float64)
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
        or not np.allclose(intrinsic[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-8)
    ):
        raise ValueError("intrinsic_cv 必须是有限有效的 OpenCV [3,3]")
    return intrinsic


@dataclass(frozen=True)
class ScalarCovarianceCalibration:
    """由冻结 validation ledger 得到的单个 covariance scale。"""

    scale_factor: float
    support_count: int
    order_statistic_k: int
    quantile_score: float
    scoring_ledger_sha256: str
    config_sha256: str
    validation_data_identity_sha256: str
    source_identity_sha256: str
    alpha: float = 0.05
    chi_square_threshold: float = 5.991
    method: str = SCALAR_COVARIANCE_CALIBRATION_METHOD

    def __post_init__(self) -> None:
        for name in (
            "scoring_ledger_sha256",
            "config_sha256",
            "validation_data_identity_sha256",
            "source_identity_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            not math.isfinite(self.scale_factor)
            or self.scale_factor < 1.0
            or not math.isfinite(self.quantile_score)
            or self.quantile_score < 0.0
        ):
            raise ValueError("covariance calibration scale/quantile 必须有限且 scale>=1")
        if (
            not isinstance(self.support_count, int)
            or isinstance(self.support_count, bool)
            or self.support_count <= 0
            or not isinstance(self.order_statistic_k, int)
            or isinstance(self.order_statistic_k, bool)
            or not 1 <= self.order_statistic_k <= self.support_count
        ):
            raise ValueError("covariance calibration support/k 无效")
        if self.alpha != 0.05 or self.chi_square_threshold != 5.991:
            raise ValueError("G2B calibration alpha/chi-square threshold 漂移")
        if self.method != SCALAR_COVARIANCE_CALIBRATION_METHOD:
            raise ValueError("G2B calibration method 漂移")
        expected_k = math.ceil((self.support_count + 1) * (1.0 - self.alpha))
        expected_scale = max(1.0, self.quantile_score / self.chi_square_threshold)
        if self.order_statistic_k != expected_k or not math.isclose(
            self.scale_factor, expected_scale, rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise ValueError("calibration k/scale 与声明的分位数方法不一致")

    @property
    def sigma_scale_factor(self) -> float:
        return float(math.sqrt(self.scale_factor))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sigma_scale_factor"] = self.sigma_scale_factor
        payload["identity_sha256"] = canonical_sha256(payload)
        return payload

    @property
    def identity_sha256(self) -> str:
        return str(self.to_dict()["identity_sha256"])

    def calibrate_sigma_xy_px(self, raw_sigma_xy_px: np.ndarray) -> np.ndarray:
        sigma = np.asarray(raw_sigma_xy_px, dtype=np.float64)
        if sigma.shape != (2,) or not np.isfinite(sigma).all() or np.any(sigma < 0.0):
            raise ValueError("raw_sigma_xy_px 必须是有限非负 [2]")
        calibrated = sigma * self.sigma_scale_factor
        if not np.isfinite(calibrated).all():
            raise ValueError("calibrated sigma 非有限")
        return calibrated

    def calibrate_covariance(self, raw_covariance: np.ndarray) -> np.ndarray:
        covariance = np.asarray(raw_covariance, dtype=np.float64)
        if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
            raise ValueError("raw covariance 必须是有限 [3,3]")
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
            raise ValueError("raw covariance 必须对称")
        covariance = (covariance + covariance.T) * 0.5
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if float(eigenvalues.min()) < -1e-12:
            raise ValueError("raw covariance 必须为 PSD")
        if float(eigenvalues.min()) < 0.0:
            # 缩放前消除容差内的负舍入误差，避免放大成实质性负方差。
            covariance = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
            covariance = (covariance + covariance.T) * 0.5
        calibrated = covariance * self.scale_factor
        if not np.isfinite(calibrated).all():
            raise ValueError("calibrated covariance 非有限")
        return calibrated


def build_stable_camera_calibration_identity(
    *,
    camera_uid: str,
    primitive_id: str,
    intrinsic_cv: np.ndarray,
    actual_base_from_camera_cv: np.ndarray,
    covariance_calibration_identity_sha256: str,
    source_training_camera: str,
    target_camera: str,
    frame_convention: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """返回稳定 identity 与逐帧 pose record；二者有意分离。"""

    for name, value in (
        ("camera_uid", camera_uid),
        ("primitive_id", primitive_id),
        ("source_training_camera", source_training_camera),
        ("target_camera", target_camera),
        ("frame_convention", frame_convention),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} 必须是非空字符串")
    _require_sha256(
        covariance_calibration_identity_sha256,
        "covariance_calibration_identity_sha256",
    )
    intrinsic = _intrinsic(intrinsic_cv)
    actual_pose = validate_se3(
        actual_base_from_camera_cv,
        "actual_base_from_camera_cv",
    ).astype(np.float64, copy=False)
    identity = {
        "version": STABLE_CAMERA_CALIBRATION_IDENTITY_VERSION,
        "camera_uid": camera_uid,
        "primitive_id": primitive_id,
        "frame_convention": frame_convention,
        "intrinsic_cv_sha256": array_sha256(intrinsic),
        "covariance_calibration_identity_sha256": (covariance_calibration_identity_sha256),
        "calibration_method": SCALAR_COVARIANCE_CALIBRATION_METHOD,
        "source_training_camera": source_training_camera,
        "target_camera": target_camera,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    pose_record = {
        "actual_base_from_camera_cv": actual_pose.tolist(),
        "actual_base_from_camera_cv_sha256": array_sha256(actual_pose),
        "excluded_from_stable_identity": True,
    }
    return identity, pose_record


def build_calibrated_provider_identity(
    *,
    checkpoint_sha256: str,
    checkpoint_parameter_sha256: str,
    checkpoint_provenance_sha256: str,
    model_config_sha256: str,
    proprio_stats_sha256: str,
    proprio_normalizer_sha256: str,
    finger_force_stats_sha256: str,
    finger_force_normalizer_sha256: str,
    adapter_config_sha256: str,
    stable_camera_calibration_identity_sha256: str,
    covariance_calibration_identity_sha256: str,
    primitive_id: str,
    geometric_motion_provider_id: str,
    source_training_camera: str,
    target_camera: str,
    frame_convention: str,
) -> dict[str, Any]:
    """构造只含静态 provenance 的 provider identity。"""

    sha_fields = {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_parameter_sha256": checkpoint_parameter_sha256,
        "checkpoint_provenance_sha256": checkpoint_provenance_sha256,
        "model_config_sha256": model_config_sha256,
        "proprio_stats_sha256": proprio_stats_sha256,
        "proprio_normalizer_sha256": proprio_normalizer_sha256,
        "finger_force_stats_sha256": finger_force_stats_sha256,
        "finger_force_normalizer_sha256": finger_force_normalizer_sha256,
        "adapter_config_sha256": adapter_config_sha256,
        "stable_camera_calibration_identity_sha256": (stable_camera_calibration_identity_sha256),
        "covariance_calibration_identity_sha256": (covariance_calibration_identity_sha256),
    }
    for name, value in sha_fields.items():
        _require_sha256(value, name)
    for name, value in (
        ("primitive_id", primitive_id),
        ("geometric_motion_provider_id", geometric_motion_provider_id),
        ("source_training_camera", source_training_camera),
        ("target_camera", target_camera),
        ("frame_convention", frame_convention),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} 必须是非空字符串")
    identity = {
        "version": CALIBRATED_PROVIDER_IDENTITY_VERSION,
        **sha_fields,
        "primitive_id": primitive_id,
        "geometric_motion_provider_id": geometric_motion_provider_id,
        "source_training_camera": source_training_camera,
        "target_camera": target_camera,
        "frame_convention": frame_convention,
        "execution_mode": CALIBRATED_PROVIDER_EXECUTION_MODE,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def build_calibrated_object_write_evidence(
    *,
    calibration: ScalarCovarianceCalibration,
    raw_sigma_xy_px: np.ndarray,
    visibility_probability: float,
    projection_validity_probability: float,
    object_mask_probability: float,
    goal_mask_probability: float,
    normalized_entropy: float,
    geometry_valid: bool,
    min_object_mask_probability: float,
    max_goal_mask_probability: float,
) -> tuple[ObjectWriteEvidence, np.ndarray]:
    """把同一校准 sigma 送入 write evidence，避免仅离线放大 covariance。"""

    calibrated_sigma = calibration.calibrate_sigma_xy_px(raw_sigma_xy_px)
    evidence = ObjectWriteEvidence(
        visibility_probability=visibility_probability,
        projection_validity_probability=projection_validity_probability,
        object_mask_probability=object_mask_probability,
        goal_mask_probability=goal_mask_probability,
        normalized_entropy=normalized_entropy,
        radial_sigma_px=float(np.linalg.norm(calibrated_sigma)),
        geometry_valid=geometry_valid,
        min_object_mask_probability=min_object_mask_probability,
        max_goal_mask_probability=max_goal_mask_probability,
    )
    return evidence, calibrated_sigma


__all__ = [
    "CALIBRATED_PROVIDER_EXECUTION_MODE",
    "CALIBRATED_PROVIDER_IDENTITY_VERSION",
    "SCALAR_COVARIANCE_CALIBRATION_METHOD",
    "STABLE_CAMERA_CALIBRATION_IDENTITY_VERSION",
    "ScalarCovarianceCalibration",
    "array_sha256",
    "build_calibrated_object_write_evidence",
    "build_calibrated_provider_identity",
    "build_stable_camera_calibration_identity",
    "canonical_sha256",
]
