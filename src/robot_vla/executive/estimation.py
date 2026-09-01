"""Observation V2 到可部署 Executive 状态的确定性时序适配层。

视觉网络只负责输出每个历史时刻的腕部 object/goal 像素与置信度。本模块使用与该图像
同时记录的相机位姿和时间戳完成 base-frame 反投影、四时刻速度拟合与 fail-closed 门禁。
它不读取仿真器 object pose、is_grasped 或任务成功标志。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robot_vla.contracts import OBSERVATION_HISTORY_LENGTH, RobotSpec
from robot_vla.executive.contracts import (
    EXECUTIVE_PREDICATES,
    DeployableStateEstimate,
    ExecutiveSnapshot,
    ModalityStatus,
    PredicateEvidence,
    PredicateSource,
    ScalarStateEstimate,
    SpatialTrackEstimate,
)
from robot_vla.observation import (
    OBSERVATION_MODALITIES,
    ObservationV2Window,
    rotation_6d_to_matrix,
)
from robot_vla.precision.geometry import normalized_uv_to_base_z_plane

WRIST_KEYPOINT_DETECTION_VERSION = "qwen-vla-wrist-keypoint-detection/v1"
DEPLOYABLE_STATE_ESTIMATOR_VERSION = "qwen-vla-deployable-state-estimator/v1"


def _probability(value: float, name: str) -> float:
    candidate = float(value)
    if not math.isfinite(candidate) or not 0.0 <= candidate <= 1.0:
        raise ValueError(f"{name} 必须是 [0,1] 内的有限概率")
    return candidate


def _positive(value: float, name: str) -> float:
    candidate = float(value)
    if not math.isfinite(candidate) or candidate <= 0.0:
        raise ValueError(f"{name} 必须是有限正数")
    return candidate


def _finite(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是有限数值")
    candidate = float(value)
    if not math.isfinite(candidate):
        raise ValueError(f"{name} 必须是有限数值")
    return candidate


def _normalized_uv_or_none(
    value: tuple[float, float] | None,
    name: str,
) -> tuple[float, float] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"{name} 必须是二元素 normalized UV 或 None")
    uv = (float(value[0]), float(value[1]))
    if any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in uv):
        raise ValueError(f"{name} 必须位于 [0,1]")
    return uv


@dataclass(frozen=True)
class WristKeypointDetection:
    """一个 wrist RGB 时刻的部署侧关键点输出，不携带三维 GT。"""

    timestamp_s: float
    object_normalized_uv: tuple[float, float] | None
    goal_normalized_uv: tuple[float, float] | None
    object_confidence: float
    goal_confidence: float
    source: PredicateSource = PredicateSource.DEPLOYABLE_ESTIMATOR
    version: str = WRIST_KEYPOINT_DETECTION_VERSION

    def __post_init__(self) -> None:
        if self.version != WRIST_KEYPOINT_DETECTION_VERSION:
            raise ValueError(
                f"wrist keypoint detection version 必须为 "
                f"{WRIST_KEYPOINT_DETECTION_VERSION}"
            )
        timestamp = _finite(self.timestamp_s, "detection timestamp_s")
        if timestamp < 0.0:
            raise ValueError("detection timestamp_s 必须非负")
        object_uv = _normalized_uv_or_none(
            self.object_normalized_uv,
            "object_normalized_uv",
        )
        goal_uv = _normalized_uv_or_none(
            self.goal_normalized_uv,
            "goal_normalized_uv",
        )
        object_confidence = _probability(
            self.object_confidence,
            "object_confidence",
        )
        goal_confidence = _probability(self.goal_confidence, "goal_confidence")
        if object_uv is None and object_confidence != 0.0:
            raise ValueError("object keypoint 缺失时 confidence 必须为 0")
        if goal_uv is None and goal_confidence != 0.0:
            raise ValueError("goal keypoint 缺失时 confidence 必须为 0")
        if self.source not in {
            PredicateSource.DEPLOYABLE_ESTIMATOR,
            PredicateSource.EVALUATOR_GT,
        }:
            raise ValueError("keypoint source 只能是 deployable estimator 或 evaluator GT")
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "object_normalized_uv", object_uv)
        object.__setattr__(self, "goal_normalized_uv", goal_uv)
        object.__setattr__(self, "object_confidence", object_confidence)
        object.__setattr__(self, "goal_confidence", goal_confidence)


@dataclass(frozen=True)
class DeployableStateEstimatorConfig:
    """需要由 E013 shadow/calibration 冻结的几何与时序门禁。

    除结构性的四时刻和至少两点速度拟合外，本类没有经验默认阈值，避免把测试参数误当成
    正式标定结果。
    """

    object_plane_base_z_m: float
    goal_plane_base_z_m: float
    min_detection_confidence: float
    max_detection_timestamp_error_s: float
    max_camera_image_skew_s: float
    max_track_age_s: float
    max_track_innovation_m: float
    max_track_speed_m_s: float
    min_track_points: int = 2
    history_length: int = OBSERVATION_HISTORY_LENGTH
    version: str = DEPLOYABLE_STATE_ESTIMATOR_VERSION

    def __post_init__(self) -> None:
        _finite(self.object_plane_base_z_m, "object_plane_base_z_m")
        _finite(self.goal_plane_base_z_m, "goal_plane_base_z_m")
        detection_confidence = _probability(
            self.min_detection_confidence,
            "min_detection_confidence",
        )
        if detection_confidence == 0.0:
            raise ValueError("min_detection_confidence 必须大于 0")
        for value, name in (
            (self.max_detection_timestamp_error_s, "max_detection_timestamp_error_s"),
            (self.max_camera_image_skew_s, "max_camera_image_skew_s"),
            (self.max_track_age_s, "max_track_age_s"),
            (self.max_track_innovation_m, "max_track_innovation_m"),
            (self.max_track_speed_m_s, "max_track_speed_m_s"),
        ):
            _positive(value, name)
        if (
            not isinstance(self.min_track_points, int)
            or isinstance(self.min_track_points, bool)
            or not 2 <= self.min_track_points <= OBSERVATION_HISTORY_LENGTH
        ):
            raise ValueError("min_track_points 必须是 [2,4] 内的整数")
        if self.history_length != OBSERVATION_HISTORY_LENGTH:
            raise ValueError(
                f"state estimator history_length 必须为 {OBSERVATION_HISTORY_LENGTH}"
            )
        if self.version != DEPLOYABLE_STATE_ESTIMATOR_VERSION:
            raise ValueError(
                f"state estimator version 必须为 {DEPLOYABLE_STATE_ESTIMATOR_VERSION}"
            )


@dataclass(frozen=True)
class DeployableOutcomeEvidence:
    """独立、已标定 monitor 给出的抓取/支撑/稳定证据。

    本层不根据未经标定的 F_L/F_R 阈值伪造这些状态。缺失字段会保持 invalid，并由后续
    Executive Predicate 门禁阻止关键动作。
    """

    grasp: ScalarStateEstimate | None = None
    support_contact: ScalarStateEstimate | None = None
    settled: ScalarStateEstimate | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.grasp, "grasp"),
            (self.support_contact, "support_contact"),
            (self.settled, "settled"),
        ):
            if value is not None and not isinstance(value, ScalarStateEstimate):
                raise TypeError(f"{name} 必须是 ScalarStateEstimate 或 None")
            if value is not None and value.source == PredicateSource.EVALUATOR_GT:
                raise ValueError(f"部署侧 {name} evidence 禁止使用 evaluator GT")


@dataclass(frozen=True)
class TemporalTrackDiagnostics:
    candidate_count: int
    accepted_count: int
    latest_age_s: float | None
    innovation_m: float | None
    speed_m_s: float | None
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class DeployableStateEstimatorResult:
    state_estimate: DeployableStateEstimate
    object_diagnostics: TemporalTrackDiagnostics
    goal_diagnostics: TemporalTrackDiagnostics
    estimator_version: str = DEPLOYABLE_STATE_ESTIMATOR_VERSION


@dataclass(frozen=True)
class DeployablePredicateThresholds:
    """State Estimate 到 Predicate 的待标定置信度与 freshness 门禁。"""

    track_confidence_min: float
    grasp_candidate_min: float
    grasp_verified_min: float
    support_contact_min: float
    support_verified_min: float
    settled_min: float
    max_track_age_s: float
    max_scalar_age_s: float

    def __post_init__(self) -> None:
        for value, name in (
            (self.track_confidence_min, "track_confidence_min"),
            (self.grasp_candidate_min, "grasp_candidate_min"),
            (self.grasp_verified_min, "grasp_verified_min"),
            (self.support_contact_min, "support_contact_min"),
            (self.support_verified_min, "support_verified_min"),
            (self.settled_min, "settled_min"),
        ):
            _probability(value, name)
        _positive(self.max_track_age_s, "max_track_age_s")
        _positive(self.max_scalar_age_s, "max_scalar_age_s")
        if self.grasp_verified_min < self.grasp_candidate_min:
            raise ValueError("grasp_verified_min 不能低于 grasp_candidate_min")
        if self.support_verified_min < self.support_contact_min:
            raise ValueError("support_verified_min 不能低于 support_contact_min")


@dataclass(frozen=True)
class _TrackPoint:
    timestamp_s: float
    position_base_m: np.ndarray
    confidence: float


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _camera_transform(window: ObservationV2Window, index: int) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_6d_to_matrix(window.wrist_rotation_6d[index])
    transform[:3, 3] = window.wrist_position[index]
    return transform


def _linear_fit(
    points: list[_TrackPoint],
    reference_timestamp_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.asarray(
        [point.timestamp_s - reference_timestamp_s for point in points],
        dtype=np.float64,
    )
    positions = np.stack([point.position_base_m for point in points]).astype(np.float64)
    weights = np.sqrt(
        np.asarray([point.confidence for point in points], dtype=np.float64)
    )
    design = np.column_stack((np.ones(len(points), dtype=np.float64), timestamps))
    weighted_design = design * weights[:, None]
    weighted_positions = positions * weights[:, None]
    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_design,
        weighted_positions,
        rcond=None,
    )
    if rank < 2:
        raise ValueError("track timestamps 退化，无法估计速度")
    return coefficients[0], coefficients[1]


def _invalid_scalar() -> ScalarStateEstimate:
    return ScalarStateEstimate(
        confidence=0.0,
        valid=False,
        age_s=None,
        source=PredicateSource.DEPLOYABLE_ESTIMATOR,
    )


class FourFrameDeployableStateEstimator:
    """把严格对齐的四时刻 wrist detections 融合为 base-frame 状态。"""

    def __init__(
        self,
        spec: RobotSpec,
        wrist_intrinsic_cv: np.ndarray,
        config: DeployableStateEstimatorConfig,
    ) -> None:
        intrinsic = np.asarray(wrist_intrinsic_cv)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("wrist_intrinsic_cv 必须是有限 [3,3]")
        if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
            raise ValueError("wrist_intrinsic_cv fx/fy 必须为正数")
        if not np.allclose(intrinsic[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-7):
            raise ValueError("wrist_intrinsic_cv 最后一行必须为 [0,0,1]")
        self.spec = spec
        self.wrist_intrinsic_cv = intrinsic.astype(np.float64, copy=True)
        self.config = config

    def _collect_points(
        self,
        window: ObservationV2Window,
        detections: tuple[WristKeypointDetection | None, ...],
        current_timestamp_s: float,
        *,
        target: str,
    ) -> tuple[list[_TrackPoint], int, list[str]]:
        if target not in {"object", "goal"}:
            raise ValueError("target 必须是 object 或 goal")
        wrist_rgb_index = OBSERVATION_MODALITIES.index("rgb_wrist")
        camera_pose_index = OBSERVATION_MODALITIES.index("wrist_camera_pose")
        image_size_hw = tuple(int(value) for value in window.rgb_wrist.shape[1:3])
        plane_z = (
            self.config.object_plane_base_z_m
            if target == "object"
            else self.config.goal_plane_base_z_m
        )
        points: list[_TrackPoint] = []
        reasons: list[str] = []
        candidate_count = 0

        for index, detection in enumerate(detections):
            if not window.history_valid[index]:
                if detection is not None:
                    raise ValueError("padding history 不得携带 keypoint detection")
                continue
            if detection is None:
                _append_reason(reasons, "detection_missing")
                continue
            if detection.source == PredicateSource.EVALUATOR_GT:
                raise ValueError("部署状态估计禁止使用 evaluator GT keypoint")

            uv = (
                detection.object_normalized_uv
                if target == "object"
                else detection.goal_normalized_uv
            )
            confidence = (
                detection.object_confidence
                if target == "object"
                else detection.goal_confidence
            )
            if uv is None:
                _append_reason(reasons, "keypoint_missing")
                continue
            candidate_count += 1
            if confidence < self.config.min_detection_confidence:
                _append_reason(reasons, "detection_below_confidence")
                continue
            if not window.modality_valid[index, wrist_rgb_index]:
                _append_reason(reasons, "wrist_rgb_invalid")
                continue
            if not window.modality_valid[index, camera_pose_index]:
                _append_reason(reasons, "wrist_camera_pose_invalid")
                continue

            image_timestamp = (
                current_timestamp_s - window.modality_age_s[index, wrist_rgb_index]
            )
            camera_timestamp = (
                current_timestamp_s - window.modality_age_s[index, camera_pose_index]
            )
            if (
                abs(detection.timestamp_s - image_timestamp)
                > self.config.max_detection_timestamp_error_s
            ):
                _append_reason(reasons, "detection_timestamp_mismatch")
                continue
            if (
                abs(image_timestamp - camera_timestamp)
                > self.config.max_camera_image_skew_s
            ):
                _append_reason(reasons, "camera_image_timestamp_skew")
                continue

            try:
                projection = normalized_uv_to_base_z_plane(
                    np.asarray(uv, dtype=np.float32),
                    self.wrist_intrinsic_cv,
                    _camera_transform(window, index),
                    image_size_hw,
                    plane_base_z_m=plane_z,
                )
            except ValueError:
                _append_reason(reasons, "projection_invalid")
                continue
            points.append(
                _TrackPoint(
                    timestamp_s=float(image_timestamp),
                    position_base_m=projection.point_base_m,
                    confidence=float(confidence),
                )
            )
        return points, candidate_count, reasons

    def _fit_track(
        self,
        points: list[_TrackPoint],
        candidate_count: int,
        reasons: list[str],
        current_timestamp_s: float,
    ) -> tuple[SpatialTrackEstimate, TemporalTrackDiagnostics]:
        ordered = sorted(points, key=lambda point: point.timestamp_s)
        if len(ordered) < self.config.min_track_points:
            _append_reason(reasons, "track_insufficient_points")
            confidence = min((point.confidence for point in ordered), default=0.0)
            return (
                SpatialTrackEstimate(
                    position_base_m=None,
                    velocity_base_m_s=None,
                    confidence=confidence,
                    valid=False,
                    age_s=None,
                ),
                TemporalTrackDiagnostics(
                    candidate_count=candidate_count,
                    accepted_count=len(ordered),
                    latest_age_s=None,
                    innovation_m=None,
                    speed_m_s=None,
                    rejection_reasons=tuple(reasons),
                ),
            )

        latest = ordered[-1]
        latest_age = current_timestamp_s - latest.timestamp_s
        if latest_age < -1e-9:
            raise ValueError("track point 不能晚于当前估计时刻")
        latest_age = max(0.0, float(latest_age))
        try:
            _, velocity = _linear_fit(ordered, latest.timestamp_s)
        except ValueError:
            _append_reason(reasons, "track_timestamp_degenerate")
            velocity = np.zeros(3, dtype=np.float64)
        speed = float(np.linalg.norm(velocity))
        innovation: float | None = None
        if len(ordered) >= 3:
            try:
                predicted, _ = _linear_fit(ordered[:-1], latest.timestamp_s)
                innovation = float(np.linalg.norm(latest.position_base_m - predicted))
            except ValueError:
                _append_reason(reasons, "track_timestamp_degenerate")

        if latest_age > self.config.max_track_age_s:
            _append_reason(reasons, "track_stale")
        if speed > self.config.max_track_speed_m_s:
            _append_reason(reasons, "track_speed_exceeded")
        if innovation is not None and innovation > self.config.max_track_innovation_m:
            _append_reason(reasons, "track_innovation_exceeded")
        valid = not any(
            reason
            in {
                "track_timestamp_degenerate",
                "track_stale",
                "track_speed_exceeded",
                "track_innovation_exceeded",
            }
            for reason in reasons
        )
        confidence = min(point.confidence for point in ordered)
        position = tuple(float(value) for value in latest.position_base_m)
        velocity_tuple = tuple(float(value) for value in velocity)
        return (
            SpatialTrackEstimate(
                position_base_m=position,
                velocity_base_m_s=velocity_tuple,
                confidence=confidence,
                valid=valid,
                age_s=latest_age,
            ),
            TemporalTrackDiagnostics(
                candidate_count=candidate_count,
                accepted_count=len(ordered),
                latest_age_s=latest_age,
                innovation_m=innovation,
                speed_m_s=speed,
                rejection_reasons=tuple(reasons),
            ),
        )

    def estimate(
        self,
        window: ObservationV2Window,
        detections: tuple[WristKeypointDetection | None, ...],
        *,
        current_timestamp_s: float,
        outcome_evidence: DeployableOutcomeEvidence | None = None,
    ) -> DeployableStateEstimatorResult:
        """估计一个 Tick；``current_timestamp_s`` 是 Window 最新控制步时间。

        运行时缺测返回 invalid，结构/来源污染直接拒绝。
        """

        timestamp = _finite(current_timestamp_s, "current_timestamp_s")
        if timestamp < 0.0:
            raise ValueError("current_timestamp_s 必须非负")
        window.validate(self.spec)
        if len(detections) != self.config.history_length:
            raise ValueError(
                f"detections 必须与四时刻 history 对齐，实际长度为 {len(detections)}"
            )
        timestamp_tolerance = max(1e-6, 1e-5 / self.spec.control_hz)
        for index in np.flatnonzero(window.history_valid):
            if window.frame_age_s[index] > timestamp + timestamp_tolerance:
                raise ValueError("current_timestamp_s 早于 Observation history")

        object_points, object_candidates, object_reasons = self._collect_points(
            window,
            detections,
            timestamp,
            target="object",
        )
        goal_points, goal_candidates, goal_reasons = self._collect_points(
            window,
            detections,
            timestamp,
            target="goal",
        )
        object_track, object_diagnostics = self._fit_track(
            object_points,
            object_candidates,
            object_reasons,
            timestamp,
        )
        goal_track, goal_diagnostics = self._fit_track(
            goal_points,
            goal_candidates,
            goal_reasons,
            timestamp,
        )

        evidence = outcome_evidence or DeployableOutcomeEvidence()
        finger_force_index = OBSERVATION_MODALITIES.index("finger_force")
        if window.modality_valid[-1, finger_force_index]:
            force = tuple(float(value) for value in window.finger_force_n[-1])
        else:
            force = (0.0, 0.0)
        state = DeployableStateEstimate(
            object_track=object_track,
            goal_track=goal_track,
            grasp=evidence.grasp or _invalid_scalar(),
            support_contact=evidence.support_contact or _invalid_scalar(),
            settled=evidence.settled or _invalid_scalar(),
            finger_force_n=force,
            timestamp_s=timestamp,
        )
        return DeployableStateEstimatorResult(
            state_estimate=state,
            object_diagnostics=object_diagnostics,
            goal_diagnostics=goal_diagnostics,
        )


def predicates_from_state_estimate(
    state: DeployableStateEstimate,
    thresholds: DeployablePredicateThresholds,
) -> tuple[PredicateEvidence, ...]:
    """只生成能由当前 State Estimate 直接支持的 Predicate。"""

    if state.uses_evaluator_gt:
        raise ValueError("部署 Predicate adapter 禁止使用 evaluator GT state")

    object_valid = (
        state.object_track.valid
        and state.object_track.age_s is not None
        and state.object_track.age_s <= thresholds.max_track_age_s
        and state.object_track.confidence >= thresholds.track_confidence_min
    )
    goal_valid = (
        state.goal_track.valid
        and state.goal_track.age_s is not None
        and state.goal_track.age_s <= thresholds.max_track_age_s
        and state.goal_track.confidence >= thresholds.track_confidence_min
    )

    def scalar_satisfied(value: ScalarStateEstimate, threshold: float) -> bool:
        return bool(
            value.valid
            and value.age_s is not None
            and value.age_s <= thresholds.max_scalar_age_s
            and value.confidence >= threshold
        )

    grasp_candidate = scalar_satisfied(state.grasp, thresholds.grasp_candidate_min)
    grasp_verified = scalar_satisfied(state.grasp, thresholds.grasp_verified_min)
    support_contact = scalar_satisfied(
        state.support_contact,
        thresholds.support_contact_min,
    )
    support_verified = scalar_satisfied(
        state.support_contact,
        thresholds.support_verified_min,
    )
    placement_verified = bool(
        goal_valid and scalar_satisfied(state.settled, thresholds.settled_min)
    )
    values = (
        (
            "object_track_valid",
            object_valid,
            state.object_track.confidence,
            state.object_track.source,
        ),
        (
            "goal_track_valid",
            goal_valid,
            state.goal_track.confidence,
            state.goal_track.source,
        ),
        (
            "precision_target_valid",
            object_valid and goal_valid,
            min(state.object_track.confidence, state.goal_track.confidence),
            PredicateSource.DEPLOYABLE_ESTIMATOR,
        ),
        (
            "grasp_candidate",
            grasp_candidate,
            state.grasp.confidence,
            state.grasp.source,
        ),
        (
            "grasp_verified",
            grasp_verified,
            state.grasp.confidence,
            state.grasp.source,
        ),
        (
            "support_contact_detected",
            support_contact,
            state.support_contact.confidence,
            state.support_contact.source,
        ),
        (
            "support_verified",
            support_verified,
            state.support_contact.confidence,
            state.support_contact.source,
        ),
        (
            "placement_verified",
            placement_verified,
            state.settled.confidence,
            state.settled.source,
        ),
    )
    return tuple(
        PredicateEvidence(
            name=name,
            satisfied=bool(satisfied),
            confidence=float(confidence),
            source=source,
        )
        for name, satisfied, confidence, source in values
    )


def build_deployable_snapshot(
    spec: RobotSpec,
    window: ObservationV2Window,
    state: DeployableStateEstimate,
    thresholds: DeployablePredicateThresholds,
    *,
    tick: int,
    timestamp_s: float,
    additional_predicates: tuple[PredicateEvidence, ...] = (),
    unsafe_or_anomalous: bool = False,
    anomaly_reason: str | None = None,
) -> ExecutiveSnapshot:
    """保留当前 modality age/validity，并组装不含隐藏 GT 的 Executive 输入。"""

    window.validate(spec)
    timestamp = _finite(timestamp_s, "snapshot timestamp_s")
    if timestamp < 0.0:
        raise ValueError("snapshot timestamp_s 必须非负")
    if state.timestamp_s > timestamp + 1e-9:
        raise ValueError("state estimate 不能晚于 snapshot")
    if state.uses_evaluator_gt:
        raise ValueError("部署 snapshot 禁止使用 evaluator GT state")
    if any(item.source == PredicateSource.EVALUATOR_GT for item in additional_predicates):
        raise ValueError("部署 snapshot 禁止使用 evaluator GT predicate")

    finger_force_index = OBSERVATION_MODALITIES.index("finger_force")
    if window.modality_valid[-1, finger_force_index]:
        if not np.allclose(
            state.finger_force_n,
            window.finger_force_n[-1],
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("state F_L/F_R 与当前 Observation V2 不一致")
    elif state.finger_force_n != (0.0, 0.0):
        raise ValueError("无效 finger_force modality 的 state F_L/F_R 必须为零")

    modalities = tuple(
        ModalityStatus(
            name=name,
            valid=bool(window.modality_valid[-1, index]),
            age_s=(
                float(window.modality_age_s[-1, index])
                if window.modality_valid[-1, index]
                else None
            ),
        )
        for index, name in enumerate(OBSERVATION_MODALITIES)
    ) + (
        ModalityStatus(
            name="controller_state",
            valid=bool(window.controller_valid.all()),
            age_s=0.0 if window.controller_valid.all() else None,
        ),
    )
    derived = predicates_from_state_estimate(state, thresholds)
    names = tuple(item.name for item in (*derived, *additional_predicates))
    if len(set(names)) != len(names):
        raise ValueError("additional_predicates 不得覆盖 State Estimate 派生 Predicate")
    unknown = set(names) - set(EXECUTIVE_PREDICATES)
    if unknown:
        raise ValueError(f"snapshot 包含未知 Predicate: {sorted(unknown)}")
    return ExecutiveSnapshot(
        tick=tick,
        timestamp_s=timestamp,
        modalities=modalities,
        predicates=(*derived, *additional_predicates),
        state_estimate=state,
        unsafe_or_anomalous=unsafe_or_anomalous,
        anomaly_reason=anomaly_reason,
    )


__all__ = [
    "DEPLOYABLE_STATE_ESTIMATOR_VERSION",
    "WRIST_KEYPOINT_DETECTION_VERSION",
    "DeployableOutcomeEvidence",
    "DeployablePredicateThresholds",
    "DeployableStateEstimatorConfig",
    "DeployableStateEstimatorResult",
    "FourFrameDeployableStateEstimator",
    "TemporalTrackDiagnostics",
    "WristKeypointDetection",
    "build_deployable_snapshot",
    "predicates_from_state_estimate",
]
