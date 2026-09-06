"""消费冻结 D046/D048 产物，输出 D049 HOME 分数或 PRIMARY 平面测量。

仅适用于 development 仿真中的 FREE_STATIC 物体中心 z=0.02m；不授予 actuator 权限。
训练、资格评分和数据采集仍由各自实验负责，此处只加载、推理及转换证据。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla.adapters import FingerForceNormalizer, FingerForceStats, ProprioNormalizer, ProprioStats
from robot_vla.contracts import RobotSpec
from robot_vla.executive.contracts import PhaseId
from robot_vla.precision.active_external_observation import ACTUAL_EXTERNAL_POSE_SOURCE, ActiveExternalObservation
from robot_vla.precision.active_front_camera import ExternalCameraMotionState as Motion, rotation_angular_distance_rad
from robot_vla.precision.active_front_memory_provider import (
    ACTIVE_FRONT_HOME_PRIMITIVE_ID, ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
    ActiveFrontScoreComponents, ActiveFrontStage2FrameEvidence, PassiveHomeScoreEvidence,
    d049_home_baseline_provider_identity, d049_primary_provider_identity,
)
from robot_vla.precision.active_front_provider import build_precision_camera_role_state
from robot_vla.precision.calibrated_front_provider import array_sha256, canonical_sha256
from robot_vla.precision.detection import precision_prediction_to_wrist_detection
from robot_vla.precision.object_memory import ObjectMemorySafetyContext
from robot_vla.precision.observability import mask_probability_at_normalized_uv
from robot_vla.precision.outliers import geometry_conditioning


def _read_bound_json(path: Path, expected: str) -> Any:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"D049 artifact SHA-256 不匹配: {path.name}")
    return json.loads(raw)


def _check_internal(payload: dict, field: str, expected: str) -> None:
    if payload.get(field) != expected or canonical_sha256(
        {k: v for k, v in payload.items() if k != field}
    ) != expected:
        raise ValueError(f"D049 artifact {field} 不匹配")


def verify_d049_bundle(bundle: str | Path) -> dict:
    """仅核验消费所需冻结组件；不重开历史 label 或重跑资格评估。"""
    root = Path(bundle)
    identity = d049_primary_provider_identity()
    bound = {
        "qualification_config.json": identity.qualification_config_raw_sha256,
        "qualification_receipt.json": identity.qualification_result_receipt_raw_sha256,
        "calibration_config.json": identity.calibration_config_raw_sha256,
        "calibration_receipt.json": identity.calibration_result_receipt_raw_sha256,
        "viewpoint_calibrations.json": identity.calibration_viewpoints_raw_sha256,
        "proprio_stats.json": identity.proprio_stats_sha256,
        "finger_force_stats.json": identity.finger_force_stats_sha256,
    }
    values = {name: _read_bound_json(root/name, digest) for name, digest in bound.items()}
    for name, field, expected in (
        ("qualification_config.json", "config_sha256", identity.qualification_config_internal_sha256),
        ("qualification_receipt.json", "receipt_sha256", identity.qualification_result_receipt_internal_sha256),
        ("calibration_config.json", "config_sha256", identity.calibration_config_internal_sha256),
        ("calibration_receipt.json", "receipt_sha256", identity.calibration_result_receipt_internal_sha256),
    ):
        _check_internal(values[name], field, expected)
    verification = json.loads((root/"qualification_verification.json").read_text())
    _check_internal(verification, "verification_sha256", identity.qualification_result_verification_sha256)
    receipt = values["qualification_receipt.json"]
    if (receipt["source_identity_sha256"] != identity.qualification_source_identity_sha256
            or receipt["primary"]["primary_viewpoint_id"] != identity.primitive_id
            or not receipt["protocol_valid"] or not receipt["gate_passed"] or not verification["verified"]):
        raise ValueError("D048 PRIMARY 资格或来源不匹配")
    execution = json.loads((root/"qualification_execution.json").read_text())
    _check_internal(execution, "receipt_sha256", receipt["execution_receipt_internal_sha256"])
    raw_predictions = (root/"qualification_predictions.jsonl").read_bytes()
    ledger = execution["prediction_ledger"]
    if (hashlib.sha256(raw_predictions).hexdigest() != ledger["raw_sha256"]
            or len(raw_predictions) != ledger["size_bytes"]
            or len(raw_predictions.splitlines()) != ledger["row_count"]):
        raise ValueError("D048 camera witness ledger hash 不匹配")
    constraints = {}
    for line in raw_predictions.splitlines():
        row = json.loads(line)
        primitive = row["viewpoint_id"]
        if primitive in (ACTIVE_FRONT_HOME_PRIMITIVE_ID, ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID):
            current = (np.asarray(row["external_intrinsic_cv"]), np.asarray(row["base_from_external_camera_cv"]))
            if primitive in constraints:
                _check_camera_geometry(*current, *constraints[primitive])
            else:
                constraints[primitive] = current
    if set(constraints) != {ACTIVE_FRONT_HOME_PRIMITIVE_ID, ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID}:
        raise ValueError("D048 缺少 HOME/PRIMARY 同源相机 witness")
    for provider_identity in (identity, d049_home_baseline_provider_identity()):
        rows = [r for r in values["viewpoint_calibrations.json"]
                if r["viewpoint_id"] == provider_identity.primitive_id]
        if (len(rows) != 1 or not rows[0]["passed"]
                or rows[0]["write_threshold"] != provider_identity.write_threshold
                or rows[0]["calibration"]["scale_factor"] != provider_identity.calibration_scale_factor):
            raise ValueError("D046 HOME/PRIMARY 校准值不匹配")
    return {"files": bound, "provider_identity_sha256": identity.sha256, "camera_constraints": constraints,
            "qualification_verification_sha256": identity.qualification_result_verification_sha256}


def _check_camera_geometry(intrinsic, pose, expected_intrinsic, expected_pose) -> None:
    if (not np.allclose(intrinsic, expected_intrinsic, rtol=0., atol=1e-6)
            or np.linalg.norm(pose[:3, 3]-expected_pose[:3, 3]) > 1e-5
            or rotation_angular_distance_rad(pose[:3, :3], expected_pose[:3, :3]) > 1e-4):
        raise ValueError("D048 相机 K/actual pose 超出冻结资格范围")


@dataclass(frozen=True)
class PlanarObjectPrediction:
    """条件于固定高度的测量；零 Z 方差不代表真实高度无误差。"""

    components: ActiveFrontScoreComponents
    normalized_uv: tuple[float, float]
    position_base_m: np.ndarray | None
    covariance_base_m2: np.ndarray | None

    @property
    def evidence(self):
        return self.components.to_object_write_evidence(geometry_valid=self.position_base_m is not None)


def decode_d049_planar_prediction(prediction: Any, *, intrinsic_cv: np.ndarray,
                                  base_from_camera_cv: np.ndarray, covariance_scale: float) -> PlanarObjectPrediction:
    """保留 D049 raw-sigma score，scale 仅乘 J diag(sigma²) Jᵀ。"""
    if not np.isfinite(covariance_scale) or covariance_scale < 1.0:
        raise ValueError("covariance_scale 必须有限且 >=1")
    if tuple(prediction.keypoints.normalized_uv.shape) != (1, 2, 2):
        raise ValueError("D049 单帧 keypoints 必须是 [1,2,2]")
    mask = prediction.mask_probability
    if mask is None or getattr(mask, "requires_grad", False):
        raise ValueError("需要同次冻结 forward 的 mask")
    if hasattr(mask, "detach"):
        mask = mask.detach().float().cpu().numpy()
    mask = np.asarray(mask)
    if mask.shape != (1, 2, 128, 128):
        raise ValueError("D049 mask 必须是 [1,2,128,128]")
    keypoint = precision_prediction_to_wrist_detection(
        prediction, keypoint_names=("object_center", "goal_center"), timestamp_s=0.,
    ).object_evidence
    uv = keypoint.normalized_uv
    components = ActiveFrontScoreComponents(
        object_visibility_probability=keypoint.visibility_probability,
        projection_validity_probability=keypoint.projection_validity_probability,
        object_mask_probability=mask_probability_at_normalized_uv(mask[0, 0], uv),
        goal_mask_probability=mask_probability_at_normalized_uv(mask[0, 1], uv),
        object_normalized_entropy=keypoint.normalized_entropy, object_sigma_xy_px=keypoint.sigma_px,
    )
    try:
        geometry = geometry_conditioning(normalized_uv=np.asarray(uv), intrinsic_cv=intrinsic_cv,
            base_from_camera_cv=base_from_camera_cv, image_size_hw=(128, 128), plane_base_z_m=0.02)
        position = np.asarray(geometry["predicted_world_point_base_m"], dtype=np.float64)
        jacobian = np.asarray(geometry["local_jacobian_xy_m_per_px"], dtype=np.float64)
        covariance = np.zeros((3, 3), dtype=np.float64)
        covariance[:2, :2] = jacobian @ np.diag(np.square(keypoint.sigma_px)) @ jacobian.T
        covariance *= covariance_scale
        if not np.isfinite(covariance).all():
            raise ValueError("非有限几何 covariance")
    except (ValueError, np.linalg.LinAlgError):
        position, covariance = None, None
    return PlanarObjectPrediction(components, tuple(uv), position, covariance)


class D049FrontProvider:
    """显式消费真实冻结包；HOME 永远只返回 baseline，PRIMARY 才返回候选证据。"""

    def __init__(self, bundle: str | Path) -> None:
        from robot_vla.precision.checkpoint import PrecisionCheckpointRole, load_torch_precision_frame_predictor
        from robot_vla.precision.provider import TorchPrecisionFramePredictorConfig

        root = Path(bundle)
        self.bundle_verification = verify_d049_bundle(root)
        identity = d049_primary_provider_identity()
        self.spec = RobotSpec()
        self.proprio = ProprioNormalizer(ProprioStats.from_json(root/"proprio_stats.json"), self.spec)
        self.force = FingerForceNormalizer(FingerForceStats.from_json(root/"finger_force_stats.json"), self.spec)
        proprio_sha = canonical_sha256({"mean": self.proprio.mean.tolist(), "std": self.proprio.std.tolist(),
                                       "clip": self.proprio.clip, "robot_spec": self.spec.to_dict()})
        force_sha = canonical_sha256({"stats": asdict(self.force.stats), "scale": self.force.scale.tolist(),
                                     "clip": self.force.clip, "robot_spec": self.spec.to_dict()})
        if proprio_sha != identity.proprio_normalizer_sha256 or force_sha != identity.finger_force_normalizer_sha256:
            raise ValueError("D049 normalizer semantic identity 不匹配")
        loaded = load_torch_precision_frame_predictor(
            root/"weights.pt", expected_checkpoint_sha256=identity.checkpoint_sha256,
            expected_provenance_sha256=identity.checkpoint_provenance_sha256,
            expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
            predictor_config=TorchPrecisionFramePredictorConfig(device="cuda", use_bf16=True, temperature=1.0),
        )
        if (loaded.receipt.parameter_state_sha256 != identity.checkpoint_parameter_sha256
                or loaded.receipt.model_config_sha256 != identity.model_config_sha256):
            raise ValueError("D049 checkpoint parameter/config identity 不匹配")
        self.predictor = loaded.predictor
        self.predictor.verify_identity()
        self.forward_count = 0
        self._execution_config = self.predictor.config
        self._predictor_identity_sha256 = self.predictor.identity.sha256
        self._episode_identity = None
        for array in (self.proprio.mean, self.proprio.std, self.force.scale):
            array.setflags(write=False)

    def _verify_runtime_identity(self, episode_id: str, generation: int) -> None:
        identity = d049_primary_provider_identity()
        proprio_sha = canonical_sha256({"mean": self.proprio.mean.tolist(), "std": self.proprio.std.tolist(),
            "clip": self.proprio.clip, "robot_spec": self.spec.to_dict()})
        force_sha = canonical_sha256({"stats": asdict(self.force.stats), "scale": self.force.scale.tolist(),
            "clip": self.force.clip, "robot_spec": self.spec.to_dict()})
        if (proprio_sha != identity.proprio_normalizer_sha256 or force_sha != identity.finger_force_normalizer_sha256
                or self.predictor.config != self._execution_config or self.predictor.device.type != "cuda"
                or self.predictor.identity.sha256 != self._predictor_identity_sha256):
            raise ValueError("D049 runtime normalizer/execution identity 漂移")
        episode = (episode_id, generation)
        if episode != self._episode_identity:
            self.predictor.verify_identity()
            self._episode_identity = episode

    def predict(self, sidecar: ActiveExternalObservation, *, physical_proprio: np.ndarray,
                base_from_tcp: np.ndarray, finger_force_n: np.ndarray, tcp_timestamp_s: float,
                episode_generation: int, source_phase: PhaseId, safety: ObjectMemorySafetyContext,
                static_plane_scope_verified: bool) -> ActiveFrontStage2FrameEvidence | PassiveHomeScoreEvidence:
        """调用方从实际传感与任务约束确认静态平面适用性，不接收 object GT。"""
        # 副本构成单次推理边界；forward 期间外部数组变动不改变已绑定的 state/geometry。
        sidecar = replace(sidecar, **{name: getattr(sidecar, name).copy() for name in (
            "rgb_external", "intrinsic_cv", "base_from_external_camera_cv",
            "actual_world_from_external_camera_gl", "commanded_world_from_external_camera_gl")})
        physical_proprio = np.asarray(physical_proprio).copy()
        base_from_tcp = np.asarray(base_from_tcp).copy()
        finger_force_n = np.asarray(finger_force_n).copy()
        self._verify_runtime_identity(sidecar.episode_id, episode_generation)
        home = sidecar.viewpoint_primitive_id == ACTIVE_FRONT_HOME_PRIMITIVE_ID
        primary = sidecar.viewpoint_primitive_id == ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID
        phase = Motion.HOME_ANCHOR if home else Motion.COLLECT
        if (not (home or primary) or sidecar.camera_motion_state is not phase or not sidecar.settled
                or sidecar.camera_uid != "base_camera" or sidecar.actual_pose_source != ACTUAL_EXTERNAL_POSE_SOURCE):
            raise ValueError("D049 仅接受实际 settled HOME_ANCHOR 或 PRIMARY COLLECT")
        expected_k, expected_pose = self.bundle_verification["camera_constraints"][sidecar.viewpoint_primitive_id]
        _check_camera_geometry(sidecar.intrinsic_cv, sidecar.base_from_external_camera_cv, expected_k, expected_pose)
        _check_camera_geometry(sidecar.intrinsic_cv, sidecar.actual_world_from_external_camera_gl,
                               sidecar.intrinsic_cv, sidecar.commanded_world_from_external_camera_gl)
        times = (sidecar.control_timestamp_s, sidecar.rgb_timestamp_s, sidecar.camera_pose_timestamp_s, tcp_timestamp_s)
        if (not all(np.isfinite(t) and t >= 0 for t in times) or max(times)-min(times) > 0.01+1e-12
                or any(t > times[0]+1e-12 for t in times[1:])):
            raise ValueError("D049 sensor timestamp 不同步或来自未来")
        if sidecar.rgb_external.shape != (128, 128, 3):
            raise ValueError("D049 RGB 必须是 [128,128,3]")
        if type(static_plane_scope_verified) is not bool or not static_plane_scope_verified:
            raise ValueError("D049 固定高度 FREE_STATIC 平面范围未经确认")
        if (safety.invalidation_reasons or not safety.pregrasp_window_open or not safety.gripper_open
                or not safety.controller_tracking_valid or np.max(finger_force_n) > 0.01
                or physical_proprio[-1] < 0.95):
            raise ValueError("D049 safe-hold/free-static 条件不满足")
        state = build_precision_camera_role_state(spec=self.spec, proprio_normalizer=self.proprio,
            finger_force_normalizer=self.force, physical_proprio=physical_proprio, base_from_tcp=base_from_tcp,
            base_from_camera_cv=sidecar.base_from_external_camera_cv, finger_force_n=finger_force_n)
        motion = np.zeros(4, dtype=np.float32)
        identity = d049_home_baseline_provider_identity() if home else d049_primary_provider_identity()
        input_digest = canonical_sha256({"version": "d049-verified-provider-input/v1",
            "sidecar_sha256": sidecar.audit_digest(), "rgb_sha256": array_sha256(sidecar.rgb_external),
            "state_sha256": array_sha256(state), "motion_sha256": array_sha256(motion),
            "tcp_timestamp_s": tcp_timestamp_s, "episode_generation": episode_generation,
            "source_phase": source_phase.value, "provider_sha256": identity.sha256,
            "static_plane_scope_verified": static_plane_scope_verified, "safety": asdict(safety)})
        prediction = self.predictor.predict(sidecar.rgb_external, state, motion, include_mask_probability=True)
        self.forward_count += 1
        planar = decode_d049_planar_prediction(prediction, intrinsic_cv=sidecar.intrinsic_cv,
            base_from_camera_cv=sidecar.base_from_external_camera_cv, covariance_scale=identity.calibration_scale_factor)
        evidence = planar.evidence
        output_digest = canonical_sha256({"input_digest": input_digest, "components": asdict(planar.components),
            "normalized_uv": planar.normalized_uv,
            "position": None if planar.position_base_m is None else planar.position_base_m.tolist(),
            "covariance": None if planar.covariance_base_m2 is None else planar.covariance_base_m2.tolist()})
        common = dict(episode_id=sidecar.episode_id, episode_generation=episode_generation,
            request_id=sidecar.request_id, observation_sequence_id=sidecar.observation_sequence_id,
            model_input_digest=input_digest, provider_output_digest=output_digest, provider_identity=identity,
            camera_motion_state=phase, settled=sidecar.settled, control_timestamp_s=times[0],
            rgb_timestamp_s=times[1], camera_pose_timestamp_s=times[2], tcp_pose_timestamp_s=times[3],
            base_from_external_camera_cv=sidecar.base_from_external_camera_cv,
            score_components=planar.components, geometry_valid=evidence.geometry_valid)
        if home:
            return PassiveHomeScoreEvidence(**common, viewpoint_primitive_id=identity.primitive_id,
                                            stored_write_score=evidence.score)
        projection_valid = planar.components.projection_validity_probability >= 0.5
        in_fov = projection_valid and all(0 <= x <= 1 for x in planar.normalized_uv)
        observable = bool(evidence.observable and projection_valid and in_fov)
        return ActiveFrontStage2FrameEvidence(**common, source_phase=source_phase,
            position_base_m=planar.position_base_m, covariance_base_m2=planar.covariance_base_m2,
            measurement_confidence=evidence.score, write_score=evidence.score,
            projection_valid=projection_valid, in_fov=in_fov, observable=observable,
            structurally_eligible=bool(observable and evidence.geometry_valid), deployable_free_static_safe=True)
