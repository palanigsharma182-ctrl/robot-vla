"""五个通用模块的合成接口回放；不加载策略、真实 provider 或 GT。"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from robot_vla.observation import GL_CAMERA_FROM_CV_CAMERA
from robot_vla.precision.active_external_observation import (
    ActiveExternalObservation, extract_active_external_observation, project_base_point,
)
from robot_vla.precision.active_front_camera import (
    ExternalCameraMotionState as Motion, sample_translation_path,
)
from robot_vla.precision.calibrated_front_provider import (
    ScalarCovarianceCalibration, build_calibrated_object_write_evidence,
    build_stable_camera_calibration_identity, canonical_sha256,
)
from robot_vla.precision.object_memory import (
    ExplicitObjectStateMemory, ObjectCandidateWindowVerifier, ObjectMeasurement,
    ObjectMemoryConfig, ObjectMemorySafetyContext, ObjectStateRequirement, resolve_object_state,
)

CAMERA = "synthetic-front-camera"
PRIMITIVE = "SYNTHETIC_CAPTURE"
INTRINSIC = np.array([[40., 0., 31.5], [0., 40., 31.5], [0., 0., 1.]])
SAFE = ObjectMemorySafetyContext(True, True, True, False, False, False, False, False)


def synthetic_calibration(scale: float = 4.0) -> ScalarCovarianceCalibration:
    """固定的人工校准元数据；不读取或伪称存在正式 validation ledger。"""
    identity = canonical_sha256({"fixture": "synthetic-five-common/v1"})
    return ScalarCovarianceCalibration(
        scale_factor=scale, support_count=40, order_statistic_k=39,
        quantile_score=scale * 5.991, scoring_ledger_sha256=identity,
        config_sha256=identity, validation_data_identity_sha256=identity,
        source_identity_sha256=identity,
    )


def measurement_from_prediction(
    observation: ActiveExternalObservation,
    *,
    position_camera_m: np.ndarray | None,
    covariance_camera_m2: np.ndarray | None,
    calibration: ScalarCovarianceCalibration,
    tcp_pose_timestamp_s: float,
    collect_started_at_s: float | None,
    visibility_probability: float = 0.95,
) -> ObjectMeasurement:
    """实验消费者：校准后的预测经实际相机坐标变换，再受运动和证据门槛约束。"""
    if observation.camera_uid != CAMERA:
        raise ValueError("观测相机与本合成 provider 的 target camera 不一致")
    if collect_started_at_s is not None and (
        not np.isfinite(collect_started_at_s) or collect_started_at_s < 0
        or collect_started_at_s > observation.control_timestamp_s
    ):
        raise ValueError("采集阶段起点必须是不晚于当前 tick 的有效时间")
    phase_bound = collect_started_at_s is not None and min(
        observation.rgb_timestamp_s, observation.camera_pose_timestamp_s,
    ) >= collect_started_at_s
    identity, _ = build_stable_camera_calibration_identity(
        camera_uid=CAMERA, primitive_id=observation.viewpoint_primitive_id,
        intrinsic_cv=observation.intrinsic_cv,
        actual_base_from_camera_cv=observation.base_from_external_camera_cv,
        covariance_calibration_identity_sha256=calibration.identity_sha256,
        source_training_camera="synthetic-camera", target_camera=CAMERA,
        frame_convention="robot-base-from-opencv-optical-camera/v1",
    )
    position = covariance = None
    projection = {"projection_valid": False, "in_frame": False}
    if position_camera_m is not None:
        xyz = np.asarray(position_camera_m, dtype=np.float64)
        if xyz.shape != (3,) or not np.isfinite(xyz).all() or covariance_camera_m2 is None:
            raise ValueError("预测要求有限 camera XYZ 与配套 covariance")
        transform = observation.base_from_external_camera_cv
        position = (transform @ np.r_[xyz, 1.])[:3]
        rotation = transform[:3, :3]
        covariance = rotation @ calibration.calibrate_covariance(covariance_camera_m2) @ rotation.T
        projection = project_base_point(observation, position)
    elif covariance_camera_m2 is not None:
        raise ValueError("缺少位置时不能提供 covariance")
    geometry_valid = bool(projection["projection_valid"] and projection["in_frame"])
    evidence, _ = build_calibrated_object_write_evidence(
        calibration=calibration, raw_sigma_xy_px=np.array([0.1, 0.1]),
        visibility_probability=visibility_probability,
        projection_validity_probability=float(projection["projection_valid"]),
        object_mask_probability=0.95 if geometry_valid else 0.0,
        goal_mask_probability=0.0, normalized_entropy=0.05,
        geometry_valid=geometry_valid, min_object_mask_probability=0.5,
        max_goal_mask_probability=0.5,
    )
    return ObjectMeasurement(
        timestamp_s=observation.control_timestamp_s,
        rgb_timestamp_s=observation.rgb_timestamp_s,
        camera_pose_timestamp_s=observation.camera_pose_timestamp_s,
        tcp_pose_timestamp_s=tcp_pose_timestamp_s,
        position_base_m=position, covariance_base_m2=covariance,
        confidence=evidence.score, projection_valid=bool(projection["projection_valid"]),
        in_fov=geometry_valid, observable=evidence.observable, geometry_valid=geometry_valid,
        write_gate_passed=bool(evidence.accepted(threshold=0.7)
                              and observation.memory_write_eligible and phase_bound),
        source_camera=CAMERA,
        # 数值工具生成的资格 identity 不授予写入权限；此消费者使用独立合成来源。
        source_model_identity="synthetic-prediction/" + identity["identity_sha256"],
    )


def synthetic_observation(
    *, episode: str, index: int, tick: float, rgb_time: float,
    motion: Motion, actual_translation: np.ndarray,
) -> ActiveExternalObservation:
    """生成明确标记的传感器 fixture，故意让 commanded 与 actual pose 不同。"""
    actual_cv = np.eye(4)
    actual_cv[:3, 3] = actual_translation
    commanded_gl = actual_cv @ GL_CAMERA_FROM_CV_CAMERA
    commanded_gl = commanded_gl.copy()
    commanded_gl[0, 3] += 0.2
    return extract_active_external_observation(
        {"sensor_data": {CAMERA: {"rgb": np.zeros((64, 64, 3), dtype=np.uint8)}},
         "sensor_param": {CAMERA: {"intrinsic_cv": INTRINSIC,
                                   "cam2world_gl": actual_cv @ GL_CAMERA_FROM_CV_CAMERA}}},
        camera_uid=CAMERA, world_from_robot_base=np.eye(4),
        commanded_world_from_external_camera_gl=commanded_gl,
        episode_id=episode, request_id="synthetic-request",
        observation_sequence_id=f"{episode}:rgb:{rgb_time}",
        camera_command_sequence_id=f"{episode}:command:{index}",
        control_tick=index, control_timestamp_s=tick, rgb_timestamp_s=rgb_time,
        camera_pose_timestamp_s=rgb_time, camera_motion_state=motion,
        viewpoint_primitive_id=PRIMITIVE, settled=motion == Motion.COLLECT,
        maximum_rotation_projection_error_frobenius=1e-6,
    )


def run_replay() -> dict:
    """跨 Episode 复用同一 Memory/verifier，覆盖实际接线及错误输入传播。"""
    calibration = synthetic_calibration()
    path = sample_translation_path(np.zeros(3), np.array([0.1, 0., 0.]), steps=4)
    initial = synthetic_observation(episode="initial", index=0, tick=0., rgb_time=0.,
                                    motion=Motion.COLLECT, actual_translation=path[-1])
    sample = measurement_from_prediction(initial, position_camera_m=np.array([0., 0., 1.]),
                                          covariance_camera_m2=np.eye(3)*1e-6,
                                          calibration=calibration, tcp_pose_timestamp_s=0.,
                                          collect_started_at_s=0.)
    config = ObjectMemoryConfig(
        max_unobserved_age_s=0.5, max_innovation_m=0.01, max_position_std_m=0.02,
        min_candidate_frames=2, max_candidate_gap_s=0.1,
        max_candidate_position_spread_m=0.005, max_sensor_skew_s=0.02,
        expected_source_camera=CAMERA, expected_source_model_identity=sample.source_model_identity,
    )
    memory, verifier = ExplicitObjectStateMemory(config), ObjectCandidateWindowVerifier(config)
    scenarios = {
        "timing_and_motion": [
            (0., 0., "move"), (0.05, 0.05, "settle"),
            (0.10, 0.10, "visible"), (0.15, 0.141, "visible"),
            (0.155, 0.141, "duplicate"), (0.16, 0.14, "reversed"),
            (0.20, 0.191, "visible"), (0.25, 0.241, "visible"),
            (0.26, 0.251, "move"), (0.27, 0.261, "return"),
            (0.30, 0.291, "low_score"), (0.35, 0.341, "missing"),
            (0.80, 0.791, "expired"),
        ],
        "source_change": [(0., 0., "visible"), (0.05, 0.05, "visible"),
                          (0.10, 0.10, "source_change"), (0.15, 0.15, "visible")],
        "contact": [(0., 0., "visible"), (0.05, 0.05, "visible"),
                    (0.10, 0.10, "contact"), (0.15, 0.15, "visible")],
        "after_reset": [(0., 0., "visible"), (0.05, 0.05, "visible")],
    }
    rows = []
    for episode, frames in scenarios.items():
        memory.reset(episode)
        verifier.reset(episode)
        collect_started_at_s = None
        for index, (tick, rgb_time, event) in enumerate(frames):
            motion = {"move": Motion.MOVE_TO_VIEW, "settle": Motion.SETTLE_AT_VIEW,
                      "return": Motion.RETURN_HOME}.get(event, Motion.COLLECT)
            translation = path[0] if event == "move" else path[-1]
            if motion != Motion.COLLECT:
                collect_started_at_s = None
            elif collect_started_at_s is None:
                collect_started_at_s = tick
            observation = synthetic_observation(
                episode=episode, index=index, tick=tick, rgb_time=rgb_time,
                motion=motion, actual_translation=translation,
            )
            # 这是人工设计的预测值，不是从仿真器获取物体真值。
            synthetic_base_prediction = np.array([0.1, 0., 1.])
            if event == "low_score":
                synthetic_base_prediction[0] += 0.3
            xyz = (np.linalg.inv(observation.base_from_external_camera_cv)
                   @ np.r_[synthetic_base_prediction, 1.])[:3]
            missing = event in ("missing", "expired")
            measurement = measurement_from_prediction(
                observation, position_camera_m=None if missing else xyz,
                covariance_camera_m2=None if missing else np.eye(3)*1e-6,
                calibration=synthetic_calibration(9.) if event == "source_change" else calibration,
                tcp_pose_timestamp_s=rgb_time, collect_started_at_s=collect_started_at_s,
                visibility_probability=0.1 if event == "low_score" else 0.95,
            )
            safety = replace(SAFE, object_contact_detected=event == "contact")
            candidate = verifier.observe(measurement, episode_id=episode, safety=safety)
            update = memory.update(candidate, episode_id=episode, safety=safety)
            navigation = resolve_object_state(update, requirement=ObjectStateRequirement.NAVIGATION)
            rows.append({
                "episode": episode, "event": event, "tick": tick, "rgb_time": rgb_time,
                "motion": motion.value, "motion_write_eligible": observation.memory_write_eligible,
                "accepted": update.measurement_accepted, "valid": update.state.valid,
                "age_s": update.state.age_s, "position_base_m": update.state.position_base_m,
                "covariance_base_m2": update.state.covariance_base_m2,
                "navigation_available": navigation.available, "memory_only": navigation.memory_only,
                "contact_authorized": navigation.contact_authorized,
                "rejections": update.rejection_reasons, "invalid_reasons": update.state.invalid_reasons,
                "source_identity": measurement.source_model_identity,
                "actual_translation": observation.actual_world_from_external_camera_gl[:3, 3].tolist(),
                "commanded_translation": observation.commanded_world_from_external_camera_gl[:3, 3].tolist(),
            })
    return {"evidence_level": "synthetic-five-module-interface-replay", "rows": rows,
            "episodes": len(scenarios), "actuation_count": 0, "provider_inference_count": 0}


if __name__ == "__main__":
    print(json.dumps(run_replay(), ensure_ascii=False, allow_nan=False, indent=2))
