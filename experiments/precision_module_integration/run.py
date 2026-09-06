"""复用现有模型/Checkpoint/V2 状态与五模块工具的合成工程验收，不训练。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from robot_vla.adapters import FingerForceNormalizer, FingerForceStats, ProprioNormalizer, ProprioStats
from robot_vla.contracts import RobotSpec
from robot_vla.observation import GL_CAMERA_FROM_CV_CAMERA
from robot_vla.precision.active_external_observation import extract_active_external_observation
from robot_vla.precision.active_front_camera import ExternalCameraMotionState as Motion, sample_translation_path
from robot_vla.precision.active_front_provider import ActiveFrontProviderAdapterConfig, build_active_front_model_input
from robot_vla.precision.calibrated_front_provider import (
    ScalarCovarianceCalibration, build_calibrated_object_evidence_from_prediction, canonical_sha256,
)
from robot_vla.precision.checkpoint import (
    PrecisionCheckpointProvenance, PrecisionCheckpointRole, load_torch_precision_frame_predictor,
    save_precision_checkpoint,
)
from robot_vla.precision.model import PrecisionThreeHeadUNet, PrecisionUNetConfig
from robot_vla.precision.provider import PrecisionGeometricMotionInput, TorchPrecisionFramePredictorConfig


def run_replay() -> dict:
    """真实小模型初始化→临时 checkpoint→冻结重载→front 输入→预测证据校准。"""
    spec = RobotSpec()
    proprio = ProprioNormalizer(ProprioStats(
        mean=(0.,)*spec.proprio_dim, std=(1.,)*spec.proprio_dim, count=1,
    ), spec)
    force = FingerForceNormalizer(FingerForceStats(
        scale_log1p_p95=(1., 1.), count=1, positive_count=(1, 1),
    ), spec)
    fixture_sha = canonical_sha256({"kind": "synthetic-interface-fixture", "seed": 0})
    root = Path(__file__).resolve().parents[2]
    source_files = ["src/robot_vla/precision/" + name + ".py" for name in (
        "model", "provider", "checkpoint", "active_front_provider", "active_front_camera",
        "active_external_observation", "calibrated_front_provider", "object_observability",
    )] + [str(Path(__file__).resolve().relative_to(root))]
    source_sha = canonical_sha256({p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source_files})
    config = PrecisionUNetConfig(encoder_channels=(8, 16), state_hidden_size=8, head_hidden_size=16)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = PrecisionThreeHeadUNet(config)
    calibration = ScalarCovarianceCalibration(
        scale_factor=4., support_count=40, order_statistic_k=39, quantile_score=4.*5.991,
        scoring_ledger_sha256=fixture_sha, config_sha256=fixture_sha,
        validation_data_identity_sha256=fixture_sha, source_identity_sha256=source_sha,
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="precision-integration-") as temporary:
        checkpoint = Path(temporary)/"synthetic.pt"
        receipt = save_precision_checkpoint(checkpoint, model, PrecisionCheckpointProvenance(
            role=PrecisionCheckpointRole.SYNTHETIC_DEBUG, data_identity_sha256=fixture_sha,
            training_config_sha256=fixture_sha, source_tree_sha256=source_sha,
            seed=0, examples_seen=0, optimizer_steps=0,
        ))
        loaded = load_torch_precision_frame_predictor(
            checkpoint, expected_checkpoint_sha256=receipt.checkpoint_sha256,
            expected_provenance_sha256=receipt.provenance_sha256,
            expected_role=PrecisionCheckpointRole.SYNTHETIC_DEBUG,
            predictor_config=TorchPrecisionFramePredictorConfig(device="cpu"),
        )
        predictor = loaded.predictor
        predictor.verify_identity()
        forward_calls = []
        hook = predictor.model.register_forward_hook(
            lambda module, args, output: forward_calls.append(None)
        )
        path = sample_translation_path(np.zeros(3), np.array([0.1, 0., 0.]), steps=2)
        for tick, phase in enumerate((Motion.MOVE_TO_VIEW, Motion.COLLECT, Motion.COLLECT)):
            forwards_before = len(forward_calls)
            t = tick / spec.control_hz
            actual_cv = np.eye(4)
            actual_cv[:3, 3] = path[min(tick, 1)]
            intrinsic = np.array([[25., 0., 15.5], [0., 25., 15.5], [0., 0., 1.]])
            sidecar = extract_active_external_observation(
                {"sensor_data": {"base_camera": {"rgb": np.full((32, 32, 3), 127, dtype=np.uint8)}},
                 "sensor_param": {"base_camera": {"intrinsic_cv": intrinsic,
                    "cam2world_gl": actual_cv @ GL_CAMERA_FROM_CV_CAMERA}}},
                camera_uid="base_camera", world_from_robot_base=np.eye(4),
                commanded_world_from_external_camera_gl=GL_CAMERA_FROM_CV_CAMERA,
                episode_id="synthetic-model-episode", request_id="synthetic-model-request",
                observation_sequence_id=f"frame-{tick}", camera_command_sequence_id=f"command-{tick}",
                control_tick=tick, control_timestamp_s=t, rgb_timestamp_s=t, camera_pose_timestamp_s=t,
                camera_motion_state=phase, viewpoint_primitive_id="SYNTHETIC_VIEW",
                settled=phase == Motion.COLLECT, maximum_rotation_projection_error_frobenius=1e-6,
            )
            if not sidecar.memory_write_eligible:
                rows.append({"tick": tick, "status": "motion-rejected",
                             "forward_count": len(forward_calls) - forwards_before})
                continue
            model_input = build_active_front_model_input(
                spec=spec, proprio_normalizer=proprio, finger_force_normalizer=force,
                config=ActiveFrontProviderAdapterConfig(), episode_id=sidecar.episode_id,
                request_id=sidecar.request_id, observation_sequence_id=sidecar.observation_sequence_id,
                primitive_id=sidecar.viewpoint_primitive_id, rgb_external=sidecar.rgb_external,
                physical_proprio=np.zeros(spec.proprio_dim, dtype=np.float32), base_from_tcp=np.eye(4),
                base_from_external_camera_cv=sidecar.base_from_external_camera_cv,
                finger_force_n=np.zeros(2, dtype=np.float32), intrinsic_cv=sidecar.intrinsic_cv,
                control_timestamp_s=t, rgb_timestamp_s=t, camera_pose_timestamp_s=t, tcp_pose_timestamp_s=t,
                geometric_motion=PrecisionGeometricMotionInput(timestamp_s=t, motion=(0., 0., 0., 0.)),
                geometric_motion_provider_id="synthetic-no-motion", camera_motion_state=phase,
                settled=sidecar.settled, actual_pose_source=sidecar.actual_pose_source,
            )
            prediction = predictor.predict(
                model_input.rgb_external, model_input.structured_state, model_input.geometric_motion,
                include_mask_probability=True,
            )
            evidence, sigma = build_calibrated_object_evidence_from_prediction(
                prediction, keypoint_names=config.keypoint_names, mask_names=config.mask_names,
                image_size_hw=(32, 32), timestamp_s=t, calibration=calibration,
                geometry_valid=False,  # 本验收没有合格 3D geometry，不能由二维预测补造。
            )
            rows.append({
                "tick": tick, "status": "qualification-evidence",
                "forward_count": len(forward_calls) - forwards_before,
                "source_camera": model_input.source_camera, "input_sha256": model_input.input_digest,
                "structured_state_shape": list(model_input.structured_state.shape),
                "mask_shape": list(prediction.mask_probability.shape),
                "score": evidence.score, "calibrated_sigma_px": sigma.tolist(),
                "qualification_only": model_input.qualification_only,
                "memory_write_eligible": model_input.memory_write_eligible,
                "geometry_valid": evidence.geometry_valid,
            })
        predictor.verify_identity()
        hook.remove()
    return {"evidence_level": "synthetic-checkpoint-model-interface", "source_sha256": source_sha,
            "checkpoint_role": "synthetic-debug", "optimizer_steps": 0,
            "checkpoint_sha256": receipt.checkpoint_sha256,
            "checkpoint_provenance_sha256": receipt.provenance_sha256,
            "calibration_sha256": calibration.identity_sha256,
            "provider_qualified": False, "memory_write_count": 0, "actuation_count": 0, "rows": rows}


if __name__ == "__main__":
    print(json.dumps(run_replay(), ensure_ascii=False, allow_nan=False, indent=2))
