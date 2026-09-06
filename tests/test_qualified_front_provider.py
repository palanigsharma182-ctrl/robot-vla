"""D049 数值与输入拒绝测试；合成预测不构成真实 provider 资格。"""

from types import SimpleNamespace

import numpy as np
import pytest

from robot_vla.precision.qualified_front_provider import decode_d049_planar_prediction, verify_d049_bundle
from robot_vla.precision.active_front_memory_provider import ACTIVE_FRONT_HOME_BASE_FROM_EXTERNAL_CAMERA_CV


def prediction(*, sigma=(0.3, 0.4)):
    return SimpleNamespace(
        keypoints=SimpleNamespace(normalized_uv=np.array([[[0.5, 0.5], [0.8, 0.8]]]),
                                  peak_probability=np.array([[0.95, 0.95]]), normalized_entropy=np.array([[0.05, 0.05]])),
        visibility_probability=np.array([[0.95, 0.95]]), projection_validity_probability=np.array([0.95]),
        keypoint_sigma_px=np.array([[sigma, sigma]]),
        mask_probability=np.stack([np.full((128, 128), 0.95), np.full((128, 128), 0.05)])[None],
    )


def decode(value, *, scale=1., pose=None):
    return decode_d049_planar_prediction(value, covariance_scale=scale,
        intrinsic_cv=np.array([[128., 0., 63.5], [0., 128., 63.5], [0., 0., 1.]]),
        base_from_camera_cv=np.asarray(ACTIVE_FRONT_HOME_BASE_FROM_EXTERNAL_CAMERA_CV) if pose is None else pose)


def test_raw_sigma_score_is_independent_of_covariance_scale():
    raw, scaled = decode(prediction()), decode(prediction(), scale=4.)
    assert raw.evidence.score == pytest.approx(2./3.)
    assert scaled.evidence.score == raw.evidence.score
    assert scaled.covariance_base_m2 == pytest.approx(4.*raw.covariance_base_m2)
    assert scaled.position_base_m == pytest.approx(raw.position_base_m)
    assert scaled.position_base_m[2] == pytest.approx(0.02)
    assert np.count_nonzero(scaled.covariance_base_m2[2]) == 0


def test_goal_mask_is_sampled_at_object_uv_not_goal_uv():
    value = prediction()
    value.mask_probability[0, 1, 98:107, 98:107] = 1.
    assert decode(value).components.goal_mask_probability == pytest.approx(0.05)
    value.mask_probability[0, 1, 62:66, 62:66] = 1.
    assert decode(value).components.goal_mask_probability == 1.
    assert not decode(value).evidence.structurally_eligible


def test_correlated_covariance_uses_anisotropic_pixel_uncertainty():
    pose = np.asarray(ACTIVE_FRONT_HOME_BASE_FROM_EXTERNAL_CAMERA_CV).copy()
    angle = 0.4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    pose[:3, :3] = rotation @ pose[:3, :3]
    result = decode(prediction(sigma=(0.1, 0.8)), pose=pose)
    assert abs(result.covariance_base_m2[0, 1]) > 1e-8
    assert np.linalg.eigvalsh(result.covariance_base_m2).min() >= -1e-12


@pytest.mark.parametrize("pose", [np.eye(4), np.diag([1., -1., -1., 1.])])
def test_parallel_or_backward_geometry_cannot_be_accepted(pose):
    # 相机和固定平面的方向反向：第一种位于平面上方却向上看，第二种位于下方却向下看。
    pose = pose.copy()
    pose[2, 3] = 1. if pose[2, 2] == 1. else 0.
    result = decode(prediction(), pose=pose)
    assert result.position_base_m is None and result.covariance_base_m2 is None
    assert not result.evidence.structurally_eligible


def test_bad_masks_batches_or_scale_are_rejected():
    value = prediction()
    with pytest.raises(ValueError, match="scale"):
        decode(value, scale=0.5)
    value.keypoints.normalized_uv = np.repeat(value.keypoints.normalized_uv, 2, axis=0)
    with pytest.raises(ValueError, match="keypoints"):
        decode(value)
    value = prediction()
    value.mask_probability[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        decode(value)


def test_bundle_rejects_replaced_qualification_config_before_loading_weights(tmp_path):
    (tmp_path/"qualification_config.json").write_text('{}')
    with pytest.raises(ValueError, match="SHA-256"):
        verify_d049_bundle(tmp_path)


@pytest.fixture
def synthetic_provider(monkeypatch):
    """仅测试边界行为；明确绕过真实权重和资格包加载。"""
    from robot_vla.precision import qualified_front_provider as q
    from robot_vla.observation import GL_CAMERA_FROM_CV_CAMERA
    from robot_vla.precision.active_external_observation import extract_active_external_observation
    p = q.D049FrontProvider.__new__(q.D049FrontProvider)
    p.spec = q.RobotSpec()
    p.proprio = q.ProprioNormalizer(q.ProprioStats(mean=(0.,)*15, std=(1.,)*15, count=1), p.spec)
    p.force = q.FingerForceNormalizer(q.FingerForceStats(scale_log1p_p95=(1.,1.), count=1, positive_count=(1,1)), p.spec)
    value = prediction()
    p.predictor = SimpleNamespace(predict=lambda *a, **kw: value)
    p.forward_count = 0
    monkeypatch.setattr(p, "_verify_runtime_identity", lambda *a: None)
    pose = np.diag([1., -1., -1., 1.]); pose[2, 3] = .6
    intrinsic = np.array([[64.,0.,64.],[0.,64.,64.],[0.,0.,1.]])
    p.bundle_verification = {"camera_constraints": {q.ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID: (intrinsic.copy(), pose.copy())}}
    sidecar = extract_active_external_observation(
        {"sensor_data": {"base_camera": {"rgb": np.full((128,128,3), 127, np.uint8)}},
         "sensor_param": {"base_camera": {"intrinsic_cv": intrinsic, "cam2world_gl": pose@GL_CAMERA_FROM_CV_CAMERA}}},
        camera_uid="base_camera", world_from_robot_base=np.eye(4),
        commanded_world_from_external_camera_gl=pose@GL_CAMERA_FROM_CV_CAMERA,
        episode_id="synthetic", request_id="synthetic", observation_sequence_id="frame-0",
        camera_command_sequence_id="command-0", control_tick=0, control_timestamp_s=0.,
        rgb_timestamp_s=0., camera_pose_timestamp_s=0., camera_motion_state=q.Motion.COLLECT,
        viewpoint_primitive_id=q.ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID, settled=True,
        maximum_rotation_projection_error_frobenius=1e-6)
    args = dict(physical_proprio=np.array([0.]*14+[1.], np.float32), base_from_tcp=np.eye(4),
        finger_force_n=np.zeros(2, np.float32), tcp_timestamp_s=0., episode_generation=1,
        source_phase=q.PhaseId.FINE_ALIGN, safety=q.ObjectMemorySafetyContext(True,True,True,False,False,False,False,False),
        static_plane_scope_verified=True)
    return p, sidecar, args, value


@pytest.mark.parametrize("kind", ["object_mask", "goal_mask", "projection", "visibility"])
def test_rejected_predictions_still_form_consistent_frames(synthetic_provider, kind):
    from robot_vla.precision.active_front_memory_provider import ActiveFrontStage2ProviderAdapter, ActiveFrontStage2Config
    p, sidecar, args, value = synthetic_provider
    if kind == "object_mask": value.mask_probability[0, 0] = .1
    if kind == "goal_mask": value.mask_probability[0, 1] = .9
    if kind == "projection": value.projection_validity_probability[:] = .1
    if kind == "visibility": value.visibility_probability[0, 0] = .1
    frame = p.predict(sidecar, **args)
    adapter = ActiveFrontStage2ProviderAdapter(ActiveFrontStage2Config(enabled=True, memory_write_allowed=True))
    assert not adapter.adapt(frame, safety=args["safety"]).eligible
    assert p.forward_count == 1


@pytest.mark.parametrize("field,index", [
    ("intrinsic_cv", (0,0)), ("base_from_external_camera_cv", (0,3)),
    ("commanded_world_from_external_camera_gl", (0,3))])
def test_camera_drift_is_rejected_before_forward(synthetic_provider, field, index):
    p, sidecar, args, _ = synthetic_provider
    getattr(sidecar, field)[index] += .1
    with pytest.raises(ValueError, match="相机"):
        p.predict(sidecar, **args)
    assert p.forward_count == 0


def test_external_mutation_during_forward_does_not_change_bound_measurement(synthetic_provider):
    p, sidecar, args, value = synthetic_provider
    expected = p.predict(sidecar, **args)
    def mutate(*a, **kw):
        sidecar.base_from_external_camera_cv[0,3] += .25
        sidecar.rgb_external[:] = 0
        args["base_from_tcp"][0,3] += .25
        return value
    p.predictor.predict = mutate
    actual = p.predict(sidecar, **args)
    assert actual.model_input_digest == expected.model_input_digest
    assert actual.provider_output_digest == expected.provider_output_digest
    np.testing.assert_array_equal(actual.position_base_m, expected.position_base_m)


@pytest.mark.parametrize("previous_episode,generation", [(None, 1), (("episode", 1), 1), (("episode", 1), 2)])
def test_predictor_replacement_cannot_reuse_loaded_identity(monkeypatch, previous_episode, generation):
    from dataclasses import asdict, replace
    from robot_vla.precision import qualified_front_provider as q
    p = q.D049FrontProvider.__new__(q.D049FrontProvider)
    p.spec = q.RobotSpec()
    p.proprio = q.ProprioNormalizer(q.ProprioStats(mean=(0.,)*15, std=(1.,)*15, count=1), p.spec)
    p.force = q.FingerForceNormalizer(q.FingerForceStats(scale_log1p_p95=(1.,1.), count=1, positive_count=(1,1)), p.spec)
    identity = replace(q.d049_primary_provider_identity(),
        proprio_normalizer_sha256=q.canonical_sha256({"mean": p.proprio.mean.tolist(), "std": p.proprio.std.tolist(),
            "clip": p.proprio.clip, "robot_spec": p.spec.to_dict()}),
        finger_force_normalizer_sha256=q.canonical_sha256({"stats": asdict(p.force.stats), "scale": p.force.scale.tolist(),
            "clip": p.force.clip, "robot_spec": p.spec.to_dict()}))
    monkeypatch.setattr(q, "d049_primary_provider_identity", lambda: identity)
    verified = []
    p.predictor = SimpleNamespace(config="same-config", device=SimpleNamespace(type="cuda"),
        identity=SimpleNamespace(sha256="a"*64), verify_identity=lambda: verified.append(True))
    p._execution_config = p.predictor.config
    p._predictor_identity_sha256 = "a"*64
    p._episode_identity = previous_episode
    p._verify_runtime_identity("episode", generation)
    p.predictor = SimpleNamespace(config="same-config", device=SimpleNamespace(type="cuda"),
        identity=SimpleNamespace(sha256="b"*64), verify_identity=lambda: verified.append(True))
    with pytest.raises(ValueError, match="identity"):
        p._verify_runtime_identity("episode", generation)
