"""Memory 条件实验的时间、掩码、梯度与观察权限回归。"""
from dataclasses import replace

import numpy as np
import pytest
import torch

from conditioning import (
    MEMORY_INPUT_KEY, MemoryBatch, snapshot_memory,
)
from probe import build_probe, run_probe
from robot_vla.executive.contracts import PhaseId
from robot_vla.precision.active_front_reobserve import (
    ActiveFrontReobserveConfig, ActiveFrontReobserveController, ActiveFrontTriggerEvidence,
    ActiveFrontTriggerReason,
)
from robot_vla.precision.object_memory import (
    ObjectMemoryConfig, ObjectMemoryMode, ObjectMemorySafetyContext, ObjectState,
)


def fixture_state():
    config = ObjectMemoryConfig(2.5, .01, .02, 3, .075, .005, .01, 'base_camera', 'model')
    safety = ObjectMemorySafetyContext(True, True, True, False, False, False, False, False)
    state = ObjectState(
        episode_id='probe', mode=ObjectMemoryMode.FREE_STATIC,
        position_base_m=(.4, .1, .03), covariance_base_m2=np.eye(3)*1e-4,
        measurement_confidence=.8, last_observed_timestamp_s=3.7, state_timestamp_s=5.9,
        observable_now=False, valid=True, accepted_update_count=1, source_camera='base_camera',
        source_model_identity='model', invalid_reasons=(),
    )
    return state, config, safety


def snapshot(state=None, config=None, safety=None, timestamp=5.9):
    s, c, safe = fixture_state()
    return snapshot_memory(state or s, config or c, safety or safe,
                           episode_id='probe', timestamp_s=timestamp)


def test_snapshot_age_covariance_and_immutability():
    value = snapshot()
    assert value.available and value.features[-3] == pytest.approx(2.2/2.5)
    assert value.features[3:9] == pytest.approx((.25, 0, 0, .25, 0, .25))
    assert value.features[-1] == 0
    with pytest.raises(AttributeError):
        value.timestamp_s = 9
    assert not snapshot(timestamp=6.21).available
    assert value.available  # 旧快照保持原值，但不能被下一次规划冒充新快照。


@pytest.mark.parametrize('event', ['object_contact_detected', 'gripper_close_commanded',
                                 'grasp_candidate', 'grasp_verified', 'object_maybe_moved'])
def test_contact_events_mask_old_position(event):
    _, _, safety = fixture_state()
    value = snapshot(safety=replace(safety, **{event: True}))
    assert not value.available and not any(value.features)


def test_future_cross_episode_and_covariance_growth():
    state, config, safety = fixture_state()
    with pytest.raises(ValueError):
        snapshot(timestamp=5.8)
    with pytest.raises(ValueError):
        snapshot_memory(state, config, safety, episode_id='another', timestamp_s=6)
    value = snapshot(config=replace(config, covariance_growth_m2_per_s=.01), timestamp=6)
    assert 'memory_uncertain' in value.reasons


def test_source_invalid_state_and_current_visibility():
    state, _, _ = fixture_state()
    assert not snapshot(state=replace(state, source_model_identity='other')).available
    invalid = replace(state, valid=False, mode=ObjectMemoryMode.INVALID,
                      invalid_reasons=('measurement_conflict',))
    assert not snapshot(state=invalid).available
    current = replace(state, observable_now=True, last_observed_timestamp_s=5.9)
    assert snapshot(state=current).features[-1] == 1
    assert snapshot(state=current, timestamp=6).features[-1] == 0


def test_real_expert_flow_and_gradient_probe():
    result = run_probe()
    assert result['action_shape'] == [2, 16, 8]
    assert result['training_steps'] == result['actuator_steps'] == 0


def test_mixed_batch_mask_blocks_inactive_values_and_does_not_persist():
    baseline, candidate, inputs, proprio = build_probe()
    mask = torch.tensor([[True], [False]])
    a = torch.zeros(2, 12)
    b = a.clone(); b[1] = 1e8
    def sample(values):
        return candidate.sample_actions({**inputs, MEMORY_INPUT_KEY: MemoryBatch(values, mask)},
            proprio, num_steps=2, generator=torch.Generator().manual_seed(19))
    assert torch.equal(sample(a), sample(b))
    original = baseline.sample_actions(inputs, proprio, num_steps=2,
                                      generator=torch.Generator().manual_seed(19))
    torch.testing.assert_close(sample(a)[1], original[1], rtol=1e-6, atol=1e-6)
    assert MEMORY_INPUT_KEY not in inputs
    assert candidate.encode_context(inputs).tokens.shape[1] == 5
    assert baseline.encode_context(inputs).tokens.shape[1] == 5


def test_rtc_and_plain_flow_share_memory_conditioned_context():
    _, candidate, inputs, proprio = build_probe()
    memory = MemoryBatch(torch.ones(2, 12)*.1, torch.ones(2, 1, dtype=torch.bool))
    payload = {**inputs, MEMORY_INPUT_KEY: memory}
    plain = candidate.sample_actions(payload, proprio, num_steps=2,
                                    generator=torch.Generator().manual_seed(9))
    rtc = candidate.sample_actions_rtc(
        payload, proprio, torch.zeros(2, 16, 8), torch.zeros(2, 16), num_steps=2,
        generator=torch.Generator().manual_seed(9),
    )
    torch.testing.assert_close(rtc.raw_action, plain, rtol=0, atol=0)
    torch.testing.assert_close(rtc.guided_action, plain, rtol=0, atol=0)


def test_masked_sample_has_no_memory_feature_gradient():
    _, candidate, inputs, proprio = build_probe()
    values = torch.zeros(2, 12, requires_grad=True)
    payload = {**inputs, MEMORY_INPUT_KEY: MemoryBatch(values, torch.tensor([[True], [False]]))}
    loss = candidate.flow_matching_loss(
        payload, proprio, torch.zeros(2, 16, 8), torch.ones(2, 16, dtype=torch.bool),
        generator=torch.Generator().manual_seed(7),
    ).loss
    loss.backward()
    assert float(values.grad[0].abs().sum()) > 0
    assert torch.equal(values.grad[1], torch.zeros(12))


@pytest.mark.parametrize('kind', ['nan', 'shape', 'mask'])
def test_malformed_input_rejected(kind):
    _, candidate, inputs, _ = build_probe()
    values = torch.zeros(2, 12); mask = torch.ones(2, 1, dtype=torch.bool)
    if kind == 'nan': values[0, 0] = float('nan')
    if kind == 'shape': values = torch.zeros(2, 11)
    if kind == 'mask': mask = mask.float()
    with pytest.raises(ValueError):
        candidate.encode_context({**inputs, MEMORY_INPUT_KEY: MemoryBatch(values, mask)})


@pytest.mark.parametrize('case', ['unavailable', 'memory', 'contact', 'phase', 'not_observed'])
def test_observation_permission_stays_with_existing_supervisor(case):
    controller = ActiveFrontReobserveController(ActiveFrontReobserveConfig(enabled=True))
    controller.reset_episode('probe', episode_generation=1)
    value = snapshot() if case == 'memory' else snapshot(timestamp=6.3)
    decisions = []
    for tick in range(3):
        evidence = ActiveFrontTriggerEvidence(
            episode_id='probe', episode_generation=1, control_tick=tick,
            timestamp_s=6.3+tick*.05,
            source_phase=PhaseId.FINAL_APPROACH if case == 'phase' else PhaseId.ACQUIRE_TRACK,
            wrist_object_measurement_usable=False, front_home_object_measurement_usable=False,
            object_memory_navigation_state_available=value.available,
            arm_hold_prerequisites_pass=True, camera_home_prerequisites_pass=True,
            failure_reason=(ActiveFrontTriggerReason.NO_QUALIFIED_WRIST_PROVIDER_IN_PARENT if case == 'not_observed'
                            else ActiveFrontTriggerReason.OBJECT_OCCLUSION),
            object_contact=case == 'contact',
        )
        decisions.append(controller.consider_trigger(evidence))
    assert decisions[-1].requestable == (case == 'unavailable')
    assert not decisions[0].requestable  # 不凭单帧失败请求相机运动。
