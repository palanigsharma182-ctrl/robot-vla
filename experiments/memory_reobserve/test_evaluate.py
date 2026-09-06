"""开发评估的请求证据和每步预算边界；不启动仿真或 GPU。"""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.memory_reobserve.evaluate import PolicyAcquisition, EvaluationController, SEEDS


def acquisition(condition='evidence'):
    return PolicyAcquisition(SEEDS[0],None,condition,None,1,160,1000.)


def evidence(scores=(.1,.1,.1),arm_hold=True):
    return [(SimpleNamespace(episode_id='episode',object_measurement_usable=False,
        geometry_valid=True,home_capture_valid=True,pose_valid=True,timestamp_valid=True,
        stored_write_score=score,control_timestamp_s=tick/20,
        model_input_digest='a'*64,provider_output_digest='b'*64),arm_hold)
        for tick,score in zip((3,4,5),scores)]


def test_scene_cue_requires_three_consecutive_low_scores_and_actual_hold():
    assert acquisition().decide_request(evidence())
    assert not acquisition().decide_request(evidence((.1,.9,.1)))
    assert not acquisition().decide_request(evidence(arm_hold=False))
    assert not acquisition('visual').decide_request(evidence())
    assert acquisition('fixed').decide_request(evidence((.9,.9,.9)))


@pytest.mark.parametrize('field',['geometry_valid','home_capture_valid','pose_valid','timestamp_valid'])
def test_invalid_home_evidence_cannot_request(field):
    rows=evidence();setattr(rows[1][0],field,False)
    with pytest.raises(ValueError):acquisition().decide_request(rows)


def test_duplicate_home_tick_cannot_count_as_new_evidence():
    rows=evidence();rows[1][0].control_timestamp_s=rows[0][0].control_timestamp_s
    with pytest.raises(ValueError,match='连续'):acquisition().decide_request(rows)


def test_policy_budget_rejects_both_chunk_actions_and_direct_hold_send():
    controller=EvaluationController.__new__(EvaluationController)
    controller.tick=20;controller.stop_tick=20;controller.episode_done=False
    assert controller.should_interrupt_before_action(np.zeros(8))
    with pytest.raises(RuntimeError,match='budget'):controller.send_action(np.zeros(8))
    assert controller.tick==20
