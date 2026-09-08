"""旧日志不含派生tracking字段时仍能准确保留停止结果。"""
import pytest
from experiments.tcp_teacher_closed_loop.metrics import trajectory_metrics, canonical_audit


def row(step,distance,holding=False):
    return dict(policy_step=step,distance_m=distance,holding=holding,
        command_q=[.1]*7,q_after=[.08]*7,action_used_memory=True,occluded=False)


def test_legacy_trace_tracking_and_denominator():
    result=trajectory_metrics([row(0,.1),row(1,.08),row(2,.06)],.1,[{},{}])
    assert result['policy_steps']==2 and result['replans']==2
    assert result['tracking_error_max_rad']==pytest.approx(.02)
    assert not result['reached'] and result['memory_actions']==2


def test_holds_excluded_from_reach_and_memory_metrics():
    result=trajectory_metrics([row(1,.08),row(1,.001,holding=True)],.1,[{}])
    assert not result['reached'] and result['minimum_distance_m']==.08
    assert result['final_distance_m']==.001 and result['memory_actions']==1


def test_zero_policy_step_still_has_valid_denominator():
    result=trajectory_metrics([row(0,.1,holding=True)],.1,[{}])
    assert result['policy_steps']==0 and result['tracking_error_max_rad'] is None
    assert not result['reached'] and result['distance_by_policy_step']=={}


def test_initial_pair_survives_disk_without_relaxing_values():
    import json
    from experiments.rgbd_memory_policy.evaluate import verify_initial_pair
    live=dict(snapshot=dict(features=(0.,)*12,reasons=('unavailable',)),q=[.1]*7,input_digest='same')
    disk=json.loads(json.dumps(live))
    verify_initial_pair(disk,canonical_audit(live))
    live['q'][0]+=.000001
    with pytest.raises(ValueError):verify_initial_pair(disk,canonical_audit(live))
