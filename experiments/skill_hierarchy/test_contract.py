"""候选边界的关键反例；仿真验证单独在云端执行。"""
import math
import numpy as np
import pytest

from experiments.skill_hierarchy.contract import (
    SKILLS, OPERATIONS, ENTRY, EXIT, BoundaryTracker, cube_orientation_error_deg,
    invariant_violations, violations,
)


def metrics(skill):
    return {k: (lo+hi)/2 for k, (lo, hi) in EXIT[skill].items()}


def test_hierarchy_and_legacy_separation():
    from robot_vla.tasks.pick_place import ATOMIC_PICK_PLACE_SKILLS
    assert len(ATOMIC_PICK_PLACE_SKILLS) == 5
    assert len(SKILLS) == 7
    assert OPERATIONS == ('pick',)*4+('carry',)+('place',)*2


@pytest.mark.parametrize('index', range(6))
def test_entire_exit_envelope_is_covered_by_next_entry(index):
    previous = EXIT[SKILLS[index]]
    for key, (lo, hi) in ENTRY[SKILLS[index+1]].items():
        assert key in previous
        assert lo <= previous[key][0] <= previous[key][1] <= hi


@pytest.mark.parametrize('skill', SKILLS)
def test_threshold_inclusive_and_missing_nan_rejected(skill):
    for key, (lo, hi) in EXIT[skill].items():
        for boundary in (lo, hi):
            m = metrics(skill); m[key] = boundary
            assert not violations(EXIT[skill], m)
        for invalid in (lo-1e-8, hi+1e-8, math.nan, math.inf):
            m = metrics(skill); m[key] = invalid
            assert key in violations(EXIT[skill], m)
        m = metrics(skill); del m[key]
        assert key in violations(EXIT[skill], m)


def test_cube_equivalent_yaw_does_not_hide_tilt():
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
    assert cube_orientation_error_deg(rz, np.eye(3)) == 0
    tilt = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
    assert cube_orientation_error_deg(tilt, np.eye(3)) == pytest.approx(90)


def test_relative_grasp_window_does_not_accept_slow_cumulative_slip():
    tracker = BoundaryTracker(active=2)
    m = metrics('grasp')
    for i in range(3):
        pose = np.eye(4); pose[0, 3] = i*.0015
        out = tracker.observe(m, pose, i*.05)
    assert out['grasp_stable'] == 0  # 相邻 1.5mm 合格，但窗口累计 3mm 不合格。
    assert tracker.active == 2
    for i in range(3, 6):
        out = tracker.observe(m, pose, i*.05)
    assert out['grasp_stable'] == 1
    assert tracker.active == 3


def test_grasp_lost_overrides_historical_success():
    tracker = BoundaryTracker(active=4, events=[{'skill': 'grasp'}])
    with pytest.raises(RuntimeError, match='grasp_lost'):
        tracker.observe(dict(metrics('transport'), held=0), np.eye(4), 0)
    assert tracker.active == 4


def test_window_reset_after_loss_and_no_repeated_frames():
    tracker = BoundaryTracker(active=2)
    m = metrics('grasp')
    tracker.observe(m, np.eye(4), 0)
    tracker.observe(dict(m, held=0), np.eye(4), .05)
    out = tracker.observe(m, np.eye(4), .10)
    assert not out['grasp_stable']
    with pytest.raises(ValueError, match='20 Hz'):
        tracker.observe(m, np.eye(4), .10)


def test_release_requires_command_and_four_consecutive_frames():
    tracker = BoundaryTracker(active=6)
    m = metrics('release')
    tracker.observe(dict(m, release_commanded=0), np.eye(4), 0)
    for i in range(1, 4):
        tracker.observe(m, np.eye(4), i*.05)
        assert tracker.active == 6
    tracker.observe(dict(m, object_speed_m_s=.02), np.eye(4), .20)
    for i in range(5, 9):
        tracker.observe(m, np.eye(4), i*.05)
    assert tracker.active == 7


def test_invalid_relative_pose_rejected():
    tracker = BoundaryTracker()
    pose = np.eye(4); pose[0, 0] = 2
    with pytest.raises(ValueError, match=r'SE\(3\)'):
        tracker.observe(metrics('approach'), pose, 0)


def test_sequential_boundaries_label_finishing_action_as_previous_skill():
    tracker = BoundaryTracker()
    # 对应探针的 before-action 标签规则：完成动作归旧技能，下一动作归新技能。
    before_action_id = tracker.active
    tracker.observe(metrics('approach'), np.eye(4), 0)
    assert before_action_id == 0
    assert tracker.active == 1
    assert len(tracker.events) == 1
