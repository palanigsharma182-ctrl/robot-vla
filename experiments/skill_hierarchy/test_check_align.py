"""Align 诊断的轴向、停止优先级与异常动作计数检查。"""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from experiments.skill_hierarchy.check_align import AlignController, PreparationMotion, perturbation_target, rotation_error
from experiments.tcp_atomic_skills.runtime import AtomicController


def test_targets_use_approach_direction_and_preserve_nominal():
    nominal = np.eye(4); target = np.eye(4); target[1, 3] = .04
    inward, axis = perturbation_target(nominal, target, 'position_in10')
    outward, _ = perturbation_target(nominal, target, 'position_out10')
    np.testing.assert_allclose(axis, [0, 1, 0])
    np.testing.assert_allclose(inward[:3, 3], [0, .01, 0])
    np.testing.assert_allclose(outward[:3, 3], [0, -.01, 0])
    turn, _ = perturbation_target(nominal, target, 'yaw_pos10')
    assert rotation_error(turn, nominal) == pytest.approx(10)
    np.testing.assert_array_equal(nominal, np.eye(4))
    with pytest.raises(ValueError):
        perturbation_target(nominal, nominal, 'standard')


def test_velocity_preparation_keeps_speed_and_compensates_endpoint():
    nominal = np.eye(4); pregrasp = np.eye(4); pregrasp[0, 3] = .04
    motion = PreparationMotion(None, None); commands = []
    motion.move = lambda pose, steps=12: commands.append((pose.copy(), steps))
    motion.hold = lambda pose: None
    goal, velocity = motion.prepare(nominal, pregrasp, 'velocity_in05')
    np.testing.assert_array_equal(goal, nominal)
    np.testing.assert_allclose(velocity, [.05, 0, 0])
    start, end = commands[0][0], commands[1][0]
    np.testing.assert_allclose((end[:3, 3]-start[:3, 3])/.30, velocity)
    assert end[0, 3] == pytest.approx(.00375)
    assert commands[1][1] == 6


@pytest.mark.parametrize('prior,active,fault,expected', [
    (None, 1, None, None),
    (None, 2, None, 'success'),
    ('tracking-invalid', 2, None, 'tracking-invalid'),
    ('step-budget-exhausted', 1, None, 'step-budget-exhausted'),
    ('step-budget-exhausted', 2, None, 'success'),
    (None, 1, 'align: opening', 'align-invariant-failure'),
])
def test_boundary_failure_stops_without_throwing_to_hold(
        monkeypatch, tmp_path, prior, active, fault, expected):
    monkeypatch.setattr(AtomicController, 'send_action', lambda self, value: None)
    def observe(metrics, *_):
        if fault:
            raise RuntimeError(fault)
        return metrics
    ctrl = AlignController.__new__(AlignController)
    ctrl.teacher = SimpleNamespace(measure=lambda **kw: {}, relative_pose=None,
                                   boundaries=SimpleNamespace(active=active, observe=observe))
    ctrl.stop_reason, ctrl.steps, ctrl.output = prior, 1, tmp_path
    ctrl.send_action([0.]*7+[1.])
    assert ctrl.stop_reason == expected
    assert ctrl.chunk_stop_requested == (expected is not None)
    rows = (tmp_path/'metrics.jsonl').read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])['fault'] == fault
