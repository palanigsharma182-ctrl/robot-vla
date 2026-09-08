"""防止把下降趋势、单指标平台或开发集退化误报为收敛。"""
import pytest
from experiments.tcp_teacher_continuation.convergence import METRICS, decision, plateau


def row(value):
    return dict(groups={s: {k: value for k in METRICS} for s in ('train', 'development')},
                selection_score=value)


def test_requires_three_consecutive_plateau_checks():
    assert decision([row(1)] * 9) == 'training'
    assert decision([row(1)] * 10) == 'offline_converged'


def test_falling_loss_is_not_converged():
    assert decision([row(.8**i) for i in range(20)]) == 'training'


def test_flat_loss_does_not_hide_action_improvement():
    history = [row(1) for _ in range(20)]
    for i, entry in enumerate(history):
        entry['groups']['train']['rotation_deg'] = .8**i
    assert decision(history) == 'training'


def test_large_oscillation_is_not_plateau():
    assert not plateau([row(1 if i % 2 else 2) for i in range(10)], 'train')


def test_small_stable_noise_is_plateau():
    assert decision([row(1 + .002*(i % 2)) for i in range(10)]) == 'offline_converged'


def test_regression_is_not_called_convergence():
    history = [row(1) for _ in range(10)]
    for i in range(4):
        entry = row(1)
        entry['groups']['development'] = {k: 1.5 + .1*i for k in METRICS}
        entry['selection_score'] = 1.5 + .1*i
        history.append(entry)
    assert decision(history) == 'development_regression'


def test_reject_nonfinite_metric():
    with pytest.raises(ValueError):
        plateau([row(float('nan')) for _ in range(8)], 'train')


def test_stably_degraded_development_is_not_convergence():
    history = [row(1)]
    for _ in range(10):
        entry = row(1)
        entry['groups']['development'] = {k: 2 for k in METRICS}
        entry['selection_score'] = 2
        history.append(entry)
    assert decision(history) == 'development_regression'
