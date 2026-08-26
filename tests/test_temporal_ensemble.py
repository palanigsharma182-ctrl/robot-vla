import numpy as np
import pytest

from robot_vla.contracts import RobotSpec
from robot_vla.execution.temporal_ensemble import (
    TemporalChunkEnsembler,
    TemporalEnsembleConfig,
)


def _chunk(spec: RobotSpec, value: float) -> np.ndarray:
    return np.full(
        (spec.action_horizon, spec.action_dim),
        value,
        dtype=np.float32,
    )


def test_first_chunk_passes_through_without_modification() -> None:
    spec = RobotSpec()
    ensembler = TemporalChunkEnsembler(spec)
    chunk = _chunk(spec, 0.25)

    output = ensembler.add_and_compose(chunk, origin_control_step=0)

    np.testing.assert_allclose(output.normalized_action, chunk)
    assert output.trace.buffer_size == 1
    assert output.trace.proposal_counts == (1,) * spec.action_horizon
    assert output.trace.newest_normalized_weights == (1.0,) * spec.action_horizon


def test_overlapping_chunks_are_aligned_by_global_step_and_newest_dominates() -> None:
    spec = RobotSpec()
    ensembler = TemporalChunkEnsembler(
        spec,
        TemporalEnsembleConfig(recency_decay=0.5),
    )
    old = np.zeros((spec.action_horizon, spec.action_dim), dtype=np.float32)
    old[:, 0] = np.arange(spec.action_horizon, dtype=np.float32) / 16.0
    ensembler.add_and_compose(old, origin_control_step=0)
    new = _chunk(spec, 1.0)

    output = ensembler.add_and_compose(new, origin_control_step=4)

    expected_first = (1.0 + 0.5 * old[4, 0]) / 1.5
    assert output.normalized_action[0, 0] == pytest.approx(expected_first)
    assert output.trace.proposal_counts[:12] == (2,) * 12
    assert output.trace.proposal_counts[12:] == (1,) * 4
    assert output.trace.newest_normalized_weights[0] == pytest.approx(2.0 / 3.0)
    assert output.normalized_action[0, 0] > old[4, 0]


def test_four_overlapping_chunks_give_newest_more_weight_than_all_old_chunks() -> None:
    spec = RobotSpec()
    ensembler = TemporalChunkEnsembler(spec)
    for origin, value in ((0, 0.0), (4, 0.2), (8, 0.4), (12, 1.0)):
        output = ensembler.add_and_compose(_chunk(spec, value), origin_control_step=origin)

    assert output.trace.proposal_counts[0] == 4
    assert output.trace.newest_normalized_weights[0] == pytest.approx(1.0 / 1.875)
    assert output.trace.newest_normalized_weights[0] > 0.5


def test_clear_removes_stale_proposals() -> None:
    spec = RobotSpec()
    ensembler = TemporalChunkEnsembler(spec)
    ensembler.add_and_compose(_chunk(spec, -1.0), origin_control_step=0)
    ensembler.clear()

    output = ensembler.add_and_compose(_chunk(spec, 1.0), origin_control_step=4)

    assert output.trace.buffer_size == 1
    np.testing.assert_allclose(output.normalized_action, 1.0)
