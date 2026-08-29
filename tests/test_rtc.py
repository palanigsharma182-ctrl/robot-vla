import math

import numpy as np
import pytest

from robot_vla.execution.rtc import (
    ChunkInferenceStrategy,
    RTCConfig,
    build_rtc_trace,
    resolve_inference_strategy,
)


def test_rtc_eq5_slot_weights_use_zero_delay_soft_overlap_and_free_tail() -> None:
    config = RTCConfig(execution_horizon=4, max_guidance_weight=10.0)

    weights = config.slot_weights(16)

    assert np.all(np.diff(weights[:12]) < 0.0)
    np.testing.assert_allclose(weights[12:], 0.0)
    c_0 = 12 / 13
    assert weights[0] == pytest.approx(c_0 * math.expm1(c_0) / math.expm1(1.0))
    assert weights[0] > 0.75
    assert 0.3 < weights[3] < 0.5


def test_rtc_trace_measures_paired_raw_guided_and_future_change() -> None:
    config = RTCConfig()
    previous = np.zeros((12, 8), dtype=np.float32)
    raw = np.full((16, 8), 0.8, dtype=np.float32)
    guided = raw.copy()
    guided[:4] = 0.2

    trace = build_rtc_trace(
        config,
        action_horizon=16,
        previous_overlap=previous,
        raw_action=raw,
        guided_action=guided,
        denoising_guidance_coefficients=(10.0, 2.0),
    )

    assert trace.previous_chunk_available is True
    assert trace.overlap_length == 12
    assert trace.raw_mean_abs_disagreement == pytest.approx(0.8)
    assert trace.prefix_mean_abs_disagreement == pytest.approx(0.2)
    assert trace.prefix_mean_abs_correction == pytest.approx(0.6)
    assert trace.future_mean_abs_correction == pytest.approx(0.0)


def test_first_rtc_replan_has_no_previous_chunk_metrics() -> None:
    config = RTCConfig()
    action = np.zeros((16, 8), dtype=np.float32)

    trace = build_rtc_trace(
        config,
        action_horizon=16,
        previous_overlap=None,
        raw_action=action,
        guided_action=action,
    )

    assert trace.previous_chunk_available is False
    assert trace.overlap_length == 0
    assert trace.raw_mean_abs_disagreement is None


def test_strategy_resolution_preserves_legacy_temporal_boolean() -> None:
    assert resolve_inference_strategy(None) == ChunkInferenceStrategy.TEMPORAL_ENSEMBLE
    assert (
        resolve_inference_strategy(None, legacy_temporal_ensemble_enabled=False)
        == ChunkInferenceStrategy.NEWEST_ONLY
    )
    with pytest.raises(ValueError, match="冲突"):
        resolve_inference_strategy(
            ChunkInferenceStrategy.RTC,
            legacy_temporal_ensemble_enabled=True,
        )
