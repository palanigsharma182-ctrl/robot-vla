from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("mani_skill")
pytest.importorskip("mplib")

from robot_vla.contracts import RobotSpec
from robot_vla.observation import OBSERVATION_MODALITIES, ObservationV2Frame
from robot_vla.precision.shadow import (
    PrecisionShadowObserver,
    _synthetic_warmup_window,
    _warm_up_provider,
)


def _valid_arm_q() -> np.ndarray:
    return np.asarray((0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0), dtype=np.float32)


class _FakeProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.windows = []
        self._records = []
        self.reset_count = 0

    @property
    def records(self):
        return tuple(self._records)

    @property
    def records_sha256(self) -> str:
        payload = f"records={len(self._records)}".encode()
        return hashlib.sha256(payload).hexdigest()

    def reset(self) -> None:
        self.reset_count += 1
        self.windows.clear()
        self._records.clear()

    def __call__(self, window):
        self.windows.append(window)
        if self.fail:
            raise RuntimeError("injected provider failure")
        self._records.append(
            SimpleNamespace(
                detections_count=int(window.history_valid.sum()),
                success=True,
                total_latency_s=0.01,
            )
        )
        return ()


def _frame(spec: RobotSpec, timestep: int) -> ObservationV2Frame:
    timestamp = timestep / spec.control_hz
    return ObservationV2Frame(
        rgb_external=np.zeros((8, 8, 3), dtype=np.uint8),
        rgb_wrist=np.full((8, 8, 3), timestep, dtype=np.uint8),
        physical_proprio=np.zeros(spec.proprio_dim, dtype=np.float32),
        base_from_tcp=np.eye(4, dtype=np.float64),
        base_from_wrist_camera=np.eye(4, dtype=np.float64),
        finger_force_n=np.zeros(2, dtype=np.float32),
        timestamp_s=timestamp,
        modality_timestamp_s=np.full(
            len(OBSERVATION_MODALITIES),
            timestamp,
            dtype=np.float64,
        ),
        modality_valid=np.ones(len(OBSERVATION_MODALITIES), dtype=np.bool_),
    )


def test_precision_shadow_observer_builds_four_frame_window_without_action() -> None:
    spec = RobotSpec()
    provider = _FakeProvider()
    observer = PrecisionShadowObserver(spec, provider, deadline_s=0.05)

    observer.reset()
    result0 = observer.observe(
        _frame(spec, 0),
        previous_command_q=_valid_arm_q(),
        previous_action=None,
    )
    result1 = observer.observe(
        _frame(spec, 1),
        previous_command_q=_valid_arm_q(),
        previous_action=np.zeros(spec.action_dim, dtype=np.float32),
    )

    assert result0 is None and result1 is None
    assert provider.reset_count == 1
    assert provider.windows[0].history_valid.tolist() == [False, False, False, True]
    assert provider.windows[1].history_valid.tolist() == [False, False, True, True]
    receipt = observer.receipt(seed=132000, observer_error_count=0)
    assert receipt.provider_call_count == 2
    assert receipt.predicted_frame_count == 3
    assert receipt.provider_failure_count == 0
    assert receipt.deadline_miss_count == 0


def test_precision_shadow_observer_records_wall_time_when_provider_raises() -> None:
    spec = RobotSpec()
    observer = PrecisionShadowObserver(
        spec,
        _FakeProvider(fail=True),
        deadline_s=0.05,
    )

    observer.reset()
    with pytest.raises(RuntimeError, match="injected provider failure"):
        observer.observe(
            _frame(spec, 0),
            previous_command_q=_valid_arm_q(),
            previous_action=None,
        )

    assert len(observer.wall_latency_s) == 1
    assert observer.wall_latency_s[0] >= 0.0


def test_precision_shadow_provider_warmup_is_four_frame_and_clears_ledger() -> None:
    spec = RobotSpec()
    provider = _FakeProvider()
    window = _synthetic_warmup_window(spec, image_size_hw=(8, 8))

    completed = _warm_up_provider(provider, window, calls=3)

    assert completed == 3
    assert window.history_valid.tolist() == [True, True, True, True]
    assert provider.reset_count == 2
    assert provider.records == ()
