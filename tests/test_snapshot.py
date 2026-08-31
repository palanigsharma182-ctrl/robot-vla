from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from robot_vla.sim.snapshot import (
    CollectionSnapshotRing,
    SnapshotContractError,
    capture_collection_snapshot,
    restore_collection_snapshot,
)


class _FakeBatchedRNG:
    def __init__(self, seed: int) -> None:
        self.rngs = [np.random.RandomState(seed)]


class _FakeController:
    def __init__(self, width: int) -> None:
        self._step = 5
        self._start_qpos = np.zeros((1, width), dtype=np.float32)
        self._target_qpos = np.full((1, width), 0.25, dtype=np.float32)


class _FakeBaseEnv:
    def __init__(self) -> None:
        self.agent = SimpleNamespace(
            controller=SimpleNamespace(
                controllers={
                    "arm": _FakeController(7),
                    "gripper": _FakeController(2),
                }
            )
        )
        self._state = {
            "actors": {"cube": np.arange(13, dtype=np.float32)[None]},
            "articulations": {"robot": np.arange(31, dtype=np.float32)[None]},
        }
        self._batched_main_rng = _FakeBatchedRNG(10)
        self._batched_episode_rng = _FakeBatchedRNG(20)
        self._episode_rng = self._batched_episode_rng.rngs[0]
        self._episode_seed = np.asarray([29_990], dtype=np.int64)
        self._elapsed_steps = np.asarray([12], dtype=np.int32)
        self._max_episode_steps = 300
        self._last_obs = _observation()

    @property
    def unwrapped(self):
        return self

    def get_state_dict(self):
        return self._state

    def set_state_dict(self, state):
        self._state = state


@dataclass(frozen=True)
class _Predicate:
    completed: int
    is_grasped: bool


class _Tracker:
    def __init__(self) -> None:
        self._active_skill_id = 1
        self._stable_grasp_steps = 0
        self._stable_place_steps = 0
        self._task_completed = False


def _observation() -> dict:
    return {
        "sensor_data": {
            "base_camera": {
                "rgb": np.zeros((1, 4, 5, 3), dtype=np.uint8),
                "segmentation": np.ones((1, 4, 5, 1), dtype=np.int16),
            },
            "hand_camera": {
                "rgb": np.full((1, 3, 4, 3), 2, dtype=np.uint8),
                "segmentation": np.full((1, 3, 4, 1), 3, dtype=np.int16),
            },
        }
    }


def _session() -> SimpleNamespace:
    return SimpleNamespace(
        observation=_observation(),
        tracker=_Tracker(),
        progress={"completed_skill_count": 1},
        previous_command_q=np.arange(7, dtype=np.float32) / 10.0,
        done=False,
    )


def _loop() -> SimpleNamespace:
    stored_chunk = SimpleNamespace(
        origin_control_step=0,
        sequence_id=0,
        normalized_action=np.zeros((16, 8), dtype=np.float32),
    )
    return SimpleNamespace(
        control_step=8,
        _consecutive_anomaly_replans=1,
        _rtc_previous_chunk=None,
        runtime=SimpleNamespace(_sample_index=3, _last_sampling_trace={"seed": 4}),
        ensembler=SimpleNamespace(_chunks=[stored_chunk], _next_sequence_id=1),
    )


def _capture(env, session, loop, *, replan_index: int):
    return capture_collection_snapshot(
        env,
        session,
        loop,
        label="boundary_crossing" if replan_index == 2 else "replan_start",
        replan_index=replan_index,
        environment_seed=29_990,
        physical_proprio=np.arange(15, dtype=np.float32) / 20.0,
        predicate_state=_Predicate(completed=1, is_grasped=False),
        contact_forces_n=(0.2, 1.5),
        camera_calibration={"version": "test/v1"},
        collection_policy_identity={
            "checkpoint_sha256": "a" * 64,
            "inference_strategy": "temporal-ensemble",
        },
    )


def test_snapshot_restores_environment_rng_tracker_and_temporal_buffer() -> None:
    env = _FakeBaseEnv()
    session = _session()
    loop = _loop()
    snapshot = _capture(env, session, loop, replan_index=2)
    expected_main_draw = env._batched_main_rng.rngs[0].randint(2**31)
    expected_episode_draw = env._batched_episode_rng.rngs[0].randint(2**31)

    env._state["actors"]["cube"][:] = -1
    env.agent.controller.controllers["arm"]._target_qpos[:] = -1
    env._elapsed_steps[:] = 99
    session.tracker._active_skill_id = 4
    session.previous_command_q[:] = -1
    loop.control_step = 100
    loop.runtime._sample_index = 100
    loop.ensembler._chunks.clear()

    restore_collection_snapshot(
        snapshot,
        env,
        session=session,
        loop=loop,
        restore_global_rng=False,
    )

    np.testing.assert_array_equal(
        env._state["actors"]["cube"],
        np.arange(13, dtype=np.float32)[None],
    )
    np.testing.assert_allclose(
        env.agent.controller.controllers["arm"]._target_qpos,
        0.25,
    )
    np.testing.assert_array_equal(env._elapsed_steps, [12])
    assert env._batched_main_rng.rngs[0].randint(2**31) == expected_main_draw
    assert env._batched_episode_rng.rngs[0].randint(2**31) == expected_episode_draw
    assert session.tracker._active_skill_id == 1
    np.testing.assert_allclose(session.previous_command_q, np.arange(7) / 10.0)
    assert loop.control_step == 8
    assert loop.runtime._sample_index == 3
    assert len(loop.ensembler._chunks) == 1
    assert snapshot.to_summary()["policy"]["temporal_buffer_size"] == 1


def test_snapshot_ring_keeps_predecessor_crossing_replan_and_boundary() -> None:
    env = _FakeBaseEnv()
    session = _session()
    loop = _loop()
    ring = CollectionSnapshotRing()
    for index in range(3):
        ring.append(_capture(env, session, loop, replan_index=index))

    assert [snapshot.replan_index for snapshot in ring.snapshots] == [0, 1, 2]
    ring.append(_capture(env, session, loop, replan_index=3))
    assert [snapshot.replan_index for snapshot in ring.snapshots] == [1, 2, 3]

    with pytest.raises(SnapshotContractError, match="不能倒退"):
        ring.append(_capture(env, session, loop, replan_index=2))
