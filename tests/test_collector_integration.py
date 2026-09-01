from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("mani_skill")
pytest.importorskip("mplib")

from robot_vla.contracts import RobotSpec
from robot_vla.data.trajectory import TrajectoryStore, load_manifest
from robot_vla.sim.collector import RECOVERY_PROFILES, TrustedPickPlaceCollector


def test_collector_writes_real_success_trajectory_without_reading_privileged_gt(
    tmp_path,
    monkeypatch,
) -> None:
    with TrustedPickPlaceCollector(tmp_path) as collector:
        monkeypatch.setattr(
            collector,
            "_read_precision_positions_base",
            lambda: pytest.fail("未启用 sidecar 时禁止读取 object/goal GT pose"),
        )
        meta = collector.collect(seed=0, split="train")

    arrays = TrajectoryStore(tmp_path, RobotSpec()).get(meta)
    assert arrays.success[-1]
    assert set(arrays.skill_id.tolist()) == {0, 1, 2, 3, 4}
    assert meta.outcome_evidence is not None
    assert meta.outcome_evidence.both_goal_visible_steps > 0
    assert load_manifest(tmp_path) == [meta]


def test_collector_isolates_shadow_observer_failure_from_expert_step() -> None:
    class RaisingObserver:
        def observe(self, frame, *, previous_command_q, previous_action) -> None:
            assert frame.rgb_wrist.shape == (8, 8, 3)
            assert previous_command_q.shape == (RobotSpec().arm_dof,)
            assert previous_action is None
            raise RuntimeError("injected observer failure")

    collector = object.__new__(TrustedPickPlaceCollector)
    collector.spec = RobotSpec()
    collector.shadow_observer = RaisingObserver()
    collector.shadow_observer_errors = []
    recorder = SimpleNamespace(
        action=[np.zeros(RobotSpec().action_dim, dtype=np.float32)],
        previous_action=[],
        previous_command_q_rad=[np.zeros(RobotSpec().arm_dof, dtype=np.float32)],
        rgb_external=[np.zeros((8, 8, 3), dtype=np.uint8)],
        rgb_wrist=[np.zeros((8, 8, 3), dtype=np.uint8)],
        proprio=[np.zeros(RobotSpec().proprio_dim, dtype=np.float32)],
        left_finger_force_n=[0.0],
        right_finger_force_n=[0.0],
    )
    session = SimpleNamespace(recorder=recorder)

    collector._observe_shadow(
        session,
        np.eye(4, dtype=np.float64),
        np.eye(4, dtype=np.float64),
    )

    assert collector.shadow_observer_errors == [
        {"type": "RuntimeError", "message": "injected observer failure"}
    ]


@pytest.mark.parametrize("recovery_profile", RECOVERY_PROFILES)
def test_collector_writes_auditable_recovery_trajectory(tmp_path, recovery_profile) -> None:
    seed = RECOVERY_PROFILES.index(recovery_profile) + 30
    with TrustedPickPlaceCollector(tmp_path) as collector:
        meta = collector.collect(
            seed=seed,
            split="train",
            recovery_profile=recovery_profile,
        )

    arrays = TrajectoryStore(tmp_path, RobotSpec()).get(meta)
    assert arrays.success[-1]
    assert set(arrays.skill_id.tolist()) == {0, 1, 2, 3, 4}
    assert meta.randomization["recovery_profile"] == recovery_profile
    evidence = meta.randomization["recovery_evidence"]
    assert 0 <= evidence["disturbance_end_step"]
    assert evidence["disturbance_end_step"] < evidence["successful_recovery_end_step"]
    assert evidence["successful_recovery_end_step"] == arrays.num_steps - 1
