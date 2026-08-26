from __future__ import annotations

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("mani_skill")
pytest.importorskip("mplib")

from robot_vla.contracts import RobotSpec
from robot_vla.data.trajectory import TrajectoryStore, load_manifest
from robot_vla.sim.collector import RECOVERY_PROFILES, TrustedPickPlaceCollector


def test_collector_writes_real_success_trajectory(tmp_path) -> None:
    with TrustedPickPlaceCollector(tmp_path) as collector:
        meta = collector.collect(seed=0, split="train")

    arrays = TrajectoryStore(tmp_path, RobotSpec()).get(meta)
    assert arrays.success[-1]
    assert set(arrays.skill_id.tolist()) == {0, 1, 2, 3, 4}
    assert meta.outcome_evidence is not None
    assert meta.outcome_evidence.both_goal_visible_steps > 0
    assert load_manifest(tmp_path) == [meta]


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
