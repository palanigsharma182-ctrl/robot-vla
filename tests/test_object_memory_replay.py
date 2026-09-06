"""验证合成消费者把观测证据、候选窗口和状态用途正确串接。"""

from pathlib import Path
import runpy

import pytest


@pytest.fixture(scope="module")
def replay() -> dict:
    runner = Path(__file__).resolve().parents[1] / "experiments/object_memory_replay/run.py"
    return runpy.run_path(str(runner))["run_replay"]()


def test_rejected_predictions_cannot_initialize_or_replace_memory(replay: dict) -> None:
    rows = [r for r in replay["rows"] if r["episode_id"] == "hold_and_expiry"]
    assert [r["measurement_accepted"] for r in rows] == [False, False, False, True, False, False, False]
    assert all(r["stored_position"] is None for r in rows[:3])
    assert all(not r["navigation_available"] for r in rows[:3])
    assert not rows[0]["write_gate_passed"]
    assert not rows[1]["write_gate_passed"]
    assert rows[3]["navigation_available"] and not rows[3]["memory_only"]
    for row in rows[4:6]:
        assert row["navigation_available"] and row["memory_only"]
        assert row["stored_position"] == (0.4, 0.1, 0.02)
    assert rows[-1]["age_s"] == pytest.approx(0.55)
    assert not rows[-1]["valid"] and not rows[-1]["navigation_available"]
    assert rows[-1]["navigation_position"] is None
    assert rows[-1]["stored_position"] == (0.4, 0.1, 0.02)  # 仅供诊断保留。


@pytest.mark.parametrize("episode,reason", [
    ("contact", "object_contact_detected"),
    ("source_change", "source_model_identity_mismatch"),
])
def test_irreversible_invalidation_survives_later_good_frames(replay: dict, episode: str, reason: str) -> None:
    rows = [r for r in replay["rows"] if r["episode_id"] == episode]
    assert rows[1]["valid"] and rows[1]["measurement_accepted"]
    for row in rows[2:]:
        assert not row["valid"] and not row["measurement_accepted"]
        assert not row["navigation_available"]
        assert reason in row["invalid_reasons"]


def test_episode_reset_clears_state_window_and_invalidation(replay: dict) -> None:
    assert len(replay["resets"]) == 4
    for reset in replay["resets"]:
        assert reset["object_position"] is None and reset["goal_position"] is None
        assert not reset["object_valid"] and reset["accepted_update_count"] == 0
    rows = [r for r in replay["rows"] if r["episode_id"] == "after_reset"]
    assert rows[0]["stored_position"] is None and not rows[0]["measurement_accepted"]
    assert rows[1]["measurement_accepted"] and rows[1]["valid"]
    assert rows[1]["stored_position"] == (0.6, 0.1, 0.02)


def test_object_updates_preserve_goal_and_navigation_grants_no_contact(replay: dict) -> None:
    assert len(replay["rows"]) == 19
    for row in replay["rows"]:
        assert row["goal_unchanged"]
        assert row["goal_position"] == (0.6, -0.1, 0.02)
        assert not row["contact_authorized"]
