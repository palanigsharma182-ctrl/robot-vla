"""五模块串联验收，断言具体行为而非只检查 runner 能否退出。"""

from pathlib import Path
import runpy
from dataclasses import replace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def replay():
    path = Path(__file__).resolve().parents[1] / "experiments/five_common_replay/run.py"
    return runpy.run_path(str(path))["run_replay"]()


def test_motion_time_calibration_and_memory_form_one_chain(replay):
    rows = [r for r in replay["rows"] if r["episode"] == "timing_and_motion"]
    assert len(rows) == 13
    assert [i for i, r in enumerate(rows) if r["accepted"]] == [3, 7]
    assert rows[3]["age_s"] == pytest.approx(0.009)
    assert rows[3]["position_base_m"] == pytest.approx((0.1, 0., 1.))
    assert np.diag(rows[3]["covariance_base_m2"]) == pytest.approx([4e-6]*3)
    for row in rows:
        assert row["actual_translation"] != row["commanded_translation"]
    for index in (4, 5):
        assert "rgb_timestamp_not_increasing" in rows[index]["rejections"]
        assert rows[index]["memory_only"]
    for index in (8, 9):
        assert not rows[index]["motion_write_eligible"]
        assert not rows[index]["accepted"] and rows[index]["memory_only"]
    for index in (10, 11):
        assert rows[index]["position_base_m"] == pytest.approx((0.1, 0., 1.))
        assert rows[index]["memory_only"]
    assert rows[-1]["age_s"] == pytest.approx(0.559)
    assert not rows[-1]["valid"] and not rows[-1]["navigation_available"]


@pytest.mark.parametrize("episode,reason", [
    ("source_change", "source_model_identity_mismatch"),
    ("contact", "object_contact_detected"),
])
def test_calibration_identity_and_contact_invalidation(replay, episode, reason):
    rows = [r for r in replay["rows"] if r["episode"] == episode]
    assert rows[1]["accepted"]
    for row in rows[2:]:
        assert not row["valid"] and not row["navigation_available"]
        assert reason in row["invalid_reasons"]
    if episode == "source_change":
        assert rows[1]["source_identity"] != rows[2]["source_identity"]
        assert rows[1]["source_identity"] == rows[3]["source_identity"]


def test_full_replay_resets_and_has_no_actuator_consumer(replay):
    assert replay["episodes"] == 4 and len(replay["rows"]) == 23
    assert replay["actuation_count"] == replay["provider_inference_count"] == 0
    assert all(not r["contact_authorized"] for r in replay["rows"])
    rows = [r for r in replay["rows"] if r["episode"] == "after_reset"]
    assert rows[0]["position_base_m"] is None and not rows[0]["accepted"]
    assert rows[1]["accepted"] and rows[1]["valid"]


def test_bridge_binds_camera_and_capture_phase_before_write():
    path = Path(__file__).resolve().parents[1] / "experiments/five_common_replay/run.py"
    runner = runpy.run_path(str(path))
    observation = runner["synthetic_observation"](
        episode="test", index=1, tick=0.05, rgb_time=0.041,
        motion=runner["Motion"].COLLECT, actual_translation=np.zeros(3),
    )
    kwargs = dict(position_camera_m=np.array([0., 0., 1.]), covariance_camera_m2=np.eye(3)*1e-6,
                  calibration=runner["synthetic_calibration"](), tcp_pose_timestamp_s=0.041,
                  collect_started_at_s=0.05)
    measurement = runner["measurement_from_prediction"](observation, **kwargs)
    assert not measurement.write_gate_passed  # 当前已稳定，也不能把运动期图像改标为采集帧。
    with pytest.raises(ValueError, match="target camera"):
        runner["measurement_from_prediction"](
            replace(observation, camera_uid="another-camera", actual_pose_source=None), **kwargs,
        )
