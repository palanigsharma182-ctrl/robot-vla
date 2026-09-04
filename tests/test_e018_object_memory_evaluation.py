from __future__ import annotations

import numpy as np

from robot_vla.precision.object_memory import ObjectMemoryConfig, ObjectMemorySafetyContext
from robot_vla.precision.object_memory_evaluation import (
    ObjectReplayFrame,
    calibrate_object_write_threshold,
    replay_object_memory,
    summarize_object_memory_replay,
)

MODEL_IDENTITY = "precision-unet/checkpoint-sha256"


def _config() -> ObjectMemoryConfig:
    return ObjectMemoryConfig(
        max_unobserved_age_s=1.0,
        max_innovation_m=0.01,
        max_position_std_m=0.02,
        min_candidate_frames=2,
        max_candidate_gap_s=0.075,
        max_candidate_position_spread_m=0.005,
        max_sensor_skew_s=0.01,
        expected_source_camera="hand_camera",
        expected_source_model_identity=MODEL_IDENTITY,
    )


def _safety(*, pregrasp: bool = True) -> ObjectMemorySafetyContext:
    return ObjectMemorySafetyContext(
        pregrasp_window_open=pregrasp,
        gripper_open=pregrasp,
        controller_tracking_valid=True,
        object_contact_detected=False,
        gripper_close_commanded=not pregrasp,
        grasp_candidate=False,
        grasp_verified=False,
        object_maybe_moved=False,
    )


def _frame(
    timestep: int,
    *,
    visible: bool,
    pregrasp: bool = True,
) -> ObjectReplayFrame:
    timestamp = timestep * 0.05
    position = (0.4, 0.1, 0.02) if visible else None
    covariance = np.eye(3, dtype=np.float64) * 1e-6 if visible else None
    return ObjectReplayFrame(
        episode_id="episode-a",
        timestep=timestep,
        timestamp_s=timestamp,
        rgb_timestamp_s=timestamp,
        camera_pose_timestamp_s=timestamp,
        tcp_pose_timestamp_s=timestamp,
        predicted_position_base_m=position,
        measurement_covariance_base_m2=covariance,
        write_score=0.9 if visible else 0.0,
        structurally_eligible=visible,
        predicted_observable=visible,
        geometry_valid=visible,
        gt_position_base_m=(0.4, 0.1, 0.02),
        gt_observable=visible,
        oracle_safe_measurement=visible,
        safety=_safety(pregrasp=pregrasp),
        source_camera="hand_camera",
        source_model_identity=MODEL_IDENTITY,
    )


def test_object_write_calibration_maximizes_coverage_without_unsafe() -> None:
    calibration = calibrate_object_write_threshold(
        scores=(0.9, 0.8, 0.7),
        structurally_eligible=np.ones(3, dtype=np.bool_),
        oracle_safe=np.asarray((True, False, True), dtype=np.bool_),
    )

    assert calibration.enabled
    assert calibration.threshold == 0.9
    assert calibration.accepted_count == 1
    assert calibration.accepted_unsafe_count == 0
    assert calibration.safe_coverage == 0.5


def test_replay_bridges_pregrasp_gap_and_invalidates_after_window() -> None:
    frames = [
        _frame(0, visible=True),
        _frame(1, visible=True),
        _frame(2, visible=False),
        _frame(3, visible=False),
        _frame(4, visible=False, pregrasp=False),
    ]
    calibration = calibrate_object_write_threshold(
        scores=(0.9, 0.9),
        structurally_eligible=np.ones(2, dtype=np.bool_),
        oracle_safe=np.ones(2, dtype=np.bool_),
    )

    records, leakage = replay_object_memory(
        frames,
        calibration=calibration,
        memory_config=_config(),
    )
    summary = summarize_object_memory_replay(
        records,
        catastrophic_world_xyz_error_m=0.02,
        reset_leakage_count=leakage,
    )

    assert not records[0].current_measurement_accepted
    assert records[1].current_measurement_accepted
    assert records[2].memory_valid and records[2].memory_only
    assert records[3].memory_valid and records[3].memory_only
    assert not records[4].memory_valid
    assert summary["direct_write_gate_count"] == 2
    assert summary["current_candidate_valid_count"] == 1
    assert summary["object_memory_valid_count"] == 3
    assert summary["paired_navigation_availability_gain_count"] == 2
    assert summary["memory_valid_while_gt_unobservable_count"] == 2
    assert summary["cold_start_gt_unobservable_count"] == 0
    assert summary["gt_unobservable_after_memory_initialization_count"] == 2
    assert summary["episodes_with_initialized_memory_count"] == 1
    assert summary["post_pregrasp_memory_valid_count"] == 0
    assert summary["memory_only_contact_authorization_count"] == 0
    assert summary["episode_reset_leakage_count"] == 0
    assert summary["memory_catastrophic_count"] == 0
