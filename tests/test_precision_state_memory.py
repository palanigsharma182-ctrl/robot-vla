from __future__ import annotations

import json

import numpy as np
import pytest

from robot_vla.cli.run_e015_precision_memory import _claim_test_evaluation_once
from robot_vla.precision.memory_evaluation import (
    GoalReplayFrame,
    calibrate_goal_write_threshold,
    replay_goal_memory,
    select_memory_max_age,
    summarize_goal_memory_replay,
)
from robot_vla.precision.observability import (
    GoalWriteEvidence,
    derive_goal_observability,
    mask_probability_at_normalized_uv,
)
from robot_vla.precision.state_memory import (
    ExplicitGoalStateMemory,
    GoalMeasurement,
    GoalMemoryConfig,
)


def _config(*, max_age_s: float = 1.0) -> GoalMemoryConfig:
    return GoalMemoryConfig(
        max_unobserved_age_s=max_age_s,
        max_innovation_m=0.01,
        max_position_std_m=0.02,
        require_covariance=True,
    )


def _measurement(
    timestamp_s: float,
    *,
    position: tuple[float, float, float] | None = (0.4, 0.1, 0.02),
    observable: bool = True,
    gate: bool = True,
    frame_semantics: str = "position/robot-base/m/v1",
) -> GoalMeasurement:
    covariance = None if position is None else np.eye(3, dtype=np.float64) * 1e-6
    return GoalMeasurement(
        timestamp_s=timestamp_s,
        position_base_m=position,
        covariance_base_m2=covariance,
        confidence=0.99 if observable else 0.0,
        goal_exists=True,
        projection_valid=True,
        in_fov=True,
        observable=observable,
        geometry_valid=position is not None,
        write_gate_passed=bool(gate and observable and position is not None),
        frame_semantics=frame_semantics,
    )


def test_visible_reliable_measurement_initializes_base_frame_memory() -> None:
    memory = ExplicitGoalStateMemory(_config())
    reset = memory.reset("episode-a")
    assert not reset.valid

    update = memory.update(_measurement(0.0), episode_id="episode-a")

    assert update.measurement_accepted
    assert update.state.valid
    assert update.state.observable_now
    assert update.state.position_base_m == pytest.approx((0.4, 0.1, 0.02))
    assert memory.world_state.estimated_goal_position_base_m == pytest.approx(
        (0.4, 0.1, 0.02)
    )


def test_short_occlusion_keeps_position_but_marks_not_observable() -> None:
    memory = ExplicitGoalStateMemory(_config(max_age_s=1.0))
    memory.reset("episode-a")
    memory.update(_measurement(0.0), episode_id="episode-a")

    held = memory.update(
        _measurement(0.5, position=None, observable=False, gate=False),
        episode_id="episode-a",
    )

    assert not held.measurement_accepted
    assert held.state.valid
    assert not held.state.observable_now
    assert held.state.position_base_m == pytest.approx((0.4, 0.1, 0.02))
    assert held.state.age_s == pytest.approx(0.5)


def test_rejected_or_conflicting_measurement_cannot_overwrite_history() -> None:
    memory = ExplicitGoalStateMemory(_config())
    memory.reset("episode-a")
    memory.update(_measurement(0.0), episode_id="episode-a")

    rejected = memory.update(
        _measurement(0.1, position=(0.401, 0.1, 0.02), gate=False),
        episode_id="episode-a",
    )
    assert not rejected.measurement_accepted
    assert rejected.state.valid
    assert rejected.state.position_base_m == pytest.approx((0.4, 0.1, 0.02))

    conflict = memory.update(
        _measurement(0.2, position=(0.5, 0.1, 0.02)),
        episode_id="episode-a",
    )
    assert not conflict.measurement_accepted
    assert conflict.rejection_reasons == ("measurement_conflict",)
    assert not conflict.state.valid
    assert conflict.state.position_base_m == pytest.approx((0.4, 0.1, 0.02))

    still_invalid = memory.update(
        _measurement(0.3, position=None, observable=False, gate=False),
        episode_id="episode-a",
    )
    assert not still_invalid.state.valid
    assert "measurement_conflict" in still_invalid.state.invalid_reasons


def test_reobservation_updates_memory_after_occlusion() -> None:
    memory = ExplicitGoalStateMemory(_config())
    memory.reset("episode-a")
    memory.update(_measurement(0.0), episode_id="episode-a")
    memory.update(
        _measurement(0.2, position=None, observable=False, gate=False),
        episode_id="episode-a",
    )

    recovered = memory.update(
        _measurement(0.4, position=(0.402, 0.1, 0.02)),
        episode_id="episode-a",
    )

    assert recovered.measurement_accepted
    assert recovered.state.valid
    assert recovered.state.observable_now
    assert recovered.state.accepted_update_count == 2
    assert recovered.state.position_base_m == pytest.approx((0.402, 0.1, 0.02))


def test_episode_reset_clears_memory_and_rejects_identity_drift() -> None:
    memory = ExplicitGoalStateMemory(_config())
    memory.reset("episode-a")
    memory.update(_measurement(0.0), episode_id="episode-a")

    reset = memory.reset("episode-b")
    assert not reset.valid
    assert reset.position_base_m is None
    assert memory.world_state.estimated_goal_position_base_m is None
    with pytest.raises(ValueError, match="episode identity"):
        memory.update(_measurement(0.1), episode_id="episode-a")


def test_frame_semantics_and_timestamp_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="robot-base"):
        _measurement(0.0, frame_semantics="position/camera/m/v1")

    memory = ExplicitGoalStateMemory(_config())
    memory.reset("episode-a")
    memory.update(_measurement(1.0), episode_id="episode-a")
    with pytest.raises(ValueError, match="不能倒退"):
        memory.update(_measurement(0.9), episode_id="episode-a")


def test_stale_memory_retains_position_but_is_not_controller_authorized() -> None:
    memory = ExplicitGoalStateMemory(_config(max_age_s=0.5))
    memory.reset("episode-a")
    memory.update(_measurement(0.0), episode_id="episode-a")
    stale = memory.update(
        _measurement(0.6, position=None, observable=False, gate=False),
        episode_id="episode-a",
    )
    assert stale.state.position_base_m == pytest.approx((0.4, 0.1, 0.02))
    assert not stale.state.valid
    assert stale.state.invalid_reasons == ("memory_stale",)
    assert memory.world_state.estimated_goal_position_base_m is None


def test_corrected_observability_checks_projected_center_not_mask_any() -> None:
    goal = np.zeros((7, 7), dtype=np.bool_)
    goal[0, 0] = True
    object_mask = np.zeros_like(goal)
    object_mask[3, 3] = True
    label = derive_goal_observability(
        goal_exists=True,
        projection_valid=True,
        projected_normalized_uv=np.asarray(((3.5 / 7.0), (3.5 / 7.0))),
        goal_mask=goal,
        object_mask=object_mask,
        legacy_visible=True,
        support_radius_px=1,
    )
    assert label.in_fov
    assert not label.observable
    assert label.center_inside_object_mask
    assert label.occlusion_type == "object_occlusion"
    assert label.legacy_contract_mismatch


def test_deployable_write_evidence_uses_predicted_signals_only() -> None:
    goal_probability = np.zeros((2, 2), dtype=np.float32)
    goal_probability[0, 0] = 1.0
    sampled = mask_probability_at_normalized_uv(
        goal_probability,
        np.asarray((0.25, 0.25), dtype=np.float32),
    )
    assert sampled == pytest.approx(1.0)

    evidence = GoalWriteEvidence(
        visibility_probability=0.9,
        projection_validity_probability=0.95,
        goal_mask_probability=sampled,
        object_mask_probability=0.05,
        normalized_entropy=0.2,
        radial_sigma_px=1.0,
        geometry_valid=True,
    )
    assert evidence.structurally_eligible
    assert evidence.score == pytest.approx(0.5)
    assert evidence.accepted(threshold=0.5)
    assert not evidence.accepted(threshold=0.5001)


def _replay_frame(
    timestep: int,
    *,
    predicted_x: float = 0.4,
    score: float = 0.8,
    gt_observable: bool = True,
    predicted_observable: bool = True,
) -> GoalReplayFrame:
    return GoalReplayFrame(
        episode_id="episode-a",
        timestep=timestep,
        timestamp_s=timestep * 0.1,
        predicted_position_base_m=(predicted_x, 0.1, 0.02),
        measurement_covariance_base_m2=tuple(
            tuple(float(value) for value in row)
            for row in (np.eye(3, dtype=np.float64) * 1e-6)
        ),
        write_score=score,
        structurally_eligible=predicted_observable,
        predicted_observable=predicted_observable,
        geometry_valid=True,
        gt_position_base_m=(0.4, 0.1, 0.02),
        gt_observable=gt_observable,
    )


def test_validation_calibration_maximizes_coverage_without_unsafe_writes() -> None:
    calibration = calibrate_goal_write_threshold(
        scores=(0.9, 0.8, 0.7, 0.6),
        structurally_eligible=np.asarray((True, True, True, True), dtype=np.bool_),
        oracle_safe=np.asarray((True, True, False, True), dtype=np.bool_),
    )
    assert calibration.enabled
    assert calibration.threshold == pytest.approx(0.8)
    assert calibration.accepted_count == 2
    assert calibration.accepted_unsafe_count == 0


def test_memory_replay_improves_occluded_availability_without_hallucinating() -> None:
    frames = (
        _replay_frame(0),
        _replay_frame(
            1,
            score=0.1,
            gt_observable=False,
            predicted_observable=False,
        ),
    )
    calibration = calibrate_goal_write_threshold(
        scores=(0.8, 0.1),
        structurally_eligible=np.asarray((True, False), dtype=np.bool_),
        oracle_safe=np.asarray((True, False), dtype=np.bool_),
    )
    records = replay_goal_memory(
        frames,
        calibration=calibration,
        memory_config=_config(max_age_s=0.5),
    )
    summary = summarize_goal_memory_replay(
        records,
        catastrophic_world_xy_error_m=0.02,
    )
    assert summary["current_measurement_valid_count"] == 1
    assert summary["memory_valid_count"] == 2
    assert summary["memory_valid_while_gt_unobservable_count"] == 1
    assert summary["memory_catastrophic_count"] == 0


def test_memory_summary_counts_uninitialized_occluded_frames() -> None:
    frame = _replay_frame(
        0,
        score=0.1,
        gt_observable=False,
        predicted_observable=False,
    )
    calibration = calibrate_goal_write_threshold(
        scores=(frame.write_score,),
        structurally_eligible=np.asarray((False,), dtype=np.bool_),
        oracle_safe=np.asarray((False,), dtype=np.bool_),
    )
    records = replay_goal_memory(
        (frame,),
        calibration=calibration,
        memory_config=_config(max_age_s=0.5),
    )
    summary = summarize_goal_memory_replay(
        records,
        catastrophic_world_xy_error_m=0.02,
    )

    assert summary["stale_or_uninitialized_occluded_count"] == 1
    assert summary["memory_unavailable_while_gt_unobservable_count"] == 1
    assert summary["memory_uninitialized_while_gt_unobservable_count"] == 1
    assert (
        summary[
            "memory_previously_initialized_but_invalid_while_gt_unobservable_count"
        ]
        == 0
    )


def test_memory_age_selection_uses_validation_only_and_prefers_safe_coverage() -> None:
    frames = tuple(
        _replay_frame(
            timestep,
            score=0.8 if timestep == 0 else 0.1,
            gt_observable=timestep == 0,
            predicted_observable=timestep == 0,
        )
        for timestep in range(4)
    )
    calibration = calibrate_goal_write_threshold(
        scores=tuple(frame.write_score for frame in frames),
        structurally_eligible=np.asarray(
            tuple(frame.structurally_eligible for frame in frames), dtype=np.bool_
        ),
        oracle_safe=np.asarray((True, False, False, False), dtype=np.bool_),
    )
    selected, reports = select_memory_max_age(
        frames,
        calibration=calibration,
        max_age_candidates_s=(0.1, 0.3, 1.0),
        max_innovation_m=0.01,
        max_position_std_m=0.02,
        require_covariance=True,
        covariance_growth_m2_per_s=0.0,
        catastrophic_world_xy_error_m=0.02,
    )
    assert selected == pytest.approx(0.3)
    assert len(reports) == 3


def test_test_evaluation_claim_is_atomic_and_cannot_be_reused(tmp_path) -> None:
    claim = tmp_path / "test_evaluation_claim.json"
    sha256 = _claim_test_evaluation_once(
        claim,
        rules_sha256="1" * 64,
        dataset_identity_sha256="2" * 64,
        seed_identity_sha256="3" * 64,
        source_tree_sha256="4" * 64,
    )

    assert len(sha256) == 64
    assert json.loads(claim.read_text(encoding="utf-8"))["status"] == (
        "claimed-before-test-read"
    )
    with pytest.raises(RuntimeError, match="禁止.*重复评估"):
        _claim_test_evaluation_once(
            claim,
            rules_sha256="1" * 64,
            dataset_identity_sha256="2" * 64,
            seed_identity_sha256="3" * 64,
            source_tree_sha256="4" * 64,
        )
