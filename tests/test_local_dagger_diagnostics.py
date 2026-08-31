import pytest

from robot_vla.contracts import PICK_AND_PLACE_SKILLS
from robot_vla.sim.local_dagger_diagnostics import (
    EPISODE_TIME_LIMIT_REASON,
    EXPERT_ACTION_BUDGET_EXHAUSTED_REASON,
    EXPERT_GRASP_APPROACH_PHASE,
    EXPERT_GRASP_STABILIZATION_PHASE,
    EXPERT_LIFT_MOTION_PHASE,
    EXPERT_LOWER_MOTION_PHASE,
    EXPERT_RELEASE_SETTLE_PHASE,
    EXPERT_TRANSPORT_MOTION_PHASE,
    POLICY_ACTION_BUDGET_EXHAUSTED_REASON,
    POLICY_BEFORE_BOUNDARY_REASON,
    POLICY_ROLLIN_PHASE,
    LocalDaggerFailureDiagnostics,
    classify_grasp_lift_failure,
)
from robot_vla.tasks.pick_place import OutcomeSnapshot, PickPlaceTaskProgress


def _progress(
    *,
    completed_skill_count: int,
    stable_grasp_steps: int = 0,
    reached: bool = False,
    grasped: bool = False,
    lifted: bool = False,
    transported: bool = False,
) -> PickPlaceTaskProgress:
    active_skill_id = min(completed_skill_count, len(PICK_AND_PLACE_SKILLS) - 1)
    return PickPlaceTaskProgress(
        active_skill_id=active_skill_id,
        active_skill_name=PICK_AND_PLACE_SKILLS[active_skill_id],
        completed_skill_count=completed_skill_count,
        task_completed=completed_skill_count == len(PICK_AND_PLACE_SKILLS),
        stable_grasp_steps=stable_grasp_steps,
        stable_place_steps=0,
        outcome=OutcomeSnapshot(
            tcp_to_object_distance_m=0.03 if reached else 0.10,
            object_height_above_support_m=0.06 if lifted else 0.0,
            object_to_goal_xy_distance_m=0.02 if transported else 0.10,
            object_to_goal_distance_m=0.10,
            object_linear_speed_m_s=0.01,
            object_angular_speed_rad_s=0.10,
            reached=reached,
            grasped=grasped,
            lifted=lifted,
            transported=transported,
            placed_now=False,
        ),
    )


def _diagnostics() -> LocalDaggerFailureDiagnostics:
    return LocalDaggerFailureDiagnostics(
        environment_seed=30_100,
        boundary_type="grasp_lift",
    )


def _observe(
    diagnostics: LocalDaggerFailureDiagnostics,
    progress: PickPlaceTaskProgress,
    *,
    terminated: bool = False,
    truncated: bool = False,
    action_source: int = 0,
) -> None:
    diagnostics.observe(
        action_step=diagnostics.action_count + 1,
        progress=progress,
        terminated=terminated,
        truncated=truncated,
        environment_success=False,
        action_source=action_source,
        gripper_opening=0.04,
    )


def test_action_steps_must_be_contiguous_and_rejection_is_fail_closed() -> None:
    diagnostics = _diagnostics()

    with pytest.raises(ValueError, match=r"expected=1, actual=2"):
        diagnostics.observe(
            action_step=2,
            progress=_progress(completed_skill_count=0),
            terminated=False,
            truncated=False,
            environment_success=False,
            action_source=0,
            gripper_opening=0.04,
        )

    result = diagnostics.to_dict()
    assert result["action_count"] == 0
    assert result["phase_action_counts"] == {}
    assert result["final_progress"] is None
    assert result["final_transition"] is None
    assert "action_budget_protocol" not in result
    assert "action_budget_usage" not in result
    assert "budget_exhaustion_phase" not in result


def test_phase_transitions_assign_subsequent_actions_to_the_new_phase() -> None:
    diagnostics = _diagnostics()
    _observe(
        diagnostics,
        _progress(
            completed_skill_count=2,
            stable_grasp_steps=2,
            reached=True,
            grasped=True,
        ),
    )

    diagnostics.set_phase(EXPERT_GRASP_APPROACH_PHASE, action_step=1)
    _observe(
        diagnostics,
        _progress(completed_skill_count=2, stable_grasp_steps=2, reached=True, grasped=True),
        action_source=1,
    )

    result = diagnostics.to_dict()
    assert result["phase_transitions"] == [
        {"action_step": 0, "phase": POLICY_ROLLIN_PHASE},
        {"action_step": 1, "phase": EXPERT_GRASP_APPROACH_PHASE},
    ]
    assert result["boundary_reached"] is True
    assert result["boundary_detection_step"] == 1
    assert result["expert_takeover_step"] == 1
    assert result["phase_action_counts"] == {
        EXPERT_GRASP_APPROACH_PHASE: 1,
        POLICY_ROLLIN_PHASE: 1,
    }
    assert result["phase_at_failure"] == EXPERT_GRASP_APPROACH_PHASE


def test_policy_failure_before_reach_is_classified_by_progress_and_terminal_signal() -> None:
    diagnostics = _diagnostics()
    _observe(
        diagnostics,
        _progress(completed_skill_count=0),
        truncated=True,
    )

    classification = classify_grasp_lift_failure(
        diagnostics.to_dict(failure_reason=POLICY_BEFORE_BOUNDARY_REASON)
    )

    assert classification == {
        "failure_family": "policy_before_stable_grasp_boundary",
        "progress_bucket": "never_completed_reach",
        "terminal_bucket": "policy_time_limit",
    }


def test_policy_failure_after_reach_without_raw_grasp_has_distinct_bucket() -> None:
    diagnostics = _diagnostics()
    _observe(
        diagnostics,
        _progress(completed_skill_count=1, reached=True),
        terminated=True,
    )

    classification = classify_grasp_lift_failure(
        diagnostics.to_dict(failure_reason=POLICY_BEFORE_BOUNDARY_REASON)
    )

    assert classification["progress_bucket"] == "reached_never_raw_grasped"
    assert classification["terminal_bucket"] == "policy_environment_termination"


def test_transient_raw_grasp_lost_before_stability_is_classified() -> None:
    diagnostics = _diagnostics()
    _observe(
        diagnostics,
        _progress(
            completed_skill_count=1,
            stable_grasp_steps=1,
            reached=True,
            grasped=True,
        ),
    )
    _observe(
        diagnostics,
        _progress(completed_skill_count=1, reached=True),
        terminated=True,
    )

    result = diagnostics.to_dict(failure_reason=POLICY_BEFORE_BOUNDARY_REASON)
    classification = classify_grasp_lift_failure(result)

    assert result["raw_grasp_loss_events"] == 1
    assert classification["progress_bucket"] == "transient_grasp_then_lost_before_stable"


def test_raw_grasp_at_terminal_without_stability_is_classified() -> None:
    diagnostics = _diagnostics()
    _observe(
        diagnostics,
        _progress(
            completed_skill_count=1,
            stable_grasp_steps=1,
            reached=True,
            grasped=True,
        ),
        truncated=True,
    )

    result = diagnostics.to_dict(failure_reason=POLICY_BEFORE_BOUNDARY_REASON)
    classification = classify_grasp_lift_failure(result)

    assert result["max_stable_grasp_steps"] == 1
    assert result["final_progress"]["raw_grasped"] is True
    assert classification["progress_bucket"] == "raw_grasp_at_terminal_not_stable"


@pytest.mark.parametrize(
    "phase",
    (
        EXPERT_GRASP_APPROACH_PHASE,
        EXPERT_GRASP_STABILIZATION_PHASE,
        EXPERT_LIFT_MOTION_PHASE,
        EXPERT_TRANSPORT_MOTION_PHASE,
        EXPERT_LOWER_MOTION_PHASE,
        EXPERT_RELEASE_SETTLE_PHASE,
    ),
)
def test_expert_time_limit_preserves_commanded_phase(phase: str) -> None:
    diagnostics = _diagnostics()
    _observe(
        diagnostics,
        _progress(completed_skill_count=2, stable_grasp_steps=2, reached=True, grasped=True),
    )
    diagnostics.set_phase(phase, action_step=1)
    _observe(
        diagnostics,
        _progress(completed_skill_count=2, stable_grasp_steps=2, reached=True, grasped=True),
        truncated=True,
        action_source=1,
    )

    classification = classify_grasp_lift_failure(
        diagnostics.to_dict(failure_reason=EPISODE_TIME_LIMIT_REASON)
    )

    assert classification == {
        "failure_family": "expert_time_limit_after_takeover",
        "progress_bucket": phase,
        "terminal_bucket": "episode_time_limit",
    }


def test_stable_grasp_without_takeover_is_reported_as_contract_violation() -> None:
    diagnostics = _diagnostics()
    _observe(
        diagnostics,
        _progress(
            completed_skill_count=2,
            stable_grasp_steps=2,
            reached=True,
            grasped=True,
        ),
        terminated=True,
    )

    result = diagnostics.to_dict(failure_reason=POLICY_BEFORE_BOUNDARY_REASON)
    classification = classify_grasp_lift_failure(result)

    assert result["phase_at_failure"] == POLICY_ROLLIN_PHASE
    assert result["skill_completion_steps"]["reach"] == 1
    assert result["skill_completion_steps"]["grasp"] == 1
    assert classification["progress_bucket"] == (
        "contract_violation_stable_grasp_without_takeover"
    )


def test_raw_grasp_segments_use_end_exclusive_action_steps() -> None:
    diagnostics = _diagnostics()
    _observe(diagnostics, _progress(completed_skill_count=1, reached=True))
    _observe(
        diagnostics,
        _progress(
            completed_skill_count=1,
            stable_grasp_steps=1,
            reached=True,
            grasped=True,
        ),
    )
    _observe(
        diagnostics,
        _progress(
            completed_skill_count=1,
            stable_grasp_steps=1,
            reached=True,
            grasped=True,
        ),
    )
    _observe(diagnostics, _progress(completed_skill_count=1, reached=True))
    _observe(
        diagnostics,
        _progress(
            completed_skill_count=1,
            stable_grasp_steps=1,
            reached=True,
            grasped=True,
        ),
        truncated=True,
    )

    result = diagnostics.to_dict(failure_reason=POLICY_BEFORE_BOUNDARY_REASON)

    assert result["raw_grasp_segments"] == [
        {"start_action_step": 2, "end_action_step_exclusive": 4},
        {"start_action_step": 5, "end_action_step_exclusive": 6},
    ]
    assert result["raw_grasp_action_count"] == 3
    assert result["raw_grasp_loss_events"] == 1
    assert result["max_consecutive_raw_grasp_steps"] == 2


def test_segmented_diagnostics_distinguishes_policy_budget_exhaustion() -> None:
    diagnostics = LocalDaggerFailureDiagnostics(
        environment_seed=30_200,
        boundary_type="grasp_lift",
        action_budget_protocol="segmented-300-180-480",
    )
    for _ in range(300):
        _observe(diagnostics, _progress(completed_skill_count=0))
    diagnostics.record_budget_exhaustion("policy", action_step=300)

    result = diagnostics.to_dict(
        failure_reason=POLICY_ACTION_BUDGET_EXHAUSTED_REASON
    )
    classification = classify_grasp_lift_failure(result)

    assert result["action_budget_protocol"]["name"] == "segmented-300-180-480"
    assert result["action_budget_usage"] == {
        "policy_actions": 300,
        "expert_actions": 0,
        "total_actions": 300,
    }
    assert result["budget_exhaustion_phase"] == "policy"
    assert classification["failure_family"] == (
        "policy_action_budget_exhausted_before_boundary"
    )


def test_segmented_diagnostics_distinguishes_expert_budget_exhaustion() -> None:
    diagnostics = LocalDaggerFailureDiagnostics(
        environment_seed=30_203,
        boundary_type="grasp_lift",
        action_budget_protocol="segmented-300-180-480",
    )
    boundary_progress = _progress(
        completed_skill_count=2,
        stable_grasp_steps=2,
        reached=True,
        grasped=True,
    )
    _observe(diagnostics, boundary_progress)
    diagnostics.set_phase(EXPERT_LIFT_MOTION_PHASE, action_step=1)
    for _ in range(180):
        _observe(diagnostics, boundary_progress, action_source=1)
    diagnostics.record_budget_exhaustion("expert", action_step=181)

    result = diagnostics.to_dict(
        failure_reason=EXPERT_ACTION_BUDGET_EXHAUSTED_REASON
    )
    classification = classify_grasp_lift_failure(result)

    assert result["action_budget_usage"] == {
        "policy_actions": 1,
        "expert_actions": 180,
        "total_actions": 181,
    }
    assert result["budget_exhaustion_phase"] == "expert"
    assert classification == {
        "failure_family": "expert_action_budget_exhausted_after_takeover",
        "progress_bucket": EXPERT_LIFT_MOTION_PHASE,
        "terminal_bucket": "expert_action_budget",
    }
