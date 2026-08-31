import pytest

from robot_vla.local_dagger_protocol import (
    EXPERT_ACTION_BUDGET_EXHAUSTED_REASON,
    POLICY_ACTION_BUDGET_EXHAUSTED_REASON,
    LocalDaggerActionBudgetProtocol,
    resolve_local_dagger_action_budget,
)


def test_protocol_enum_freezes_legacy_and_segmented_limits() -> None:
    legacy = resolve_local_dagger_action_budget(
        LocalDaggerActionBudgetProtocol.LEGACY
    )
    segmented = resolve_local_dagger_action_budget("segmented-300-180-480")

    assert legacy.planned_metadata() is None
    assert legacy.usage_metadata(total_actions=300, expert_takeover_step=None) is None
    assert segmented.planned_metadata() == {
        "version": "robot-vla-local-dagger-segmented-action-budget/v1",
        "name": "segmented-300-180-480",
        "action_unit": "actual_environment_action",
        "policy_action_limit": 300,
        "expert_action_limit": 180,
        "environment_action_limit": 480,
        "deadline_semantics": "success_must_precede_environment_truncation",
    }
    assert segmented.usage_metadata(
        total_actions=479,
        expert_takeover_step=300,
    ) == {
        "policy_actions": 300,
        "expert_actions": 179,
        "total_actions": 479,
    }


def test_policy_action_300_stops_without_masking_same_step_boundary() -> None:
    plan = resolve_local_dagger_action_budget("segmented-300-180-480")

    assert plan.policy_budget_exhausted_after_action(
        policy_actions=299,
        boundary_reached=False,
    ) is False
    assert plan.policy_budget_exhausted_after_action(
        policy_actions=300,
        boundary_reached=False,
    ) is True
    assert plan.policy_budget_exhausted_after_action(
        policy_actions=300,
        boundary_reached=True,
    ) is False


def test_expert_action_180_preserves_truncation_and_success_priority() -> None:
    plan = resolve_local_dagger_action_budget("segmented-300-180-480")

    assert plan.expert_budget_exhausted_after_action(
        expert_actions=179,
        task_completed=False,
        truncated=False,
    ) is False
    assert plan.expert_budget_exhausted_after_action(
        expert_actions=180,
        task_completed=False,
        truncated=False,
    ) is True
    assert plan.expert_budget_exhausted_after_action(
        expert_actions=180,
        task_completed=True,
        truncated=False,
    ) is False
    assert plan.expert_budget_exhausted_after_action(
        expert_actions=180,
        task_completed=True,
        truncated=True,
    ) is False
    assert plan.environment_hard_deadline_reached_after_action(
        total_actions=479
    ) is False
    assert plan.environment_hard_deadline_reached_after_action(
        total_actions=480
    ) is True


def test_usage_and_protocol_inputs_fail_closed() -> None:
    plan = resolve_local_dagger_action_budget("segmented-300-180-480")

    with pytest.raises(ValueError, match="takeover step"):
        plan.usage_metadata(total_actions=100, expert_takeover_step=101)
    with pytest.raises(ValueError, match="未知 Local DAgger"):
        resolve_local_dagger_action_budget("segmented-300-181-481")

    assert "300 Action" in POLICY_ACTION_BUDGET_EXHAUSTED_REASON
    assert "180 Action" in EXPERT_ACTION_BUDGET_EXHAUSTED_REASON
