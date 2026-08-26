import pytest

from robot_vla.contracts import PICK_AND_PLACE_SKILLS
from robot_vla.tasks.pick_place import (
    ATOMIC_PICK_PLACE_SKILLS,
    PickPlacePredicateConfig,
    PickPlaceState,
    PickPlaceTaskTracker,
    build_pick_place_task,
    evaluate_pick_place_outcomes,
)


def _state(
    *,
    tcp=(0.0, 0.0, 0.20),
    obj=(0.0, 0.0, 0.02),
    goal=(0.10, 0.0, 0.02),
    linear=(0.0, 0.0, 0.0),
    angular=(0.0, 0.0, 0.0),
    grasped=False,
) -> PickPlaceState:
    return PickPlaceState(
        tcp_position=tcp,
        object_position=obj,
        goal_position=goal,
        object_linear_velocity=linear,
        object_angular_velocity=angular,
        support_center_z_m=0.02,
        is_grasped=grasped,
    )


def test_atomic_definitions_and_combination_task_use_fixed_order() -> None:
    task = build_pick_place_task(1)

    assert tuple(skill.name for skill in ATOMIC_PICK_PLACE_SKILLS) == PICK_AND_PLACE_SKILLS
    assert tuple(skill.skill_id for skill in ATOMIC_PICK_PLACE_SKILLS) == tuple(range(5))
    assert task.skill_names == PICK_AND_PLACE_SKILLS
    assert task.task_id == "pick-cube-to-region"
    assert "green target region" in task.instruction


def test_snapshot_predicates_require_physical_outcomes() -> None:
    config = PickPlacePredicateConfig()

    reached = evaluate_pick_place_outcomes(
        _state(tcp=(0.0, 0.0, 0.05)),
        config,
    )
    lifted = evaluate_pick_place_outcomes(
        _state(tcp=(0.0, 0.0, 0.08), obj=(0.0, 0.0, 0.08), grasped=True),
        config,
    )
    transported = evaluate_pick_place_outcomes(
        _state(
            tcp=(0.10, 0.0, 0.08),
            obj=(0.10, 0.0, 0.08),
            grasped=True,
        ),
        config,
    )
    placed = evaluate_pick_place_outcomes(
        _state(obj=(0.10, 0.0, 0.02), grasped=False),
        config,
    )

    assert reached.reached is True
    assert lifted.lifted is True and lifted.transported is False
    assert transported.transported is True
    assert placed.placed_now is True


def test_task_tracker_requires_stable_grasp_and_stable_released_place() -> None:
    tracker = PickPlaceTaskTracker(
        PickPlacePredicateConfig(stable_grasp_steps=2, stable_place_steps=3)
    )

    assert tracker.update(_state()).active_skill_name == "reach"
    assert tracker.update(_state(tcp=(0.0, 0.0, 0.05))).active_skill_name == "grasp"
    assert tracker.update(_state(tcp=(0.0, 0.0, 0.03), grasped=True)).active_skill_name == "grasp"
    assert tracker.update(_state(tcp=(0.0, 0.0, 0.03), grasped=True)).active_skill_name == "lift"
    assert tracker.update(
        _state(tcp=(0.0, 0.0, 0.08), obj=(0.0, 0.0, 0.08), grasped=True)
    ).active_skill_name == "transport"
    assert tracker.update(
        _state(tcp=(0.10, 0.0, 0.08), obj=(0.10, 0.0, 0.08), grasped=True)
    ).active_skill_name == "place"

    moving_release = _state(
        obj=(0.10, 0.0, 0.02),
        linear=(0.02, 0.0, 0.0),
        grasped=False,
    )
    assert tracker.update(moving_release).stable_place_steps == 0
    settled = _state(obj=(0.10, 0.0, 0.02), grasped=False)
    assert tracker.update(settled).task_completed is False
    assert tracker.update(settled).task_completed is False
    final = tracker.update(settled)

    assert final.task_completed is True
    assert final.completed_skill_count == 5


def test_place_cannot_complete_while_cube_is_still_grasped() -> None:
    config = PickPlacePredicateConfig()
    outcome = evaluate_pick_place_outcomes(
        _state(obj=(0.10, 0.0, 0.02), grasped=True),
        config,
    )

    assert outcome.placed_now is False


def test_predicate_config_rejects_unversioned_or_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="Predicate version"):
        PickPlacePredicateConfig(version="unversioned")
    with pytest.raises(ValueError, match="有限正数"):
        PickPlacePredicateConfig(reach_distance_m=0.0)
