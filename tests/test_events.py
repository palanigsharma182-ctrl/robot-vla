from dataclasses import replace

import numpy as np

from robot_vla.data.events import EventDetectionConfig, detect_trajectory_events


def _event_state(arrays):
    steps = arrays.num_steps
    return replace(
        arrays,
        robot_object_contact_force_n=np.zeros(steps, dtype=np.float32),
        support_contact_force_n=np.zeros(steps, dtype=np.float32),
        is_grasped=np.zeros(steps, dtype=np.bool_),
        object_position_m=np.zeros((steps, 3), dtype=np.float32),
        object_linear_velocity_m_s=np.zeros((steps, 3), dtype=np.float32),
        object_angular_velocity_rad_s=np.zeros((steps, 3), dtype=np.float32),
        commanded_joint_target_rad=np.zeros((steps, 7), dtype=np.float32),
        applied_joint_correction_rad=np.zeros((steps, 7), dtype=np.float32),
    )


def test_gripper_command_events_are_available_on_legacy_trajectory(arrays_factory) -> None:
    arrays = arrays_factory()
    action = arrays.action.copy()
    action[:, -1] = np.asarray([1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    arrays = replace(arrays, action=action)

    events = detect_trajectory_events(arrays)

    assert events.event_state_available is False
    assert np.flatnonzero(events.by_type["grasp_command"]).tolist() == [2]
    assert np.flatnonzero(events.by_type["release_command"]).tolist() == [4]
    assert events.counts["contact"] == 0


def test_state_transition_marks_the_action_that_caused_contact_and_place(
    arrays_factory,
) -> None:
    arrays = _event_state(arrays_factory())
    robot_contact = arrays.robot_object_contact_force_n.copy()
    robot_contact[2:] = 1.0
    support_contact = arrays.support_contact_force_n.copy()
    support_contact[:2] = 1.0
    support_contact[4:] = 1.0
    grasped = arrays.is_grasped.copy()
    grasped[2:4] = True
    linear = arrays.object_linear_velocity_m_s.copy()
    linear[3, 0] = 0.2
    arrays = replace(
        arrays,
        robot_object_contact_force_n=robot_contact,
        support_contact_force_n=support_contact,
        is_grasped=grasped,
        object_linear_velocity_m_s=linear,
    )

    events = detect_trajectory_events(
        arrays,
        EventDetectionConfig(
            linear_velocity_jump_threshold_m_s=0.1,
            angular_velocity_jump_threshold_rad_s=None,
        ),
    )

    assert np.flatnonzero(events.by_type["contact"]).tolist() == [1]
    assert np.flatnonzero(events.by_type["pickup"]).tolist() == [1]
    assert np.flatnonzero(events.by_type["place"]).tolist() == [3]
    assert np.flatnonzero(events.by_type["linear_velocity_jump"]).tolist() == [2, 3]
    assert events.event_mask[1]


def test_multiple_event_types_still_produce_one_binary_event_mask(arrays_factory) -> None:
    arrays = _event_state(arrays_factory())
    action = arrays.action.copy()
    action[:, -1] = 1.0
    action[1:, -1] = 0.0
    contact = arrays.robot_object_contact_force_n.copy()
    contact[2:] = 1.0
    arrays = replace(
        arrays,
        action=action,
        robot_object_contact_force_n=contact,
    )

    events = detect_trajectory_events(arrays)

    assert events.by_type["grasp_command"][1]
    assert events.by_type["contact"][1]
    assert events.event_mask.dtype == np.bool_
    assert int(events.event_mask[1]) == 1
