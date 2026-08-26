import pytest

from robot_vla.contracts import (
    FRANKA_ARM_JOINT_NAMES,
    MODEL_ARCH,
    PROMPT_VERSION,
    QWEN_MODEL_ID,
    QWEN_REVISION,
    TRAJECTORY_SCHEMA_VERSION,
    RobotSpec,
    TaskSpec,
)


def test_version_identity_is_explicit() -> None:
    assert TRAJECTORY_SCHEMA_VERSION == "robot-vla-trajectory/v2"
    assert MODEL_ARCH == "qwen_vla_late_fusion_v1"
    assert PROMPT_VERSION == "qwen-vla-prompt/v1"
    assert QWEN_MODEL_ID == "Qwen/Qwen3.5-2B"
    assert QWEN_REVISION == "15852e8c16360a2fea060d615a32b45270f8a8fc"


def test_robot_spec_fixes_franka_dimensions_limits_and_rates() -> None:
    spec = RobotSpec()

    assert spec.arm_joint_names == FRANKA_ARM_JOINT_NAMES
    assert spec.active_joint_names[-2:] == (
        "panda_finger_joint1",
        "panda_finger_joint2",
    )
    assert spec.proprio_dim == 15
    assert spec.action_dim == 8
    assert spec.chunk_duration_s == pytest.approx(0.8)
    assert spec.replan_hz == pytest.approx(5.0)
    assert spec.effective_joint_delta_limits_rad == pytest.approx((0.05,) * 7)
    assert spec.joint_position_limits_rad[3] == pytest.approx((-3.0718, -0.0698))
    assert spec.joint_velocity_limits_rad_s == pytest.approx(
        (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)
    )


def test_robot_spec_serialization_round_trip() -> None:
    spec = RobotSpec()

    restored = RobotSpec.from_dict(spec.to_dict())

    assert restored == spec


def test_robot_spec_rejects_joint_order_drift() -> None:
    wrong_order = list(FRANKA_ARM_JOINT_NAMES)
    wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]

    with pytest.raises(ValueError, match="joint 顺序"):
        RobotSpec(arm_joint_names=tuple(wrong_order))


def test_task_spec_exposes_versioned_skill_mapping() -> None:
    task = TaskSpec(
        task_id="pick_cube_to_region",
        task_group_id="pick-and-place",
        instruction="Pick up the red cube and place it in the target region.",
    )

    assert task.skill_name(-1) == "unknown"
    assert task.skill_name(0) == "reach"
    assert task.skill_name(4) == "place"
    assert TaskSpec.from_dict(task.to_dict()) == task

    with pytest.raises(ValueError, match="未知 skill_id"):
        task.skill_name(5)
