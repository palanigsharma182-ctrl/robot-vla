"""TCP Chunk 诊断指标的合成反例；不依赖 GPU、仿真或机器人资源。"""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.tcp_chunk_diagnostic.metrics import summarize_chunk


def transform(translation=(0.0, 0.0, 0.0), rotvec=(0.0, 0.0, 0.0)):
    value = np.eye(4)
    value[:3, 3] = translation
    value[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    return value


class CartesianKinematics:
    """只供指标测试使用的确定性 FK/IK；q 的前六维直接编码 TCP pose。"""

    def __init__(self, fail_on_call=None):
        self.fail_on_call = fail_on_call
        self.calls = 0
        self.inverse_targets = []

    def pose_base(self, q):
        q = np.asarray(q, dtype=np.float64)
        return transform(q[:3], q[3:6])

    def inverse(self, target, reference):
        self.calls += 1
        self.inverse_targets.append(np.asarray(target, dtype=np.float64).copy())
        if self.calls == self.fail_on_call:
            raise ValueError("synthetic IK rejection")
        q = np.asarray(reference, dtype=np.float64).copy()
        q[:3] = target[:3, 3]
        q[3:6] = Rotation.from_matrix(target[:3, :3]).as_rotvec()
        return q


def empty_chunk():
    return np.zeros((16, 7), dtype=np.float64)


def test_noncommuting_rotations_and_fixed_anchor_cumulative_coordinates():
    teacher = empty_chunk()
    student = empty_chunk()
    # 此顺序在固定 anchor 坐标系中分别得到 Ry @ Rx 和 Rx @ Ry，不能交换。
    teacher[0, 3] = np.pi / 2
    teacher[1, 4] = np.pi / 2
    student[0, 4] = np.pi / 2
    student[1, 3] = np.pi / 2
    # 平移也应始终沿 anchor 轴，而不是沿已经转动的 TCP 轴。
    teacher[0, :3] = [.001, 0, 0]
    teacher[1, :3] = [0, .002, 0]
    student[:2, :3] = teacher[:2, :3]
    anchor = transform((.1, .2, .3), (0, 0, np.pi / 2))
    actual_q = np.array([.1, .2, .3, 0, 0, np.pi / 2, 0])
    kinematics = CartesianKinematics()

    result = summarize_chunk(student, teacher, anchor, actual_q, kinematics)

    # 前四次 inverse 是教师路径；第二个目标为 anchor + R_anchor @ [1, 2, 0] mm。
    np.testing.assert_allclose(
        kinematics.inverse_targets[1][:3, 3], [.098, .201, .3], atol=1e-12
    )
    assert result["action_error"]["translation_error_mm"][:2] == pytest.approx([0.0, 0.0])
    assert result["cumulative_target_error"]["rotation_geodesic_error_deg"][1] > 1.0
    json.dumps(result)


def test_pure_translation_has_zero_pose_error_and_reports_both_amplitudes():
    teacher = empty_chunk()
    student = empty_chunk()
    teacher[:4, :3] = [[.001, 0, 0], [0, .002, 0], [0, 0, -.003], [.004, 0, 0]]
    student[:] = teacher
    teacher[:, 6] = .25
    student[:, 6] = .25

    result = summarize_chunk(student, teacher, np.eye(4), np.zeros(7), CartesianKinematics())

    assert result["action_error"]["primary_translation_error_mm"] == pytest.approx([0.0] * 4)
    assert result["action_error"]["primary_rotation_geodesic_error_deg"] == pytest.approx([0.0] * 4)
    assert result["action_error"]["gripper_mae"] == pytest.approx(0.0)
    assert result["cumulative_target_error"]["primary_translation_error_mm"] == pytest.approx([0.0] * 4)
    assert result["amplitudes"]["teacher"]["max_abs_translation_component_m"] == pytest.approx(.004)
    assert result["amplitudes"]["student"]["max_abs_rotvec_component_rad"] == pytest.approx(0.0)
    assert result["kinematics"]["teacher"]["pass_005"] is True
    assert result["kinematics"]["student"]["pass_01"] is True


def test_ik_third_step_failure_keeps_four_step_denominator_and_threshold_boundary():
    chunk = empty_chunk()
    chunk[:4, 0] = .001
    kinematics = CartesianKinematics(fail_on_call=3)

    result = summarize_chunk(chunk, chunk, np.eye(4), np.zeros(7), kinematics)
    teacher_ik = result["kinematics"]["teacher"]

    assert teacher_ik["denominator"] == 4
    assert teacher_ik["attempted_steps"] == 3
    assert teacher_ik["successful_steps"] == 2
    assert teacher_ik["failed_or_unavailable_steps"] == 2
    assert teacher_ik["first_failure_step"] == 3
    assert teacher_ik["first_failure_step_index"] == 2
    assert "synthetic IK rejection" in teacher_ik["first_failure_reason"]
    assert teacher_ik["sequential_joint_deltas"][2:] == [None, None]
    assert teacher_ik["pass_005"] is False
    assert teacher_ik["pass_01"] is False

    at_limit = empty_chunk()
    at_limit[0, 0] = .05
    boundary = summarize_chunk(at_limit, at_limit, np.eye(4), np.zeros(7), CartesianKinematics())
    assert boundary["kinematics"]["teacher"]["max_joint_delta"] == pytest.approx(.05)
    assert boundary["kinematics"]["teacher"]["pass_005"] is True
    over_limit = at_limit.copy()
    over_limit[0, 0] = .0500002
    over = summarize_chunk(over_limit, over_limit, np.eye(4), np.zeros(7), CartesianKinematics())
    assert over["kinematics"]["teacher"]["pass_005"] is False
    assert over["kinematics"]["teacher"]["pass_01"] is True
    assert over["kinematics"]["teacher"]["first_over_005_step"] == 1


@pytest.mark.parametrize(
    "predicted,teacher,anchor,actual_q",
    [
        (np.where(np.indices((16, 7))[0] == 0, np.nan, 0.0), empty_chunk(), np.eye(4), np.zeros(7)),
        (empty_chunk(), np.where(np.indices((16, 7))[1] == 0, np.inf, 0.0), np.eye(4), np.zeros(7)),
        (empty_chunk(), empty_chunk(), np.full((4, 4), np.nan), np.zeros(7)),
        (empty_chunk(), empty_chunk(), np.eye(4), np.array([0, 0, 0, 0, 0, 0, np.inf])),
    ],
)
def test_nonfinite_inputs_are_rejected(predicted, teacher, anchor, actual_q):
    with pytest.raises(ValueError, match="有限"):
        summarize_chunk(predicted, teacher, anchor, actual_q, CartesianKinematics())


def test_false_ik_success_and_stale_anchor_are_rejected():
    class WrongIK(CartesianKinematics):
        def inverse(self, target, reference):
            return reference.copy()
    chunk = empty_chunk()
    chunk[:4, 0] = .001
    result = summarize_chunk(chunk, chunk, np.eye(4), np.zeros(7), WrongIK())
    assert result['kinematics']['student']['pass_01'] is False
    assert result['kinematics']['student']['first_ik_failure_step'] == 1
    with pytest.raises(ValueError, match='anchor'):
        summarize_chunk(chunk, chunk, transform((.1, 0, 0)), np.zeros(7), CartesianKinematics())


def test_only_executed_prefix_is_composed_with_float32_fk_roundoff():
    anchor = np.eye(4)
    anchor[:3, :3] *= 1 + 6e-7
    result = summarize_chunk(empty_chunk(), empty_chunk(), anchor, np.zeros(7), CartesianKinematics())
    assert len(result['cumulative_target_error']['translation_error_mm']) == 4
    assert len(result['action_error']['translation_error_mm']) == 16
