"""TCP Action Chunk 的纯离线诊断指标。

这个模块只比较同一 anchor 下的教师与学生 Chunk，不调用执行器、仿真或
任何 ground-truth 目标。关节检查也只经由传入的 FK/IK 对象完成，因此结果
只能说明离线计划是否可解以及关节增量是否落在既有阈值内。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.tcp_memory_control.geometry import apply_delta, pose


HORIZON = 16
PRIMARY_STEPS = 4
JOINT_DELTA_THRESHOLDS_RAD = (0.05, 0.1)
JOINT_DELTA_TOLERANCE_RAD = 1e-7  # 与 TCPChunkExecutor 的计划步长判定一致。


def _finite_array(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """转换并验证诊断输入，避免把 NaN/Inf 误报成很小的误差。"""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}必须是数值数组") from error
    if array.shape != shape:
        raise ValueError(f"{name}必须为{shape}，实际为{array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}必须全部有限")
    return array


def _rotation_geodesic_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    """计算两个旋转矩阵间的最短 SO(3) 角距离，单位为 degree。"""
    relative = first @ second.T
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec())))


def _rotvec_geodesic_error_deg(
    first_rotvecs: np.ndarray, second_rotvecs: np.ndarray
) -> list[float]:
    """逐步比较 rotvec 所代表的旋转，而非直接比较 rotvec 坐标。"""
    first = Rotation.from_rotvec(first_rotvecs)
    second = Rotation.from_rotvec(second_rotvecs)
    relative = first * second.inv()
    return [float(value) for value in np.degrees(np.linalg.norm(relative.as_rotvec(), axis=1))]


def _cumulative_targets(chunk: np.ndarray, anchor: np.ndarray) -> list[np.ndarray]:
    """用共同 anchor 组合实际执行的四步；未执行尾部不作累计几何结论。"""
    current = anchor
    targets: list[np.ndarray] = []
    for delta in chunk[:PRIMARY_STEPS]:
        current = apply_delta(current, delta[:6], anchor)
        targets.append(current)
    return targets


def _failure_reason(error: Exception) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _evaluate_kinematics(
    chunk: np.ndarray,
    targets: list[np.ndarray],
    actual_q: np.ndarray,
    kinematics: Any,
) -> dict[str, Any]:
    """在固定四步分母内做顺序 IK/FK 诊断。

    任一步 IK、返回关节向量或 FK 无效后，后续步骤不会用陈旧参考强行求解；
    它们仍计入 ``denominator``，并以 ``None`` 保留在逐步数组中。这样失败
    Chunk 绝不会因已成功的前缀而被算作通过。
    """
    sequential_joint_deltas: list[list[float] | None] = [None] * PRIMARY_STEPS
    per_step_joint_delta_max: list[float | None] = [None] * PRIMARY_STEPS
    fk_translation_error_mm: list[float | None] = [None] * PRIMARY_STEPS
    fk_rotation_geodesic_error_deg: list[float | None] = [None] * PRIMARY_STEPS

    reference_q = actual_q.copy()
    attempted_steps = 0
    successful_steps = 0
    first_ik_failure_step: int | None = None
    first_ik_failure_step_index: int | None = None
    first_ik_failure_reason: str | None = None
    first_over_005_step: int | None = None
    first_over_005_step_index: int | None = None
    first_over_01_step: int | None = None
    first_over_01_step_index: int | None = None

    for index, target in enumerate(targets[:PRIMARY_STEPS]):
        attempted_steps += 1
        try:
            next_q = _finite_array(
                kinematics.inverse(target, reference_q),
                name="IK返回关节",
                shape=(7,),
            )
            fk_target = pose(kinematics.pose_base(next_q))
            if (np.linalg.norm(fk_target[:3, 3] - target[:3, 3]) > 1e-4
                    or np.linalg.norm(Rotation.from_matrix(
                        fk_target[:3, :3] @ target[:3, :3].T).as_rotvec()) > 1e-3):
                raise ValueError('IK解未通过FK回代')
        except Exception as error:
            first_ik_failure_step = index + 1
            first_ik_failure_step_index = index
            first_ik_failure_reason = _failure_reason(error)
            break

        joint_delta = next_q - reference_q
        max_for_step = float(np.max(np.abs(joint_delta)))
        sequential_joint_deltas[index] = [float(value) for value in joint_delta]
        per_step_joint_delta_max[index] = max_for_step
        fk_translation_error_mm[index] = float(
            np.linalg.norm(fk_target[:3, 3] - target[:3, 3]) * 1000.0
        )
        fk_rotation_geodesic_error_deg[index] = _rotation_geodesic_error_deg(
            fk_target[:3, :3], target[:3, :3]
        )
        successful_steps += 1

        if max_for_step > JOINT_DELTA_THRESHOLDS_RAD[0] + JOINT_DELTA_TOLERANCE_RAD and first_over_005_step is None:
            first_over_005_step = index + 1
            first_over_005_step_index = index
        if max_for_step > JOINT_DELTA_THRESHOLDS_RAD[1] + JOINT_DELTA_TOLERANCE_RAD and first_over_01_step is None:
            first_over_01_step = index + 1
            first_over_01_step_index = index
        reference_q = next_q

    finite_maxima = [value for value in per_step_joint_delta_max if value is not None]
    max_joint_delta = float(max(finite_maxima)) if finite_maxima else None
    fully_solved = successful_steps == PRIMARY_STEPS
    pass_005 = bool(
        fully_solved
        and max_joint_delta is not None
        and max_joint_delta <= JOINT_DELTA_THRESHOLDS_RAD[0] + JOINT_DELTA_TOLERANCE_RAD
    )
    pass_01 = bool(
        fully_solved
        and max_joint_delta is not None
        and max_joint_delta <= JOINT_DELTA_THRESHOLDS_RAD[1] + JOINT_DELTA_TOLERANCE_RAD
    )

    # ``first_failure`` 采用更严格的 0.05 rad 执行合同；阈值字段仍让使用者
    # 明确判断 0.1 rad 下的结果。IK/FK 失败会使两个合同都不通过。
    failure_candidates: list[tuple[int, str]] = []
    if first_ik_failure_step is not None:
        failure_candidates.append((first_ik_failure_step, first_ik_failure_reason or "IK/FK失败"))
    if first_over_005_step is not None:
        failure_candidates.append((first_over_005_step, "joint_delta_exceeds_0.05_rad"))
    first_failure_step: int | None
    first_failure_reason: str | None
    if failure_candidates:
        first_failure_step, first_failure_reason = min(failure_candidates, key=lambda item: item[0])
    else:
        first_failure_step, first_failure_reason = None, None

    return {
        "joint_delta_tolerance_rad": JOINT_DELTA_TOLERANCE_RAD,
        "denominator": PRIMARY_STEPS,
        "total_steps": PRIMARY_STEPS,
        "attempted_steps": attempted_steps,
        "successful_steps": successful_steps,
        "unattempted_steps": PRIMARY_STEPS - attempted_steps,
        "failed_or_unavailable_steps": PRIMARY_STEPS - successful_steps,
        "sequential_joint_deltas": sequential_joint_deltas,
        "per_step_max_joint_delta": per_step_joint_delta_max,
        "max_joint_delta": max_joint_delta,
        "fk_translation_error_mm": fk_translation_error_mm,
        "fk_rotation_geodesic_error_deg": fk_rotation_geodesic_error_deg,
        "first_failure_step": first_failure_step,
        "first_failure_step_index": None if first_failure_step is None else first_failure_step - 1,
        "first_failure_reason": first_failure_reason,
        "first_ik_failure_step": first_ik_failure_step,
        "first_ik_failure_step_index": first_ik_failure_step_index,
        "first_ik_failure_reason": first_ik_failure_reason,
        "first_over_005_step": first_over_005_step,
        "first_over_005_step_index": first_over_005_step_index,
        "first_over_01_step": first_over_01_step,
        "first_over_01_step_index": first_over_01_step_index,
        "pass_005": pass_005,
        "pass_01": pass_01,
    }


def summarize_chunk(
    predicted_physical: Any,
    teacher_physical: Any,
    anchor: Any,
    actual_q: Any,
    kinematics: Any,
) -> dict[str, Any]:
    """汇总一个学生/教师 TCP Chunk 的可 JSON 序列化离线指标。

    Args:
        predicted_physical: 学生输出，有限 ``float[16, 7]``，通道为 ``xyz``、
            ``rotvec`` 和 gripper。
        teacher_physical: 同一观察/anchor 下的教师 Chunk，格式与学生相同。
        anchor: ``base_from_tcp`` 的有效 ``float[4, 4]`` pose；两条 Chunk 都从它
            开始累积，不以学生或教师各自的中间 pose 作为共同基准。
        actual_q: 当前实际的 7 维关节状态。顺序 IK 的第一步以它为参考，后续
            步以之前的 IK 解为参考。
        kinematics: 提供 ``inverse(target, reference_q)`` 与 ``pose_base(q)`` 的
            纯运动学对象。

    Returns:
        只含 Python 标量、list、dict、str、bool 和 ``None`` 的字典。动作和累计
        动作误差保留16步，累计pose误差只计算实际执行的前四步。IK
        结果位于 ``kinematics.teacher`` 与 ``kinematics.student``；其中 step 为
        人类可读的一起始编号，``*_index`` 为零起始索引。
    """
    predicted = _finite_array(
        predicted_physical, name="predicted_physical", shape=(HORIZON, 7)
    )
    teacher = _finite_array(
        teacher_physical, name="teacher_physical", shape=(HORIZON, 7)
    )
    # 先单独报出非有限 anchor；随后仍复用几何合同检查 SE(3) 结构。
    validated_anchor = pose(_finite_array(anchor, name="anchor", shape=(4, 4)))
    validated_actual_q = _finite_array(actual_q, name="actual_q", shape=(7,))
    if not np.allclose(pose(kinematics.pose_base(validated_actual_q)), validated_anchor,
                       atol=2e-5, rtol=0):
        raise ValueError('anchor与actual_q的FK不一致')

    translation_error_mm = np.linalg.norm(predicted[:, :3] - teacher[:, :3], axis=1) * 1000.0
    rotation_error_deg = _rotvec_geodesic_error_deg(predicted[:, 3:6], teacher[:, 3:6])
    gripper_absolute_error = np.abs(predicted[:, 6] - teacher[:, 6])

    predicted_targets = _cumulative_targets(predicted, validated_anchor)
    teacher_targets = _cumulative_targets(teacher, validated_anchor)
    cumulative_translation_error_mm = [
        float(np.linalg.norm(predicted_target[:3, 3] - teacher_target[:3, 3]) * 1000.0)
        for predicted_target, teacher_target in zip(predicted_targets, teacher_targets)
    ]
    cumulative_rotation_error_deg = [
        _rotation_geodesic_error_deg(predicted_target[:3, :3], teacher_target[:3, :3])
        for predicted_target, teacher_target in zip(predicted_targets, teacher_targets)
    ]

    return {
        "schema": "tcp-chunk-diagnostic-metrics/v1",
        "horizon": HORIZON,
        "primary_steps": PRIMARY_STEPS,
        "action_error": {
            "translation_error_mm": [float(value) for value in translation_error_mm],
            "rotation_geodesic_error_deg": rotation_error_deg,
            "gripper_absolute_error": [float(value) for value in gripper_absolute_error],
            "gripper_mae": float(np.mean(gripper_absolute_error)),
            "primary_translation_error_mm": [
                float(value) for value in translation_error_mm[:PRIMARY_STEPS]
            ],
            "primary_rotation_geodesic_error_deg": rotation_error_deg[:PRIMARY_STEPS],
            "primary_gripper_mae": float(np.mean(gripper_absolute_error[:PRIMARY_STEPS])),
        },
        "amplitudes": {
            "teacher": {
                "max_abs_translation_component_m": float(np.max(np.abs(teacher[:, :3]))),
                "max_abs_rotvec_component_rad": float(np.max(np.abs(teacher[:, 3:6]))),
            },
            "student": {
                "max_abs_translation_component_m": float(np.max(np.abs(predicted[:, :3]))),
                "max_abs_rotvec_component_rad": float(np.max(np.abs(predicted[:, 3:6]))),
            },
        },
        "cumulative_target_error": {
            "translation_error_mm": cumulative_translation_error_mm,
            "rotation_geodesic_error_deg": cumulative_rotation_error_deg,
            "primary_translation_error_mm": cumulative_translation_error_mm[:PRIMARY_STEPS],
            "primary_rotation_geodesic_error_deg": cumulative_rotation_error_deg[:PRIMARY_STEPS],
        },
        "kinematics": {
            "teacher": _evaluate_kinematics(
                teacher, teacher_targets, validated_actual_q, kinematics
            ),
            "student": _evaluate_kinematics(
                predicted, predicted_targets, validated_actual_q, kinematics
            ),
        },
    }
