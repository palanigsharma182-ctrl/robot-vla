"""以合成预测测量回放 Object Memory；不加载模型、GT 或控制器。"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from robot_vla.precision.object_memory import (
    DualPrecisionWorldState,
    ExplicitObjectStateMemory,
    ObjectCandidateWindowVerifier,
    ObjectMeasurement,
    ObjectMemoryConfig,
    ObjectMemorySafetyContext,
    ObjectStateRequirement,
    resolve_object_state,
)
from robot_vla.precision.object_observability import ObjectWriteEvidence
from robot_vla.precision.state_memory import (
    ExplicitGoalStateMemory,
    GoalMeasurement,
    GoalMemoryConfig,
)

SOURCE = "synthetic-object-provider/v1"
CAMERA = "synthetic-wrist-camera"
WRITE_THRESHOLD = 0.7  # 仅用于合成接口验证，不是部署或实验选定阈值。
POSITION = (0.4, 0.1, 0.02)
SAFE = ObjectMemorySafetyContext(
    pregrasp_window_open=True,
    gripper_open=True,
    controller_tracking_valid=True,
    object_contact_detected=False,
    gripper_close_commanded=False,
    grasp_candidate=False,
    grasp_verified=False,
    object_maybe_moved=False,
)


def synthetic_measurement(timestamp_s: float, event: str) -> tuple[ObjectMeasurement, float]:
    """事件仅改变合成预测，不通过离线 observability label 生成写入许可。"""
    missing = event in ("occluded", "expired")
    geometry_valid = not missing and event != "bad_geometry"
    evidence = ObjectWriteEvidence(
        visibility_probability=0.1 if event == "low_score" else 0.95,
        projection_validity_probability=0.95,
        object_mask_probability=0.0 if missing else 0.95,
        goal_mask_probability=0.0,
        normalized_entropy=0.05,
        radial_sigma_px=0.1,
        geometry_valid=geometry_valid,
    )
    position = None if missing else POSITION
    if event == "low_score":
        position = (0.8, 0.1, 0.02)  # 拒绝的异位预测不能覆盖已有状态。
    elif event == "new_position":
        position = (0.6, 0.1, 0.02)
    measurement = ObjectMeasurement(
        timestamp_s=timestamp_s,
        rgb_timestamp_s=timestamp_s,
        camera_pose_timestamp_s=timestamp_s,
        tcp_pose_timestamp_s=timestamp_s,
        position_base_m=position,
        covariance_base_m2=None if missing else np.eye(3) * 1e-6,
        confidence=evidence.score,
        projection_valid=True,
        in_fov=True,
        observable=evidence.observable,
        geometry_valid=geometry_valid,
        write_gate_passed=evidence.accepted(threshold=WRITE_THRESHOLD),
        source_camera=CAMERA,
        source_model_identity="synthetic-other-provider/v1" if event == "source_change" else SOURCE,
    )
    return measurement, evidence.score


def run_replay() -> dict:
    """同一组 memory/verifier 显式 reset 后重用，记录每一步的真实接口输出。"""
    config = ObjectMemoryConfig(
        max_unobserved_age_s=0.5,
        max_innovation_m=0.01,
        max_position_std_m=0.02,
        min_candidate_frames=2,
        max_candidate_gap_s=0.1,
        max_candidate_position_spread_m=0.005,
        max_sensor_skew_s=0.01,
        expected_source_camera=CAMERA,
        expected_source_model_identity=SOURCE,
    )
    memory = ExplicitObjectStateMemory(config)
    verifier = ObjectCandidateWindowVerifier(config)
    goal = ExplicitGoalStateMemory(GoalMemoryConfig(
        max_unobserved_age_s=2.0, max_innovation_m=0.01, max_position_std_m=0.02,
    ))
    scenarios = {
        "hold_and_expiry": [
            (0.0, "low_score"), (0.05, "bad_geometry"),
            (0.10, "visible"), (0.15, "visible"),
            (0.20, "low_score"), (0.25, "occluded"), (0.70, "expired"),
        ],
        "contact": [
            (0.0, "visible"), (0.05, "visible"), (0.10, "contact"),
            (0.15, "visible"), (0.20, "visible"),
        ],
        "source_change": [
            (0.0, "visible"), (0.05, "visible"), (0.10, "source_change"),
            (0.15, "visible"), (0.20, "visible"),
        ],
        "after_reset": [(0.0, "new_position"), (0.05, "new_position")],
    }
    rows = []
    resets = []
    for episode_id, frames in scenarios.items():
        memory.reset(episode_id)
        verifier.reset(episode_id)
        goal.reset(episode_id)
        resets.append({
            "episode_id": episode_id,
            "object_position": memory.state.position_base_m,
            "object_valid": memory.state.valid,
            "accepted_update_count": memory.state.accepted_update_count,
            "goal_position": goal.state.position_base_m,
        })
        for timestamp_s, event in frames:
            # 独立、明确标记的合成 Goal 测量，用于验证两份状态在同一 Episode 共存。
            goal.update(GoalMeasurement(
                timestamp_s=timestamp_s,
                position_base_m=(0.6, -0.1, 0.02),
                covariance_base_m2=np.eye(3) * 1e-6,
                confidence=0.95,
                goal_exists=True, projection_valid=True, in_fov=True,
                observable=True, geometry_valid=True, write_gate_passed=True,
                source="synthetic-goal-provider/v1",
            ), episode_id=episode_id)
            goal_before = goal.state
            measurement, score = synthetic_measurement(timestamp_s, event)
            safety = replace(SAFE, object_contact_detected=event == "contact")
            candidate = verifier.observe(measurement, episode_id=episode_id, safety=safety)
            update = memory.update(candidate, episode_id=episode_id, safety=safety)
            navigation = resolve_object_state(update, requirement=ObjectStateRequirement.NAVIGATION)
            world = DualPrecisionWorldState(goal=goal.state, object=memory.state)
            rows.append({
                "episode_id": episode_id, "event": event, "timestamp_s": timestamp_s,
                "write_score": score, "write_gate_passed": measurement.write_gate_passed,
                "measurement_accepted": update.measurement_accepted,
                "mode": update.state.mode.value, "valid": update.state.valid,
                "stored_position": update.state.position_base_m,
                "age_s": update.state.age_s,
                "navigation_position": navigation.position_base_m,
                "navigation_available": navigation.available,
                "memory_only": navigation.memory_only,
                "contact_authorized": navigation.contact_authorized,
                "rejection_reasons": update.rejection_reasons,
                "invalid_reasons": update.state.invalid_reasons,
                "goal_unchanged": goal.state == goal_before,
                "goal_position": world.estimated_goal_position_base_m,
            })
    return {
        "evidence_level": "synthetic-interface-replay",
        "source_identity": SOURCE,
        "write_threshold": WRITE_THRESHOLD,
        "max_unobserved_age_s": config.max_unobserved_age_s,
        "resets": resets, "rows": rows,
    }


if __name__ == "__main__":
    print(json.dumps(run_replay(), ensure_ascii=False, indent=2, allow_nan=False))
