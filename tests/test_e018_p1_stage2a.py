from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from robot_vla.contracts import RobotSpec
from robot_vla.executive.contracts import PhaseId
from robot_vla.observation import (
    OBSERVATION_MODALITIES,
    ObservationV2Frame,
    ObservationV2History,
    opengl_camera_to_opencv,
)
from robot_vla.precision.active_front_memory_provider import (
    ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
)
from robot_vla.precision.active_front_reobserve import (
    ActiveFrontDecisionReason,
    ActiveFrontReobserveConfig,
    ActiveFrontReobserveController,
    ActiveFrontReobserveState,
    ActiveFrontSafetyEvidence,
    ActiveFrontSignal,
    ActiveFrontTriggerEvidence,
    ActiveFrontTriggerReason,
    HomeV2BarrierFrame,
    Stage2MemoryCandidateReceipt,
)
from robot_vla.precision.calibrated_front_provider import canonical_sha256
from robot_vla.precision.e018_p1_stage2a import (
    STAGE2A_INTEGRATION_SMOKE_GO,
    Stage2AActionHistoryRuntime,
    Stage2AExecutionProgress,
    _STAGE2A_COMPLETE_ARTIFACT_FILES,
    _array_sha256,
    _build_observation_v2_window_identity,
    _new_stage2a_replay_controller,
    _record_stage2a_failure_evidence,
    _stage2a_episode_id,
    _verify_stage2a_camera_authorization,
    _verify_stage2a_controller_receipt,
    _verify_stage2a_exact_file_tree,
    _verify_stage2a_safety_record,
    _verify_stage2a_source_recheck_identity,
    _verify_stage2a_trigger_replay,
    _stage2a_safety_evidence_record,
    build_absent_wrist_capability_record,
    build_trigger_evidence_from_capability,
    home_observation_payload_identity,
    load_e018_p1_stage2a_config,
    run_e018_p1_stage2a_integration_smoke,
    verify_stage2a_action_history_audit,
    verify_stage2a_failure_evidence,
    verify_stage2a_observation_v2_window_identity,
)
from robot_vla.precision.object_memory import ObjectMemoryMode, ObjectState


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STAGE2_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "e018_p1_stage2a_primary_memory_development_v1.json"
)


def _uninitialized_state(episode_id: str) -> ObjectState:
    return ObjectState(
        episode_id=episode_id,
        mode=ObjectMemoryMode.UNINITIALIZED,
        position_base_m=None,
        covariance_base_m2=None,
        measurement_confidence=0.0,
        last_observed_timestamp_s=None,
        state_timestamp_s=0.0,
        observable_now=False,
        valid=False,
        accepted_update_count=0,
        source_camera=None,
        source_model_identity=None,
        invalid_reasons=("memory_uninitialized",),
    )


def _valid_state(episode_id: str, *, timestamp_s: float) -> ObjectState:
    return ObjectState(
        episode_id=episode_id,
        mode=ObjectMemoryMode.FREE_STATIC,
        position_base_m=(0.45, 0.0, 0.03),
        covariance_base_m2=(
            (1e-4, 0.0, 0.0),
            (0.0, 1e-4, 0.0),
            (0.0, 0.0, 1e-4),
        ),
        measurement_confidence=0.9,
        last_observed_timestamp_s=timestamp_s,
        state_timestamp_s=timestamp_s,
        observable_now=False,
        valid=True,
        accepted_update_count=1,
        source_camera="base_camera",
        source_model_identity="stage2a-test-provider",
        invalid_reasons=(),
    )


def _observation(
    external_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    *,
    wrist_world_from_gl: np.ndarray | None = None,
) -> dict[str, object]:
    wrist_pose = (
        np.eye(4, dtype=np.float64)
        if wrist_world_from_gl is None
        else np.asarray(wrist_world_from_gl, dtype=np.float64)
    )
    return {
        "sensor_data": {
            "base_camera": {"rgb": external_rgb[None]},
            "hand_camera": {"rgb": wrist_rgb[None]},
        },
        "sensor_param": {
            "base_camera": {"cam2world_gl": np.eye(4, dtype=np.float64)[None]},
            "hand_camera": {"cam2world_gl": wrist_pose[None]},
        },
    }


def _trigger_transaction(
    episode_id: str,
) -> tuple[dict[str, object], ActiveFrontReobserveController]:
    loaded = load_e018_p1_stage2a_config(STAGE2_CONFIG)
    controller = _new_stage2a_replay_controller(loaded, episode_id=episode_id)
    memory = _uninitialized_state(episode_id)
    request_id = f"{episode_id}-active-front-01"
    observation = _observation(
        np.zeros((4, 5, 3), dtype=np.uint8),
        np.ones((4, 5, 3), dtype=np.uint8),
    )
    records = []
    decisions = []
    for tick in range(3):
        record = build_absent_wrist_capability_record(
            observation=observation,
            episode_id=episode_id,
            episode_generation=1,
            request_id=request_id,
            record_role="trigger",
            observation_sequence_id=f"{episode_id}-trigger-home-{tick:02d}",
            timestamp_s=tick * 0.05,
            memory_state=memory,
        )
        decision = controller.consider_trigger(
            build_trigger_evidence_from_capability(
                record,
                control_tick=tick,
                arm_hold_prerequisites_pass=True,
                camera_home_prerequisites_pass=True,
            )
        )
        public = {
            "version": "e018-p1-stage2a-capability-trigger-decision/v1",
            "control_tick": tick,
            "timestamp_s": tick * 0.05,
            "capability_evidence_identity_sha256": record.digest,
            "requestable": decision.requestable,
            "reason": decision.reason.value,
            "consecutive_unusable_ticks": decision.consecutive_unusable_ticks,
        }
        public["decision_sha256"] = canonical_sha256(public)
        records.append(record)
        decisions.append(public)
    source = build_absent_wrist_capability_record(
        observation=observation,
        episode_id=episode_id,
        episode_generation=1,
        request_id=request_id,
        record_role="source_recheck",
        observation_sequence_id=f"{episode_id}-source-recheck-home",
        timestamp_s=0.151,
        memory_state=memory,
    )
    transaction: dict[str, object] = {
        "request_id": request_id,
        "trigger_wrist_capability_records": [value.to_dict() for value in records],
        "trigger_wrist_capability": records[-1].to_dict(),
        "source_recheck_wrist_capability": source.to_dict(),
        "trigger_decisions": decisions,
        "capability_absence_trigger_reason": (
            ActiveFrontTriggerReason.NO_QUALIFIED_WRIST_PROVIDER_IN_PARENT.value
        ),
    }
    return transaction, controller


def _requested_controller(
    episode_id: str,
) -> tuple[ActiveFrontReobserveController, object]:
    controller = ActiveFrontReobserveController(
        ActiveFrontReobserveConfig(
            enabled=True,
            selected_primitive_id=ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
            consecutive_unusable_ticks=3,
            maximum_attempts_per_episode=1,
            home_v2_barrier_frames=4,
            allow_capability_absent_trigger=True,
        )
    )
    controller.reset_episode(episode_id, episode_generation=1)
    decision = None
    for tick in range(3):
        decision = controller.consider_trigger(
            ActiveFrontTriggerEvidence(
                episode_id=episode_id,
                episode_generation=1,
                control_tick=tick,
                timestamp_s=tick * 0.05,
                source_phase=PhaseId.ACQUIRE_TRACK,
                wrist_object_measurement_usable=False,
                front_home_object_measurement_usable=False,
                object_memory_navigation_state_available=False,
                arm_hold_prerequisites_pass=True,
                camera_home_prerequisites_pass=True,
                failure_reason=(
                    ActiveFrontTriggerReason.NO_QUALIFIED_WRIST_PROVIDER_IN_PARENT
                ),
            )
        )
    assert decision is not None and decision.request is not None
    return controller, decision.request


def _window_packet(
    episode_id: str,
) -> tuple[RobotSpec, dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    spec = RobotSpec()
    history = ObservationV2History(spec)
    home_evidence: list[dict[str, object]] = []
    motion_rows: list[dict[str, object]] = []
    ids: list[str] = []
    arm_q = np.asarray((0.0, -0.5, 0.0, -1.0, 0.0, 1.0, 0.0), dtype=np.float32)
    arm_dq = np.asarray((0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07), dtype=np.float32)
    finger_q = np.asarray((0.04, 0.04), dtype=np.float32)
    wrist_world_from_gl = np.eye(4, dtype=np.float64)
    wrist_world_from_gl[0, 3] = 0.2
    base_from_wrist = opengl_camera_to_opencv(wrist_world_from_gl)
    for index, frame_index in enumerate((88, 89, 90, 91)):
        timestamp = 4.5 + index * 0.05
        external = np.full((4, 5, 3), index, dtype=np.uint8)
        wrist = np.full((4, 5, 3), index + 10, dtype=np.uint8)
        observation = _observation(
            external,
            wrist,
            wrist_world_from_gl=wrist_world_from_gl,
        )
        payload = home_observation_payload_identity(
            observation,
            timestamp_s=timestamp,
        )
        sequence_id = f"{episode_id}-home-v2-{frame_index:02d}"
        ids.append(sequence_id)
        row: dict[str, object] = {
            "frame_index": frame_index,
            "episode_id": episode_id,
            "world_from_robot_base": np.eye(4, dtype=np.float64).tolist(),
            "tcp_current_world": np.eye(4, dtype=np.float64).tolist(),
            "arm_current_q_rad": arm_q.tolist(),
            "arm_current_dq_rad_s": arm_dq.tolist(),
            "finger_joint_positions_m": finger_q.tolist(),
            "finger_force_left_n": 0.0,
            "finger_force_right_n": 0.0,
        }
        evidence: dict[str, object] = {
            "observation_sequence_id": sequence_id,
            "control_timestamp_s": timestamp,
            "route_frame_index": frame_index,
            "motion_row_sha256": canonical_sha256(row),
            "home_observation_payload": payload,
        }
        evidence["evidence_sha256"] = canonical_sha256(evidence)
        motion_rows.append(row)
        home_evidence.append(evidence)
        history.append(
            ObservationV2Frame(
                rgb_external=external,
                rgb_wrist=wrist,
                physical_proprio=np.concatenate(
                    (arm_q, arm_dq, np.ones(1, dtype=np.float32))
                ),
                base_from_tcp=np.eye(4, dtype=np.float32),
                base_from_wrist_camera=base_from_wrist.astype(np.float32),
                finger_force_n=np.zeros(2, dtype=np.float32),
                timestamp_s=timestamp,
                modality_timestamp_s=np.full(
                    len(OBSERVATION_MODALITIES), timestamp, dtype=np.float64
                ),
                modality_valid=np.ones(
                    len(OBSERVATION_MODALITIES), dtype=np.bool_
                ),
            )
        )
    window = history.snapshot(
        "E018 Stage 2A fresh HOME shadow replan; no actuator",
        previous_command_q=None,
        previous_action=None,
    )
    identity = _build_observation_v2_window_identity(
        window,
        spec=spec,
        episode_id=episode_id,
        episode_generation=1,
        observation_sequence_ids=ids,
        home_evidence=home_evidence,
    )
    return spec, identity, home_evidence, motion_rows


def _resign_window(identity: dict[str, object]) -> None:
    primitive = dict(identity)
    primitive.pop("window_sha256", None)
    identity["window_sha256"] = canonical_sha256(primitive)


def _resign_evidence(evidence: dict[str, object]) -> None:
    primitive = dict(evidence)
    primitive.pop("evidence_sha256", None)
    evidence["evidence_sha256"] = canonical_sha256(primitive)


def test_stage2a_config_import_and_identity_are_stable() -> None:
    loaded = load_e018_p1_stage2a_config(STAGE2_CONFIG)
    assert loaded.payload["splits"]["integration_smoke"] == [76901, 76910]
    assert loaded.payload["execution"]["allow_capability_absent_trigger"] is True
    assert len(loaded.raw_sha256) == len(loaded.canonical_sha256) == 64


def test_capability_absence_requires_explicit_stage2_override() -> None:
    episode_id = "stage2a-default-disabled"
    record = build_absent_wrist_capability_record(
        observation=_observation(
            np.zeros((4, 5, 3), dtype=np.uint8),
            np.zeros((4, 5, 3), dtype=np.uint8),
        ),
        episode_id=episode_id,
        episode_generation=1,
        request_id=f"{episode_id}-active-front-01",
        record_role="trigger",
        observation_sequence_id=f"{episode_id}-trigger-home-00",
        timestamp_s=0.0,
        memory_state=_uninitialized_state(episode_id),
    )
    controller = ActiveFrontReobserveController(
        ActiveFrontReobserveConfig(enabled=True, consecutive_unusable_ticks=3)
    )
    controller.reset_episode(episode_id, episode_generation=1)
    decision = controller.consider_trigger(
        build_trigger_evidence_from_capability(
            record,
            control_tick=0,
            arm_hold_prerequisites_pass=True,
            camera_home_prerequisites_pass=True,
        )
    )
    assert decision.requestable is False
    assert decision.reason is ActiveFrontDecisionReason.FAILURE_NOT_VIEWPOINT_RESOLVABLE


def test_invalid_sensor_never_uses_capability_absence_override() -> None:
    controller, _ = _requested_controller("discarded-controller")
    del controller
    episode_id = "stage2a-invalid-sensor"
    checked = ActiveFrontReobserveController(
        ActiveFrontReobserveConfig(
            enabled=True,
            consecutive_unusable_ticks=3,
            allow_capability_absent_trigger=True,
        )
    )
    checked.reset_episode(episode_id, episode_generation=1)
    evidence = ActiveFrontTriggerEvidence(
        episode_id=episode_id,
        episode_generation=1,
        control_tick=0,
        timestamp_s=0.0,
        source_phase=PhaseId.ACQUIRE_TRACK,
        wrist_object_measurement_usable=False,
        front_home_object_measurement_usable=False,
        object_memory_navigation_state_available=False,
        arm_hold_prerequisites_pass=True,
        camera_home_prerequisites_pass=True,
        failure_reason=ActiveFrontTriggerReason.INVALID_SENSOR_OR_POSE,
    )
    decision = checked.consider_trigger(evidence)
    assert decision.requestable is False
    assert decision.reason is ActiveFrontDecisionReason.FAILURE_NOT_VIEWPOINT_RESOLVABLE


def test_three_plus_one_capability_records_replay_without_visual_failure_claim() -> None:
    episode_id = "stage2a-trigger-replay"
    transaction, _ = _trigger_transaction(episode_id)
    replay = _new_stage2a_replay_controller(
        load_e018_p1_stage2a_config(STAGE2_CONFIG),
        episode_id=episode_id,
    )
    triggers, source = _verify_stage2a_trigger_replay(
        transaction,
        controller=replay,
        episode_id=episode_id,
    )
    assert len(triggers) == 3
    assert len({value.digest for value in (*triggers, source)}) == 4
    assert [value["requestable"] for value in transaction["trigger_decisions"]] == [
        False,
        False,
        True,
    ]
    assert all(value.frame_evaluated is False for value in (*triggers, source))


def test_trigger_alias_tamper_is_rejected_even_when_record_is_resigned() -> None:
    episode_id = "stage2a-trigger-tamper"
    transaction, _ = _trigger_transaction(episode_id)
    transaction["trigger_wrist_capability"] = copy.deepcopy(
        transaction["trigger_wrist_capability_records"][1]
    )
    replay = _new_stage2a_replay_controller(
        load_e018_p1_stage2a_config(STAGE2_CONFIG),
        episode_id=episode_id,
    )
    with pytest.raises(ValueError, match="trigger alias"):
        _verify_stage2a_trigger_replay(
            transaction,
            controller=replay,
            episode_id=episode_id,
        )


def test_observation_v2_window_replays_from_four_home_frames() -> None:
    episode_id = "stage2a-home-window"
    spec, identity, evidence, rows = _window_packet(episode_id)
    window = verify_stage2a_observation_v2_window_identity(
        identity,
        spec=spec,
        home_evidence=evidence,
        home_motion_rows=rows,
        expected_episode_id=episode_id,
        expected_episode_generation=1,
    )
    assert window.history_valid.tolist() == [True, True, True, True]
    assert window.modality_valid.all()
    assert window.controller_valid.tolist() == [False, False]
    assert np.allclose(np.diff(window.frame_timestamp_s), 0.05)


@pytest.mark.parametrize(
    "field",
    ["physical_proprio", "tcp_position", "wrist_position", "finger_force_n"],
)
def test_observation_v2_missing_required_array_is_rejected(field: str) -> None:
    episode_id = f"stage2a-missing-{field}"
    spec, identity, evidence, rows = _window_packet(episode_id)
    del identity["arrays"][field]
    _resign_window(identity)
    with pytest.raises(ValueError, match="arrays"):
        verify_stage2a_observation_v2_window_identity(
            identity,
            spec=spec,
            home_evidence=evidence,
            home_motion_rows=rows,
            expected_episode_id=episode_id,
            expected_episode_generation=1,
        )


def test_observation_v2_resigned_force_witness_tamper_is_rejected() -> None:
    episode_id = "stage2a-force-tamper"
    spec, identity, evidence, rows = _window_packet(episode_id)
    rows[2]["finger_force_left_n"] = 0.005
    evidence[2]["motion_row_sha256"] = canonical_sha256(rows[2])
    _resign_evidence(evidence[2])
    identity["home_evidence_digests"][2] = evidence[2]["evidence_sha256"]
    _resign_window(identity)
    with pytest.raises(ValueError, match="raw witness"):
        verify_stage2a_observation_v2_window_identity(
            identity,
            spec=spec,
            home_evidence=evidence,
            home_motion_rows=rows,
            expected_episode_id=episode_id,
            expected_episode_generation=1,
        )


def test_observation_v2_resigned_dq_tamper_is_rejected() -> None:
    episode_id = "stage2a-dq-tamper"
    spec, identity, evidence, rows = _window_packet(episode_id)
    proprio = identity["arrays"]["physical_proprio"]
    values = np.asarray(proprio["values"], dtype=np.float32)
    values[:, spec.arm_dof : 2 * spec.arm_dof] = np.float32(0.314159)
    proprio["values"] = values.tolist()
    proprio["array_sha256"] = _array_sha256(values)
    _resign_window(identity)
    with pytest.raises(ValueError, match="raw witness"):
        verify_stage2a_observation_v2_window_identity(
            identity,
            spec=spec,
            home_evidence=evidence,
            home_motion_rows=rows,
            expected_episode_id=episode_id,
            expected_episode_generation=1,
        )


def test_observation_v2_resigned_wrist_pose_tamper_is_rejected() -> None:
    episode_id = "stage2a-wrist-pose-tamper"
    spec, identity, evidence, rows = _window_packet(episode_id)
    wrist = evidence[1]["home_observation_payload"]["cameras"]["wrist"]
    wrist["cam2world_gl"][0][3] += 0.1
    pose = np.asarray(wrist["cam2world_gl"], dtype=np.dtype(wrist["cam2world_gl_dtype"]))
    pose = pose.reshape(tuple(wrist["cam2world_gl_shape"]))
    wrist["cam2world_gl_sha256"] = _array_sha256(pose)
    _resign_evidence(evidence[1])
    identity["home_evidence_digests"][1] = evidence[1]["evidence_sha256"]
    _resign_window(identity)
    with pytest.raises(ValueError, match="raw witness"):
        verify_stage2a_observation_v2_window_identity(
            identity,
            spec=spec,
            home_evidence=evidence,
            home_motion_rows=rows,
            expected_episode_id=episode_id,
            expected_episode_generation=1,
        )


def test_observation_v2_old_controller_state_is_rejected_after_resigning() -> None:
    episode_id = "stage2a-stale-controller"
    spec, identity, evidence, rows = _window_packet(episode_id)
    controller_valid = identity["arrays"]["controller_valid"]
    controller_valid["values"] = [True, False]
    controller_valid["array_sha256"] = _array_sha256(
        np.asarray(controller_valid["values"], dtype=np.bool_)
    )
    previous = identity["arrays"]["previous_command_q"]
    previous["values"] = identity["arrays"]["physical_proprio"]["values"][-1][:7]
    previous["array_sha256"] = _array_sha256(
        np.asarray(previous["values"], dtype=np.float32)
    )
    _resign_window(identity)
    with pytest.raises(ValueError, match="controller state"):
        verify_stage2a_observation_v2_window_identity(
            identity,
            spec=spec,
            home_evidence=evidence,
            home_motion_rows=rows,
            expected_episode_id=episode_id,
            expected_episode_generation=1,
        )


def test_observation_v2_home_digest_order_and_episode_tamper_are_rejected() -> None:
    episode_id = "stage2a-home-order"
    spec, identity, evidence, rows = _window_packet(episode_id)
    tampered = copy.deepcopy(identity)
    tampered["home_evidence_digests"][0], tampered["home_evidence_digests"][1] = (
        tampered["home_evidence_digests"][1],
        tampered["home_evidence_digests"][0],
    )
    _resign_window(tampered)
    with pytest.raises(ValueError, match="digest顺序"):
        verify_stage2a_observation_v2_window_identity(
            tampered,
            spec=spec,
            home_evidence=evidence,
            home_motion_rows=rows,
            expected_episode_id=episode_id,
            expected_episode_generation=1,
        )
    with pytest.raises(ValueError, match="identity"):
        verify_stage2a_observation_v2_window_identity(
            identity,
            spec=spec,
            home_evidence=evidence,
            home_motion_rows=rows,
            expected_episode_id=f"{episode_id}-other",
            expected_episode_generation=1,
        )


def test_source_recheck_is_independent_and_bound_to_last_home_payload() -> None:
    episode_id = "stage2a-source-recheck"
    transaction, _ = _trigger_transaction(episode_id)
    replay = _new_stage2a_replay_controller(
        load_e018_p1_stage2a_config(STAGE2_CONFIG), episode_id=episode_id
    )
    triggers, _ = _verify_stage2a_trigger_replay(
        transaction, controller=replay, episode_id=episode_id
    )
    _, _, home_evidence, _ = _window_packet(episode_id)
    wrist_world_from_gl = np.eye(4, dtype=np.float64)
    wrist_world_from_gl[0, 3] = 0.2
    final_observation = _observation(
        np.full((4, 5, 3), 3, dtype=np.uint8),
        np.full((4, 5, 3), 13, dtype=np.uint8),
        wrist_world_from_gl=wrist_world_from_gl,
    )
    timestamp = float(home_evidence[-1]["control_timestamp_s"]) + 0.001
    source = build_absent_wrist_capability_record(
        observation=final_observation,
        episode_id=episode_id,
        episode_generation=1,
        request_id=f"{episode_id}-active-front-01",
        record_role="source_recheck",
        observation_sequence_id=f"{episode_id}-source-recheck-home",
        timestamp_s=timestamp,
        memory_state=_uninitialized_state(episode_id),
    )
    _verify_stage2a_source_recheck_identity(
        source,
        trigger_records=triggers,
        final_home_evidence=home_evidence[-1],
        episode_id=episode_id,
        request_id=f"{episode_id}-active-front-01",
    )
    stale = build_absent_wrist_capability_record(
        observation=final_observation,
        episode_id=episode_id,
        episode_generation=1,
        request_id=f"{episode_id}-active-front-01",
        record_role="source_recheck",
        observation_sequence_id=f"{episode_id}-source-recheck-home",
        timestamp_s=timestamp + 0.01,
        memory_state=_uninitialized_state(episode_id),
    )
    with pytest.raises(ValueError, match="fresh HOME"):
        _verify_stage2a_source_recheck_identity(
            stale,
            trigger_records=triggers,
            final_home_evidence=home_evidence[-1],
            episode_id=episode_id,
            request_id=f"{episode_id}-active-front-01",
        )


def test_action_history_before_after_and_fresh_home_receipts_recompute() -> None:
    episode_id = "stage2a-action-history"
    controller, request = _requested_controller(episode_id)
    runtime = Stage2AActionHistoryRuntime(episode_id)
    reset, reset_audit = runtime.invalidate_for_active_request(request)
    assert verify_stage2a_action_history_audit(reset_audit) == reset
    spec, identity, evidence, rows = _window_packet(episode_id)
    verify_stage2a_observation_v2_window_identity(
        identity,
        spec=spec,
        home_evidence=evidence,
        home_motion_rows=rows,
        expected_episode_id=episode_id,
        expected_episode_generation=1,
    )
    resume, resume_audit = runtime.generate_fresh_shadow_replan(
        request,
        home_evidence=evidence,
        observation_v2_window_identity=identity,
        memory_state=_valid_state(episode_id, timestamp_s=4.65),
        source_phase=PhaseId.ACQUIRE_TRACK,
    )
    assert verify_stage2a_action_history_audit(resume_audit) == resume
    assert reset_audit["after"] == resume_audit["before"]
    assert resume.observation_v2_window_sha256 == identity["window_sha256"]
    assert resume.stale_action_chunk_resumed is False
    del controller


def test_resigned_resume_window_digest_tamper_is_rejected() -> None:
    episode_id = "stage2a-resume-window-tamper"
    _, request = _requested_controller(episode_id)
    runtime = Stage2AActionHistoryRuntime(episode_id)
    runtime.invalidate_for_active_request(request)
    _, identity, evidence, _ = _window_packet(episode_id)
    _, audit = runtime.generate_fresh_shadow_replan(
        request,
        home_evidence=evidence,
        observation_v2_window_identity=identity,
        memory_state=_valid_state(episode_id, timestamp_s=4.65),
        source_phase=PhaseId.ACQUIRE_TRACK,
    )
    tampered = copy.deepcopy(audit)
    tampered["receipt"]["observation_v2_window_sha256"] = "f" * 64
    primitive = dict(tampered)
    primitive.pop("audit_sha256")
    tampered["audit_sha256"] = canonical_sha256(primitive)
    with pytest.raises(ValueError, match="fresh HOME shadow replan"):
        verify_stage2a_action_history_audit(tampered)


def test_resigned_reset_sentinel_tamper_is_rejected() -> None:
    episode_id = "stage2a-reset-tamper"
    _, request = _requested_controller(episode_id)
    runtime = Stage2AActionHistoryRuntime(episode_id)
    _, audit = runtime.invalidate_for_active_request(request)
    tampered = copy.deepcopy(audit)
    tampered["before"]["action_chunk_identity_sha256"] = "a" * 64
    tampered["before_sha256"] = canonical_sha256(tampered["before"])
    primitive = dict(tampered)
    primitive.pop("audit_sha256")
    tampered["audit_sha256"] = canonical_sha256(primitive)
    with pytest.raises(ValueError, match="ready sentinel"):
        verify_stage2a_action_history_audit(tampered)


def test_resigned_raw_safety_witness_tamper_is_rejected() -> None:
    episode_id = "stage2a-safety-tamper"
    controller, _ = _requested_controller(episode_id)
    row: dict[str, object] = {
        "frame_index": 0,
        "arm_joint_max_drift_rad": 0.0,
        "tcp_position_drift_m": 0.0,
        "tcp_orientation_drift_rad": 0.0,
        "minimum_finger_joint_position_m": 0.04,
        "finger_object_contact_force_n": 0.0,
        "arm_anchor_q_rad": [0.0] * 7,
        "arm_current_q_rad": [0.0] * 7,
        "tcp_anchor_world": np.eye(4).tolist(),
        "tcp_current_world": np.eye(4).tolist(),
        "finger_joint_positions_m": [0.04, 0.04],
        "finger_force_left_n": 0.0,
        "finger_force_right_n": 0.0,
    }
    evidence = _stage2a_safety_evidence_record(row, controller=controller)
    _verify_stage2a_safety_record(row, evidence, controller=controller)
    row["arm_current_q_rad"][0] = 0.1
    evidence["motion_row_sha256"] = canonical_sha256(row)
    primitive = dict(evidence)
    primitive.pop("evidence_sha256")
    evidence["evidence_sha256"] = canonical_sha256(primitive)
    with pytest.raises(ValueError, match="raw witness"):
        _verify_stage2a_safety_record(row, evidence, controller=controller)


def test_pre_command_authorization_requires_current_controller_transition() -> None:
    episode_id = "stage2a-pre-command"
    controller, request = _requested_controller(episode_id)
    runtime = Stage2AActionHistoryRuntime(episode_id)
    reset, _ = runtime.invalidate_for_active_request(request)
    controller.begin(reset)
    safety = ActiveFrontSafetyEvidence()
    controller.advance(ActiveFrontSignal.CAMERA_LEASE_ACQUIRED, safety=safety)
    valid_state = controller.state
    assert valid_state is ActiveFrontReobserveState.SELECT_FROZEN_PRIMITIVE
    row = {
        "frame_index": 1,
        "camera_motion_state": "move_to_view",
        "viewpoint_primitive_id": ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
        "request_id": request.request_id,
        "camera_command_sequence_id": request.camera_command_sequence_id,
    }
    record = {
        "version": "e018-p1-stage2a-camera-command-authorization/v1",
        "frame_index": 1,
        "camera_motion_state": "move_to_view",
        "viewpoint_primitive_id": ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
        "controller_state_before_command": "move_to_view",
        "external_camera_owner": "active_reobserve",
        "camera_lease_held": True,
        "active_window_open": True,
        "request_id": request.request_id,
        "camera_command_sequence_id": request.camera_command_sequence_id,
        "selected_primitive_id": ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
        "authorized": True,
    }
    record["authorization_sha256"] = canonical_sha256(record)
    with pytest.raises(ValueError, match="authorization"):
        _verify_stage2a_camera_authorization(row, record, controller=controller)
    controller.advance(
        ActiveFrontSignal.FROZEN_PRIMITIVE_SELECTED,
        safety=safety,
        selected_primitive_id=ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
    )
    _verify_stage2a_camera_authorization(row, record, controller=controller)


def test_complete_controller_chain_and_cross_receipt_replay() -> None:
    episode_id = "stage2a-complete-controller"
    controller, request = _requested_controller(episode_id)
    runtime = Stage2AActionHistoryRuntime(episode_id)
    reset, _ = runtime.invalidate_for_active_request(request)
    controller.begin(reset)
    safety = ActiveFrontSafetyEvidence()
    controller.advance(ActiveFrontSignal.CAMERA_LEASE_ACQUIRED, safety=safety)
    controller.advance(
        ActiveFrontSignal.FROZEN_PRIMITIVE_SELECTED,
        safety=safety,
        selected_primitive_id=ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
    )
    controller.advance(ActiveFrontSignal.MOVE_COMPLETE, safety=safety)
    controller.advance(ActiveFrontSignal.SETTLE_COMPLETE, safety=safety)
    controller.advance(ActiveFrontSignal.COLLECTION_COMPLETE, safety=safety)
    candidate = Stage2MemoryCandidateReceipt(
        request_id=request.request_id,
        candidate_digest="b" * 64,
        commit_eligible=True,
        rejection_reasons=(),
        memory_write_deferred=True,
        live_memory_write_executed=False,
        provider_forward_count=3,
        collect_frame_digests=("1" * 64, "2" * 64, "3" * 64),
    )
    controller.advance(
        ActiveFrontSignal.SHADOW_CANDIDATE_STAGED,
        safety=safety,
        shadow_candidate_receipt=candidate,
    )
    controller.advance(ActiveFrontSignal.RETURN_HOME_COMPLETE, safety=safety)
    spec, identity, evidence, rows = _window_packet(episode_id)
    verify_stage2a_observation_v2_window_identity(
        identity,
        spec=spec,
        home_evidence=evidence,
        home_motion_rows=rows,
        expected_episode_id=episode_id,
        expected_episode_generation=1,
    )
    for item in evidence:
        controller.accept_home_v2_barrier_frame(
            HomeV2BarrierFrame(
                observation_sequence_id=item["observation_sequence_id"],
                camera_at_home=True,
                fresh_observation_v2_frame=True,
                captured_after_return=True,
                contains_alternate_or_motion_rgb=False,
            )
        )
    controller.advance(
        ActiveFrontSignal.SOURCE_INVARIANTS_VERIFIED,
        safety=safety,
        source_phase=PhaseId.ACQUIRE_TRACK,
        source_invariants_passed=True,
    )
    resume, _ = runtime.generate_fresh_shadow_replan(
        request,
        home_evidence=evidence,
        observation_v2_window_identity=identity,
        memory_state=_valid_state(episode_id, timestamp_s=4.65),
        source_phase=PhaseId.ACQUIRE_TRACK,
    )
    receipt = controller.complete_stage2_memory_write(
        resume,
        memory_write_count=1,
        provider_forward_count=4,
    )
    public = {**receipt.as_dict(), "audit_digest": receipt.audit_digest}
    _verify_stage2a_controller_receipt(public, receipt)
    assert receipt.state_trace == (
        "idle",
        "requested",
        "acquire_camera_lease_and_hold_arm",
        "select_frozen_primitive",
        "move_to_view",
        "settle_at_view",
        "collect",
        "stage_candidate",
        "return_home",
        "verify_home_and_arm_hold",
        "recheck_source_invariants",
        "commit_and_resume",
        "complete_stage2_memory_write",
    )
    tampered = copy.deepcopy(public)
    tampered["state_trace"].remove("return_home")
    with pytest.raises(ValueError, match="controller receipt"):
        _verify_stage2a_controller_receipt(tampered, receipt)


def test_injected_failure_preserves_bounded_recoverable_progress(
    tmp_path: Path,
) -> None:
    loaded = load_e018_p1_stage2a_config(STAGE2_CONFIG)
    source: dict[str, object] = {
        "git_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
    }
    source["identity_sha256"] = canonical_sha256(source)
    episode_id = _stage2a_episode_id(76903)
    progress = Stage2AExecutionProgress(
        current_seed=76903,
        episode_id=episode_id,
        request_id=f"{episode_id}-active-front-01",
        current_frame_index=46,
        last_processed_frame_index=45,
        last_authorized_frame_index=46,
        controller_state=ActiveFrontReobserveState.COLLECT.value,
        orchestrator_state="collecting",
        provider_forward_count=2,
        memory_write_count=0,
    )
    try:
        raise RuntimeError("injected-stage2a-failure:" + "x" * 10_000)
    except RuntimeError as error:
        evidence = _record_stage2a_failure_evidence(
            output_root=tmp_path,
            error=error,
            progress=progress,
            stage2_config=loaded,
            source_identity=source,
        )

    stored = json.loads((tmp_path / "FAILURE.json").read_text(encoding="utf-8"))
    verified = verify_stage2a_failure_evidence(stored)
    assert stored == evidence
    assert verified["progress"] == progress.as_dict()
    assert verified["error_type"] == "RuntimeError"
    assert stored["traceback"]["truncated"] is True
    assert len(stored["traceback"]["tail"]) == 8192
    assert len(stored["error"]) == 1024
    assert all(
        stored[name] == 0
        for name in (
            "fresh_test_reads",
            "runtime_object_gt_reads",
            "goal_gt_reads",
            "offline_label_reads",
        )
    )


def test_exact_artifact_tree_rejects_extra_symlink_and_hardlink(tmp_path: Path) -> None:
    for name in _STAGE2A_COMPLETE_ARTIFACT_FILES:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    _verify_stage2a_exact_file_tree(tmp_path)
    extra = tmp_path / "extra.json"
    extra.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact file tree"):
        _verify_stage2a_exact_file_tree(tmp_path)
    extra.unlink()
    target = tmp_path / "RUN_STARTED.json"
    target.unlink()
    os.symlink(tmp_path / "execution_receipt.json", target)
    with pytest.raises(RuntimeError, match="symlink"):
        _verify_stage2a_exact_file_tree(tmp_path)
    target.unlink()
    os.link(tmp_path / "execution_receipt.json", target)
    with pytest.raises(RuntimeError, match="hardlink"):
        _verify_stage2a_exact_file_tree(tmp_path)


def test_stage2a_cli_has_no_arbitrary_seed_and_checks_go_before_reads() -> None:
    from robot_vla.cli.run_e018_p1_stage2a import build_parser

    parser = build_parser()
    common = [
        "--config",
        "missing-config.json",
        "--qualification-config",
        "missing-qualification.json",
        "--qualification-public-execution-root",
        "missing-execution",
        "--qualification-result-root",
        "missing-result",
        "--g0c-config",
        "missing-g0c.json",
        "--data-config",
        "missing-data.json",
    ]
    args = parser.parse_args(
        [
            "smoke",
            *common,
            "--stats-root",
            "missing-stats",
            "--selected-checkpoint",
            "missing.pt",
            "--repository-root",
            "missing-repo",
            "--output",
            "missing-output",
            "--expected-config-raw-sha256",
            "a" * 64,
            "--expected-config-canonical-sha256",
            "b" * 64,
            "--expected-source-git-commit",
            "deadbeef",
            "--expected-source-identity-sha256",
            "c" * 64,
            "--integration-smoke-go",
            STAGE2A_INTEGRATION_SMOKE_GO,
        ]
    )
    assert not hasattr(args, "seed")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "smoke",
                *common,
                "--seed",
                "77001",
                "--stats-root",
                "missing-stats",
                "--selected-checkpoint",
                "missing.pt",
                "--repository-root",
                "missing-repo",
                "--output",
                "missing-output",
                "--expected-config-raw-sha256",
                "a" * 64,
                "--expected-config-canonical-sha256",
                "b" * 64,
                "--expected-source-git-commit",
                "deadbeef",
                "--expected-source-identity-sha256",
                "c" * 64,
                "--integration-smoke-go",
                STAGE2A_INTEGRATION_SMOKE_GO,
            ]
        )
    with pytest.raises(PermissionError, match="exact GO"):
        run_e018_p1_stage2a_integration_smoke(
            stage2_config_path="missing-config.json",
            qualification_config_path="missing-qualification.json",
            qualification_public_execution_root="missing-execution",
            qualification_result_root="missing-result",
            g0c_config_path="missing-g0c.json",
            data_config_path="missing-data.json",
            stats_root="missing-stats",
            selected_checkpoint_path="missing.pt",
            repository_root="missing-repo",
            output_root="missing-output",
            expected_stage2_config_raw_sha256="a" * 64,
            expected_stage2_config_canonical_sha256="b" * 64,
            expected_source_git_commit="deadbeef",
            expected_source_identity_sha256="c" * 64,
            integration_smoke_go="DENY",
        )
