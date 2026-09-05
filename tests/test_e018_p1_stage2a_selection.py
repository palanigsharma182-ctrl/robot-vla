from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from robot_vla.cli.run_e018_p1_stage2a_selection import build_parser
from robot_vla.executive.contracts import PhaseId
from robot_vla.precision import e018_p1_stage2a_selection_runtime as selection_runtime
from robot_vla.precision.active_front_camera import ExternalCameraMotionState
from robot_vla.precision.active_front_memory_provider import (
    ACTIVE_FRONT_HOME_BASE_FROM_EXTERNAL_CAMERA_CV,
    ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
    ActiveFrontScoreComponents,
    ActiveFrontStage2Config,
    ActiveFrontStage2FrameEvidence,
    PassiveBaselineEvidence,
    PassiveHomeScoreEvidence,
    d049_home_baseline_provider_identity,
    d049_primary_provider_identity,
)
from robot_vla.precision.active_front_reobserve import (
    ActiveFrontReobserveRequest,
    ActiveFrontSafetyEvidence,
    ActiveFrontTriggerReason,
    HomeV2BarrierFrame,
)
from robot_vla.precision.calibrated_front_provider import canonical_sha256
from robot_vla.precision.e018_p1_stage2a import (
    E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID,
    E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXPERIMENT_ID,
    STAGE2A_SELECTION_PREFLIGHT_SEED,
    STAGE2A_SELECTION_SEEDS,
    Stage2AExecutionProgress,
    Stage2ARouteTransaction,
)
from robot_vla.precision.e018_p1_stage2a_selection import (
    STAGE2A_SELECTION_GAINS,
    STAGE2A_SELECTION_GO,
    STAGE2A_SELECTION_PREFLIGHT_GO,
    CapturedSelectionRoute,
    GainBranchOutcome,
    Stage2ASelectionJournal,
    Stage2ASelectionPreflightJournal,
    _begin_gain_replay,
    load_e018_p1_stage2a_selection_config,
    replay_all_gain_branches,
    score_gain_branches,
)
from robot_vla.precision.object_memory import ObjectMemorySafetyContext
from robot_vla.precision.object_observability import ObjectObservabilityLabel

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SELECTION_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "e018_p1_stage2a_information_gain_selection_development_v1.json"
)


def _components(score: float) -> ActiveFrontScoreComponents:
    return ActiveFrontScoreComponents(
        object_visibility_probability=score,
        projection_validity_probability=1.0,
        object_mask_probability=0.98,
        goal_mask_probability=0.01,
        object_normalized_entropy=0.2,
        object_sigma_xy_px=(0.1, 0.1),
    )


def _memory_safety() -> ObjectMemorySafetyContext:
    return ObjectMemorySafetyContext(
        pregrasp_window_open=True,
        gripper_open=True,
        controller_tracking_valid=True,
        object_contact_detected=False,
        gripper_close_commanded=False,
        grasp_candidate=False,
        grasp_verified=False,
        object_maybe_moved=False,
    )


def _captured_route(*, raw_gain: float = 0.03) -> CapturedSelectionRoute:
    seed = 77001
    episode_id = f"e018-p1-stage2a-selection-development-seed-{seed}"
    request = ActiveFrontReobserveRequest(
        episode_id=episode_id,
        episode_generation=1,
        request_id=f"{episode_id}-active-front-01",
        source_phase=PhaseId.ACQUIRE_TRACK,
        resume_phase=PhaseId.ACQUIRE_TRACK,
        trigger_tick=3,
        trigger_timestamp_s=0.0,
        trigger_reason=(
            ActiveFrontTriggerReason.NO_QUALIFIED_WRIST_PROVIDER_IN_PARENT
        ),
        attempt_index=1,
        selected_primitive_id=ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
        camera_command_sequence_id=f"{episode_id}-camera-00",
    )
    home_score = 0.67
    home = PassiveHomeScoreEvidence(
        episode_id=episode_id,
        episode_generation=1,
        request_id=request.request_id,
        observation_sequence_id=f"{episode_id}-route-frame-00",
        model_input_digest="1" * 64,
        provider_output_digest="2" * 64,
        provider_identity=d049_home_baseline_provider_identity(),
        viewpoint_primitive_id="HOME__CENTER",
        camera_motion_state=ExternalCameraMotionState.HOME_ANCHOR,
        settled=True,
        score_components=_components(home_score),
        stored_write_score=home_score,
        geometry_valid=True,
        control_timestamp_s=0.0,
        rgb_timestamp_s=0.0,
        camera_pose_timestamp_s=0.0,
        tcp_pose_timestamp_s=0.0,
        base_from_external_camera_cv=np.asarray(
            ACTIVE_FRONT_HOME_BASE_FROM_EXTERNAL_CAMERA_CV,
            dtype=np.float64,
        ),
    )
    baseline = PassiveBaselineEvidence(
        episode_id=episode_id,
        episode_generation=1,
        request_id=request.request_id,
        timestamp_s=0.0,
        wrist_object_measurement_usable=False,
        wrist_evidence_identity_sha256="3" * 64,
        home_front=home,
        object_memory_navigation_state_available=False,
        object_memory_age_s=None,
        object_memory_source_identity=None,
    )
    primary_score = home_score + raw_gain
    primary: list[ActiveFrontStage2FrameEvidence] = []
    for index, timestamp_s in enumerate((0.10, 0.15, 0.20)):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = (0.3, -0.16, 0.48)
        primary.append(
            ActiveFrontStage2FrameEvidence(
                episode_id=episode_id,
                episode_generation=1,
                request_id=request.request_id,
                source_phase=PhaseId.ACQUIRE_TRACK,
                observation_sequence_id=f"{episode_id}-route-frame-{45 + index:02d}",
                model_input_digest=f"{4 + index:x}" * 64,
                provider_output_digest=f"{8 + index:x}" * 64,
                provider_identity=d049_primary_provider_identity(),
                camera_motion_state=ExternalCameraMotionState.COLLECT,
                settled=True,
                control_timestamp_s=timestamp_s,
                rgb_timestamp_s=timestamp_s,
                camera_pose_timestamp_s=timestamp_s,
                tcp_pose_timestamp_s=timestamp_s,
                base_from_external_camera_cv=transform,
                position_base_m=(0.400 + index * 0.001, 0.1, 0.02),
                covariance_base_m2=np.eye(3, dtype=np.float64) * 1e-6,
                measurement_confidence=primary_score,
                write_score=primary_score,
                score_components=_components(primary_score),
                projection_valid=True,
                in_fov=True,
                observable=True,
                geometry_valid=True,
                structurally_eligible=True,
                deployable_free_static_safe=True,
            )
        )
    home_frames = tuple(
        HomeV2BarrierFrame(
            observation_sequence_id=f"{episode_id}-home-v2-{88 + index:02d}",
            camera_at_home=True,
            fresh_observation_v2_frame=True,
            captured_after_return=True,
            contains_alternate_or_motion_rgb=False,
        )
        for index in range(4)
    )
    home_timestamps = (0.25, 0.30, 0.35, 0.40)
    home_evidence = tuple(
        {
            "observation_sequence_id": frame.observation_sequence_id,
            "control_timestamp_s": timestamp,
            "evidence_sha256": f"{index + 10:x}" * 64,
        }
        for index, (frame, timestamp) in enumerate(
            zip(home_frames, home_timestamps, strict=True)
        )
    )
    window = {
        "episode_id": episode_id,
        "episode_generation": 1,
        "observation_sequence_ids": [
            frame.observation_sequence_id for frame in home_frames
        ],
        "home_evidence_digests": [
            value["evidence_sha256"] for value in home_evidence
        ],
    }
    window["window_sha256"] = canonical_sha256(window)
    provisional = CapturedSelectionRoute(
        seed=seed,
        episode_id=episode_id,
        request=request,
        passive_baseline=baseline,
        primary_frames=tuple(primary),
        collect_memory_safety=tuple(_memory_safety() for _ in range(3)),
        home_frames=home_frames,
        home_timestamps_s=home_timestamps,
        home_memory_safety=tuple(_memory_safety() for _ in range(4)),
        home_active_safety=tuple(ActiveFrontSafetyEvidence() for _ in range(4)),
        final_active_safety=ActiveFrontSafetyEvidence(),
        final_memory_safety=_memory_safety(),
        source_recheck_evidence_digest="e" * 64,
        source_recheck_timestamp_s=0.401,
        return_home_timestamp_s=0.201,
        home_evidence=home_evidence,
        observation_v2_window_identity=window,
        route_protocol_safety_valid=True,
        physical_route_count=1,
        captured_provider_forward_count=4,
        raw_candidate_digest_at_gain_0_02="f" * 64,
        raw_candidate_commit_eligible_at_gain_0_02=True,
        raw_candidate_rejection_reasons_at_gain_0_02=(),
    )
    _, orchestrator, _ = _begin_gain_replay(provisional, 0.02)
    candidate = orchestrator.pending_candidate
    assert candidate is not None
    return replace(
        provisional,
        raw_candidate_digest_at_gain_0_02=candidate.digest,
        raw_candidate_commit_eligible_at_gain_0_02=candidate.commit_eligible,
        raw_candidate_rejection_reasons_at_gain_0_02=candidate.rejection_reasons,
    )


def _private_capture(*, observable: bool = True) -> dict[str, object]:
    observability = ObjectObservabilityLabel(
        object_exists=True,
        projection_valid=True,
        in_fov=True,
        observable=observable,
        legacy_visible=True,
        center_inside_object_mask=observable,
        center_inside_goal_mask=False,
        local_object_visible_fraction=1.0 if observable else 0.0,
        object_mask_area_fraction=1.0 / (128 * 128),
        occlusion_type="observable" if observable else "other_occlusion_or_background",
    )
    return {
        "gt_object_exists": True,
        "gt_observable": observable,
        "gt_object_position_base_m": [0.402, 0.1, 0.02],
        "gt_object_projection_valid": True,
        "gt_object_projected_normalized_uv": [0.5, 0.5],
        "gt_object_mask_sha256": "a" * 64,
        "gt_object_visible_pixel_count": 1,
        "gt_object_observability": observability.to_dict(),
        "is_grasped": False,
        "robot_object_contact_force_n": 0.0,
        "goal_gt_read_count": 0,
        "test_data_read": False,
        "object_linear_speed_m_s": 0.01,
        "object_angular_speed_rad_s": 0.5,
        "object_motion_event": False,
    }


def _private_label(
    label_index: int,
    *,
    observable: bool,
) -> dict[str, object]:
    seed = STAGE2A_SELECTION_SEEDS[label_index // 3]
    frame = (45, 46, 47)[label_index % 3]
    value = {
        **_private_capture(observable=observable),
        "version": "e018-p1-stage2a-min-information-gain-selection-execution/v1",
        "label_index": label_index,
        "prediction_row_index": (label_index // 3) * 4 + 1 + label_index % 3,
        "seed": seed,
        "route_frame_index": frame,
        "rgb_sha256": "b" * 64,
        "actual_pose_sha256": "c" * 64,
        "provider_output_digest": "d" * 64,
        "prediction_commit_receipt_sha256": "e" * 64,
        "transaction_identity_sha256": "f" * 64,
        "motion_predicate_version": "pick-and-place-predicates/v1",
        "motion_linear_threshold_m_s": 0.01,
        "motion_angular_threshold_rad_s": 0.5,
        "contact_threshold_n": 0.01,
        "privileged_captured_at_unix_ns": label_index + 1,
    }
    value["label_sha256"] = canonical_sha256(value)
    return value


def _branches(
    *,
    commit_seed_gain: tuple[int, float] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in STAGE2A_SELECTION_SEEDS:
        route_digest = canonical_sha256({"seed": seed, "route": "shared"})
        for gain in STAGE2A_SELECTION_GAINS:
            commit = commit_seed_gain == (seed, gain)
            rows.append(
                GainBranchOutcome(
                    seed=seed,
                    gain=gain,
                    route_evidence_digest=route_digest,
                    route_protocol_safety_valid=True,
                    candidate_commit_eligible=commit,
                    memory_commit_count=int(commit),
                    navigation_state_available=commit,
                    fresh_shadow_action_generation_count=int(commit),
                    committed_position_base_m=(0.402, 0.1, 0.02) if commit else None,
                    provider_forward_count=0,
                ).to_dict()
            )
    return rows


def _different_valid_producer_process() -> dict[str, object]:
    value = selection_runtime._new_process_identity("pass-a-producer")
    value["pid"] += 1_000_000
    instance = {
        key: value[key]
        for key in (
            "version",
            "boot_id_sha256",
            "pid",
            "proc_start_time_ticks",
        )
    }
    value["process_instance_sha256"] = canonical_sha256(instance)
    value.pop("process_token_sha256")
    value["process_token_sha256"] = canonical_sha256(value)
    return value


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(selection_runtime._json_bytes(value))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(selection_runtime._jsonl_line_bytes(row) for row in rows)
    )


def _synthetic_pass_b_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    producer: dict[str, object] | None = None,
) -> dict[str, object]:
    loaded = load_e018_p1_stage2a_selection_config(SELECTION_CONFIG)
    source_identity = "1" * 64
    parent_identity = "2" * 64
    transaction_identity = "3" * 64
    producer = producer or _different_valid_producer_process()
    private_role = selection_runtime._artifact_role_identity_sha256(
        role="private_labels",
        config_canonical_sha256=loaded.canonical_sha256,
        source_identity_sha256=source_identity,
        parent_verification_sha256=parent_identity,
    )
    public_result: dict[str, object] = {
        "verified": True,
        "source_git_commit": "4" * 40,
        "source_identity_sha256": source_identity,
        "parent_verification_sha256": parent_identity,
        "transaction_identity_sha256": transaction_identity,
        "public_artifact_role_identity_sha256": (
            selection_runtime._artifact_role_identity_sha256(
                role="public_execution",
                config_canonical_sha256=loaded.canonical_sha256,
                source_identity_sha256=source_identity,
                parent_verification_sha256=parent_identity,
            )
        ),
        "private_artifact_role_identity_sha256": private_role,
        "producer_process_identity": producer,
        "public_completion_marker_sha256": "5" * 64,
        "verification_sha256": "6" * 64,
    }
    monkeypatch.setattr(
        selection_runtime,
        "verify_e018_p1_stage2a_selection_public",
        lambda **_: copy.deepcopy(public_result),
    )
    public_root = tmp_path / "public_execution"
    private_root = tmp_path / "private_labels"
    result_root = tmp_path / "result"
    public_root.mkdir()
    (public_root / "prediction_commits").mkdir()
    (private_root / "label_commits").mkdir(parents=True)

    provider_rows: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []
    for prediction_index in range(100):
        provider_digest = hashlib.sha256(
            f"provider-{prediction_index}".encode()
        ).hexdigest()
        provider_rows.append({"provider_output_digest": provider_digest})
        receipt = {
            "provider_output_digest": provider_digest,
            "prediction_fsync_completed_at_unix_ns": 1_000 + prediction_index,
        }
        receipt["commit_receipt_sha256"] = canonical_sha256(receipt)
        receipts.append(receipt)
        _write_json(
            public_root
            / "prediction_commits"
            / f"{prediction_index:06d}.commit.json",
            receipt,
        )
    _write_jsonl(public_root / "provider_output_ledger.jsonl", provider_rows)

    pose = np.eye(4, dtype=np.float64)
    pose_digest = selection_runtime._array_sha256(pose)
    rgb_digest = "7" * 64
    camera_rows = [
        {
            "rgb_sha256": rgb_digest,
            "actual_base_from_external_camera_cv": pose.tolist(),
        }
        for _ in range(2300)
    ]
    _write_jsonl(public_root / "camera_pose_ledger.jsonl", camera_rows)
    _write_jsonl(public_root / "gain_branch_ledger.jsonl", _branches())

    inventory_rows: list[dict[str, object]] = []
    for label_index in range(75):
        prediction_index = (label_index // 3) * 4 + 1 + label_index % 3
        label = _private_label(label_index, observable=False)
        label.update(
            {
                "rgb_sha256": rgb_digest,
                "actual_pose_sha256": pose_digest,
                "provider_output_digest": provider_rows[prediction_index][
                    "provider_output_digest"
                ],
                "prediction_commit_receipt_sha256": receipts[
                    prediction_index
                ]["commit_receipt_sha256"],
                "transaction_identity_sha256": transaction_identity,
                "privileged_captured_at_unix_ns": 2_000 + label_index,
            }
        )
        label.pop("label_sha256")
        label["label_sha256"] = canonical_sha256(label)
        label_path = (
            private_root / "label_commits" / f"{label_index:06d}.json"
        )
        raw = selection_runtime._json_bytes(label)
        label_path.write_bytes(raw)
        inventory = {
            "label_index": label_index,
            "prediction_row_index": prediction_index,
            "seed": STAGE2A_SELECTION_SEEDS[label_index // 3],
            "route_frame_index": (45, 46, 47)[label_index % 3],
            "path": f"label_commits/{label_index:06d}.json",
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "scoring_primitive_sha256": (
                selection_runtime._scoring_primitive_sha256(label)
            ),
        }
        inventory["row_sha256"] = canonical_sha256(inventory)
        inventory_rows.append(inventory)
    inventory_value = {
        "version": (
            "e018-p1-stage2a-min-information-gain-selection-execution/v1"
        ),
        "label_count": 75,
        "rows": inventory_rows,
    }
    inventory_value["inventory_sha256"] = canonical_sha256(inventory_value)
    _write_json(public_root / "private_label_inventory.json", inventory_value)
    capture_state = {
        "version": (
            "e018-p1-stage2a-min-information-gain-selection-execution/v1"
        ),
        "status": "capture-complete-write-only-not-opened",
        "transaction_identity_sha256": transaction_identity,
        "prediction_commit_count": 100,
        "privileged_access_started_count": 75,
        "privileged_capture_count": 75,
        "rerun_under_same_identity_allowed": False,
    }
    capture_state["state_sha256"] = canonical_sha256(capture_state)
    _write_json(private_root / "capture_state.json", capture_state)
    return {
        "public": public_root,
        "private": private_root,
        "result": result_root,
        "public_result": public_result,
    }


def _score_synthetic(artifact: dict[str, object], result_root: Path | None = None):
    return selection_runtime.run_e018_p1_stage2a_selection_score_private(
        selection_config_path=SELECTION_CONFIG,
        stage2a_config_path=Path("unused-stage2a.json"),
        qualification_config_path=Path("unused-qualification.json"),
        public_root=artifact["public"],
        private_root=artifact["private"],
        result_root=result_root or artifact["result"],
        expected_source_git_commit="4" * 40,
        expected_source_identity_sha256="1" * 64,
        exact_go_token=STAGE2A_SELECTION_GO,
    )


def _verify_synthetic_result(
    artifact: dict[str, object],
    *,
    result_root: Path | None = None,
):
    return selection_runtime.verify_e018_p1_stage2a_selection_result(
        selection_config_path=SELECTION_CONFIG,
        stage2a_config_path=Path("unused-stage2a.json"),
        qualification_config_path=Path("unused-qualification.json"),
        public_root=artifact["public"],
        result_root=result_root or artifact["result"],
        expected_source_git_commit="4" * 40,
        expected_source_identity_sha256="1" * 64,
    )


def _rewrite_private_label_and_inventory(
    artifact: dict[str, object],
    *,
    label_index: int,
    changes: dict[str, object],
) -> None:
    private_root = Path(artifact["private"])
    public_root = Path(artifact["public"])
    label_path = private_root / "label_commits" / f"{label_index:06d}.json"
    label = json.loads(label_path.read_text())
    label.update(changes)
    label.pop("label_sha256")
    label["label_sha256"] = canonical_sha256(label)
    raw = selection_runtime._json_bytes(label)
    label_path.write_bytes(raw)

    inventory_path = public_root / "private_label_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    row = inventory["rows"][label_index]
    row.update(
        {
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "scoring_primitive_sha256": (
                selection_runtime._scoring_primitive_sha256(label)
            ),
        }
    )
    row.pop("row_sha256")
    row["row_sha256"] = canonical_sha256(row)
    inventory.pop("inventory_sha256")
    inventory["inventory_sha256"] = canonical_sha256(inventory)
    _write_json(inventory_path, inventory)


def _synthetic_route_summary() -> dict[str, object]:
    diagnostics = {
        "alternate_rgb_mean_abs_difference": 0.0,
        "return_home_rgb_mean_abs_difference": 0.0,
        "alternate_displacement_m": 0.0,
        "requested_orientation_offset_rad": 0.0,
        "actual_orientation_offset_rad": 0.0,
        "alternate_target_orientation_error_rad": 0.0,
        "object_visible_pixels_collect_min": None,
        "goal_visible_pixels_collect_min": None,
    }
    gates: dict[str, object] = {}
    for name, fields in selection_runtime._SELECTION_ROUTE_GATE_ACTUAL_KEYS.items():
        if fields is None:
            actual: object = 0.0
        else:
            actual = {}
            for field in fields:
                if field == "orientation_id":
                    actual[field] = "PITCH_UP"
                elif field == "final_settled":
                    actual[field] = True
                elif field in {
                    "move_ticks_each_leg",
                    "unique_actual_positions",
                    "collect_frames",
                    "eligible_collect_frames",
                }:
                    actual[field] = 0
                else:
                    actual[field] = 0.0
        gates[name] = {"actual": actual, "required": "frozen", "passed": True}
    return {
        "version": "e018-p1-stage2a-primary-memory-integration-smoke/v1",
        "episode_id": "e018-p1-stage2a-selection-development-seed-77001",
        "seed": 77001,
        "alternate_viewpoint_id": "LEFT_LOW__PITCH_UP",
        "alternate_orientation_id": "PITCH_UP",
        "yaw_offset_rad": 0.0,
        "pitch_offset_rad": 0.0,
        "roll_offset_rad": 0.0,
        "status": "passed",
        "passed": True,
        "frame_count": 92,
        "control_hz": 20,
        "motion_ticks_each_leg": 40,
        "route_simulated_duration_s": 4.55,
        "gates": gates,
        "diagnostics": diagnostics,
        "test_split_status": "prohibited-unread",
        "provider_forward_count": 4,
        "memory_write_count": 0,
        "formal_claim_allowed": False,
        "classification": (
            "formal-development-selection-capture-only-no-test-no-actuation/v1"
        ),
        "offline_segmentation_diagnostics": False,
        "runtime_object_gt_reads": 0,
        "goal_gt_reads": 0,
        "fresh_test_reads": 0,
    }


def test_selection_config_is_frozen_no_test_no_actuation() -> None:
    loaded = load_e018_p1_stage2a_selection_config(SELECTION_CONFIG)
    assert loaded.payload["experiment"]["id"] == (
        E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID
    )
    assert loaded.payload["gain_selection"]["minimum_common_denominator_routes"] == 1
    assert loaded.payload["parents"]["replication_state"] == "REPLICATED"
    assert loaded.payload["permissions"]["fresh_test_reads"] == 0
    assert loaded.payload["permissions"]["arm_tcp_actuation"] == 0


def test_selection_transaction_entry_rejects_unfrozen_identity_and_seed() -> None:
    kwargs = {
        "provider": None,
        "stage2_config": None,
        "qualification_config": {},
        "data_config": {},
        "base_env": None,
        "spec": None,
        "proprio_normalizer": None,
        "finger_force_normalizer": None,
        "execution_progress": None,
    }
    with pytest.raises(PermissionError, match="experiment identity"):
        Stage2ARouteTransaction.for_information_gain_selection_capture(
            experiment_identity="wrong",
            seed=77001,
            **kwargs,
        )
    with pytest.raises(ValueError, match="77001..77025"):
        Stage2ARouteTransaction.for_information_gain_selection_capture(
            experiment_identity=E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID,
            seed=77026,
            **kwargs,
        )
    with pytest.raises(ValueError, match="76901..76910"):
        Stage2ARouteTransaction(seed=77001, **kwargs)


def test_raw_gain_003_replays_one_route_as_accept_reject_reject() -> None:
    captured = _captured_route(raw_gain=0.03)
    forward_count_before = captured.captured_provider_forward_count
    branches = replay_all_gain_branches(captured)

    assert [value.gain for value in branches] == [0.02, 0.05, 0.10]
    assert [value.memory_commit_count for value in branches] == [1, 0, 0]
    assert [value.fresh_shadow_action_generation_count for value in branches] == [1, 0, 0]
    assert [value.candidate_commit_eligible for value in branches] == [True, False, False]
    assert all(value.provider_forward_count == 0 for value in branches)
    assert captured.physical_route_count == 1
    assert captured.captured_provider_forward_count == forward_count_before == 4

    reversed_branches = replay_all_gain_branches(
        captured, gain_order=(0.10, 0.05, 0.02)
    )
    assert [value.to_dict() for value in reversed_branches] == [
        value.to_dict() for value in branches
    ]


def test_captured_route_public_round_trip_preserves_replay_digest() -> None:
    captured = _captured_route(raw_gain=0.03)

    restored = CapturedSelectionRoute.from_public_dict(captured.to_public_dict())

    assert restored.route_evidence_digest == captured.route_evidence_digest
    assert [value.to_dict() for value in replay_all_gain_branches(restored)] == [
        value.to_dict() for value in replay_all_gain_branches(captured)
    ]


def test_resigned_raw_candidate_digest_tamper_is_rejected_by_replay() -> None:
    captured = _captured_route(raw_gain=0.03)
    tampered = replace(
        captured,
        raw_candidate_digest_at_gain_0_02="0" * 64,
    )
    resigned = CapturedSelectionRoute.from_public_dict(tampered.to_public_dict())

    with pytest.raises(RuntimeError, match="capture-time raw identity"):
        replay_all_gain_branches(resigned)


def test_route_digest_binds_collect_safety_input() -> None:
    captured = _captured_route(raw_gain=0.03)
    changed_safety = replace(
        captured.collect_memory_safety[0],
        object_contact_detected=True,
    )
    changed = replace(
        captured,
        collect_memory_safety=(
            changed_safety,
            *captured.collect_memory_safety[1:],
        ),
    )

    assert changed.route_evidence_digest != captured.route_evidence_digest


def test_private_motion_capture_uses_strict_frozen_thresholds(monkeypatch) -> None:
    class Cube:
        linear_velocity = np.asarray([[0.01, 0.0, 0.0]])
        angular_velocity = np.asarray([[0.5, 0.0, 0.0]])

    class Env:
        cube = Cube()

    monkeypatch.setattr(
        selection_runtime,
        "capture_qualification_object_label",
        lambda **_: _private_capture(),
    )
    label = selection_runtime.capture_selection_private_label(
        observation={},
        base_env=Env(),
        prediction={},
        data_config={},
    )
    assert label["object_linear_speed_m_s"] == 0.01
    assert label["object_angular_speed_rad_s"] == 0.5
    assert label["object_motion_event"] is False

    Env.cube.linear_velocity = np.asarray(
        [[float(np.nextafter(0.01, np.inf)), 0.0, 0.0]]
    )
    moved = selection_runtime.capture_selection_private_label(
        observation={},
        base_env=Env(),
        prediction={},
        data_config={},
    )
    assert moved["object_motion_event"] is True


def test_selection_contact_and_gain_thresholds_are_exact() -> None:
    class Controller:
        active_window_open = True

    row = {
        "arm_joint_max_drift_rad": 0.0,
        "tcp_position_drift_m": 0.0,
        "tcp_orientation_drift_rad": 0.0,
        "minimum_finger_joint_position_m": 0.04,
        "finger_object_contact_force_n": 0.01,
    }
    exact = selection_runtime._stage2a._stage2a_active_safety(
        row,
        controller=Controller(),
        contact_comparison_tolerance_n=0.0,
    )
    row["finger_object_contact_force_n"] = float(
        np.nextafter(0.01, np.inf)
    )
    above = selection_runtime._stage2a._stage2a_active_safety(
        row,
        controller=Controller(),
        contact_comparison_tolerance_n=0.0,
    )
    assert exact.contact_absent is True
    assert above.contact_absent is False

    for threshold in STAGE2A_SELECTION_GAINS:
        config = ActiveFrontStage2Config.development(
            min_information_gain=threshold,
            information_gain_comparison_tolerance=0.0,
        )
        assert config.information_gain_is_sufficient(threshold) is True
        assert config.information_gain_is_sufficient(
            float(np.nextafter(threshold, -np.inf))
        ) is False


def test_prediction_commit_prefix_chain_replays_all_100_rows(tmp_path: Path) -> None:
    journal = Stage2ASelectionJournal(
        public_root=tmp_path / "public",
        private_root=tmp_path / "private",
        config_canonical_sha256="1" * 64,
        transaction_identity_sha256="2" * 64,
    )
    for row_index in range(100):
        seed = STAGE2A_SELECTION_SEEDS[row_index // 4]
        frame = (0, 45, 46, 47)[row_index % 4]
        provider_digest = hashlib.sha256(f"provider-{row_index}".encode()).hexdigest()
        input_digest = hashlib.sha256(f"input-{row_index}".encode()).hexdigest()
        journal.commit_prediction(
            {
                "provider_output_digest": provider_digest,
                "model_input_digest": input_digest,
            },
            seed=seed,
            route_frame_index=frame,
            provider_output_digest=provider_digest,
            model_input_digest=input_digest,
        )
    rows = [
        json.loads(line)
        for line in (tmp_path / "public" / "provider_output_ledger.jsonl")
        .read_text()
        .splitlines()
    ]

    receipts = selection_runtime._verify_prediction_commit_chain(
        tmp_path / "public",
        rows,
        expected_transaction_identity_sha256="2" * 64,
    )

    assert len(receipts) == 100
    assert receipts[-1]["row_index"] == 99


def test_route_invalid_no_commit_cannot_shrink_denominator_then_select() -> None:
    captured = replace(
        _captured_route(raw_gain=0.03),
        route_protocol_safety_valid=False,
    )
    route_branches = replay_all_gain_branches(captured)
    branches = _branches()
    branches[:3] = [branch.to_dict() for branch in route_branches]
    labels = [
        _private_label(index, observable=index < 3) for index in range(75)
    ]

    _, summary = score_gain_branches(branches, labels)

    assert summary["common_denominator_count"] == 0
    assert summary["selected_gain"] is None
    assert all(row["protocol_violation_count"] == 1 for row in summary["per_gain"])
    assert all(row["eligible"] is False for row in summary["per_gain"])


def test_prediction_fsync_then_three_private_labels_return_only_metadata(
    tmp_path: Path,
) -> None:
    journal = Stage2ASelectionJournal(
        public_root=tmp_path / "public",
        private_root=tmp_path / "private",
        config_canonical_sha256="1" * 64,
        transaction_identity_sha256="2" * 64,
    )
    frames = (0, 45, 46, 47)
    for row_index, frame in enumerate(frames):
        digest = f"{row_index + 3:x}" * 64
        receipt = journal.commit_prediction(
            {
                "provider_output_digest": digest,
                "model_input_digest": f"{row_index + 7:x}" * 64,
            },
            seed=77001,
            route_frame_index=frame,
            provider_output_digest=digest,
            model_input_digest=f"{row_index + 7:x}" * 64,
        )
        if frame != 0:
            metadata = journal.capture_private_label_after_prediction(
                prediction_receipt=receipt,
                seed=77001,
                route_frame_index=frame,
                rgb_sha256="a" * 64,
                actual_pose_sha256="b" * 64,
                provider_output_digest=digest,
                privileged_getter=_private_capture,
            )
            assert "gt_observable" not in metadata
            assert set(metadata) == {
                "label_index",
                "prediction_row_index",
                "seed",
                "route_frame_index",
                "path",
                "raw_sha256",
                "size_bytes",
                "scoring_primitive_sha256",
                "row_sha256",
            }
    assert journal.prediction_count == 4
    assert journal.privileged_access_started_count == 3
    assert journal.label_count == 3
    state = json.loads((tmp_path / "private" / "capture_state.json").read_text())
    assert state["privileged_access_started_count"] == 3
    assert state["privileged_capture_count"] == 3


def test_preflight_progress_identity_and_seed_are_mutually_isolated() -> None:
    progress = Stage2AExecutionProgress()
    with pytest.raises(ValueError, match="77001..77025"):
        progress.begin_information_gain_selection(
            STAGE2A_SELECTION_PREFLIGHT_SEED,
            experiment_identity=E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID,
        )
    with pytest.raises(ValueError, match="76891"):
        progress.begin_information_gain_preflight(
            STAGE2A_SELECTION_SEEDS[0],
            experiment_identity=(
                E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXPERIMENT_ID
            ),
        )
    with pytest.raises(PermissionError, match="identity"):
        progress.begin_information_gain_selection(
            STAGE2A_SELECTION_SEEDS[0],
            experiment_identity=(
                E018_P1_STAGE2A_SELECTION_PREFLIGHT_EXPERIMENT_ID
            ),
        )
    with pytest.raises(PermissionError, match="identity"):
        progress.begin_information_gain_preflight(
            STAGE2A_SELECTION_PREFLIGHT_SEED,
            experiment_identity=E018_P1_STAGE2A_SELECTION_EXPERIMENT_ID,
        )


def test_preflight_journal_is_fixed_4_3_and_rejects_formal_freeze(
    tmp_path: Path,
) -> None:
    journal = Stage2ASelectionPreflightJournal(
        public_root=tmp_path / "public",
        private_root=tmp_path / "private",
        config_canonical_sha256="1" * 64,
        transaction_identity_sha256="2" * 64,
    )
    for row_index, frame in enumerate((0, 45, 46, 47)):
        provider_digest = f"{row_index + 3:x}" * 64
        input_digest = f"{row_index + 7:x}" * 64
        receipt = journal.commit_prediction(
            {
                "provider_output_digest": provider_digest,
                "model_input_digest": input_digest,
            },
            seed=STAGE2A_SELECTION_PREFLIGHT_SEED,
            route_frame_index=frame,
            provider_output_digest=provider_digest,
            model_input_digest=input_digest,
        )
        if frame in (45, 46, 47):
            journal.capture_private_label_after_prediction(
                prediction_receipt=receipt,
                seed=STAGE2A_SELECTION_PREFLIGHT_SEED,
                route_frame_index=frame,
                rgb_sha256="a" * 64,
                actual_pose_sha256="b" * 64,
                provider_output_digest=provider_digest,
                privileged_getter=_private_capture,
            )
    with pytest.raises(PermissionError, match="formal freeze"):
        journal.freeze()
    provider, private = journal.finalize_preflight_capture()
    assert provider["row_count"] == 4
    assert private["label_count"] == 3


def test_formal_and_preflight_go_tokens_cannot_cross_authorize() -> None:
    common = {
        "selection_config_path": SELECTION_CONFIG,
        "repository_root": REPOSITORY_ROOT,
        "expected_config_raw_sha256": "1" * 64,
        "expected_config_canonical_sha256": "2" * 64,
        "expected_source_git_commit": "3" * 40,
        "expected_source_identity_sha256": "4" * 64,
    }
    with pytest.raises(PermissionError, match="exact GO"):
        selection_runtime._assert_capture_authority(
            **common,
            exact_go_token=STAGE2A_SELECTION_PREFLIGHT_GO,
        )
    with pytest.raises(PermissionError, match="preflight token"):
        selection_runtime._assert_preflight_authority(
            **common,
            exact_preflight_token=STAGE2A_SELECTION_GO,
        )
    with pytest.raises(PermissionError, match="preflight token"):
        selection_runtime._assert_preflight_authority(
            **common,
            exact_preflight_token=(
                "E018_P1_STAGE2A_PASS_A_ONE_ROUTE_PREFLIGHT_GO_76891_V1"
            ),
        )


def test_preflight_stats_identity_binds_actual_raw_files(
    tmp_path: Path,
) -> None:
    proprio = tmp_path / "proprio_stats.json"
    force = tmp_path / "finger_force_stats.json"
    proprio.write_bytes(b"proprio\n")
    force.write_bytes(b"force\n")
    data_config = {
        "data_identity": {
            "proprio_stats_sha256": hashlib.sha256(
                proprio.read_bytes()
            ).hexdigest(),
            "finger_force_stats_sha256": hashlib.sha256(
                force.read_bytes()
            ).hexdigest(),
        }
    }
    identity = selection_runtime._build_preflight_stats_identity(
        tmp_path,
        data_config,
    )
    assert identity["stats_identity_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in identity.items()
            if key != "stats_identity_sha256"
        }
    )
    force.write_bytes(b"drifted\n")
    with pytest.raises(RuntimeError, match="冻结 data config"):
        selection_runtime._build_preflight_stats_identity(
            tmp_path,
            data_config,
        )


def test_preflight_cli_has_fixed_seed_and_no_seed_option() -> None:
    parser = build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    preflight = subparsers.choices["preflight-one-route"]
    assert "--seed" not in preflight._option_string_actions
    assert "--preflight-go" in preflight._option_string_actions


def test_privileged_getter_failure_persists_consumed_identity(tmp_path: Path) -> None:
    journal = Stage2ASelectionJournal(
        public_root=tmp_path / "public",
        private_root=tmp_path / "private",
        config_canonical_sha256="1" * 64,
        transaction_identity_sha256="2" * 64,
    )
    for row_index, frame in enumerate((0, 45)):
        digest = f"{row_index + 3:x}" * 64
        receipt = journal.commit_prediction(
            {
                "provider_output_digest": digest,
                "model_input_digest": f"{row_index + 7:x}" * 64,
            },
            seed=77001,
            route_frame_index=frame,
            provider_output_digest=digest,
            model_input_digest=f"{row_index + 7:x}" * 64,
        )
    with pytest.raises(RuntimeError, match="injected-private-read"):
        journal.capture_private_label_after_prediction(
            prediction_receipt=receipt,
            seed=77001,
            route_frame_index=45,
            rgb_sha256="a" * 64,
            actual_pose_sha256="b" * 64,
            provider_output_digest="4" * 64,
            privileged_getter=lambda: (_ for _ in ()).throw(
                RuntimeError("injected-private-read")
            ),
        )
    state = json.loads((tmp_path / "private" / "capture_state.json").read_text())
    assert state["privileged_access_started_count"] == 1
    assert state["privileged_capture_count"] == 0
    assert state["rerun_under_same_identity_allowed"] is False


def test_selection_denominator_zero_is_null_and_no_commit_unsafe_is_zero() -> None:
    labels = [_private_label(index, observable=False) for index in range(75)]
    _, summary = score_gain_branches(_branches(), labels)
    assert summary["common_denominator_count"] == 0
    assert summary["selected_gain"] is None
    assert all(row["unsafe_recovery_count"] == 0 for row in summary["per_gain"])


def test_selection_denominator_one_can_select_gain() -> None:
    labels = [
        _private_label(index, observable=index < 3) for index in range(75)
    ]
    _, summary = score_gain_branches(
        _branches(commit_seed_gain=(77001, 0.02)), labels
    )
    assert summary["common_denominator_count"] == 1
    assert summary["selected_gain"] == 0.02


def test_selection_equal_recovered_count_prefers_larger_gain() -> None:
    branches = _branches()
    route_digest = branches[0]["route_evidence_digest"]
    for index, gain in enumerate(STAGE2A_SELECTION_GAINS):
        branches[index] = GainBranchOutcome(
            seed=77001,
            gain=gain,
            route_evidence_digest=route_digest,
            route_protocol_safety_valid=True,
            candidate_commit_eligible=True,
            memory_commit_count=1,
            navigation_state_available=True,
            fresh_shadow_action_generation_count=1,
            committed_position_base_m=(0.402, 0.1, 0.02),
            provider_forward_count=0,
        ).to_dict()
    labels = [
        _private_label(index, observable=index < 3) for index in range(75)
    ]

    _, summary = score_gain_branches(branches, labels)

    assert [row["recovered_count"] for row in summary["per_gain"]] == [1, 1, 1]
    assert summary["selected_gain"] == 0.10


@pytest.mark.parametrize(
    ("xyz_error_m", "expected_recovered", "expected_catastrophic"),
    (
        (0.005, True, False),
        (float(np.nextafter(0.005, np.inf)), False, False),
        (0.020, False, False),
        (float(np.nextafter(0.020, np.inf)), False, True),
    ),
)
def test_scoring_uses_exact_frozen_xyz_thresholds(
    xyz_error_m: float,
    expected_recovered: bool,
    expected_catastrophic: bool,
) -> None:
    branches = _branches()
    route_digest = branches[0]["route_evidence_digest"]
    branches[0] = GainBranchOutcome(
        seed=77001,
        gain=0.02,
        route_evidence_digest=route_digest,
        route_protocol_safety_valid=True,
        candidate_commit_eligible=True,
        memory_commit_count=1,
        navigation_state_available=True,
        fresh_shadow_action_generation_count=1,
        committed_position_base_m=(xyz_error_m, 0.0, 0.0),
        provider_forward_count=0,
    ).to_dict()
    labels = [
        _private_label(index, observable=index < 3) for index in range(75)
    ]
    for index in range(3):
        label = copy.deepcopy(labels[index])
        label["gt_object_position_base_m"] = [0.0, 0.0, 0.0]
        label.pop("label_sha256")
        label["label_sha256"] = canonical_sha256(label)
        labels[index] = label

    scored, _ = score_gain_branches(branches, labels)

    row = scored[0]
    assert row["xyz_error_m"] == xyz_error_m
    assert row["recovered"] is expected_recovered
    assert row["false_recovery"] is (not expected_recovered)
    assert row["catastrophic_recovery"] is expected_catastrophic


def test_unobservable_commit_is_false_unsafe_and_eliminates_gain() -> None:
    labels = [_private_label(index, observable=False) for index in range(75)]
    _, summary = score_gain_branches(
        _branches(commit_seed_gain=(77001, 0.02)), labels
    )
    by_gain = {row["gain"]: row for row in summary["per_gain"]}
    assert by_gain[0.02]["false_recovery_count"] == 1
    assert by_gain[0.02]["unsafe_recovery_count"] == 1
    assert by_gain[0.02]["eligible"] is False


@pytest.mark.parametrize("tamper", ["observable", "mask_count", "goal_field"])
def test_private_label_resigned_tamper_is_rejected(tamper: str) -> None:
    labels = [_private_label(index, observable=False) for index in range(75)]
    target = copy.deepcopy(labels[0])
    if tamper == "observable":
        target["gt_observable"] = True
    elif tamper == "mask_count":
        target["gt_object_visible_pixel_count"] = 0
    else:
        target["goal_position_base_m"] = [0.0, 0.0, 0.0]
    target.pop("label_sha256")
    target["label_sha256"] = canonical_sha256(target)
    labels[0] = target
    with pytest.raises(RuntimeError):
        score_gain_branches(_branches(), labels)


def test_three_gain_route_binding_resigned_tamper_is_rejected() -> None:
    branches = _branches()
    tampered = copy.deepcopy(branches[1])
    tampered["route_evidence_digest"] = "0" * 64
    tampered.pop("branch_sha256")
    tampered["branch_sha256"] = canonical_sha256(tampered)
    branches[1] = tampered
    labels = [_private_label(index, observable=False) for index in range(75)]
    with pytest.raises(RuntimeError, match="同一 route"):
        score_gain_branches(branches, labels)


def test_branch_resigned_commit_state_tamper_is_rejected() -> None:
    branches = _branches()
    tampered = copy.deepcopy(branches[0])
    tampered["memory_commit_count"] = 1
    tampered["candidate_commit_eligible"] = False
    tampered["navigation_state_available"] = True
    tampered["fresh_shadow_action_generation_count"] = 1
    tampered["committed_position_base_m"] = [0.402, 0.1, 0.02]
    tampered.pop("branch_sha256")
    tampered["branch_sha256"] = canonical_sha256(tampered)
    branches[0] = tampered
    labels = [_private_label(index, observable=False) for index in range(75)]
    with pytest.raises(RuntimeError, match="状态语义"):
        score_gain_branches(branches, labels)


def test_branch_resigned_route_invalid_commit_is_rejected_fail_closed() -> None:
    branches = _branches(commit_seed_gain=(77001, 0.02))
    tampered = copy.deepcopy(branches[0])
    tampered["route_protocol_safety_valid"] = False
    tampered.pop("branch_sha256")
    tampered["branch_sha256"] = canonical_sha256(tampered)
    branches[0] = tampered
    labels = [_private_label(index, observable=True) for index in range(75)]

    with pytest.raises(RuntimeError, match="状态语义"):
        score_gain_branches(branches, labels)


@pytest.mark.parametrize("level", ["summary", "diagnostics", "gates", "gate_actual"])
def test_selection_route_summary_rejects_resigned_nested_extra_key(
    level: str,
) -> None:
    summary = _synthetic_route_summary()
    if level == "summary":
        summary["private_object_position"] = [0.0, 0.0, 0.0]
    elif level == "diagnostics":
        summary["diagnostics"]["private_object_position"] = [0.0, 0.0, 0.0]
    elif level == "gates":
        summary["gates"]["private_oracle_gate"] = {
            "actual": 0,
            "required": "none",
            "passed": True,
        }
    else:
        summary["gates"]["actual_dynamic_pose_observed"]["actual"][
            "private_object_position"
        ] = [0.0, 0.0, 0.0]
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        selection_runtime._verify_selection_route_summary_schema(
            summary,
            seed=77001,
            episode_id=(
                "e018-p1-stage2a-selection-development-seed-77001"
            ),
        )


def test_selection_route_summary_exact_nested_schema_accepts_baseline() -> None:
    summary = _synthetic_route_summary()
    verified = selection_runtime._verify_selection_route_summary_schema(
        summary,
        seed=77001,
        episode_id="e018-p1-stage2a-selection-development-seed-77001",
    )
    assert verified == summary


def test_pass_b_rejects_same_os_process_before_consuming_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = selection_runtime._new_process_identity("pass-a-producer")
    artifact = _synthetic_pass_b_artifact(
        tmp_path,
        monkeypatch,
        producer=producer,
    )

    with pytest.raises(RuntimeError, match="不同 OS 进程"):
        _score_synthetic(artifact)

    assert not (artifact["private"] / "SCORING_CONSUMED.json").exists()


def test_pass_b_wrong_go_token_cannot_create_consumption_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _synthetic_pass_b_artifact(tmp_path, monkeypatch)
    with pytest.raises(PermissionError, match="exact GO"):
        selection_runtime.run_e018_p1_stage2a_selection_score_private(
            selection_config_path=SELECTION_CONFIG,
            stage2a_config_path=Path("unused-stage2a.json"),
            qualification_config_path=Path("unused-qualification.json"),
            public_root=artifact["public"],
            private_root=artifact["private"],
            result_root=artifact["result"],
            expected_source_git_commit="4" * 40,
            expected_source_identity_sha256="1" * 64,
            exact_go_token="wrong",
        )
    assert not (artifact["private"] / "SCORING_CONSUMED.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("transaction_identity_sha256", "8" * 64),
        ("provider_output_digest", "9" * 64),
        ("prediction_commit_receipt_sha256", "a" * 64),
    ],
)
def test_pass_b_rejects_resigned_private_public_binding_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    artifact = _synthetic_pass_b_artifact(tmp_path, monkeypatch)
    _rewrite_private_label_and_inventory(
        artifact,
        label_index=0,
        changes={field: value},
    )

    with pytest.raises(RuntimeError, match="public binding"):
        _score_synthetic(artifact)

    marker = json.loads(
        (artifact["private"] / "SCORING_CONSUMED.json").read_text()
    )
    assert marker["status"] == "PASS_B_FAILED_IDENTITY_PERMANENTLY_CONSUMED"
    assert marker["label_open_started_count"] == 1
    assert marker["label_open_completed_count"] == 0
    assert marker["rerun_under_same_identity_allowed"] is False


@pytest.mark.parametrize("tamper", ["path", "order"])
def test_pass_b_rejects_private_inventory_path_or_order_before_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    artifact = _synthetic_pass_b_artifact(tmp_path, monkeypatch)
    inventory_path = artifact["public"] / "private_label_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    if tamper == "path":
        inventory["rows"][0]["path"] = "label_commits/000001.json"
        row = inventory["rows"][0]
        row.pop("row_sha256")
        row["row_sha256"] = canonical_sha256(row)
    else:
        inventory["rows"][0], inventory["rows"][1] = (
            inventory["rows"][1],
            inventory["rows"][0],
        )
    inventory.pop("inventory_sha256")
    inventory["inventory_sha256"] = canonical_sha256(inventory)
    _write_json(inventory_path, inventory)

    with pytest.raises(RuntimeError, match="identity/order"):
        _score_synthetic(artifact)

    assert not (artifact["private"] / "SCORING_CONSUMED.json").exists()


def test_pass_b_rejects_resigned_private_inventory_raw_hash_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _synthetic_pass_b_artifact(tmp_path, monkeypatch)
    inventory_path = artifact["public"] / "private_label_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    row = inventory["rows"][0]
    row["raw_sha256"] = "b" * 64
    row.pop("row_sha256")
    row["row_sha256"] = canonical_sha256(row)
    inventory.pop("inventory_sha256")
    inventory["inventory_sha256"] = canonical_sha256(inventory)
    _write_json(inventory_path, inventory)

    with pytest.raises(RuntimeError, match="public binding"):
        _score_synthetic(artifact)

    marker = json.loads(
        (artifact["private"] / "SCORING_CONSUMED.json").read_text()
    )
    assert marker["status"] == "PASS_B_FAILED_IDENTITY_PERMANENTLY_CONSUMED"
    assert marker["label_open_started_count"] == 1
    assert marker["label_open_completed_count"] == 0


def test_pass_b_rejects_tampered_producer_process_identity_before_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _synthetic_pass_b_artifact(tmp_path, monkeypatch)
    artifact["public_result"]["producer_process_identity"]["pid"] += 1

    with pytest.raises(RuntimeError, match="process identity"):
        _score_synthetic(artifact)

    assert not (artifact["private"] / "SCORING_CONSUMED.json").exists()


def test_process_identity_from_new_subprocess_is_distinct() -> None:
    producer = selection_runtime._new_process_identity("pass-a-producer")
    command = (
        "import json; "
        "from robot_vla.precision.e018_p1_stage2a_selection_runtime "
        "import _new_process_identity; "
        "print(json.dumps(_new_process_identity('pass-b-scorer')))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    scorer = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", command],
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
        )
    )

    selection_runtime._verify_process_identity(
        scorer,
        role="pass-b-scorer",
    )
    assert (
        scorer["process_instance_sha256"]
        != producer["process_instance_sha256"]
    )


def test_pass_b_exact_once_and_result_verifier_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _synthetic_pass_b_artifact(tmp_path, monkeypatch)
    original = selection_runtime._read_private_label_exactly_once
    opened: list[str] = []

    def checked_open(path: Path):
        marker = json.loads(
            (artifact["private"] / "SCORING_CONSUMED.json").read_text()
        )
        assert marker["label_open_started_count"] == len(opened) + 1
        assert marker["label_open_completed_count"] == len(opened)
        opened.append(path.name)
        return original(path)

    monkeypatch.setattr(
        selection_runtime,
        "_read_private_label_exactly_once",
        checked_open,
    )
    result = _score_synthetic(artifact)

    assert result["verified"] is True
    assert result["complete_marker_verified"] is True
    assert result["selected_gain"] is None
    assert result["stage2b_continuation_required"] is True
    assert len(opened) == 75
    assert {path.name for path in artifact["result"].iterdir()} == {
        "scored_gain_branches.jsonl",
        "selection_summary.json",
        "result_receipt.json",
        "RESULT_COMPLETE.json",
    }
    marker = json.loads(
        (artifact["private"] / "SCORING_CONSUMED.json").read_text()
    )
    assert marker["label_open_started_count"] == 75
    assert marker["label_open_completed_count"] == 75
    assert marker["rerun_under_same_identity_allowed"] is False

    retry_root = tmp_path / "retry-after-complete"
    retry_artifact = {
        **artifact,
        "public": retry_root / "public_execution",
        "private": retry_root / "private_labels",
        "result": retry_root / "result",
    }
    shutil.copytree(artifact["public"], retry_artifact["public"])
    shutil.copytree(artifact["private"], retry_artifact["private"])
    with pytest.raises(RuntimeError, match="已消费"):
        _score_synthetic(retry_artifact)


def test_pass_b_mid_label_failure_remains_permanently_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _synthetic_pass_b_artifact(tmp_path, monkeypatch)
    original = selection_runtime._read_private_label_exactly_once
    calls = 0

    def fail_fifth(path: Path):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("injected-label-open-failure")
        return original(path)

    monkeypatch.setattr(
        selection_runtime,
        "_read_private_label_exactly_once",
        fail_fifth,
    )
    with pytest.raises(RuntimeError, match="injected-label-open-failure"):
        _score_synthetic(artifact)
    marker = json.loads(
        (artifact["private"] / "SCORING_CONSUMED.json").read_text()
    )
    assert marker["status"] == "PASS_B_FAILED_IDENTITY_PERMANENTLY_CONSUMED"
    assert marker["label_open_started_count"] == 5
    assert marker["label_open_completed_count"] == 4
    assert marker["rerun_under_same_identity_allowed"] is False

    retry_root = tmp_path / "retry-after-failure"
    retry_artifact = {
        **artifact,
        "public": retry_root / "public_execution",
        "private": retry_root / "private_labels",
        "result": retry_root / "result",
    }
    shutil.copytree(artifact["public"], retry_artifact["public"])
    shutil.copytree(artifact["private"], retry_artifact["private"])
    with pytest.raises(RuntimeError, match="已消费"):
        _score_synthetic(retry_artifact)


@pytest.mark.parametrize("layout", ["different-parent", "extra-top-level"])
def test_pass_b_rejects_noncanonical_artifact_layout_before_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
) -> None:
    artifact = _synthetic_pass_b_artifact(tmp_path, monkeypatch)
    if layout == "different-parent":
        other_parent = tmp_path / "other-artifact"
        other_parent.mkdir()
        artifact["result"] = other_parent / "result"
        expected = "同一真实父目录"
    else:
        (tmp_path / "unexpected").mkdir()
        expected = "top-level directories"

    with pytest.raises(RuntimeError, match=expected):
        _score_synthetic(artifact)

    assert not (artifact["private"] / "SCORING_CONSUMED.json").exists()


@pytest.mark.parametrize("phase", ["reserve", "post-completion"])
def test_pass_b_budget_failure_is_permanently_consumed_without_complete_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    artifact = _synthetic_pass_b_artifact(tmp_path, monkeypatch)
    cap = load_e018_p1_stage2a_selection_config(SELECTION_CONFIG).payload[
        "budgets"
    ]["combined_artifact_bytes_max"]
    calls = 0

    def injected_size(_: Path) -> int:
        nonlocal calls
        calls += 1
        if phase == "reserve" or calls > 1:
            return cap + 1
        return 0

    monkeypatch.setattr(
        selection_runtime,
        "_combined_artifact_bytes",
        injected_size,
    )
    with pytest.raises(RuntimeError, match="artifact budget"):
        _score_synthetic(artifact)

    assert not (artifact["result"] / "RESULT_COMPLETE.json").exists()
    marker = json.loads(
        (artifact["private"] / "SCORING_CONSUMED.json").read_text()
    )
    assert marker["status"] == "PASS_B_FAILED_IDENTITY_PERMANENTLY_CONSUMED"
    assert marker["rerun_under_same_identity_allowed"] is False


def test_post_consumption_publisher_failure_freezes_identity_and_removes_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _synthetic_pass_b_artifact(tmp_path, monkeypatch)

    def fail_result_ledger(*_: object, **__: object) -> None:
        raise RuntimeError("injected-post-consumption-publisher-failure")

    monkeypatch.setattr(selection_runtime, "_atomic_jsonl", fail_result_ledger)
    with pytest.raises(
        RuntimeError,
        match="injected-post-consumption-publisher-failure",
    ):
        _score_synthetic(artifact)

    marker = json.loads(
        (artifact["private"] / "SCORING_CONSUMED.json").read_text()
    )
    assert marker["status"] == "PASS_B_FAILED_IDENTITY_PERMANENTLY_CONSUMED"
    assert marker["label_open_started_count"] == 75
    assert marker["label_open_completed_count"] == 75
    assert marker["failure"]["error_type"] == "RuntimeError"
    assert "injected-post-consumption" in marker["failure"]["message"]
    assert marker["rerun_under_same_identity_allowed"] is False
    assert not (artifact["result"] / "RESULT_COMPLETE.json").exists()

    retry_root = tmp_path / "retry-after-publisher-failure"
    retry_artifact = {
        **artifact,
        "public": retry_root / "public_execution",
        "private": retry_root / "private_labels",
        "result": retry_root / "result",
    }
    shutil.copytree(artifact["public"], retry_artifact["public"])
    shutil.copytree(artifact["private"], retry_artifact["private"])
    with pytest.raises(RuntimeError, match="已消费"):
        _score_synthetic(retry_artifact)


def test_result_artifact_verifies_after_copy_and_role_swap_differs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _synthetic_pass_b_artifact(tmp_path, monkeypatch)
    _score_synthetic(artifact)
    copied_root = tmp_path / "drive-copy"
    copied_public = copied_root / "public_execution"
    copied_private = copied_root / "private_labels"
    copied_result = copied_root / "result"
    shutil.copytree(artifact["public"], copied_public)
    shutil.copytree(artifact["private"], copied_private)
    shutil.copytree(artifact["result"], copied_result)

    verified = selection_runtime.verify_e018_p1_stage2a_selection_result(
        selection_config_path=SELECTION_CONFIG,
        stage2a_config_path=Path("unused-stage2a.json"),
        qualification_config_path=Path("unused-qualification.json"),
        public_root=copied_public,
        result_root=copied_result,
        expected_source_git_commit="4" * 40,
        expected_source_identity_sha256="1" * 64,
    )
    assert verified["verified"] is True
    public_role = artifact["public_result"][
        "public_artifact_role_identity_sha256"
    ]
    private_role = artifact["public_result"][
        "private_artifact_role_identity_sha256"
    ]
    assert public_role != private_role


def test_result_verifier_rejects_non_sibling_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _synthetic_pass_b_artifact(tmp_path, monkeypatch)
    _score_synthetic(artifact)
    isolated_root = tmp_path / "isolated-copy"
    isolated_root.mkdir()
    copied_result = isolated_root / "result"
    shutil.copytree(artifact["result"], copied_result)

    with pytest.raises(RuntimeError, match="同一真实父目录"):
        _verify_synthetic_result(artifact, result_root=copied_result)


@pytest.mark.parametrize("tree_tamper", ["extra", "symlink", "hardlink"])
def test_result_verifier_rejects_exact_tree_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tree_tamper: str,
) -> None:
    artifact = _synthetic_pass_b_artifact(tmp_path, monkeypatch)
    _score_synthetic(artifact)
    result_root = artifact["result"]
    extra = result_root / "unexpected.json"
    if tree_tamper == "extra":
        extra.write_text("{}\n")
    elif tree_tamper == "symlink":
        extra.symlink_to("selection_summary.json")
    else:
        os.link(result_root / "selection_summary.json", extra)

    with pytest.raises(RuntimeError, match="exact tree|symlink|hardlink"):
        _verify_synthetic_result(artifact)


def test_result_verifier_recomputes_summary_from_resigned_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _synthetic_pass_b_artifact(tmp_path, monkeypatch)
    _score_synthetic(artifact)
    scored_path = artifact["result"] / "scored_gain_branches.jsonl"
    rows = [json.loads(line) for line in scored_path.read_text().splitlines()]
    rows[0]["protocol_violation_count"] = 1
    rows[0].pop("scored_row_sha256")
    rows[0]["scored_row_sha256"] = canonical_sha256(rows[0])
    _write_jsonl(scored_path, rows)

    with pytest.raises(RuntimeError, match="独立重算"):
        _verify_synthetic_result(artifact)


def test_result_verifier_rejects_resigned_completion_marker_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _synthetic_pass_b_artifact(tmp_path, monkeypatch)
    _score_synthetic(artifact)
    marker_path = artifact["result"] / "RESULT_COMPLETE.json"
    marker = json.loads(marker_path.read_text())
    marker["transaction_identity_sha256"] = "f" * 64
    marker.pop("marker_sha256")
    marker["marker_sha256"] = canonical_sha256(marker)
    _write_json(marker_path, marker)

    with pytest.raises(RuntimeError, match="completion marker"):
        _verify_synthetic_result(artifact)


def test_selection_cli_keeps_private_scoring_and_result_verifier_narrow() -> None:
    common = [
        "--selection-config",
        "selection.json",
        "--stage2a-config",
        "stage2a.json",
        "--qualification-config",
        "qualification.json",
        "--expected-source-git-commit",
        "4" * 40,
        "--expected-source-identity-sha256",
        "1" * 64,
        "--public-root",
        "public",
    ]
    score = build_parser().parse_args(
        [
            "score-private",
            *common,
            "--private-root",
            "private",
            "--result-root",
            "result",
            "--selection-go",
            STAGE2A_SELECTION_GO,
        ]
    )
    verify = build_parser().parse_args(
        ["verify-result", *common, "--result-root", "result"]
    )
    for args in (score, verify):
        assert not hasattr(args, "selected_checkpoint")
        assert not hasattr(args, "stats_root")
        assert not hasattr(args, "provider")
    assert not hasattr(verify, "private_root")
