"""E018-P1 G1：受限主动观察 supervisor 的纯合同 replay gate。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from robot_vla.executive.contracts import PhaseId
from robot_vla.precision import e018_p1_g0 as _g0
from robot_vla.precision.active_front_reobserve import (
    ACTIVE_FRONT_REOBSERVE_VERSION,
    ALLOWED_ACTIVE_SOURCE_PHASES,
    POST_ACTIVE_WINDOW_PHASES,
    ActionHistoryResetReceipt,
    ActionHistoryResumeReceipt,
    ActiveFrontDecisionReason,
    ActiveFrontFailure,
    ActiveFrontReobserveConfig,
    ActiveFrontReobserveController,
    ActiveFrontReobserveState,
    ActiveFrontSafetyEvidence,
    ActiveFrontSignal,
    ActiveFrontTriggerEvidence,
    ActiveFrontTriggerReason,
    HomeV2BarrierFrame,
    Stage1ShadowCandidateReceipt,
)

E018_P1_G1_CONFIG_VERSION = "e018-p1-g1-control-shadow-replay-development/v2"
E018_P1_G1_RESULT_VERSION = "e018-p1-g1-control-shadow-replay-result/v2"
E018_P1_G1_GATE = "G1_RESTRICTED_ACTIVE_CONTROL_CONTRACT"
_PARENT_CONFIG_SHA256 = "80a672bf63a43b9714d0757ba5e54bb491e85fa6a01368c4f77686ffa421c929"
_PARENT_RECEIPT_SHA256 = "afc8aa9cf91cd235136706a1d10c385e68b0fa6942a8a5c3041098a9ab7768dc"
_FAILURE_CASES = (
    "primitive_identity_mismatch",
    "contact_during_motion",
    "polluted_home_frame",
    "source_invariant_failed",
    "stale_action_history_resume",
    "motion_timeout",
)
_SOURCE_FILES = (
    "src/robot_vla/precision/active_front_reobserve.py",
    "src/robot_vla/precision/e018_p1_g1.py",
    "src/robot_vla/cli/run_e018_p1_g1.py",
)


def load_e018_p1_g1_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"E018-P1 G1 config 不存在: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = _g0._require_keys(
        config,
        {
            "version",
            "status",
            "scope",
            "parent_capability",
            "supervisor",
            "replay",
            "execution",
        },
        "E018-P1 G1 config",
    )
    if config["version"] != E018_P1_G1_CONFIG_VERSION:
        raise ValueError("E018-P1 G1 config version 漂移")
    if config["status"] != "development-only-stage1-shadow-no-live-consumers":
        raise ValueError("E018-P1 G1 只能以 development-only shadow 状态运行")
    scope = _g0._require_keys(
        config["scope"],
        {
            "gate",
            "test_split_allowed",
            "formal_claim_allowed",
            "provider_inference_allowed",
            "memory_read_allowed",
            "memory_write_allowed",
            "executive_integration_allowed",
            "camera_actuation_allowed",
            "arm_actuation_allowed",
        },
        "scope",
    )
    if scope != {
        "gate": E018_P1_G1_GATE,
        "test_split_allowed": False,
        "formal_claim_allowed": False,
        "provider_inference_allowed": False,
        "memory_read_allowed": False,
        "memory_write_allowed": False,
        "executive_integration_allowed": False,
        "camera_actuation_allowed": False,
        "arm_actuation_allowed": False,
    }:
        raise ValueError("G1 scope 必须禁止 test/formal/provider/Memory/Executive/actuation")
    parent = _g0._require_keys(
        config["parent_capability"],
        {"gate", "config_version", "config_sha256", "receipt_sha256", "gate_passed"},
        "parent_capability",
    )
    if parent != {
        "gate": "G1A_DYNAMIC_EXTERNAL_OBSERVATION_CAPABILITY",
        "config_version": "e018-p1-g1a-dynamic-external-observation-probe/v2",
        "config_sha256": _PARENT_CONFIG_SHA256,
        "receipt_sha256": _PARENT_RECEIPT_SHA256,
        "gate_passed": True,
    }:
        raise ValueError("G1 parent G1A identity 漂移")
    supervisor = _g0._require_keys(
        config["supervisor"],
        {
            "version",
            "enabled",
            "selected_primitive_id",
            "consecutive_unusable_ticks",
            "cooldown_ticks",
            "maximum_attempts_per_episode",
            "home_v2_barrier_frames",
            "allowed_source_phases",
            "post_active_window_phases",
            "candidate_mode",
        },
        "supervisor",
    )
    expected_phases = [phase.value for phase in sorted(ALLOWED_ACTIVE_SOURCE_PHASES, key=str)]
    expected_post_phases = sorted(phase.value for phase in POST_ACTIVE_WINDOW_PHASES)
    if (
        supervisor["version"] != ACTIVE_FRONT_REOBSERVE_VERSION
        or supervisor["enabled"] is not True
        or supervisor["selected_primitive_id"] != "LEFT_LOW__YAW_LEFT"
        or supervisor["consecutive_unusable_ticks"] != 3
        or supervisor["cooldown_ticks"] != 20
        or supervisor["maximum_attempts_per_episode"] != 1
        or supervisor["home_v2_barrier_frames"] != 4
        or sorted(supervisor["allowed_source_phases"]) != expected_phases
        or sorted(supervisor["post_active_window_phases"]) != expected_post_phases
        or supervisor["candidate_mode"] != "shadow-only-no-provider-no-memory/v1"
    ):
        raise ValueError("G1 supervisor identity/边界漂移")
    ActiveFrontReobserveConfig(
        enabled=True,
        selected_primitive_id=supervisor["selected_primitive_id"],
        consecutive_unusable_ticks=supervisor["consecutive_unusable_ticks"],
        cooldown_ticks=supervisor["cooldown_ticks"],
        maximum_attempts_per_episode=supervisor["maximum_attempts_per_episode"],
        home_v2_barrier_frames=supervisor["home_v2_barrier_frames"],
    )
    replay = _g0._require_keys(
        config["replay"],
        {
            "evidence_source",
            "success_source_phases",
            "failure_cases",
            "repeat_success_replays",
        },
        "replay",
    )
    if (
        replay["evidence_source"] != "synthetic-contract-evidence-no-provider/v1"
        or replay["success_source_phases"]
        != [PhaseId.ACQUIRE_TRACK.value, PhaseId.STABILIZE_PREGRASP.value]
        or tuple(replay["failure_cases"]) != _FAILURE_CASES
        or replay["repeat_success_replays"] != 2
    ):
        raise ValueError("G1 replay cases/order 漂移")
    execution = _g0._require_keys(
        config["execution"],
        {
            "device",
            "simulator_steps_allowed",
            "physical_robot_actuation_allowed",
            "simulated_camera_actuation_allowed",
            "provider_inference_allowed",
            "memory_read_allowed",
            "memory_write_allowed",
            "test_data_read_allowed",
        },
        "execution",
    )
    if execution != {
        "device": "cpu",
        "simulator_steps_allowed": False,
        "physical_robot_actuation_allowed": False,
        "simulated_camera_actuation_allowed": False,
        "provider_inference_allowed": False,
        "memory_read_allowed": False,
        "memory_write_allowed": False,
        "test_data_read_allowed": False,
    }:
        raise ValueError("G1 execution 必须是无 simulator/provider/Memory/test/actuation replay")
    return config


def _controller(config: dict[str, Any], *, episode_id: str) -> ActiveFrontReobserveController:
    values = config["supervisor"]
    controller = ActiveFrontReobserveController(
        ActiveFrontReobserveConfig(
            enabled=True,
            selected_primitive_id=values["selected_primitive_id"],
            consecutive_unusable_ticks=values["consecutive_unusable_ticks"],
            cooldown_ticks=values["cooldown_ticks"],
            maximum_attempts_per_episode=values["maximum_attempts_per_episode"],
            home_v2_barrier_frames=values["home_v2_barrier_frames"],
        )
    )
    controller.reset_episode(episode_id, episode_generation=1)
    return controller


def _trigger(
    controller: ActiveFrontReobserveController,
    *,
    episode_id: str,
    phase: PhaseId,
):
    decision = None
    for tick in range(3):
        decision = controller.consider_trigger(
            ActiveFrontTriggerEvidence(
                episode_id=episode_id,
                episode_generation=1,
                control_tick=tick,
                timestamp_s=tick * 0.05,
                source_phase=phase,
                wrist_object_measurement_usable=False,
                front_home_object_measurement_usable=False,
                object_memory_navigation_state_available=False,
                arm_hold_prerequisites_pass=True,
                camera_home_prerequisites_pass=True,
                failure_reason=ActiveFrontTriggerReason.OBJECT_OCCLUSION,
            )
        )
    if decision is None or decision.request is None:
        raise RuntimeError("G1 replay 未形成预期 active request")
    return decision.request


def _begin(controller: ActiveFrontReobserveController, request: Any) -> None:
    controller.begin(
        ActionHistoryResetReceipt(
            episode_id=request.episode_id,
            request_id=request.request_id,
            reset_control_tick=request.trigger_tick,
            generation_before=4,
            generation_after=5,
            action_chunk_cleared=True,
            temporal_ensemble_cleared=True,
            rtc_overlap_cleared=True,
            command_reference_invalidated=True,
        )
    )


def _advance_to_candidate(controller: ActiveFrontReobserveController, request: Any) -> None:
    controller.advance(ActiveFrontSignal.CAMERA_LEASE_ACQUIRED)
    controller.advance(
        ActiveFrontSignal.FROZEN_PRIMITIVE_SELECTED,
        selected_primitive_id=request.selected_primitive_id,
    )
    controller.advance(ActiveFrontSignal.MOVE_COMPLETE)
    controller.advance(ActiveFrontSignal.SETTLE_COMPLETE)
    controller.advance(ActiveFrontSignal.COLLECTION_COMPLETE)


def _stage_and_return(controller: ActiveFrontReobserveController, request: Any) -> None:
    controller.advance(
        ActiveFrontSignal.SHADOW_CANDIDATE_STAGED,
        shadow_candidate_receipt=Stage1ShadowCandidateReceipt(
            request_id=request.request_id,
            candidate_digest="stage1-shadow-no-provider-candidate",
            shadow_only=True,
            live_memory_write_executed=False,
            provider_forward_count=0,
        ),
    )
    controller.advance(ActiveFrontSignal.RETURN_HOME_COMPLETE)


def _barrier(controller: ActiveFrontReobserveController, *, polluted: bool = False) -> None:
    for index in range(4):
        controller.accept_home_v2_barrier_frame(
            HomeV2BarrierFrame(
                observation_sequence_id=f"fresh-home-v2-{index}",
                camera_at_home=True,
                fresh_observation_v2_frame=True,
                captured_after_return=True,
                contains_alternate_or_motion_rgb=polluted and index == 0,
            )
        )
        if polluted:
            return


def _success_receipt(
    config: dict[str, Any],
    *,
    phase: PhaseId,
    episode_id: str,
):
    controller = _controller(config, episode_id=episode_id)
    request = _trigger(controller, episode_id=episode_id, phase=phase)
    _begin(controller, request)
    _advance_to_candidate(controller, request)
    _stage_and_return(controller, request)
    _barrier(controller)
    controller.advance(
        ActiveFrontSignal.SOURCE_INVARIANTS_VERIFIED,
        source_phase=phase,
        source_invariants_passed=True,
    )
    return controller.complete_no_write_resume(
        ActionHistoryResumeReceipt(
            episode_id=episode_id,
            request_id=request.request_id,
            generation=6,
            home_observation_sequence_ids=tuple(f"fresh-home-v2-{i}" for i in range(4)),
            generated_from_fresh_home_v2=True,
            stale_action_chunk_resumed=False,
        )
    )


def _failure_receipt(config: dict[str, Any], case: str):
    episode_id = f"g1-failure-{case}"
    controller = _controller(config, episode_id=episode_id)
    request = _trigger(controller, episode_id=episode_id, phase=PhaseId.ACQUIRE_TRACK)
    _begin(controller, request)
    if case == "primitive_identity_mismatch":
        controller.advance(ActiveFrontSignal.CAMERA_LEASE_ACQUIRED)
        controller.advance(
            ActiveFrontSignal.FROZEN_PRIMITIVE_SELECTED,
            selected_primitive_id="RIGHT_LOW__YAW_RIGHT",
        )
    elif case == "contact_during_motion":
        controller.advance(ActiveFrontSignal.CAMERA_LEASE_ACQUIRED)
        controller.advance(
            ActiveFrontSignal.FROZEN_PRIMITIVE_SELECTED,
            selected_primitive_id=request.selected_primitive_id,
        )
        controller.advance(
            ActiveFrontSignal.MOVE_COMPLETE,
            safety=ActiveFrontSafetyEvidence(contact_absent=False),
        )
        controller.complete_failsafe_return(home_verified=True)
    elif case == "polluted_home_frame":
        _advance_to_candidate(controller, request)
        _stage_and_return(controller, request)
        _barrier(controller, polluted=True)
    elif case == "source_invariant_failed":
        _advance_to_candidate(controller, request)
        _stage_and_return(controller, request)
        _barrier(controller)
        controller.advance(
            ActiveFrontSignal.SOURCE_INVARIANTS_VERIFIED,
            source_phase=PhaseId.ACQUIRE_TRACK,
            source_invariants_passed=False,
        )
    elif case == "stale_action_history_resume":
        _advance_to_candidate(controller, request)
        _stage_and_return(controller, request)
        _barrier(controller)
        controller.advance(
            ActiveFrontSignal.SOURCE_INVARIANTS_VERIFIED,
            source_phase=PhaseId.ACQUIRE_TRACK,
            source_invariants_passed=True,
        )
        return controller.complete_no_write_resume(
            ActionHistoryResumeReceipt(
                episode_id=episode_id,
                request_id=request.request_id,
                generation=5,
                home_observation_sequence_ids=tuple(
                    f"fresh-home-v2-{i}" for i in range(4)
                ),
                generated_from_fresh_home_v2=True,
                stale_action_chunk_resumed=False,
            )
        )
    elif case == "motion_timeout":
        controller.advance(ActiveFrontSignal.CAMERA_LEASE_ACQUIRED)
        controller.advance(
            ActiveFrontSignal.FROZEN_PRIMITIVE_SELECTED,
            selected_primitive_id=request.selected_primitive_id,
        )
        controller.fail(ActiveFrontFailure.TIMEOUT, camera_at_home=False)
        controller.complete_failsafe_return(home_verified=True)
    else:
        raise ValueError(f"未知 G1 failure case: {case}")
    if controller.state is ActiveFrontReobserveState.FAILSAFE_RETURN:
        controller.complete_failsafe_return(home_verified=True)
    return controller.receipt()


def _source_identity(repository_root: Path) -> dict[str, Any]:
    safe_repository = str(repository_root.resolve())
    git = ("git", "-c", f"safe.directory={safe_repository}")
    identity = {
        "git_commit": subprocess.run(
            [*git, "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "git_status": subprocess.run(
            [*git, "status", "--porcelain", "--untracked-files=all"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines(),
        "source_file_sha256": {
            path: _g0._file_sha256(repository_root / path) for path in _SOURCE_FILES
        },
    }
    identity["worktree_clean"] = not identity["git_status"]
    identity["identity_sha256"] = _g0._canonical_sha256(identity)
    return identity


def run_e018_p1_g1(
    *,
    config_path: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    config = load_e018_p1_g1_config(config_path)
    repository = Path(repository_root)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"E018-P1 G1 output 已存在: {output}")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    source_identity = _source_identity(repository)
    success_receipts = []
    deterministic = True
    for phase_value in config["replay"]["success_source_phases"]:
        phase = PhaseId(phase_value)
        first = _success_receipt(config, phase=phase, episode_id=f"g1-success-{phase.value}")
        second = _success_receipt(config, phase=phase, episode_id=f"g1-success-{phase.value}")
        deterministic &= first.audit_digest == second.audit_digest
        success_receipts.append(first)
    failure_receipts = [_failure_receipt(config, case) for case in _FAILURE_CASES]

    post_window_replays = []
    for phase in sorted(POST_ACTIVE_WINDOW_PHASES, key=lambda item: item.value):
        episode_id = f"g1-post-window-{phase.value}"
        blocked = _controller(config, episode_id=episode_id)
        first = blocked.consider_trigger(
            ActiveFrontTriggerEvidence(
                episode_id=episode_id,
                episode_generation=1,
                control_tick=0,
                timestamp_s=0.0,
                source_phase=phase,
                wrist_object_measurement_usable=False,
                front_home_object_measurement_usable=False,
                object_memory_navigation_state_available=False,
                arm_hold_prerequisites_pass=True,
                camera_home_prerequisites_pass=True,
                failure_reason=ActiveFrontTriggerReason.OBJECT_OCCLUSION,
            )
        )
        recovered = blocked.consider_trigger(
            ActiveFrontTriggerEvidence(
                episode_id=episode_id,
                episode_generation=1,
                control_tick=1,
                timestamp_s=0.05,
                source_phase=PhaseId.ACQUIRE_TRACK,
                wrist_object_measurement_usable=False,
                front_home_object_measurement_usable=False,
                object_memory_navigation_state_available=False,
                arm_hold_prerequisites_pass=True,
                camera_home_prerequisites_pass=True,
                failure_reason=ActiveFrontTriggerReason.OBJECT_OCCLUSION,
            )
        )
        post_window_replays.append(
            {
                "phase": phase.value,
                "initial_reason": first.reason.value,
                "recovered_phase": PhaseId.ACQUIRE_TRACK.value,
                "recovered_reason": recovered.reason.value,
                "active_window_open": blocked.active_window_open,
                "test_data_read": False,
            }
        )
    all_receipts = [*success_receipts, *failure_receipts]
    gate_checks = {
        "two_allowed_source_phases_complete": len(success_receipts) == 2
        and all(
            receipt.status == "complete-stage1-shadow-no-write"
            for receipt in success_receipts
        ),
        "success_replay_digest_stable": deterministic,
        "four_fresh_home_frames_before_resume": all(
            len(receipt.home_observation_sequence_ids) == 4 for receipt in success_receipts
        ),
        "action_history_generation_advanced": all(
            receipt.action_history_generation_after_reset
            == receipt.action_history_generation_before + 1
            and receipt.resumed_action_history_generation
            == receipt.action_history_generation_after_reset + 1
            for receipt in success_receipts
        ),
        "all_failure_cases_fail_safe": len(failure_receipts) == len(_FAILURE_CASES)
        and all(receipt.status == "failed-safe-hold-no-write" for receipt in failure_receipts),
        "no_live_consumers": all(
            receipt.memory_read_count == 0
            and receipt.memory_write_count == 0
            and receipt.provider_forward_count == 0
            and receipt.test_read_count == 0
            for receipt in all_receipts
        ),
        "all_post_active_window_phases_close_latch": (
            len(post_window_replays) == len(POST_ACTIVE_WINDOW_PHASES)
            and all(
                row["initial_reason"]
                == ActiveFrontDecisionReason.DISALLOWED_SOURCE_PHASE.value
                and row["recovered_reason"]
                == ActiveFrontDecisionReason.ACTIVE_WINDOW_CLOSED.value
                and row["active_window_open"] is False
                for row in post_window_replays
            )
        ),
        "no_actuation_executed": True,
    }
    summary = {
        "version": E018_P1_G1_RESULT_VERSION,
        "status": "complete-development-only",
        "gate": E018_P1_G1_GATE,
        "gate_passed": all(gate_checks.values()),
        "config_sha256": _g0._canonical_sha256(config),
        "parent_capability": config["parent_capability"],
        "source_identity": source_identity,
        "gate_checks": gate_checks,
        "success_replay_count": len(success_receipts),
        "failure_replay_count": len(failure_receipts),
        "post_active_window_phase_count": len(post_window_replays),
        "success_receipt_digests": [receipt.audit_digest for receipt in success_receipts],
        "failure_cases": list(_FAILURE_CASES),
        "failure_reasons": [
            None if receipt.failure is None else receipt.failure.value
            for receipt in failure_receipts
        ],
        "simulator_step_count": 0,
        "camera_actuation_count": 0,
        "arm_actuation_count": 0,
        "provider_forward_count": 0,
        "memory_read_count": 0,
        "memory_write_count": 0,
        "test_read_count": 0,
        "formal_claim_allowed": False,
        "scope_limits": {
            "proves": [
                "pure trigger/latch/state/barrier/history contracts are deterministic",
                "declared failure injections terminate in no-write SafeHold",
            ],
            "does_not_prove": [
                (
                    "runtime integration, camera actuation, provider quality, Memory "
                    "update, or task benefit"
                ),
            ],
        },
    }
    _g0._atomic_jsonl(
        output / "replay_receipts.jsonl",
        [receipt.as_dict() for receipt in all_receipts],
    )
    _g0._atomic_jsonl(
        output / "blocked_post_active_window_decisions.jsonl",
        post_window_replays,
    )
    _g0._atomic_json(output / "config_snapshot.json", config)
    _g0._atomic_json(output / "summary.json", summary)
    artifacts = sorted(output.glob("*"), key=lambda path: path.name)
    receipt = {
        "version": E018_P1_G1_RESULT_VERSION,
        "status": "complete-development-only",
        "gate_passed": summary["gate_passed"],
        "config_sha256": summary["config_sha256"],
        "files": {path.name: _g0._file_sha256(path) for path in artifacts},
        "test_split_status": "prohibited-unread",
        "formal_claim_allowed": False,
    }
    receipt["receipt_sha256"] = _g0._canonical_sha256(receipt)
    _g0._atomic_json(output / "receipt.json", receipt)
    return summary


__all__ = [
    "E018_P1_G1_CONFIG_VERSION",
    "E018_P1_G1_GATE",
    "E018_P1_G1_RESULT_VERSION",
    "load_e018_p1_g1_config",
    "run_e018_p1_g1",
]
