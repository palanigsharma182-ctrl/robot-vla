from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_vla.sim.local_dagger_diagnostics import (
    EPISODE_TIME_LIMIT_REASON,
    EXPERT_LIFT_MOTION_PHASE,
    EXPERT_LOWER_MOTION_PHASE,
    EXPERT_RELEASE_SETTLE_PHASE,
    EXPERT_TRANSPORT_MOTION_PHASE,
    LOCAL_DAGGER_DIAGNOSTIC_FORMAT,
    POLICY_BEFORE_BOUNDARY_REASON,
    POLICY_ROLLIN_PHASE,
)
from scripts.analyze_e012_gl_failure_diagnostics import (
    ANALYSIS_FORMAT,
    COLLECTION_FORMAT,
    EXPERT_INCOMPLETE_REASON,
    FORMAL_GL_SEEDS,
    MPLIB_PATH_REASON,
    POOL_FORMAT,
    REPLAY_FORMAT,
    SNAPSHOT_REASON_PREFIX,
    analyze_gl_failure_diagnostics,
    main,
)

FORMAL_SOURCE = "source-tree-sha256:" + "a" * 64
REPLAY_SOURCE = "source-tree-sha256:" + "b" * 64
CHECKPOINT_SHA = "c" * 64


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pool_config() -> dict:
    return {
        "inference_strategy": "temporal-ensemble",
        "max_anomaly_replans": 3,
        "num_flow_steps": 10,
        "paired_clean_expert_required": True,
        "qwen_context_layer": 12,
        "recency_decay": 0.5,
        "sampling_seed_base": 52_012,
        "snapshot_round_trip_required": True,
    }


def _episode_config(seed: int) -> dict:
    return {
        "boundary_type": "grasp_lift",
        "environment_seed": seed,
        "episode_sampling_seed": 7_000_000 + seed,
        "max_anomaly_replans": 3,
        "num_flow_steps": 10,
        "paired_clean_expert_required": True,
        "qwen_context_layer": 12,
        "recency_decay": 0.5,
        "sampling_seed_base": 52_012,
        "snapshot_round_trip_required": True,
    }


def _checkpoint() -> dict:
    return {"path": "/frozen/e011.pt", "sha256": CHECKPOINT_SHA}


def _base_record(seed: int, *, source: str, status: str, reason: str | None) -> dict:
    record = {
        "format": COLLECTION_FORMAT,
        "source_revision": source,
        "base_dataset": "/frozen/d0",
        "checkpoint": _checkpoint(),
        "config": _episode_config(seed),
        "status": status,
    }
    if reason is not None:
        record["failure"] = {"type": "EpisodeRejected", "reason": reason}
    return record


def _accepted_record(seed: int, index: int) -> tuple[dict, int, int]:
    takeover = 80 + index * 10
    success_delta = 120 + index
    num_steps = takeover + success_delta
    record = _base_record(seed, source=FORMAL_SOURCE, status="accepted", reason=None)
    record.update(
        {
            "eligible_for_risk_selection": True,
            "result": {
                "boundary": {"boundary_type": "grasp_lift", "control_step": takeover},
                "snapshot_round_trip": {"passed": True},
                "trajectory": {
                    "trajectory_id": f"accepted-{seed}",
                    "num_steps": num_steps,
                    "local_dagger": {
                        "boundary_type": "grasp_lift",
                        "boundary_detection_step": takeover,
                        "expert_takeover_step": takeover,
                        "training_window_start": takeover,
                        "training_window_end": takeover + 64,
                        "rollin_seed": seed,
                        "rollin_policy_checkpoint_sha256": CHECKPOINT_SHA,
                        "expert_recovery_success": True,
                    },
                    "outcome_evidence": {"task_completed": True},
                },
            },
        }
    )
    return record, takeover, num_steps


def _replan_traces(
    policy_steps: int, *, reach_step: int | None, grasp_step: int | None
) -> list[dict]:
    traces = []
    control_step = 0
    index = 0
    while control_step < policy_steps:
        executed = min(4, policy_steps - control_step)

        def completed(step: int) -> int:
            if grasp_step is not None and step >= grasp_step:
                return 2
            if reach_step is not None and step >= reach_step:
                return 1
            return 0

        traces.append(
            {
                "replan_index": index,
                "control_step": control_step,
                "sampling_seed": 9_000 + index,
                "executed_steps": executed,
                "completed_skill_count_before": completed(control_step),
                "completed_skill_count_after": completed(control_step + executed),
                "temporal_buffer_size": min(index + 1, 4),
                "temporal_max_proposal_spread": 0.1,
                "replan_required": False,
            }
        )
        control_step += executed
        index += 1
    return traces


def _diagnostic_common(
    *,
    seed: int,
    reason: str,
    action_count: int,
    phase_transitions: list[dict],
    phase_action_counts: dict[str, int],
    max_completed: int,
    skill_steps: dict[str, int | None],
    segments: list[dict[str, int]],
    loss_events: int,
    max_stable: int,
    final_raw: bool,
    terminated: bool,
    truncated: bool,
    action_source: int,
    policy_steps: int,
) -> dict:
    raw_count = sum(
        segment["end_action_step_exclusive"] - segment["start_action_step"] for segment in segments
    )
    max_run = max(
        (
            segment["end_action_step_exclusive"] - segment["start_action_step"]
            for segment in segments
        ),
        default=0,
    )
    boundary_step = skill_steps["grasp"]
    replans = _replan_traces(
        policy_steps,
        reach_step=skill_steps["reach"],
        grasp_step=skill_steps["grasp"],
    )
    return {
        "format": LOCAL_DAGGER_DIAGNOSTIC_FORMAT,
        "environment_seed": seed,
        "boundary_type": "grasp_lift",
        "failure_reason": reason,
        "observation_scope": "local_dagger_collection_session",
        "action_count": action_count,
        "failure_control_step": action_count,
        "boundary_skill": "grasp",
        "boundary_reached": boundary_step is not None,
        "boundary_detection_step": boundary_step,
        "expert_takeover_step": boundary_step,
        "phase_at_failure": phase_transitions[-1]["phase"],
        "phase_action_counts": phase_action_counts,
        "phase_transitions": phase_transitions,
        "skill_completion_steps": skill_steps,
        "max_completed_skill_count": max_completed,
        "max_stable_grasp_steps": max_stable,
        "ever_raw_grasped": raw_count > 0,
        "raw_grasp_action_count": raw_count,
        "raw_grasp_loss_events": loss_events,
        "raw_grasp_segments": segments,
        "max_consecutive_raw_grasp_steps": max_run,
        "first_raw_grasp_action_step": None if not segments else segments[0]["start_action_step"],
        "last_raw_grasp_action_step": (
            None if not segments else segments[-1]["end_action_step_exclusive"] - 1
        ),
        "raw_grasp_rising_edge_count": len(segments),
        "ever_lifted": False,
        "ever_transported": False,
        "min_tcp_to_object_distance_m": 0.01,
        "max_object_height_above_support_m": 0.05,
        "max_object_linear_speed_m_s": 0.2,
        "max_object_angular_speed_rad_s": 0.3,
        "final_progress": {
            "active_skill_id": max_completed,
            "active_skill_name": "grasp" if max_completed == 1 else "lift",
            "completed_skill_count": max_completed,
            "stable_grasp_steps": max_stable,
            "task_completed": False,
            "reached": max_completed >= 1,
            "raw_grasped": final_raw,
            "lifted": False,
            "transported": False,
            "tcp_to_object_distance_m": 0.01,
            "object_height_above_support_m": 0.05,
            "object_linear_speed_m_s": 0.2,
            "object_angular_speed_rad_s": 0.3,
        },
        "final_transition": {
            "action_step": action_count,
            "phase": phase_transitions[-1]["phase"],
            "terminated": terminated,
            "truncated": truncated,
            "environment_success": False,
            "action_source": action_source,
            "gripper_opening": 0.0,
        },
        "policy_replan_count": len(replans),
        "policy_replan_required_count": 0,
        "policy_replan_traces": replans,
    }


def _policy_diagnostics(seed: int, variant: int) -> dict:
    skills = {skill: None for skill in ("reach", "grasp", "lift", "transport", "place")}
    segments: list[dict[str, int]] = []
    loss_events = 0
    max_stable = 0
    final_raw = False
    if variant == 0:
        action_count = 300
        terminated, truncated = False, True
    elif variant == 1:
        action_count = 120
        terminated, truncated = True, False
        skills["reach"] = 40
    elif variant == 2:
        action_count = 120
        terminated, truncated = True, False
        skills["reach"] = 40
        segments = [{"start_action_step": 60, "end_action_step_exclusive": 62}]
        loss_events = 1
        max_stable = 1
    else:
        action_count = 300
        terminated, truncated = False, True
        skills["reach"] = 40
        segments = [{"start_action_step": 299, "end_action_step_exclusive": 301}]
        max_stable = 1
        final_raw = True
    completed = int(skills["reach"] is not None)
    return _diagnostic_common(
        seed=seed,
        reason=POLICY_BEFORE_BOUNDARY_REASON,
        action_count=action_count,
        phase_transitions=[{"action_step": 0, "phase": POLICY_ROLLIN_PHASE}],
        phase_action_counts={POLICY_ROLLIN_PHASE: action_count},
        max_completed=completed,
        skill_steps=skills,
        segments=segments,
        loss_events=loss_events,
        max_stable=max_stable,
        final_raw=final_raw,
        terminated=terminated,
        truncated=truncated,
        action_source=0,
        policy_steps=action_count,
    )


def _expert_diagnostics(seed: int, final_phase: str) -> dict:
    transition_plan = [
        (0, POLICY_ROLLIN_PHASE),
        (100, "expert_grasp_stabilization"),
        (108, EXPERT_LIFT_MOTION_PHASE),
        (150, EXPERT_TRANSPORT_MOTION_PHASE),
        (200, EXPERT_LOWER_MOTION_PHASE),
        (250, EXPERT_RELEASE_SETTLE_PHASE),
    ]
    final_index = next(
        index for index, (_, phase) in enumerate(transition_plan) if phase == final_phase
    )
    transitions = [
        {"action_step": step, "phase": phase} for step, phase in transition_plan[: final_index + 1]
    ]
    counts = {}
    for index, transition in enumerate(transitions):
        end = transitions[index + 1]["action_step"] if index + 1 < len(transitions) else 300
        counts[transition["phase"]] = end - transition["action_step"]
    skills = {
        "reach": 50,
        "grasp": 100,
        "lift": None,
        "transport": None,
        "place": None,
    }
    return _diagnostic_common(
        seed=seed,
        reason=EPISODE_TIME_LIMIT_REASON,
        action_count=300,
        phase_transitions=transitions,
        phase_action_counts=counts,
        max_completed=2,
        skill_steps=skills,
        segments=[{"start_action_step": 99, "end_action_step_exclusive": 301}],
        loss_events=0,
        max_stable=202,
        final_raw=True,
        terminated=False,
        truncated=True,
        action_source=1,
        policy_steps=100,
    )


def _build_fixture(root: Path) -> tuple[Path, Path]:
    formal_root = root / "formal-gl"
    replay_root = root / "replay-gl"
    formal_root.mkdir(parents=True)
    replay_root.mkdir(parents=True)
    formal_experiment = {
        "format": POOL_FORMAT,
        "source_revision": FORMAL_SOURCE,
        "checkpoint": _checkpoint(),
        "base_dataset": {"path": "/frozen/d0", "audit": {"dataset_sha256": "d" * 64}},
        "model_cache": "/frozen/qwen",
        "boundary_type": "grasp_lift",
        "environment_seeds": list(FORMAL_GL_SEEDS),
        "config": _pool_config(),
    }
    _write_json(formal_root / "experiment.json", formal_experiment)

    formal_rows: list[dict] = []
    formal_records: dict[int, tuple[dict, Path]] = {}
    for index, seed in enumerate(FORMAL_GL_SEEDS):
        record_path = formal_root / "candidates" / f"seed-{seed:06d}" / "record.json"
        if index < 10:
            record, takeover, _ = _accepted_record(seed, index)
            row = {
                "environment_seed": seed,
                "boundary_type": "grasp_lift",
                "status": "accepted",
                "record": str(record_path.resolve()),
                "episode_sampling_seed": _episode_config(seed)["episode_sampling_seed"],
                "eligible_for_risk_selection": True,
                "expert_takeover_step": takeover,
                "snapshot_round_trip_passed": True,
                "trajectory_audit": "passed",
            }
        elif index < 81:
            record = _base_record(
                seed,
                source=FORMAL_SOURCE,
                status="rejected",
                reason=POLICY_BEFORE_BOUNDARY_REASON,
            )
            row = {
                "environment_seed": seed,
                "boundary_type": "grasp_lift",
                "status": "rejected",
                "record": str(record_path.resolve()),
                "episode_sampling_seed": _episode_config(seed)["episode_sampling_seed"],
                "eligible_for_risk_selection": False,
                "failure": record["failure"],
            }
        elif index < 97:
            record = _base_record(
                seed,
                source=FORMAL_SOURCE,
                status="rejected",
                reason=EPISODE_TIME_LIMIT_REASON,
            )
            row = {
                "environment_seed": seed,
                "boundary_type": "grasp_lift",
                "status": "rejected",
                "record": str(record_path.resolve()),
                "episode_sampling_seed": _episode_config(seed)["episode_sampling_seed"],
                "eligible_for_risk_selection": False,
                "failure": record["failure"],
            }
        else:
            reasons = (
                EXPERT_INCOMPLETE_REASON,
                MPLIB_PATH_REASON,
                SNAPSHOT_REASON_PREFIX + "synthetic",
            )
            record = _base_record(
                seed,
                source=FORMAL_SOURCE,
                status="rejected",
                reason=reasons[index - 97],
            )
            row = {
                "environment_seed": seed,
                "boundary_type": "grasp_lift",
                "status": "rejected",
                "record": str(record_path.resolve()),
                "episode_sampling_seed": _episode_config(seed)["episode_sampling_seed"],
                "eligible_for_risk_selection": False,
                "failure": record["failure"],
            }
        _write_json(record_path, record)
        formal_records[seed] = (record, record_path)
        formal_rows.append(row)
    _write_jsonl(formal_root / "collection_candidates.jsonl", formal_rows)

    target_seeds = list(FORMAL_GL_SEEDS[10:97])
    selected_candidates = []
    for seed in target_seeds:
        formal_record, formal_path = formal_records[seed]
        selected_candidates.append(
            {
                "environment_seed": seed,
                "boundary_type": "grasp_lift",
                "status": formal_record["status"],
                "failure": formal_record["failure"],
                "source_revision": formal_record["source_revision"],
                "base_dataset": formal_record["base_dataset"],
                "checkpoint": formal_record["checkpoint"],
                "config": formal_record["config"],
                "record": str(formal_path.resolve()),
                "record_sha256": _sha256(formal_path),
            }
        )
    replay_experiment = {
        "format": REPLAY_FORMAT,
        "purpose": "exploratory GL failure decomposition; not training data",
        "source_revision": REPLAY_SOURCE,
        "replay_source_revision": REPLAY_SOURCE,
        "formal_source_revision": FORMAL_SOURCE,
        "checkpoint": _checkpoint(),
        "base_dataset": formal_experiment["base_dataset"],
        "model_cache": "/frozen/qwen",
        "config": _pool_config(),
        "environment_seeds": target_seeds,
        "target_boundary_type": "grasp_lift",
        "target_failure_reasons": [
            POLICY_BEFORE_BOUNDARY_REASON,
            EPISODE_TIME_LIMIT_REASON,
        ],
        "selected_count": 87,
        "selected_candidates": selected_candidates,
        "formal_pool": {
            "path": str(formal_root.resolve()),
            "experiment": formal_experiment,
            "experiment_sha256": _sha256(formal_root / "experiment.json"),
            "collection_candidates": str((formal_root / "collection_candidates.jsonl").resolve()),
            "collection_candidates_sha256": _sha256(formal_root / "collection_candidates.jsonl"),
        },
        "execution": {},
    }
    _write_json(replay_root / "experiment.json", replay_experiment)

    replay_rows: list[dict] = []
    phases = (
        EXPERT_LIFT_MOTION_PHASE,
        EXPERT_TRANSPORT_MOTION_PHASE,
        EXPERT_LOWER_MOTION_PHASE,
        EXPERT_RELEASE_SETTLE_PHASE,
    )
    for target_index, seed in enumerate(target_seeds):
        formal_record, formal_path = formal_records[seed]
        reason = formal_record["failure"]["reason"]
        replay_record = _base_record(
            seed,
            source=REPLAY_SOURCE,
            status="rejected",
            reason=reason,
        )
        if reason == POLICY_BEFORE_BOUNDARY_REASON:
            diagnostics = _policy_diagnostics(seed, target_index % 4)
        else:
            diagnostics = _expert_diagnostics(seed, phases[(target_index - 71) % 4])
        replay_record["failure_diagnostics"] = diagnostics
        replay_path = replay_root / "candidates" / f"seed-{seed:06d}" / "record.json"
        _write_json(replay_path, replay_record)
        reconciliation = {
            "classification": "matched",
            "reconciled": True,
            "exact_match": True,
            "status_matches": True,
            "reason_matches": True,
            "failure_type_matches": True,
            "config_matches": True,
            "checkpoint_matches": True,
            "base_dataset_matches": True,
            "source_revision_matches": True,
            "identity_contract_matches": True,
            "failure_diagnostics_matches": True,
            "failure_record_contract_matches": True,
            "subprocess_contract_matches": True,
            "subprocess_returncode": 1,
        }
        replay_rows.append(
            {
                "format": "robot-vla-local-dagger-failure-replay-candidate/v1",
                "environment_seed": seed,
                "boundary_type": "grasp_lift",
                "status": "rejected",
                "failure": replay_record["failure"],
                "record": str(replay_path.resolve()),
                "failure_diagnostics": diagnostics,
                "source_revision": REPLAY_SOURCE,
                "config": replay_record["config"],
                "checkpoint": replay_record["checkpoint"],
                "base_dataset": replay_record["base_dataset"],
                "reconciled": True,
                "original": {
                    "status": formal_record["status"],
                    "failure": formal_record["failure"],
                    "record": str(formal_path.resolve()),
                    "record_sha256": _sha256(formal_path),
                    "source_revision": FORMAL_SOURCE,
                },
                "reconciliation": reconciliation,
            }
        )
    _write_jsonl(replay_root / "replay_candidates.jsonl", replay_rows)
    return formal_root, replay_root


@pytest.fixture
def synthetic_roots(tmp_path: Path) -> tuple[Path, Path]:
    return _build_fixture(tmp_path)


def test_analyze_builds_additive_auditable_decomposition(
    synthetic_roots: tuple[Path, Path],
) -> None:
    formal_root, replay_root = synthetic_roots

    result = analyze_gl_failure_diagnostics(formal_root, replay_root)

    assert result["format"] == ANALYSIS_FORMAT
    assert result["canonical_formal_population"]["count"] == 100
    assert result["canonical_formal_population"]["status_reason_counts"] == {
        "accepted": 10,
        "policy_before_stable_grasp_boundary": 71,
        "episode_time_limit_after_takeover": 16,
        "expert_incomplete_pick_and_place": 1,
        "mplib_no_trusted_screw_path": 1,
        "snapshot_round_trip": 1,
    }
    assert result["boundary_reach_lower_bound"]["qualifier"] == "at_least"
    assert result["boundary_reach_lower_bound"]["count"] == 29
    policy = result["failure_decomposition"]["policy_before_stable_grasp_boundary"]
    assert policy["total"] == 71
    assert policy["additivity"]["passed"] is True
    assert len(policy["progress_by_terminal"]) >= 4
    expert = result["failure_decomposition"]["expert_time_limit_after_takeover"]
    assert expert["total"] == 16
    assert [row["count"] for row in expert["commanded_phase"]] == [4, 4, 4, 4]
    timing = result["accepted_survivor_timing"]
    assert timing["expert_takeover_step"] == {"min": 80, "median": 125, "max": 170}
    assert timing["takeover_to_full_success_steps"] == {
        "min": 120,
        "median": 124.5,
        "max": 129,
    }
    assert result["reconciliation"]["matched_count"] == 87
    assert result["reconciliation"]["exact_match"] is True
    assert len(result["unidentifiable_items"]) >= 4
    assert "survivors" in result["survivorship_caveat"]


def test_rejects_duplicate_formal_seed(synthetic_roots: tuple[Path, Path]) -> None:
    formal_root, replay_root = synthetic_roots
    path = formal_root / "collection_candidates.jsonl"
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    path.write_text(path.read_text(encoding="utf-8") + first_line + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="重复 seed"):
        analyze_gl_failure_diagnostics(formal_root, replay_root)


def test_rejects_missing_replay_target(synthetic_roots: tuple[Path, Path]) -> None:
    formal_root, replay_root = synthetic_roots
    path = replay_root / "replay_candidates.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="应为 87"):
        analyze_gl_failure_diagnostics(formal_root, replay_root)


def test_rejects_per_seed_reason_mismatch(synthetic_roots: tuple[Path, Path]) -> None:
    formal_root, replay_root = synthetic_roots
    seed = FORMAL_GL_SEEDS[10]
    path = replay_root / "candidates" / f"seed-{seed:06d}" / "record.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["failure"]["reason"] = EPISODE_TIME_LIMIT_REASON
    _write_json(path, record)

    with pytest.raises(ValueError, match="failure 未逐 seed 对齐"):
        analyze_gl_failure_diagnostics(formal_root, replay_root)


def test_rejects_action_continuity_mismatch(synthetic_roots: tuple[Path, Path]) -> None:
    formal_root, replay_root = synthetic_roots
    seed = FORMAL_GL_SEEDS[10]
    record_path = replay_root / "candidates" / f"seed-{seed:06d}" / "record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["failure_diagnostics"]["phase_action_counts"][POLICY_ROLLIN_PHASE] -= 1
    _write_json(record_path, record)
    rows_path = replay_root / "replay_candidates.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    row = next(item for item in rows if item["environment_seed"] == seed)
    row["failure_diagnostics"] = record["failure_diagnostics"]
    _write_jsonl(rows_path, rows)

    with pytest.raises(ValueError, match="phase_action_counts"):
        analyze_gl_failure_diagnostics(formal_root, replay_root)


def test_cli_writes_requested_json(
    synthetic_roots: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    formal_root, replay_root = synthetic_roots
    output = tmp_path / "result" / "decomposition.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_e012_gl_failure_diagnostics.py",
            "--formal-root",
            str(formal_root),
            "--replay-root",
            str(replay_root),
            "--output",
            str(output),
        ],
    )

    main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["format"] == ANALYSIS_FORMAT
    assert payload["reconciliation"]["exact_match"] is True
    assert "e012_gl_failure_diagnostics_analyzed" in capsys.readouterr().out
