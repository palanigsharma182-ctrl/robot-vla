#!/usr/bin/env python3
"""聚合并严格审计 E012 Grasp→Lift 失败诊断重放。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from robot_vla.contracts import PICK_AND_PLACE_SKILLS
from robot_vla.sim.local_dagger_diagnostics import (
    EPISODE_TIME_LIMIT_REASON,
    LOCAL_DAGGER_DIAGNOSTIC_FORMAT,
    LOCAL_DAGGER_DIAGNOSTIC_PHASES,
    POLICY_BEFORE_BOUNDARY_REASON,
    POLICY_ROLLIN_PHASE,
    classify_grasp_lift_failure,
)

ANALYSIS_FORMAT = "robot-vla-e012-gl-failure-decomposition/v1"
POOL_FORMAT = "robot-vla-local-dagger-pool/v1"
COLLECTION_FORMAT = "robot-vla-local-dagger-collection/v1"
REPLAY_FORMAT = "robot-vla-local-dagger-failure-replay/v1"

FORMAL_GL_SEEDS = tuple(range(30_100, 30_200))
TARGET_FAILURE_REASONS = (
    POLICY_BEFORE_BOUNDARY_REASON,
    EPISODE_TIME_LIMIT_REASON,
)
EXPECTED_CANONICAL_COUNTS = {
    "accepted": 10,
    "policy_before_stable_grasp_boundary": 71,
    "episode_time_limit_after_takeover": 16,
    "expert_incomplete_pick_and_place": 1,
    "mplib_no_trusted_screw_path": 1,
    "snapshot_round_trip": 1,
}

EXPERT_INCOMPLETE_REASON = "Local DAgger Expert 未完成完整 Pick-and-Place"
MPLIB_PATH_REASON = "MPlib 无法规划可信 screw 路径"
SNAPSHOT_REASON_PREFIX = "Boundary snapshot round-trip 未通过："
POST_BOUNDARY_REASON_CODES = {
    "expert_incomplete_pick_and_place",
    "mplib_no_trusted_screw_path",
    "snapshot_round_trip",
}


def _fail(message: str) -> None:
    raise ValueError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON 不允许非有限常量: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )
    _require(isinstance(value, dict), f"{path}: 顶层必须是 JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: JSONL 无效") from error
        _require(isinstance(row, dict), f"{path}:{line_number}: row 必须是 object")
        rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{label} 必须是 int")
    resolved = int(value)
    if minimum is not None:
        _require(resolved >= minimum, f"{label} 必须 >= {minimum}")
    return resolved


def _strict_bool(value: Any, label: str) -> bool:
    _require(isinstance(value, bool), f"{label} 必须是 bool")
    return bool(value)


def _finite_number(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} 必须是有限数值",
    )
    return float(value)


def _index_rows(
    rows: Iterable[dict[str, Any]],
    *,
    label: str,
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        seed = _strict_int(
            row.get("environment_seed"), f"{label}[{row_number}].environment_seed", minimum=0
        )
        _require(seed not in indexed, f"{label}: 重复 seed {seed}")
        indexed[seed] = row
    return indexed


def _candidate_record_path(root: Path, seed: int) -> Path:
    path = root / "candidates" / f"seed-{seed:06d}" / "record.json"
    if not path.is_file():
        raise FileNotFoundError(f"seed {seed}: 找不到 candidate record: {path}")
    return path


def _validate_record_reference(row: dict[str, Any], expected: Path, label: str) -> None:
    reference = row.get("record")
    _require(isinstance(reference, str) and reference, f"{label}.record 缺失")
    reference_path = Path(reference)
    if reference_path.is_file():
        _require(
            reference_path.resolve() == expected.resolve(),
            f"{label}.record 指向了其他文件",
        )
    else:
        expected_suffix = Path("candidates") / expected.parent.name / "record.json"
        _require(
            tuple(reference_path.parts[-3:]) == tuple(expected_suffix.parts),
            f"{label}.record 既不可访问，也不是预期 candidate 后缀",
        )


def _canonical_reason(status: str, failure_reason: str | None) -> str:
    if status == "accepted":
        _require(failure_reason is None, "accepted candidate 不应有 failure reason")
        return "accepted"
    _require(status == "rejected", f"不支持的 formal status: {status}")
    _require(
        isinstance(failure_reason, str) and failure_reason, "rejected candidate 缺 failure reason"
    )
    if failure_reason == POLICY_BEFORE_BOUNDARY_REASON:
        return "policy_before_stable_grasp_boundary"
    if failure_reason == EPISODE_TIME_LIMIT_REASON:
        return "episode_time_limit_after_takeover"
    if failure_reason == EXPERT_INCOMPLETE_REASON:
        return "expert_incomplete_pick_and_place"
    if failure_reason == MPLIB_PATH_REASON:
        return "mplib_no_trusted_screw_path"
    if failure_reason.startswith(SNAPSHOT_REASON_PREFIX):
        return "snapshot_round_trip"
    _fail(f"未登记的 formal GL failure reason: {failure_reason}")
    raise AssertionError("unreachable")


def _record_failure(record: dict[str, Any], label: str) -> dict[str, str] | None:
    status = record.get("status")
    _require(status in {"accepted", "rejected"}, f"{label}.status 必须是 accepted/rejected")
    failure = record.get("failure")
    if status == "accepted":
        _require(failure is None, f"{label}: accepted record 不应有 failure")
        return None
    _require(isinstance(failure, dict), f"{label}: rejected record 缺 failure")
    _require(
        isinstance(failure.get("type"), str) and failure["type"],
        f"{label}.failure.type 缺失",
    )
    _require(
        isinstance(failure.get("reason"), str) and failure["reason"],
        f"{label}.failure.reason 缺失",
    )
    return {"type": str(failure["type"]), "reason": str(failure["reason"])}


def _validate_pool_experiment(experiment: dict[str, Any]) -> None:
    _require(experiment.get("format") == POOL_FORMAT, "formal experiment format 不兼容")
    _require(experiment.get("boundary_type") == "grasp_lift", "formal root 不是 grasp_lift")
    seeds = experiment.get("environment_seeds")
    _require(isinstance(seeds, list), "formal experiment.environment_seeds 缺失")
    resolved = tuple(
        _strict_int(value, f"formal experiment.environment_seeds[{index}]", minimum=0)
        for index, value in enumerate(seeds)
    )
    _require(len(set(resolved)) == len(resolved), "formal experiment seeds 重复")
    _require(resolved == FORMAL_GL_SEEDS, "formal GL seeds 必须完整等于 30100..30199")
    _require(
        isinstance(experiment.get("source_revision"), str) and experiment["source_revision"],
        "formal source_revision 缺失",
    )
    checkpoint = experiment.get("checkpoint")
    _require(isinstance(checkpoint, dict), "formal checkpoint 缺失")
    checkpoint_sha = checkpoint.get("sha256")
    _require(
        isinstance(checkpoint_sha, str)
        and len(checkpoint_sha) == 64
        and all(character in "0123456789abcdef" for character in checkpoint_sha),
        "formal checkpoint sha256 无效",
    )
    config = experiment.get("config")
    _require(isinstance(config, dict), "formal config 缺失")
    expected_config = {
        "inference_strategy": "temporal-ensemble",
        "qwen_context_layer": 12,
        "sampling_seed_base": 52_012,
        "num_flow_steps": 10,
        "recency_decay": 0.5,
        "max_anomaly_replans": 3,
        "snapshot_round_trip_required": True,
        "paired_clean_expert_required": True,
    }
    _require(config == expected_config, "formal GL frozen config 漂移")


def _validate_collection_record_identity(
    record: dict[str, Any],
    *,
    experiment: dict[str, Any],
    seed: int,
    label: str,
) -> None:
    _require(record.get("format") == COLLECTION_FORMAT, f"{label}.format 不兼容")
    _require(
        record.get("source_revision") == experiment["source_revision"],
        f"{label}.source_revision 漂移",
    )
    checkpoint = record.get("checkpoint")
    _require(isinstance(checkpoint, dict), f"{label}.checkpoint 缺失")
    _require(
        checkpoint.get("sha256") == experiment["checkpoint"]["sha256"],
        f"{label}.checkpoint sha256 漂移",
    )
    config = record.get("config")
    _require(isinstance(config, dict), f"{label}.config 缺失")
    expected = experiment["config"]
    checks = {
        "environment_seed": seed,
        "boundary_type": "grasp_lift",
        "sampling_seed_base": expected["sampling_seed_base"],
        "num_flow_steps": expected["num_flow_steps"],
        "recency_decay": expected["recency_decay"],
        "max_anomaly_replans": expected["max_anomaly_replans"],
        "qwen_context_layer": expected["qwen_context_layer"],
        "snapshot_round_trip_required": True,
        "paired_clean_expert_required": True,
    }
    for key, expected_value in checks.items():
        _require(config.get(key) == expected_value, f"{label}.config.{key} 漂移")
    _strict_int(
        config.get("episode_sampling_seed"), f"{label}.config.episode_sampling_seed", minimum=0
    )


def _validate_accepted_record(
    record: dict[str, Any], row: dict[str, Any], seed: int
) -> dict[str, int]:
    label = f"formal seed {seed}"
    result = record.get("result")
    _require(isinstance(result, dict), f"{label}.result 缺失")
    trajectory = result.get("trajectory")
    _require(isinstance(trajectory, dict), f"{label}.result.trajectory 缺失")
    local_dagger = trajectory.get("local_dagger")
    _require(isinstance(local_dagger, dict), f"{label}.local_dagger 缺失")
    takeover = _strict_int(
        local_dagger.get("expert_takeover_step"),
        f"{label}.expert_takeover_step",
        minimum=1,
    )
    num_steps = _strict_int(trajectory.get("num_steps"), f"{label}.num_steps", minimum=1)
    _require(num_steps > takeover, f"{label}: success 必须发生在 takeover 之后")
    _require(local_dagger.get("boundary_type") == "grasp_lift", f"{label}: boundary_type 漂移")
    _require(local_dagger.get("rollin_seed") == seed, f"{label}: rollin_seed 漂移")
    _require(
        local_dagger.get("rollin_policy_checkpoint_sha256") == record["checkpoint"]["sha256"],
        f"{label}: provenance checkpoint 漂移",
    )
    _require(
        local_dagger.get("boundary_detection_step") == takeover,
        f"{label}: boundary/takeover 不同步",
    )
    _require(
        local_dagger.get("training_window_start") == takeover,
        f"{label}: training window 未从 takeover 开始",
    )
    _require(
        local_dagger.get("expert_recovery_success") is True, f"{label}: expert recovery 未成功"
    )
    boundary = result.get("boundary")
    _require(isinstance(boundary, dict), f"{label}.boundary 缺失")
    _require(boundary.get("control_step") == takeover, f"{label}: boundary.control_step 不同步")
    evidence = trajectory.get("outcome_evidence")
    _require(
        isinstance(evidence, dict) and evidence.get("task_completed") is True,
        f"{label}: 缺完整成功证据",
    )
    _require(row.get("expert_takeover_step") == takeover, f"{label}: index takeover 漂移")
    _require(row.get("snapshot_round_trip_passed") is True, f"{label}: snapshot gate 未通过")
    _require(row.get("trajectory_audit") == "passed", f"{label}: trajectory audit 未通过")
    return {
        "environment_seed": seed,
        "expert_takeover_step": takeover,
        "num_steps": num_steps,
        "takeover_to_success_steps": num_steps - takeover,
    }


def _load_formal(formal_root: Path) -> dict[str, Any]:
    experiment = _load_json(formal_root / "experiment.json")
    _validate_pool_experiment(experiment)
    rows = _load_jsonl(formal_root / "collection_candidates.jsonl")
    indexed = _index_rows(rows, label="formal collection_candidates")
    _require(len(indexed) == 100, f"formal candidates 应为 100，实际 {len(indexed)}")
    _require(set(indexed) == set(FORMAL_GL_SEEDS), "formal candidates seeds 不完整")

    records: dict[int, dict[str, Any]] = {}
    record_paths: dict[int, Path] = {}
    canonical_candidates: list[dict[str, Any]] = []
    accepted_timing: list[dict[str, int]] = []
    canonical_counts: Counter[str] = Counter()
    target_seeds: list[int] = []

    for seed in FORMAL_GL_SEEDS:
        row = indexed[seed]
        label = f"formal seed {seed}"
        _require(row.get("boundary_type") == "grasp_lift", f"{label}: index boundary_type 漂移")
        path = _candidate_record_path(formal_root, seed)
        _validate_record_reference(row, path, label)
        record = _load_json(path)
        _validate_collection_record_identity(
            record,
            experiment=experiment,
            seed=seed,
            label=label,
        )
        failure = _record_failure(record, label)
        status = str(record["status"])
        _require(row.get("status") == status, f"{label}: index/record status 不一致")
        if failure is None:
            _require(row.get("failure") is None, f"{label}: accepted index 不应有 failure")
            _require(
                row.get("eligible_for_risk_selection") is True,
                f"{label}: accepted formal candidate 必须 eligible",
            )
            timing = _validate_accepted_record(record, row, seed)
            accepted_timing.append(timing)
            raw_reason = None
        else:
            _require(row.get("failure") == failure, f"{label}: index/record failure 不一致")
            _require(
                row.get("eligible_for_risk_selection") is False, f"{label}: rejected 不应 eligible"
            )
            raw_reason = failure["reason"]
            if raw_reason in TARGET_FAILURE_REASONS:
                target_seeds.append(seed)
        _require(
            row.get("episode_sampling_seed") == record["config"]["episode_sampling_seed"],
            f"{label}: episode_sampling_seed 漂移",
        )
        code = _canonical_reason(status, raw_reason)
        canonical_counts[code] += 1
        canonical_candidates.append(
            {
                "environment_seed": seed,
                "status": status,
                "reason_code": code,
                "failure_reason": raw_reason,
            }
        )
        records[seed] = record
        record_paths[seed] = path

    _require(
        dict(canonical_counts) == EXPECTED_CANONICAL_COUNTS,
        "formal GL canonical profile 漂移: "
        f"expected={EXPECTED_CANONICAL_COUNTS}, actual={dict(canonical_counts)}",
    )
    _require(len(target_seeds) == 87, "formal target failure seeds 必须是 87")
    return {
        "experiment": experiment,
        "rows": indexed,
        "records": records,
        "record_paths": record_paths,
        "canonical_candidates": canonical_candidates,
        "canonical_counts": dict(canonical_counts),
        "target_seeds": tuple(target_seeds),
        "accepted_timing": accepted_timing,
    }


def _validate_phase_and_action_continuity(diagnostics: dict[str, Any], label: str) -> None:
    action_count = _strict_int(diagnostics.get("action_count"), f"{label}.action_count", minimum=1)
    _require(
        diagnostics.get("failure_control_step") == action_count,
        f"{label}.failure_control_step 与 action_count 不一致",
    )
    phase_at_failure = diagnostics.get("phase_at_failure")
    _require(phase_at_failure in LOCAL_DAGGER_DIAGNOSTIC_PHASES, f"{label}.phase_at_failure 无效")

    transitions = diagnostics.get("phase_transitions")
    _require(isinstance(transitions, list) and transitions, f"{label}.phase_transitions 缺失")
    normalized: list[tuple[int, str]] = []
    previous_step = -1
    previous_phase_index = -1
    for index, transition in enumerate(transitions):
        _require(isinstance(transition, dict), f"{label}.phase_transitions[{index}] 无效")
        step = _strict_int(
            transition.get("action_step"),
            f"{label}.phase_transitions[{index}].action_step",
            minimum=0,
        )
        phase = transition.get("phase")
        _require(
            phase in LOCAL_DAGGER_DIAGNOSTIC_PHASES,
            f"{label}.phase_transitions[{index}].phase 无效",
        )
        phase_index = LOCAL_DAGGER_DIAGNOSTIC_PHASES.index(phase)
        _require(step >= previous_step, f"{label}: phase action_step 倒退")
        _require(phase_index > previous_phase_index, f"{label}: phase 必须单调前进且不重复")
        _require(step <= action_count, f"{label}: phase transition 超过 action_count")
        normalized.append((step, str(phase)))
        previous_step = step
        previous_phase_index = phase_index
    _require(
        normalized[0] == (0, POLICY_ROLLIN_PHASE),
        f"{label}: 首个 phase 必须是 step 0 policy_rollin",
    )
    _require(
        normalized[-1][1] == phase_at_failure, f"{label}: phase_at_failure 与最后 transition 不一致"
    )

    derived_counts: Counter[str] = Counter()
    for index, (start, phase) in enumerate(normalized):
        end = normalized[index + 1][0] if index + 1 < len(normalized) else action_count
        _require(end >= start, f"{label}: phase duration 为负")
        if end > start:
            derived_counts[phase] += end - start
    raw_counts = diagnostics.get("phase_action_counts")
    _require(isinstance(raw_counts, dict), f"{label}.phase_action_counts 缺失")
    observed_counts: dict[str, int] = {}
    for phase, value in raw_counts.items():
        _require(phase in LOCAL_DAGGER_DIAGNOSTIC_PHASES, f"{label}: 未知 phase_action_counts key")
        count = _strict_int(value, f"{label}.phase_action_counts.{phase}", minimum=0)
        if count:
            observed_counts[str(phase)] = count
    _require(
        observed_counts == dict(derived_counts),
        f"{label}: phase_action_counts 与 transition 不一致",
    )
    _require(
        sum(observed_counts.values()) == action_count,
        f"{label}: phase actions 未加总到 action_count",
    )

    final_transition = diagnostics.get("final_transition")
    _require(isinstance(final_transition, dict), f"{label}.final_transition 缺失")
    _require(
        final_transition.get("action_step") == action_count, f"{label}: final action_step 不连续"
    )
    _require(
        final_transition.get("phase") == phase_at_failure, f"{label}: final transition phase 漂移"
    )

    replans = diagnostics.get("policy_replan_traces")
    _require(isinstance(replans, list) and replans, f"{label}.policy_replan_traces 缺失")
    _require(
        diagnostics.get("policy_replan_count") == len(replans),
        f"{label}.policy_replan_count 不一致",
    )
    _require(
        diagnostics.get("policy_replan_required_count")
        == sum(bool(trace.get("replan_required")) for trace in replans),
        f"{label}.policy_replan_required_count 不一致",
    )
    policy_actions = 0
    previous_completed_after: int | None = None
    for index, trace in enumerate(replans):
        _require(isinstance(trace, dict), f"{label}.policy_replan_traces[{index}] 无效")
        _require(trace.get("replan_index") == index, f"{label}: replan_index 不连续")
        _require(
            trace.get("control_step") == policy_actions, f"{label}: replan control_step 不连续"
        )
        executed = _strict_int(
            trace.get("executed_steps"), f"{label}.replan[{index}].executed_steps", minimum=1
        )
        before = _strict_int(
            trace.get("completed_skill_count_before"),
            f"{label}.replan[{index}].completed_before",
            minimum=0,
        )
        after = _strict_int(
            trace.get("completed_skill_count_after"),
            f"{label}.replan[{index}].completed_after",
            minimum=0,
        )
        _require(after >= before, f"{label}: replan skill progress 倒退")
        _strict_bool(trace.get("replan_required"), f"{label}.replan[{index}].replan_required")
        sampling_seed = trace.get("sampling_seed")
        if sampling_seed is not None:
            _strict_int(sampling_seed, f"{label}.replan[{index}].sampling_seed", minimum=0)
        buffer_size = trace.get("temporal_buffer_size")
        if buffer_size is not None:
            _strict_int(buffer_size, f"{label}.replan[{index}].temporal_buffer_size", minimum=1)
        spread = trace.get("temporal_max_proposal_spread")
        if spread is not None:
            _finite_number(spread, f"{label}.replan[{index}].temporal_max_proposal_spread")
        if previous_completed_after is not None:
            _require(before == previous_completed_after, f"{label}: replan skill progress 不连续")
        previous_completed_after = after
        policy_actions += executed
    _require(
        policy_actions == observed_counts.get(POLICY_ROLLIN_PHASE, 0),
        f"{label}: replan executed_steps 未加总到 policy actions",
    )
    _require(
        previous_completed_after is not None
        and previous_completed_after <= diagnostics.get("max_completed_skill_count"),
        f"{label}: Policy replan skill progress 超过 diagnostic max",
    )


def _validate_grasp_segments(diagnostics: dict[str, Any], label: str) -> None:
    action_count = int(diagnostics["action_count"])
    raw_count = _strict_int(
        diagnostics.get("raw_grasp_action_count"), f"{label}.raw_grasp_action_count", minimum=0
    )
    loss_events = _strict_int(
        diagnostics.get("raw_grasp_loss_events"), f"{label}.raw_grasp_loss_events", minimum=0
    )
    segments = diagnostics.get("raw_grasp_segments")
    _require(isinstance(segments, list), f"{label}.raw_grasp_segments 缺失")
    lengths: list[int] = []
    previous_end = 0
    normalized: list[tuple[int, int]] = []
    for index, segment in enumerate(segments):
        _require(isinstance(segment, dict), f"{label}.raw_grasp_segments[{index}] 无效")
        start = _strict_int(
            segment.get("start_action_step"), f"{label}.segment[{index}].start", minimum=1
        )
        end = _strict_int(
            segment.get("end_action_step_exclusive"), f"{label}.segment[{index}].end", minimum=2
        )
        _require(start < end <= action_count + 1, f"{label}: raw grasp segment 范围无效")
        _require(start > previous_end, f"{label}: raw grasp segments 重叠或不连续")
        normalized.append((start, end))
        lengths.append(end - start)
        previous_end = end
    _require(
        sum(lengths) == raw_count, f"{label}: raw grasp segment 未加总到 raw_grasp_action_count"
    )
    _require(
        bool(diagnostics.get("ever_raw_grasped")) == (raw_count > 0),
        f"{label}: ever_raw_grasped 不一致",
    )
    _require(
        diagnostics.get("max_consecutive_raw_grasp_steps") == max(lengths, default=0),
        f"{label}: max_consecutive_raw_grasp_steps 不一致",
    )
    final_progress = diagnostics.get("final_progress")
    _require(isinstance(final_progress, dict), f"{label}.final_progress 缺失")
    final_raw = _strict_bool(
        final_progress.get("raw_grasped"), f"{label}.final_progress.raw_grasped"
    )
    open_segments = sum(end == action_count + 1 for _, end in normalized)
    _require(
        open_segments == int(final_raw), f"{label}: open raw grasp segment 与 final state 不一致"
    )
    _require(
        loss_events == len(normalized) - open_segments, f"{label}: raw_grasp_loss_events 不一致"
    )
    _require(
        diagnostics.get("raw_grasp_rising_edge_count") == len(normalized),
        f"{label}: raw_grasp_rising_edge_count 不一致",
    )
    expected_first = None if not normalized else normalized[0][0]
    expected_last = None if not normalized else normalized[-1][1] - 1
    _require(
        diagnostics.get("first_raw_grasp_action_step") == expected_first,
        f"{label}: first_raw_grasp_action_step 不一致",
    )
    _require(
        diagnostics.get("last_raw_grasp_action_step") == expected_last,
        f"{label}: last_raw_grasp_action_step 不一致",
    )
    max_stable = _strict_int(
        diagnostics.get("max_stable_grasp_steps"), f"{label}.max_stable_grasp_steps", minimum=0
    )
    _require(
        max_stable <= max(lengths, default=0), f"{label}: stable grasp 超过 raw grasp 连续长度"
    )


def _validate_skill_progress(diagnostics: dict[str, Any], label: str) -> None:
    action_count = int(diagnostics["action_count"])
    max_completed = _strict_int(
        diagnostics.get("max_completed_skill_count"),
        f"{label}.max_completed_skill_count",
        minimum=0,
    )
    _require(
        max_completed <= len(PICK_AND_PLACE_SKILLS), f"{label}: max_completed_skill_count 越界"
    )
    completion = diagnostics.get("skill_completion_steps")
    _require(isinstance(completion, dict), f"{label}.skill_completion_steps 缺失")
    _require(set(completion) == set(PICK_AND_PLACE_SKILLS), f"{label}: skill completion keys 漂移")
    resolved: list[int | None] = []
    last_step = 0
    for index, skill in enumerate(PICK_AND_PLACE_SKILLS):
        value = completion[skill]
        if index < max_completed:
            step = _strict_int(value, f"{label}.skill_completion_steps.{skill}", minimum=1)
            _require(
                step >= last_step and step <= action_count,
                f"{label}: skill completion step 次序无效",
            )
            resolved.append(step)
            last_step = step
        else:
            _require(value is None, f"{label}: 未完成 skill {skill} 不应有 completion step")
            resolved.append(None)
    final_progress = diagnostics["final_progress"]
    final_completed = _strict_int(
        final_progress.get("completed_skill_count"),
        f"{label}.final_progress.completed_skill_count",
        minimum=0,
    )
    _require(final_completed == max_completed, f"{label}: final/max completed skill 不一致")
    _strict_int(
        final_progress.get("stable_grasp_steps"),
        f"{label}.final_progress.stable_grasp_steps",
        minimum=0,
    )
    _strict_bool(final_progress.get("task_completed"), f"{label}.final_progress.task_completed")


def _validate_diagnostics(
    diagnostics: dict[str, Any],
    *,
    seed: int,
    failure_reason: str,
) -> None:
    label = f"replay seed {seed} diagnostics"
    _require(diagnostics.get("format") == LOCAL_DAGGER_DIAGNOSTIC_FORMAT, f"{label}.format 不兼容")
    _require(diagnostics.get("environment_seed") == seed, f"{label}.environment_seed 漂移")
    _require(diagnostics.get("boundary_type") == "grasp_lift", f"{label}.boundary_type 漂移")
    _require(diagnostics.get("failure_reason") == failure_reason, f"{label}.failure_reason 漂移")
    _require(
        diagnostics.get("observation_scope") == "local_dagger_collection_session",
        f"{label}.observation_scope 漂移",
    )
    _require(diagnostics.get("boundary_skill") == "grasp", f"{label}.boundary_skill 漂移")
    _validate_phase_and_action_continuity(diagnostics, label)
    _validate_grasp_segments(diagnostics, label)
    _validate_skill_progress(diagnostics, label)

    for key in (
        "min_tcp_to_object_distance_m",
        "max_object_height_above_support_m",
        "max_object_linear_speed_m_s",
        "max_object_angular_speed_rad_s",
    ):
        _finite_number(diagnostics.get(key), f"{label}.{key}")
    final_transition = diagnostics["final_transition"]
    terminated = _strict_bool(final_transition.get("terminated"), f"{label}.terminated")
    truncated = _strict_bool(final_transition.get("truncated"), f"{label}.truncated")
    success = _strict_bool(
        final_transition.get("environment_success"), f"{label}.environment_success"
    )
    action_source = _strict_int(
        final_transition.get("action_source"), f"{label}.action_source", minimum=0
    )
    _require(action_source in {0, 1}, f"{label}: action_source 无效")
    _finite_number(final_transition.get("gripper_opening"), f"{label}.gripper_opening")
    _require(not success, f"{label}: rejected replay 不应有 environment_success")
    _require(
        diagnostics["final_progress"]["task_completed"] is False,
        f"{label}: rejected replay 不应 task_completed",
    )

    if failure_reason == POLICY_BEFORE_BOUNDARY_REASON:
        _require(
            terminated != truncated,
            f"{label}: policy terminal 必须且只能有一个 terminated/truncated",
        )
        _require(
            diagnostics["phase_at_failure"] == POLICY_ROLLIN_PHASE,
            f"{label}: boundary 前失败必须在 policy_rollin",
        )
        _require(action_source == 0, f"{label}: boundary 前失败的最终 action 必须来自 Policy")
        _require(
            diagnostics["max_completed_skill_count"] < 2, f"{label}: 已完成稳定 Grasp 却未 takeover"
        )
        _require(
            diagnostics["policy_replan_traces"][-1]["completed_skill_count_after"]
            == diagnostics["max_completed_skill_count"],
            f"{label}: Policy terminal progress 不一致",
        )
        _require(
            diagnostics.get("boundary_reached") is False, f"{label}: boundary_reached 应为 false"
        )
        _require(
            diagnostics.get("boundary_detection_step") is None,
            f"{label}: boundary_detection_step 应为 null",
        )
        _require(
            diagnostics.get("expert_takeover_step") is None,
            f"{label}: expert_takeover_step 应为 null",
        )
        if truncated:
            _require(
                diagnostics["action_count"] == 300,
                f"{label}: policy time limit 必须发生在 300 steps",
            )
    elif failure_reason == EPISODE_TIME_LIMIT_REASON:
        _require(truncated and not terminated, f"{label}: Episode time limit terminal 契约不成立")
        _require(
            diagnostics["action_count"] == 300, f"{label}: Expert time limit 必须发生在 300 steps"
        )
        _require(
            diagnostics["phase_at_failure"] != POLICY_ROLLIN_PHASE,
            f"{label}: Expert time limit 必须在 takeover 后",
        )
        _require(action_source == 1, f"{label}: Expert time limit 最终 action 必须来自 Expert")
        _require(
            diagnostics["max_completed_skill_count"] >= 2,
            f"{label}: Expert time limit 缺稳定 Grasp boundary",
        )
        _require(
            diagnostics["policy_replan_traces"][-1]["completed_skill_count_after"] == 2,
            f"{label}: Policy 必须在 Grasp boundary 精确停止",
        )
        transitions = diagnostics["phase_transitions"]
        takeover_step = transitions[1]["action_step"]
        _require(
            diagnostics.get("boundary_reached") is True, f"{label}: boundary_reached 应为 true"
        )
        _require(
            diagnostics.get("boundary_detection_step") == takeover_step,
            f"{label}: boundary_detection_step 不同步",
        )
        _require(
            diagnostics.get("expert_takeover_step") == takeover_step,
            f"{label}: expert_takeover_step 不同步",
        )
        _require(
            diagnostics["skill_completion_steps"]["grasp"] == takeover_step,
            f"{label}: Grasp boundary 与 Expert takeover step 不同步",
        )
    else:
        _fail(f"{label}: 不是预注册的两类目标失败")


def _validate_replay_experiment(
    experiment: dict[str, Any],
    *,
    formal: dict[str, Any],
) -> None:
    _require(experiment.get("format") == REPLAY_FORMAT, "replay experiment format 不兼容")
    _require(
        experiment.get("purpose") == "exploratory GL failure decomposition; not training data",
        "replay purpose 漂移",
    )
    _require(experiment.get("target_boundary_type") == "grasp_lift", "replay target boundary 漂移")
    _require(
        experiment.get("formal_source_revision") == formal["experiment"]["source_revision"],
        "replay formal_source_revision 漂移",
    )
    replay_source = experiment.get("replay_source_revision")
    _require(isinstance(replay_source, str) and replay_source, "replay_source_revision 缺失")
    _require(experiment.get("source_revision") == replay_source, "replay source_revision 自相矛盾")
    _require(
        experiment.get("checkpoint") == formal["experiment"]["checkpoint"],
        "replay checkpoint identity 漂移",
    )
    _require(
        experiment.get("base_dataset") == formal["experiment"]["base_dataset"],
        "replay base_dataset identity 漂移",
    )
    _require(
        experiment.get("config") == formal["experiment"]["config"], "replay frozen config 漂移"
    )
    seeds = experiment.get("environment_seeds")
    _require(isinstance(seeds, list), "replay environment_seeds 缺失")
    resolved = tuple(
        _strict_int(value, f"replay environment_seeds[{index}]", minimum=0)
        for index, value in enumerate(seeds)
    )
    _require(
        resolved == formal["target_seeds"], "replay environment_seeds 与 87 target failures 不一致"
    )
    _require(experiment.get("selected_count") == 87, "replay selected_count 必须是 87")
    reasons = experiment.get("target_failure_reasons")
    _require(reasons == list(TARGET_FAILURE_REASONS), "replay target failure reasons 漂移")
    _require(
        experiment.get("model_cache") == formal["experiment"].get("model_cache"),
        "replay model_cache identity 漂移",
    )

    selected = experiment.get("selected_candidates")
    _require(
        isinstance(selected, list) and len(selected) == 87,
        "replay selected_candidates 必须是 87 条",
    )
    selected_by_seed = _index_rows(selected, label="replay selected_candidates")
    _require(
        set(selected_by_seed) == set(formal["target_seeds"]),
        "replay selected_candidates seeds 漂移",
    )
    for seed in formal["target_seeds"]:
        frozen = selected_by_seed[seed]
        record = formal["records"][seed]
        _require(
            frozen.get("boundary_type") == "grasp_lift", f"selected seed {seed}: boundary_type 漂移"
        )
        for key in ("status", "failure", "source_revision", "base_dataset", "checkpoint", "config"):
            _require(frozen.get(key) == record.get(key), f"selected seed {seed}: {key} 漂移")
        _require(
            frozen.get("record_sha256") == _sha256_file(formal["record_paths"][seed]),
            f"selected seed {seed}: record_sha256 漂移",
        )
        reference = frozen.get("record")
        _require(
            isinstance(reference, str)
            and Path(reference).resolve() == formal["record_paths"][seed].resolve(),
            f"selected seed {seed}: record path 漂移",
        )

    formal_pool = experiment.get("formal_pool")
    _require(isinstance(formal_pool, dict), "replay formal_pool identity 缺失")
    _require(
        Path(str(formal_pool.get("path"))).resolve()
        == formal["record_paths"][FORMAL_GL_SEEDS[0]].parents[2].resolve(),
        "replay formal_pool.path 漂移",
    )
    _require(
        formal_pool.get("experiment") == formal["experiment"],
        "replay embedded formal experiment 漂移",
    )
    formal_root = formal["record_paths"][FORMAL_GL_SEEDS[0]].parents[2]
    _require(
        formal_pool.get("experiment_sha256") == _sha256_file(formal_root / "experiment.json"),
        "replay formal experiment SHA256 漂移",
    )
    _require(
        formal_pool.get("collection_candidates_sha256")
        == _sha256_file(formal_root / "collection_candidates.jsonl"),
        "replay formal candidates SHA256 漂移",
    )


def _validate_replay_row_mirrors(
    row: dict[str, Any],
    *,
    replay_record: dict[str, Any],
    replay_path: Path,
    formal_record: dict[str, Any],
    formal_path: Path,
    seed: int,
) -> None:
    label = f"replay seed {seed} index"
    _require(
        row.get("format") == "robot-vla-local-dagger-failure-replay-candidate/v1",
        f"{label}.format 不兼容",
    )
    _require(
        row.get("boundary_type") == replay_record.get("config", {}).get("boundary_type"),
        f"{label}.boundary_type 与 record config 不一致",
    )
    for key in (
        "status",
        "failure",
        "failure_diagnostics",
        "source_revision",
        "config",
        "checkpoint",
        "base_dataset",
    ):
        _require(row.get(key) == replay_record.get(key), f"{label}.{key} 与 record 不一致")
    original = row.get("original")
    _require(isinstance(original, dict), f"{label}.original 缺失")
    _require(original.get("status") == formal_record["status"], f"{label}.original.status 漂移")
    _require(original.get("failure") == formal_record["failure"], f"{label}.original.failure 漂移")
    _require(
        original.get("source_revision") == formal_record["source_revision"],
        f"{label}.original.source_revision 漂移",
    )
    _require(
        original.get("record_sha256") == _sha256_file(formal_path),
        f"{label}.original.record_sha256 漂移",
    )
    original_record_ref = original.get("record")
    _require(
        isinstance(original_record_ref, str) and original_record_ref,
        f"{label}.original.record 缺失",
    )
    _require(
        Path(original_record_ref).resolve() == formal_path.resolve(),
        f"{label}.original.record 漂移",
    )
    reconciliation = row.get("reconciliation")
    _require(isinstance(reconciliation, dict), f"{label}.reconciliation 缺失")
    _require(row.get("reconciled") is True, f"{label}.reconciled 未通过")
    _require(
        reconciliation.get("classification") == "matched",
        f"{label}: runner classification 不是 matched",
    )
    _require(reconciliation.get("reconciled") is True, f"{label}: runner reconciliation 未通过")
    _require(
        reconciliation.get("exact_match") is True, f"{label}: runner reconciliation 未 exact-match"
    )
    for key in (
        "status_matches",
        "reason_matches",
        "failure_type_matches",
        "config_matches",
        "checkpoint_matches",
        "base_dataset_matches",
        "source_revision_matches",
        "identity_contract_matches",
        "failure_diagnostics_matches",
        "failure_record_contract_matches",
        "subprocess_contract_matches",
    ):
        _require(reconciliation.get(key) is True, f"{label}.reconciliation.{key} 未通过")
    returncode = _strict_int(
        reconciliation.get("subprocess_returncode"),
        f"{label}.reconciliation.subprocess_returncode",
    )
    _require(returncode != 0, f"{label}: rejected replay subprocess 必须非零退出")
    _validate_record_reference(row, replay_path, label)


def _load_replay(replay_root: Path, formal: dict[str, Any]) -> dict[str, Any]:
    experiment = _load_json(replay_root / "experiment.json")
    _validate_replay_experiment(experiment, formal=formal)
    rows = _load_jsonl(replay_root / "replay_candidates.jsonl")
    indexed = _index_rows(rows, label="replay_candidates")
    expected_seeds = set(formal["target_seeds"])
    _require(len(indexed) == 87, f"replay target candidates 应为 87，实际 {len(indexed)}")
    _require(set(indexed) == expected_seeds, "replay seeds 未精确覆盖 87 target failures")

    records: dict[int, dict[str, Any]] = {}
    classifications: dict[int, dict[str, str]] = {}
    reconciliation_rows: list[dict[str, Any]] = []
    replay_sources: set[str] = set()
    for seed in formal["target_seeds"]:
        row = indexed[seed]
        replay_path = _candidate_record_path(replay_root, seed)
        record = _load_json(replay_path)
        formal_record = formal["records"][seed]
        label = f"replay seed {seed}"
        _require(record.get("format") == COLLECTION_FORMAT, f"{label}.format 不兼容")
        _require(
            record.get("status") == formal_record["status"] == "rejected", f"{label}.status 未对齐"
        )
        _require(
            record.get("failure") == formal_record["failure"], f"{label}.failure 未逐 seed 对齐"
        )
        _require(record.get("config") == formal_record["config"], f"{label}.config 未逐 seed 对齐")
        replay_checkpoint = record.get("checkpoint")
        _require(isinstance(replay_checkpoint, dict), f"{label}.checkpoint 缺失")
        _require(
            replay_checkpoint == formal_record["checkpoint"],
            f"{label}.checkpoint 未逐 seed 完整对齐",
        )
        _require(
            record.get("base_dataset") == formal_record.get("base_dataset"),
            f"{label}.base_dataset 未对齐",
        )
        replay_source = record.get("source_revision")
        _require(isinstance(replay_source, str) and replay_source, f"{label}.source_revision 缺失")
        replay_sources.add(str(replay_source))
        diagnostics = record.get("failure_diagnostics")
        _require(isinstance(diagnostics, dict), f"{label}.failure_diagnostics 缺失")
        reason = str(record["failure"]["reason"])
        _validate_diagnostics(diagnostics, seed=seed, failure_reason=reason)
        classification = classify_grasp_lift_failure(diagnostics)
        expected_family = (
            "policy_before_stable_grasp_boundary"
            if reason == POLICY_BEFORE_BOUNDARY_REASON
            else "expert_time_limit_after_takeover"
        )
        _require(
            classification["failure_family"] == expected_family,
            f"{label}: classifier family 不一致",
        )
        _require(
            classification["progress_bucket"] != "contract_violation_stable_grasp_without_takeover",
            f"{label}: stable Grasp 后未 takeover 契约违反",
        )
        _validate_replay_row_mirrors(
            row,
            replay_record=record,
            replay_path=replay_path,
            formal_record=formal_record,
            formal_path=formal["record_paths"][seed],
            seed=seed,
        )
        classifications[seed] = classification
        records[seed] = record
        reconciliation_rows.append(
            {
                "environment_seed": seed,
                "status_match": True,
                "failure_reason_match": True,
                "config_match": True,
                "checkpoint_sha256_match": True,
                "diagnostics_valid": True,
                "exact_match": True,
            }
        )
    _require(len(replay_sources) == 1, "87 replay records 必须共享同一 source_revision")
    only_source = next(iter(replay_sources))
    _require(
        only_source == experiment["replay_source_revision"],
        "replay record/experiment source_revision 不一致",
    )
    return {
        "experiment": experiment,
        "rows": indexed,
        "records": records,
        "classifications": classifications,
        "reconciliation_rows": reconciliation_rows,
    }


def _distribution(values: list[int]) -> dict[str, int | float]:
    _require(values, "distribution 不能为空")
    median = statistics.median(values)
    return {
        "min": min(values),
        "median": int(median) if float(median).is_integer() else float(median),
        "max": max(values),
    }


def _accepted_timing_summary(rows: list[dict[str, int]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["environment_seed"])
    _require(len(ordered) == 10, "accepted timing 必须有 10 条")
    return {
        "count": len(ordered),
        "expert_takeover_step": _distribution([row["expert_takeover_step"] for row in ordered]),
        "takeover_to_full_success_steps": _distribution(
            [row["takeover_to_success_steps"] for row in ordered]
        ),
        "trajectory_num_steps": _distribution([row["num_steps"] for row in ordered]),
        "per_seed": ordered,
        "definition": "takeover_to_full_success_steps = trajectory.num_steps - expert_takeover_step",
        "survivorship_caveat": (
            "该 timing 只由 10 条完整成功的 accepted survivors 估计；"
            "不可将其外推为 90 条 rejected episode 的 boundary timing 或成功所需时间。"
        ),
    }


def _decompose_failures(formal: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    cross: dict[tuple[str, str], list[int]] = defaultdict(list)
    policy_progress: dict[str, list[int]] = defaultdict(list)
    policy_terminal: dict[str, list[int]] = defaultdict(list)
    expert_phases: dict[str, list[int]] = defaultdict(list)
    for seed in formal["target_seeds"]:
        classification = replay["classifications"][seed]
        reason = formal["records"][seed]["failure"]["reason"]
        if reason == POLICY_BEFORE_BOUNDARY_REASON:
            progress = classification["progress_bucket"]
            terminal = classification["terminal_bucket"]
            cross[(progress, terminal)].append(seed)
            policy_progress[progress].append(seed)
            policy_terminal[terminal].append(seed)
        elif reason == EPISODE_TIME_LIMIT_REASON:
            expert_phases[classification["progress_bucket"]].append(seed)

    cross_rows = [
        {
            "progress_bucket": progress,
            "terminal_bucket": terminal,
            "count": len(seeds),
            "seeds": sorted(seeds),
        }
        for (progress, terminal), seeds in sorted(cross.items())
    ]
    progress_rows = [
        {"bucket": bucket, "count": len(seeds), "seeds": sorted(seeds)}
        for bucket, seeds in sorted(policy_progress.items())
    ]
    terminal_rows = [
        {"bucket": bucket, "count": len(seeds), "seeds": sorted(seeds)}
        for bucket, seeds in sorted(policy_terminal.items())
    ]
    phase_rows = [
        {"commanded_phase": phase, "count": len(seeds), "seeds": sorted(seeds)}
        for phase, seeds in sorted(
            expert_phases.items(),
            key=lambda item: LOCAL_DAGGER_DIAGNOSTIC_PHASES.index(item[0]),
        )
    ]
    policy_cross_total = sum(row["count"] for row in cross_rows)
    policy_progress_total = sum(row["count"] for row in progress_rows)
    policy_terminal_total = sum(row["count"] for row in terminal_rows)
    expert_total = sum(row["count"] for row in phase_rows)
    _require(
        policy_cross_total == policy_progress_total == policy_terminal_total == 71,
        "71 policy failures 分解未加总",
    )
    _require(expert_total == 16, "16 Expert time-limit phases 未加总")
    return {
        "policy_before_stable_grasp_boundary": {
            "total": 71,
            "progress_by_terminal": cross_rows,
            "progress_marginal": progress_rows,
            "terminal_marginal": terminal_rows,
            "additivity": {
                "cross_total": policy_cross_total,
                "progress_total": policy_progress_total,
                "terminal_total": policy_terminal_total,
                "expected_total": 71,
                "passed": True,
            },
        },
        "expert_time_limit_after_takeover": {
            "total": 16,
            "commanded_phase": phase_rows,
            "phase_semantics": (
                "commanded_phase 是失败 action 发生时 collector 正在执行的 Expert 调用阶段，"
                "不等价于该阶段对应 Predicate 已完成。"
            ),
            "additivity": {
                "phase_total": expert_total,
                "expected_total": 16,
                "passed": True,
            },
        },
        "all_target_failures": {
            "policy_total": 71,
            "expert_time_limit_total": 16,
            "total": 87,
            "expected_total": 87,
            "passed": policy_cross_total + expert_total == 87,
        },
    }


def _boundary_reach_lower_bound(formal: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    accepted = [
        candidate["environment_seed"]
        for candidate in formal["canonical_candidates"]
        if candidate["reason_code"] == "accepted"
    ]
    replay_confirmed = [
        seed
        for seed in formal["target_seeds"]
        if formal["records"][seed]["failure"]["reason"] == EPISODE_TIME_LIMIT_REASON
        and replay["records"][seed]["failure_diagnostics"]["max_completed_skill_count"] >= 2
    ]
    formal_post_boundary = [
        candidate["environment_seed"]
        for candidate in formal["canonical_candidates"]
        if candidate["reason_code"] in POST_BOUNDARY_REASON_CODES
    ]
    all_seeds = sorted(set(accepted) | set(replay_confirmed) | set(formal_post_boundary))
    _require(len(accepted) == 10, "boundary reach accepted evidence 必须是 10")
    _require(len(replay_confirmed) == 16, "boundary reach replay evidence 必须是 16")
    _require(len(formal_post_boundary) == 3, "boundary reach post-boundary reasons 必须是 3")
    _require(len(all_seeds) == 29, "boundary reach lower bound 必须是 29")
    return {
        "metric": "formal_gl_boundary_reach_lower_bound",
        "qualifier": "at_least",
        "count": 29,
        "denominator": 100,
        "rate": 0.29,
        "definition": (
            "稳定 Grasp boundary 已发生的可证最低数：formal accepted，"
            "与 formal 逐 seed 对齐的 Expert time-limit replay，"
            "以及按 collector 控制流只可在 boundary 后发生的其他 formal rejection。"
        ),
        "evidence": {
            "formal_accepted": {"count": len(accepted), "seeds": accepted},
            "replay_confirmed_expert_time_limit": {
                "count": len(replay_confirmed),
                "seeds": replay_confirmed,
            },
            "formal_post_boundary_only_rejections": {
                "count": len(formal_post_boundary),
                "seeds": formal_post_boundary,
            },
        },
        "seeds": all_seeds,
        "caveat": (
            "这是可证下界，不是新的 success rate；"
            "71 条 boundary-before failures 的重放只用于分解未到达过程。"
        ),
    }


def analyze_gl_failure_diagnostics(formal_root: Path, replay_root: Path) -> dict[str, Any]:
    """读取两个冻结根目录，验证后返回可 JSON 序列化的聚合结果。"""

    formal_root = Path(formal_root).resolve()
    replay_root = Path(replay_root).resolve()
    formal = _load_formal(formal_root)
    replay = _load_replay(replay_root, formal)
    decomposition = _decompose_failures(formal, replay)
    boundary_reach = _boundary_reach_lower_bound(formal, replay)
    accepted_timing = _accepted_timing_summary(formal["accepted_timing"])
    reconciliation_rows = replay["reconciliation_rows"]
    _require(len(reconciliation_rows) == 87, "reconciliation rows 必须是 87")

    return {
        "format": ANALYSIS_FORMAT,
        "experiment_id": "E012a-GL-failure-diagnostics-v1",
        "analysis_scope": {
            "formal_candidates": 100,
            "diagnostic_replay_targets": 87,
            "boundary_type": "grasp_lift",
            "historical_formal_result_mutated": False,
            "diagnostic_replay_is_training_data": False,
        },
        "identities": {
            "formal_source_revision": formal["experiment"]["source_revision"],
            "replay_source_revision": replay["experiment"]["replay_source_revision"],
            "source_revision_note": (
                "formal source 是历史结果身份；replay source 包含无干预 diagnostic observer，"
                "二者允许不同，但 87 条 replay 内必须一致。"
            ),
            "checkpoint": formal["experiment"]["checkpoint"],
            "base_dataset": formal["experiment"]["base_dataset"],
            "frozen_collection_config": formal["experiment"]["config"],
            "formal_experiment_sha256": _sha256_file(formal_root / "experiment.json"),
            "formal_candidates_sha256": _sha256_file(formal_root / "collection_candidates.jsonl"),
            "replay_experiment_sha256": _sha256_file(replay_root / "experiment.json"),
            "replay_candidates_sha256": _sha256_file(replay_root / "replay_candidates.jsonl"),
        },
        "canonical_formal_population": {
            "count": 100,
            "status_reason_counts": formal["canonical_counts"],
            "candidates": formal["canonical_candidates"],
            "additivity": {
                "summed_count": sum(formal["canonical_counts"].values()),
                "expected_count": 100,
                "passed": sum(formal["canonical_counts"].values()) == 100,
            },
        },
        "boundary_reach_lower_bound": boundary_reach,
        "failure_decomposition": decomposition,
        "accepted_survivor_timing": accepted_timing,
        "reconciliation": {
            "formal_target_count": 87,
            "replay_count": 87,
            "matched_count": len(reconciliation_rows),
            "status_match_count": sum(row["status_match"] for row in reconciliation_rows),
            "failure_reason_match_count": sum(
                row["failure_reason_match"] for row in reconciliation_rows
            ),
            "config_match_count": sum(row["config_match"] for row in reconciliation_rows),
            "checkpoint_sha256_match_count": sum(
                row["checkpoint_sha256_match"] for row in reconciliation_rows
            ),
            "valid_diagnostics_count": sum(row["diagnostics_valid"] for row in reconciliation_rows),
            "exact_match": all(row["exact_match"] for row in reconciliation_rows),
            "per_seed": reconciliation_rows,
        },
        "unidentifiable_items": [
            (
                "formal rejected records 没有 step-level telemetry；细分来自同 seed/同配置的"
                " deterministic diagnostic replay，不是对历史 trajectory 的事后解析。"
            ),
            (
                "commanded Expert phase 只标识失败时的 collector callsite，"
                "不能证明对应 Lift/Transport/Place Predicate 已成功。"
            ),
            (
                "3 条非目标 rejection（Expert incomplete、MPlib、snapshot）未纳入 87-seed replay，"
                "因而只能依 collector 控制流证明它们在 boundary 后，不做更细因果归类。"
            ),
            (
                "互斥 progress/terminal 类别是观测性分解，不识别 Action Chunk 冲突"
                "或任一控制参数对失败的因果效应。"
            ),
        ],
        "survivorship_caveat": accepted_timing["survivorship_caveat"],
        "decision_use": (
            "本文件可用于冻结 protocol amendment 的证据；"
            "不得改写 formal E012a 结果、当作 D1 样本，或直接解释为训练收益。"
        ),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = analyze_gl_failure_diagnostics(args.formal_root, args.replay_root)
    _atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "event": "e012_gl_failure_diagnostics_analyzed",
                "output": str(args.output.resolve()),
                "formal_candidates": payload["analysis_scope"]["formal_candidates"],
                "replay_targets": payload["analysis_scope"]["diagnostic_replay_targets"],
                "reconciliation_exact_match": payload["reconciliation"]["exact_match"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
