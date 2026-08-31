"""E012 Local DAgger 失败重放的无干预诊断记录。"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from robot_vla.contracts import PICK_AND_PLACE_SKILLS
from robot_vla.local_dagger_protocol import (
    EXPERT_ACTION_BUDGET_EXHAUSTED_REASON,
    LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD,
    LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD,
    POLICY_ACTION_BUDGET_EXHAUSTED_REASON,
    LocalDaggerActionBudgetPlan,
    LocalDaggerActionBudgetProtocol,
    resolve_local_dagger_action_budget,
)

LOCAL_DAGGER_DIAGNOSTIC_FORMAT = "robot-vla-local-dagger-failure-diagnostics/v1"
POLICY_BEFORE_BOUNDARY_REASON = "Policy 在目标 boundary 前终止或截断"
EPISODE_TIME_LIMIT_REASON = "Episode 在可信成功前达到时间上限"

POLICY_ROLLIN_PHASE = "policy_rollin"
EXPERT_GRASP_APPROACH_PHASE = "expert_grasp_approach"
EXPERT_GRASP_STABILIZATION_PHASE = "expert_grasp_stabilization"
EXPERT_LIFT_MOTION_PHASE = "expert_lift_motion"
EXPERT_TRANSPORT_MOTION_PHASE = "expert_transport_motion"
EXPERT_LOWER_MOTION_PHASE = "expert_lower_motion"
EXPERT_RELEASE_SETTLE_PHASE = "expert_release_settle"

LOCAL_DAGGER_DIAGNOSTIC_PHASES = (
    POLICY_ROLLIN_PHASE,
    EXPERT_GRASP_APPROACH_PHASE,
    EXPERT_GRASP_STABILIZATION_PHASE,
    EXPERT_LIFT_MOTION_PHASE,
    EXPERT_TRANSPORT_MOTION_PHASE,
    EXPERT_LOWER_MOTION_PHASE,
    EXPERT_RELEASE_SETTLE_PHASE,
)


def _finite_float(value: Any, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} 必须为有限数值")
    return resolved


@dataclass
class LocalDaggerFailureDiagnostics:
    """只聚合 Predicate/phase 元数据，不读取或保存 RGB。"""

    environment_seed: int
    boundary_type: str
    action_budget_protocol: LocalDaggerActionBudgetProtocol | str = (
        LocalDaggerActionBudgetProtocol.LEGACY
    )
    current_phase: str = POLICY_ROLLIN_PHASE
    action_count: int = 0
    expert_takeover_step: int | None = None
    phase_action_counts: Counter[str] = field(default_factory=Counter)
    phase_transitions: list[dict[str, Any]] = field(default_factory=list)
    skill_completion_steps: list[int | None] = field(
        default_factory=lambda: [None] * len(PICK_AND_PLACE_SKILLS)
    )
    max_completed_skill_count: int = 0
    max_stable_grasp_steps: int = 0
    raw_grasp_action_count: int = 0
    raw_grasp_loss_events: int = 0
    raw_grasp_segments: list[dict[str, int]] = field(default_factory=list)
    ever_lifted: bool = False
    ever_transported: bool = False
    min_tcp_to_object_distance_m: float | None = None
    max_object_height_above_support_m: float | None = None
    max_object_linear_speed_m_s: float = 0.0
    max_object_angular_speed_rad_s: float = 0.0
    final_progress: dict[str, Any] | None = None
    final_transition: dict[str, Any] | None = None
    policy_replan_traces: list[dict[str, Any]] = field(default_factory=list)
    budget_exhaustion_phase: str | None = None
    action_budget: LocalDaggerActionBudgetPlan = field(init=False, repr=False)
    _open_raw_grasp_start: int | None = field(default=None, repr=False)
    _previous_raw_grasped: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.environment_seed < 0:
            raise ValueError("environment_seed 不能为负数")
        if not self.boundary_type:
            raise ValueError("boundary_type 不能为空")
        self.action_budget = resolve_local_dagger_action_budget(
            self.action_budget_protocol
        )
        self.set_phase(self.current_phase, action_step=0)

    def set_phase(self, phase: str, *, action_step: int) -> None:
        if phase not in LOCAL_DAGGER_DIAGNOSTIC_PHASES:
            raise ValueError(f"未知 Local DAgger diagnostic phase: {phase}")
        if action_step != self.action_count:
            raise ValueError("diagnostic phase 必须在当前 action 边界切换")
        if self.phase_transitions and self.current_phase == phase:
            return
        if self.current_phase != POLICY_ROLLIN_PHASE and phase == POLICY_ROLLIN_PHASE:
            raise ValueError("Expert takeover 后 diagnostic phase 不得返回 Policy")
        if self.current_phase == POLICY_ROLLIN_PHASE and phase != POLICY_ROLLIN_PHASE:
            if self.expert_takeover_step is not None:
                raise RuntimeError("diagnostic expert_takeover_step 已存在")
            self.expert_takeover_step = action_step
        self.current_phase = phase
        self.phase_transitions.append({"action_step": int(action_step), "phase": phase})

    def observe(
        self,
        *,
        action_step: int,
        progress: Any,
        terminated: bool,
        truncated: bool,
        environment_success: bool,
        action_source: int,
        gripper_opening: float,
    ) -> None:
        """记录一个执行后的 control step；调用方不得据此改变控制流。"""

        if action_step != self.action_count + 1:
            raise ValueError(
                "diagnostic action_step 必须连续："
                f"expected={self.action_count + 1}, actual={action_step}"
            )
        completed = int(progress.completed_skill_count)
        if not 0 <= completed <= len(PICK_AND_PLACE_SKILLS):
            raise ValueError("completed_skill_count 超出技能范围")
        stable_grasp_steps = int(progress.stable_grasp_steps)
        if stable_grasp_steps < 0:
            raise ValueError("stable_grasp_steps 不能为负数")

        outcome = progress.outcome
        tcp_distance = _finite_float(
            outcome.tcp_to_object_distance_m,
            "tcp_to_object_distance_m",
        )
        object_height = _finite_float(
            outcome.object_height_above_support_m,
            "object_height_above_support_m",
        )
        linear_speed = _finite_float(
            outcome.object_linear_speed_m_s,
            "object_linear_speed_m_s",
        )
        angular_speed = _finite_float(
            outcome.object_angular_speed_rad_s,
            "object_angular_speed_rad_s",
        )
        raw_grasped = bool(outcome.grasped)

        self.action_count = action_step
        self.phase_action_counts[self.current_phase] += 1
        for skill_index in range(self.max_completed_skill_count, completed):
            if self.skill_completion_steps[skill_index] is None:
                self.skill_completion_steps[skill_index] = action_step
        self.max_completed_skill_count = max(self.max_completed_skill_count, completed)
        self.max_stable_grasp_steps = max(
            self.max_stable_grasp_steps,
            stable_grasp_steps,
        )

        if raw_grasped:
            self.raw_grasp_action_count += 1
            if not self._previous_raw_grasped:
                self._open_raw_grasp_start = action_step
        elif self._previous_raw_grasped:
            if self._open_raw_grasp_start is None:
                raise RuntimeError("raw grasp segment 状态不一致")
            self.raw_grasp_segments.append(
                {
                    "start_action_step": self._open_raw_grasp_start,
                    "end_action_step_exclusive": action_step,
                }
            )
            self.raw_grasp_loss_events += 1
            self._open_raw_grasp_start = None
        self._previous_raw_grasped = raw_grasped

        self.ever_lifted = self.ever_lifted or bool(outcome.lifted)
        self.ever_transported = self.ever_transported or bool(outcome.transported)
        self.min_tcp_to_object_distance_m = (
            tcp_distance
            if self.min_tcp_to_object_distance_m is None
            else min(self.min_tcp_to_object_distance_m, tcp_distance)
        )
        self.max_object_height_above_support_m = (
            object_height
            if self.max_object_height_above_support_m is None
            else max(self.max_object_height_above_support_m, object_height)
        )
        self.max_object_linear_speed_m_s = max(
            self.max_object_linear_speed_m_s,
            linear_speed,
        )
        self.max_object_angular_speed_rad_s = max(
            self.max_object_angular_speed_rad_s,
            angular_speed,
        )
        self.final_progress = {
            "active_skill_id": int(progress.active_skill_id),
            "active_skill_name": str(progress.active_skill_name),
            "completed_skill_count": completed,
            "stable_grasp_steps": stable_grasp_steps,
            "task_completed": bool(progress.task_completed),
            "reached": bool(outcome.reached),
            "raw_grasped": raw_grasped,
            "lifted": bool(outcome.lifted),
            "transported": bool(outcome.transported),
            "tcp_to_object_distance_m": tcp_distance,
            "object_height_above_support_m": object_height,
            "object_linear_speed_m_s": linear_speed,
            "object_angular_speed_rad_s": angular_speed,
        }
        self.final_transition = {
            "action_step": action_step,
            "phase": self.current_phase,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "environment_success": bool(environment_success),
            "action_source": int(action_source),
            "gripper_opening": _finite_float(gripper_opening, "gripper_opening"),
        }

    def record_policy_replan(self, trace: dict[str, Any]) -> None:
        replan_index = int(trace["replan_index"])
        if self.policy_replan_traces:
            expected = int(self.policy_replan_traces[-1]["replan_index"]) + 1
            if replan_index != expected:
                raise ValueError("policy diagnostic replan_index 必须连续")
        elif replan_index != 0:
            raise ValueError("首个 policy diagnostic replan_index 必须为 0")
        self.policy_replan_traces.append(dict(trace))

    def record_budget_exhaustion(self, phase: str, *, action_step: int) -> None:
        """记录 segmented budget 拒绝来源；legacy 不得伪造该诊断。"""

        if not self.action_budget.amended:
            raise ValueError("legacy diagnostics 不支持 segmented budget exhaustion")
        if phase not in {"policy", "expert"}:
            raise ValueError("budget exhaustion phase 必须是 policy/expert")
        if action_step != self.action_count:
            raise ValueError("budget exhaustion 必须记录在当前 action step")
        if self.budget_exhaustion_phase is not None:
            raise RuntimeError("budget exhaustion 已记录")
        if phase == "policy":
            if self.expert_takeover_step is not None:
                raise ValueError("Policy budget exhaustion 不能发生在 takeover 后")
            consumed = self.action_count
            limit = self.action_budget.policy_action_limit
        else:
            if self.expert_takeover_step is None:
                raise ValueError("Expert budget exhaustion 必须发生在 takeover 后")
            consumed = self.action_count - self.expert_takeover_step
            limit = self.action_budget.expert_action_limit
        if limit is None or consumed < limit:
            raise ValueError("budget exhaustion 发生在冻结上限之前")
        self.budget_exhaustion_phase = phase

    def to_dict(self, *, failure_reason: str | None = None) -> dict[str, Any]:
        segments = [dict(item) for item in self.raw_grasp_segments]
        if self._open_raw_grasp_start is not None:
            segments.append(
                {
                    "start_action_step": self._open_raw_grasp_start,
                    "end_action_step_exclusive": self.action_count + 1,
                }
            )
        max_raw_grasp_run = max(
            (
                item["end_action_step_exclusive"] - item["start_action_step"]
                for item in segments
            ),
            default=0,
        )
        target_completed = 1 if self.boundary_type == "reach_grasp" else 2
        boundary_skill = PICK_AND_PLACE_SKILLS[target_completed - 1]
        boundary_detection_step = self.skill_completion_steps[target_completed - 1]
        if (
            self.expert_takeover_step is not None
            and boundary_detection_step != self.expert_takeover_step
        ):
            raise RuntimeError("diagnostic boundary 与 Expert takeover step 不一致")
        result = {
            "format": LOCAL_DAGGER_DIAGNOSTIC_FORMAT,
            "environment_seed": self.environment_seed,
            "boundary_type": self.boundary_type,
            "failure_reason": failure_reason,
            "observation_scope": "local_dagger_collection_session",
            "action_count": self.action_count,
            "failure_control_step": self.action_count,
            "boundary_skill": boundary_skill,
            "boundary_reached": boundary_detection_step is not None,
            "boundary_detection_step": boundary_detection_step,
            "expert_takeover_step": self.expert_takeover_step,
            "phase_at_failure": self.current_phase,
            "phase_action_counts": dict(sorted(self.phase_action_counts.items())),
            "phase_transitions": [dict(item) for item in self.phase_transitions],
            "skill_completion_steps": {
                skill: self.skill_completion_steps[index]
                for index, skill in enumerate(PICK_AND_PLACE_SKILLS)
            },
            "max_completed_skill_count": self.max_completed_skill_count,
            "max_stable_grasp_steps": self.max_stable_grasp_steps,
            "ever_raw_grasped": self.raw_grasp_action_count > 0,
            "raw_grasp_action_count": self.raw_grasp_action_count,
            "raw_grasp_loss_events": self.raw_grasp_loss_events,
            "raw_grasp_segments": segments,
            "max_consecutive_raw_grasp_steps": max_raw_grasp_run,
            "first_raw_grasp_action_step": (
                None if not segments else segments[0]["start_action_step"]
            ),
            "last_raw_grasp_action_step": (
                None
                if not segments
                else segments[-1]["end_action_step_exclusive"] - 1
            ),
            "raw_grasp_rising_edge_count": len(segments),
            "ever_lifted": self.ever_lifted,
            "ever_transported": self.ever_transported,
            "min_tcp_to_object_distance_m": self.min_tcp_to_object_distance_m,
            "max_object_height_above_support_m": self.max_object_height_above_support_m,
            "max_object_linear_speed_m_s": self.max_object_linear_speed_m_s,
            "max_object_angular_speed_rad_s": self.max_object_angular_speed_rad_s,
            "final_progress": None if self.final_progress is None else dict(self.final_progress),
            "final_transition": (
                None if self.final_transition is None else dict(self.final_transition)
            ),
            "policy_replan_count": len(self.policy_replan_traces),
            "policy_replan_required_count": sum(
                bool(item.get("replan_required"))
                for item in self.policy_replan_traces
            ),
            "policy_replan_traces": [dict(item) for item in self.policy_replan_traces],
        }
        planned = self.action_budget.planned_metadata()
        if planned is not None:
            usage = self.action_budget.usage_metadata(
                total_actions=self.action_count,
                expert_takeover_step=self.expert_takeover_step,
            )
            if usage is None:
                raise RuntimeError("amended diagnostics 缺少 action-budget usage")
            result[LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD] = planned
            result[LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD] = usage
            result["budget_exhaustion_phase"] = self.budget_exhaustion_phase
        return result


def classify_grasp_lift_failure(diagnostics: dict[str, Any]) -> dict[str, str]:
    """将重放诊断分成互斥 progress 与 terminal 两个维度。"""

    if diagnostics.get("format") != LOCAL_DAGGER_DIAGNOSTIC_FORMAT:
        raise ValueError("Local DAgger failure diagnostics format 不兼容")
    if diagnostics.get("boundary_type") != "grasp_lift":
        raise ValueError("首版 failure decomposition 只支持 grasp_lift")
    failure_reason = diagnostics.get("failure_reason")
    final_transition = diagnostics.get("final_transition") or {}

    if failure_reason == POLICY_BEFORE_BOUNDARY_REASON:
        completed = int(diagnostics["max_completed_skill_count"])
        ever_raw_grasped = bool(diagnostics["ever_raw_grasped"])
        loss_events = int(diagnostics["raw_grasp_loss_events"])
        final_raw_grasped = bool((diagnostics.get("final_progress") or {}).get("raw_grasped"))
        if completed >= 2:
            progress_bucket = "contract_violation_stable_grasp_without_takeover"
        elif completed == 0:
            progress_bucket = "never_completed_reach"
        elif not ever_raw_grasped:
            progress_bucket = "reached_never_raw_grasped"
        elif loss_events > 0:
            progress_bucket = "transient_grasp_then_lost_before_stable"
        elif final_raw_grasped:
            progress_bucket = "raw_grasp_at_terminal_not_stable"
        else:
            progress_bucket = "transient_grasp_unresolved"

        if bool(final_transition.get("truncated")):
            terminal_bucket = "policy_time_limit"
        elif bool(final_transition.get("terminated")):
            terminal_bucket = "policy_environment_termination"
        else:
            terminal_bucket = "policy_terminal_signal_unresolved"
        return {
            "failure_family": "policy_before_stable_grasp_boundary",
            "progress_bucket": progress_bucket,
            "terminal_bucket": terminal_bucket,
        }

    if failure_reason == EPISODE_TIME_LIMIT_REASON:
        phase = str(diagnostics.get("phase_at_failure"))
        if phase not in LOCAL_DAGGER_DIAGNOSTIC_PHASES:
            phase = "unknown_phase"
        return {
            "failure_family": "expert_time_limit_after_takeover",
            "progress_bucket": phase,
            "terminal_bucket": "episode_time_limit",
        }

    if failure_reason == POLICY_ACTION_BUDGET_EXHAUSTED_REASON:
        return {
            "failure_family": "policy_action_budget_exhausted_before_boundary",
            "progress_bucket": "stable_boundary_not_reached",
            "terminal_bucket": "policy_action_budget",
        }

    if failure_reason == EXPERT_ACTION_BUDGET_EXHAUSTED_REASON:
        phase = str(diagnostics.get("phase_at_failure"))
        if phase not in LOCAL_DAGGER_DIAGNOSTIC_PHASES:
            phase = "unknown_phase"
        return {
            "failure_family": "expert_action_budget_exhausted_after_takeover",
            "progress_bucket": phase,
            "terminal_bucket": "expert_action_budget",
        }

    return {
        "failure_family": "other_rejection",
        "progress_bucket": "not_applicable",
        "terminal_bucket": "not_applicable",
    }


__all__ = [
    "EPISODE_TIME_LIMIT_REASON",
    "EXPERT_ACTION_BUDGET_EXHAUSTED_REASON",
    "EXPERT_GRASP_APPROACH_PHASE",
    "EXPERT_GRASP_STABILIZATION_PHASE",
    "EXPERT_LIFT_MOTION_PHASE",
    "EXPERT_LOWER_MOTION_PHASE",
    "EXPERT_RELEASE_SETTLE_PHASE",
    "EXPERT_TRANSPORT_MOTION_PHASE",
    "LOCAL_DAGGER_DIAGNOSTIC_FORMAT",
    "LOCAL_DAGGER_DIAGNOSTIC_PHASES",
    "POLICY_ACTION_BUDGET_EXHAUSTED_REASON",
    "POLICY_BEFORE_BOUNDARY_REASON",
    "POLICY_ROLLIN_PHASE",
    "LocalDaggerFailureDiagnostics",
    "classify_grasp_lift_failure",
]
