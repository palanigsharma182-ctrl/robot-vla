"""固定 Pick-and-Place semantic proposal 到可审计 task graph 的编译器。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from robot_vla.executive.contracts import (
    EXECUTIVE_PLAN_VERSION,
    CompiledTaskPlan,
    ControllerOwner,
    CriticalAction,
    PhaseId,
    PhaseSpec,
    SemanticPlanProposal,
    SubtaskId,
    SubtaskSpec,
)

PICK_PLACE_TASK_ID = "pick-cube-to-region"
CANONICAL_PICK_PLACE_SUBTASKS = (
    SubtaskId.APPROACH_AND_ALIGN,
    SubtaskId.ACQUIRE_AND_VERIFY,
    SubtaskId.TRANSFER_HELD_OBJECT,
    SubtaskId.DEPOSIT_AND_VERIFY,
)


@dataclass(frozen=True)
class PlanCompilerConfig:
    """P0/P1 shadow 配置；正式稳定 Tick 和 timeout 必须在 E013 标定后冻结。"""

    stable_ticks_required: int = 2
    phase_timeout_ticks: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.stable_ticks_required, int)
            or isinstance(self.stable_ticks_required, bool)
            or self.stable_ticks_required <= 0
        ):
            raise ValueError("stable_ticks_required 必须为正整数")
        if self.phase_timeout_ticks is not None and (
            not isinstance(self.phase_timeout_ticks, int)
            or isinstance(self.phase_timeout_ticks, bool)
            or self.phase_timeout_ticks <= 0
        ):
            raise ValueError("phase_timeout_ticks 必须为正整数或 None")

    def to_dict(self) -> dict[str, int | None]:
        return {
            "stable_ticks_required": self.stable_ticks_required,
            "phase_timeout_ticks": self.phase_timeout_ticks,
        }


class PlanCompilationError(ValueError):
    """Semantic proposal 无法安全编译时的 fail-closed 错误。"""


def _phase(
    phase: PhaseId,
    subtask: SubtaskId,
    owner: ControllerOwner,
    *,
    required: tuple[str, ...],
    entry: tuple[str, ...],
    exit: tuple[str, ...],
    invariant: tuple[str, ...] = (),
    config: PlanCompilerConfig,
    critical: CriticalAction = CriticalAction.NONE,
) -> PhaseSpec:
    return PhaseSpec(
        phase=phase,
        subtask=subtask,
        controller_owner=owner,
        required_modalities=required,
        entry_predicates=entry,
        exit_predicates=exit,
        invariant_predicates=invariant,
        stable_ticks_required=config.stable_ticks_required,
        timeout_ticks=config.phase_timeout_ticks,
        critical_action=critical,
    )


def _build_phase_specs(config: PlanCompilerConfig) -> tuple[PhaseSpec, ...]:
    motion = ("proprio", "tcp_pose", "controller_state")
    external = motion + ("rgb_external", "wrist_camera_pose")
    wrist = motion + ("rgb_wrist", "wrist_camera_pose")
    contact = wrist + ("finger_force",)
    return (
        _phase(
            PhaseId.ACQUIRE_TRACK,
            SubtaskId.APPROACH_AND_ALIGN,
            ControllerOwner.SAFE_HOLD,
            required=external,
            entry=(),
            exit=("object_track_valid",),
            config=config,
        ),
        _phase(
            PhaseId.COARSE_APPROACH,
            SubtaskId.APPROACH_AND_ALIGN,
            ControllerOwner.ACTION_CHUNK,
            required=external,
            entry=("object_track_valid", "goal_track_valid"),
            exit=("coarse_reach_complete",),
            invariant=("object_track_valid", "goal_track_valid"),
            config=config,
        ),
        _phase(
            PhaseId.FINE_ALIGN,
            SubtaskId.APPROACH_AND_ALIGN,
            ControllerOwner.PRECISION,
            required=wrist,
            entry=("precision_target_valid",),
            exit=("fine_alignment_complete",),
            invariant=("precision_target_valid",),
            config=config,
        ),
        _phase(
            PhaseId.STABILIZE_PREGRASP,
            SubtaskId.APPROACH_AND_ALIGN,
            ControllerOwner.PRECISION,
            required=wrist,
            entry=("pregrasp_pose_valid",),
            exit=("pregrasp_stable",),
            invariant=("pregrasp_pose_valid",),
            config=config,
        ),
        _phase(
            PhaseId.FINAL_APPROACH,
            SubtaskId.ACQUIRE_AND_VERIFY,
            ControllerOwner.PRECISION,
            required=wrist,
            entry=("pregrasp_stable",),
            exit=("close_ready",),
            invariant=("precision_target_valid",),
            config=config,
        ),
        _phase(
            PhaseId.CLOSE_UNTIL_CONTACT,
            SubtaskId.ACQUIRE_AND_VERIFY,
            ControllerOwner.FORCE_GUARD,
            required=contact,
            entry=("close_authorized",),
            exit=("finger_contact_detected",),
            invariant=("close_authorized",),
            config=config,
            critical=CriticalAction.CLOSE_GRIPPER,
        ),
        _phase(
            PhaseId.SEAT_AND_BALANCE,
            SubtaskId.ACQUIRE_AND_VERIFY,
            ControllerOwner.FORCE_GUARD,
            required=contact,
            entry=("finger_contact_detected",),
            exit=("grasp_balanced",),
            invariant=("finger_contact_detected",),
            config=config,
        ),
        _phase(
            PhaseId.VERIFY_GRASP,
            SubtaskId.ACQUIRE_AND_VERIFY,
            ControllerOwner.SAFE_HOLD,
            required=contact,
            entry=("grasp_candidate",),
            exit=("grasp_verified",),
            invariant=("grasp_candidate",),
            config=config,
        ),
        _phase(
            PhaseId.LIFT_CLEARANCE,
            SubtaskId.TRANSFER_HELD_OBJECT,
            ControllerOwner.ACTION_CHUNK,
            required=contact,
            entry=("lift_authorized",),
            exit=("lift_clearance_reached",),
            invariant=("lift_authorized", "grasp_verified"),
            config=config,
            critical=CriticalAction.LIFT,
        ),
        _phase(
            PhaseId.MOVE_TO_GOAL,
            SubtaskId.TRANSFER_HELD_OBJECT,
            ControllerOwner.ACTION_CHUNK,
            required=external + ("finger_force",),
            entry=("grasp_verified", "goal_track_valid"),
            exit=("goal_region_reached",),
            invariant=("grasp_verified", "goal_track_valid"),
            config=config,
        ),
        _phase(
            PhaseId.ALIGN_FOR_DEPOSIT,
            SubtaskId.TRANSFER_HELD_OBJECT,
            ControllerOwner.PRECISION,
            required=wrist + ("finger_force",),
            entry=("precision_target_valid",),
            exit=("deposit_alignment_complete",),
            invariant=("precision_target_valid", "grasp_verified"),
            config=config,
        ),
        _phase(
            PhaseId.STABILIZE_HELD,
            SubtaskId.TRANSFER_HELD_OBJECT,
            ControllerOwner.PRECISION,
            required=wrist + ("finger_force",),
            entry=("grasp_verified",),
            exit=("held_pose_stable",),
            invariant=("grasp_verified",),
            config=config,
        ),
        _phase(
            PhaseId.LOWER_TO_SUPPORT,
            SubtaskId.DEPOSIT_AND_VERIFY,
            ControllerOwner.FORCE_GUARD,
            required=contact,
            entry=("held_pose_stable",),
            exit=("support_contact_detected",),
            invariant=("grasp_verified",),
            config=config,
        ),
        _phase(
            PhaseId.CONFIRM_SUPPORT,
            SubtaskId.DEPOSIT_AND_VERIFY,
            ControllerOwner.FORCE_GUARD,
            required=contact,
            entry=("support_contact_detected",),
            exit=("support_verified",),
            invariant=("support_contact_detected",),
            config=config,
        ),
        _phase(
            PhaseId.RELEASE,
            SubtaskId.DEPOSIT_AND_VERIFY,
            ControllerOwner.FORCE_GUARD,
            required=contact,
            entry=("release_authorized",),
            exit=("object_released",),
            invariant=("release_authorized", "support_verified"),
            config=config,
            critical=CriticalAction.RELEASE_GRIPPER,
        ),
        _phase(
            PhaseId.RETRACT,
            SubtaskId.DEPOSIT_AND_VERIFY,
            ControllerOwner.ACTION_CHUNK,
            required=external,
            entry=("object_released",),
            exit=("tcp_retracted",),
            invariant=("object_released",),
            config=config,
        ),
        _phase(
            PhaseId.VERIFY_SETTLED,
            SubtaskId.DEPOSIT_AND_VERIFY,
            ControllerOwner.SAFE_HOLD,
            required=external,
            entry=("object_released",),
            exit=("placement_verified",),
            invariant=("object_released",),
            config=config,
        ),
        _phase(
            PhaseId.SAFE_HOLD,
            SubtaskId.RECOVER_OR_HOLD,
            ControllerOwner.SAFE_HOLD,
            required=("proprio", "controller_state"),
            entry=(),
            exit=("hold_confirmed",),
            config=config,
        ),
        _phase(
            PhaseId.REOBSERVE,
            SubtaskId.RECOVER_OR_HOLD,
            ControllerOwner.SAFE_HOLD,
            required=(
                "rgb_external",
                "rgb_wrist",
                "proprio",
                "tcp_pose",
                "wrist_camera_pose",
                "finger_force",
                "controller_state",
            ),
            entry=("hold_confirmed",),
            exit=("modalities_recovered",),
            config=config,
        ),
        _phase(
            PhaseId.DIAGNOSE,
            SubtaskId.RECOVER_OR_HOLD,
            ControllerOwner.SAFE_HOLD,
            required=("proprio", "controller_state"),
            entry=("modalities_recovered",),
            exit=(),
            invariant=("modalities_recovered",),
            config=config,
        ),
    )


def _build_subtask_specs() -> tuple[SubtaskSpec, ...]:
    return (
        SubtaskSpec(
            subtask=SubtaskId.APPROACH_AND_ALIGN,
            phases=(
                PhaseId.ACQUIRE_TRACK,
                PhaseId.COARSE_APPROACH,
                PhaseId.FINE_ALIGN,
                PhaseId.STABILIZE_PREGRASP,
            ),
            allowed_next=(
                SubtaskId.ACQUIRE_AND_VERIFY,
                SubtaskId.RECOVER_OR_HOLD,
            ),
        ),
        SubtaskSpec(
            subtask=SubtaskId.ACQUIRE_AND_VERIFY,
            phases=(
                PhaseId.FINAL_APPROACH,
                PhaseId.CLOSE_UNTIL_CONTACT,
                PhaseId.SEAT_AND_BALANCE,
                PhaseId.VERIFY_GRASP,
            ),
            allowed_next=(
                SubtaskId.TRANSFER_HELD_OBJECT,
                SubtaskId.RECOVER_OR_HOLD,
            ),
        ),
        SubtaskSpec(
            subtask=SubtaskId.TRANSFER_HELD_OBJECT,
            phases=(
                PhaseId.LIFT_CLEARANCE,
                PhaseId.MOVE_TO_GOAL,
                PhaseId.ALIGN_FOR_DEPOSIT,
                PhaseId.STABILIZE_HELD,
            ),
            allowed_next=(
                SubtaskId.DEPOSIT_AND_VERIFY,
                SubtaskId.RECOVER_OR_HOLD,
            ),
        ),
        SubtaskSpec(
            subtask=SubtaskId.DEPOSIT_AND_VERIFY,
            phases=(
                PhaseId.LOWER_TO_SUPPORT,
                PhaseId.CONFIRM_SUPPORT,
                PhaseId.RELEASE,
                PhaseId.RETRACT,
                PhaseId.VERIFY_SETTLED,
            ),
            allowed_next=(SubtaskId.RECOVER_OR_HOLD,),
        ),
        SubtaskSpec(
            subtask=SubtaskId.RECOVER_OR_HOLD,
            phases=(PhaseId.SAFE_HOLD, PhaseId.REOBSERVE, PhaseId.DIAGNOSE),
            allowed_next=CANONICAL_PICK_PLACE_SUBTASKS,
        ),
    )


class PickPlacePlanCompiler:
    """只接受冻结的单物体 Pick-and-Place 顺序，拒绝 Qwen 自由扩图。"""

    def __init__(self, config: PlanCompilerConfig | None = None) -> None:
        self.config = config or PlanCompilerConfig()

    def compile(self, proposal: SemanticPlanProposal) -> CompiledTaskPlan:
        if proposal.task_id != PICK_PLACE_TASK_ID:
            raise PlanCompilationError(f"不支持的 task_id: {proposal.task_id}")
        if proposal.requested_subtasks != CANONICAL_PICK_PLACE_SUBTASKS:
            raise PlanCompilationError(
                "首版 requested_subtasks 必须严格等于冻结 Pick-and-Place 顺序"
            )
        subtasks = _build_subtask_specs()
        phases = _build_phase_specs(self.config)
        identity_payload = {
            "version": EXECUTIVE_PLAN_VERSION,
            "proposal": proposal.to_dict(),
            "compiler_config": self.config.to_dict(),
            "subtasks": [item.to_dict() for item in subtasks],
            "phases": [item.to_dict() for item in phases],
        }
        canonical = json.dumps(
            identity_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        plan_id = hashlib.sha256(canonical).hexdigest()
        return CompiledTaskPlan(
            plan_id=plan_id,
            proposal=proposal,
            subtasks=subtasks,
            phases=phases,
        )


__all__ = [
    "CANONICAL_PICK_PLACE_SUBTASKS",
    "PICK_PLACE_TASK_ID",
    "PickPlacePlanCompiler",
    "PlanCompilationError",
    "PlanCompilerConfig",
]
