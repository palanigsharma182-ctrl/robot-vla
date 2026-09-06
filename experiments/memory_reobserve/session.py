"""一次运行中请求的开发接线；资格缺失与视角故障分开记录。"""
from dataclasses import asdict

import numpy as np

from experiments.memory_conditioning.conditioning import snapshot_memory
from robot_vla.executive.contracts import PhaseId
from robot_vla.precision.active_front_reobserve import (
    ActiveFrontReobserveConfig, ActiveFrontReobserveController, ActiveFrontTriggerEvidence,
    ActiveFrontTriggerReason,
)
from robot_vla.precision.object_memory import ObjectMemorySafetyContext
from robot_vla.precision.active_front_memory_provider import ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID

ENGINEERING_SEEDS = tuple(range(1001200, 1001204))
VISUAL_FALLBACK_REASONS = frozenset({
    "memory_stale", "pregrasp_window_closed", "gripper_not_open",
    "object_contact_detected", "gripper_close_commanded",
})


class MemoryContextInvalid(RuntimeError):
    """本步尚未发送，旧 Memory 动作必须被撤销。"""


def frame_safety(frame, *, close_commanded=False, tracking_valid=False):
    """只读真实 proprio/接触，不读仿真物体位置或成功标签。"""
    return ObjectMemorySafetyContext(
        pregrasp_window_open=not close_commanded,
        gripper_open=bool(frame.physical_proprio[-1] >= .975),
        controller_tracking_valid=bool(tracking_valid),
        object_contact_detected=bool(np.max(frame.finger_force_n) > .01),
        gripper_close_commanded=bool(close_commanded),
        grasp_candidate=False, grasp_verified=False, object_maybe_moved=False,
    )


class MemoryRouteSession:
    allowed_seeds = ENGINEERING_SEEDS

    def __init__(self, seed, runtime, *, return_steps=40, post_replans=2, visual_fallback=False):
        if return_steps not in (10, 40) or not 2 <= post_replans <= 12:
            raise ValueError("只允许明确的仿真返回时序诊断范围")
        self.return_steps, self.post_replans = return_steps, post_replans
        self.visual_fallback = visual_fallback
        self.visual_only = False
        if seed not in self.allowed_seeds:
            raise ValueError('只允许冻结的新工程开发场景')
        self.seed, self.runtime = seed, runtime
        self.supervisor = ActiveFrontReobserveController(ActiveFrontReobserveConfig(
            enabled=True, allow_capability_absent_trigger=True,
            selected_primitive_id=ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID))
        self.trigger_records = []
        self.latest_tick = None
        self.latest_timestamp = None
        self.request = None
        self.snapshot = None
        self.send_records = []
        self.closed = False
        self.interruption_reason = None
        self.cleanup_records = []
        self.tracking_valid = False

    def reset(self, episode):
        self.episode = episode
        self.supervisor.reset_episode(episode, episode_generation=1)
        self.runtime.reset_memory_episode(episode)

    def observe_trigger(self, frame, tick, *, memory, close_commanded, camera_home, tracking):
        if self.latest_tick is not None and (tick != self.latest_tick + 1
                or frame.timestamp_s <= self.latest_timestamp):
            raise ValueError('触发器要求连续的新鲜控制帧')
        if abs(frame.timestamp_s - tick / 20) > 1e-8:
            raise ValueError("控制tick和实际帧时间不匹配")
        self.latest_tick, self.latest_timestamp = tick, frame.timestamp_s
        self.tracking_valid = bool(tracking["valid"])
        safe = frame_safety(frame, close_commanded=close_commanded, tracking_valid=self.tracking_valid)
        snapshot = snapshot_memory(memory.state, memory.config, safe,
            episode_id=self.episode, timestamp_s=frame.timestamp_s)
        evidence = ActiveFrontTriggerEvidence(self.episode, 1, tick, frame.timestamp_s,
            PhaseId.ACQUIRE_TRACK, False, False, snapshot.available,
            not bool(safe.invalidation_reasons), camera_home,
            ActiveFrontTriggerReason.NO_QUALIFIED_WRIST_PROVIDER_IN_PARENT,
            object_contact=safe.object_contact_detected, gripper_close_commanded=close_commanded)
        decision = self.supervisor.consider_trigger(evidence)
        self.trigger_records.append(dict(evidence=asdict(evidence), decision=asdict(decision),
            reason_scope='qualified-provider-absent; not observed occlusion',
            home_scope='D049 HOME score-only, never a usable measurement', tracking=tracking))
        if decision.requestable:
            self.request = decision.request
            self.interruption_reason = "capability-trigger"
        return decision.requestable

    def bind(self, frame, memory, instruction, *, safety=None):
        safe = safety or frame_safety(frame, close_commanded=self.closed, tracking_valid=self.tracking_valid)
        self.snapshot = self.runtime.bind_frame(frame, memory, safe, instruction)
        if self.visual_only and self.snapshot.available:
            raise RuntimeError('视觉回退不得重新启用旧 Memory')
        return self.snapshot

    def before_send(self, action, frame, memory):
        if self.snapshot is None or not self.snapshot.available:
            return
        safe = frame_safety(frame, close_commanded=bool(action[-1] < .95 or self.closed), tracking_valid=self.tracking_valid)
        current = snapshot_memory(memory.state, memory.config, safe,
            episode_id=self.episode, timestamp_s=frame.timestamp_s)
        self.send_records.append(dict(timestamp_s=frame.timestamp_s,
            measurement_timestamp_s=current.last_observed_timestamp_s,
            available=current.available, reasons=current.reasons))
        if not current.available:
            self.interruption_reason = ','.join(current.reasons)
            memory.invalidate_for_safety(episode_id=self.episode, timestamp_s=frame.timestamp_s, reasons=current.reasons)
            raise MemoryContextInvalid('Memory-conditioned chunk失效，必须清理后重规划：'+','.join(current.reasons))

    def after_send(self, frame, action, memory):
        self.closed |= bool(action[-1] < .95 or np.max(frame.finger_force_n) > .01)
        if self.snapshot is None or not self.snapshot.available:
            return False
        # 在动作边界中断；下一规划使用实际最新时间重新屏蔽，不执行旧 Chunk 的余量。
        safe = frame_safety(frame, close_commanded=self.closed, tracking_valid=self.tracking_valid)
        current = snapshot_memory(memory.state, memory.config, safe,
            episode_id=self.episode, timestamp_s=frame.timestamp_s)
        if not current.available:
            self.interruption_reason = ",".join(current.reasons)
            memory.invalidate_for_safety(episode_id=self.episode, timestamp_s=frame.timestamp_s, reasons=current.reasons)
        return not current.available

    def cleanup_after_execution(self, loop):
        """只对已批准的 Memory 失效原因切换视觉；控制/工程故障保持暂停。"""
        if self.interruption_reason and self.interruption_reason != 'capability-trigger':
            reasons = set(self.interruption_reason.split(','))
            fallback = (self.visual_fallback and not loop.observation_paused
                and bool(reasons) and reasons <= VISUAL_FALLBACK_REASONS)
            if fallback:
                loop.clear_action_history()
            elif not loop.observation_paused:
                loop.pause_for_observation()
            empty = (loop.ensembler.buffer_size == 0 and loop._rtc_previous_chunk is None
                and loop.executor.previous_command_q is None and loop.executor.previous_action is None)
            self.cleanup_records.append(dict(reason=self.interruption_reason, all_history_empty=empty,
                control_step=loop.control_step, paused=loop.observation_paused,
                next_mode='visual-only' if fallback else 'hold'))
            if not empty:
                raise RuntimeError('失效后旧动作历史未清理')
            if fallback:
                self.visual_only = True
                self.interruption_reason = None
            return True
        return False
