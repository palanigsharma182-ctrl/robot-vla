"""Oracle Geometry Reach 的独立 Atomic Reach 评估与距离轨迹。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from robot_vla.adapters import FrankaObservationAdapter
from robot_vla.contracts import RobotSpec
from robot_vla.evaluation.atomic import AtomicSkillEpisodeResult
from robot_vla.evaluation.maniskill import (
    _read_online_observation,
    _reset_atomic_time_limit,
    _TrackingManiSkillController,
)
from robot_vla.execution import RecedingHorizonChunkExecutor
from robot_vla.runtime import QwenVLAReplanLoop


@dataclass(frozen=True)
class ReachDiagnosticEpisodeResult:
    """Atomic Reach 原结果加每个真实环境控制步的 TCP→object 距离。"""

    episode: AtomicSkillEpisodeResult
    distance_trace_m: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.episode.skill_name != "reach":
            raise ValueError("Reach diagnostic 只能保存 reach Episode")
        expected = self.episode.policy_environment_steps + 1
        if len(self.distance_trace_m) != expected:
            raise ValueError(
                f"distance trace 应包含初始点和每个环境步，共 {expected} 点，"
                f"实际 {len(self.distance_trace_m)}"
            )
        if not all(math.isfinite(value) and value >= 0.0 for value in self.distance_trace_m):
            raise ValueError("distance trace 必须是有限非负米制距离")
        if not math.isclose(
            self.distance_trace_m[-1],
            self.episode.final_tcp_to_object_distance_m,
            abs_tol=1e-6,
            rel_tol=1e-5,
        ):
            raise ValueError("distance trace 终点与 Atomic Episode final distance 不一致")

    @property
    def initial_tcp_to_object_distance_m(self) -> float:
        return self.distance_trace_m[0]

    @property
    def final_tcp_to_object_distance_m(self) -> float:
        return self.distance_trace_m[-1]

    @property
    def minimum_tcp_to_object_distance_m(self) -> float:
        return min(self.distance_trace_m)

    def to_dict(self) -> dict[str, Any]:
        value = self.episode.to_dict()
        value.update(
            {
                "initial_tcp_to_object_distance_m": (
                    self.initial_tcp_to_object_distance_m
                ),
                "minimum_tcp_to_object_distance_m": (
                    self.minimum_tcp_to_object_distance_m
                ),
                "distance_trace_m": list(self.distance_trace_m),
            }
        )
        return value


class _DistanceTraceController(_TrackingManiSkillController):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.distance_trace_m = [float(self.progress.outcome.tcp_to_object_distance_m)]

    def send_action(self, controller_action):
        steps_before = self.environment_steps
        super().send_action(controller_action)
        if self.environment_steps == steps_before + 1:
            self.distance_trace_m.append(
                float(self.progress.outcome.tcp_to_object_distance_m)
            )


def run_reach_diagnostic_episode(
    env: Any,
    runtime: Any,
    spec: RobotSpec,
    *,
    seed: int,
    instruction: str,
    sampling_seed_base: int,
    preparation: Any,
    max_policy_steps: int,
    temporal_ensemble_enabled: bool = True,
    recency_decay: float = 0.5,
    max_anomaly_replans: int = 3,
) -> ReachDiagnosticEpisodeResult:
    """复用正式执行/安全层，只增加逐环境步 distance trace。"""

    if max_policy_steps <= 0:
        raise ValueError("max_policy_steps 必须为正数")
    if preparation.progress.completed_skill_count != 0:
        raise ValueError("Atomic Reach 前置状态必须没有已完成技能")
    _reset_atomic_time_limit(env)
    started = time.monotonic()
    controller = _DistanceTraceController(
        env,
        spec,
        preparation.observation,
        preparation.tracker,
        preparation.progress,
    )
    loop = QwenVLAReplanLoop(
        runtime,
        RecedingHorizonChunkExecutor(spec),
        temporal_ensemble_enabled=temporal_ensemble_enabled,
        recency_decay=recency_decay,
        max_anomaly_replans=max_anomaly_replans,
    )
    observation_adapter = FrankaObservationAdapter(spec)
    replans = 0
    sampling_seeds: list[int] = []
    action_chunks = 0
    failure_stage: str | None = None
    error: str | None = None
    saturation_count = 0
    requested_abs_max: float | None = None
    applied_abs_max: float | None = None
    anomaly_replan_count = 0
    ensemble_max_buffer_size = 0
    ensemble_max_proposal_spread = 0.0
    ensemble_min_newest_weight: float | None = None

    def target_completed() -> bool:
        return controller.progress.completed_skill_count >= 1

    while (
        not controller.done
        and not target_completed()
        and controller.environment_steps < max_policy_steps
    ):
        replans += 1
        online_observation = _read_online_observation(
            controller.observation,
            env.unwrapped,
            observation_adapter,
            instruction,
        )
        result = loop.replan_and_execute(online_observation, controller)
        execution = result.execution
        anomaly_replan_count += int(execution.replan_required)
        if result.ensemble_trace is not None:
            ensemble_max_buffer_size = max(
                ensemble_max_buffer_size,
                result.ensemble_trace.buffer_size,
            )
            ensemble_max_proposal_spread = max(
                ensemble_max_proposal_spread,
                result.ensemble_trace.max_proposal_spread,
            )
            newest_weight = min(
                result.ensemble_trace.newest_normalized_weights[: spec.execute_steps]
            )
            ensemble_min_newest_weight = (
                newest_weight
                if ensemble_min_newest_weight is None
                else min(ensemble_min_newest_weight, newest_weight)
            )
        if result.sampling is not None:
            sampling_seeds.append(result.sampling.seed)
        action_chunks += int(result.action_chunk is not None)
        saturation_count += execution.correction_saturation_steps
        if execution.requested_correction_abs_max_rad is not None:
            requested_abs_max = max(
                requested_abs_max or 0.0,
                execution.requested_correction_abs_max_rad,
            )
        if execution.applied_correction_abs_max_rad is not None:
            applied_abs_max = max(
                applied_abs_max or 0.0,
                execution.applied_correction_abs_max_rad,
            )
        if not execution.success:
            failure_stage = execution.failure_stage
            error = execution.error
            break

    progress = controller.progress
    success = progress.completed_skill_count >= 1
    stage_categories = {
        "inference": "inference_error",
        "initial_observation": "controller_observation_error",
        "step_observation": "controller_observation_error",
        "chunk_safety": "action_safety_rejection",
        "step_safety": "action_safety_rejection",
        "controller_step": "controller_error",
        "replan_anomaly_exhausted": "replan_anomaly_exhausted",
    }
    failure_category = None if success else stage_categories.get(
        failure_stage,
        "reach_failed",
    )
    outcome = progress.outcome
    episode = AtomicSkillEpisodeResult(
        seed=seed,
        skill_name="reach",
        instruction=instruction,
        sampling_seed_base=sampling_seed_base,
        success=success,
        failure_category=failure_category,
        failure_stage=failure_stage,
        error=error,
        preparation_steps=preparation.preparation_steps,
        initial_completed_skill_count=0,
        final_completed_skill_count=progress.completed_skill_count,
        policy_environment_steps=controller.environment_steps,
        replans=replans,
        sampling_seeds=tuple(sampling_seeds),
        action_chunks=action_chunks,
        tracking_correction_saturation_count=saturation_count,
        tracking_correction_requested_abs_max_rad=requested_abs_max,
        tracking_correction_applied_abs_max_rad=applied_abs_max,
        final_is_grasped=outcome.grasped,
        final_tcp_to_object_distance_m=outcome.tcp_to_object_distance_m,
        final_object_height_above_support_m=outcome.object_height_above_support_m,
        final_object_to_goal_xy_distance_m=outcome.object_to_goal_xy_distance_m,
        final_object_to_goal_distance_m=outcome.object_to_goal_distance_m,
        final_object_linear_speed_m_s=outcome.object_linear_speed_m_s,
        final_object_angular_speed_rad_s=outcome.object_angular_speed_rad_s,
        wall_time_s=time.monotonic() - started,
        anomaly_replan_count=anomaly_replan_count,
        temporal_ensemble_max_buffer_size=ensemble_max_buffer_size,
        temporal_ensemble_max_proposal_spread=ensemble_max_proposal_spread,
        temporal_ensemble_min_newest_weight=ensemble_min_newest_weight,
    )
    return ReachDiagnosticEpisodeResult(
        episode=episode,
        distance_trace_m=tuple(controller.distance_trace_m),
    )


def summarize_reach_diagnostics(
    results: list[ReachDiagnosticEpisodeResult],
) -> dict[str, Any]:
    if not results:
        raise ValueError("不能汇总空 Reach diagnostic")
    seeds = [result.episode.seed for result in results]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Reach diagnostic 包含重复 seed")
    successes = sum(result.episode.success for result in results)
    return {
        "episodes": len(results),
        "successes": successes,
        "success_rate": successes / len(results),
        "mean_initial_tcp_to_object_distance_m": sum(
            result.initial_tcp_to_object_distance_m for result in results
        )
        / len(results),
        "mean_final_tcp_to_object_distance_m": sum(
            result.final_tcp_to_object_distance_m for result in results
        )
        / len(results),
        "minimum_tcp_to_object_distance_m": min(
            result.minimum_tcp_to_object_distance_m for result in results
        ),
        "mean_policy_environment_steps": sum(
            result.episode.policy_environment_steps for result in results
        )
        / len(results),
        "total_replans": sum(result.episode.replans for result in results),
        "seeds": seeds,
        "per_seed": [result.to_dict() for result in results],
    }


__all__ = [
    "ReachDiagnosticEpisodeResult",
    "run_reach_diagnostic_episode",
    "summarize_reach_diagnostics",
]
