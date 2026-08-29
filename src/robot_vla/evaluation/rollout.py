"""不依赖仿真器的闭环结果契约、失败分类和聚合。"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

from robot_vla.contracts import PICK_AND_PLACE_SKILLS
from robot_vla.execution.rtc import ChunkInferenceStrategy

ROLLOUT_FORMAT = "robot-vla-maniskill-rollout/v1"
ROLLOUT_SEED_GROUPS = ("test", "unseen")


@dataclass(frozen=True)
class RolloutEpisodeSpec:
    seed_group: str
    seed: int
    instruction: str

    def __post_init__(self) -> None:
        if self.seed_group not in ROLLOUT_SEED_GROUPS:
            raise ValueError(f"seed_group 必须是 {ROLLOUT_SEED_GROUPS}")
        if self.seed < 0 or not self.instruction.strip():
            raise ValueError("Rollout seed 必须非负且 instruction 不能为空")


@dataclass(frozen=True)
class RolloutEpisodeResult:
    seed_group: str
    seed: int
    instruction: str
    sampling_seed_base: int
    success: bool
    environment_success: bool
    predicate_success: bool
    failure_category: str | None
    failure_stage: str | None
    error: str | None
    environment_steps: int
    replans: int
    sampling_seeds: tuple[int, ...]
    action_chunks: int
    normalized_action_abs_max: float | None
    physical_arm_delta_abs_max_rad: float | None
    gripper_target_min: float | None
    gripper_target_max: float | None
    completed_skill_count: int
    skill_completed: tuple[bool, ...]
    terminated: bool
    truncated: bool
    final_is_grasped: bool
    stable_grasp_steps: int
    stable_place_steps: int
    final_tcp_to_object_distance_m: float
    final_object_height_above_support_m: float
    final_object_to_goal_xy_distance_m: float
    final_object_to_goal_distance_m: float
    final_object_linear_speed_m_s: float
    final_object_angular_speed_rad_s: float
    wall_time_s: float
    execution_diagnostic: dict[str, Any] | None = None
    tracking_correction_saturation_count: int = 0
    tracking_correction_requested_abs_max_rad: float | None = None
    tracking_correction_applied_abs_max_rad: float | None = None
    anomaly_replan_count: int = 0
    temporal_ensemble_max_buffer_size: int = 0
    temporal_ensemble_max_proposal_spread: float = 0.0
    temporal_ensemble_min_newest_weight: float | None = None
    inference_strategy: str = ChunkInferenceStrategy.TEMPORAL_ENSEMBLE.value
    skill_completion_environment_steps: tuple[int | None, ...] = (None,) * len(
        PICK_AND_PLACE_SKILLS
    )
    replan_traces: tuple[dict[str, Any], ...] = ()
    format: str = ROLLOUT_FORMAT

    def __post_init__(self) -> None:
        RolloutEpisodeSpec(self.seed_group, self.seed, self.instruction)
        if self.format != ROLLOUT_FORMAT:
            raise ValueError(f"Rollout format 必须为 {ROLLOUT_FORMAT}")
        if self.sampling_seed_base < 0 or self.environment_steps < 0 or self.replans < 0:
            raise ValueError("采样 seed、环境步数和 Replan 数不能为负数")
        if self.action_chunks < 0 or self.action_chunks > self.replans:
            raise ValueError("action_chunks 必须位于 [0,replans]")
        action_statistics = (
            self.normalized_action_abs_max,
            self.physical_arm_delta_abs_max_rad,
            self.gripper_target_min,
            self.gripper_target_max,
        )
        if self.action_chunks == 0 and any(value is not None for value in action_statistics):
            raise ValueError("没有 Action Chunk 时不能记录 Action 统计")
        if self.action_chunks > 0:
            if any(value is None or not math.isfinite(value) for value in action_statistics):
                raise ValueError("Action Chunk 统计必须是有限数值")
            if not 0.0 <= self.normalized_action_abs_max <= 1.0 + 1e-5:
                raise ValueError("normalized Action 最大绝对值超出范围")
            if self.physical_arm_delta_abs_max_rad < 0.0:
                raise ValueError("physical arm delta 最大绝对值不能为负数")
            if not 0.0 <= self.gripper_target_min <= self.gripper_target_max <= 1.0:
                raise ValueError("gripper target 统计超出 [0,1]")
        if not 0 <= self.completed_skill_count <= len(PICK_AND_PLACE_SKILLS):
            raise ValueError("completed_skill_count 超出原子技能范围")
        if len(self.skill_completed) != len(PICK_AND_PLACE_SKILLS):
            raise ValueError("skill_completed 长度与原子技能不一致")
        ChunkInferenceStrategy(self.inference_strategy)
        if len(self.skill_completion_environment_steps) != len(PICK_AND_PLACE_SKILLS):
            raise ValueError("skill completion step 长度与原子技能不一致")
        expected_skills = tuple(
            index < self.completed_skill_count for index in range(len(PICK_AND_PLACE_SKILLS))
        )
        if self.skill_completed != expected_skills:
            raise ValueError("skill_completed 必须是单调前缀")
        if any(step is not None for step in self.skill_completion_environment_steps):
            for index, step in enumerate(self.skill_completion_environment_steps):
                if index < self.completed_skill_count:
                    if step is None or step < 0 or step > self.environment_steps:
                        raise ValueError("已完成技能必须记录有效 environment step")
                elif step is not None:
                    raise ValueError("未完成技能不能记录 completion step")
            completed_steps = self.skill_completion_environment_steps[: self.completed_skill_count]
            if any(left > right for left, right in pairwise(completed_steps)):
                raise ValueError("skill completion step 必须单调")
        if len(self.replan_traces) > self.replans:
            raise ValueError("replan trace 数不能超过 replans")
        if self.success != (self.environment_success and self.predicate_success):
            raise ValueError("完整成功必须同时满足环境和项目 Predicate")
        if self.success != (self.failure_category is None):
            raise ValueError("成功 Episode 不能有失败类别，失败 Episode 必须有失败类别")
        if self.wall_time_s < 0 or not math.isfinite(self.wall_time_s):
            raise ValueError("wall_time_s 必须是有限非负数")
        if self.execution_diagnostic is not None:
            if self.failure_stage not in {
                "chunk_safety",
                "step_safety",
                "replan_anomaly_exhausted",
            }:
                raise ValueError("execution_diagnostic 目前只允许记录 Action 安全拒绝")
            if not isinstance(self.execution_diagnostic.get("kind"), str):
                raise ValueError("execution_diagnostic 必须包含字符串 kind")
        if self.tracking_correction_saturation_count < 0:
            raise ValueError("tracking correction 饱和次数不能为负数")
        if self.anomaly_replan_count < 0 or self.temporal_ensemble_max_buffer_size < 0:
            raise ValueError("anomaly/temporal ensemble 计数不能为负数")
        if (
            not math.isfinite(self.temporal_ensemble_max_proposal_spread)
            or self.temporal_ensemble_max_proposal_spread < 0
        ):
            raise ValueError("temporal ensemble proposal spread 必须是有限非负数")
        if self.temporal_ensemble_min_newest_weight is not None and not (
            0.0 < self.temporal_ensemble_min_newest_weight <= 1.0
        ):
            raise ValueError("temporal ensemble newest weight 必须位于 (0,1]")
        correction_statistics = (
            self.tracking_correction_requested_abs_max_rad,
            self.tracking_correction_applied_abs_max_rad,
        )
        if any(
            value is not None and (value < 0 or not math.isfinite(value))
            for value in correction_statistics
        ):
            raise ValueError("tracking correction 统计必须是有限非负数")
        if (
            self.tracking_correction_requested_abs_max_rad is not None
            and self.tracking_correction_applied_abs_max_rad is not None
            and self.tracking_correction_applied_abs_max_rad
            > self.tracking_correction_requested_abs_max_rad + 1e-6
        ):
            raise ValueError("实际 tracking correction 不能大于请求修正")
        finite_values = (
            self.final_tcp_to_object_distance_m,
            self.final_object_height_above_support_m,
            self.final_object_to_goal_xy_distance_m,
            self.final_object_to_goal_distance_m,
            self.final_object_linear_speed_m_s,
            self.final_object_angular_speed_rad_s,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("最终 Outcome 数值必须有限")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RolloutEpisodeResult:
        data = dict(value)
        data["sampling_seeds"] = tuple(int(seed) for seed in data["sampling_seeds"])
        data["skill_completed"] = tuple(bool(item) for item in data["skill_completed"])
        data["skill_completion_environment_steps"] = tuple(
            None if step is None else int(step)
            for step in data.get(
                "skill_completion_environment_steps",
                (None,) * len(PICK_AND_PLACE_SKILLS),
            )
        )
        data["replan_traces"] = tuple(
            dict(trace) for trace in data.get("replan_traces", ())
        )
        return cls(**data)


def classify_rollout_failure(
    *,
    completed_skill_count: int,
    predicate_success: bool,
    environment_success: bool,
    failure_stage: str | None,
    final_is_grasped: bool,
    final_object_to_goal_distance_m: float,
    place_distance_m: float,
    final_object_linear_speed_m_s: float,
    static_linear_speed_m_s: float,
    final_object_angular_speed_rad_s: float,
    static_angular_speed_rad_s: float,
) -> str | None:
    """优先报告系统错误；正常超时则归因到最早未完成的原子技能。"""

    if predicate_success and environment_success:
        return None
    if predicate_success != environment_success:
        return "predicate_mismatch"
    stage_categories = {
        "inference": "inference_error",
        "initial_observation": "controller_observation_error",
        "step_observation": "controller_observation_error",
        "chunk_safety": "action_safety_rejection",
        "step_safety": "action_safety_rejection",
        "controller_step": "controller_error",
        "replan_anomaly_exhausted": "replan_anomaly_exhausted",
        "rollout": "rollout_error",
    }
    if failure_stage is not None:
        return stage_categories.get(failure_stage, "execution_error")
    if completed_skill_count == 0:
        return "reach_failed"
    if completed_skill_count == 1:
        return "grasp_failed"
    if completed_skill_count == 2:
        return "lift_failed"
    if completed_skill_count == 3:
        return "transport_failed"
    if completed_skill_count == 4:
        if final_is_grasped:
            return "release_failed"
        if final_object_to_goal_distance_m > place_distance_m:
            return "place_position_failed"
        if (
            final_object_linear_speed_m_s > static_linear_speed_m_s
            or final_object_angular_speed_rad_s > static_angular_speed_rad_s
        ):
            return "place_stability_failed"
        return "place_stability_timeout"
    return "predicate_mismatch"


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        raise ValueError("Wilson interval 需要非空样本")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _summarize_group(results: list[RolloutEpisodeResult]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        raise ValueError("不能聚合空 Rollout 结果")
    successes = sum(result.success for result in results)
    skill_successes = {
        skill: sum(result.skill_completed[index] for result in results)
        for index, skill in enumerate(PICK_AND_PLACE_SKILLS)
    }
    reach_count = skill_successes["reach"]
    grasp_count = skill_successes["grasp"]
    lift_count = skill_successes["lift"]
    transport_count = skill_successes["transport"]

    def mean_or_none(values: list[int]) -> float | None:
        return None if not values else sum(values) / len(values)

    reach_steps = [
        result.skill_completion_environment_steps[0]
        for result in results
        if result.skill_completion_environment_steps[0] is not None
    ]
    reach_to_grasp_steps = [
        result.skill_completion_environment_steps[1]
        - result.skill_completion_environment_steps[0]
        for result in results
        if result.skill_completion_environment_steps[0] is not None
        and result.skill_completion_environment_steps[1] is not None
    ]
    lift_to_transport_steps = [
        result.skill_completion_environment_steps[3]
        - result.skill_completion_environment_steps[2]
        for result in results
        if result.skill_completion_environment_steps[2] is not None
        and result.skill_completion_environment_steps[3] is not None
    ]
    replan_traces = [trace for result in results for trace in result.replan_traces]
    rtc_traces = [trace for trace in replan_traces if trace.get("rtc_enabled")]

    def max_trace_value(name: str) -> float | None:
        values = [
            float(trace[name])
            for trace in rtc_traces
            if trace.get(name) is not None
        ]
        return None if not values else max(values)

    return {
        "episodes": total,
        "successes": successes,
        "success_rate": successes / total,
        "success_rate_wilson_95": _wilson_interval(successes, total),
        "skill_successes": skill_successes,
        "skill_success_rates": {
            skill: count / total for skill, count in skill_successes.items()
        },
        "grasp_given_reach": {
            "numerator": grasp_count,
            "denominator": reach_count,
            "rate": None if reach_count == 0 else grasp_count / reach_count,
            "wilson_95": None
            if reach_count == 0
            else _wilson_interval(grasp_count, reach_count),
        },
        "transport_given_lift": {
            "numerator": transport_count,
            "denominator": lift_count,
            "rate": None if lift_count == 0 else transport_count / lift_count,
            "wilson_95": None
            if lift_count == 0
            else _wilson_interval(transport_count, lift_count),
        },
        "mean_completed_skill_count": sum(
            result.completed_skill_count for result in results
        )
        / total,
        "mean_steps_to_reach": mean_or_none(reach_steps),
        "mean_steps_reach_to_grasp": mean_or_none(reach_to_grasp_steps),
        "mean_steps_lift_to_transport": mean_or_none(lift_to_transport_steps),
        "inference_strategy_counts": dict(
            sorted(Counter(result.inference_strategy for result in results).items())
        ),
        "rtc_replans": len(rtc_traces),
        "rtc_replans_with_previous_chunk": sum(
            bool(trace.get("previous_chunk_available")) for trace in rtc_traces
        ),
        "rtc_raw_max_abs_disagreement": max_trace_value("raw_max_abs_disagreement"),
        "rtc_prefix_max_abs_disagreement": max_trace_value(
            "prefix_max_abs_disagreement"
        ),
        "rtc_prefix_max_abs_correction": max_trace_value("prefix_max_abs_correction"),
        "rtc_future_max_abs_correction": max_trace_value("future_max_abs_correction"),
        "failure_counts": dict(
            sorted(Counter(result.failure_category for result in results if not result.success).items())
        ),
        "mean_environment_steps": sum(result.environment_steps for result in results) / total,
        "mean_replans": sum(result.replans for result in results) / total,
        "tracking_correction_saturation_count": sum(
            result.tracking_correction_saturation_count for result in results
        ),
        "episodes_with_tracking_correction_saturation": sum(
            result.tracking_correction_saturation_count > 0 for result in results
        ),
        "anomaly_replan_count": sum(result.anomaly_replan_count for result in results),
        "episodes_with_anomaly_replan": sum(
            result.anomaly_replan_count > 0 for result in results
        ),
        "temporal_ensemble_max_buffer_size": max(
            result.temporal_ensemble_max_buffer_size for result in results
        ),
        "temporal_ensemble_max_proposal_spread": max(
            result.temporal_ensemble_max_proposal_spread for result in results
        ),
        "tracking_correction_requested_abs_max_rad": max(
            (
                result.tracking_correction_requested_abs_max_rad
                for result in results
                if result.tracking_correction_requested_abs_max_rad is not None
            ),
            default=None,
        ),
        "tracking_correction_applied_abs_max_rad": max(
            (
                result.tracking_correction_applied_abs_max_rad
                for result in results
                if result.tracking_correction_applied_abs_max_rad is not None
            ),
            default=None,
        ),
        "total_wall_time_s": sum(result.wall_time_s for result in results),
    }


def summarize_rollouts(results: list[RolloutEpisodeResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("不能汇总空 Rollout")
    identities = [(result.seed_group, result.seed) for result in results]
    if len(identities) != len(set(identities)):
        raise ValueError("Rollout 结果包含重复 seed_group/seed")
    groups = {
        group: _summarize_group([result for result in results if result.seed_group == group])
        for group in ROLLOUT_SEED_GROUPS
        if any(result.seed_group == group for result in results)
    }
    return {
        "format": ROLLOUT_FORMAT,
        "overall": _summarize_group(results),
        "groups": groups,
    }
