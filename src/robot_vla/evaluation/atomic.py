"""独立原子技能闭环评估结果契约与聚合。"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from robot_vla.contracts import PICK_AND_PLACE_SKILLS
from robot_vla.execution.rtc import ChunkInferenceStrategy

ATOMIC_ROLLOUT_FORMAT = "robot-vla-maniskill-atomic-rollout/v1"


def derive_atomic_sampling_seed(base_seed: int, seed: int, skill_name: str) -> int:
    if base_seed < 0 or seed < 0 or skill_name not in PICK_AND_PLACE_SKILLS:
        raise ValueError("原子评估 sampling seed 或技能无效")
    identity = f"{base_seed}:atomic:{skill_name}:{seed}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % (2**63 - 1)


@dataclass(frozen=True)
class AtomicSkillEpisodeResult:
    seed: int
    skill_name: str
    instruction: str
    sampling_seed_base: int
    success: bool
    failure_category: str | None
    failure_stage: str | None
    error: str | None
    preparation_steps: int
    initial_completed_skill_count: int
    final_completed_skill_count: int
    policy_environment_steps: int
    replans: int
    sampling_seeds: tuple[int, ...]
    action_chunks: int
    tracking_correction_saturation_count: int
    tracking_correction_requested_abs_max_rad: float | None
    tracking_correction_applied_abs_max_rad: float | None
    final_is_grasped: bool
    final_tcp_to_object_distance_m: float
    final_object_height_above_support_m: float
    final_object_to_goal_xy_distance_m: float
    final_object_to_goal_distance_m: float
    final_object_linear_speed_m_s: float
    final_object_angular_speed_rad_s: float
    wall_time_s: float
    anomaly_replan_count: int = 0
    temporal_ensemble_max_buffer_size: int = 0
    temporal_ensemble_max_proposal_spread: float = 0.0
    temporal_ensemble_min_newest_weight: float | None = None
    inference_strategy: str = ChunkInferenceStrategy.TEMPORAL_ENSEMBLE.value
    replan_traces: tuple[dict[str, Any], ...] = ()
    format: str = ATOMIC_ROLLOUT_FORMAT

    def __post_init__(self) -> None:
        if self.format != ATOMIC_ROLLOUT_FORMAT:
            raise ValueError(f"原子 Rollout format 必须为 {ATOMIC_ROLLOUT_FORMAT}")
        if self.seed < 0 or self.sampling_seed_base < 0 or not self.instruction.strip():
            raise ValueError("原子 Rollout seed 和 instruction 无效")
        ChunkInferenceStrategy(self.inference_strategy)
        if self.skill_name not in PICK_AND_PLACE_SKILLS:
            raise ValueError(f"未知原子技能: {self.skill_name}")
        target = PICK_AND_PLACE_SKILLS.index(self.skill_name)
        if self.initial_completed_skill_count != target:
            raise ValueError("原子评估起始状态必须精确完成目标技能的全部前置技能")
        if not target <= self.final_completed_skill_count <= len(PICK_AND_PLACE_SKILLS):
            raise ValueError("原子评估最终技能进度无效")
        if self.success != (self.final_completed_skill_count >= target + 1):
            raise ValueError("原子技能成功必须与最终技能进度一致")
        if self.success != (self.failure_category is None):
            raise ValueError("原子技能成功与失败类别不一致")
        counts = (
            self.preparation_steps,
            self.policy_environment_steps,
            self.replans,
            self.action_chunks,
            self.tracking_correction_saturation_count,
        )
        if any(value < 0 for value in counts) or self.action_chunks > self.replans:
            raise ValueError("原子评估步数或计数无效")
        if self.anomaly_replan_count < 0 or self.temporal_ensemble_max_buffer_size < 0:
            raise ValueError("原子评估 anomaly/temporal ensemble 计数无效")
        if len(self.replan_traces) > self.replans:
            raise ValueError("原子 replan trace 数不能超过 replans")
        if (
            not math.isfinite(self.temporal_ensemble_max_proposal_spread)
            or self.temporal_ensemble_max_proposal_spread < 0
        ):
            raise ValueError("原子评估 temporal ensemble spread 无效")
        if self.temporal_ensemble_min_newest_weight is not None and not (
            0.0 < self.temporal_ensemble_min_newest_weight <= 1.0
        ):
            raise ValueError("原子评估 temporal ensemble newest weight 无效")
        correction_values = (
            self.tracking_correction_requested_abs_max_rad,
            self.tracking_correction_applied_abs_max_rad,
        )
        if any(
            value is not None and (value < 0 or not math.isfinite(value))
            for value in correction_values
        ):
            raise ValueError("原子评估 tracking correction 统计无效")
        finite_values = (
            self.final_tcp_to_object_distance_m,
            self.final_object_height_above_support_m,
            self.final_object_to_goal_xy_distance_m,
            self.final_object_to_goal_distance_m,
            self.final_object_linear_speed_m_s,
            self.final_object_angular_speed_rad_s,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("原子评估 Outcome 必须是有限数值")
        if self.wall_time_s < 0 or not math.isfinite(self.wall_time_s):
            raise ValueError("原子评估耗时必须是有限非负数")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AtomicSkillEpisodeResult:
        data = dict(value)
        data["sampling_seeds"] = tuple(int(seed) for seed in data["sampling_seeds"])
        data["replan_traces"] = tuple(
            dict(trace) for trace in data.get("replan_traces", ())
        )
        return cls(**data)


def summarize_atomic_rollouts(results: list[AtomicSkillEpisodeResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("不能汇总空原子 Rollout")
    identities = [(result.skill_name, result.seed) for result in results]
    if len(identities) != len(set(identities)):
        raise ValueError("原子 Rollout 包含重复 skill/seed")
    groups: dict[str, Any] = {}
    for skill_name in PICK_AND_PLACE_SKILLS:
        selected = [result for result in results if result.skill_name == skill_name]
        if not selected:
            continue
        successes = sum(result.success for result in selected)
        rtc_traces = [
            trace
            for result in selected
            for trace in result.replan_traces
            if trace.get("rtc_enabled")
        ]
        groups[skill_name] = {
            "episodes": len(selected),
            "successes": successes,
            "success_rate": successes / len(selected),
            "failure_counts": dict(
                sorted(
                    Counter(
                        result.failure_category for result in selected if not result.success
                    ).items()
                )
            ),
            "mean_policy_environment_steps": sum(
                result.policy_environment_steps for result in selected
            )
            / len(selected),
            "tracking_correction_saturation_count": sum(
                result.tracking_correction_saturation_count for result in selected
            ),
            "anomaly_replan_count": sum(
                result.anomaly_replan_count for result in selected
            ),
            "temporal_ensemble_max_buffer_size": max(
                result.temporal_ensemble_max_buffer_size for result in selected
            ),
            "temporal_ensemble_max_proposal_spread": max(
                result.temporal_ensemble_max_proposal_spread for result in selected
            ),
            "inference_strategy_counts": dict(
                sorted(Counter(result.inference_strategy for result in selected).items())
            ),
            "rtc_replans": len(rtc_traces),
            "rtc_replans_with_previous_chunk": sum(
                bool(trace.get("previous_chunk_available")) for trace in rtc_traces
            ),
        }
    return {
        "format": ATOMIC_ROLLOUT_FORMAT,
        "episodes": len(results),
        "successes": sum(result.success for result in results),
        "groups": groups,
    }


__all__ = [
    "ATOMIC_ROLLOUT_FORMAT",
    "AtomicSkillEpisodeResult",
    "derive_atomic_sampling_seed",
    "summarize_atomic_rollouts",
]
