"""E012 replay/DAgger 配对闭环与原子评估的纯函数分析。"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from robot_vla.contracts import PICK_AND_PLACE_SKILLS
from robot_vla.evaluation.atomic import AtomicSkillEpisodeResult
from robot_vla.evaluation.rollout import RolloutEpisodeResult, summarize_rollouts

PAIRED_EVALUATION_FORMAT = "robot-vla-e012-paired-evaluation/v1"
PAIRED_PROTOCOL_SEEDS = {
    "checkpoint-validation": {
        "full_chain": tuple(range(31_000, 31_020)),
        "atomic": tuple(range(31_020, 31_025)),
    },
    "stage-a": {
        "full_chain": tuple(range(32_000, 32_020)),
        "atomic": tuple(range(32_020, 32_025)),
    },
    "stage-a-plus-b": {
        "full_chain": tuple(range(32_000, 32_020))
        + tuple(range(32_100, 32_130)),
        "atomic": tuple(range(32_020, 32_025)),
    },
}

_SYSTEM_FAILURE_CATEGORIES = {
    "controller_error",
    "controller_observation_error",
    "execution_error",
    "inference_error",
    "predicate_mismatch",
    "rollout_error",
}


def _wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> list[float] | None:
    if total == 0:
        return None
    if total < 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval 计数无效")
    rate = successes / total
    scale = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / scale
    half = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / scale
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def exact_paired_test(dagger_wins: int, replay_wins: int) -> dict[str, Any]:
    """对 discordant pair 做双侧 exact McNemar/binomial 检验。"""

    if dagger_wins < 0 or replay_wins < 0:
        raise ValueError("paired wins 不能为负数")
    discordant = dagger_wins + replay_wins
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(dagger_wins, replay_wins) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "method": "two-sided-exact-mcnemar-binomial",
        "discordant_pairs": discordant,
        "p_value": p_value,
    }


def _paired_counts(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    dagger_wins = replay_wins = success_ties = failure_ties = 0
    for replay_success, dagger_success in pairs:
        if dagger_success and not replay_success:
            dagger_wins += 1
        elif replay_success and not dagger_success:
            replay_wins += 1
        elif replay_success:
            success_ties += 1
        else:
            failure_ties += 1
    return {
        "pairs": len(pairs),
        "dagger_wins": dagger_wins,
        "replay_wins": replay_wins,
        "net_dagger_wins": dagger_wins - replay_wins,
        "success_ties": success_ties,
        "failure_ties": failure_ties,
        "exact_paired_test": exact_paired_test(dagger_wins, replay_wins),
    }


def _sampling_prefix_matches(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    common = min(len(left), len(right))
    return left[:common] == right[:common]


def _rollout_issue_flags(result: RolloutEpisodeResult) -> dict[str, bool]:
    failure_category = result.failure_category
    return {
        "system": result.error is not None
        or failure_category in _SYSTEM_FAILURE_CATEGORIES,
        "safety": failure_category == "action_safety_rejection",
        "tracking": result.tracking_correction_saturation_count > 0,
        "anomaly": result.anomaly_replan_count > 0
        or failure_category == "replan_anomaly_exhausted",
    }


def _atomic_issue_flags(result: AtomicSkillEpisodeResult) -> dict[str, bool]:
    failure_category = result.failure_category
    return {
        "system": result.error is not None
        or failure_category in _SYSTEM_FAILURE_CATEGORIES,
        "safety": failure_category == "action_safety_rejection",
        "tracking": result.tracking_correction_saturation_count > 0,
        "anomaly": result.anomaly_replan_count > 0
        or failure_category == "replan_anomaly_exhausted",
    }


def _count_issues(flags: list[dict[str, bool]]) -> dict[str, int]:
    return {
        f"episodes_with_{name}": sum(row[name] for row in flags)
        for name in ("system", "safety", "tracking", "anomaly")
    }


def _new_issue_counts(
    replay_flags: list[dict[str, bool]], dagger_flags: list[dict[str, bool]]
) -> dict[str, int]:
    return {
        f"new_{name}_episodes": sum(
            dagger[name] and not replay[name]
            for replay, dagger in zip(replay_flags, dagger_flags, strict=True)
        )
        for name in ("system", "safety", "tracking", "anomaly")
    }


def _rate(successes: int, total: int) -> dict[str, Any]:
    return {
        "numerator": successes,
        "denominator": total,
        "rate": None if total == 0 else successes / total,
        "wilson_95": _wilson_interval(successes, total),
    }


def _model_full_chain_summary(results: list[RolloutEpisodeResult]) -> dict[str, Any]:
    overall = summarize_rollouts(results)["overall"]
    total = len(results)
    unconditional = {
        skill: _rate(
            sum(result.skill_completed[index] for result in results),
            total,
        )
        for index, skill in enumerate(PICK_AND_PLACE_SKILLS)
    }
    unconditional["full"] = _rate(sum(result.success for result in results), total)
    issue_flags = [_rollout_issue_flags(result) for result in results]
    return {
        "episodes": total,
        "unconditional": unconditional,
        "conditionals": {
            "grasp_given_reach": overall["grasp_given_reach"],
            "lift_given_grasp": overall["lift_given_grasp"],
            "transport_given_lift": overall["transport_given_lift"],
        },
        "mean_completed_skill_count": overall["mean_completed_skill_count"],
        "stage_timing": {
            "mean_steps_to_reach": overall["mean_steps_to_reach"],
            "mean_steps_reach_to_grasp": overall["mean_steps_reach_to_grasp"],
            "mean_steps_lift_to_transport": overall[
                "mean_steps_lift_to_transport"
            ],
        },
        "issues": {
            **_count_issues(issue_flags),
            "tracking_correction_saturation_count": sum(
                result.tracking_correction_saturation_count for result in results
            ),
            "anomaly_replan_count": sum(
                result.anomaly_replan_count for result in results
            ),
        },
        "failure_counts": dict(
            sorted(
                Counter(
                    result.failure_category
                    for result in results
                    if result.failure_category is not None
                ).items()
            )
        ),
    }


def analyze_full_chain_pair(
    replay_results: list[RolloutEpisodeResult],
    dagger_results: list[RolloutEpisodeResult],
) -> dict[str, Any]:
    if not replay_results or not dagger_results:
        raise ValueError("paired full-chain 结果不能为空")
    replay_by_id = {
        (result.seed_group, result.seed): result for result in replay_results
    }
    dagger_by_id = {
        (result.seed_group, result.seed): result for result in dagger_results
    }
    if len(replay_by_id) != len(replay_results) or len(dagger_by_id) != len(
        dagger_results
    ):
        raise ValueError("paired full-chain 结果包含重复 seed identity")
    if set(replay_by_id) != set(dagger_by_id):
        raise ValueError("replay/DAgger full-chain seed identity 不一致")

    per_seed: list[dict[str, Any]] = []
    replay_issue_flags: list[dict[str, bool]] = []
    dagger_issue_flags: list[dict[str, bool]] = []
    for identity in sorted(replay_by_id):
        replay = replay_by_id[identity]
        dagger = dagger_by_id[identity]
        if replay.instruction != dagger.instruction:
            raise ValueError(f"paired seed {identity} 的 instruction 不一致")
        if replay.sampling_seed_base != dagger.sampling_seed_base:
            raise ValueError(f"paired seed {identity} 的 Flow sampling base 不一致")
        if not _sampling_prefix_matches(replay.sampling_seeds, dagger.sampling_seeds):
            raise ValueError(f"paired seed {identity} 的 Flow sampling seed prefix 不一致")
        replay_flags = _rollout_issue_flags(replay)
        dagger_flags = _rollout_issue_flags(dagger)
        replay_issue_flags.append(replay_flags)
        dagger_issue_flags.append(dagger_flags)
        per_seed.append(
            {
                "seed_group": replay.seed_group,
                "seed": replay.seed,
                "replay": {
                    "completed_skill_count": replay.completed_skill_count,
                    "skill_completed": dict(
                        zip(PICK_AND_PLACE_SKILLS, replay.skill_completed, strict=True)
                    ),
                    "success": replay.success,
                    "failure_category": replay.failure_category,
                    "issues": replay_flags,
                },
                "dagger": {
                    "completed_skill_count": dagger.completed_skill_count,
                    "skill_completed": dict(
                        zip(PICK_AND_PLACE_SKILLS, dagger.skill_completed, strict=True)
                    ),
                    "success": dagger.success,
                    "failure_category": dagger.failure_category,
                    "issues": dagger_flags,
                },
                "completed_skill_delta": dagger.completed_skill_count
                - replay.completed_skill_count,
            }
        )

    unconditional_paired = {
        skill: _paired_counts(
            [
                (replay.skill_completed[index], dagger.skill_completed[index])
                for replay, dagger in (
                    (replay_by_id[identity], dagger_by_id[identity])
                    for identity in sorted(replay_by_id)
                )
            ]
        )
        for index, skill in enumerate(PICK_AND_PLACE_SKILLS)
    }
    unconditional_paired["full"] = _paired_counts(
        [
            (replay_by_id[identity].success, dagger_by_id[identity].success)
            for identity in sorted(replay_by_id)
        ]
    )

    handoffs: dict[str, Any] = {}
    for skill_index in range(1, 4):
        skill = PICK_AND_PLACE_SKILLS[skill_index]
        predecessor = PICK_AND_PLACE_SKILLS[skill_index - 1]
        common_identities = [
            identity
            for identity in sorted(replay_by_id)
            if replay_by_id[identity].skill_completed[skill_index - 1]
            and dagger_by_id[identity].skill_completed[skill_index - 1]
        ]
        handoffs[skill] = {
            "predecessor": predecessor,
            "common_predecessor_support": len(common_identities),
            "common_predecessor_seeds": [
                {"seed_group": group, "seed": seed}
                for group, seed in common_identities
            ],
            "handoff_on_common_support": _paired_counts(
                [
                    (
                        replay_by_id[identity].skill_completed[skill_index],
                        dagger_by_id[identity].skill_completed[skill_index],
                    )
                    for identity in common_identities
                ]
            ),
            "predecessor_paired_on_all_seeds": unconditional_paired[predecessor],
        }

    replay_mean = sum(
        result.completed_skill_count for result in replay_results
    ) / len(replay_results)
    dagger_mean = sum(
        result.completed_skill_count for result in dagger_results
    ) / len(dagger_results)
    return {
        "replay": _model_full_chain_summary(replay_results),
        "dagger": _model_full_chain_summary(dagger_results),
        "unconditional_paired": unconditional_paired,
        "handoffs": handoffs,
        "mean_completed_skill_count_delta": dagger_mean - replay_mean,
        "completed_skill_count_paired": {
            "dagger_wins": sum(row["completed_skill_delta"] > 0 for row in per_seed),
            "replay_wins": sum(row["completed_skill_delta"] < 0 for row in per_seed),
            "ties": sum(row["completed_skill_delta"] == 0 for row in per_seed),
        },
        "issue_deltas": {
            **{
                key: _count_issues(dagger_issue_flags)[key]
                - _count_issues(replay_issue_flags)[key]
                for key in _count_issues(replay_issue_flags)
            },
            **_new_issue_counts(replay_issue_flags, dagger_issue_flags),
            "tracking_correction_saturation_count_delta": sum(
                result.tracking_correction_saturation_count
                for result in dagger_results
            )
            - sum(
                result.tracking_correction_saturation_count
                for result in replay_results
            ),
            "anomaly_replan_count_delta": sum(
                result.anomaly_replan_count for result in dagger_results
            )
            - sum(result.anomaly_replan_count for result in replay_results),
        },
        "per_seed": per_seed,
    }


def analyze_atomic_pair(
    replay_results: list[AtomicSkillEpisodeResult],
    dagger_results: list[AtomicSkillEpisodeResult],
) -> dict[str, Any]:
    if not replay_results or not dagger_results:
        raise ValueError("paired atomic 结果不能为空")
    replay_by_id = {(row.skill_name, row.seed): row for row in replay_results}
    dagger_by_id = {(row.skill_name, row.seed): row for row in dagger_results}
    if len(replay_by_id) != len(replay_results) or len(dagger_by_id) != len(
        dagger_results
    ):
        raise ValueError("paired atomic 结果包含重复 skill/seed")
    if set(replay_by_id) != set(dagger_by_id):
        raise ValueError("replay/DAgger atomic identity 不一致")

    per_seed: list[dict[str, Any]] = []
    replay_issue_flags: list[dict[str, bool]] = []
    dagger_issue_flags: list[dict[str, bool]] = []
    for identity in sorted(replay_by_id):
        replay = replay_by_id[identity]
        dagger = dagger_by_id[identity]
        if replay.instruction != dagger.instruction:
            raise ValueError(f"paired atomic {identity} 的 instruction 不一致")
        if replay.sampling_seed_base != dagger.sampling_seed_base:
            raise ValueError(f"paired atomic {identity} 的 Flow sampling base 不一致")
        if not _sampling_prefix_matches(replay.sampling_seeds, dagger.sampling_seeds):
            raise ValueError(f"paired atomic {identity} 的 Flow sampling seed prefix 不一致")
        replay_flags = _atomic_issue_flags(replay)
        dagger_flags = _atomic_issue_flags(dagger)
        replay_issue_flags.append(replay_flags)
        dagger_issue_flags.append(dagger_flags)
        per_seed.append(
            {
                "skill": replay.skill_name,
                "seed": replay.seed,
                "replay_success": replay.success,
                "dagger_success": dagger.success,
                "replay_failure_category": replay.failure_category,
                "dagger_failure_category": dagger.failure_category,
                "replay_issues": replay_flags,
                "dagger_issues": dagger_flags,
            }
        )
    by_skill = {
        skill: _paired_counts(
            [
                (replay_by_id[(skill, seed)].success, dagger_by_id[(skill, seed)].success)
                for seed in sorted(
                    seed for row_skill, seed in replay_by_id if row_skill == skill
                )
            ]
        )
        for skill in PICK_AND_PLACE_SKILLS
        if any(row_skill == skill for row_skill, _ in replay_by_id)
    }
    return {
        "by_skill": by_skill,
        "issue_deltas": {
            **{
                key: _count_issues(dagger_issue_flags)[key]
                - _count_issues(replay_issue_flags)[key]
                for key in _count_issues(replay_issue_flags)
            },
            **_new_issue_counts(replay_issue_flags, dagger_issue_flags),
            "tracking_correction_saturation_count_delta": sum(
                result.tracking_correction_saturation_count
                for result in dagger_results
            )
            - sum(
                result.tracking_correction_saturation_count
                for result in replay_results
            ),
            "anomaly_replan_count_delta": sum(
                result.anomaly_replan_count for result in dagger_results
            )
            - sum(result.anomaly_replan_count for result in replay_results),
        },
        "per_seed": per_seed,
    }


def _validate_protocol_seeds(
    protocol: str,
    full_chain: list[RolloutEpisodeResult],
    atomic: list[AtomicSkillEpisodeResult] | None,
) -> None:
    if protocol == "descriptive":
        return
    try:
        expected = PAIRED_PROTOCOL_SEEDS[protocol]
    except KeyError as error:
        raise ValueError(f"未知 E012 paired protocol: {protocol}") from error
    observed_full = {(row.seed_group, row.seed) for row in full_chain}
    expected_full = {("unseen", seed) for seed in expected["full_chain"]}
    if observed_full != expected_full:
        raise ValueError(f"{protocol} full-chain seed registry 不一致")
    if atomic is None:
        raise ValueError(f"{protocol} 要求 atomic guardrail")
    observed_atomic = {(row.skill_name, row.seed) for row in atomic}
    expected_atomic = {
        (skill, seed)
        for skill in PICK_AND_PLACE_SKILLS
        for seed in expected["atomic"]
    }
    if observed_atomic != expected_atomic:
        raise ValueError(f"{protocol} atomic seed registry/五技能覆盖不一致")


def _gate_checks(
    protocol: str,
    full_chain: dict[str, Any],
    atomic: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if protocol not in {"stage-a", "stage-a-plus-b"}:
        return None
    if atomic is None:
        raise ValueError(f"{protocol} gate 缺少 atomic 结果")
    issues = full_chain["issue_deltas"]
    atomic_groups = atomic["by_skill"]
    required_atomic = ("grasp", "lift", "place")
    if any(skill not in atomic_groups for skill in required_atomic):
        raise ValueError("E012 gate 缺少 Grasp/Lift/Place atomic guardrail")
    paired = full_chain["unconditional_paired"]
    if protocol == "stage-a":
        checks = {
            "no_new_system_failure": issues["new_system_episodes"] == 0,
            "no_new_safety_failure": issues["new_safety_episodes"] == 0,
            "no_new_tracking_failure": issues["new_tracking_episodes"] == 0,
            "atomic_grasp_not_regressed": atomic_groups["grasp"][
                "net_dagger_wins"
            ]
            >= 0,
            "atomic_lift_not_regressed": atomic_groups["lift"][
                "net_dagger_wins"
            ]
            >= 0,
            "atomic_place_not_regressed": atomic_groups["place"][
                "net_dagger_wins"
            ]
            >= 0,
            "full_chain_reach_delta_at_least_minus_1": paired["reach"][
                "net_dagger_wins"
            ]
            >= -1,
            "full_chain_grasp_not_regressed": paired["grasp"][
                "net_dagger_wins"
            ]
            >= 0,
            "full_chain_lift_not_regressed": paired["lift"]["net_dagger_wins"]
            >= 0,
            "grasp_or_lift_improves_by_at_least_2": max(
                paired["grasp"]["net_dagger_wins"],
                paired["lift"]["net_dagger_wins"],
            )
            >= 2,
            "mean_completed_skills_improved": full_chain[
                "mean_completed_skill_count_delta"
            ]
            > 0.0,
        }
    else:
        checks = {
            "no_new_system_failure": issues["new_system_episodes"] == 0,
            "no_new_safety_failure": issues["new_safety_episodes"] == 0,
            "no_new_tracking_failure": issues["new_tracking_episodes"] == 0,
            "atomic_grasp_not_regressed": atomic_groups["grasp"][
                "net_dagger_wins"
            ]
            >= 0,
            "atomic_lift_not_regressed": atomic_groups["lift"][
                "net_dagger_wins"
            ]
            >= 0,
            "atomic_place_not_regressed": atomic_groups["place"][
                "net_dagger_wins"
            ]
            >= 0,
            "full_chain_reach_delta_at_least_minus_2": paired["reach"][
                "net_dagger_wins"
            ]
            >= -2,
            "full_chain_grasp_not_regressed": paired["grasp"][
                "net_dagger_wins"
            ]
            >= 0,
            "full_chain_lift_not_regressed": paired["lift"]["net_dagger_wins"]
            >= 0,
            "grasp_or_lift_has_positive_paired_net_wins": max(
                paired["grasp"]["net_dagger_wins"],
                paired["lift"]["net_dagger_wins"],
            )
            > 0,
            "mean_completed_skills_improved": full_chain[
                "mean_completed_skill_count_delta"
            ]
            > 0.0,
        }
    return {"passed": all(checks.values()), "checks": checks}


def analyze_e012_pair(
    replay_full_chain: list[RolloutEpisodeResult],
    dagger_full_chain: list[RolloutEpisodeResult],
    *,
    replay_atomic: list[AtomicSkillEpisodeResult] | None = None,
    dagger_atomic: list[AtomicSkillEpisodeResult] | None = None,
    protocol: str = "descriptive",
) -> dict[str, Any]:
    if (replay_atomic is None) != (dagger_atomic is None):
        raise ValueError("replay/DAgger atomic 结果必须成对提供")
    _validate_protocol_seeds(protocol, replay_full_chain, replay_atomic)
    _validate_protocol_seeds(protocol, dagger_full_chain, dagger_atomic)
    full_chain = analyze_full_chain_pair(replay_full_chain, dagger_full_chain)
    atomic = (
        None
        if replay_atomic is None or dagger_atomic is None
        else analyze_atomic_pair(replay_atomic, dagger_atomic)
    )
    return {
        "format": PAIRED_EVALUATION_FORMAT,
        "protocol": protocol,
        "full_chain": full_chain,
        "atomic_guardrail": atomic,
        "gate": _gate_checks(protocol, full_chain, atomic),
    }


__all__ = [
    "PAIRED_EVALUATION_FORMAT",
    "PAIRED_PROTOCOL_SEEDS",
    "analyze_atomic_pair",
    "analyze_e012_pair",
    "analyze_full_chain_pair",
    "exact_paired_test",
]
