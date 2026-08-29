"""E012a paired boundary risk components、mid-rank percentile 与确定性选样。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

RISK_CONTRACT_VERSION = "robot-vla-local-dagger-risk/v1"

RISK_COMPONENT_UNITS = {
    "reach_grasp": {
        "tcp_object_xy_error_m": "m",
        "relative_z_deviation_m": "m",
        "tcp_linear_speed_m_s": "m/s",
        "joint_velocity_rms_rad_s": "rad/s",
        "gripper_opening_deviation": "opening_ratio",
        "arm_mean_pairwise_disagreement": "normalized_action",
        "gripper_mean_pairwise_disagreement": "normalized_action",
    },
    "grasp_lift": {
        "object_tcp_relative_position_deviation_m": "m",
        "object_linear_speed_m_s": "m/s",
        "object_angular_speed_rad_s": "rad/s",
        "joint_velocity_rms_rad_s": "rad/s",
        "gripper_opening_deviation": "opening_ratio",
        "contact_grasp_instability": "binary",
        "arm_mean_pairwise_disagreement": "normalized_action",
        "gripper_mean_pairwise_disagreement": "normalized_action",
    },
}


@dataclass(frozen=True)
class RiskSelectionResult:
    boundary_type: str
    scored_candidates: tuple[dict[str, Any], ...]
    high_risk_seeds: tuple[int, ...]
    low_risk_seeds: tuple[int, ...]
    version: str = RISK_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "boundary_type": self.boundary_type,
            "component_units": dict(RISK_COMPONENT_UNITS[self.boundary_type]),
            "scored_candidates": list(self.scored_candidates),
            "high_risk_seeds": list(self.high_risk_seeds),
            "low_risk_seeds": list(self.low_risk_seeds),
            "selected_seeds": list(self.high_risk_seeds + self.low_risk_seeds),
        }


def _finite_nonnegative(value: Any, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"risk component {name} 必须是有限非负数")
    return resolved


def compute_paired_risk_components(
    boundary_type: str,
    policy_boundary: Mapping[str, Any],
    expert_boundary: Mapping[str, Any],
) -> dict[str, float]:
    """按预注册公式计算原始 risk components，不做 percentile 或权重。"""

    if boundary_type not in RISK_COMPONENT_UNITS:
        raise ValueError(f"未知 boundary_type: {boundary_type}")
    if policy_boundary.get("boundary_type") != boundary_type:
        raise ValueError("Policy boundary identity 不一致")
    if expert_boundary.get("boundary_type") != boundary_type:
        raise ValueError("Expert boundary identity 不一致")

    gripper_deviation = abs(
        float(policy_boundary["gripper_opening"])
        - float(expert_boundary["gripper_opening"])
    )
    if boundary_type == "reach_grasp":
        components = {
            "tcp_object_xy_error_m": policy_boundary["tcp_object_xy_error_m"],
            "relative_z_deviation_m": abs(
                float(policy_boundary["tcp_object_relative_z_m"])
                - float(expert_boundary["tcp_object_relative_z_m"])
            ),
            "tcp_linear_speed_m_s": policy_boundary["tcp_linear_speed_m_s"],
            "joint_velocity_rms_rad_s": policy_boundary[
                "joint_velocity_rms_rad_s"
            ],
            "gripper_opening_deviation": gripper_deviation,
            "arm_mean_pairwise_disagreement": policy_boundary[
                "arm_mean_pairwise_disagreement"
            ],
            "gripper_mean_pairwise_disagreement": policy_boundary[
                "gripper_mean_pairwise_disagreement"
            ],
        }
    else:
        policy_relative = np.asarray(
            policy_boundary["tcp_object_relative_xyz_m"],
            dtype=np.float64,
        )
        expert_relative = np.asarray(
            expert_boundary["tcp_object_relative_xyz_m"],
            dtype=np.float64,
        )
        if policy_relative.shape != (3,) or expert_relative.shape != (3,):
            raise ValueError("GL object-TCP relative position 必须是 3D")
        stable_grasp = bool(policy_boundary["is_grasped"]) and float(
            policy_boundary["robot_object_contact_force_n"]
        ) > 0.0
        components = {
            "object_tcp_relative_position_deviation_m": float(
                np.linalg.norm(policy_relative - expert_relative)
            ),
            "object_linear_speed_m_s": policy_boundary["object_linear_speed_m_s"],
            "object_angular_speed_rad_s": policy_boundary[
                "object_angular_speed_rad_s"
            ],
            "joint_velocity_rms_rad_s": policy_boundary[
                "joint_velocity_rms_rad_s"
            ],
            "gripper_opening_deviation": gripper_deviation,
            "contact_grasp_instability": 0.0 if stable_grasp else 1.0,
            "arm_mean_pairwise_disagreement": policy_boundary[
                "arm_mean_pairwise_disagreement"
            ],
            "gripper_mean_pairwise_disagreement": policy_boundary[
                "gripper_mean_pairwise_disagreement"
            ],
        }
    expected = set(RISK_COMPONENT_UNITS[boundary_type])
    if set(components) != expected:
        raise RuntimeError("risk component schema 与 manifest 不一致")
    return {
        name: _finite_nonnegative(value, name)
        for name, value in components.items()
    }


def midrank_empirical_percentiles(values: Sequence[float]) -> tuple[float, ...]:
    """最小/最大非 tie 值映射到 0/1；tie 使用零基位置的平均 rank。"""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("mid-rank percentile 至少需要两个有限标量")
    output = np.empty(array.size, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    sorted_values = array[order]
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and sorted_values[end] == sorted_values[start]:
            end += 1
        midrank = ((start + end - 1) * 0.5) / (array.size - 1)
        output[order[start:end]] = midrank
        start = end
    return tuple(float(value) for value in output)


def score_and_select_risk_candidates(
    boundary_type: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    high_count: int = 14,
    low_count: int = 6,
) -> RiskSelectionResult:
    """只接受 paired/full-recovery/audit 已通过的候选，并返回 20 个互异 seed。"""

    if boundary_type not in RISK_COMPONENT_UNITS:
        raise ValueError(f"未知 boundary_type: {boundary_type}")
    if high_count <= 0 or low_count <= 0:
        raise ValueError("high_count/low_count 必须为正数")
    if len(candidates) < high_count + low_count:
        raise ValueError("完整恢复且 paired/audit 通过的候选不足选样配额")
    seeds = [int(candidate["environment_seed"]) for candidate in candidates]
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("candidate environment_seed 必须非负且唯一")
    component_names = tuple(RISK_COMPONENT_UNITS[boundary_type])
    raw_components: list[dict[str, float]] = []
    for candidate in candidates:
        values = {
            name: _finite_nonnegative(candidate["risk_components"][name], name)
            for name in component_names
        }
        if set(candidate["risk_components"]) != set(component_names):
            raise ValueError("candidate risk component schema 不完整")
        raw_components.append(values)

    percentiles_by_component = {
        name: midrank_empirical_percentiles(
            [components[name] for components in raw_components]
        )
        for name in component_names
    }
    rows: list[dict[str, Any]] = []
    for index, (candidate, components) in enumerate(
        zip(candidates, raw_components, strict=True)
    ):
        percentiles = {
            name: percentiles_by_component[name][index]
            for name in component_names
        }
        rows.append(
            {
                **dict(candidate),
                "risk_components": components,
                "risk_component_percentiles": percentiles,
                "risk_score": float(np.mean(tuple(percentiles.values()))),
                "selection_stratum": None,
            }
        )

    high_order = sorted(rows, key=lambda row: (-row["risk_score"], row["environment_seed"]))
    high = high_order[:high_count]
    high_seeds = {row["environment_seed"] for row in high}
    low_order = sorted(
        (row for row in rows if row["environment_seed"] not in high_seeds),
        key=lambda row: (row["risk_score"], row["environment_seed"]),
    )
    low = low_order[:low_count]
    selected = {
        row["environment_seed"]: "high" for row in high
    } | {row["environment_seed"]: "low" for row in low}
    for row in rows:
        row["selection_stratum"] = selected.get(row["environment_seed"])
    rows.sort(key=lambda row: row["environment_seed"])
    return RiskSelectionResult(
        boundary_type=boundary_type,
        scored_candidates=tuple(rows),
        high_risk_seeds=tuple(row["environment_seed"] for row in high),
        low_risk_seeds=tuple(row["environment_seed"] for row in low),
    )


__all__ = [
    "RISK_COMPONENT_UNITS",
    "RISK_CONTRACT_VERSION",
    "RiskSelectionResult",
    "compute_paired_risk_components",
    "midrank_empirical_percentiles",
    "score_and_select_risk_candidates",
]
