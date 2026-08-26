"""qwen-vla-v0.1 跨数据、模型和控制模块共享的稳定契约。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

TRAJECTORY_SCHEMA_VERSION = "robot-vla-trajectory/v2"
MODEL_ARCH = "qwen_vla_late_fusion_v1"
PROMPT_VERSION = "qwen-vla-prompt/v1"
QWEN_MODEL_ID = "Qwen/Qwen3.5-2B"
QWEN_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
OUTCOME_PREDICATE_VERSION = "pick-and-place-predicates/v1"
PROPRIO_STATS_VERSION = "franka-proprio-zscore/v1"

UNKNOWN_SKILL_ID = -1
PICK_AND_PLACE_SKILLS = ("reach", "grasp", "lift", "transport", "place")

FRANKA_ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
FRANKA_GRIPPER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
FRANKA_JOINT_POSITION_LIMITS_RAD = (
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
)
FRANKA_JOINT_VELOCITY_LIMITS_RAD_S = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)


def _finite_number(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数值")


@dataclass(frozen=True)
class RobotSpec:
    """ManiSkill Franka 的观测、动作和控制时间契约。"""

    embodiment: str = "maniskill-franka-panda-v1"
    arm_joint_names: tuple[str, ...] = FRANKA_ARM_JOINT_NAMES
    gripper_joint_names: tuple[str, ...] = FRANKA_GRIPPER_JOINT_NAMES
    joint_position_limits_rad: tuple[tuple[float, float], ...] = (
        FRANKA_JOINT_POSITION_LIMITS_RAD
    )
    joint_velocity_limits_rad_s: tuple[float, ...] = FRANKA_JOINT_VELOCITY_LIMITS_RAD_S
    joint_delta_limit_rad: tuple[float, ...] = (0.05,) * 7
    maniskill_arm_delta_range_rad: float = 0.1
    gripper_joint_position_range_m: tuple[float, float] = (0.0, 0.04)
    action_horizon: int = 16
    control_hz: float = 20.0
    execute_steps: int = 4

    def __post_init__(self) -> None:
        if not self.embodiment.strip():
            raise ValueError("embodiment 不能为空")
        if self.arm_joint_names != FRANKA_ARM_JOINT_NAMES:
            raise ValueError(f"Franka arm joint 顺序必须为 {FRANKA_ARM_JOINT_NAMES}")
        if self.gripper_joint_names != FRANKA_GRIPPER_JOINT_NAMES:
            raise ValueError(f"Franka gripper joint 顺序必须为 {FRANKA_GRIPPER_JOINT_NAMES}")
        if len(self.joint_position_limits_rad) != self.arm_dof:
            raise ValueError("joint_position_limits_rad 长度必须等于 arm_dof")
        if len(self.joint_velocity_limits_rad_s) != self.arm_dof:
            raise ValueError("joint_velocity_limits_rad_s 长度必须等于 arm_dof")
        if len(self.joint_delta_limit_rad) != self.arm_dof:
            raise ValueError("joint_delta_limit_rad 长度必须等于 arm_dof")

        for index, (lower, upper) in enumerate(self.joint_position_limits_rad):
            _finite_number(lower, f"joint[{index}] lower")
            _finite_number(upper, f"joint[{index}] upper")
            if lower >= upper:
                raise ValueError(f"joint[{index}] 位置下限必须小于上限")
        for name, values in (
            ("joint_velocity_limits_rad_s", self.joint_velocity_limits_rad_s),
            ("joint_delta_limit_rad", self.joint_delta_limit_rad),
        ):
            if any(value <= 0 for value in values):
                raise ValueError(f"{name} 必须全部为正数")
            for value in values:
                _finite_number(value, name)

        if self.maniskill_arm_delta_range_rad <= 0:
            raise ValueError("maniskill_arm_delta_range_rad 必须为正数")
        gripper_lower, gripper_upper = self.gripper_joint_position_range_m
        if gripper_lower < 0 or gripper_lower >= gripper_upper:
            raise ValueError("gripper_joint_position_range_m 无效")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon 必须为正整数")
        if self.control_hz <= 0:
            raise ValueError("control_hz 必须为正数")
        if not 1 <= self.execute_steps <= self.action_horizon:
            raise ValueError("execute_steps 必须位于 [1, action_horizon]")

    @property
    def arm_dof(self) -> int:
        return len(self.arm_joint_names)

    @property
    def proprio_dim(self) -> int:
        return self.arm_dof * 2 + 1

    @property
    def action_dim(self) -> int:
        return self.arm_dof + 1

    @property
    def active_joint_names(self) -> tuple[str, ...]:
        return self.arm_joint_names + self.gripper_joint_names

    @property
    def chunk_duration_s(self) -> float:
        return self.action_horizon / self.control_hz

    @property
    def replan_hz(self) -> float:
        return self.control_hz / self.execute_steps

    @property
    def effective_joint_delta_limits_rad(self) -> tuple[float, ...]:
        return tuple(
            min(configured, velocity / self.control_hz)
            for configured, velocity in zip(
                self.joint_delta_limit_rad,
                self.joint_velocity_limits_rad_s,
                strict=True,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RobotSpec:
        data = dict(value)
        for key in (
            "arm_joint_names",
            "gripper_joint_names",
            "joint_velocity_limits_rad_s",
            "joint_delta_limit_rad",
            "gripper_joint_position_range_m",
        ):
            if key in data:
                data[key] = tuple(data[key])
        if "joint_position_limits_rad" in data:
            data["joint_position_limits_rad"] = tuple(
                tuple(pair) for pair in data["joint_position_limits_rad"]
            )
        return cls(**data)


@dataclass(frozen=True)
class TaskSpec:
    """语言任务、技能 ID 映射和 Outcome Predicate 版本。"""

    task_id: str
    task_group_id: str
    instruction: str
    skill_names: tuple[str, ...] = PICK_AND_PLACE_SKILLS
    outcome_predicate_version: str = OUTCOME_PREDICATE_VERSION

    def __post_init__(self) -> None:
        for name, value in (
            ("task_id", self.task_id),
            ("task_group_id", self.task_group_id),
            ("instruction", self.instruction),
            ("outcome_predicate_version", self.outcome_predicate_version),
        ):
            if not value.strip():
                raise ValueError(f"{name} 不能为空")
        if not self.skill_names or any(not skill.strip() for skill in self.skill_names):
            raise ValueError("skill_names 必须包含非空技能名称")
        if len(set(self.skill_names)) != len(self.skill_names):
            raise ValueError("skill_names 不能重复")

    def skill_name(self, skill_id: int) -> str:
        if skill_id == UNKNOWN_SKILL_ID:
            return "unknown"
        if not 0 <= skill_id < len(self.skill_names):
            raise ValueError(f"未知 skill_id: {skill_id}")
        return self.skill_names[skill_id]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TaskSpec:
        data = dict(value)
        if "skill_names" in data:
            data["skill_names"] = tuple(data["skill_names"])
        return cls(**data)
