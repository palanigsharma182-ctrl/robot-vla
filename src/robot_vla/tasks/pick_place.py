"""D012 的 reach→grasp→lift→transport→place 可判定状态机。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robot_vla.contracts import (
    OUTCOME_PREDICATE_VERSION,
    PICK_AND_PLACE_SKILLS,
    TaskSpec,
)

PICK_PLACE_INSTRUCTIONS = (
    "Pick up the red cube and place it in the green target region.",
    "Move the red cube into the green target region.",
    "Grasp the red cube and set it down in the green target area.",
)


@dataclass(frozen=True)
class AtomicSkillDefinition:
    skill_id: int
    name: str
    prerequisite: str
    outcome: str


ATOMIC_PICK_PLACE_SKILLS = (
    AtomicSkillDefinition(0, "reach", "cube and TCP are valid", "TCP enters cube neighborhood"),
    AtomicSkillDefinition(1, "grasp", "reach completed", "stable two-finger grasp"),
    AtomicSkillDefinition(2, "lift", "grasp completed", "cube clears support height"),
    AtomicSkillDefinition(3, "transport", "lift completed", "held cube enters target XY area"),
    AtomicSkillDefinition(4, "place", "transport completed", "released cube settles in target"),
)


def _finite_xyz(value: tuple[float, float, float], name: str) -> None:
    if len(value) != 3 or not np.isfinite(np.asarray(value, dtype=np.float64)).all():
        raise ValueError(f"{name} 必须是 3 维有限坐标")


@dataclass(frozen=True)
class PickPlacePredicateConfig:
    version: str = OUTCOME_PREDICATE_VERSION
    reach_distance_m: float = 0.04
    lift_clearance_m: float = 0.05
    transport_xy_distance_m: float = 0.04
    place_distance_m: float = 0.025
    static_linear_speed_m_s: float = 0.01
    static_angular_speed_rad_s: float = 0.5
    stable_grasp_steps: int = 2
    stable_place_steps: int = 4

    def __post_init__(self) -> None:
        if self.version != OUTCOME_PREDICATE_VERSION:
            raise ValueError(f"Predicate version 必须为 {OUTCOME_PREDICATE_VERSION}")
        for name in (
            "reach_distance_m",
            "lift_clearance_m",
            "transport_xy_distance_m",
            "place_distance_m",
            "static_linear_speed_m_s",
            "static_angular_speed_rad_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是有限正数")
        if self.stable_grasp_steps <= 0 or self.stable_place_steps <= 0:
            raise ValueError("稳定帧数必须为正整数")


@dataclass(frozen=True)
class PickPlaceState:
    tcp_position: tuple[float, float, float]
    object_position: tuple[float, float, float]
    goal_position: tuple[float, float, float]
    object_linear_velocity: tuple[float, float, float]
    object_angular_velocity: tuple[float, float, float]
    support_center_z_m: float
    is_grasped: bool

    def __post_init__(self) -> None:
        for name in (
            "tcp_position",
            "object_position",
            "goal_position",
            "object_linear_velocity",
            "object_angular_velocity",
        ):
            _finite_xyz(getattr(self, name), name)
        if not math.isfinite(self.support_center_z_m):
            raise ValueError("support_center_z_m 必须是有限数值")
        if not isinstance(self.is_grasped, bool):
            raise TypeError("is_grasped 必须为 bool")


@dataclass(frozen=True)
class OutcomeSnapshot:
    tcp_to_object_distance_m: float
    object_height_above_support_m: float
    object_to_goal_xy_distance_m: float
    object_to_goal_distance_m: float
    object_linear_speed_m_s: float
    object_angular_speed_rad_s: float
    reached: bool
    grasped: bool
    lifted: bool
    transported: bool
    placed_now: bool


def evaluate_pick_place_outcomes(
    state: PickPlaceState,
    config: PickPlacePredicateConfig,
) -> OutcomeSnapshot:
    tcp = np.asarray(state.tcp_position, dtype=np.float64)
    obj = np.asarray(state.object_position, dtype=np.float64)
    goal = np.asarray(state.goal_position, dtype=np.float64)
    linear_velocity = np.asarray(state.object_linear_velocity, dtype=np.float64)
    angular_velocity = np.asarray(state.object_angular_velocity, dtype=np.float64)
    tcp_distance = float(np.linalg.norm(tcp - obj))
    height = float(obj[2] - state.support_center_z_m)
    goal_delta = obj - goal
    goal_xy_distance = float(np.linalg.norm(goal_delta[:2]))
    goal_distance = float(np.linalg.norm(goal_delta))
    linear_speed = float(np.linalg.norm(linear_velocity))
    angular_speed = float(np.linalg.norm(angular_velocity))
    lifted = state.is_grasped and height >= config.lift_clearance_m
    transported = (
        lifted
        and goal_xy_distance <= config.transport_xy_distance_m
    )
    placed_now = (
        not state.is_grasped
        and goal_distance <= config.place_distance_m
        and linear_speed <= config.static_linear_speed_m_s
        and angular_speed <= config.static_angular_speed_rad_s
    )
    return OutcomeSnapshot(
        tcp_to_object_distance_m=tcp_distance,
        object_height_above_support_m=height,
        object_to_goal_xy_distance_m=goal_xy_distance,
        object_to_goal_distance_m=goal_distance,
        object_linear_speed_m_s=linear_speed,
        object_angular_speed_rad_s=angular_speed,
        reached=tcp_distance <= config.reach_distance_m,
        grasped=state.is_grasped,
        lifted=lifted,
        transported=transported,
        placed_now=placed_now,
    )


@dataclass(frozen=True)
class PickPlaceTaskProgress:
    active_skill_id: int
    active_skill_name: str
    completed_skill_count: int
    task_completed: bool
    stable_grasp_steps: int
    stable_place_steps: int
    outcome: OutcomeSnapshot


class PickPlaceTaskTracker:
    """单调推进技能阶段；失败或条件消失不会伪造后续完成。"""

    def __init__(self, config: PickPlacePredicateConfig | None = None) -> None:
        self.config = config or PickPlacePredicateConfig()
        self._active_skill_id = 0
        self._stable_grasp_steps = 0
        self._stable_place_steps = 0
        self._task_completed = False

    @property
    def active_skill_id(self) -> int:
        return self._active_skill_id

    @property
    def task_completed(self) -> bool:
        return self._task_completed

    def update(self, state: PickPlaceState) -> PickPlaceTaskProgress:
        outcome = evaluate_pick_place_outcomes(state, self.config)
        if not self._task_completed:
            if self._active_skill_id == 0 and outcome.reached:
                self._active_skill_id = 1
            elif self._active_skill_id == 1:
                self._stable_grasp_steps = (
                    self._stable_grasp_steps + 1 if outcome.grasped else 0
                )
                if self._stable_grasp_steps >= self.config.stable_grasp_steps:
                    self._active_skill_id = 2
            elif self._active_skill_id == 2 and outcome.lifted:
                self._active_skill_id = 3
            elif self._active_skill_id == 3 and outcome.transported:
                self._active_skill_id = 4
            elif self._active_skill_id == 4:
                self._stable_place_steps = (
                    self._stable_place_steps + 1 if outcome.placed_now else 0
                )
                if self._stable_place_steps >= self.config.stable_place_steps:
                    self._task_completed = True

        return PickPlaceTaskProgress(
            active_skill_id=self._active_skill_id,
            active_skill_name=PICK_AND_PLACE_SKILLS[self._active_skill_id],
            completed_skill_count=(
                len(PICK_AND_PLACE_SKILLS)
                if self._task_completed
                else self._active_skill_id
            ),
            task_completed=self._task_completed,
            stable_grasp_steps=self._stable_grasp_steps,
            stable_place_steps=self._stable_place_steps,
            outcome=outcome,
        )


def build_pick_place_task(instruction_index: int = 0) -> TaskSpec:
    if not 0 <= instruction_index < len(PICK_PLACE_INSTRUCTIONS):
        raise ValueError(f"instruction_index 应位于 [0,{len(PICK_PLACE_INSTRUCTIONS) - 1}]")
    return TaskSpec(
        task_id="pick-cube-to-region",
        task_group_id="pick-and-place",
        instruction=PICK_PLACE_INSTRUCTIONS[instruction_index],
        skill_names=PICK_AND_PLACE_SKILLS,
        outcome_predicate_version=OUTCOME_PREDICATE_VERSION,
    )
