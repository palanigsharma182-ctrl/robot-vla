"""七细技能候选合同；只供实验标注/评估，不改变 canonical 五阶段语义。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import numpy as np

VERSION = "pick-carry-place/seven-skills-v1"
SKILLS = ("approach", "align", "grasp", "lift", "transport", "lower", "release")
OPERATIONS = ("pick", "pick", "pick", "pick", "carry", "place", "place")

# 所有范围为闭区间。位置/速度为模长，z 为有符号偏差；角度为 degree。
# None 表示该维度不作固定数值限制，不代表工作空间/碰撞约束可被绕过。
ENTRY = {
    "approach": {"held": (0, 0)},
    "align": {"held": (0, 0), "pregrasp_distance_m": (0, .060),
              "opening": (.8, 1), "tcp_speed_m_s": (0, .25)},
    "grasp": {"held": (0, 0), "pregrasp_distance_m": (0, .020),
              "orientation_error_deg": (0, 10), "opening": (.8, 1),
              "tcp_speed_m_s": (0, .15)},
    "lift": {"held": (1, 1), "grasp_stable": (1, 1), "tcp_speed_m_s": (0, .15)},
    "transport": {"held": (1, 1), "grasp_stable": (1, 1),
                  "clearance_m": (.060, .180), "tcp_speed_m_s": (0, .25)},
    "lower": {"held": (1, 1), "grasp_stable": (1, 1), "goal_xy_m": (0, .030),
              "clearance_m": (.060, .180), "tcp_speed_m_s": (0, .15)},
    "release": {"held": (1, 1), "grasp_stable": (1, 1), "goal_xy_m": (0, .020),
                "goal_z_m": (-.003, .012), "tcp_speed_m_s": (0, .08)},
}
EXIT = {
    "approach": {"held": (0, 0), "pregrasp_distance_m": (0, .040),
                 "opening": (.9, 1), "tcp_speed_m_s": (0, .20)},
    "align": {"held": (0, 0), "pregrasp_distance_m": (0, .008),
              "orientation_error_deg": (0, 5), "opening": (.9, 1),
              "tcp_speed_m_s": (0, .08)},
    "grasp": {"held": (1, 1), "grasp_stable": (1, 1), "tcp_speed_m_s": (0, .08)},
    "lift": {"held": (1, 1), "grasp_stable": (1, 1), "clearance_m": (.080, .160),
             "support_force_n": (0, .1), "tcp_speed_m_s": (0, .20)},
    "transport": {"held": (1, 1), "grasp_stable": (1, 1), "goal_xy_m": (0, .015),
                  "clearance_m": (.080, .160), "tcp_speed_m_s": (0, .12)},
    "lower": {"held": (1, 1), "grasp_stable": (1, 1), "goal_xy_m": (0, .010),
              "goal_z_m": (-.002, .008), "tcp_speed_m_s": (0, .04)},
    # 最终物体距离、速度及连续四帧沿用原完整任务语义；另要求确实发送释放命令。
    "release": {"held": (0, 0), "release_commanded": (1, 1), "goal_distance_m": (0, .025),
                "object_speed_m_s": (0, .01), "object_angular_speed_rad_s": (0, .5)},
}
PERTURBATIONS = {
    "translation_norm_m": [.005, .010, .020],
    "yaw_abs_deg": [5, 10, 20], "tilt_norm_deg": [3, 5, 10],
    "linear_velocity_increment_norm_m_s": [.02, .05, .10],
    "angular_velocity_increment_norm_deg_s": [5, 15, 30],
    "held_relative_translation_m": [.001, .002, .004],
    "held_relative_rotation_deg": [2, 5, 10],
}
GRASP_WINDOW = 3  # 20 Hz 下三个观测点覆盖 0.10 秒，不是可靠抓取认证。
GRASP_TRANSLATION_M = .002
GRASP_ROTATION_DEG = 5.


def rotation_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    """两个旋转间的主角；输入来自已验证的 SE(3) 或仿真器。"""
    return math.degrees(math.acos(float(np.clip((np.trace(a.T @ b)-1)/2, -1, 1))))


def cube_orientation_error_deg(actual: np.ndarray, target: np.ndarray) -> float:
    """仅方块抓取采用绕目标接近轴的四重对称；滑移检测不折叠对称性。"""
    candidates = []
    for k in range(4):
        t = k * math.pi/2
        rz = np.array([[math.cos(t), -math.sin(t), 0],
                       [math.sin(t), math.cos(t), 0], [0, 0, 1]])
        candidates.append(rotation_distance_deg(actual, target @ rz))
    return min(candidates)


def violations(rules: dict, metrics: dict) -> list[str]:
    failed = []
    for key, (lo, hi) in rules.items():
        value = float(metrics.get(key, math.nan))
        if not math.isfinite(value) or not lo <= value <= hi:
            failed.append(key)
    return failed


def invariant_violations(skill: str, metrics: dict) -> list[str]:
    """实时状态优先于历史完成事件；丢失抓持进入恢复诊断。"""
    if skill in ("lift", "transport", "lower"):
        return ["grasp_lost"] if violations({"held": (1, 1)}, metrics) else []
    if skill in ("approach", "align"):
        return violations({"held": (0, 0), "opening": (.8, 1)}, metrics)
    return []


@dataclass
class BoundaryTracker:
    active: int = 0
    events: list[dict] = field(default_factory=list)
    held_history: deque = field(default_factory=lambda: deque(maxlen=GRASP_WINDOW))
    stable_release_frames: int = 0
    last_time: float | None = None

    def observe(self, metrics: dict, tcp_from_object: np.ndarray, timestamp_s: float) -> dict:
        """每个真实控制 tick 调用一次；先补充抓持稳定证据，再判定当前技能出口。"""
        if not math.isfinite(timestamp_s):
            raise ValueError("时间必须有限")
        if self.last_time is not None and not math.isclose(timestamp_s-self.last_time, .05, abs_tol=1e-6):
            raise ValueError("候选合同要求连续 20 Hz 观测，不允许重复帧凑稳定窗口")
        self.last_time = timestamp_s
        pose = np.asarray(tcp_from_object, dtype=float)
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5)
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
                or not math.isclose(np.linalg.det(pose[:3, :3]), 1, abs_tol=1e-5)):
            raise ValueError("相对位姿必须为有效 SE(3)")
        if metrics.get("held") == 1:
            self.held_history.append(pose.copy())
        else:
            self.held_history.clear()
        pairs = [(a, b) for i, a in enumerate(self.held_history) for b in list(self.held_history)[i+1:]]
        translation = max((float(np.linalg.norm(a[:3, 3]-b[:3, 3])) for a, b in pairs), default=0.)
        angle = max((rotation_distance_deg(a[:3, :3], b[:3, :3]) for a, b in pairs), default=0.)
        m = dict(metrics, grasp_stable=int(len(self.held_history) == GRASP_WINDOW
                 and translation <= GRASP_TRANSLATION_M and angle <= GRASP_ROTATION_DEG),
                 relative_drift_m=translation, relative_drift_deg=angle)
        if self.active == len(SKILLS):
            return m
        skill = SKILLS[self.active]
        faults = invariant_violations(skill, m)
        if faults:
            raise RuntimeError(f"{skill}: {','.join(faults)}")
        good = not violations(EXIT[skill], m)
        if skill == "release":
            self.stable_release_frames = self.stable_release_frames+1 if good else 0
            good = self.stable_release_frames >= 4
        if good:
            next_skill = SKILLS[self.active+1] if self.active+1 < len(SKILLS) else None
            handoff_errors = violations(ENTRY[next_skill], m) if next_skill else []
            # 不通过额外动作把不合法的前一出口修饰成下一入口。
            if handoff_errors:
                raise RuntimeError(f"{skill} -> {next_skill} 交接不兼容: {handoff_errors}")
            self.events.append(dict(skill=skill, time_s=timestamp_s, metrics=m))
            self.active += 1
        return m


def contract_document() -> dict:
    return dict(version=VERSION, status="candidate-thresholds-not-certified-robustness",
                skills=list(SKILLS), operations=list(OPERATIONS), entry=ENTRY, exit=EXIT,
                perturbations=PERTURBATIONS, control_hz=20, grasp_window=GRASP_WINDOW,
                grasp_translation_m=GRASP_TRANSLATION_M, grasp_rotation_deg=GRASP_ROTATION_DEG,
                release_stable_frames=4, sidecar_field="fine_skill_id",
                privileged_evaluation_only=True)
