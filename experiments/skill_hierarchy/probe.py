"""有界云端教师探针：新分段和独立准备共用首次到达出口的停止逻辑。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.skill_hierarchy.contract import (
    VERSION, SKILLS, ENTRY, EXIT, BoundaryTracker, contract_document,
    cube_orientation_error_deg, violations,
)
from robot_vla.sim.collector import TrustedPickPlaceCollector, _numpy


class BoundaryReached(Exception):
    """已执行的动作完成了当前技能，中断剩余教师路径。"""


class HierarchyTeacher(TrustedPickPlaceCollector):
    def initialize(self, seed: int):
        self.session = self._start_session(seed, record_action_provenance=True)
        self.poses = self._phase_poses()  # 固定 episode 的目标，不能随实际 TCP 漂移重算。
        self.pregrasp_world = self.poses[1].to_transformation_matrix()
        self.boundaries = BoundaryTracker()
        self.rows = []
        self.last_tcp_world = None
        self.metrics = self.measure(open_command=1.)
        self.metrics = self.boundaries.observe(self.metrics, self.relative_pose, 0.)

    def measure(self, *, open_command: float) -> dict:
        state = self._read_predicate_state()
        tcp = _numpy(self.base_env.agent.tcp_pose.to_transformation_matrix())[0].astype(float)
        obj = _numpy(self.base_env.cube.pose.to_transformation_matrix())[0].astype(float)
        self.relative_pose = np.linalg.inv(tcp) @ obj
        speed = 0. if self.last_tcp_world is None else float(np.linalg.norm(tcp[:3, 3]-self.last_tcp_world)/.05)
        self.last_tcp_world = tcp[:3, 3].copy()
        _, support = self._read_contact_forces()
        q = _numpy(self.base_env.agent.robot.get_qpos())[0]
        delta = np.asarray(state.object_position)-np.asarray(state.goal_position)
        return dict(held=int(state.is_grasped), opening=float(np.clip(np.mean(q[-2:])/.04, 0, 1)),
                    pregrasp_distance_m=float(np.linalg.norm(tcp[:3, 3]-self.pregrasp_world[:3, 3])),
                    orientation_error_deg=cube_orientation_error_deg(tcp[:3, :3], self.pregrasp_world[:3, :3]),
                    tcp_speed_m_s=speed, clearance_m=state.object_position[2]-state.support_center_z_m,
                    goal_xy_m=float(np.linalg.norm(delta[:2])), goal_z_m=float(delta[2]),
                    goal_distance_m=float(np.linalg.norm(delta)), support_force_n=support,
                    object_speed_m_s=float(np.linalg.norm(state.object_linear_velocity)),
                    object_angular_speed_rad_s=float(np.linalg.norm(state.object_angular_velocity)),
                    release_commanded=int(open_command >= .9 and self.boundaries.active == 6))

    def _step_with_target(self, session, target_q, label, gripper_opening):
        skill_id = self.boundaries.active
        row = dict(action_index=len(self.rows), fine_skill_id=skill_id,
                   before=self.metrics.copy(), commanded_joint_target_rad=np.asarray(target_q).tolist(),
                   gripper_command=gripper_opening)
        super()._step_with_target(session, target_q, label, gripper_opening)
        # 异常停止也必须保留刚发出的命令锚点，包括 settle 分支。
        session.previous_command_q = np.asarray(target_q).copy()
        self.metrics = self.measure(open_command=gripper_opening)
        row['after'] = self.metrics.copy()
        self.rows.append(row)
        self.metrics = self.boundaries.observe(self.metrics, self.relative_pose, len(self.rows)/20.)
        row['after'] = self.metrics.copy()
        if self.boundaries.active != skill_id:
            raise BoundaryReached()

    def run_skill(self):
        skill_id = self.boundaries.active
        skill = SKILLS[skill_id]
        invalid = violations(ENTRY[skill], self.metrics)
        if invalid:
            raise RuntimeError(f'{skill} 入口不合法: {invalid}')
        grasp, pregrasp, lift, transport, lower = self.poses
        try:
            if skill in ('approach', 'align'):
                self._move_to_pose(self.session, pregrasp, gripper_opening=1.)
            elif skill == 'grasp':
                self._move_to_pose(self.session, grasp, gripper_opening=1.)
                self._hold(self.session, gripper_opening=0., steps=16)
            elif skill in ('lift', 'transport', 'lower'):
                target = {'lift': lift, 'transport': transport, 'lower': lower}[skill]
                self._move_to_pose(self.session, target, gripper_opening=0.)
            else:
                self._hold(self.session, gripper_opening=1., steps=24)
            # 目标已到达但速度/短窗口未稳定，最多额外等待 1 秒。
            self._hold(self.session, gripper_opening=1. if skill in ('approach', 'align', 'release') else 0., steps=20)
        except BoundaryReached:
            return
        raise RuntimeError(f'{skill} 教师路径及等待后未完成: {violations(EXIT[skill], self.metrics)}')

    def run_until(self, stop_before: int = 7):
        """分段采集与独立技能准备使用同一入口；stop_before 前不执行目标技能动作。"""
        if not 0 <= stop_before <= 7:
            raise ValueError('stop_before 必须为 0..7')
        while self.boundaries.active < stop_before:
            self.run_skill()
        if self.boundaries.active != stop_before:
            raise RuntimeError('初始状态跳过了目标技能')


def save(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    contract = contract_document()
    save(args.output/'contract.json', contract)
    started = time.monotonic()
    # 新 development 身份，不消费旧 confirmation/final test；不追加 seed 寻找成功。
    units = [(seed, 7) for seed in range(1800000, 1800004)] + [(1800000, i) for i in (2, 5, 6)]
    result = dict(version=VERSION, status='running', data_use='development',
                  scope='nominal-teacher-boundaries-only; no-perturbation-certification',
                  units=[], max_total_steps=4200,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(Path('experiments/skill_hierarchy').glob('*.py'))})
    for seed, stop in units:
        # SAPIEN reset 后上一次接触缓存仍可被查询到；每单元重建 scene，
        # 保留零准备动作的真实初始状态，不靠额外 env.step 掩盖缓存问题。
        with HierarchyTeacher(None, max_episode_steps=600) as teacher:
            row = dict(seed=seed, stop_before=stop, status='running')
            result['units'].append(row)
            try:
                teacher.initialize(seed)
                teacher.run_until(stop)
                row['status'] = 'passed'
            except Exception as exc:
                row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            finally:
                row.update(steps=len(getattr(teacher, 'rows', [])),
                           events=getattr(getattr(teacher, 'boundaries', None), 'events', []),
                           final_metrics=getattr(teacher, 'metrics', {}))
                save(args.output/f'{seed}-before-{stop}.json', getattr(teacher, 'rows', []))
                save(args.output/'result.json', result)
    # 检查独立准备与同 seed 完整轨迹前缀完全一致，不用后处理修补边界。
    full = args.output/'1800000-before-7.json'
    full_rows = json.loads(full.read_text())
    for row in result['units'][4:]:
        prep = json.loads((args.output/f"1800000-before-{row['stop_before']}.json").read_text())
        row['prefix_parity'] = prep == full_rows[:len(prep)]
        row['target_skill_actions'] = sum(r['fine_skill_id'] >= row['stop_before'] for r in prep)
        if not row['prefix_parity'] or row['target_skill_actions']:
            row['status'] = 'failed'
    result.update(status='passed' if all(r['status'] == 'passed' for r in result['units']) else 'failed',
                  elapsed_s=time.monotonic()-started)
    save(args.output/'result.json', result)
    print(json.dumps(dict(status=result['status'], elapsed_s=result['elapsed_s'],
                         units=[{k:v for k,v in r.items() if k not in ('events', 'final_metrics')}
                                for r in result['units']]), ensure_ascii=False))
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
