"""教师标准交接及实际运动扰动后的 Align 闭环；共享学生不接收技能 ID。"""
import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from experiments.skill_hierarchy.contract import BoundaryTracker, ENTRY, EXIT, violations
from experiments.skill_hierarchy.full_task import load_student, verify_upstream
from experiments.skill_hierarchy.probe import HierarchyTeacher
from experiments.tcp_atomic_skills.protocol import save, sha, verify_source
from experiments.tcp_atomic_skills.runtime import AtomicController, AtomicExecutor, execute_plan, load_parent, predict
from experiments.tcp_memory_control.kinematics import TCPKinematics
from experiments.tcp_memory_control.protocol import sampling_seed
from robot_vla.sim.collector import AtomicPreparation, _numpy
from robot_vla.tasks.pick_place import build_pick_place_task

SEEDS = tuple(range(1811000, 1811004))
CASES = ('standard', 'settled', 'position_in10', 'position_out10',
         'yaw_neg10', 'yaw_pos10', 'velocity_in05', 'velocity_out05')
LIMIT = 120
VELOCITY_LEAD_SECONDS = .075  # 首轮纯准备实测约 3.7 mm / 0.05 m/s 的伺服滞后。


def rotation_error(a, b):
    return float(np.linalg.norm(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).as_rotvec())*180/np.pi)


def perturbation_target(nominal, pregrasp, case):
    """位置沿当前 TCP 指向预抓取点；姿态绕名义 TCP 的局部 Z 轴。"""
    direction = pregrasp[:3, 3]-nominal[:3, 3]
    norm = float(np.linalg.norm(direction))
    if norm < .008:
        raise ValueError('名义起点已接近 Align 出口，不能用于本次入口诊断')
    direction /= norm
    goal = nominal.copy()
    if case.startswith('position_'):
        goal[:3, 3] += direction*(.01 if case == 'position_in10' else -.01)
    if case.startswith('yaw_'):
        angle = -10 if case == 'yaw_neg10' else 10
        goal[:3, :3] = goal[:3, :3] @ Rotation.from_euler('z', angle, degrees=True).as_matrix()
    return goal, direction


class PreparationMotion:
    """仅用真实 env.step 准备扰动；不直接修改物体、关节位置或速度。"""
    def __init__(self, teacher, fk):
        self.teacher, self.fk, self.rows = teacher, fk, []

    def q(self):
        return _numpy(self.teacher.base_env.agent.robot.get_qpos())[0, :7].copy()

    def send(self, target):
        if len(self.rows) >= 100:
            raise RuntimeError('扰动准备超过 100 步')
        before = self.q(); old_pose = self.fk.pose_base(before)
        action = np.r_[(target-before)/.1, 1.].astype(np.float32)
        if np.max(np.abs(action)) > 1+1e-6:
            raise ValueError('准备动作超出关节增量限制')
        obs, _, terminated, truncated, _ = self.teacher.env.step(action)
        self.teacher.session.observation = obs
        self.teacher.session.progress = self.teacher.session.tracker.update(self.teacher._read_predicate_state())
        self.teacher.metrics = self.teacher.measure(open_command=1.)
        after = self.q(); new_pose = self.fk.pose_base(after)
        row = dict(step=len(self.rows)+1, command_q=np.asarray(target).tolist(), q_after=after.tolist(),
                   metrics=self.teacher.metrics,
                   tcp_velocity_base_m_s=((new_pose[:3, 3]-old_pose[:3, 3])/.05).tolist())
        self.rows.append(row)
        if np.max(np.abs(after-target)) > .05 or not np.isfinite(after).all():
            raise RuntimeError('扰动准备跟踪异常')
        if bool(terminated.item()) or bool(truncated.item()):
            raise RuntimeError('扰动准备中环境终止')
        if self.teacher.metrics['held'] or self.teacher.metrics['opening'] < .8:
            raise RuntimeError('扰动准备改变了空爪状态')

    def move(self, goal, steps=12):
        start = self.fk.pose_base(self.q()).astype(float)
        turn = Rotation.from_matrix(start[:3, :3].T @ goal[:3, :3]).as_rotvec()
        reference = self.q()
        for fraction in np.linspace(1/steps, 1, steps):
            pose = start.copy()
            pose[:3, 3] += fraction*(goal[:3, 3]-start[:3, 3])
            pose[:3, :3] = start[:3, :3] @ Rotation.from_rotvec(fraction*turn).as_matrix()
            reference = self.fk.inverse(pose, reference)
            self.send(reference)

    def hold(self, goal, steps=12):
        target = self.fk.inverse(goal, self.q())
        for _ in range(steps):
            self.send(target)

    def prepare(self, nominal, pregrasp, case):
        goal, direction = perturbation_target(nominal, pregrasp, case)
        desired_velocity = np.zeros(3)
        if case == 'standard':
            return goal, desired_velocity
        if case.startswith('velocity_'):
            desired_velocity = direction*(.05 if case == 'velocity_in05' else -.05)
            endpoint = nominal.copy()
            endpoint[:3, 3] += desired_velocity*VELOCITY_LEAD_SECONDS
            lead = endpoint.copy(); lead[:3, 3] -= desired_velocity*.30
            self.move(lead); self.hold(lead)
            # 六个连续控制间隔匀速经过名义点，保留实际动量直接交给学生。
            self.move(endpoint, steps=6)
        else:
            if case != 'settled':
                self.move(goal)
            self.hold(goal)
        return goal, desired_velocity


class AlignController(AtomicController):
    def __init__(self, teacher, preparation, instruction, output):
        self.teacher = teacher
        super().__init__(teacher.env, preparation, 5, LIMIT, instruction, output)

    def send_action(self, value):
        super().send_action(value)
        self.teacher.metrics = self.teacher.measure(open_command=float((value[-1]+1)/2))
        fault = None
        try:
            self.teacher.metrics = self.teacher.boundaries.observe(
                self.teacher.metrics, self.teacher.relative_pose, self.steps/20.)
        except RuntimeError as exc:
            fault = str(exc)
        if self.stop_reason != 'tracking-invalid':
            if fault:
                self.stop_reason = 'align-invariant-failure'
            elif self.teacher.boundaries.active == 2:
                self.stop_reason = 'success'
        # 不将边界异常再抛给通用执行器，避免异常后额外动作和计数歧义。
        self.chunk_stop_requested = self.stop_reason is not None
        with (self.output/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=self.steps, active=self.teacher.boundaries.active,
                                    metrics=self.teacher.metrics, fault=fault))+'\n')


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'model-cache', 'continued', 'parent-training', 'training', 'output', 'source-manifest'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--case', choices=CASES, action='append')
    args = parser.parse_args(); source = verify_source(args.source_manifest)
    args.output.mkdir(exist_ok=False); started = time.monotonic()
    cases = args.case or CASES
    if len(set(cases)) != len(cases):
        raise ValueError('不允许重复评估同一个条件')
    rows = [dict(case=case, seed=seed, status='not_run', success=False) for case in cases for seed in SEEDS]
    result = dict(status='running', data_use='four known tuning development scenes; no final test',
                  source_sha256=source, cases=list(cases), records=rows, max_student_steps_per_unit=LIMIT,
                  max_units=len(rows), max_wall_seconds=1200, training=False,
                  velocity_preparation_lead_seconds=VELOCITY_LEAD_SECONDS,
                  success='first Align EXIT and Grasp ENTRY; student-only after physical preparation')

    def persist():
        result['elapsed_s'] = time.monotonic()-started
        save(args.output/'result.json', result)

    def stop(*_):
        raise InterruptedError('停止诊断，保留所有已执行和未执行单元')

    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop); persist()
    try:
        config = json.loads((args.training/'config.json').read_text())
        trained = json.loads((args.training/'result.json').read_text())
        base, policy, upstream = load_parent(args)
        verify_upstream(upstream, config['upstream'])
        load_student(policy, args.training/'latest.pt', trained, config)
        result['checkpoint_sha256'] = trained['checkpoint_sha256']
        fk = TCPKinematics(); nominal_states = {}
        for row in rows:
            if time.monotonic()-started > 1200:
                raise TimeoutError('到达 20 分钟诊断上限')
            folder = args.output/f'{row["case"]}-{row["seed"]}'; folder.mkdir()
            controller = None; motion = None; plans = []; row['status'] = 'preparing'; persist()
            try:
                with HierarchyTeacher(None, max_episode_steps=800) as teacher:
                    teacher.initialize(row['seed']); teacher.run_until(1)
                    prep_steps = len(teacher.rows)
                    if any(x['fine_skill_id'] != 0 for x in teacher.rows):
                        raise ValueError('标准起点准备执行了 Align 教师动作')
                    motion = PreparationMotion(teacher, fk)
                    q = motion.q(); nominal = fk.pose_base(q).astype(float)
                    pregrasp = np.linalg.inv(fk.world_from_base) @ teacher.pregrasp_world
                    if row['seed'] not in nominal_states:
                        nominal_states[row['seed']] = q.copy()
                    difference = float(np.max(np.abs(q-nominal_states[row['seed']])))
                    if difference > 1e-6:
                        raise ValueError('同场景标准教师起点不能复现')
                    row.update(teacher_preparation_steps=prep_steps, nominal_joint_parity_max_abs=difference,
                               nominal_metrics=dict(teacher.metrics))
                    goal, desired_velocity = motion.prepare(nominal, pregrasp, row['case'])
                    actual = fk.pose_base(motion.q())
                    actual_velocity = np.asarray(motion.rows[-1]['tcp_velocity_base_m_s']) if motion.rows else None
                    row.update(perturbation_preparation_steps=len(motion.rows), entry_metrics=dict(teacher.metrics),
                        position_shift_mm=float(np.linalg.norm(actual[:3, 3]-nominal[:3, 3])*1000),
                        orientation_shift_deg=rotation_error(actual, nominal),
                        preparation_target_error_mm=float(np.linalg.norm(actual[:3, 3]-goal[:3, 3])*1000),
                        desired_velocity_base_m_s=desired_velocity.tolist(),
                        actual_velocity_base_m_s=None if actual_velocity is None else actual_velocity.tolist())
                    invalid = violations(ENTRY['align'], teacher.metrics)
                    row['entry_violations'] = invalid
                    if invalid:
                        row['status'] = 'entry_invalid'; continue
                    if row['case'] != 'standard':
                        if (row['preparation_target_error_mm'] > 3 or rotation_error(actual, goal) > 1
                                or np.linalg.norm(actual_velocity-desired_velocity) > .02):
                            row['status'] = 'perturbation_not_realized'; continue
                    # 单独的评估时钟从实际交接观测开始；不伪造先前完成动作。
                    teacher.boundaries = BoundaryTracker(active=1)
                    teacher.metrics = teacher.boundaries.observe(teacher.metrics, teacher.relative_pose, 0.)
                    if teacher.boundaries.active != 1:
                        row['status'] = 'already_at_exit'; continue
                    prep = AtomicPreparation(teacher.session.observation, teacher.session.tracker,
                                             teacher.session.progress, prep_steps+len(motion.rows))
                    controller = AlignController(teacher, prep, build_pick_place_task(row['seed'] % 3).instruction, folder)
                    save(folder/'initial.json', controller.audit())
                    executor = AtomicExecutor(fk); row['status'] = 'running'; persist()
                    while controller.stop_reason is None:
                        online = controller.online()
                        observations = dict(rgb_external=online.rgb_external, rgb_wrist=online.rgb_wrist,
                                            physical_proprio=online.physical_proprio)
                        np.savez_compressed(folder/'last_observation.npz', **observations)
                        if not plans:
                            np.savez_compressed(folder/'initial_observation.npz', **observations)
                        anchor = fk.pose_base(controller.read_state().joint_positions)
                        noise_seed = sampling_seed(row['seed'], 100000+len(plans))
                        physical = predict(base, policy, online, noise_seed)
                        plan = dict(step=controller.steps, sampling_seed=noise_seed,
                                    physical=physical.tolist(), base_from_tcp=anchor.tolist())
                        plans.append(plan)
                        try:
                            plan['execution'] = execute_plan(executor, controller, physical, anchor)
                        except ValueError as exc:
                            plan['rejection'] = str(exc); controller.stop_reason = 'plan-rejected'
                    row.update(status='completed', **controller.result(),
                               fine_completed=teacher.boundaries.active, final_metrics=teacher.metrics,
                               remaining_exit_violations=violations(EXIT['align'], teacher.metrics),
                               events=teacher.boundaries.events)
            except torch.cuda.OutOfMemoryError:
                raise
            except (ValueError, RuntimeError) as exc:
                row.update(status='error', success=False, error=f'{type(exc).__name__}: {exc}')
            finally:
                save(folder/'plans.json', plans)
                if motion is not None:
                    save(folder/'preparation.json', motion.rows)
                persist()
            print(json.dumps({k: row.get(k) for k in ('case', 'seed', 'status', 'success', 'policy_steps', 'stop_reason')}), flush=True)
        result['status'] = 'completed'
    except BaseException as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        persist()


if __name__ == '__main__':
    main()
