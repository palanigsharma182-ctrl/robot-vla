"""完整教师命令经学生的 TCP/IK 执行链重放；同时检查七技能实时结果。"""
import argparse
from pathlib import Path
import json
import time
import numpy as np

from experiments.skill_hierarchy.probe import HierarchyTeacher
from experiments.skill_hierarchy.collect import TRAIN_SEEDS, DEV_SEEDS
from experiments.tcp_atomic_skills.data import verify_commands, command_chunk, action_spec
from experiments.tcp_atomic_skills.runtime import AtomicController, AtomicExecutor, execute_plan
from experiments.tcp_atomic_skills.protocol import save, sha, verify_source
from experiments.tcp_memory_control.kinematics import TCPKinematics
from robot_vla.sim.collector import AtomicPreparation
from robot_vla.data.trajectory import TrajectoryStore, load_manifest
from robot_vla.contracts import RobotSpec


class ReplayController(AtomicController):
    def __init__(self, teacher, *args):
        self.teacher = teacher
        super().__init__(teacher.env, *args)

    def send_action(self, value):
        super().send_action(value)
        teacher = self.teacher
        teacher.metrics = teacher.measure(open_command=float((value[-1]+1)/2))
        teacher.metrics = teacher.boundaries.observe(teacher.metrics, teacher.relative_pose, self.steps/20.)
        if teacher.boundaries.active == 7:
            self.stop_reason = 'success'


def main():
    parser = argparse.ArgumentParser()
    for field in ('collection', 'output', 'source-manifest'):
        parser.add_argument('--'+field, type=Path, required=True)
    args = parser.parse_args(); source = verify_source(args.source_manifest)
    args.output.mkdir(exist_ok=False)
    collection = json.loads((args.collection/'collection.json').read_text())
    if collection['status'] != 'completed' or not collection['training_ready']:
        raise ValueError('教师批次或标签审计未就绪')
    rows = [dict(seed=s, status='not_run') for s in (TRAIN_SEEDS[0], DEV_SEEDS[0])]
    result = dict(status='running', data_use='train/development diagnostic',
                  source_sha256=source, collection_sha256=sha(args.collection/'collection.json'), records=rows)
    start = time.monotonic(); fk = TCPKinematics()
    store = TrajectoryStore(args.collection/'dataset', RobotSpec(), cache_size=1)
    entries = load_manifest(args.collection/'dataset')
    try:
        for row in rows:
            record = next(r for r in collection['records'] if r['seed'] == row['seed'])
            if record['status'] != 'completed':
                row.update(status='failed', error='预定重放场景教师未成功，不替换 seed'); continue
            entry = next(e for e in entries if e.trajectory_id == record['trajectory_id'])
            if sha(args.collection/'dataset'/entry.file) != record['sha256']:
                raise ValueError('教师轨迹 SHA 不一致')
            arrays = store.get(entry); verify_commands(arrays)
            targets = [fk.pose_base(q) for q in arrays.commanded_joint_target_rad]
            folder = args.output/str(row['seed']); folder.mkdir()
            with HierarchyTeacher(None, max_episode_steps=600) as teacher:
                teacher.initialize(row['seed'])
                session = teacher.session
                preparation = AtomicPreparation(session.observation, session.tracker, session.progress, 0)
                ctrl = ReplayController(teacher, preparation, 5, arrays.num_steps, entry.task.instruction, folder)
                executor = AtomicExecutor(fk); plans = []
                try:
                    np.testing.assert_allclose(ctrl.online().physical_proprio, arrays.proprio[0], atol=1e-6, rtol=0)
                    while ctrl.stop_reason is None:
                        anchor = fk.pose_base(ctrl.read_state().joint_positions)
                        labels, mask, _ = command_chunk(anchor, targets, arrays.action[:, -1], ctrl.steps)
                        physical = action_spec().denormalize(labels)
                        n = int(mask.sum()); physical[n:, :6] = 0; physical[n:, 6] = physical[n-1, 6]
                        plans.append(dict(step=ctrl.steps, execution=execute_plan(executor, ctrl, physical, anchor)))
                    row.update(status='passed' if ctrl.result()['success'] and teacher.boundaries.active == 7 else 'failed')
                except (ValueError, RuntimeError) as exc:
                    row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                finally:
                    row.update(**ctrl.result(), seven_completed=teacher.boundaries.active,
                               events=teacher.boundaries.events)
                    save(folder/'plans.json', plans)
                    save(args.output/'result.json', result)
        result['status'] = 'passed' if all(r['status']=='passed' for r in rows) else 'failed'
    finally:
        result['elapsed_s'] = time.monotonic()-start
        save(args.output/'result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}), flush=True)
    if result['status'] != 'passed': raise SystemExit(1)


if __name__ == '__main__':
    main()
