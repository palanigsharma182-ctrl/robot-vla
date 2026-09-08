"""把两条新教师完整任务的命令目标经TCP/IK重放，覆盖真实抓取和释放。"""
import argparse
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np

from experiments.tcp_atomic_skills.data import command_chunk, verify_commands, action_spec
from experiments.tcp_atomic_skills.protocol import TRAIN_SEEDS, save, verify_source
from experiments.tcp_atomic_skills.runtime import AtomicController, AtomicExecutor, execute_plan
from experiments.tcp_memory_control.geometry import TCPActionSpec
from experiments.tcp_memory_control.execution import TCPExecutionCandidate
from experiments.tcp_memory_control.kinematics import TCPKinematics


def main():
    from robot_vla.sim.collector import TrustedPickPlaceCollector
    from robot_vla.data.trajectory import TrajectoryStore, load_manifest
    from robot_vla.contracts import RobotSpec
    p = argparse.ArgumentParser()
    for key in ('collection', 'output', 'source-manifest'):
        p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args(); source = verify_source(args.source_manifest)
    args.output.mkdir(exist_ok=False); start = time.monotonic()
    root = args.collection/'dataset'; entries = load_manifest(root, split='train')
    store = TrajectoryStore(root, RobotSpec()); fk = TCPKinematics()
    records = [dict(seed=s, status='not_run') for s in TRAIN_SEEDS[:2]]
    result = dict(status='running', records=records, source_sha256=source)
    try:
        with TrustedPickPlaceCollector(None, max_episode_steps=600) as collector:
            for row in records:
                meta = next(e for e in entries if int(e.randomization['seed']) == row['seed'])
                arrays = store.get(meta); verify_commands(arrays)
                poses = [fk.pose_base(q) for q in arrays.commanded_joint_target_rad]
                prep = collector.prepare_atomic(seed=row['seed'], skill_name='reach')
                folder = args.output/str(row['seed']); folder.mkdir()
                ctrl = AtomicController(collector.env, prep, 5, arrays.num_steps, meta.task.instruction, folder)
                np.testing.assert_allclose(ctrl.online().physical_proprio, arrays.proprio[0], atol=1e-6, rtol=0)
                executor = AtomicExecutor(fk); plans = []; row['status'] = 'running'
                try:
                    while ctrl.stop_reason is None:
                        anchor = fk.pose_base(ctrl.read_state().joint_positions)
                        action, mask, _ = command_chunk(anchor, poses, arrays.action[:, -1], ctrl.steps)
                        physical = action_spec().denormalize(action)
                        # 不足四步时仅为规划补保持；控制器在原轨迹长度前中止，不发送padding。
                        n = int(mask.sum()); physical[n:, :6] = 0.; physical[n:, 6] = physical[n-1, 6]
                        plans.append(dict(step=ctrl.steps, physical=physical.tolist(),
                            execution=execute_plan(executor, ctrl, physical, anchor)))
                    row.update(status='completed', **ctrl.result())
                except ValueError as error:
                    row.update(status='stopped', error=str(error), **ctrl.result())
                finally:
                    save(folder/'plans.json', plans); save(args.output/'result.json', result)
        result['status'] = 'passed' if all(r.get('success') for r in records) else 'failed'
        if result['status'] != 'passed':
            raise RuntimeError('完整教师TCP接触重放未通过，暂不启动依赖训练')
    finally:
        result['elapsed_s'] = time.monotonic()-start
        save(args.output/'result.json', result)


if __name__ == '__main__':
    main()
