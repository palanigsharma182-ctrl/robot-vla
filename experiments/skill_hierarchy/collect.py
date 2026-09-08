"""七技能教师数据：真实观测/命令、独立语义 sidecar 和全部 BC 窗口。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from experiments.skill_hierarchy.contract import VERSION, SKILLS, BoundaryTracker, contract_document
from experiments.tcp_atomic_skills.data import command_chunk, verify_commands
from experiments.tcp_atomic_skills.protocol import save, sha, verify_source

TRAIN_SEEDS = tuple(range(1810000, 1810128))
DEV_SEEDS = tuple(range(1811000, 1811032))
SMOKE_SEEDS = (1819000, 1819001)
SCHEMA = 'seven-skill-teacher-bc/v1'


def validate_sidecar(arrays, sidecar):
    """逐动作重算七段出口，拒绝长度错位、提前切段和失真的命令来源。"""
    if sidecar['version'] != VERSION or sidecar['source'] != 'privileged-teacher-evaluation':
        raise ValueError('七技能 sidecar 版本或用途错误')
    rows = sidecar['rows']
    if len(rows) != arrays.num_steps or not rows:
        raise ValueError('sidecar 与轨迹长度不一致')
    tracker = BoundaryTracker()
    tracker.observe(rows[0]['before'], np.array(rows[0]['tcp_from_object_before']), 0.)
    for i, row in enumerate(rows):
        if row['action_index'] != i or row['fine_skill_id'] != tracker.active:
            raise ValueError(f'动作 {i} 的七段标签错位')
        if not np.allclose(row['commanded_joint_target_rad'], arrays.commanded_joint_target_rad[i], atol=1e-7, rtol=0):
            raise ValueError('sidecar 命令与轨迹不同')
        if row['gripper_command'] != float(arrays.action[i, -1]):
            raise ValueError('sidecar 夹爪命令与轨迹不同')
        measured = tracker.observe(row['after'], np.array(row['tcp_from_object_after']), (i+1)/20.)
        for key in ('grasp_stable', 'relative_drift_m', 'relative_drift_deg'):
            if not np.isclose(measured[key], row['after'][key], atol=1e-10, rtol=0):
                raise ValueError(f'{key} 无法由原始相对位姿复算')
        if i and (row['before'] != rows[i-1]['after']
                  or row['tcp_from_object_before'] != rows[i-1]['tcp_from_object_after']):
            raise ValueError('相邻动作状态不连续')
    if tracker.active != 7 or tracker.events != sidecar['events']:
        raise ValueError('七段事件不完整或不能复算')
    ids = np.array([r['fine_skill_id'] for r in rows], np.int16)
    if set(ids) != set(range(7)):
        raise ValueError('七段中存在空动作段')
    return ids


def build_labels(arrays, fine_ids, fk):
    """保留每个有效锚点和跨技能连续未来动作；仅 episode 末尾做 padding。"""
    verify_commands(arrays)
    targets = [fk.pose_base(q) for q in arrays.commanded_joint_target_rad]
    actions, masks = [], []
    max_error = 0.
    for i in range(arrays.num_steps):
        actual = fk.pose_base(arrays.proprio[i, :7])
        action, mask, error = command_chunk(actual, targets, arrays.action[:, -1], i)
        actions.append(action); masks.append(mask); max_error = max(max_error, error)
    return dict(action=np.stack(actions), action_mask=np.stack(masks),
                fine_skill_id=fine_ids, anchor=np.arange(arrays.num_steps, dtype=np.int32)), max_error


def main():
    from experiments.skill_hierarchy.probe import HierarchyTeacher
    from experiments.tcp_memory_control.kinematics import TCPKinematics
    from robot_vla.sim import PICK_CUBE_TO_REGION_ENV_ID
    from robot_vla.sim.collector import EpisodeRejected, _numpy
    from robot_vla.contracts import OBSERVATION_V2_VERSION, FINGER_FORCE_SENSOR_VERSION, RobotSpec
    from robot_vla.data.events import EVENT_STATE_CONTRACT_VERSION
    from robot_vla.data.trajectory import TrajectoryMeta, OutcomeEvidence, TrajectoryStore
    from robot_vla.data.writer import TrajectoryDatasetWriter
    from robot_vla.tasks.pick_place import build_pick_place_task

    class Teacher(HierarchyTeacher):
        def initialize(self, seed):
            super().initialize(seed)
            # canonical 的 action_source 数组专属 Local DAgger；纯教师不伪称 DAgger。
            self.session.recorder.record_action_provenance = False

        def _step_with_target(self, session, target_q, label, gripper_opening):
            before = self.relative_pose.copy(); n = len(self.rows)
            try:
                super()._step_with_target(session, target_q, label, gripper_opening)
            finally:
                if len(self.rows) == n+1:
                    self.rows[-1].update(tcp_from_object_before=before.tolist(),
                                        tcp_from_object_after=self.relative_pose.tolist())

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    source = verify_source(args.source_manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('sidecars', 'labels', 'failures'):
        (args.output/name).mkdir()
    planned = ([('train', s) for s in SMOKE_SEEDS] if args.smoke else
               [('train', s) for s in TRAIN_SEEDS]+[('val', s) for s in DEV_SEEDS])
    protocol = dict(schema=SCHEMA, contract=contract_document(), planned=[list(x) for x in planned],
                    data_use={'train': 'BC fitting', 'val': 'tuning development; not final test'},
                    max_episode_steps=600, max_wall_seconds=1800, retries_per_seed=0,
                    action_schema='tcp-anchor-command-delta-rotvec-gripper/v1',
                    translation_limit_m=.025, rotation_component_limit_rad=.1,
                    horizon=16, execute_steps=4, sampling='all episode anchors; seven skill buckets',
                    student_inputs='real RGB and proprio; no privileged state or fine_skill_id input',
                    perturbations='none; randomized native task scenes only',
                    teacher='MPlib screw planning; seven predicate first-hit boundaries',
                    minimum_success_fraction_per_split=.90)
    save(args.output/'protocol.json', protocol)
    records = [dict(split=s, seed=n, status='not_run') for s,n in planned]
    start = time.monotonic()
    result = dict(schema=SCHEMA, status='running', source_sha256=source,
                  protocol_sha256=sha(args.output/'protocol.json'), records=records)
    def persist():
        result['elapsed_s'] = time.monotonic()-start
        save(args.output/'collection.json', result)
    persist()
    fk = TCPKinematics()
    writer = TrajectoryDatasetWriter(args.output/'dataset', RobotSpec())
    store = TrajectoryStore(args.output/'dataset', RobotSpec(), cache_size=1)
    try:
        for row in records:
            if time.monotonic()-start >= protocol['max_wall_seconds']:
                raise TimeoutError('教师批次达到 30 分钟墙钟上限')
            row['status'] = 'running'; persist()
            with Teacher(None, max_episode_steps=600) as teacher:
                try:
                    teacher.initialize(row['seed'])
                    calibration = teacher._camera_calibration(teacher.session.observation)
                    initial = _numpy(teacher.base_env.cube.pose.p)[0].tolist()
                    goal = _numpy(teacher.base_env.goal_site.pose.p)[0].tolist()
                    teacher.run_until(7)
                    session = teacher.session
                    if not session.done or not session.progress.task_completed:
                        raise EpisodeRejected('七段完成但原完整任务终止证据未完成')
                    arrays = session.recorder.build()
                    identity = f"seven-skills-seed-{row['seed']}"
                    sidecar = dict(version=VERSION, source='privileged-teacher-evaluation',
                                   trajectory_id=identity, seed=row['seed'],
                                   world_from_pregrasp=teacher.pregrasp_world.tolist(),
                                   events=teacher.boundaries.events, rows=teacher.rows)
                    fine_ids = validate_sidecar(arrays, sidecar)
                    outcome = session.progress.outcome
                    meta = TrajectoryMeta(
                        trajectory_id=identity, source_episode_id=f"maniskill-seed-{row['seed']}",
                        file=f'trajectories/{identity}.npz', split=row['split'],
                        scene_id=f"{PICK_CUBE_TO_REGION_ENV_ID}:seed={row['seed']}",
                        task=build_pick_place_task(row['seed'] % 3), num_steps=arrays.num_steps,
                        camera_calibration=calibration,
                        randomization=dict(seed=row['seed'], environment_id=PICK_CUBE_TO_REGION_ENV_ID,
                            control_mode='pd_joint_delta_pos', event_state_contract_version=EVENT_STATE_CONTRACT_VERSION,
                            observation_contract_version=OBSERVATION_V2_VERSION,
                            finger_force_sensor_version=FINGER_FORCE_SENSOR_VERSION,
                            cube_initial_position_m=initial, goal_position_m=goal,
                            hierarchy_version=VERSION, teacher_source='MPlib seven-skill teacher'),
                        outcome_evidence=OutcomeEvidence(
                            predicate_version=session.tracker.config.version,
                            task_completed=session.progress.task_completed, final_is_released=not outcome.grasped,
                            stable_place_steps=session.progress.stable_place_steps,
                            external_goal_visible_steps=session.recorder.external_goal_visible_steps,
                            wrist_goal_visible_steps=session.recorder.wrist_goal_visible_steps,
                            both_goal_visible_steps=session.recorder.both_goal_visible_steps,
                            final_object_to_goal_distance_m=outcome.object_to_goal_distance_m,
                            final_object_linear_speed_m_s=outcome.object_linear_speed_m_s,
                            final_object_angular_speed_rad_s=outcome.object_angular_speed_rad_s))
                    path = writer.write(meta, arrays)
                    row.update(trajectory_id=identity, file=meta.file, sha256=sha(path), steps=arrays.num_steps)
                    sidecar['trajectory_sha256'] = row['sha256']
                    side_path = args.output/'sidecars'/f'{identity}.json'; save(side_path, sidecar)
                    row.update(sidecar=str(side_path.relative_to(args.output)), sidecar_sha256=sha(side_path))
                    # 独立从磁盘重新加载，验证实际写出的数据，而非只检查内存对象。
                    loaded = store.get(meta)
                    labels, error = build_labels(loaded, fine_ids, fk)
                    label_path = args.output/'labels'/f'{identity}.npz'
                    np.savez_compressed(label_path, **labels)
                    with np.load(label_path, allow_pickle=False) as check:
                        if any(not np.array_equal(check[k], v) for k,v in labels.items()):
                            raise ValueError('BC 标签写入重载不一致')
                    row.update(status='completed', label_file=str(label_path.relative_to(args.output)),
                               label_sha256=sha(label_path), skill_frames=np.bincount(fine_ids, minlength=7).tolist(),
                               maximum_roundtrip_error=error, outcome=meta.outcome_evidence.to_dict())
                except (EpisodeRejected, RuntimeError, ValueError) as exc:
                    row.update(status='teacher_rejected' if 'file' not in row else 'label_or_export_invalid',
                               error=f'{type(exc).__name__}: {exc}', recorded_actions=len(getattr(teacher,'rows',[])))
                    save(args.output/'failures'/f"{row['seed']}.json", dict(error=row['error'],
                        traceback=traceback.format_exc(), rows=getattr(teacher,'rows',[]),
                        events=getattr(getattr(teacher,'boundaries',None),'events',[])))
                persist()
                print(json.dumps({k:v for k,v in row.items() if k != 'outcome'}, ensure_ascii=False), flush=True)
        summary = {}
        for split in sorted({s for s,_ in planned}):
            selected = [r for r in records if r['split'] == split]
            good = [r for r in selected if r['status'] == 'completed']
            summary[split] = dict(planned=len(selected), completed=len(good),
                skill_frames=np.sum([r['skill_frames'] for r in good], axis=0).tolist() if good else [0]*7)
        result.update(status='completed', summary=summary,
                      training_ready=all(v['completed']/v['planned'] >= .90 for v in summary.values())
                      and not any(r['status']=='label_or_export_invalid' for r in records))
    except BaseException as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        persist()
    if not result['training_ready']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
