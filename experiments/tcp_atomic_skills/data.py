"""五技能真实命令转 TCP 标签；夹爪目标与尾部 mask 按原记录保留。"""
from pathlib import Path
import json
import numpy as np

from experiments.tcp_atomic_skills.protocol import PROTOCOL, TRAIN_SEEDS, DEV_SEEDS, sha
from experiments.tcp_memory_control.geometry import TCPActionSpec, pose_delta, apply_delta
from robot_vla.contracts import RobotSpec
from robot_vla.data.trajectory import TrajectoryStore, load_manifest


def action_spec():
    return TCPActionSpec(translation_limit_m=PROTOCOL['tcp_translation_limit_m'])


def command_chunk(actual_pose, target_poses, gripper, anchor_index):
    """锚点 actual→首个 command，其后为相邻 command；不能用 actual 差分替代。"""
    n = len(target_poses)
    if len(gripper) != n or not 0 <= anchor_index < n:
        raise ValueError('命令长度或锚点无效')
    spec = action_spec(); length = min(spec.horizon, n - anchor_index)
    physical = np.zeros((spec.horizon, 7), np.float64)
    physical[:, 6] = .5  # masked padding，归一化后为零。
    previous = actual_pose
    maximum_roundtrip = 0.
    for k in range(length):
        target = target_poses[anchor_index + k]
        physical[k] = np.r_[pose_delta(previous, target, actual_pose), gripper[anchor_index + k]]
        reconstructed = apply_delta(previous, physical[k, :6], actual_pose)
        maximum_roundtrip = max(maximum_roundtrip, float(np.abs(reconstructed - target).max()))
        previous = target
    normalized = spec.normalize(physical)
    mask = np.arange(spec.horizon) < length
    # SAPIEN FK为float32姿态；此处矩阵误差预算不改变IK的物理位置/角度阈值。
    if maximum_roundtrip > 4e-6:
        raise ValueError('TCP标签FK往返不一致')
    return normalized, mask, maximum_roundtrip


def verify_commands(arrays):
    target, previous = arrays.commanded_joint_target_rad, arrays.previous_command_q_rad
    if target is None or previous is None:
        raise ValueError('缺少真实commanded target或previous command，不能重建标签')
    if not arrays.observation_valid.all() or not arrays.previous_command_valid.all():
        raise ValueError('本轮教师数据必须逐步具有有效观察与command provenance')
    if not np.allclose(target - previous, arrays.action[:, :7], rtol=0, atol=1e-6):
        raise ValueError('joint label与实际command来源不符')
    if not np.allclose(previous[1:], target[:-1], rtol=0, atol=1e-6):
        raise ValueError('跨步command不连续')
    if not np.allclose(np.diff(arrays.timestamp_action), .05, rtol=0, atol=1e-8):
        raise ValueError('20 Hz标签时间不连续')
    if any(not np.allclose(getattr(arrays, key), arrays.timestamp_action, rtol=0, atol=1e-8)
           for key in ('timestamp_external', 'timestamp_wrist', 'timestamp_proprio')):
        raise ValueError('观察与动作时间未对齐')
    if not np.all(np.isin(arrays.skill_id, range(5))):
        raise ValueError('未知技能标签')


def load_examples(collection, fk):
    collection = Path(collection); root = collection/'dataset'
    record = json.loads((collection/'collection.json').read_text())
    if (record['status'] != 'completed'
        or record['protocol']['train_seeds'] != list(TRAIN_SEEDS)
        or record['protocol']['development_seeds'] != list(DEV_SEEDS)
        or record['protocol']['schema'] != PROTOCOL['schema']):
        raise ValueError('采集未完成或协议改变')
    expected = [(s, 'train') for s in TRAIN_SEEDS] + [(s, 'val') for s in DEV_SEEDS]
    if [(r['seed'], r['split']) for r in record['records']] != expected:
        raise ValueError('采集分母改变')
    completed = {r['trajectory_id']: r for r in record['records'] if r['status'] == 'completed'}
    entries = load_manifest(root)
    if {e.trajectory_id for e in entries} != set(completed):
        raise ValueError('manifest与采集记录不一致')
    store = TrajectoryStore(root, RobotSpec(), cache_size=1)
    buckets = {s: {i: [] for i in range(5)} for s in ('train', 'val')}
    hashes = {}; counts = {s: [0]*5 for s in buckets}
    for index, entry in enumerate(entries):
        row = completed[entry.trajectory_id]
        digest = sha(root/entry.file)
        if digest != row['sha256'] or entry.split != row['split']:
            raise ValueError('轨迹SHA或split不符')
        hashes[entry.file] = digest
        arrays = store.get(entry); verify_commands(arrays)
        for i, skill in enumerate(arrays.skill_id):
            buckets[entry.split][int(skill)].append((index, i))
            counts[entry.split][int(skill)] += 1
    rng = np.random.default_rng(PROTOCOL['seed']); selected = {}; by_entry = {}
    for split in buckets:
        if sum(e.split == split for e in entries) < PROTOCOL['minimum_teacher_episodes'][split]:
            raise ValueError('完整教师轨迹不足，不能启动本轮训练')
        size = PROTOCOL['train_windows_per_skill' if split == 'train' else 'development_windows_per_skill']
        selected[split] = []
        for skill, bucket in buckets[split].items():
            if len(bucket) < size:
                raise ValueError(f'{split}/{skill}窗口不足')
            for k in rng.choice(len(bucket), size=size, replace=False):
                index, t = bucket[int(k)]
                selected[split].append(dict(trajectory_id=entries[index].trajectory_id, anchor=t, skill_id=skill))
                by_entry.setdefault(index, []).append((split, t, skill))
    output = {'train': [], 'development': []}; maximum_roundtrip = 0.
    for index, selection in sorted(by_entry.items()):
        entry = entries[index]; arrays = store.get(entry)
        targets = [fk.pose_base(q) for q in arrays.commanded_joint_target_rad]
        for split, t, skill in sorted(selection):
            anchor = fk.pose_base(arrays.proprio[t, :7])
            try:
                action, mask, error = command_chunk(anchor, targets, arrays.action[:, -1], t)
            except ValueError as error:
                raise ValueError(f'{entry.trajectory_id}/{t}/{skill}: {error}') from error
            maximum_roundtrip = max(maximum_roundtrip, error)
            output['train' if split == 'train' else 'development'].append(dict(
                seed=int(entry.randomization['seed']), trajectory_id=entry.trajectory_id, anchor=t, skill_id=skill,
                rgb_external=arrays.rgb_external[t].copy(), rgb_wrist=arrays.rgb_wrist[t].copy(),
                physical_proprio=arrays.proprio[t].copy(), instruction=entry.task.instruction,
                action=action, action_mask=mask, features=np.zeros(12, np.float32), available=False))
    return output, dict(collection_sha256=sha(collection/'collection.json'),
        manifest_sha256=sha(root/'manifest.jsonl'), files=hashes, skill_frames=counts,
        selected=selected, maximum_roundtrip_error=maximum_roundtrip,
        memory='masked; no qualified dynamic Memory in these recorded trajectories')
