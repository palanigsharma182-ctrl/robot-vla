"""五技能完整任务的隔离开发协议；不更改历史 Reach 夹具或 canonical。"""
import hashlib
import json
from pathlib import Path

SKILLS = ('reach', 'grasp', 'lift', 'transport', 'place')
TRAIN_SEEDS = tuple(range(1700000, 1700024))
DEV_SEEDS = tuple(range(1700100, 1700108))
EVAL_SEEDS = tuple(range(1700200, 1700204))
CASES = tuple((s, i, i + 1, 160) for i, s in enumerate(SKILLS)) + tuple(
    (SKILLS[i] + '+' + SKILLS[i + 1], i, i + 2, 240) for i in range(4)
) + (('full-task', 0, 5, 400),)
PROTOCOL = dict(schema='tcp-five-skills-development/v1', train_seeds=list(TRAIN_SEEDS),
    development_seeds=list(DEV_SEEDS), evaluation_seeds=list(EVAL_SEEDS), cases=[list(c) for c in CASES],
    steps=4096, learning_rate=1e-5, seed=1700042, train_windows_per_skill=256,
    development_windows_per_skill=32, accumulation=6,
    sampling='one window per each of five skills plus one original Reach window per update',
    memory='new five-skill data masked: no recorded qualified Memory; original Reach replay unchanged',
    conditioning='full-task language and actual observations; no ground-truth skill input',
    action='tcp-anchor-command-delta-rotvec-gripper/v1', tcp_translation_limit_m=.025,
    horizon=16, execute_steps=4,
    control_hz=20, joint_limit_rad=.1, tracking_limit_rad=.05, sampling_steps=10,
    selection='final planned update; no development checkpoint selection',
    development_purpose='new tuning development; dataset val field only encodes storage split',
    minimum_teacher_episodes=dict(train=16, val=4))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def verify_source(path):
    entries = json.loads(Path(path).read_text())
    required = ['experiments/tcp_atomic_skills/protocol.py', 'src/robot_vla/sim/collector.py',
                'experiments/tcp_memory_control/geometry.py', 'experiments/tcp_memory_control/kinematics.py']
    if not all(x in entries for x in required) or any(sha(f) != h for f, h in entries.items()):
        raise ValueError('实际源码与冻结快照不符')
    return sha(path)
