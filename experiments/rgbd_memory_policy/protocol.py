"""第二阶段隔离抓前接近实验；首轮运行前冻结，不改变第一阶段结果。"""
import hashlib
import json

from experiments.front_rgbd_memory.geometry import PROVIDER_ID

PROTOCOL = dict(
    schema='rgbd-memory-policy/v1', seeds=list(range(1500000, 1500024)),
    train_seeds=[1500000+i for i in range(24) if i % 3 != 2],
    development_seeds=[1500000+i for i in range(24) if i % 3 == 2],
    rollout_seeds=list(range(1500100, 1500104)),
    teacher_steps=96, policy_steps=88, horizon=16, execute_steps=4, anchor_stride=8,
    occlusion_start=16, occlusion_end=76, train_steps_per_arm=256,
    accumulation=2, learning_rate=1e-5, memory_dropout=.25, seed=42,
    approach_offset_m=.08, reach_threshold_m=.02,
    arms=['visual', 'memory'], evaluation_arms=['visual', 'memory', 'memory-masked'],
    provider=PROVIDER_ID, qwen_frozen=True, adapter_frozen=True,
    geometry='unchanged stage-one known 4cm red cube / three-plane provider',
    scope='kinematic no-collision static target; open-gripper pregrasp approach only',
    feedback='fresh RGB-D/Memory/proprio every real control step; new Qwen context each replan',
    success_claim='development evidence only; not full pick-place or physical control',
)


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def occluded(step):
    return PROTOCOL['occlusion_start'] <= step < PROTOCOL['occlusion_end']
