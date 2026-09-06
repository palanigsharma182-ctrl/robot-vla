"""用户批准的两个耦合修改作为一个联合变量；不作单因素归因。"""
from dataclasses import asdict
import hashlib
import json
from experiments.rgbd_memory_policy.protocol import PROTOCOL as DATA_PROTOCOL
from experiments.tcp_memory_control.geometry import TCPActionSpec, FEATURE_SCHEMA

PROTOCOL = dict(
    schema='tcp-memory-joint-intervention/v1', arms=['joint-world','tcp-relative'],
    train_seeds=DATA_PROTOCOL['train_seeds'], development_seeds=DATA_PROTOCOL['development_seeds'],
    rollout_seeds=list(range(1600100,1600104)), control_seeds=[1600000],
    steps=256, accumulation=2, learning_rate=1e-5, memory_dropout=.25, seed=42,
    horizon=16, execute_steps=4, control_hz=20, policy_steps=88, reach_threshold_m=.02,
    action_spec=asdict(TCPActionSpec()), feature_schema=FEATURE_SCHEMA,
    provider=DATA_PROTOCOL['provider'], offset_base_m=[0.,0.,.08],
    shared_initialization='same upstream trunk; both action input/output projections reset; same memory initialization',
    command_reference='both arms reset to actual at every replan; first label uses actual-to-next-command, later labels successive commanded targets',
    flow_sampling='same per-microbatch seed and flow time; action dimensions differ, no channelwise noise equivalence claim',
    data='same verified 24 teacher trajectories; FK of actual and commanded joints; no object GT in model inputs',
    variable='base Memory + joint action versus TCP-relative target Memory + TCP pose delta action',
    primary='per-control-step Reach and paired final distance; raw FlowMSE not compared across action spaces',
    scope='bounded development pregrasp, same static no-collision fixture, gripper held open',
)


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sampling_seed(scene,index):
    # 不同场景/规划使用不重叠派生流；同场景两arm仍使用同一派生规则。
    return int(identity(['tcp-memory-sampling/v1',int(scene),int(index)])[:15],16)
