"""同一批真实教师轨迹的两种表达；FK重建机器人位姿，不向学生补物体GT。"""
from pathlib import Path
import numpy as np
from experiments.rgbd_memory_policy.data import load_examples
from experiments.tcp_memory_control.geometry import relative_features, pose_delta, TCPActionSpec
from robot_vla.adapters import ActionAdapter
from robot_vla.contracts import RobotSpec
from experiments.tcp_memory_control.protocol import PROTOCOL


def prepare_examples(root, kinematics):
    examples, hashes, denominator = load_examples(root)
    poses = {}
    for entry in denominator:
        if entry['status'] != 'completed':
            raise ValueError('共同数据不完整，不用不同分母比较两arm')
        seed = entry['seed']
        with np.load(Path(root)/str(seed)/'sequence.npz',allow_pickle=False) as z:
            actual = [kinematics.pose_base(q[:7]) for q in z['physical_proprio']]
            commanded = [kinematics.pose_base(q) for q in z['commanded_joint_target_rad']]
            previous = [kinematics.pose_base(q) for q in z['previous_command_q_rad']]
            joint_targets=z['commanded_joint_target_rad'].copy()
            actual_q=z['physical_proprio'][:,:7].copy()
        poses[seed] = (actual, commanded, previous, joint_targets, actual_q)
    spec = TCPActionSpec()
    for rows in examples.values():
        for x in rows:
            actual, commanded, previous, joint_targets, actual_q = poses[x['seed']];i = x['anchor'];anchor = actual[i]
            # 两arm每次重规划均从可观测actual状态起步，避免隐藏command reference。
            physical = np.array([np.r_[pose_delta(anchor if j==i else previous[j], commanded[j], anchor),1.]
                for j in range(i,i+spec.horizon)])
            x['action']=x['action'].copy()
            x['action'][0]=ActionAdapter(RobotSpec()).normalize(np.r_[joint_targets[i]-actual_q[i],1.].astype(np.float32),strict=True)
            x['tcp_action'] = spec.normalize(physical)
            x['tcp_features'] = relative_features(x['snapshot'],anchor,PROTOCOL['offset_base_m'])
            x['base_from_tcp'] = anchor
    return examples, hashes, denominator
