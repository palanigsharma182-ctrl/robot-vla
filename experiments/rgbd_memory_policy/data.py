"""新轨迹的因果时间、动作参考及Memory快照核验；不读取旧test。"""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from experiments.front_rgbd_memory.memory import candidate_config
from experiments.memory_conditioning.conditioning import MEMORY_SCHEMA
from experiments.memory_reobserve.runtime import observation_digest
from experiments.g2c_memory_integration.vla import sha256
from experiments.rgbd_memory_policy.protocol import PROTOCOL, identity
from robot_vla.adapters import ActionAdapter
from robot_vla.contracts import RobotSpec
from robot_vla.runtime.policy_runtime import OnlineObservation


def require(condition,message):
    if not condition:raise ValueError(message)


def validate_sequence(data,meta):
    n=len(data['normalized_action']);horizon=PROTOCOL['horizon']
    require(n==PROTOCOL['teacher_steps'],'序列未完成冻结的教师步数')
    require(data['normalized_action'].shape==(n,8),'action shape不符')
    require(data['physical_proprio'].shape==(n,15),'proprio shape不符')
    require(data['timestamp_s'].shape==(n,),'timestamp shape不符')
    require(data['memory_features'].shape==(n,12),'Memory shape不符')
    require(data['memory_available'].shape==(n,) and data['memory_available'].dtype==np.bool_,'Memory mask不符')
    require(len(meta['snapshots'])==n and meta['student_uses_gt'] is False,'快照数或GT边界不符')
    require(meta['protocol_sha256']==identity(PROTOCOL),'数据协议不符')
    require(meta['memory_config']==asdict(candidate_config()),'Memory配置不符')
    require(np.allclose(np.diff(data['timestamp_s']),.05,rtol=0,atol=1e-8),'时间不连续')
    previous=np.concatenate([data['initial_previous_command_q_rad'][None],data['commanded_joint_target_rad'][:-1]])
    require(np.allclose(previous,data['previous_command_q_rad'],rtol=0,atol=1e-6),'上一command reference不一致')
    labels=np.concatenate([data['commanded_joint_target_rad']-previous,np.ones((n,1))],axis=1).astype(np.float32)
    require(np.allclose(ActionAdapter(RobotSpec()).normalize(labels,strict=True),data['normalized_action'],rtol=0,atol=1e-6),'标签不是commanded-target差分')
    require(list(data['anchors'])==list(range(0,n-horizon+1,PROTOCOL['anchor_stride'])),'anchor不符')
    for i,s in enumerate(meta['snapshots']):
        require(s['schema']==MEMORY_SCHEMA and s['episode_id']==meta['episode_id'],'快照schema/Episode不符')
        require(s['timestamp_s']==data['timestamp_s'][i],'快照和输入时间不一致')
        require(bool(data['memory_available'][i])==s['available'],'快照mask不符')
        require(np.allclose(data['memory_features'][i],s['features'],rtol=1e-6,atol=1e-7),'快照数值不符')
        if s['available']:
            require(not s['reasons'] and 0<=s['timestamp_s']-s['last_observed_timestamp_s']<=2.5+1e-9,'可用快照含未来/过期观察')
            require(s['source_model_identity']==PROTOCOL['provider'] and s['source_camera']=='base_camera','快照来源不符')
        else:require(not np.any(data['memory_features'][i]),'不可用Memory没有清零')


def validate_frame_binding(data,meta,observations,instruction):
    require(len({r['timestamp_s'] for r in observations})==len(observations),'传感器时间重复')
    by_time={r['timestamp_s']:r for r in observations}
    for i,s in enumerate(meta['snapshots']):
        require(s['timestamp_s'] in by_time,'缺少输入时刻的传感器记录')
        row=by_time[s['timestamp_s']]
        online=OnlineObservation(data['rgb_external'][i],data['rgb_wrist'][i],data['physical_proprio'][i],instruction)
        require(row['input_digest']==observation_digest(online),'RGB/proprio/指令与输入时刻不一致')
        require(row['snapshot']==s,'Memory不是该输入时刻实际读出的快照')


def load_examples(root):
    from experiments.rgbd_memory_policy.stream import INSTRUCTION
    root=Path(root)
    require(json.loads((root/'protocol.json').read_text())==PROTOCOL,'采集协议不符')
    records=json.loads((root/'collection.json').read_text())['records']
    require([r['seed'] for r in records]==PROTOCOL['seeds'],'采集分母不完整')
    result={'train':[],'development':[]};hashes={};seen=set()
    for r in records:
        seed=r['seed'];folder=root/str(seed);path=folder/'sequence.npz'
        # 不完整教师轨迹保留为失败分母，不冒充冻结的96步序列。
        if r['status']!='completed':continue
        require(path.is_file() and r.get('samples',0)>0,'完成记录缺少样本')
        require(r['split']==('train' if seed in PROTOCOL['train_seeds'] else 'development'),'split不符')
        meta=json.loads((folder/'sequence.json').read_text())
        require(meta['episode_id']==f'rgbd-policy-train-{seed}','Episode与seed不符')
        hashes[str(seed)]={'sequence':sha256(path),'metadata':sha256(folder/'sequence.json'),'observations':sha256(folder/'observations.jsonl')}
        require(hashes[str(seed)]['sequence'] not in seen,'不同seed存在重复序列')
        seen.add(hashes[str(seed)]['sequence'])
        observations=[json.loads(line) for line in (folder/'observations.jsonl').read_text().splitlines()]
        with np.load(path,allow_pickle=False) as archive:
            # 每个数组只解压一次；逐帧摘要核验不能反复解压整条RGB序列。
            d={name:archive[name] for name in archive.files}
            validate_sequence(d,meta);validate_frame_binding(d,meta,observations,INSTRUCTION)
            require(len(d['anchors'])==r['samples'],'样本分母不符')
            for i in d['anchors']:
                i=int(i);s=meta['snapshots'][i]
                result[r['split']].append(dict(seed=seed,anchor=i,snapshot=s,
                    rgb_external=d['rgb_external'][i].copy(),rgb_wrist=d['rgb_wrist'][i].copy(),
                    physical_proprio=d['physical_proprio'][i].copy(),action=d['normalized_action'][i:i+PROTOCOL['horizon']].copy()))
    return result,hashes,records
