"""实验训练适配：旧五技能 train 回放保持标签，新序列逐帧核验 Memory。"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.memory_reobserve.protocol import PROTOCOL
from experiments.memory_conditioning.conditioning import MEMORY_SCHEMA
from robot_vla.adapters import ActionAdapter
from robot_vla.contracts import RobotSpec
from robot_vla.data.dataset import ActionChunkDataset
from robot_vla.data.trajectory import load_manifest


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def new_examples(root):
    root=Path(root)
    protocol=json.loads((root/'protocol.json').read_text())
    if protocol != PROTOCOL:
        raise ValueError('数据必须匹配冻结的新32场景协议')
    rows=json.loads((root/'collection.json').read_text())['records']
    if [r['seed'] for r in rows] != PROTOCOL['seeds']:
        raise ValueError('采集分母缺失、重复或替换')
    examples={'train':[], 'development':[]}; hashes={}; denominator=rows; seen={}
    for row in rows:
        seed=row['seed']; folder=root/str(seed); path=folder/'sequence.npz'
        capture=row.get('result',{}).get('capture') or {}
        if path.exists() != bool(capture.get('samples',0)):
            raise ValueError('样本文件与采集记录不匹配')
        if not path.exists():continue
        meta=json.loads((folder/'sequence.json').read_text())
        episode=f'g2c-main-engineering-{seed}'
        if meta.get('seed')!=seed or meta.get('episode_id')!=episode or row['result'].get('seed')!=seed:
            raise ValueError('数据 seed/Episode 与采集身份不一致')
        if meta['schema']!='memory-reobserve-sequence/v1' or meta['model_inputs_use_privileged_pose']:
            raise ValueError('序列 schema 或可部署数据边界不符')
        from robot_vla.precision.active_front_memory_provider import build_stage2_object_memory_config
        expected_config=json.loads(json.dumps(asdict(build_stage2_object_memory_config())))
        if meta['memory_config'] != expected_config:
            raise ValueError('Memory 特征缩放/资格配置不一致')
        hashes[str(seed)]=dict(npz=digest(path), metadata=digest(folder/'sequence.json'),
            result=digest(folder/'result.json'))
        if hashes[str(seed)]['npz'] in seen:
            raise ValueError('不同 seed 不能复用同一训练/开发序列')
        seen[hashes[str(seed)]['npz']]=seed
        with np.load(path,allow_pickle=False) as data:
            n=len(data['normalized_action']); records=meta['records']
            if len(records)!=n or data['normalized_action'].shape!=(n,8):
                raise ValueError('序列长度/动作维度不符')
            if (data['memory_features'].shape!=(n,12) or data['memory_available'].shape!=(n,)
                    or data['memory_available'].dtype!=np.bool_ or data['physical_proprio'].shape!=(n,15)):
                raise ValueError('输入 shape/mask 不符')
            if not np.allclose(np.diff(data['timestamp_s']),.05,rtol=0,atol=1e-8):
                raise ValueError('教师序列时间不连续')
            previous=np.concatenate((data['initial_previous_command_q_rad'][None],data['commanded_joint_target_rad'][:-1]))
            labels=np.concatenate((data['commanded_joint_target_rad']-previous,np.ones((n,1))),axis=1).astype(np.float32)
            expected=ActionAdapter(RobotSpec()).normalize(labels,strict=True)
            if not np.allclose(expected,data['normalized_action'],rtol=0,atol=1e-6):
                raise ValueError('标签不是实际 commanded-target delta')
            for i,record in enumerate(records):
                snapshot=record['snapshot']
                if (snapshot.get('episode_id')!=episode or snapshot['schema']!=MEMORY_SCHEMA or snapshot['timestamp_s']!=data['timestamp_s'][i]
                    or snapshot['available']!=bool(data['memory_available'][i])
                    or not np.allclose(snapshot['features'],data['memory_features'][i],rtol=1e-6,atol=1e-7)):
                    raise ValueError('快照与同帧数据不一致')
                if not snapshot['available'] and np.any(data['memory_features'][i]):
                    raise ValueError('不可用 Memory 未清零')
                if 'memory_stale' in snapshot['reasons']:
                    observed=snapshot.get('last_observed_timestamp_s')
                    if observed is None or snapshot['timestamp_s']-observed<=PROTOCOL['memory_max_age_s']+1e-12:
                        raise ValueError('memory_stale 原因与实际年龄不符')
                if snapshot['available'] and (row['result']['memory_write_count']!=1
                    or not row['result'].get('commit') or snapshot['reasons']
                    or data['timestamp_s'][i]-snapshot['last_observed_timestamp_s']>PROTOCOL['memory_max_age_s']+1e-12):
                    raise ValueError('可用 Memory 无有效提交或已经过期')
            anchors=data['anchors'].tolist()
            if anchors!=list(range(0,n-15,4)) or len(anchors)!=capture['samples']:
                raise ValueError('锚点不符合真实未来16步窗口')
            split='train' if seed in PROTOCOL['train_seeds'] else 'development'
            for i in anchors:
                examples[split].append(dict(seed=seed,anchor=i,source='g2c-sequence',skill_id=0,
                    timestamp_s=records[i]['snapshot']['timestamp_s'],
                    invalidation_reasons=records[i]['snapshot']['reasons'],
                    age_s=(None if records[i]['snapshot'].get('last_observed_timestamp_s') is None else
                        records[i]['snapshot']['timestamp_s']-records[i]['snapshot']['last_observed_timestamp_s']),
                    rgb_external=data['rgb_external'][i].copy(),rgb_wrist=data['rgb_wrist'][i].copy(),
                    physical_proprio=data['physical_proprio'][i].copy(),action=data['normalized_action'][i:i+16].copy(),
                    action_mask=np.ones(16,bool),features=data['memory_features'][i].copy(),
                    available=bool(data['memory_available'][i]),instruction='pick the cube and place it in the target region'))
    return examples,hashes,denominator


def replay_examples(root, normalizer):
    """只读 train；每技能128个预定窗口，保留跨技能标签，不修改 action mask。"""
    entries=load_manifest(root,split='train')
    if len(entries)!=176 or any(e.local_dagger is not None for e in entries):
        raise ValueError('本批只复用已盘点的176条 D0 train')
    dataset=ActionChunkDataset(str(root),entries,RobotSpec(),normalizer)
    buckets={i:[] for i in range(5)}
    by_entry={i:[] for i in range(len(entries))}
    for index,(entry,t) in enumerate(dataset.index):by_entry[entry].append((index,t))
    for entry,pairs in by_entry.items():
        arrays=dataset.store.get(entries[entry])
        for index,t in pairs:buckets[int(arrays.skill_id[t])].append(index)
    rng=np.random.default_rng(42)
    selected=sorted(int(i) for skill in range(5) for i in rng.choice(buckets[skill],128,replace=False))
    examples=[]
    for index in selected:
        x=dataset[index]
        examples.append(dict(seed=x['trajectory_id'],anchor=x['timestep'],source='d0',skill_id=x['skill_id'],
            rgb_external=x['rgb_external'],rgb_wrist=x['rgb_wrist'],proprio=x['proprio'],action=x['action'],
            action_mask=x['action_mask'],features=np.zeros(12,np.float32),available=False,instruction=x['instruction']))
    identity=dict(manifest_sha256=digest(Path(root)/'manifest.jsonl'),selected_indices=selected,
        sample_ids=[dict(episode=x['seed'],timestep=x['anchor'],skill_id=x['skill_id']) for x in examples])
    return examples,identity
