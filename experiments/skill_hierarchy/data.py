"""BC 仅按需读取真实观测与动作；七技能 ID 只用于采样和诊断。"""
from pathlib import Path
import json
import numpy as np

from experiments.skill_hierarchy.collect import SCHEMA, TRAIN_SEEDS, DEV_SEEDS
from experiments.tcp_atomic_skills.protocol import sha
from robot_vla.contracts import RobotSpec
from robot_vla.data.trajectory import TrajectoryStore, load_manifest


def example(arrays, labels, index, *, seed, trajectory_id, instruction):
    """沿用现有共享 TCP BC 的输入形式；不从 privileged sidecar 构造模型输入。"""
    return dict(seed=seed, trajectory_id=trajectory_id, anchor=index,
                skill_id=int(labels['fine_skill_id'][index]),
                rgb_external=arrays.rgb_external[index].copy(), rgb_wrist=arrays.rgb_wrist[index].copy(),
                physical_proprio=arrays.proprio[index].copy(), instruction=instruction,
                action=labels['action'][index].copy(), action_mask=labels['action_mask'][index].copy(),
                features=np.zeros(12, np.float32), available=False)


class SevenSkillWindows:
    """保存轻量索引及标签，一次只缓存一条 RGB 轨迹；保留全部有效锚点。"""
    def __init__(self, collection, split):
        if split not in ('train', 'val'):
            raise ValueError('这里只读取 train 或 tuning development')
        self.collection = Path(collection)
        result = json.loads((self.collection/'collection.json').read_text())
        protocol = json.loads((self.collection/'protocol.json').read_text())
        expected = [('train',s) for s in TRAIN_SEEDS]+[('val',s) for s in DEV_SEEDS]
        if (result['schema'] != SCHEMA or result['status'] != 'completed' or not result['training_ready']
                or [(r['split'],r['seed']) for r in result['records']] != expected
                or protocol['planned'] != [list(x) for x in expected]
                or sha(self.collection/'protocol.json') != result['protocol_sha256']):
            raise ValueError('数据集身份、完整分母或审计状态不符')
        entries = load_manifest(self.collection/'dataset')
        records = {r['trajectory_id']:r for r in result['records'] if 'file' in r}
        if {e.trajectory_id for e in entries} != set(records):
            raise ValueError('manifest 与采集分母不符')
        self.entries = []; self.labels = []; self.index = []; self.buckets = [[] for _ in range(7)]
        self.hashes = {}
        for entry in entries:
            row = records[entry.trajectory_id]
            if row['status'] != 'completed' or entry.split != row['split'] or entry.randomization['seed'] != row['seed']:
                raise ValueError('manifest 的 split/seed 或标签状态不一致')
            if entry.split != split: continue
            for prefix, key, digest in [('dataset', 'file', 'sha256'), ('', 'label_file', 'label_sha256')]:
                path = self.collection/prefix/row[key]
                if sha(path) != row[digest]: raise ValueError(f'{key} SHA 不一致')
                self.hashes[str(path.relative_to(self.collection))] = row[digest]
            with np.load(self.collection/row['label_file'], allow_pickle=False) as payload:
                labels = {k:payload[k] for k in payload.files}
            n = entry.num_steps
            if (labels['action'].shape != (n,16,7) or labels['action'].dtype != np.float32
                    or labels['action_mask'].dtype != np.bool_
                    or labels['action_mask'].shape != (n,16)
                    or labels['fine_skill_id'].shape != (n,)
                    or not np.all(np.isin(labels['fine_skill_id'],range(7)))
                    or not np.array_equal(labels['anchor'],np.arange(n))
                    or not np.isfinite(labels['action']).all()
                    or np.abs(labels['action']).max()>1+1e-6):
                raise ValueError('BC 标签 shape/dtype/数值非法')
            expected_mask = np.arange(16)[None,:] < np.minimum(16,n-np.arange(n))[:,None]
            if not np.array_equal(labels['action_mask'],expected_mask):
                raise ValueError('mask 必须只在 episode 尾部截断')
            e = len(self.entries); self.entries.append(entry);self.labels.append(labels)
            for t,skill in enumerate(labels['fine_skill_id']):
                self.buckets[int(skill)].append(len(self.index));self.index.append((e,t))
        if any(not bucket for bucket in self.buckets): raise ValueError('至少一个技能没有样本')
        self.store = TrajectoryStore(self.collection/'dataset', RobotSpec(), cache_size=1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        e,t = self.index[index]; entry = self.entries[e]
        return example(self.store.get(entry), self.labels[e], t, seed=entry.randomization['seed'],
                       trajectory_id=entry.trajectory_id, instruction=entry.task.instruction)


def main():
    import argparse
    from experiments.tcp_atomic_skills.protocol import save, verify_source
    parser=argparse.ArgumentParser()
    for name in ('collection','output','source-manifest'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();source=verify_source(args.source_manifest)
    args.output.mkdir(exist_ok=False)
    result=dict(status='running',source_sha256=source,splits={})
    for split in ('train','val'):
        dataset=SevenSkillWindows(args.collection,split)
        sampled=[]
        for bucket in dataset.buckets:
            for index in (bucket[0],bucket[-1]):
                item=dataset[index]
                if (item['physical_proprio'].shape != (15,) or item['available']
                    or np.any(item['features']) or item['action'].shape != (16,7)
                    or not np.isfinite(item['physical_proprio']).all()
                    or any(item[k].ndim!=3 or item[k].shape[-1]!=3 or item[k].dtype!=np.uint8
                           for k in ('rgb_external','rgb_wrist'))):
                    raise ValueError('训练实际读取的样本接口无效')
                sampled.append(dict(trajectory_id=item['trajectory_id'],anchor=item['anchor'],skill_id=item['skill_id']))
        result['splits'][split]=dict(episodes=len(dataset.entries),windows=len(dataset),
            skill_windows=[len(b) for b in dataset.buckets],consumer_samples=sampled,files=dataset.hashes)
        save(args.output/'result.json',result)
    result['status']='passed';save(args.output/'result.json',result)
    print(json.dumps({s:{k:v for k,v in r.items() if k not in ('files','consumer_samples')}
                     for s,r in result['splits'].items()}),flush=True)


if __name__=='__main__':
    main()
