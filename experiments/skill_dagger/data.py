"""接管数据的监督边界：真实学生前缀留档，只有教师段参与拟合。"""
from dataclasses import fields
from pathlib import Path
import json
import numpy as np

from experiments.tcp_atomic_skills.data import command_chunk, verify_commands
from experiments.tcp_atomic_skills.protocol import sha
from robot_vla.data.trajectory import TrajectoryArrays


def training_seeds(collection, count):
    result=json.loads((Path(collection)/'collection.json').read_text())
    seeds=sorted(r['seed'] for r in result['records'] if r['split']=='train' and r['status']=='completed')
    if len(seeds)<count or len(seeds)!=len(set(seeds)): raise ValueError('合格train场景不足或重复')
    return seeds[:count]


def teacher_windows(arrays, takeover, fk):
    verify_commands(arrays)
    if not 0<=takeover<arrays.num_steps: raise ValueError('教师接管索引无效')
    targets=[fk.pose_base(q) for q in arrays.commanded_joint_target_rad]
    anchors=list(range(takeover,min(takeover+16,arrays.num_steps-3)))
    if not anchors: raise ValueError('接管段不足4个真实教师动作')
    actions=[];masks=[]
    for t in anchors:
        action,mask,_=command_chunk(fk.pose_base(arrays.proprio[t,:7]),targets,arrays.action[:,-1],t)
        actions.append(action);masks.append(mask)
    return dict(anchor=np.array(anchors,np.int32),action=np.stack(actions),action_mask=np.stack(masks))


def save_arrays(path,arrays):
    np.savez_compressed(path,**{f.name:getattr(arrays,f.name) for f in fields(arrays)
        if getattr(arrays,f.name) is not None})


class CorrectiveWindows:
    def __init__(self,root):
        self.root=Path(root);result=json.loads((self.root/'collection.json').read_text())
        if result['status']!='completed' or not result['pilot_passed']: raise ValueError('纠偏采集尚未通过')
        self.records=[];self.index=[];self.cached=None;self.arrays=None;self.labels=None
        for r in result['records']:
            if r['status']!='recovered': continue
            for key in ('trajectory','labels'):
                if sha(self.root/r[key])!=r[key+'_sha256']: raise ValueError('纠偏数据SHA不符')
            with np.load(self.root/r['labels'],allow_pickle=False) as x:
                anchors=x['anchor']
                if np.any(anchors<r['takeover']) or len(anchors)!=r['windows']: raise ValueError('学生动作泄漏为监督')
            e=len(self.records);self.records.append(r);self.index.extend((e,i) for i in range(r['windows']))
        if not self.index: raise ValueError('没有合格纠偏样本')

    def __len__(self): return len(self.index)

    def __getitem__(self,index):
        e,i=self.index[index];r=self.records[e]
        if e!=self.cached:
            with np.load(self.root/r['trajectory'],allow_pickle=False) as x:
                self.arrays={k:x[k] for k in ('rgb_external','rgb_wrist','proprio')}
            with np.load(self.root/r['labels'],allow_pickle=False) as x:self.labels={k:x[k] for k in x.files}
            self.cached=e
        t=int(self.labels['anchor'][i])
        return dict(seed=r['seed'],trajectory_id=f"dagger-{r['seed']}-{r['prefix']}",anchor=t,skill_id=2,
            instruction=r['instruction'],rgb_external=self.arrays['rgb_external'][t].copy(),
            rgb_wrist=self.arrays['rgb_wrist'][t].copy(),physical_proprio=self.arrays['proprio'][t].copy(),
            action=self.labels['action'][i].copy(),action_mask=self.labels['action_mask'][i].copy(),
            features=np.zeros(12,np.float32),available=False)
