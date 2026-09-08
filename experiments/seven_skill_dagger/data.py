"""独立技能视图：只截断监督，不改历史全任务数据或冻结特征。"""
import numpy as np


def skill_targets(labels, anchor, skill):
    ids = labels['fine_skill_id']
    if not 0 <= anchor < len(ids) or int(ids[anchor]) != skill:
        raise ValueError('窗口不属于指定策略')
    end = anchor + 1
    while end < min(anchor + 16, len(ids)) and int(ids[end]) == skill:
        end += 1
    mask = labels['action_mask'][anchor].copy()
    mask &= np.arange(16) < end - anchor
    if not mask[0]:
        raise ValueError('当前技能没有有效动作')
    action = labels['action'][anchor].copy()
    action[~mask] = 0
    return action, mask


def independent_schedule(bucket, steps, accumulation, seed, *, offset=0):
    """按窗口数续接确定性遍历；offset 不改变随机种子或训练噪声。"""
    if not bucket or steps < 1 or accumulation < 1 or offset < 0:
        raise ValueError('技能桶和更新数必须有效')
    rng = np.random.default_rng(seed)
    indices = []
    end = offset + steps * accumulation
    while len(indices) < end:
        indices.extend(rng.permutation(bucket).tolist())
    return np.asarray(indices[offset:end]).reshape(steps, accumulation)


class CorrectiveWindows:
    """仅接受同技能、指定学生生成的train纠偏；逐文件核验接管与动作来源。"""
    def __init__(self, root, skill, student_sha, allowed_seeds):
        from pathlib import Path
        import json
        from experiments.tcp_atomic_skills.protocol import sha
        self.root = Path(root)
        self.student_sha = student_sha
        result = json.loads((self.root/'collection.json').read_text())
        if (result.get('status') != 'completed' or result.get('mode') != 'collect'
                or result.get('checkpoint_sha256') != student_sha):
            raise ValueError('纠偏采集状态或学生身份不一致')
        self.records = []
        self.index = []
        self.labels = []
        self.cached = None
        self.arrays = None
        seen = set()
        for row in result['records']:
            if row['skill'] != skill or row['split'] != 'train' or row['seed'] not in allowed_seeds:
                raise ValueError('跨技能或非训练场景数据')
            unit = (row['seed'], row['skill'], row['case'])
            if unit in seen:
                raise ValueError('采集单元重复')
            seen.add(unit)
            if row['status'] != 'recovered':
                continue
            if not row.get('success') or not row.get('handoff_parity'):
                raise ValueError('缺少恢复成功或真实接管一致性证据')
            for key in ('trajectory', 'labels', 'boundaries'):
                path = self.root/row[key]
                if not path.resolve().is_relative_to(self.root.resolve()) or sha(path) != row[key+'_sha256']:
                    raise ValueError('纠偏文件路径或SHA不一致')
            boundary = json.loads((self.root/row['boundaries']).read_text())
            with np.load(self.root/row['labels'], allow_pickle=False) as payload:
                labels = {k:payload[k] for k in payload.files}
            validate_corrective_labels(row, labels, boundary['rows'], skill)
            e = len(self.records)
            self.records.append(row)
            self.labels.append(labels)
            self.index.extend((e, i) for i in range(row['windows']))
        if not self.index:
            raise ValueError('没有合格纠偏样本')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        e, i = self.index[index]
        row, labels = self.records[e], self.labels[e]
        if e != self.cached:
            with np.load(self.root/row['trajectory'], allow_pickle=False) as payload:
                self.arrays = {k:payload[k] for k in ('rgb_external', 'rgb_wrist', 'proprio')}
            if any(len(v) != row['teacher_end'] for v in self.arrays.values()):
                raise ValueError('实际轨迹长度与监督终点不一致')
            self.cached = e
        t = int(labels['anchor'][i])
        return dict(seed=row['seed'], trajectory_id=f"correction-{row['seed']}-{row['skill']}-{row['case']}",
                    anchor=t, skill_id=row['skill'], instruction=row['instruction'],
                    rgb_external=self.arrays['rgb_external'][t].copy(),
                    rgb_wrist=self.arrays['rgb_wrist'][t].copy(),
                    physical_proprio=self.arrays['proprio'][t].copy(),
                    action=labels['action'][i].copy(), action_mask=labels['action_mask'][i].copy(),
                    features=np.zeros(12, np.float32), available=False)


def validate_corrective_labels(row, labels, rows, skill):
    """拒绝学生监督、跨技能目标、越过真实教师尾部和伪造未来mask。"""
    begin, end = row['takeover'], row['teacher_end']
    if not 0 <= begin < end or row['windows'] != end-begin:
        raise ValueError('接管段边界或窗口数非法')
    anchors = np.arange(begin, end)
    n = len(anchors)
    if not np.array_equal(labels['anchor'], anchors):
        raise ValueError('教师锚点缺失、重复或包含学生动作')
    action, mask, ids = (labels[k] for k in ('action','action_mask','fine_skill_id'))
    expected = np.arange(16)[None, :] < np.minimum(16, end-anchors)[:, None]
    if (action.shape != (n,16,7) or action.dtype != np.float32 or not np.isfinite(action).all()
            or np.abs(action).max() > 1+1e-6 or mask.dtype != np.bool_
            or not np.array_equal(mask, expected) or ids.shape != (n,) or not np.all(ids == skill)):
        raise ValueError('纠偏标签shape/dtype/数值/技能或尾mask非法')
    start = row['recording_start_tick']
    segment = rows[start+begin:start+end]
    if len(segment) != n or any(r.get('source') != 'teacher' or r['fine_skill_id'] != skill for r in segment):
        raise ValueError('监督不完全来自当前技能的真实教师段')
    if any(r['action_index'] != start+begin+i for i,r in enumerate(segment)):
        raise ValueError('动作索引与接管时间不一致')


class CorrectiveReplay:
    """跨学生轮次保留纠偏；同一父策略的同一采集单元不重复计入。"""
    def __init__(self, parts):
        if not parts or any(len(part) == 0 for part in parts):
            raise ValueError('纠偏回放来源为空')
        seen = set()
        for part in parts:
            for row in part.records:
                unit = (part.student_sha, row['seed'], row['skill'], row['case'])
                if unit in seen:
                    raise ValueError('跨轮回放重复计入同一学生采集单元')
                seen.add(unit)
        self.parts = parts
        self.ends = np.cumsum([len(part) for part in parts])

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        part = int(np.searchsorted(self.ends, index, side='right'))
        offset = 0 if part == 0 else int(self.ends[part-1])
        item = dict(self.parts[part][index-offset])
        item['trajectory_id'] = f"round-{part}-{item['trajectory_id']}"
        return item
