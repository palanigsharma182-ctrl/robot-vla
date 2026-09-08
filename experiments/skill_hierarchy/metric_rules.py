"""指标 BC 的纯数据规则；不读取 GPU，不介入动作执行。"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import numpy as np

VERSION = 'metric-bc-v2'
CHANNELS = ('translation_error_mm', 'rotation_delta_error_deg', 'gripper_mae')
TAU = np.array([1., .2, .05])
FLOOR = np.array([.1, .01, .005])
PHASES = ('switch', 'entry', 'exit', 'interior')


@dataclass
class CommandEvents:
    state: str = 'UNKNOWN'
    pending: str | None = None
    count: int = 0
    start: int = 0
    events: list = field(default_factory=list)
    raw: list = field(default_factory=list)

    def observe(self, opening, tick):
        if not np.isfinite(opening) or not 0 <= opening <= 1:
            raise ValueError('夹爪命令无效')
        self.raw.append(float(opening))
        target = 'CLOSED' if opening <= .3 else ('OPEN' if opening >= .7 else None)
        if target is None or target == self.state:
            self.pending = None; self.count = 0
            return None
        if target != self.pending:
            self.pending, self.count, self.start = target, 0, tick
        self.count += 1
        if self.count != 3:
            return None
        event = dict(state=target, previous=self.state, start=self.start, confirmed=tick)
        self.state = target; self.pending = None; self.count = 0
        self.events.append(event)
        return event

    def chatter(self):
        runs = []
        for end in range(9, len(self.raw)):
            a = np.array(self.raw[end-9:end+1]) >= .5
            if np.count_nonzero(a[1:] != a[:-1]) >= 4:
                if runs and end-9 <= runs[-1][1]: runs[-1][1] = end
                else: runs.append([end-9, end])
        return runs


def event_groups(ids, opening):
    """技能×互斥事件桶；真实命令去抖后使用候选起点。"""
    ids = np.asarray(ids, dtype=int)
    if len(ids) != len(opening) or not np.isin(ids, range(7)).all():
        raise ValueError('技能标签与命令长度不符')
    phase = np.full(len(ids), 3, dtype=int)
    edges = [0, *list(np.flatnonzero(ids[1:] != ids[:-1])+1), len(ids)]
    for a, b in zip(edges[:-1], edges[1:]):
        phase[max(a,b-8):b] = 2
        phase[a:min(b,a+8)] = 1
    commands = CommandEvents()
    for t, u in enumerate(opening):
        event = commands.observe(float(u), t)
        # 初始 OPEN 仅建立已知状态，不把 episode 开头误标成释放切换。
        if event and not (event['previous'] == 'UNKNOWN' and event['state'] == 'OPEN'):
            c = event['start']; phase[max(0,c-8):min(len(ids),c+9)] = 0
    return ids*4+phase


def group_index(dataset):
    groups=[]; scenes=[]
    for entry, labels in zip(dataset.entries, dataset.labels):
        # 标签归一化夹爪 [-1,1] 恢复到真实命令 [0,1]。
        g = event_groups(labels['fine_skill_id'], (labels['action'][:,0,6]+1)/2)
        groups.extend(g.tolist()); scenes.extend([entry.randomization['seed']]*len(g))
    return np.array(groups, int), np.array(scenes, int)


def aggregate_errors(records, groups):
    out = {}
    for g in range(28):
        chosen = [r for r in records if groups[r['index']] == g]
        by_scene = {}
        for r in chosen:
            by_scene.setdefault(r['seed'], []).append([r[k] for k in CHANNELS])
        if by_scene:
            values = np.array([np.mean(v, axis=0) for _,v in sorted(by_scene.items())])
            out[str(g)] = dict(scenes=sorted(by_scene), windows=len(chosen),
                median=np.median(values,axis=0).tolist(), mean=values.mean(axis=0).tolist(),
                p90=np.quantile(values,.9,axis=0).tolist())
    return out


def related(g):
    return [2] if g%4 == 0 else [0,1]


def failure_key(g):
    skill, event = divmod(g,4)
    if skill == 0 and event in (1,2): return 'approach'
    if skill == 1 and event in (1,2): return 'align'
    if skill in (1,2) and event == 0: return 'gripper'
    return None


def focus_distribution(base, previous_focus, grades):
    """先构造可行目标，再凸插值约束每桶概率变化，保留所有上限。"""
    base=np.asarray(base,float); old=np.asarray(previous_focus,float)
    w=np.asarray(grades,float)
    if w.sum(): w=np.minimum(.1, .5*w/w.sum())
    for s in range(7):
        excess=w[s*4:s*4+4].sum()-w.sum()/7
        if excess > .3-1/7:
            w *= (.3-1/7)/excess
    oldp=(1-old.sum())*base+old
    newp=(1-w.sum())*base+w
    change=float(np.max(np.abs(newp-oldp)))
    if change>.05: w=old+(.05/change)*(w-old)
    p=(1-w.sum())*base+w
    if (w.min() < -1e-9 or w.max()>.1+1e-9 or w.sum()>.5+1e-9
            or np.max(p.reshape(7,4).sum(axis=1))>.3+1e-9
            or np.max(np.abs(p-oldp))>.05+1e-9 or not np.isclose(p.sum(),1)):
        raise ValueError('采样概率约束失败')
    return w, p


class FeedbackSampler:
    """基础队列与 focus RNG 分离；完整状态可随 checkpoint 恢复。"""
    def __init__(self, groups, scenes, baseline):
        self.groups=np.asarray(groups); self.scenes=np.asarray(scenes)
        self.baseline=np.asarray(baseline).reshape(-1)
        self.cursor=0; self.rng=np.random.default_rng(1820042)
        self.focus=np.zeros(28); self.grades=np.zeros(28,dtype=int)
        self.exposure=np.zeros(len(groups),dtype=int); self.extra=np.zeros(len(groups),dtype=int)
        self.history=[]; self.references={}; self.retired=set(); self.decisions=[]
        self.base=np.array([np.count_nonzero(self.groups==g)/np.count_nonzero(self.groups//4==g//4)/7
                            for g in range(28)])

    def draw(self, step, adaptive):
        rows=[]
        for k in range(7):
            extra=False
            focus_slot=adaptive and self.focus.sum()>0 and k < (3 if step%2==0 else 4)
            g=None
            if focus_slot:
                z=self.rng.random()
                if z<2*self.focus.sum():
                    g=int(np.searchsorted(np.cumsum(2*self.focus),z,side='right'))
            if g is None:
                idx=int(self.baseline[self.cursor]); self.cursor+=1
            else:
                scene=int(self.rng.choice(np.unique(self.scenes[self.groups==g])))
                idx=int(self.rng.choice(np.flatnonzero((self.groups==g)&(self.scenes==scene))))
                extra=True
            rows.append((idx,extra))
        return rows

    def account(self, rows):
        for i, extra in rows:
            self.exposure[i]+=1; self.extra[i]+=int(extra)

    def coverage(self,g, start=None):
        mask=self.groups==g
        a=self.exposure if start is None else self.exposure-np.array(start['exposure'])
        f=self.extra if start is None else self.extra-np.array(start['extra'])
        used=mask&(a>0)
        return dict(exposure=int(a[mask].sum()), focus=int(f[mask].sum()), unique=int(used.sum()),
                    scenes=int(len(np.unique(self.scenes[used]))))

    def update(self, feedback, step):
        if not feedback['valid']: raise ValueError('无效反馈禁止更新分布')
        self.history.append(copy.deepcopy(feedback)); notes=[]
        if len(self.history)<2: return []
        prev=self.history[-2]
        for g in range(12):
            key=failure_key(g); st=str(g)
            if key is None or st not in feedback['m7'] or st not in prev['m7']: continue
            if len(np.unique(self.scenes[self.groups==g]))<8: continue
            now, before=feedback['m7'][st],prev['m7'][st]
            if min(len(now['scenes']),len(before['scenes']))<4: continue
            ix=related(g); e=np.array(now['median']); ep=np.array(before['median'])
            failures=set(feedback['failures'][key])&set(prev['failures'][key])
            persistent=len(failures)>=2
            low=bool(np.all(e[ix]<TAU[ix]) and np.all(ep[ix]<TAU[ix]))
            count=self.coverage(g)
            sparse=set(feedback.get('sparse',{}).get(key,[]))&set(prev.get('sparse',{}).get(key,[]))&failures
            family=[j for j in range(12) if failure_key(j)==key and np.any(self.groups==j)]
            family_ready=True
            for j in family:
                a=feedback['m7'].get(str(j)); b=prev['m7'].get(str(j)); cov=self.coverage(j); ji=related(j)
                if (a is None or b is None or min(len(a['scenes']),len(b['scenes']))<4
                        or not np.all(np.array(a['median'])[ji]<TAU[ji])
                        or not np.all(np.array(b['median'])[ji]<TAU[ji])
                        or cov['exposure']<128 or cov['unique']<32 or cov['scenes']<8): family_ready=False
            corrective=(persistent and low and step>=512 and family_ready and len(sparse)>=2)
            reason=None
            if corrective:
                self.retired.add(g); reason='corrective_data_candidate'
            ref=self.references.get(st)
            if ref and step-ref['step']>=512 and g not in self.retired:
                cov=self.coverage(g,ref); origin=np.array(ref['error']); ri=[i for i in ix if origin[i]>=TAU[i]]
                near=(origin-e)/np.maximum(origin,FLOOR)
                earlier=(origin-ep)/np.maximum(origin,FLOOR)
                view='F' if key=='approach' else 'S'
                no_task_gain=all(feedback0['tasks'][view]['successes']<=ref['tasks'][view]['successes']
                    and feedback0['tasks'][view]['stable']<=ref['tasks'][view]['stable']
                    and len(feedback0['failures'][key])>=ref['failure_count'] for feedback0 in (prev,feedback))
                if cov['exposure']>=128 and cov['focus']>=64 and cov['unique']>=32 and cov['scenes']>=8:
                    if persistent and no_task_gain and ri and np.all(near[ri]<.05) and np.all(earlier[ri]<.05):
                        self.retired.add(g); reason='resampling_exhausted_check_labels_features_optimization'
                    elif persistent and no_task_gain: reason='offline_only_progress'
                else: reason='exposure_insufficient'
            high=bool(np.any(e[ix]>=TAU[ix]) and np.any(ep[ix]>=TAU[ix]))
            if g in self.retired: self.grades[g]=0
            elif persistent and high:
                grade=1
                if ref and step-ref['step']>=256:
                    o=np.array(ref['error']); ri=[i for i in ix if o[i]>=TAU[i]]
                    if ri and np.all(((o-e)/np.maximum(o,FLOOR))[ri]<.05): grade=2
                self.grades[g]=min(grade,self.grades[g] or 1) if reason=='offline_only_progress' else grade
            else:
                now_eligible=len(feedback['failures'][key])>=2 and np.any(e[ix]>=TAU[ix])
                old_eligible=len(prev['failures'][key])>=2 and np.any(ep[ix]>=TAU[ix])
                if not now_eligible and not old_eligible: self.grades[g]=max(0,self.grades[g]-1)
            if reason: notes.append(dict(group=g,reason=reason,coverage=count,persistent_scenes=sorted(failures)))
        self.focus,p=focus_distribution(self.base,self.focus,self.grades)
        for g in range(28):
            st=str(g); key=failure_key(g)
            if self.focus[g]>0 and st not in self.references:
                self.references[st]=dict(step=step,error=feedback['m7'][st]['median'],
                    exposure=self.exposure.tolist(),extra=self.extra.tolist(),tasks=copy.deepcopy(feedback['tasks']),
                    failure_count=len(feedback['failures'][key]))
        self.decisions.append(dict(step=step,notes=notes,focus=self.focus.tolist(),probabilities=p.tolist(),grades=self.grades.tolist()))
        return notes

    def state(self):
        return dict(cursor=self.cursor,rng=self.rng.bit_generator.state,focus=self.focus.tolist(),grades=self.grades.tolist(),
            exposure=self.exposure.tolist(),extra=self.extra.tolist(),history=self.history,references=self.references,
            retired=sorted(self.retired),decisions=self.decisions)

    def restore(self,state):
        self.cursor=state['cursor']; self.rng.bit_generator.state=state['rng']
        for key in ('focus','grades','exposure','extra'): setattr(self,key,np.array(state[key],dtype=float if key=='focus' else int))
        self.history=copy.deepcopy(state['history']); self.references=copy.deepcopy(state['references'])
        self.retired=set(state['retired']); self.decisions=copy.deepcopy(state['decisions'])
