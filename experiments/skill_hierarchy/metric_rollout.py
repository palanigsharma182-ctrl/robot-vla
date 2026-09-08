"""只以 canonical 完成为终止，七技能和夹爪事件在旁路测量。"""
from collections import deque
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.skill_hierarchy.contract import ENTRY, EXIT, SKILLS, violations, cube_orientation_error_deg, rotation_distance_deg
from experiments.skill_hierarchy.metric_rules import CommandEvents
from experiments.skill_hierarchy.probe import HierarchyTeacher
from experiments.tcp_atomic_skills.runtime import AtomicController, AtomicExecutor, predict, execute_plan
from experiments.tcp_atomic_skills.protocol import save
from experiments.tcp_memory_control.protocol import sampling_seed
from robot_vla.sim.collector import AtomicPreparation, _numpy
from robot_vla.tasks.pick_place import build_pick_place_task


class RunBudget:
    def __init__(self, deadline, output):
        self.deadline=deadline; self.output=Path(output); self.student=0; self.teacher=0; self.episodes=0; self.stopped=False
    def check(self):
        if self.stopped or (self.deadline is not None and time.time()>=self.deadline):
            raise InterruptedError('停止信号或墙钟上限')
    def consume(self,kind):
        self.check()
        if self.student+self.teacher>=132480: raise InterruptedError('累计控制步上限')
        if kind=='student':
            if self.student>=115200: raise InterruptedError('学生控制步上限')
            self.student+=1
        else:
            if self.teacher>=17280: raise InterruptedError('教师准备步上限')
            self.teacher+=1
    def save(self):
        save(self.output/'budget.json',dict(deadline_unix=self.deadline,student_step_reservations=self.student,
            teacher_step_reservations=self.teacher,episodes=self.episodes,stopped=self.stopped,updated_unix=time.time()))


class PreparedTeacher(HierarchyTeacher):
    budget=None
    def _step_with_target(self,*args,**kwargs):
        if len(self.rows)>=120: raise ValueError('标准 Approach 准备超过 120 步')
        self.budget.consume('teacher')
        self.last_preparation_tcp_before=_numpy(self.base_env.agent.tcp_pose.to_transformation_matrix())[0].astype(float)
        return super()._step_with_target(*args,**kwargs)


class Observer:
    def __init__(self, teacher):
        self.teacher=teacher; self.initial_object=_numpy(teacher.base_env.cube.pose.to_transformation_matrix())[0].astype(float)
        self.grasp=teacher.poses[0].to_transformation_matrix().astype(float)
        self.pregrasp=teacher.pregrasp_world.astype(float)
        self.previous_tcp=getattr(teacher,'last_preparation_tcp_before',None); self.held_history=deque(maxlen=3)
        self.active=0; self.ever_stable=False; self.hits={}; self.commands=CommandEvents()
        for i,row in enumerate(teacher.rows):
            self.commands.observe(row['gripper_command'],i-len(teacher.rows))
        self.commands.events.clear(); self.commands.raw.clear()
        self.rows=[]; self.opportunities=[]; self.op_start=None; self.op_count=0; self.op_latched=False
        self.closed_events=[]; self.last_held=False; self.grasp_lost=False

    def read(self, tick, command=None):
        teacher=self.teacher
        tcp=_numpy(teacher.base_env.agent.tcp_pose.to_transformation_matrix())[0].astype(float)
        obj=_numpy(teacher.base_env.cube.pose.to_transformation_matrix())[0].astype(float)
        if tick==0: m=dict(teacher.metrics)
        else: m=teacher.measure(open_command=command)
        relative=np.linalg.inv(tcp)@obj
        if m['held']: self.held_history.append(relative)
        else: self.held_history.clear()
        pairs=[(a,b) for i,a in enumerate(self.held_history) for b in list(self.held_history)[i+1:]]
        stable=(len(self.held_history)==3 and max(np.linalg.norm(a[:3,3]-b[:3,3]) for a,b in pairs)<=.002
                and max(rotation_distance_deg(a[:3,:3],b[:3,:3]) for a,b in pairs)<=5)
        if self.ever_stable and self.last_held and not m['held'] and command is not None and command<.7:
            self.grasp_lost=True
        # NumPy 比较可能返回 bool_；结果必须始终保持原生 bool，含不稳定分支。
        self.last_held=bool(m['held']); self.ever_stable = bool(self.ever_stable or stable)
        m.update(grasp_stable=int(stable),release_commanded=int(self.ever_stable and command is not None and command>=.9))
        for name in SKILLS:
            if not violations(EXIT[name],m): self.hits.setdefault(name,tick)
        if self.active<7 and not violations(EXIT[SKILLS[self.active]],m):
            if self.active==6 or not violations(ENTRY[SKILLS[self.active+1]],m): self.active+=1
        velocity=None if self.previous_tcp is None else (tcp[:3,3]-self.previous_tcp[:3,3])/.05
        direction=self.pregrasp[:3,3]-tcp[:3,3]; distance=np.linalg.norm(direction)
        radial=None if velocity is None or distance<.001 else float(velocity@direction/distance)
        self.previous_tcp=tcp.copy()
        fixed_valid=(np.linalg.norm(obj[:3,3]-self.initial_object[:3,3])<=.005
                     and cube_orientation_error_deg(obj[:3,:3],self.initial_object[:3,:3])<=5)
        grasp_distance=float(np.linalg.norm(tcp[:3,3]-self.grasp[:3,3]))
        grasp_angle=cube_orientation_error_deg(tcp[:3,:3],self.grasp[:3,:3])
        opportunity=bool(fixed_valid and not m['held'] and grasp_distance<=.01 and grasp_angle<=10 and m['tcp_speed_m_s']<=.08)
        before_state=self.commands.state
        event=None if command is None else self.commands.observe(command,tick)
        if event and event['state']=='CLOSED': self.closed_events.append(dict(event))
        if command is not None:
            if not opportunity: self.op_latched=False
            self.op_count=self.op_count+1 if opportunity else 0
            if self.op_start is not None:
                status=None
                if stable or self.commands.state=='CLOSED': status='closed_or_grasped'
                elif not opportunity: status='lost_opportunity' if fixed_valid else 'fixed_target_invalid'
                elif tick-self.op_start>=20: status='missing_close'
                if status:
                    self.opportunities.append(dict(start=self.op_start,end=tick,status=status))
                    self.op_start=None
                    # 同一连续机会段只判定一次，直到几何条件真正中断。
                    self.op_latched=opportunity
            if self.op_count==3 and not self.op_latched:
                if self.commands.state=='CLOSED':
                    self.opportunities.append(dict(start=tick,end=tick,status='closed_or_grasped')); self.op_latched=True
                else: self.op_start=tick
        row=dict(tick=tick,metrics=m,grasp_distance_m=grasp_distance,grasp_angle_deg=grasp_angle,
            fixed_target_valid=bool(fixed_valid),opportunity=opportunity,command=command,
            command_state_before=before_state,command_state=self.commands.state,command_event=event,
            relative_tcp=(np.linalg.inv(self.pregrasp)@tcp).tolist(),radial_velocity=radial,
            strict_active=self.active)
        self.rows.append(row)
        return row

    def result(self):
        rows=self.rows; final=rows[-1]['tick']; closes=[]
        for event in self.closed_events:
            begin=event['start']; end=begin+40
            stable=any(r['metrics']['grasp_stable'] for r in rows if begin<=r['tick']<=end)
            status='grasp_after_close' if stable else ('failed_close' if final>=end else 'censored')
            closes.append(dict(**event,status=status))
        opp=list(self.opportunities)
        if self.op_start is not None: opp.append(dict(start=self.op_start,end=final,status='censored'))
        return dict(ever_stable=self.ever_stable,grasp_lost=self.grasp_lost,hits=self.hits,
            close_events=closes,opportunities=opp,chatter=self.commands.chatter(),
            late_close=[e for e in closes if any(o['status']=='missing_close' and o['end']<e['confirmed'] for o in opp)],
            strict_active=self.active,nearest_pregrasp_m=min(r['metrics']['pregrasp_distance_m'] for r in rows),
            not_reached=not self.closed_events and not opp)


class MetricController(AtomicController):
    def __init__(self,teacher,preparation,instruction,output,budget):
        super().__init__(teacher.env,preparation,5,400,instruction,output)
        self.budget=budget; self.observer=Observer(teacher); self.observer.read(0)
    def send_action(self,value):
        self.budget.consume('student')
        super().send_action(value)
        row=self.observer.read(self.steps,float((value[-1]+1)/2))
        with (self.output/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    def should_interrupt_before_action(self,value):
        self.budget.check()
        return super().should_interrupt_before_action(value)


def support_count(row, support):
    if row is None or row['radial_velocity'] is None: return None
    p=np.array(row['relative_tcp']); scenes=set()
    for item in support:
        if item['held']!=row['metrics']['held'] or item['command_state']!=row['command_state']: continue
        if abs(item['radial_velocity']-row['radial_velocity'])>.02: continue
        q=item['relative_tcp']
        if np.linalg.norm(q[:3,3]-p[:3,3])<=.01 and cube_orientation_error_deg(q[:3,:3],p[:3,:3])<=10:
            scenes.add(item['seed'])
    return len(scenes)


def feedback_records(output,seeds,resume=False):
    """只复用完整单元；中断尝试留档，其预算由运行级账本保留。"""
    output=Path(output)
    expected=[(s,start) for s in seeds for start in ('F','S')]
    if resume and output.exists():
        records=json.loads((output/'result.json').read_text())['records']
        if [(r['seed'],r['start']) for r in records]!=expected:
            raise ValueError('恢复闭环场景身份不匹配')
        for row in records:
            if row['status']=='completed':
                if not all(k in row for k in ('success','ever_stable','execution_error','failures','support')):
                    raise ValueError('已完成闭环记录不完整')
                continue
            folder=output/f"{row['seed']}-{row['start']}"
            if folder.exists():
                folder.rename(folder.with_name(folder.name+'-interrupted'))
            seed,start=row['seed'],row['start']
            row.clear(); row.update(seed=seed,start=start,status='not_run',success=False)
        return records
    output.mkdir(exist_ok=False)
    return [dict(seed=s,start=start,status='not_run',success=False) for s,start in expected]


def evaluate_closed(base,policy,fk,seeds,output,budget,support,resume=False):
    output=Path(output)
    records=feedback_records(output,seeds,resume)
    def persist(): save(output/'result.json',dict(records=records,status='running')); budget.save()
    persist()
    for row in records:
        if row['status']=='completed': continue
        budget.check()
        if budget.episodes>=288: raise InterruptedError('episode 上限')
        budget.episodes+=1
        folder=output/f"{row['seed']}-{row['start']}"; folder.mkdir(); ctrl=None; plans=[]
        row['status']='preparing'; persist()
        try:
            with PreparedTeacher(None,max_episode_steps=520) as teacher:
                teacher.budget=budget; teacher.initialize(row['seed'])
                if row['start']=='S':
                    teacher.run_until(1)
                    if violations(ENTRY['align'],teacher.metrics) or any(r['fine_skill_id']!=0 for r in teacher.rows):
                        raise ValueError('标准 Align 入口准备不合法')
                n=len(teacher.rows); session=teacher.session
                prep=AtomicPreparation(session.observation,session.tracker,session.progress,n)
                ctrl=MetricController(teacher,prep,build_pick_place_task(row['seed']%3).instruction,folder,budget)
                save(folder/'initial.json',ctrl.audit()); executor=AtomicExecutor(fk)
                while ctrl.stop_reason is None:
                    budget.check(); anchor=fk.pose_base(ctrl.read_state().joint_positions)
                    seed=sampling_seed(row['seed'],len(plans))
                    action=predict(base,policy,ctrl.online(),seed)
                    plan=dict(step=ctrl.steps,sampling_seed=seed,physical=action.tolist()); plans.append(plan)
                    try:
                        plan['execution']=execute_plan(executor,ctrl,action,anchor)
                    except ValueError as exc:
                        if not any(message in str(exc) for message in ('IK失败或超出关节范围','IK解未通过FK回代','TCP动作对应的关节增量过大')):
                            raise
                        ctrl.stop_reason='ik-or-execution-constraint'
                        plan['execution']=dict(success=False,executed_steps=0,error=str(exc))
                summary=ctrl.observer.result()
                row.update(status='completed',**ctrl.result(),**summary)
                row['execution_error']=ctrl.stop_reason not in ('success','step-budget-exhausted','environment-terminal')
                bad=not row['success'] and not row['ever_stable'] and not row['execution_error']
                row['failures']=[]; row['support']={}
                if bad and row['start']=='F' and 'approach' not in summary['hits']: row['failures'].append('approach')
                if bad and row['start']=='S' and 'align' not in summary['hits']: row['failures'].append('align')
                failed=[e for e in summary['close_events'] if e['status']=='failed_close']
                missed=[e for e in summary['opportunities'] if e['status']=='missing_close']
                # 夹爪反馈统一使用 S，避免混合 F/S 同场景而改变分母。
                if bad and row['start']=='S' and (failed or missed): row['failures'].append('gripper')
                for key in row['failures']:
                    if key=='gripper': tick=failed[0]['start']-1 if failed else missed[0]['end']
                    else: tick=120
                    state=next((r for r in ctrl.observer.rows if r['tick']==tick),None)
                    row['support'][key]=support_count(state,support)
                save(folder/'metrics.json',ctrl.observer.rows)
        except InterruptedError:
            row['status']='interrupted'; raise
        except torch.cuda.OutOfMemoryError:
            row['status']='error'; raise
        except Exception as exc:
            row.update(status='execution_error' if ctrl else 'preparation_failed',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if ctrl is not None:
                save(folder/'plans.json',plans)
                if 'policy_steps' not in row: row.update(policy_steps=ctrl.steps)
            persist()
    tasks={start:dict(successes=sum(r['success'] for r in records if r['start']==start),
            stable=sum(r['ever_stable'] for r in records if r['start']==start),
            execution_errors=sum(r['execution_error'] for r in records if r['start']==start),denominator=len(seeds)) for start in ('F','S')}
    failures={k:sorted({r['seed'] for r in records if k in r['failures']}) for k in ('approach','align','gripper')}
    sparse={k:sorted({r['seed'] for r in records if k in r['failures'] and r['support'].get(k) is not None and r['support'][k]<4}) for k in failures}
    result=dict(status='completed',valid=True,tasks=tasks,failures=failures,sparse=sparse,records=records)
    save(output/'result.json',result)
    return result
