"""七个独立技能策略的真实状态交接、教师恢复和旁路验收。"""
from dataclasses import fields
from pathlib import Path
import json
import time

import numpy as np
import torch

from experiments.skill_hierarchy.contract import SKILLS, ENTRY, EXIT, violations
from experiments.skill_hierarchy.probe import HierarchyTeacher, BoundaryReached
from experiments.tcp_atomic_skills.runtime import AtomicController, AtomicExecutor, predict, execute_plan
from experiments.tcp_atomic_skills.data import command_chunk, verify_commands
from experiments.tcp_atomic_skills.protocol import save
from experiments.tcp_memory_control.protocol import sampling_seed
from robot_vla.sim.collector import AtomicPreparation, EpisodeRejected, _numpy
from robot_vla.data.trajectory import TrajectoryArrays


class Budget:
    def __init__(self, output, limit=150000, seconds=14400):
        self.output=Path(output); self.limit=limit; self.deadline=time.time()+seconds
        self.student=0; self.teacher=0; self.episodes=0; self.stopped=False

    def check(self):
        if self.stopped or time.time()>=self.deadline:
            raise InterruptedError('停止请求或本作业墙钟上限')

    def consume(self, kind):
        self.check()
        if self.student+self.teacher>=self.limit:
            raise InterruptedError('本作业控制步上限')
        if kind=='student': self.student+=1
        elif kind=='teacher': self.teacher+=1
        else: raise ValueError('未知预算类别')

    def save(self):
        save(self.output/'budget.json',dict(student=self.student,teacher=self.teacher,
             episodes=self.episodes,limit=self.limit,deadline=self.deadline,stopped=self.stopped))


def shifted_tcp_position(tcp_position, object_position, desired_object_position):
    """持物目标平移保留真实TCP到物体偏置，不假定方块位于夹爪中心。"""
    return np.asarray(tcp_position)+np.asarray(desired_object_position)-np.asarray(object_position)


class SkillTeacher(HierarchyTeacher):
    def measure(self, *, open_command):
        result=super().measure(open_command=open_command)
        tcp=_numpy(self.base_env.agent.tcp_pose.p)[0]
        result['grasp_distance_m']=float(np.linalg.norm(tcp-np.asarray(self.poses[0].p)))
        return result

    def initialize(self, seed):
        super().initialize(seed)
        self.seed=seed
        self.session.recorder.record_action_provenance=False
        self.recovery_start=None

    def _step_with_target(self, session, target_q, label, gripper_opening):
        if len(self.rows)>=900:
            raise EpisodeRejected('单session教师与学生总动作上限')
        if self.recovery_start is not None and len(self.rows)-self.recovery_start>=160:
            raise EpisodeRejected('教师恢复160动作上限')
        self.budget.consume('teacher')
        previous_rows = len(self.rows)
        try:
            super()._step_with_target(session,target_q,label,gripper_opening)
        finally:
            if len(self.rows) > previous_rows:
                self.rows[-1]['source'] = 'teacher'
            # BoundaryReached属于正常完成，也必须校验刚执行动作的跟踪误差。
            error=float(np.max(np.abs(np.asarray(target_q)-self._actual_arm_q())))
            if not np.isfinite(error) or error>.05:
                raise EpisodeRejected('教师跟踪误差超过0.05rad')

    def recover_skill(self, skill):
        """固定技能合同，从当前真实状态重新规划；持物阶段保持闭爪。"""
        import sapien
        if self.boundaries.active!=skill or self.session.done:
            return False
        state=self._read_predicate_state()
        if (skill<=1 and state.is_grasped) or (3<=skill<=5 and not state.is_grasped):
            return False
        self.recovery_start=len(self.rows)
        opening=1. if skill in (0,1,6) else 0.
        try:
            if skill in (0,1):
                self._move_to_pose(self.session,self.poses[1],gripper_opening=1.)
            elif skill==2:
                if not state.is_grasped:
                    grasp=sapien.Pose(_numpy(self.base_env.cube.pose.p)[0],self.poses[0].q)
                    self._move_to_pose(self.session,grasp,gripper_opening=1.)
                self._hold(self.session,gripper_opening=0.,steps=20)
            elif skill in (3,4,5):
                tcp=self.base_env.agent.tcp_pose
                tcp_p=_numpy(tcp.p)[0]; tcp_q=_numpy(tcp.q)[0]
                desired=np.array(state.object_position,dtype=float)
                if skill in (4,5): desired[:2]=np.array(state.goal_position)[:2]
                desired[2]=(state.goal_position[2]+.005 if skill==5 else state.support_center_z_m+.12)
                goal=sapien.Pose(shifted_tcp_position(tcp_p,state.object_position,desired),tcp_q)
                self._move_to_pose(self.session,goal,gripper_opening=0.)
            else:
                self._hold(self.session,gripper_opening=1.,steps=24)
            self._hold(self.session,gripper_opening=opening,steps=20)
        except BoundaryReached:
            return self.boundaries.active==skill+1
        return False


class SkillController(AtomicController):
    def __init__(self, teacher, stop_at, limit, instruction, output, record=False, fraction=None):
        self.teacher=teacher; self.stop_at=stop_at; self.record=record
        self.fraction=fraction; self.start_metrics=teacher.metrics.copy(); self.start_skill=teacher.boundaries.active
        self.start_tick=len(teacher.rows)
        s=teacher.session
        super().__init__(teacher.env,AtomicPreparation(s.observation,s.tracker,s.progress,len(teacher.rows)),
                         5,limit,instruction,output)

    def should_interrupt_before_action(self, value):
        if super().should_interrupt_before_action(value):
            return True
        # 只在纠偏采集时截获即将发出的提前闭爪，保留动作前真实状态给教师。
        # 普通独立评估仍完整执行学生动作，不用接管掩盖失败。
        if self.record and self.start_skill in (0, 1) and (float(value[-1])+1)/2 < .8:
            self.stop_reason = 'collection-before-close'
            self.chunk_stop_requested = True
            return True
        return False

    def send_action(self, value):
        t=self.teacher; s=t.session
        t.budget.consume('student')
        before=t.metrics.copy(); skill=t.boundaries.active
        actual=self.read_state().joint_positions.copy()
        correction=np.asarray(value[:7],np.float32)*self.spec.maniskill_arm_delta_range_rad
        command=actual+correction; opening=float((value[-1]+1)/2)
        if self.record:
            label=np.r_[command-s.previous_command_q,opening].astype(np.float32)
            forces=t._read_contact_forces(); tcp,camera=t._read_observation_v2_poses(s.observation)
            s.recorder.record_before_action(s.observation,label,s.progress.active_skill_id,t._read_predicate_state(),
                *forces,command,correction,base_from_tcp=tcp,base_from_wrist_camera=camera,
                finger_force_n=t._last_finger_force_n.copy(),previous_command_q_rad=s.previous_command_q.copy())
        super().send_action(value)
        obs,_,terminated,truncated,info=self.last_step_output
        if self.record: s.recorder.record_after_action(terminated,truncated,info)
        s.observation=obs; s.progress=self.progress; s.previous_command_q=command.astype(np.float32)
        s.done=bool(terminated.item()) or bool(truncated.item())
        t.metrics=t.measure(open_command=opening); fault=None
        try:
            t.metrics=t.boundaries.observe(t.metrics,t.relative_pose,(self.start_tick+self.steps)/20.)
        except RuntimeError as exc:
            fault=str(exc)
        t.rows.append(dict(action_index=len(t.rows),fine_skill_id=skill,before=before,after=t.metrics.copy(),
                           commanded_joint_target_rad=command.tolist(),gripper_command=opening,source='student'))
        if self.stop_reason!='tracking-invalid':
            if fault: self.stop_reason='skill-invariant-failure'
            elif t.boundaries.active>=self.stop_at: self.stop_reason='success'
            elif s.done: self.stop_reason='environment-terminal'
            elif self.fraction is not None and skill_progress(self.start_skill,self.start_metrics,t.metrics)>=self.fraction:
                self.stop_reason='collection-progress-reached'
            elif self.steps>=self.limit: self.stop_reason='step-budget-exhausted'
            else: self.stop_reason=None
        self.chunk_stop_requested=self.stop_reason is not None
        with (self.output/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=self.steps,active=t.boundaries.active,metrics=t.metrics,fault=fault))+'\n')


def rollout(base, policy, teacher, fk, skill, limit, instruction, output, *, record=False, stop_at=None, fraction=None):
    controller=SkillController(teacher,skill+1 if stop_at is None else stop_at,limit,instruction,output,record,fraction)
    executor=AtomicExecutor(fk); plans=[]
    while controller.stop_reason is None:
        teacher.budget.check()
        anchor=fk.pose_base(controller.read_state().joint_positions)
        seed=sampling_seed(teacher.seed,skill*10000+len(plans))
        action=predict(base,policy,controller.online(),seed)
        plan=dict(step=controller.steps,seed=seed,physical=action.tolist()); plans.append(plan)
        try:
            plan['execution']=execute_plan(executor,controller,action,anchor)
        except ValueError as exc:
            if not any(m in str(exc) for m in ('IK失败或超出关节范围','IK解未通过FK回代','TCP动作对应的关节增量过大')):
                raise
            controller.stop_reason='ik-or-execution-constraint'; plan['error']=str(exc)
    save(Path(output)/'plans.json',plans)
    return controller


def skill_progress(skill, initial, current):
    """采集分层用连续进度，不能替代合同完成判据。"""
    if skill in (0,1):
        target=.040 if skill==0 else .008
        a=initial['pregrasp_distance_m']-target; b=current['pregrasp_distance_m']-target
    elif skill==2:
        # 最后接近约5cm，结合闭爪进度；抓持是否成立仍只看合同。
        return float(np.clip(.5*(1-current['grasp_distance_m']/.05)+.5*(1-current['opening']),0,1))
    elif skill==3:
        a=.080-initial['clearance_m']; b=.080-current['clearance_m']
    elif skill==4:
        a=initial['goal_xy_m']-.015; b=current['goal_xy_m']-.015
    elif skill==5:
        a=abs(initial['goal_z_m']-.005); b=abs(current['goal_z_m']-.005)
    else:
        return float(np.clip(current['opening'],0,1))
    return float(np.clip(1-b/max(a,1e-6),0,1))


def corrective_labels(arrays, takeover, teacher_end, fine_ids, fk):
    """保留完整真实教师段，包括尾部1至3步标签，绝不填造未来动作。"""
    verify_commands(arrays)
    if not 0<=takeover<teacher_end<=arrays.num_steps:
        raise ValueError('教师段边界非法')
    if len(fine_ids)!=arrays.num_steps:
        raise ValueError('动作与细技能标注不对齐')
    targets=[fk.pose_base(q) for q in arrays.commanded_joint_target_rad[:teacher_end]]
    anchors=np.arange(takeover,teacher_end,dtype=np.int32)
    actions=[]; masks=[]
    for i in anchors:
        action,mask,_=command_chunk(fk.pose_base(arrays.proprio[i,:7]),targets,arrays.action[:teacher_end,-1],int(i))
        actions.append(action); masks.append(mask)
    return dict(anchor=anchors,action=np.stack(actions),action_mask=np.stack(masks),
                fine_skill_id=np.asarray(fine_ids,dtype=np.int16)[anchors])


def slice_arrays(arrays, start):
    """只落盘目标技能入口后的真实数组；时间和previous-command原值保留。"""
    return TrajectoryArrays(**{f.name:None if getattr(arrays,f.name) is None else getattr(arrays,f.name)[start:]
                               for f in fields(arrays)})


def summarize(records):
    return {skill:dict(planned=sum(r['skill']==i for r in records),
             completed=sum(r['skill']==i and r['status']=='completed' for r in records),
             successes=sum(r['skill']==i and r.get('success',False) for r in records),
             statuses={s:sum(r['skill']==i and r['status']==s for r in records)
                       for s in sorted({r['status'] for r in records})}) for i,skill in enumerate(SKILLS)}
