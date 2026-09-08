"""第一轮接管式DAgger：固定学生真实roll-in，教师在同一场景重新规划。"""
from collections import deque
from dataclasses import asdict
from pathlib import Path
import json
import shutil
import numpy as np

from experiments.skill_hierarchy.contract import rotation_distance_deg
from experiments.tcp_atomic_skills.runtime import AtomicController,AtomicExecutor,predict,execute_plan
from experiments.tcp_atomic_skills.protocol import save,sha
from experiments.tcp_memory_control.protocol import sampling_seed
from experiments.skill_dagger.data import training_seeds,teacher_windows,save_arrays
from robot_vla.sim.collector import TrustedPickPlaceCollector,AtomicPreparation,EpisodeRejected,_numpy
from robot_vla.tasks.pick_place import build_pick_place_task


class RecoveryComplete(Exception): pass


class RecoveryTeacher(TrustedPickPlaceCollector):
    def initialize(self,seed):
        self.session=self._start_session(seed,record_action_provenance=False)
        self.teacher_steps=0;self.teacher_limit=120;self.relative_history=deque(maxlen=3)
        self.recovering=False;self.phase='prefix';self.teacher_trace=[]

    def stable(self):
        state=self._read_predicate_state()
        if not state.is_grasped:self.relative_history.clear();return False
        tcp=_numpy(self.base_env.agent.tcp_pose.to_transformation_matrix())[0].astype(float)
        obj=_numpy(self.base_env.cube.pose.to_transformation_matrix())[0].astype(float)
        self.relative_history.append(np.linalg.inv(tcp)@obj)
        pairs=[(a,b) for i,a in enumerate(self.relative_history) for b in list(self.relative_history)[i+1:]]
        return bool(len(self.relative_history)==3 and all(
            np.linalg.norm(a[:3,3]-b[:3,3])<=.002 and rotation_distance_deg(a[:3,:3],b[:3,:3])<=5 for a,b in pairs))

    def _step_with_target(self,session,target_q,label,gripper_opening):
        if self.teacher_steps>=self.teacher_limit: raise EpisodeRejected('教师120动作预算耗尽')
        self.budget.check();self.budget.consume('teacher')
        self.teacher_steps+=1
        super()._step_with_target(session,target_q,label,gripper_opening)
        session.previous_command_q=np.asarray(target_q,dtype=np.float32).copy()
        tracking=float(np.max(np.abs(np.asarray(target_q)-self._actual_arm_q())))
        if not np.isfinite(tracking) or tracking>.05:raise EpisodeRejected('教师实际跟踪误差超过0.05rad')
        stable=self.stable()
        self.teacher_trace.append(dict(step=self.teacher_steps,phase=self.phase,held=bool(self._read_predicate_state().is_grasped),
            stable=stable,canonical_completed=session.progress.completed_skill_count))
        if stable and self.teacher_steps>=4 and session.progress.completed_skill_count>=2: raise RecoveryComplete()

    def recover(self):
        """目标由接管时实际物体状态生成，不能沿用旧示范的轨迹索引。"""
        self.recovering=True;self.relative_history.clear()
        try:
            if self._read_predicate_state().is_grasped:
                self.phase='stabilize';self._hold(self.session,gripper_opening=0.,steps=16)
            else:
                self.phase='open';self._hold(self.session,gripper_opening=1.,steps=3)
                grasp,pregrasp,*_=self._phase_poses()
                self.recovery_targets=dict(grasp=grasp.to_transformation_matrix().tolist(),pregrasp=pregrasp.to_transformation_matrix().tolist())
                self.phase='approach';self._move_to_pose(self.session,pregrasp,gripper_opening=1.)
                self.phase='grasp';self._move_to_pose(self.session,grasp,gripper_opening=1.)
                self.phase='close';self._hold(self.session,gripper_opening=0.,steps=16)
        except RecoveryComplete:return True
        return False


class RecordedStudent(AtomicController):
    def __init__(self,teacher,prefix,instruction,output):
        s=teacher.session
        super().__init__(teacher.env,AtomicPreparation(s.observation,s.tracker,s.progress,0),5,prefix,instruction,output)
        self.teacher=teacher;self.stable_now=False

    def send_action(self,value):
        t=self.teacher;s=t.session;t.budget.check();t.budget.consume('student')
        actual=self.read_state().joint_positions.copy()
        correction=np.asarray(value[:7],np.float32)*self.spec.maniskill_arm_delta_range_rad
        target=actual+correction;opening=float((value[-1]+1)/2)
        label=np.r_[target-s.previous_command_q,opening].astype(np.float32)
        forces=t._read_contact_forces();tcp,camera=t._read_observation_v2_poses(s.observation)
        s.recorder.record_before_action(s.observation,label,s.progress.active_skill_id,t._read_predicate_state(),
            *forces,target,correction,base_from_tcp=tcp,base_from_wrist_camera=camera,
            finger_force_n=t._last_finger_force_n.copy(),previous_command_q_rad=s.previous_command_q.copy())
        super().send_action(value)
        observation,_,terminated,truncated,info=self.last_step_output
        s.recorder.record_after_action(terminated,truncated,info)
        s.observation=observation;s.progress=self.progress;s.previous_command_q=target.astype(np.float32)
        s.done=bool(terminated.item()) or bool(truncated.item())
        self.stable_now=t.stable()


def pilot_gate(records):
    good=[r for r in records[:16] if r['status']=='recovered']
    return len(records)>=16 and len(good)>=12 and len({r['seed'] for r in good})>=6


def collect(base,policy,fk,collection,output,budget,parent_sha):
    output=Path(output);output.mkdir(exist_ok=False)
    seeds=training_seeds(collection,32)
    records=[dict(seed=s,prefix=k,status='not_run') for s in seeds for k in (60,120)]
    result=dict(schema='skill-dagger-r1',status='running',pilot_passed=False,records=records,
        planned_pilot=16,planned_total=64,max_control_steps=15360,student_sha256=parent_sha)
    def persist():save(output/'collection.json',result);budget.save()
    persist()
    for unit,row in enumerate(records):
        if unit==16:
            result['pilot_passed']=pilot_gate(records)
            if not result['pilot_passed']:
                result['status']='pilot_failed';persist();return result
        budget.check()
        if shutil.disk_usage(output).free<8*1024**3:raise RuntimeError('纠偏采集磁盘余量不足8GiB')
        folder=output/f"{row['seed']}-{row['prefix']}";folder.mkdir()
        row['status']='running';row['instruction']=build_pick_place_task(row['seed']%3).instruction;persist()
        with RecoveryTeacher(None,max_episode_steps=260) as teacher:
            teacher.budget=budget;teacher.initialize(row['seed'])
            controller=RecordedStudent(teacher,row['prefix'],row['instruction'],folder)
            plans=[];takeover=None;handoff=None
            try:
                executor=AtomicExecutor(fk)
                while controller.stop_reason is None:
                    budget.check();anchor=fk.pose_base(controller.read_state().joint_positions)
                    action=predict(base,policy,controller.online(),sampling_seed(row['seed'],len(plans)))
                    plan=dict(step=controller.steps,physical=action.tolist());plans.append(plan)
                    try:plan['execution']=execute_plan(executor,controller,action,anchor)
                    except ValueError as exc:
                        if not any(m in str(exc) for m in ('IK失败或超出关节范围','IK解未通过FK回代','TCP动作对应的关节增量过大')):raise
                        controller.stop_reason='ik-or-execution-constraint';plan['error']=str(exc)
                row['student_result']=controller.result()
                if controller.steps!=row['prefix'] or controller.stop_reason!='step-budget-exhausted':
                    row['status']='prefix_failed'
                elif controller.stable_now or teacher.session.done:
                    row['status']='no_candidate'
                else:
                    takeover=len(teacher.session.recorder.action);row['takeover']=takeover
                    online=controller.online()
                    handoff=(online.rgb_external.copy(),online.rgb_wrist.copy(),online.physical_proprio.copy())
                    row['handoff_observation_sha256']=controller.audit()['observation_sha256']
                    row['handoff_state']=asdict(teacher._read_predicate_state())
                    recovered=teacher.recover()
                    row['status']='recovered' if recovered else 'teacher_failed'
            except EpisodeRejected as exc:row.update(status='teacher_failed' if takeover is not None else 'prefix_failed',error=str(exc))
            except BaseException as exc:
                row.update(status='error',error=f'{type(exc).__name__}: {exc}');raise
            finally:
                row.update(student_steps=controller.steps,teacher_steps=teacher.teacher_steps)
                save(folder/'plans.json',plans);save(folder/'teacher_trace.json',teacher.teacher_trace)
                if hasattr(teacher,'recovery_targets'):save(folder/'targets.json',teacher.recovery_targets)
                if teacher.session.recorder.action and not teacher.session.recorder._pending_transition:
                    arrays=teacher.session.recorder.build();path=folder/'trajectory.npz';save_arrays(path,arrays)
                    row.update(trajectory=str(path.relative_to(output)),trajectory_sha256=sha(path))
                    if row['status']=='recovered':
                        try:
                            if not (np.array_equal(arrays.rgb_external[takeover],handoff[0])
                                    and np.array_equal(arrays.rgb_wrist[takeover],handoff[1])
                                    and np.allclose(arrays.proprio[takeover],handoff[2],atol=1e-7,rtol=0)):
                                raise ValueError('接管实际观测与首个教师标签锚点不同')
                            labels=teacher_windows(arrays,takeover,fk)
                            path=folder/'labels.npz';np.savez_compressed(path,**labels)
                            row.update(labels=str(path.relative_to(output)),labels_sha256=sha(path),windows=len(labels['anchor']),handoff_observation_parity=True)
                        except ValueError as exc:row.update(status='label_invalid',error=str(exc))
                persist()
    result.update(status='completed',pilot_passed=pilot_gate(records));persist();return result
