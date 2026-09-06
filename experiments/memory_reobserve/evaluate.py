"""新 development 场景的五技能/相邻组合/整任务，保留全部条件与失败分母。"""
import argparse
from dataclasses import asdict, dataclass, replace
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.g2c_memory_integration.run import run
from experiments.memory_reobserve.controller import MemoryDevelopmentController
from experiments.memory_reobserve.protocol import Acquisition
from experiments.memory_reobserve.runtime import load_trained_runtime, MemoryConditionedRuntime
from experiments.memory_reobserve.session import MemoryRouteSession, frame_safety
from robot_vla.adapters import FrankaObservationAdapter
from robot_vla.contracts import RobotSpec, PICK_AND_PLACE_SKILLS
from robot_vla.evaluation.maniskill import _read_observation_v2_frame, _read_predicate_state, _reset_atomic_time_limit
from robot_vla.execution.chunk_executor import RecedingHorizonChunkExecutor
from robot_vla.executive.contracts import PhaseId
from robot_vla.precision.active_front_reobserve import ActiveFrontTriggerEvidence, ActiveFrontTriggerReason
from robot_vla.precision.active_front_memory_provider import build_stage2_object_memory_config
from robot_vla.precision.object_memory import ExplicitObjectStateMemory
from robot_vla.runtime.control_loop import QwenVLAReplanLoop
from robot_vla.runtime.policy_runtime import OnlineObservation

SEEDS=tuple(range(1010200,1010204))
CASES=tuple((name,i,i+1,160) for i,name in enumerate(PICK_AND_PLACE_SKILLS))+tuple(
    (PICK_AND_PLACE_SKILLS[i]+'+'+PICK_AND_PLACE_SKILLS[i+1],i,i+2,240) for i in range(4))+(
    ('full-task',0,5,400),)
HOME_CUE_THRESHOLD=.6123920381069183
EVALUATION_PROTOCOL=dict(seeds=list(SEEDS),cases=[list(c) for c in CASES],
    conditions=['visual','fixed','evidence'],home_cue_threshold=HOME_CUE_THRESHOLD,
    cue_scope='unqualified HOME score candidate AND qualified wrist absent; not a qualified 3D measurement',
    cue_threshold_origin='fixed development candidate borrowed from existing write-score value, not calibrated request threshold',
    maximum_process_seconds=2600,attempts_per_episode=1,return_steps=10,
    visual_fallback=True,selection='report all; no checkpoint/seed replacement',
    interpretation='initial three real HOME ticks; no learned request head or repeated refresh; development only',
    atomic_scope='expert preparation only at start; no teacher actions or reset between paired skills')


class EvaluationSession(MemoryRouteSession):
    allowed_seeds=SEEDS


class EvaluationController(MemoryDevelopmentController):
    def __init__(self,*args,preparation,target_completed,max_steps,**kwargs):
        super().__init__(*args,**kwargs)
        self.tracker=preparation.tracker;self.progress=preparation.progress
        self.target_completed=target_completed
        self.stop_tick=self.tick+max_steps
        self.environment_terminated=False;self.environment_truncated=False
        self.completion_ticks={i:0 for i in range(preparation.progress.completed_skill_count)}
        self.trigger_enabled=False

    def send_action(self,value):
        if self.tick>=self.stop_tick:
            raise RuntimeError('policy-step-budget-exhausted')
        super().send_action(value)
        _,_,terminated,truncated,_=self.last_step_output
        self.environment_terminated|=bool(terminated.item())
        self.environment_truncated|=bool(truncated.item())
        old=self.progress.completed_skill_count
        # GT 仅在评估输出路径更新 predicate，不进入图像、Memory 或动作模型。
        self.progress=self.tracker.update(_read_predicate_state(self.env.unwrapped))
        for i in range(old,self.progress.completed_skill_count):self.completion_ticks[i]=self.tick
        if self.progress.completed_skill_count>=self.target_completed:
            self.episode_done=True;self.chunk_stop_requested=True
        if self.tick>=self.stop_tick:self.chunk_stop_requested=True

    def should_interrupt_before_action(self,value):
        if self.episode_done or self.tick>=self.stop_tick:return True
        return super().should_interrupt_before_action(value)


def rollout(*,runtime,seed,preparation,target_completed,max_steps,deadline,env,frame,memory,safety,episode,tick,output):
    session=EvaluationSession(seed,runtime,return_steps=10,visual_fallback=True)
    session.reset(episode)
    session.tracking_valid=bool(safety.controller_tracking_valid)
    controller=EvaluationController(env,RobotSpec(),session=session,memory=memory,
        initial_frame=frame,initial_tick=tick,home_sidecar=None,home_constraints=None,
        preparation=preparation,target_completed=target_completed,max_steps=max_steps)
    loop=QwenVLAReplanLoop(runtime,RecedingHorizonChunkExecutor(RobotSpec()))
    loop.control_step=tick;started_tick=tick; traces=[];failure=None
    while not controller.episode_done and controller.tick-started_tick<max_steps:
        if time.monotonic()>=deadline:failure='process-budget-exhausted';break
        if loop.observation_paused:failure='memory-or-control-hold';break
        try:
            controller.prepare_visual_replan(loop)
            if controller.tick>=controller.stop_tick:
                failure='step-budget-exhausted';break
            snapshot=session.bind(controller.frame,memory,'pick the cube and place it in the target region')
            result=loop.replan_and_execute(OnlineObservation(controller.frame.rgb_external,
                controller.frame.rgb_wrist,controller.frame.physical_proprio,
                'pick the cube and place it in the target region'),controller)
            traces.append(dict(tick_before=round(snapshot.timestamp_s*20),tick_after=controller.tick,
                snapshot=asdict(snapshot),execution=asdict(result.execution),
                completed_skill_count=controller.progress.completed_skill_count,
                sample_index=None if result.sampling is None else result.sampling.sample_index))
            session.cleanup_after_execution(loop)
            if not result.execution.success:
                failure=result.execution.failure_stage;break
        except Exception as error:
            failure=type(error).__name__+': '+str(error)
            session.interruption_reason='evaluation-error'
            session.cleanup_after_execution(loop);break
    final=controller.progress.completed_skill_count
    terminal_reason=('environment-terminated' if controller.environment_terminated else
        'environment-truncated' if controller.environment_truncated else
        'step-budget-exhausted' if controller.tick>=controller.stop_tick else 'stopped')
    result=dict(success=final>=target_completed,initial_completed=preparation.progress.completed_skill_count,
        final_completed=final,target_completed=target_completed,policy_steps=controller.tick-started_tick,
        observation_prefix_steps=started_tick,preparation_steps=preparation.preparation_steps,
        failure=None if final>=target_completed else failure or terminal_reason,
        terminal_reason=terminal_reason,stop_tick=controller.tick,
        completion_ticks=controller.completion_ticks,memory_inferences=sum(r['snapshot']['available'] for r in traces),
        visual_inferences=sum(not r['snapshot']['available'] for r in traces),cleanup=session.cleanup_records,
        final_predicate=asdict(_read_predicate_state(env.unwrapped)))
    (output/'policy.json').write_text(json.dumps(dict(result=result,traces=traces),indent=2)+'\n')
    return result


@dataclass(frozen=True)
class PolicyAcquisition(Acquisition):
    runtime: object
    condition: str
    preparation: object
    target_completed: int
    max_steps: int
    deadline: float

    def __post_init__(self):
        if self.seed not in SEEDS or self.condition not in EVALUATION_PROTOCOL['conditions']:
            raise ValueError('开发评估身份不符')
        object.__setattr__(self,'request_records',[])
        object.__setattr__(self,'request',None)

    def decide_request(self,evidence):
        # 三条实际 HOME 控制帧。高分仍不声称拥有 usable 3D measurement。
        session=EvaluationSession(self.seed,self.runtime)
        supervisor=session.supervisor
        supervisor.reset_episode(evidence[0][0].episode_id,episode_generation=1)
        if len(evidence)!=3:raise ValueError('触发必须检查三个真实 HOME 帧')
        previous=None
        for home,arm_hold in evidence:
            camera_home=bool(home.geometry_valid and home.home_capture_valid and home.pose_valid and home.timestamp_valid)
            if home.object_measurement_usable or not camera_home:
                raise ValueError('HOME 必须是实际几何有效的 score-only evidence')
            cue=self.condition=='fixed' or (self.condition=='evidence' and home.stored_write_score<HOME_CUE_THRESHOLD)
            reason=(ActiveFrontTriggerReason.NO_QUALIFIED_WRIST_PROVIDER_IN_PARENT if cue else ActiveFrontTriggerReason.UNKNOWN)
            t=home.control_timestamp_s;tick=round(t*20)
            if (abs(t-tick/20)>1e-8 or previous is not None and tick!=previous+1):
                raise ValueError('HOME trigger 帧必须是连续真实控制 tick')
            previous=tick
            item=ActiveFrontTriggerEvidence(home.episode_id,1,tick,t,PhaseId.ACQUIRE_TRACK,
                False,False,False,bool(arm_hold),camera_home,reason)
            decision=supervisor.consider_trigger(item)
            self.request_records.append(dict(timestamp_s=t,home_score=home.stored_write_score,
                provider_input=home.model_input_digest,provider_output=home.provider_output_digest,
                cue_candidate=cue,decision=asdict(decision),scope=EVALUATION_PROTOCOL['cue_scope']))
            if decision.requestable:object.__setattr__(self,'request',decision.request)
        return self.request is not None

    def capture(self,**kwargs):
        return rollout(runtime=self.runtime,seed=self.seed,preparation=self.preparation,
            target_completed=self.target_completed,max_steps=self.max_steps,deadline=self.deadline,**kwargs)


def main():
    from robot_vla.sim.collector import TrustedPickPlaceCollector
    p=argparse.ArgumentParser()
    for name in ('training','upstream','model-cache','bundle','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'protocol.json').write_text(json.dumps(EVALUATION_PROTOCOL,indent=2)+'\n')
    training=json.loads((a.training/'result.json').read_text())
    if training['status']!='training-completed':raise ValueError('两条件训练未完成')
    started=time.monotonic();deadline=started+2450
    rows=[dict(case=case,seed=seed,condition=condition,status='not-run')
        for case,_,_,_ in CASES for seed in SEEDS for condition in EVALUATION_PROTOCOL['conditions']]
    units={(r['case'],r['seed'],r['condition']):r for r in rows}
    def persist():
        temporary=a.output/'result.tmp'
        temporary.write_text(json.dumps(dict(protocol=EVALUATION_PROTOCOL,records=rows,
            elapsed_s=time.monotonic()-started),indent=2)+'\n')
        temporary.replace(a.output/'result.json')
    persist()
    for arm,conditions in [('visual',['visual']),('memory',['fixed','evidence'])]:
        base,identity=load_trained_runtime(a.upstream,a.model_cache,a.training/(arm+'.pt'),training['arms'][arm]['checkpoint_sha256'],
            expected_training_identity_sha256=training['training_identity_sha256'],expected_arm=arm)
        for case,start,target,max_steps in CASES:
            for seed in SEEDS:
                for condition in conditions:
                    row=units[(case,seed,condition)]
                    row['checkpoint_sha256']=identity['candidate_sha256']
                    folder=a.output/f'{condition}-{case}-{seed}'
                    if time.monotonic()>=deadline:
                        row.update(status='budget-not-run');persist();continue
                    row['status']='started';persist()
                    collector=None
                    try:
                        collector=TrustedPickPlaceCollector(None,RobotSpec(),max_episode_steps=1000)
                        preparation=collector.prepare_atomic(seed=seed,skill_name=PICK_AND_PLACE_SKILLS[start])
                        _reset_atomic_time_limit(collector.env)
                        sample_seed=int.from_bytes(hashlib.sha256(f'{seed}/{case}/42'.encode()).digest()[:8],'big')%(2**63-1)
                        runtime=MemoryConditionedRuntime(base.policy,base.processor_adapter,base.proprio_normalizer,
                            base.spec,base.device,replace(base.config,sampling_seed=sample_seed,starting_sample_index=0))
                        if time.monotonic()>=deadline:raise TimeoutError('preparation reached process deadline')
                        if start==0:
                            acquisition=PolicyAcquisition(seed,runtime,condition,preparation,target,max_steps,deadline)
                            result=run(a.bundle,folder,acquisition=acquisition,environment=collector.env)
                            row.update(status='completed',result=result['capture'],memory_write_count=result['memory_write_count'],
                                request=acquisition.request is not None,request_records=acquisition.request_records)
                        else:
                            folder.mkdir(parents=True,exist_ok=False)
                            memory=ExplicitObjectStateMemory(build_stage2_object_memory_config())
                            episode=f'{condition}-{case}-{seed}';memory.reset(episode)
                            frame=_read_observation_v2_frame(preparation.observation,collector.env.unwrapped,
                                FrankaObservationAdapter(base.spec),base.spec,control_step=0)
                            result=rollout(runtime=runtime,seed=seed,preparation=preparation,target_completed=target,
                                max_steps=max_steps,deadline=deadline,env=collector.env,frame=frame,memory=memory,
                                safety=frame_safety(frame),episode=episode,tick=0,output=folder)
                            row.update(status='completed',result=result,memory_write_count=0,request=False,
                                memory_scope='prepared later skill has no reconstructed Memory; masked from start')
                    except Exception as error:
                        row.update(status='error',error_type=type(error).__name__,error=str(error))
                    finally:
                        if collector is not None:collector.close()
                    persist()
                    print(json.dumps({k:row[k] for k in ('case','seed','condition','status')},ensure_ascii=False),flush=True)
        del base;gc.collect();torch.cuda.empty_cache()
    persist()
    if any(r['status']!='completed' for r in rows):raise SystemExit(2)


if __name__=='__main__':main()
