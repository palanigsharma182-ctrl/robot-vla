"""同初始场景联合对照；逐个真实控制步记录TCP、动作和Reach，不只看chunk末端。"""
from dataclasses import asdict
from pathlib import Path
import json
import time
import numpy as np
import torch
from experiments.g2c_memory_integration.vla import load_runtime,sha256
from experiments.memory_conditioning.conditioning import MemoryBatch,MEMORY_INPUT_KEY
from experiments.rgbd_memory_policy.stream import make_env,setup_scene,RGBDController
from experiments.rgbd_memory_policy.evaluate import initial_audit,verify_initial_pair
from experiments.rgbd_memory_policy.train import save_json
from experiments.memory_reobserve.runtime import observation_digest
from experiments.tcp_memory_control.geometry import relative_features,TCPActionSpec
from experiments.tcp_memory_control.kinematics import TCPKinematics
from experiments.tcp_memory_control.execution import (
    TCPExecutionCandidate, EXECUTION_CONFIG, require_execution_source, compatible_training_source)
from experiments.tcp_memory_control.policy import build_policy,ARMS
from experiments.tcp_memory_control.protocol import PROTOCOL,sampling_seed,identity
from experiments.tcp_memory_control.train import restore,verify_source
from robot_vla.execution.chunk_executor import RecedingHorizonChunkExecutor
from robot_vla.adapters import ActionAdapter
from robot_vla.contracts import RobotSpec
from robot_vla.runtime.policy_runtime import _move_model_inputs


def evaluation_protocol(expanded=False):
    """扩展开发对照与训练合同分开；旧场景只做复查，新场景承担主比较。"""
    return dict(
        schema='joint-baseline-vs-tcp-01-reach/v1' if expanded else 'tcp-memory-evaluation/v2',
        arms=list(ARMS), historical_seeds=list(PROTOCOL['rollout_seeds']),
        new_development_seeds=list(range(1600200,1600216)) if expanded else [],
        primary='new-development per-control-step Reach <=0.02m; all planned seeds retained',
        meaningful_signal='Reach improvement >=0.25 and paired exact two-sided p<=0.05; exploratory only',
        secondary=['completed_88_steps','per-seed minimum distance','paired common-prefix distance',
                   'tracking error','measured joint velocity','memory-conditioned actions'],
        control_hz=20,policy_steps=88,checkpoint_selection='reuse both last-256-step checkpoints; no retraining',
        comparison='base Memory + joint8 at .05 versus TCP-relative Memory + TCP7 at .1',
        inference_limit='combined scheme only; not isolated effect of .1 or all atomic skills',
        tracking_error_limit_rad=.05,
    )


class TraceController(RGBDController):
    def __init__(self,env,episode,output,metric_target_world):
        super().__init__(env,episode,output)
        # 此GT只供写出评估距离；不会传入观测、Memory、模型或动作生成。
        self.metric_target_world=np.asarray(metric_target_world)
        self.control_trace=[]

    def metric_distance(self):
        current=self.env.unwrapped.agent.tcp_pose.p.cpu().numpy()[0]
        return float(np.linalg.norm(current-self.metric_target_world))

    def send_action(self,value):
        q_before=self.read_state().joint_positions.copy()
        super().send_action(value)
        world_from_base=self.env.unwrapped.agent.robot.pose.to_transformation_matrix()[0].cpu().numpy()
        tcp_world=self.env.unwrapped.agent.tcp_pose.p.cpu().numpy()[0]
        row=dict(tick=self.tick,policy_step=self.policy_step,holding=self.holding,
            q_before=q_before.tolist(),q_after=self.frame.physical_proprio[:7].tolist(),
            base_from_tcp=self.frame.base_from_tcp.tolist(),controller_action=np.asarray(value).tolist(),
            command_q=self.last_target.tolist(),distance_m=self.metric_distance(),
            world_from_base=world_from_base.tolist(),tcp_world=tcp_world.tolist(),
            metric_target_world=self.metric_target_world.tolist(),
            joint_velocity_rad_s=self.frame.physical_proprio[7:14].tolist(),
            tracking_error_rad=(self.last_target-self.frame.physical_proprio[:7]).tolist(),
            action_used_memory=self.rows[-1]['action_used_memory'],occluded=self.current_occlusion)
        self.control_trace.append(row)
        with (self.output/'control-trace.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')


def frame_check(controller,fk):
    actual=fk.pose_base(controller.read_state().joint_positions)
    if not np.allclose(actual,controller.frame.base_from_tcp,atol=2e-5,rtol=0):
        raise ValueError('FK与实际TCP link/frame不一致')


def smoke(args):
    """六个平移/旋转轴的实际控制smoke；不加载模型，不以此宣称学习收益。"""
    source=verify_source(args.source_manifest);require_execution_source(args.source_manifest)
    args.output.mkdir(parents=True,exist_ok=False)
    env=make_env();fk=TCPKinematics();rows=[]
    try:
        position=setup_scene(env,PROTOCOL['control_seeds'][0])
        controller=TraceController(env,'tcp-control-smoke',args.output,position+np.array(PROTOCOL['offset_base_m']))
        controller.warmup(position);frame_check(controller,fk)
        executor=TCPExecutionCandidate(fk)
        for axis in range(6):
            anchor=controller.frame.base_from_tcp.copy();controller.bind(False)
            chunk=np.zeros((16,7),np.float32);chunk[:,6]=1.
            chunk[:4,axis]=.001 if axis<3 else .005
            result=executor.execute(chunk,controller,anchor)
            if (not result.success or result.executed_steps!=4 or controller.stop_reason
                or result.correction_saturation_steps or result.replan_required):
                raise ValueError('TCP smoke执行失败')
            frame_check(controller,fk)
            rows.append(dict(axis=axis,execution=asdict(result),targets=executor.last_targets,
                actual_base_from_tcp=controller.frame.base_from_tcp.tolist()))
            save_json(args.output/'result.json',dict(status='running',axes=rows))
        save_json(args.output/'result.json',dict(status='completed',axes=rows,actuator_steps=controller.policy_step,
            fk_urdf_sha256=sha256(fk.urdf_path),task_success_claim=False,
            execution_config=EXECUTION_CONFIG,runtime_source_sha256=source))
    finally:env.close()


def evaluate(args):
    source=verify_source(args.source_manifest)
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    trained=json.loads((args.training/'result.json').read_text())
    ident=json.loads((args.training/'identity.json').read_text())
    if trained['status']!='completed' or identity(ident)!=trained['identity_sha256']:
        raise ValueError('共同训练未完成/身份不符')
    training_source=compatible_training_source(args.source_manifest,
        getattr(args,'training_source_manifest',None),ident['source_manifest_sha256'])
    base,upstream=load_runtime(args.checkpoint,args.model_cache);fk=TCPKinematics()
    if (ident['upstream']!=json.loads(json.dumps(upstream)) or ident['urdf_sha256']!=sha256(fk.urdf_path)):
        raise ValueError('运行上游或运动学不同于训练')
    eval_protocol=evaluation_protocol(getattr(args,'compare_baseline',False))
    seeds=eval_protocol['historical_seeds']+eval_protocol['new_development_seeds']
    ledger=[dict(arm=arm,seed=seed,status='not_run',
        split='new-development' if seed in eval_protocol['new_development_seeds'] else 'historical-recheck',
        checkpoint_sha256=trained['results'][arm]['checkpoint_sha256']) for arm in ARMS for seed in seeds]
    def save():save_json(args.output/'result.json',dict(protocol=PROTOCOL,records=ledger,
        evaluation_protocol=eval_protocol,
        execution_config=EXECUTION_CONFIG,runtime_source_sha256=source,
        training_source_sha256=training_source,training_identity_sha256=trained['identity_sha256'],
        elapsed_s=time.monotonic()-started))
    save();env=make_env();paired={}
    try:
        for arm in ARMS:
            policy=build_policy(base.policy,arm).to('cuda')
            restore(policy,args.training/(arm+'.pt'),trained['results'][arm]['checkpoint_sha256'],
                trained['identity_sha256'],arm,training_source);policy.eval()
            for entry in [x for x in ledger if x['arm']==arm]:
                seed=entry['seed'];folder=args.output/f'{arm}-{seed}';folder.mkdir();plans=[]
                position=setup_scene(env,seed)
                controller=TraceController(env,f'tcp-eval-{arm}-{seed}',folder,position+np.array(PROTOCOL['offset_base_m']))
                entry['status']='running';save()
                try:
                    controller.warmup(position);frame_check(controller,fk)
                    state=initial_audit(controller)
                    if seed in paired:verify_initial_pair(paired[seed],state)
                    else:paired[seed]=state
                    entry.update(initial_state=state,initial_distance_m=controller.metric_distance())
                    executor=TCPExecutionCandidate(fk) if arm=='tcp-relative' else RecedingHorizonChunkExecutor(RobotSpec())
                    fallback_pending=False;fallbacks=0
                    while controller.policy_step<PROTOCOL['policy_steps'] and controller.stop_reason is None:
                        online=controller.online();snapshot=controller.bind(True);anchor=controller.frame.base_from_tcp.copy()
                        if snapshot.timestamp_s!=controller.frame.timestamp_s or snapshot.episode_id!=controller.episode:
                            raise ValueError('Memory未绑定本次实际帧')
                        features=relative_features(asdict(snapshot),anchor,PROTOCOL['offset_base_m']) if arm=='tcp-relative' else np.array(snapshot.features,np.float32)
                        processed=base.processor_adapter.encode(online.rgb_external,online.rgb_wrist,online.instruction)
                        inputs=_move_model_inputs(processed.model_inputs,base.device)
                        inputs[MEMORY_INPUT_KEY]=MemoryBatch(torch.tensor(features[None],device='cuda'),
                            torch.tensor([[snapshot.available]],device='cuda'))
                        proprio=torch.tensor(base.proprio_normalizer.normalize(online.physical_proprio)[None],device='cuda')
                        seed_action=sampling_seed(seed,len(plans))
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                            prediction=policy.sample_actions(inputs,proprio,num_steps=10,
                                generator=torch.Generator(device='cuda').manual_seed(seed_action))[0].float().cpu().numpy()
                        adapter=TCPActionSpec() if arm=='tcp-relative' else ActionAdapter(RobotSpec())
                        physical=adapter.denormalize(prediction);physical[:,-1]=1.
                        plan=dict(step_before=controller.policy_step,input_digest=observation_digest(online),
                            memory_available=snapshot.available,features=features.tolist(),snapshot=asdict(snapshot),
                            base_from_tcp_anchor=anchor.tolist(),actual_q=controller.read_state().joint_positions.tolist(),
                            sampling_seed=seed_action,physical_chunk=physical.tolist())
                        plans.append(plan)
                        try:
                            # 共同规则：当前实际状态可观测；第一步监督也由同一参考导出。
                            executor.reset()
                            execution=executor.execute(physical,controller,anchor) if arm=='tcp-relative' else executor.execute(physical,controller)
                        except ValueError as error:
                            plan['rejection']=str(error);entry.update(status='stopped',ending_reason=str(error));break
                        plan['execution']=asdict(execution)
                        if arm=='tcp-relative':plan['ik_targets']=executor.last_targets
                        if execution.correction_saturation_steps or execution.replan_required:
                            entry.update(status='stopped',ending_reason='correction-saturation-or-anomaly');break
                        if fallback_pending and not snapshot.available and execution.executed_steps>0:
                            fallbacks+=1;fallback_pending=False
                        if controller.chunk_stop_requested:
                            executor.reset()
                            fallback_pending |= snapshot.available and not controller.snapshot.available and controller.stop_reason is None
                        if not execution.success or execution.executed_steps==0:
                            entry.update(status='stopped',ending_reason='executor-failure-or-no-progress');break
                    steps=[x for x in controller.control_trace if x['policy_step']>0 and not x['holding']]
                    distances=[entry['initial_distance_m']]+[x['distance_m'] for x in steps]
                    if entry['status']=='running':
                        entry['status']='completed' if controller.policy_step>=PROTOCOL['policy_steps'] and controller.stop_reason is None else 'stopped'
                        entry['ending_reason']=controller.stop_reason
                    entry.update(policy_steps=controller.policy_step,replans=len(plans),final_distance_m=controller.metric_distance(),
                        minimum_distance_m=min(distances),reached=bool(min(distances)<=PROTOCOL['reach_threshold_m']),
                        memory_actions=sum(x['action_used_memory'] for x in steps),
                        occluded_memory_actions=sum(x['action_used_memory'] and x['occluded'] for x in steps),visual_fallbacks=fallbacks)
                except Exception as error:
                    entry.update(status='error',error_type=type(error).__name__,error=str(error));raise
                finally:
                    save_json(folder/'plans.json',plans);save()
                print(json.dumps({k:v for k,v in entry.items() if k!='initial_state'}),flush=True)
            del policy;torch.cuda.empty_cache()
    finally:env.close()
