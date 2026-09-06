"""新场景实际重规划与执行：visual / memory / 同权重屏蔽Memory。"""
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.g2c_memory_integration.vla import load_runtime,sha256
from experiments.memory_conditioning.conditioning import MEMORY_INPUT_KEY, MemoryBatch, MemoryConditionedPolicy
from experiments.memory_reobserve.runtime import MemoryConditionedRuntime
from experiments.rgbd_memory_policy.protocol import PROTOCOL
from experiments.rgbd_memory_policy.stream import make_env,setup_scene,RGBDController
from experiments.front_rgbd_memory.runner import array
from experiments.rgbd_memory_policy.train import restore,save_json
from robot_vla.execution.chunk_executor import RecedingHorizonChunkExecutor
from robot_vla.runtime.policy_runtime import _move_model_inputs


def verify_initial_pair(reference,current):
    if reference!=current:
        raise ValueError('三条件实际初态/首帧输入/未屏蔽Memory未配对')


def initial_audit(controller):
    snapshot=asdict(controller.snapshot);snapshot.pop('episode_id')
    return dict(input_digest=controller.rows[-1]['input_digest'],snapshot=snapshot,
        q=controller.frame.physical_proprio[:7].tolist(),tcp=controller.frame.base_from_tcp.tolist(),
        object_world=array(controller.env.unwrapped.cube.pose.to_transformation_matrix())[0].tolist())


def evaluate(checkpoint,model_cache,training,output,source_manifest):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(Path(source_manifest).read_text());manifest_sha=sha256(source_manifest)
    if any(sha256(path)!=digest for path,digest in manifest.items()):raise ValueError('评估实际源码与指定manifest不一致')
    training=Path(training);trained=json.loads((training/'result.json').read_text())
    if trained['status']!='completed':raise ValueError('两臂未完成，禁止比较')
    base,upstream_identity=load_runtime(Path(checkpoint),Path(model_cache))
    policy=MemoryConditionedPolicy(base.policy.context_encoder,base.policy.expert,base.policy.adapter).to('cuda')
    policy.context_encoder.requires_grad_(False);policy.adapter.requires_grad_(False)
    ledger=[dict(arm=arm,seed=seed,status='not_run') for arm in PROTOCOL['evaluation_arms'] for seed in PROTOCOL['rollout_seeds']]
    started=time.monotonic();paired_initials={}
    def save():save_json(output/'result.json',dict(protocol=PROTOCOL,records=ledger,elapsed_s=time.monotonic()-started))
    save();env=make_env()
    try:
        for entry in ledger:
            arm,seed=entry['arm'],entry['seed'];weights='visual' if arm=='visual' else 'memory'
            restore(policy,training/(weights+'.pt'),trained['results'][weights]['checkpoint_sha256'],
                trained['training_identity_sha256'],weights,base.proprio_normalizer,upstream_identity,manifest_sha)
            folder=output/f'{arm}-{seed}';folder.mkdir();entry['status']='running';save()
            controller=None
            try:
                position=setup_scene(env,seed)
                episode=f'rgbd-policy-eval-{arm}-{seed}'
                runtime=MemoryConditionedRuntime(policy,base.processor_adapter,base.proprio_normalizer,
                    base.spec,base.device,replace(base.config,sampling_seed=seed))
                runtime.reset_memory_episode(episode)
                controller=RGBDController(env,episode,folder);controller.warmup(position)
                initial_state=initial_audit(controller)
                if seed in paired_initials:
                    verify_initial_pair(paired_initials[seed],initial_state);entry['initial_pair_status']='matched'
                else:
                    paired_initials[seed]=initial_state;entry['initial_pair_status']='reference'
                entry['initial_state']=initial_state
                executor=RecedingHorizonChunkExecutor(base.spec)
                # GT只用于独立距离评估，不传入runtime或动作生成。
                target=position+np.array([0,0,PROTOCOL['approach_offset_m']])
                def distance():return float(np.linalg.norm(env.unwrapped.agent.tcp_pose.p.cpu().numpy()[0]-target))
                initial_distance=distance();distances=[];plans=[];counterfactual=None;fallbacks=0;reacquired=False
                invalidation_interrupts=0;pending_fallback=False
                had_memory=False;lost_memory=False;masked_after_loss=0
                while controller.policy_step<PROTOCOL['policy_steps']:
                    if controller.episode_done or controller.closed or not controller.rows[-1]['tracking_valid']:break
                    online=controller.online();snap=controller.bind(arm=='memory')
                    if snap.available:
                        reacquired |= lost_memory;had_memory=True
                    elif had_memory:
                        lost_memory=True;masked_after_loss+=1
                    runtime._bind_snapshot(online,snap,frame_timestamp_s=controller.frame.timestamp_s)
                    chunk=runtime.infer_action_chunk(online)
                    if snap.available and counterfactual is None:
                        processed=runtime.processor_adapter.encode(online.rgb_external,online.rgb_wrist,online.instruction)
                        inputs=_move_model_inputs(processed.model_inputs,runtime.device)
                        inputs[MEMORY_INPUT_KEY]=MemoryBatch(torch.zeros((1,12),device=runtime.device),
                            torch.zeros((1,1),dtype=torch.bool,device=runtime.device))
                        proprio=torch.tensor(runtime.proprio_normalizer.normalize(online.physical_proprio)[None],device=runtime.device)
                        with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                            masked=policy.sample_actions(inputs,proprio,num_steps=runtime.config.num_flow_steps,
                                generator=torch.Generator(device=runtime.device).manual_seed(chunk.sampling.seed))
                        counterfactual=float(np.linalg.norm(chunk.normalized_action[:,:7]-masked[0,:,:7].float().cpu().numpy()))
                    physical=chunk.physical_action.copy();physical[:,-1]=1.
                    step_before=controller.policy_step
                    execution=executor.execute(physical,controller)
                    plans.append(dict(step_before=step_before,memory_available=snap.available,
                        snapshot=asdict(snap),sampling=asdict(chunk.sampling),execution=asdict(execution)))
                    distances.append(distance())
                    if pending_fallback and not snap.available and execution.executed_steps>0:
                        fallbacks+=1;pending_fallback=False
                    if controller.chunk_stop_requested:
                        # 没有ensemble或RTC缓存；清空唯一的command reference，下一次用新帧重新推理。
                        executor.reset()
                        invalidated=snap.available and not controller.snapshot.available
                        invalidation_interrupts+=int(invalidated)
                        if invalidated and controller.stop_reason is None:pending_fallback=True
                    if not execution.success:break
                    if execution.executed_steps==0:raise RuntimeError('零步执行无时间进展，停止而非重复旧快照')
                policy_rows=[r for r in controller.rows if r['policy_step']>0]
                ending_reason=controller.stop_reason or ('executor-failure' if any(not p['execution']['success'] for p in plans) else None)
                safely_completed=controller.policy_step>=PROTOCOL['policy_steps'] and ending_reason is None
                entry.update(status='completed' if safely_completed else 'stopped',ending_reason=ending_reason,
                    policy_steps=controller.policy_step,replans=len(plans),initial_distance_m=initial_distance,
                    final_distance_m=distance(),minimum_distance_m=min(distances,default=initial_distance),
                    reached=bool(min(distances,default=initial_distance)<=PROTOCOL['reach_threshold_m']),
                    memory_replans=sum(p['memory_available'] for p in plans),
                    memory_actions=sum(r['action_used_memory'] for r in policy_rows),
                    occluded_memory_actions=sum(r['action_used_memory'] and r['occluded'] for r in policy_rows),
                    commits=sum(r['memory']['accepted'] for r in controller.rows),
                    memory_invalidation_interrupts=invalidation_interrupts,visual_fallbacks=fallbacks,reacquired_after_loss=reacquired,
                    masked_replans_after_loss=masked_after_loss,
                    counterfactual_joint_chunk_l2=counterfactual,
                    execution_failures=sum(not p['execution']['success'] for p in plans))
                save_json(folder/'plans.json',plans);save_json(folder/'runtime-reads.json',runtime.memory_reads)
            except Exception as e:
                entry.update(status='error',error_type=type(e).__name__,error=str(e));raise
            finally:save()
            print(json.dumps(entry),flush=True)
    finally:env.close()
    return ledger
