"""固定16个已有development场景的旧/新TCP闭环对照，不修改执行器。"""
from dataclasses import asdict
from pathlib import Path
import argparse
import gc
import json
import signal
import shutil
import time

import numpy as np
import torch

from experiments.g2c_memory_integration.vla import load_runtime, sha256
from experiments.memory_conditioning.conditioning import MEMORY_INPUT_KEY, MemoryBatch
from experiments.memory_reobserve.runtime import observation_digest
from experiments.rgbd_memory_policy.evaluate import initial_audit, verify_initial_pair
from experiments.rgbd_memory_policy.stream import INSTRUCTION, make_env, setup_scene
from experiments.tcp_chunk_diagnostic.run import save
from experiments.tcp_memory_control.data import prepare_examples
from experiments.tcp_memory_control.evaluate import TraceController, frame_check
from experiments.tcp_memory_control.execution import TCPExecutionCandidate
from experiments.tcp_memory_control.geometry import TCPActionSpec, relative_features
from experiments.tcp_memory_control.kinematics import TCPKinematics
from experiments.tcp_memory_control.policy import build_policy
from experiments.tcp_memory_control.protocol import PROTOCOL, sampling_seed, identity
from experiments.tcp_memory_control.train import restore, verify_source
from experiments.tcp_teacher_closed_loop.checkpoint import (
    BEST_SHA, PARENT_SHA, CONFIG_SHA, EVALUATION_SHA, load_best,
)
from experiments.tcp_teacher_closed_loop.metrics import trajectory_metrics, canonical_audit
from robot_vla.runtime.policy_runtime import _move_model_inputs

SEEDS = list(range(1600200,1600216))
ARMS = ('before', 'after')


def inputs_for(base, online, snapshot, anchor):
    features = relative_features(asdict(snapshot), anchor, PROTOCOL['offset_base_m'])
    processed = base.processor_adapter.encode(online.rgb_external, online.rgb_wrist, online.instruction)
    inputs = _move_model_inputs(processed.model_inputs, base.device)
    inputs[MEMORY_INPUT_KEY] = MemoryBatch(torch.tensor(features[None],device='cuda'),
        torch.tensor([[snapshot.available]],device='cuda'))
    proprio = torch.tensor(base.proprio_normalizer.normalize(online.physical_proprio)[None],device='cuda')
    return inputs, proprio, features


@torch.no_grad()
def predict(policy, inputs, proprio, seed):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        return policy.sample_actions(inputs,proprio,num_steps=10,
            generator=torch.Generator(device='cuda').manual_seed(seed))[0].float().cpu().numpy()


def smoke(output):
    """实际相机观察、六轴动作和FK一致性检查；不读取学生权重。"""
    output.mkdir(exist_ok=False)
    env=make_env();fk=TCPKinematics();rows=[]
    try:
        position=setup_scene(env,1600000)
        controller=TraceController(env,'tcp-continuation-smoke',output,position+np.array(PROTOCOL['offset_base_m']))
        controller.warmup(position);frame_check(controller,fk)
        executor=TCPExecutionCandidate(fk)
        for axis in range(6):
            anchor=controller.frame.base_from_tcp.copy();controller.bind(False)
            chunk=np.zeros((16,7),np.float32);chunk[:,6]=1.
            chunk[:4,axis]=.001 if axis<3 else .005
            result=executor.execute(chunk,controller,anchor)
            if (not result.success or result.executed_steps!=4 or controller.stop_reason
                or result.correction_saturation_steps or result.replan_required):
                raise ValueError('六轴控制smoke失败')
            frame_check(controller,fk)
            rows.append(dict(axis=axis,execution=asdict(result)))
        save(output/'result.json',dict(status='completed',axes=rows,policy_steps=controller.policy_step,
            external_shape=list(controller.frame.rgb_external.shape),wrist_shape=list(controller.frame.rgb_wrist.shape)))
    finally:env.close()


def assess_scene(base, policy, fk, env, entry, output, paired):
    folder=output/f'{entry["arm"]}-{entry["seed"]}';folder.mkdir()
    plans=[];seed=entry['seed'];position=setup_scene(env,seed)
    controller=TraceController(env,f'tcp-continuation-{entry["arm"]}-{seed}',folder,
                               position+np.array(PROTOCOL['offset_base_m']))
    try:
        controller.warmup(position);frame_check(controller,fk)
        state=canonical_audit(initial_audit(controller))
        save(folder/'initial-state.json',state)
        if seed in paired:verify_initial_pair(paired[seed],state)
        else:paired[seed]=state
        entry.update(initial_state=state,initial_distance_m=controller.metric_distance())
        executor=TCPExecutionCandidate(fk)
        while controller.policy_step<88 and controller.stop_reason is None:
            online=controller.online();snapshot=controller.bind(True);anchor=controller.frame.base_from_tcp.copy()
            if snapshot.timestamp_s!=controller.frame.timestamp_s or snapshot.episode_id!=controller.episode:
                raise ValueError('Memory与当前实际帧不一致')
            inputs,proprio,features=inputs_for(base,online,snapshot,anchor)
            seed_action=sampling_seed(seed,len(plans))
            prediction=predict(policy,inputs,proprio,seed_action)
            physical=TCPActionSpec().denormalize(prediction);physical[:,-1]=1.
            plan=dict(step_before=controller.policy_step,input_digest=observation_digest(online),
                memory_available=snapshot.available,features=features.tolist(),snapshot=asdict(snapshot),
                base_from_tcp_anchor=anchor.tolist(),actual_q=controller.read_state().joint_positions.tolist(),
                sampling_seed=seed_action,physical_chunk=physical.tolist())
            plans.append(plan)
            try:
                executor.reset();execution=executor.execute(physical,controller,anchor)
            except ValueError as error:
                plan['rejection']=str(error);entry.update(status='stopped',ending_reason=str(error));break
            plan.update(execution=asdict(execution),ik_targets=executor.last_targets)
            if execution.correction_saturation_steps or execution.replan_required:
                entry.update(status='stopped',ending_reason='correction-saturation-or-anomaly');break
            if controller.chunk_stop_requested:executor.reset()
            if not execution.success or execution.executed_steps==0:
                entry.update(status='stopped',ending_reason='executor-failure-or-no-progress');break
        if entry['status']=='running':
            entry.update(status='completed' if controller.policy_step>=88 and controller.stop_reason is None else 'stopped',
                         ending_reason=controller.stop_reason)
        entry.update(trajectory_metrics(controller.control_trace,entry['initial_distance_m'],plans))
    finally:save(folder/'plans.json',plans)


def evaluate(args):
    args.output.mkdir(exist_ok=False)
    started=time.monotonic();ledger=[dict(arm=a,seed=s,status='not_run',stage='first4' if s in SEEDS[:4] else 'remaining12')
                                    for s in SEEDS for a in ARMS]
    result=dict(status='preflight',records=ledger,protocol=dict(seeds=SEEDS,arms=ARMS,
        control_hz=20,policy_steps=88,execute_steps=4,joint_delta_limit_rad=.1,tracking_limit_rad=.05,
        reach_threshold_m=.02,sampling_steps=10,split='previously-used development; posthoc comparison',
        selection='frozen best at continuation step 58880',checkpoint_sha256=dict(before=PARENT_SHA,after=BEST_SHA)))
    def checkpoint():
        result['elapsed_s']=time.monotonic()-started;save(args.output/'result.json',result)
    env=None
    try:
        source=verify_source(args.source_manifest)
        extras=json.loads(args.extra_manifest.read_text())
        if any(sha256(path)!=digest for path,digest in extras.items()):raise ValueError('本轮新增源码身份不符')
        config_path=args.continued/'config.json';reference_path=args.continued/'evaluation-00058880.json'
        if sha256(config_path)!=CONFIG_SHA or sha256(reference_path)!=EVALUATION_SHA:raise ValueError('最佳点评估或配置身份改变')
        config=json.loads(config_path.read_text());prior=json.loads((args.training/'result.json').read_text())
        ident=json.loads((args.training/'identity.json').read_text())
        if prior['status']!='completed' or identity(ident)!=prior['identity_sha256']:raise ValueError('原训练身份无效')
        base,upstream=load_runtime(args.checkpoint,args.model_cache);fk=TCPKinematics()
        if json.loads(json.dumps(upstream))!=ident['upstream'] or sha256(fk.urdf_path)!=ident['urdf_sha256']:
            raise ValueError('上游模型或FK不同于训练')
        policies={a:build_policy(base.policy,'tcp-relative').to('cuda') for a in ARMS}
        restore(policies['before'],args.training/'tcp-relative.pt',PARENT_SHA,prior['identity_sha256'],'tcp-relative',source)
        result['continued_identity']=load_best(policies['after'],args.continued/'best.pt',config,prior['identity_sha256'],source)
        for p in policies.values():p.eval()
        # 使用已存在教师窗口核验加载与输入路径，模型不接收评估GT。
        raw,hashes,denominator=prepare_examples(args.data,fk)
        if hashes!=ident['data_sha256'] or denominator!=ident['denominator']:raise ValueError('教师数据身份改变')
        references=json.loads(reference_path.read_text())['records'];parity=[]
        old={(x['seed'],x['anchor']):x for x in prior['results']['tcp-relative']['development']}
        for x in raw['development'][:2]:
            encoded=base.processor_adapter.encode(x['rgb_external'],x['rgb_wrist'],INSTRUCTION)
            inputs=_move_model_inputs(encoded.model_inputs,base.device)
            inputs[MEMORY_INPUT_KEY]=MemoryBatch(torch.tensor(x['tcp_features'][None],device='cuda'),
                torch.tensor([[x['snapshot']['available']]],device='cuda'))
            proprio=torch.tensor(base.proprio_normalizer.normalize(x['physical_proprio'])[None],device='cuda')
            before=predict(policies['before'],inputs,proprio,sampling_seed(x['seed'],x['anchor']))
            after=predict(policies['after'],inputs,proprio,sampling_seed(x['seed'],x['anchor']))
            expected=next(r['predicted_physical'] for r in references if r['split']=='development' and r['seed']==x['seed'] and r['anchor']==x['anchor'])
            difference=float(np.max(np.abs(TCPActionSpec().denormalize(after)-expected)))
            old_difference=abs(float(np.abs(before[:4]-x['tcp_action'][:4]).mean())-old[(x['seed'],x['anchor'])]['sampled_first4_mae_normalized'])
            if difference>1e-6 or old_difference>1e-5:raise ValueError('checkpoint加载后预测不一致')
            parity.append(dict(seed=x['seed'],anchor=x['anchor'],new_max_abs_difference=difference,old_mae_difference=old_difference))
        result.update(status='running',parity=parity,original_source_sha256=source,extra_source_sha256=sha256(args.extra_manifest))
        del raw;gc.collect();paired={}
        if args.reuse:
            previous=json.loads((args.reuse/'result.json').read_text())
            if identity(previous['protocol'])!=identity(result['protocol']) or previous['original_source_sha256']!=source:
                raise ValueError('不能恢复不同协议或上游来源的场景')
            consumed=[x for x in previous['records'] if x['status']!='not_run']
            if len(consumed)!=1 or (consumed[0]['arm'],consumed[0]['seed'],consumed[0].get('error'))!=('before',1600200,"'tracking_error_rad'"):
                raise ValueError('仅恢复已知的首场景汇总字段错误')
            entry=ledger[0];entry.update(consumed[0])
            old_folder=args.reuse/'before-1600200'
            trace=[json.loads(line) for line in (old_folder/'control-trace.jsonl').read_text().splitlines()]
            plans=json.loads((old_folder/'plans.json').read_text())
            if not plans or plans[-1].get('rejection')!=entry['ending_reason']:
                raise ValueError('原始计划不足以确定停止原因')
            entry.update(trajectory_metrics(trace,entry['initial_distance_m'],plans),status='stopped',
                recovered_from_aggregation_error=True)
            entry.pop('error');entry.pop('error_type')
            shutil.copytree(old_folder,args.output/'before-1600200')
            paired[1600200]=entry['initial_state']
            result['reused_evidence_sha256']={name:sha256(args.reuse/name) for name in
                ['result.json','before-1600200/control-trace.jsonl','before-1600200/plans.json']}
        checkpoint();env=make_env()
        for entry in ledger:
            if entry['status'] in ('completed','stopped'):continue
            entry['status']='running';checkpoint()
            try:assess_scene(base,policies[entry['arm']],fk,env,entry,args.output,paired)
            except BaseException as error:
                entry.update(status='error',error_type=type(error).__name__,error=str(error));raise
            checkpoint()
            print(json.dumps({k:v for k,v in entry.items() if k not in ('initial_state','distance_by_policy_step')}),flush=True)
        result['status']='completed'
    except BaseException as error:
        result.update(status='incomplete',error_type=type(error).__name__,error=str(error));raise
    finally:
        checkpoint()
        if env is not None:env.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['smoke','evaluate'])
    for n in ('output','source-manifest','extra-manifest','training','continued','checkpoint','model-cache','data'):
        parser.add_argument('--'+n,type=Path,required=n=='output')
    parser.add_argument('--reuse',type=Path)
    args=parser.parse_args()
    def stop(*_):raise InterruptedError('收到停止请求；保留已完成场景，不重采样')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.stage=='smoke':smoke(args.output)
    else:evaluate(args)


if __name__=='__main__':main()
