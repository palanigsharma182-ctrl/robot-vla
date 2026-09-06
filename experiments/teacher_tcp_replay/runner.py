"""教师TCP标签原样重放：不做模型预测，不按在线误差改写动作。"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
from scipy.spatial.transform import Rotation
from experiments.g2c_memory_integration.vla import sha256
from experiments.memory_reobserve.runtime import observation_digest
from experiments.rgbd_memory_policy.protocol import PROTOCOL as DATA_PROTOCOL
from experiments.rgbd_memory_policy.stream import make_env, setup_scene
from experiments.rgbd_memory_policy.train import save_json
from experiments.tcp_memory_control.evaluate import TraceController, frame_check
from experiments.tcp_memory_control.execution import TCPExecutionCandidate
from experiments.tcp_memory_control.geometry import TCPActionSpec, pose_delta
from experiments.tcp_memory_control.kinematics import TCPKinematics
from experiments.tcp_memory_control.train import verify_source
from experiments.tcp_memory_control.protocol import identity as semantic_identity

APPROVED_TRAINING_IDENTITY = '7c458d4ac1607ed372dc5dd331c2b51c9b066fdc5b0cbb7c82da89621279d817'

PROTOCOL = dict(schema='recorded-teacher-tcp-replay/v1', seeds=list(DATA_PROTOCOL['seeds']),
    policy_steps=88, source_steps=96, horizon=16, execute_steps=4, control_hz=20,
    command_delta_limit_rad=.1, tracking_error_limit_rad=.05, reach_threshold_m=.02,
    labels='same actual-first / commanded-successive FK labels as TCP training, float32 normalized',
    replay='immutable labels indexed by actual executed step, expressed at live TCP anchor without re-aiming',
    tail='only unexecuted future slots beyond source length are neutral padding',
    model_inference=False, memory_conditioned_prediction=False,
    scope='original 24 train/development scenes; diagnostic replay, not independent generalization')


def action_digest(value):
    return hashlib.sha256(np.asarray(value, dtype='<f8').tobytes()).hexdigest()


def require_replay_sources(manifest):
    required={'experiments/teacher_tcp_replay/runner.py',
              'experiments/tcp_memory_control/execution.py'}
    if not required.issubset(manifest):
        raise ValueError('源码快照没有教师入口或0.1执行模块')


def validate_training_identity(ident, records):
    if semantic_identity(ident)!=APPROVED_TRAINING_IDENTITY:
        raise ValueError('并非本轮批准的原训练身份')
    if records!=ident['denominator']:
        raise ValueError('collection分母或历史指标与冻结训练身份不符')
    if ([x['seed'] for x in records]!=PROTOCOL['seeds']
        or any(x['status']!='completed' for x in records)
        or any(x['split']!=('train' if x['seed'] in DATA_PROTOCOL['train_seeds'] else 'development') for x in records)):
        raise ValueError('必须保留原24条分母和16train/8development划分')


class TeacherActions:
    """一次性从已记录关节目标生成标签；之后不接受在线状态或GT目标。"""
    def __init__(self, data, fk):
        self.data = data
        self.actual = np.stack([fk.pose_base(q[:7]) for q in data['physical_proprio']])
        self.commanded = np.stack([fk.pose_base(q) for q in data['commanded_joint_target_rad']])
        self.previous = np.stack([fk.pose_base(q) for q in data['previous_command_q_rad']])
        self.spec = TCPActionSpec()

    def normalized_chunk(self, index):
        n = len(self.actual)
        if not 0 <= index < PROTOCOL['policy_steps'] or n-index < 4:
            raise ValueError('教师实际执行前缀不完整')
        real = min(self.spec.horizon, n-index)
        physical = np.zeros((self.spec.horizon,7)); physical[:,6] = 1.
        anchor = self.actual[index]
        for offset in range(real):
            j = index+offset
            previous = anchor if offset == 0 else self.previous[j]
            physical[offset,:6] = pose_delta(previous,self.commanded[j],anchor)
        return self.spec.normalize(physical), real

    def frozen_chunks(self):
        values = []
        for index in range(PROTOCOL['policy_steps']):
            normalized, real = self.normalized_chunk(index)
            physical = self.spec.denormalize(normalized)
            physical.setflags(write=False)
            values.append((physical,real,action_digest(physical)))
        return values


def load_teachers(root, training_identity, fk):
    ident = json.loads(Path(training_identity).read_text())
    if sha256(fk.urdf_path) != ident['urdf_sha256']:
        raise ValueError('当前URDF不属于教师标签训练身份')
    root=Path(root)
    records=json.loads((root/'collection.json').read_text())['records']
    validate_training_identity(ident,records)
    teachers={}; hashes={}
    names=['physical_proprio','commanded_joint_target_rad','previous_command_q_rad','timestamp_s']
    for entry in records:
        seed=entry['seed'];folder=root/str(seed)
        expected=ident['data_sha256'][str(seed)]
        paths={'sequence':folder/'sequence.npz','metadata':folder/'sequence.json','observations':folder/'observations.jsonl'}
        actual={k:sha256(p) for k,p in paths.items()}
        if actual!=expected:raise ValueError('教师数据与已核验训练身份不一致')
        hashes[str(seed)]=actual
        with np.load(paths['sequence'],allow_pickle=False) as archive:
            data={name:archive[name].copy() for name in names}
        if (data['physical_proprio'].shape!=(96,15)
            or data['commanded_joint_target_rad'].shape!=(96,7)
            or data['previous_command_q_rad'].shape!=(96,7)
            or not all(np.isfinite(x).all() for x in data.values())
            or not np.allclose(np.diff(data['timestamp_s']),.05,atol=1e-8,rtol=0)
            or not np.array_equal(data['previous_command_q_rad'][1:],data['commanded_joint_target_rad'][:-1])):
            raise ValueError('教师时间/状态/command参考不完整')
        observations=[json.loads(line) for line in paths['observations'].read_text().splitlines()]
        initial=[x for x in observations if abs(x['timestamp_s']-data['timestamp_s'][0])<1e-9]
        if len(initial)!=1:raise ValueError('教师初始观测时间无法唯一匹配')
        teachers[seed]=(TeacherActions(data,fk),initial[0]['input_digest'],entry)
    return teachers, hashes


def run(args):
    source=verify_source(args.source_manifest)
    manifest=json.loads(args.source_manifest.read_text())
    require_replay_sources(manifest)
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();fk=TCPKinematics()
    teachers,hashes=load_teachers(args.data,args.training_identity,fk)
    chunks={seed:teacher.frozen_chunks() for seed,(teacher,_,_) in teachers.items()}
    identity=dict(protocol=PROTOCOL,source_sha256=source,teacher_hashes=hashes,
        training_identity_sha256=APPROVED_TRAINING_IDENTITY,
        training_identity_file_sha256=sha256(args.training_identity),urdf_sha256=sha256(fk.urdf_path))
    save_json(args.output/'identity.json',identity)
    label_hashes={}
    for seed,values in chunks.items():
        np.savez_compressed(args.output/f'teacher-actions-{seed}.npz',
            physical=np.stack([x[0] for x in values]),valid_lengths=[x[1] for x in values])
        label_hashes[str(seed)]=[x[2] for x in values]
    save_json(args.output/'teacher-action-hashes.json',label_hashes)
    ledger=[dict(seed=seed,split=teachers[seed][2]['split'],status='not_run') for seed in PROTOCOL['seeds']]
    def save():save_json(args.output/'result.json',dict(protocol=PROTOCOL,records=ledger,elapsed_s=time.monotonic()-started))
    save()
    if args.stage=='preflight':return
    env=make_env()
    try:
        for entry in ledger:
            seed=entry['seed'];teacher,initial_digest,original=teachers[seed]
            folder=args.output/str(seed);folder.mkdir();plans=[];entry['status']='running';save()
            try:
                position=setup_scene(env,seed);target=position+np.array([0.,0.,.08])
                controller=TraceController(env,f'teacher-tcp-replay-{seed}',folder,target)
                controller.warmup(position);frame_check(controller,fk)
                if (not np.allclose(controller.frame.physical_proprio,teacher.data['physical_proprio'][0],atol=1e-6,rtol=0)
                    or observation_digest(controller.online())!=initial_digest):
                    raise ValueError('重放初态与原教师观测不匹配')
                world_from_base=env.unwrapped.agent.robot.pose.to_transformation_matrix()[0].cpu().numpy()
                original_world=np.stack([(world_from_base@pose)[:3,3] for pose in teacher.actual[:89]])
                original_distances=np.linalg.norm(original_world-target,axis=1)
                entry.update(initial_digest=initial_digest,initial_distance_m=controller.metric_distance(),
                    original_teacher_prefix_reached=bool(original_distances.min()<=.02),
                    original_teacher_prefix_min_distance_m=float(original_distances.min()),
                    original_teacher_prefix_final_distance_m=float(original_distances[-1]),
                    original_teacher_96_final_distance_m=original['final_teacher_distance_m'])
                executor=TCPExecutionCandidate(fk)
                while controller.policy_step<88 and controller.stop_reason is None:
                    index=controller.policy_step;physical,real,digest=chunks[seed][index]
                    snapshot=controller.bind(True);anchor=controller.frame.base_from_tcp.copy()
                    if action_digest(physical)!=digest:raise ValueError('教师动作被改写')
                    plan=dict(source_index=index,source_valid_length=real,teacher_action_sha256=digest,
                        physical_chunk=physical.tolist(),recorded_anchor=teacher.actual[index].tolist(),live_anchor=anchor.tolist(),
                        anchor_translation_drift_m=float(np.linalg.norm(anchor[:3,3]-teacher.actual[index,:3,3])),
                        anchor_rotation_drift_rad=float(np.linalg.norm(Rotation.from_matrix(anchor[:3,:3]@teacher.actual[index,:3,:3].T).as_rotvec())),
                        memory_available=snapshot.available,memory_conditioned_prediction=False)
                    plans.append(plan)
                    try:result=executor.execute(physical,controller,anchor)
                    except ValueError as error:
                        plan['rejection']=str(error);entry.update(status='stopped',ending_reason=str(error));break
                    if action_digest(physical)!=digest:raise ValueError('执行器改写教师动作')
                    plan.update(execution=asdict(result),ik_targets=executor.last_targets)
                    if result.correction_saturation_steps or result.replan_required:
                        entry.update(status='stopped',ending_reason='correction-saturation-or-anomaly');break
                    if not result.success or result.executed_steps==0:
                        entry.update(status='stopped',ending_reason='executor-failure-or-no-progress');break
                steps=[x for x in controller.control_trace if x['policy_step']>0 and not x['holding']]
                distances=[entry['initial_distance_m']]+[x['distance_m'] for x in steps]
                if entry['status']=='running':
                    entry.update(status='completed' if controller.policy_step==88 and controller.stop_reason is None else 'stopped',ending_reason=controller.stop_reason)
                entry.update(policy_steps=controller.policy_step,reached=bool(min(distances)<=.02),
                    minimum_distance_m=min(distances),final_distance_m=distances[-1],
                    peak_tracking_error_rad=max((max(abs(v) for v in x['tracking_error_rad']) for x in steps),default=0.),
                    peak_measured_joint_velocity_rad_s=max((max(abs(v) for v in x['joint_velocity_rad_s']) for x in steps),default=0.))
            except Exception as error:
                entry.update(status='error',error_type=type(error).__name__,error=str(error));raise
            finally:save_json(folder/'plans.json',plans);save()
            print(json.dumps(entry),flush=True)
    finally:env.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','replay'])
    for name in ['data','training-identity','source-manifest','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    def stop(*_):raise TimeoutError('累计资源预算触发停止')
    signal.signal(signal.SIGTERM,stop);run(args)


if __name__=='__main__':main()
