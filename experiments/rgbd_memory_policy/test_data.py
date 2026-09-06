"""只验证新增数据边界，不把合成样本作为训练或机器人结果。"""
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from experiments.front_rgbd_memory.memory import candidate_config
from experiments.memory_conditioning.conditioning import MEMORY_SCHEMA
from experiments.rgbd_memory_policy.data import validate_sequence,validate_frame_binding
from experiments.memory_reobserve.runtime import observation_digest
from robot_vla.runtime.policy_runtime import OnlineObservation
from experiments.rgbd_memory_policy.protocol import PROTOCOL,identity,occluded
from robot_vla.adapters import ActionAdapter
from robot_vla.contracts import RobotSpec


def example():
    n=PROTOCOL['teacher_steps'];timestamps=np.arange(n)*.05+.45
    initial=np.array([0.,0.,0.,-1.,0.,1.,0.],np.float32)
    targets=np.tile(initial,(n,1));targets[:,0]+=np.arange(1,n+1)*.001
    previous=np.concatenate([initial[None],targets[:-1]])
    labels=np.concatenate([targets-previous,np.ones((n,1))],axis=1).astype(np.float32)
    features=np.tile(np.array([.4,0,.12,.01,0,0,.01,0,.01,0,.5,1.],np.float32),(n,1))
    data=dict(normalized_action=ActionAdapter(RobotSpec()).normalize(labels,strict=True),
        physical_proprio=np.zeros((n,15),np.float32),commanded_joint_target_rad=targets,
        previous_command_q_rad=previous,initial_previous_command_q_rad=initial,
        memory_features=features,memory_available=np.ones(n,bool),timestamp_s=timestamps,
        anchors=np.arange(0,n-PROTOCOL['horizon']+1,PROTOCOL['anchor_stride']))
    snapshots=[dict(schema=MEMORY_SCHEMA,episode_id='synthetic',timestamp_s=float(t),
        last_observed_timestamp_s=float(t),features=features[i].tolist(),available=True,reasons=[],
        source_model_identity=PROTOCOL['provider'],source_camera='base_camera') for i,t in enumerate(timestamps)]
    meta=dict(snapshots=snapshots,student_uses_gt=False,protocol_sha256=identity(PROTOCOL),
        memory_config=asdict(candidate_config()),episode_id='synthetic')
    return data,meta


def test_executed_target_reference_and_current_snapshot():
    validate_sequence(*example())


@pytest.mark.parametrize('case',['future','cross_episode','label_reference','nonzero_unavailable','source','late_time'])
def test_reject_corrupted_training_boundary(case):
    data,meta=example();data=deepcopy(data);meta=deepcopy(meta)
    if case=='future':meta['snapshots'][0]['last_observed_timestamp_s']+=.05
    elif case=='cross_episode':meta['snapshots'][0]['episode_id']='other'
    elif case=='label_reference':data['previous_command_q_rad'][0,0]+=.02
    elif case=='nonzero_unavailable':
        data['memory_available'][0]=False;meta['snapshots'][0].update(available=False,reasons=['missing'])
    elif case=='source':meta['snapshots'][0]['source_model_identity']='GT-or-other-provider'
    else:meta['snapshots'][0]['timestamp_s']+=.05
    with pytest.raises(ValueError):validate_sequence(data,meta)


def test_image_and_proprio_are_bound_to_actual_input_tick():
    data,meta=example();n=PROTOCOL['teacher_steps']
    data['rgb_external']=np.zeros((n,4,4,3),np.uint8);data['rgb_wrist']=data['rgb_external'].copy()
    rows=[dict(timestamp_s=s['timestamp_s'],snapshot=s,input_digest=observation_digest(
        OnlineObservation(data['rgb_external'][i],data['rgb_wrist'][i],data['physical_proprio'][i],'test')))
        for i,s in enumerate(meta['snapshots'])]
    validate_frame_binding(data,meta,rows,'test')
    data['rgb_external'][0,0,0,0]=1
    with pytest.raises(ValueError,match='RGB/proprio'):validate_frame_binding(data,meta,rows,'test')


def test_occlusion_schedule_allows_real_expiry_and_new_observations():
    assert not occluded(15) and occluded(16) and occluded(75) and not occluded(76)
    assert (PROTOCOL['occlusion_end']-PROTOCOL['occlusion_start'])/20>candidate_config().max_unobserved_age_s
    assert set(PROTOCOL['train_seeds']).isdisjoint(PROTOCOL['development_seeds'])
    assert set(PROTOCOL['seeds']).isdisjoint(PROTOCOL['rollout_seeds'])
