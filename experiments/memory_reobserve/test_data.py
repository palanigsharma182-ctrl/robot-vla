"""数据入口的反例：模块身份、未来标签与不可用 Memory 掩码。"""
from dataclasses import asdict
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from experiments.memory_reobserve.data import new_examples
from experiments.memory_reobserve.protocol import PROTOCOL, Acquisition
from robot_vla.precision.active_front_memory_provider import build_stage2_object_memory_config
from experiments.memory_conditioning.conditioning import MEMORY_SCHEMA
from robot_vla.adapters import ActionAdapter
from robot_vla.contracts import RobotSpec


def fixture(root):
    seed=PROTOCOL['train_seeds'][0]; folder=root/str(seed);folder.mkdir()
    n=20; q=np.zeros((n,7),np.float32);q[:,0]=np.arange(1,n+1)*.001
    previous=np.concatenate((np.zeros((1,7),np.float32),q[:-1]))
    labels=ActionAdapter(RobotSpec()).normalize(np.c_[q-previous,np.ones(n)].astype(np.float32),strict=True)
    episode=f'g2c-main-engineering-{seed}'
    records=[dict(snapshot=dict(episode_id=episode,schema=MEMORY_SCHEMA,timestamp_s=(10+i)/20,
        available=False,features=[0.]*12,reasons=['memory_uninitialized'])) for i in range(n)]
    values=dict(normalized_action=labels,commanded_joint_target_rad=q,
        initial_previous_command_q_rad=np.zeros(7,np.float32),timestamp_s=np.arange(10,10+n)/20,
        anchors=np.array([0,4]),memory_features=np.zeros((n,12),np.float32),memory_available=np.zeros(n,bool),
        physical_proprio=np.zeros((n,15),np.float32),rgb_external=np.zeros((n,8,8,3),np.uint8),
        rgb_wrist=np.zeros((n,8,8,3),np.uint8))
    np.savez(folder/'sequence.npz',**values)
    (folder/'sequence.json').write_text(json.dumps(dict(schema='memory-reobserve-sequence/v1',
        seed=seed,episode_id=episode,
        model_inputs_use_privileged_pose=False,memory_config=asdict(build_stage2_object_memory_config()),records=records)))
    (folder/'result.json').write_text('{}')
    (root/'protocol.json').write_text(json.dumps(PROTOCOL))
    rows=[dict(seed=s,status='completed',result={'seed':s,'capture':{'samples':2},'memory_write_count':0}) if s==seed
          else dict(seed=s,status='completed',result={'capture':{'samples':0}}) for s in PROTOCOL['seeds']]
    (root/'collection.json').write_text(json.dumps(dict(records=rows)))
    return folder,values


def test_cli_uses_shared_acquisition_identity():
    path=Path(__file__).with_name('collect.py')
    loaded=runpy.run_path(str(path),run_name='isolated_cli_import')
    assert loaded['Acquisition'] is Acquisition
    assert isinstance(loaded['Acquisition'](PROTOCOL['train_seeds'][0]),Acquisition)


def test_masked_sequences_keep_real_action_labels(tmp_path):
    _,values=fixture(tmp_path)
    rows,_,_=new_examples(tmp_path)
    assert len(rows['train'])==2 and not rows['development']
    assert not rows['train'][1]['available']
    np.testing.assert_allclose(rows['train'][1]['action'],values['normalized_action'][4:20])


@pytest.mark.parametrize('corruption',['label','time','mask','denominator'])
def test_corrupt_data_fails_before_training(tmp_path,corruption):
    folder,values=fixture(tmp_path)
    if corruption=='label':values['normalized_action'][0,0]+=.5
    if corruption=='time':values['timestamp_s'][1]+=.02
    if corruption=='mask':values['memory_features'][0,0]=.1
    if corruption=='denominator':
        p=tmp_path/'collection.json';r=json.loads(p.read_text());r['records'].pop();p.write_text(json.dumps(r))
    np.savez(folder/'sequence.npz',**values)
    with pytest.raises(ValueError):new_examples(tmp_path)


def test_stale_reason_requires_actual_expired_measurement(tmp_path):
    folder,_=fixture(tmp_path)
    p=folder/'sequence.json';meta=json.loads(p.read_text())
    s=meta['records'][0]['snapshot'];s['reasons']=['memory_stale'];s['last_observed_timestamp_s']=s['timestamp_s']-.1
    p.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='实际年龄'):new_examples(tmp_path)


def test_sequence_cannot_be_relabelled_as_another_seed(tmp_path):
    folder,_=fixture(tmp_path)
    p=folder/'sequence.json';meta=json.loads(p.read_text());meta['seed']+=1;p.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='seed/Episode'):new_examples(tmp_path)
