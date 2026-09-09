"""训练入口的来源隔离与保存重载；两步合成输入只作工程 smoke。"""
import json
import numpy as np
import pytest
from experiments.align_precision_vision.train import KeypointTrainingData,train,digest


def manifest(tmp_path):
    file=tmp_path/'train.npz'
    np.savez(file,rgb=np.zeros((16,16,3),np.uint8),pixel_uv=np.full((8,2),8.,np.float32),visible=np.ones(8,bool))
    records=[dict(id='one',scene='train-scene',split='train',file=file.name,sha256=digest(file)),
             dict(id='unread',scene='development-scene',split='development',file='not-opened.npz',sha256='unused')]
    data=dict(schema='align-precision-keypoints/v1',object_model='upright-cube-4cm',pixel_convention='zero-based-pixel-centers',records=records)
    p=tmp_path/'manifest.json';p.write_text(json.dumps(data));return p,data


def test_train_only_synthetic_smoke_and_reload(tmp_path):
    p,_=manifest(tmp_path)
    assert len(KeypointTrainingData(p).rows)==1
    result=train(p,tmp_path/'out',steps=2,wall_seconds=60,device='cpu',channels=(8,16))
    assert result['status']=='completed' and result['strict_reload'] and not result['accepted']


@pytest.mark.parametrize('mutation', ['overlap','duplicate','hash','path','test'])
def test_reject_invalid_data_provenance(tmp_path,mutation):
    p,data=manifest(tmp_path);row=data['records'][0]
    if mutation=='overlap':data['records'][1]['scene']=row['scene']
    elif mutation=='duplicate':data['records'].append(dict(row))
    elif mutation=='hash':row['sha256']='wrong'
    elif mutation=='path':row['file']='../outside.npz'
    elif mutation=='test':row['split']='test'
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError):KeypointTrainingData(p)
