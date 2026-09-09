"""完整分母、隔离split与有界续训恢复的针对性回归。"""
import json
from datetime import datetime,timedelta,timezone
import numpy as np
import pytest
import torch
from experiments.align_precision_vision.test_labels import dataset
from experiments.align_precision_vision.train import train,digest
from experiments.align_precision_vision.route_evaluate import load_frames,evaluate_frames,compare_routes
from experiments.align_precision_vision.train_continuation import train_continuation,selection_key,plateau_step,schedule,validate_parent


def metrics(accuracy,coverage,p90):
    return dict(accuracy_coverage_8mm_5deg=accuracy,coverage=coverage,position_error_mm={'p90':p90})


def test_fixed_selection_and_zero_coverage_plateau():
    empty=metrics(0,0,None);bad=metrics(0,.5,50);good=metrics(.1,.2,8)
    assert selection_key(good)>selection_key(bad)>selection_key(empty)
    assert plateau_step(empty,empty)
    assert not plateau_step(empty,good)
    assert not plateau_step(metrics(.1,.2,10),metrics(.1,.2,9))
    assert plateau_step(metrics(.1,.2,10),metrics(.105,.2,9.9))


def test_schedule_continues_the_parent_permutation_stream():
    whole=schedule(42,7,0,30)
    assert whole[10:]==schedule(42,7,10,20)
    assert sorted(whole[:7])==list(range(7))


def test_routes_preserve_failed_frames_and_null_reprojection(tmp_path):
    manifest,data=dataset(tmp_path);first=data['records'][0];row=dict(first,id='invisible',scene='invisible',file='missing.npz')
    np.savez(tmp_path/row['file'],rgb=np.zeros((101,101,3),np.uint8),pixel_uv=np.full((8,2),np.nan,np.float32),visible=np.zeros(8,bool))
    row['sha256']=digest(tmp_path/row['file']);data['records'].append(row);manifest.write_text(json.dumps(data))
    result=compare_routes(manifest,None,tmp_path/'routes.json',keypoints='gt-diagnostic',device='cpu',
        deadline_utc=(datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat())
    for group in result['groups'].values():assert group['all']['frames']==2
    assert result['groups']['pnp-only']['all']['coverage']==.5
    assert result['groups']['static-hold']['all']['coverage']==.5
    assert result['groups']['multiview']['all']['coverage']==0
    assert result['groups']['multiview']['all']['position_error_mm']['p90'] is None


def test_loading_rejects_duplicate_time_and_split_leakage(tmp_path):
    manifest,data=dataset(tmp_path);row=dict(data['records'][0],id='copy',file='copy.npz')
    (tmp_path/row['file']).write_bytes((tmp_path/'sample.npz').read_bytes());row['sha256']=digest(tmp_path/row['file'])
    data['records'].append(row);manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='重复时间'):load_frames(manifest)
    row['split']='train';manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='跨split'):load_frames(manifest)


def test_training_restores_optimizer_and_saves_reloadable_last(tmp_path):
    manifest,data=dataset(tmp_path);row=dict(data['records'][0],id='train',scene='train',split='train',file='train.npz')
    (tmp_path/row['file']).write_bytes((tmp_path/'sample.npz').read_bytes());row['sha256']=digest(tmp_path/row['file'])
    data['records'].append(row);manifest.write_text(json.dumps(data))
    train(manifest,tmp_path/'parent',steps=2,wall_seconds=60,device='cpu',channels=(4,8))
    parent=tmp_path/'parent/final.pt';before=digest(parent)
    deadline=(datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat()
    result=train_continuation(manifest,parent,tmp_path/'continued',steps=2,wall_seconds=60,
                              eval_interval=1,device='cpu',deadline_utc=deadline)
    assert result['global_step']==4 and result['additional_updates']==2 and result['strict_reload']
    assert result['stop_reason']=='update_limit' and digest(parent)==before
    payload=torch.load(tmp_path/'continued/last.pt',weights_only=True)
    assert payload['step']==4 and payload['completed'] and not payload['accepted']
    assert all(int(state['step'])==4 for state in payload['optimizer']['state'].values())
    assert len(json.loads((tmp_path/'continued/history.json').read_text()))==3
    bad=torch.load(parent,weights_only=True);bad['config']['source_sha256']['vision.py']='changed'
    with pytest.raises(ValueError,match='配方源码'):validate_parent(bad,digest(manifest))
    stopped=train_continuation(manifest,parent,tmp_path/'time-stopped',steps=2,wall_seconds=1e-9,
        eval_interval=1,device='cpu',deadline_utc=deadline)
    assert stopped['stop_reason']=='stage_wall_limit' and stopped['additional_updates']==0
    assert stopped['best_step'] is None and not (tmp_path/'time-stopped/best.pt').exists()
    assert not stopped['development_selected']
    assert (tmp_path/'time-stopped/last.pt').is_file()


def test_expired_deadline_does_not_create_run(tmp_path):
    with pytest.raises(TimeoutError,match='截止'):
        train_continuation('unused','unused',tmp_path/'expired',deadline_utc='2000-01-01T00:00:00+00:00')
    assert not (tmp_path/'expired').exists()


def test_prediction_checks_the_same_evaluation_budget(tmp_path):
    manifest,_=dataset(tmp_path);frames=load_frames(manifest);calls=[]
    def stopped():
        calls.append(1)
        if len(calls)>1:raise TimeoutError('prediction-budget')
    with pytest.raises(TimeoutError,match='prediction-budget'):
        evaluate_frames(frames,None,stop_check=stopped)
