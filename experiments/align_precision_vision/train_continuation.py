"""现有RGB定位器有界续训，定期development评估与原子恢复点。"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from experiments.align_precision_vision.train import KeypointTrainingData,digest
from experiments.align_precision_vision.vision import AlignLocalizer,localization_loss
from experiments.align_precision_vision.route_evaluate import load_frames,evaluate_frames,write_json


def selection_key(metrics):
    """空有效集合的p90不视为0；所有best均只属于development选择。"""
    tail=metrics['position_error_mm']['p90']
    return (metrics['accuracy_coverage_8mm_5deg'],metrics['coverage'],-float('inf') if tail is None else -tail)


def plateau_step(previous,current):
    gain=current['accuracy_coverage_8mm_5deg']-previous['accuracy_coverage_8mm_5deg']
    before=previous['position_error_mm']['p90'];after=current['position_error_mm']['p90']
    if before is None:tail_gain=1. if after is not None else 0.
    elif after is None:tail_gain=-float('inf')
    else:tail_gain=(before-after)/max(abs(before),1e-12)
    return gain<.01 and tail_gain<.02


def schedule(seed,rows,start,steps):
    """重建原train.py全排列采样前缀，接着原全局更新号继续。"""
    rng=np.random.default_rng(seed);values=[]
    while len(values)<start+steps:values.extend(rng.permutation(rows).tolist())
    return values[start:start+steps]


def save_checkpoint(path,model,optimizer,config,step,losses,history,*,role,status):
    payload=dict(format='align-precision-localizer/v1',model=model.state_dict(),optimizer=optimizer.state_dict(),
        config=config,step=step,completed=True,accepted=False,checkpoint_role=role,training_status=status,
        losses=losses,evaluation_history=history,torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [])
    path=Path(path);temporary=path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(path)


def validate_parent(payload,identity):
    if (payload.get('format')!='align-precision-localizer/v1' or not payload.get('completed')
        or payload.get('config',{}).get('manifest_sha256')!=identity):raise ValueError('续训checkpoint/manifest身份错误')
    config=payload['config']
    for name in ('train.py','vision.py','geometry.py'):
        if config.get('source_sha256',{}).get(name)!=digest(Path(__file__).parent/name):
            raise ValueError('父checkpoint训练配方源码不同: '+name)
    if (not isinstance(payload.get('step'),int) or isinstance(payload['step'],bool) or payload['step']<0
        or not isinstance(config.get('seed'),int) or config['seed']<0
        or not np.isfinite(config.get('learning_rate',np.nan)) or config['learning_rate']<=0
        or not config.get('channels') or any(not isinstance(c,int) or c<=0 for c in config['channels'])):
        raise ValueError('父checkpoint更新号/配置错误')
    for field in ('model','optimizer','torch_rng'):
        if field not in payload:raise ValueError('父checkpoint缺失状态: '+field)
    if not isinstance(payload['torch_rng'],torch.Tensor) or payload['torch_rng'].dtype!=torch.uint8:
        raise ValueError('父checkpoint RNG状态错误')


class EvaluationBudgetStop(Exception):
    """到阶段/批次边界时放弃未完整评估，但保存当前训练恢复点。"""


def train_continuation(manifest,checkpoint,output,*,steps=20000,wall_seconds=5400,eval_interval=1000,
                       device='cuda',deadline_utc='2026-09-10T02:00:00+00:00'):
    if min(steps,wall_seconds,eval_interval)<=0 or steps>20000 or wall_seconds>5400:
        raise ValueError('超过单阶段预算或非法参数')
    deadline=datetime.fromisoformat(deadline_utc)
    if deadline.tzinfo is None:raise ValueError('截止时间必须带时区')
    if datetime.now(timezone.utc)>=deadline:raise TimeoutError('已到整个批次截止时间')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    data=KeypointTrainingData(manifest);frames=load_frames(manifest,'development')
    payload=torch.load(checkpoint,map_location=device,weights_only=True)
    validate_parent(payload,data.identity)
    parent=payload['config'];seed=int(parent['seed']);start=int(payload['step'])
    torch.manual_seed(seed);model=AlignLocalizer(channels=tuple(parent['channels'])).to(device)
    model.load_state_dict(payload['model'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=parent['learning_rate']);optimizer.load_state_dict(payload['optimizer'])
    torch.set_rng_state(payload['torch_rng'].cpu())
    if device.startswith('cuda') and payload.get('cuda_rng'):torch.cuda.set_rng_state_all([v.cpu() for v in payload['cuda_rng']])
    config=dict(parent,manifest_sha256=data.identity,continuation_parent_sha256=digest(checkpoint),
        continuation_start_step=start,additional_updates_limit=steps,eval_interval=eval_interval,
        wall_seconds=wall_seconds,deadline_utc=deadline.isoformat(),
        optimizer='restore parent AdamW state; same lr/loss/architecture/batch=1',
        selection='development PnP-only: max accuracy coverage, max coverage, min position p90; earliest tie',
        plateau='5 consecutive evaluations with accuracy-coverage gain <.01 and position-p90 relative gain <.02',
        source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')})
    write_json(output/'config.json',config)
    samples=schedule(seed,len(data.rows),start,steps);write_json(output/'schedule.json',samples)
    history=[];losses=[];completed=0;flat=0;stop_requested=[]
    def stop(signum,frame):stop_requested.append(signum)
    old_handlers={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
    best_key=None;best_step=start;previous=None
    def budget_reason():
        if stop_requested:return 'signal_stop'
        if datetime.now(timezone.utc)>=deadline:return 'batch_deadline'
        if time.monotonic()-started>=wall_seconds:return 'stage_wall_limit'
        return None
    def evaluate(step):
        nonlocal best_key,best_step,previous,flat
        def check():
            reason=budget_reason()
            if reason:raise EvaluationBudgetStop(reason)
        check()
        remaining=min(600.,wall_seconds-(time.monotonic()-started),deadline.timestamp()-time.time())
        try:assessment=evaluate_frames(frames,model,routes=False,wall_seconds=remaining,stop_check=check)
        except TimeoutError:
            reason=budget_reason()
            if reason:raise EvaluationBudgetStop(reason)
            raise
        check()
        metrics=assessment['groups']['pnp-only']['all'];key=selection_key(metrics)
        flat=flat+1 if previous is not None and plateau_step(previous,metrics) else 0
        row=dict(global_step=step,additional_updates=completed,elapsed_s=time.monotonic()-started,
            loss_last_interval_mean=float(np.mean(losses[-eval_interval:])) if losses else None,
            metrics=metrics,prediction_metrics=assessment['prediction_metrics'],plateau_count=flat)
        history.append(row);previous=metrics
        write_json(output/f'evaluation-{step:06d}.json',assessment);write_json(output/'history.json',history)
        if best_key is None or key>best_key:
            best_key=key;best_step=step
            save_checkpoint(output/'best.pt',model,optimizer,config,step,losses,history,role='development-best',status='running')
        save_checkpoint(output/'last.pt',model,optimizer,config,step,losses,history,role='last',status='running')
        write_json(output/'status.json',dict(status='running',global_step=step,additional_updates=completed,best_step=best_step,
            elapsed_s=time.monotonic()-started,plateau_count=flat,metrics=metrics))
        print(json.dumps(row),flush=True)
    try:
        reason=budget_reason()
        if not reason:
            try:evaluate(start)
            except EvaluationBudgetStop as exc:reason=str(exc)
        for index in samples:
            if reason:break
            model.train();image,uv,visible=data.get(index,device)
            optimizer.zero_grad(set_to_none=True);loss=localization_loss(model(image),uv,visible)
            if not torch.isfinite(loss):raise ValueError('非有限训练loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            completed+=1;losses.append(float(loss.detach()))
            if completed%eval_interval==0:
                try:evaluate(start+completed)
                except EvaluationBudgetStop as exc:reason=str(exc)
                if not reason and flat>=5:reason='plateau_zero_coverage' if previous['coverage']==0 else 'plateau'
            reason=reason or budget_reason()
        reason=reason or 'update_limit'
        # 非评估边界停止只保存恢复点，不超时追加评估或把未经评估权重选作best。
        save_checkpoint(output/'last.pt',model,optimizer,config,start+completed,losses,history,role='last',status=reason)
        restored=torch.load(output/'last.pt',map_location=device,weights_only=True);model.load_state_dict(restored['model'],strict=True)
        result=dict(status='completed',stop_reason=reason,additional_updates=completed,global_step=start+completed,
            best_step=best_step if best_key is not None else None,
            best_checkpoint_sha256=digest(output/'best.pt') if (output/'best.pt').exists() else None,last_checkpoint_sha256=digest(output/'last.pt'),
            elapsed_s=time.monotonic()-started,strict_reload=True,accepted=False,development_selected=best_key is not None,
            remaining_batch_seconds=max(0,deadline.timestamp()-time.time()))
        write_json(output/'result.json',result);write_json(output/'status.json',result);print(json.dumps(result),flush=True)
        return result
    except BaseException as exc:
        result=dict(status='error',additional_updates=completed,global_step=start+completed,
                    error=f'{type(exc).__name__}: {exc}',elapsed_s=time.monotonic()-started)
        write_json(output/'result.json',result);write_json(output/'status.json',result);raise
    finally:
        for sig,handler in old_handlers.items():signal.signal(sig,handler)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=20000);p.add_argument('--wall-seconds',type=int,default=5400)
    p.add_argument('--eval-interval',type=int,default=1000);p.add_argument('--device',default='cuda')
    p.add_argument('--deadline-utc',default='2026-09-10T02:00:00+00:00')
    train_continuation(**vars(p.parse_args()))
