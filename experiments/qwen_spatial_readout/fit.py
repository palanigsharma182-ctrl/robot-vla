"""仅训练小型空间读出器，保留全部初始化、错配和无效预测分母。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from robot_vla.diagnostics.qwen_spatial_probe import unproject_gl_camera_to_world_plane
from robot_vla.diagnostics.v2_online_geometry_probe import OnlineVisualTargetProbe, online_geometry_probe_loss

INIT_SEEDS = (42, 43, 44)
TRAIN_STEPS, BATCH_SIZE = 400, 32
PLANE_Z_M = .02


def validate_cache(cached, rgb, rows, manifest_sha):
    """在启动CUDA前拒绝清单重排、RGB错配及损坏的特征张量。"""
    expected_ids = [row['sample_id'] for row in rows]
    for artifact in (cached, rgb):
        if artifact.get('sample_manifest_sha256') != manifest_sha or artifact.get('sample_ids') != expected_ids:
            raise ValueError('缓存样本身份或顺序与当前manifest不一致')
    dimensions = dict(layer12=2048, layer24=2048, adapter12=720)
    if set(cached['features']) != set(dimensions):
        raise ValueError('特征条件不完整')
    centers = cached['centers']
    if centers.ndim!=2 or centers.shape[1]!=2 or not torch.isfinite(centers).all():
        raise ValueError('token坐标无效')
    for name, dimension in dimensions.items():
        tensor = cached['features'][name]
        if tuple(tensor.shape)!=(len(rows),len(centers),dimension) or not torch.isfinite(tensor).all():
            raise ValueError('特征shape或数值无效: '+name)
    if len(rgb['predictions'])!=len(rows):
        raise ValueError('RGB预测数量不完整')
    for uv in rgb['predictions']:
        if uv is not None and (np.asarray(uv).shape!=(2,) or not np.isfinite(uv).all()):
            raise ValueError('RGB预测无效')


def train_statistics(features, train_indices):
    """均值和标准差只来自训练场景，不能让development参与归一化。"""
    selected = features[train_indices].float()
    return selected.mean((0, 1)), selected.std((0, 1), unbiased=False).clamp_min(1e-4)


def scene_derangement(rows):
    """固定scene循环置换，保持phase；映射不读取标签或可见性。"""
    scenes = sorted({r['scene'] for r in rows})
    if len(scenes) < 2:
        raise ValueError('错配至少需要两个场景')
    next_scene = dict(zip(scenes, scenes[1:]+scenes[:1]))
    lookup = {(r['scene'], r['phase']):i for i, r in enumerate(rows)}
    result = [lookup[next_scene[r['scene']], r['phase']] for r in rows]
    assert all(rows[j]['scene'] != r['scene'] and rows[j]['phase'] == r['phase'] for r,j in zip(rows,result))
    assert len(set(result)) == len(rows)
    return result


def predicted_world(uv, row):
    """转换只使用冻结平面和标定，不读取该帧GT高度。"""
    cal = row['calibration']
    return unproject_gl_camera_to_world_plane(
        np.asarray(uv), PLANE_Z_M, cal['intrinsic_external'], cal['world_from_external'], *row['image_size'],
    )


def error_records(predictions, rows):
    records = []
    for uv, row in zip(predictions, rows):
        item = {k:row[k] for k in ('sample_id', 'scene', 'phase', 'visibility')}
        item.update(predicted_uv=uv, uv_error=None, pixel_error=None, xy_error_m=None)
        if uv is not None and np.isfinite(uv).all():
            delta = np.asarray(uv)-np.asarray(row['uv'])
            item['uv_error'] = float(np.linalg.norm(delta))
            item['pixel_error'] = float(np.linalg.norm(delta*np.asarray(row['image_size'])[::-1]))
            try:
                world = predicted_world(uv, row)
                item['xy_error_m'] = float(np.linalg.norm(world[:2]-np.asarray(row['object_position_world_m'])[:2]))
            except ValueError as error:
                item['invalid_unprojection'] = str(error)
        records.append(item)
    if len(records) != len(rows):
        raise ValueError('预测数量不完整')
    return records


def metrics(records):
    summary = dict(samples=len(records), scenes=len({r['scene'] for r in records}))
    for name in ('uv_error', 'pixel_error', 'xy_error_m'):
        values = [r[name] for r in records if r[name] is not None]
        summary[name] = dict(valid=len(values), invalid=len(records)-len(values),
                             median=float(np.median(values)) if values else None,
                             p90=float(np.percentile(values,90)) if values else None)
    for cm in (1,2,4):
        passed = sum(r['xy_error_m'] is not None and r['xy_error_m'] < cm/100 for r in records)
        summary[f'within_{cm}cm'] = dict(passed=passed, total=len(records), fraction=passed/len(records))
    return summary


def summarize(predictions, rows):
    records = error_records(predictions, rows)
    result = metrics(records)
    for key in ('phase', 'visibility', 'scene'):
        result['by_'+key] = {str(value):metrics([r for r in records if r[key]==value]) for value in sorted({r[key] for r in records})}
    return result, records


class Readout(nn.Module):
    """低容量selector/坐标头；可选32维非线性投影作预定有界复核。"""
    def __init__(self, dimension, nonlinear=False):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(dimension,32), nn.SiLU()) if nonlinear else nn.Identity()
        self.probe = OnlineVisualTargetProbe(32 if nonlinear else dimension, target_count=1)

    def forward(self, tokens):
        return self.probe(self.project(tokens))


def train_head(features, centers, labels, train_indices, seed, nonlinear):
    torch.manual_seed(seed)
    model = Readout(features.shape[-1], nonlinear).to(features.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed+1000)
    history = []
    for step in range(TRAIN_STEPS):
        indices = train_indices[torch.randint(len(train_indices), (BATCH_SIZE,), generator=generator)]
        output = model(features[indices])
        loss = online_geometry_probe_loss(output, labels[indices], centers.expand(BATCH_SIZE,-1,-1), selector_loss_weight=.1)
        optimizer.zero_grad(set_to_none=True)
        loss.loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step in (0, TRAIN_STEPS-1):
            history.append(dict(step=step+1, loss=float(loss.loss), coordinate_loss=float(loss.coordinate_loss),
                                selector_loss=float(loss.selector_loss), gradient_norm=float(norm)))
    model.eval()
    return model, history


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'features', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest_text = (args.data/'samples.jsonl').read_text()
    rows = [json.loads(line) for line in manifest_text.splitlines()]
    assert len(rows) == 192 and len({r['sample_id'] for r in rows}) == 192
    # 此固定桌面协议预先要求中心在视野内；失败时停止，不删帧或夹紧GT。
    assert all(r['projected'] and np.isfinite(r['uv']).all() for r in rows), 'out-of-view标签不能直接训练sigmoid读出器'
    train = torch.tensor([i for i,r in enumerate(rows) if r['split']=='train'])
    dev = torch.tensor([i for i,r in enumerate(rows) if r['split']=='development'])
    train_scenes, dev_scenes = ({rows[i]['scene'] for i in indices.tolist()} for indices in (train,dev))
    assert len(train_scenes)==32 and len(dev_scenes)==16 and not train_scenes & dev_scenes
    assert all(len([r for r in rows if r['scene']==scene])==4 for scene in train_scenes|dev_scenes)
    dev_rows = [rows[i] for i in dev.tolist()]
    permutation = scene_derangement(dev_rows)
    cached = torch.load(args.features/'features.pt', map_location='cpu', weights_only=True)
    rgb_artifact = json.loads((args.features/'rgb_predictions.json').read_text())
    validate_cache(cached, rgb_artifact, rows, hashlib.sha256(manifest_text.encode()).hexdigest())
    centers = cached['centers'].float().to('cuda')
    labels = torch.tensor([r['uv'] for r in rows], device='cuda').unsqueeze(1)
    results, predictions = {}, {}
    rgb = rgb_artifact['predictions']
    prior = labels[train].mean(0)[0].cpu().tolist()
    for name, values in (('rgb',[rgb[i] for i in dev.tolist()]), ('train_mean',[prior]*len(dev)),
                         ('rgb_shuffled',[rgb[dev[i]] for i in permutation])):
        results[name], predictions[name] = summarize(values,dev_rows)
    representations = dict(cached['features'])
    representations['grid_only'] = centers.cpu().expand(len(rows),-1,-1)
    for name, raw in representations.items():
        mean, std = train_statistics(raw, train)
        features = (raw.float().to('cuda')-mean.to('cuda'))/std.to('cuda')
        linear_medians = []
        for nonlinear in (False, True):
            if nonlinear and (name=='grid_only' or np.mean(linear_medians) <= .02):
                continue
            for seed in INIT_SEEDS:
                key = f'{name}_{"mlp32" if nonlinear else "linear"}_{seed}'
                model, history = train_head(features, centers, labels, train, seed, nonlinear)
                with torch.no_grad():
                    values = model(features[dev]).predicted_uv[:,0].cpu().tolist()
                    train_values = model(features[train]).predicted_uv[:,0].cpu().tolist()
                results[key], predictions[key] = summarize(values, dev_rows)
                results[key]['training'] = dict(history=history, parameters=sum(p.numel() for p in model.parameters()),
                    steps=TRAIN_STEPS, batch_size=BATCH_SIZE, seed=seed,
                    train_metrics=metrics(error_records(train_values, [rows[i] for i in train.tolist()])))
                results[key+'_shuffled'], predictions[key+'_shuffled'] = summarize([values[i] for i in permutation],dev_rows)
                if not nonlinear:
                    median = results[key]['xy_error_m']['median']
                    linear_medians.append(median if median is not None else float('inf'))
                torch.save(dict(model=model.cpu().state_dict(), mean=mean, std=std, nonlinear=nonlinear,
                                dimension=raw.shape[-1], seed=seed), args.output/(key+'.pt'))
                print(json.dumps(dict(condition=key, median_xy_m=results[key]['xy_error_m']['median'],
                                      elapsed_s=time.monotonic()-started)), flush=True)
        del features
    (args.output/'summary.json').write_text(json.dumps(dict(
        results=results, dev_permutation=permutation, init_seeds=INIT_SEEDS, plane_z_m=PLANE_Z_M,
        mlp_trigger='mean of 3 linear development median XY errors > .02m; no grid-only MLP; development tuning diagnostic',
        elapsed_s=time.monotonic()-started, samples=len(rows), train_samples=len(train), dev_samples=len(dev),
    ), indent=2, allow_nan=False)+'\n')
    (args.output/'predictions.json').write_text(json.dumps(predictions,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
