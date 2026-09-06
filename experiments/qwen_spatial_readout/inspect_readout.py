"""结果出现后的CPU诊断：分离selector、坐标回归与网格分辨率；不再拟合。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.qwen_spatial_readout.fit import Readout, summarize, validate_cache


def main():
    parser = argparse.ArgumentParser()
    for name in ('data','features','fit','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    manifest = (args.data/'samples.jsonl').read_text()
    rows = [json.loads(line) for line in manifest.splitlines()]
    cached = torch.load(args.features/'features.pt',map_location='cpu',weights_only=True)
    rgb = json.loads((args.features/'rgb_predictions.json').read_text())
    validate_cache(cached,rgb,rows,hashlib.sha256(manifest.encode()).hexdigest())
    dev = [i for i,r in enumerate(rows) if r['split']=='development']
    dev_rows = [rows[i] for i in dev]
    centers = cached['centers'].float()
    uv = torch.tensor([r['uv'] for r in dev_rows])
    nearest = torch.cdist(uv,centers).argmin(-1)
    train_uv = torch.tensor([r['uv'] for r in rows if r['split']=='train'])
    train_nearest = torch.cdist(train_uv,centers).argmin(-1)
    majority = int(torch.bincount(train_nearest,minlength=len(centers)).argmax())
    oracle_metrics,_ = summarize(centers[nearest].tolist(),dev_rows)
    result = dict(status='posthoc development diagnosis, CPU only, no refitting or checkpoint selection',
                  oracle_nearest_grid=oracle_metrics, nearest_grid_label_counts=torch.bincount(nearest,minlength=len(centers)).tolist(),
                  train_majority_token=majority, train_majority_dev_accuracy=float((nearest==majority).float().mean()),
                  conditions={})
    for path in sorted(args.fit.glob('*.pt')):
        payload = torch.load(path,map_location='cpu',weights_only=True)
        model = Readout(payload['dimension'],payload['nonlinear'])
        model.load_state_dict(payload['model'],strict=True)
        model.eval().requires_grad_(False)
        representation = path.stem.split('_linear_')[0].split('_mlp32_')[0]
        raw = centers.expand(len(rows),-1,-1) if representation=='grid_only' else cached['features'][representation].float()
        with torch.no_grad():
            output = model((raw[dev]-payload['mean'])/payload['std'])
            logits = output.selector_logits[:,0]
            selected = logits.argmax(-1)
            soft_uv = logits.softmax(-1)@centers
            entropy = -(logits.log_softmax(-1)*logits.softmax(-1)).sum(-1)
        hard_metrics,_ = summarize(centers[selected].tolist(),dev_rows)
        soft_metrics,_ = summarize(soft_uv.tolist(),dev_rows)
        correct = selected==nearest
        chosen = torch.nonzero(correct,as_tuple=False).flatten().tolist()
        selected_correct_metrics = summarize(output.predicted_uv[chosen,0].tolist(),[dev_rows[i] for i in chosen])[0] if chosen else None
        result['conditions'][path.stem] = dict(
            selector_nearest_accuracy=float(correct.float().mean()),
            phase0_accuracy=float(correct[torch.tensor([r['phase']==0 for r in dev_rows])].float().mean()),
            selector_entropy_mean=float(entropy.mean()),
            coordinate_metrics_when_selection_correct=selected_correct_metrics,
            hard_grid_readout=hard_metrics, soft_grid_readout=soft_metrics,
        )
    # x模式保留已有事后诊断，不覆盖冻结的训练结果。
    with args.output.open('x') as file:
        json.dump(result,file,indent=2,allow_nan=False)
        file.write('\n')
    print(json.dumps(dict(conditions=len(result['conditions']),oracle_grid_median_xy_m=oracle_metrics['xy_error_m']['median'])))


if __name__=='__main__':
    main()
