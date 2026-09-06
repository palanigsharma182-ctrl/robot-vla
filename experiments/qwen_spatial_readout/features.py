"""一次冻结Qwen前向缓存三种表示；GT不参与特征或RGB参考定位。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from experiments.oracle_reach_control.runner import digest
from robot_vla.contracts import QWEN_MODEL_ID, QWEN_REVISION
from robot_vla.diagnostics.qwen_layer_reach import FrozenQwenLayerPairContextEncoder
from robot_vla.diagnostics.qwen_spatial_probe import build_external_visual_token_layout
from robot_vla.model.factory import load_frozen_qwen_v01
from robot_vla.model.qwen_context import QwenContext, QwenVLAAdapter
from robot_vla.model.qwen_processor import QwenVLAProcessorAdapter

ADAPTER_CHECKPOINT_SHA = 'a542076f291e29b68e3d28930b15c40396d511a44eb358c2eaeb4e113c041ad6'


def red_centroid(rgb):
    """固定颜色规则，不读分割GT或目标坐标；无法定位时显式缺失。"""
    values = np.asarray(rgb, dtype=np.float32)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    mask = (red > 80) & (red > 1.4*green) & (red > 1.4*blue)
    ys, xs = np.where(mask)
    if len(xs) < 4:
        return None
    return [(float(xs.mean())+.5)/values.shape[1], (float(ys.mean())+.5)/values.shape[0]]


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'model-cache', 'checkpoint', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if digest(args.checkpoint) != ADAPTER_CHECKPOINT_SHA:
        raise ValueError('Adapter上游checkpoint身份错误')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    metadata = payload['metadata']
    assert metadata['qwen'] == dict(model_id=QWEN_MODEL_ID, revision=QWEN_REVISION)
    assert metadata['qwen_context_hidden_state'] == 12
    adapter = QwenVLAAdapter()
    adapter.load_state_dict(payload['model']['adapter'], strict=True)
    del payload
    adapter.to('cuda').requires_grad_(False).eval()
    processor = QwenVLAProcessorAdapter.from_pretrained(cache_dir=str(args.model_cache), local_files_only=True)
    assert metadata['processor_config'] == asdict(processor.config)
    encoder = FrozenQwenLayerPairContextEncoder(load_frozen_qwen_v01(
        cache_dir=str(args.model_cache), local_files_only=True, device='cuda',
    ))
    image_token_id = processor.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    manifest_text = (args.data/'samples.jsonl').read_text()
    manifest_sha = hashlib.sha256(manifest_text.encode()).hexdigest()
    rows = [json.loads(x) for x in manifest_text.splitlines()]
    identity = dict(sample_manifest_sha256=manifest_sha, sample_ids=[r['sample_id'] for r in rows])
    features = {name: [] for name in ('layer12', 'layer24', 'adapter12')}
    rgb_predictions = []
    centers = None
    started = time.monotonic()
    for index, row in enumerate(rows):
        with np.load(args.data/'images'/f"{row['sample_id']}.npz", allow_pickle=False) as data:
            assert set(data.files) == {'rgb_external', 'rgb_wrist'}
            external, wrist = data['rgb_external'], data['rgb_wrist']
        assert hashlib.sha256(external.tobytes()).hexdigest() == row['rgb_external_sha256']
        assert hashlib.sha256(wrist.tobytes()).hexdigest() == row['rgb_wrist_sha256']
        encoded = processor.encode(external, wrist, row['instruction'])
        inputs = {k:v.to('cuda') if isinstance(v, torch.Tensor) else v for k,v in encoded.model_inputs.items()}
        layout = build_external_visual_token_layout(
            inputs['input_ids'], inputs['image_grid_thw'], image_token_id=image_token_id,
            merge_size=processor.config.merge_size,
        )
        current_centers = layout.normalized_centers[layout.mask].cpu()
        if centers is None:
            centers = current_centers
        torch.testing.assert_close(current_centers, centers, rtol=0, atol=0)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            pair = encoder(inputs)
            assert bool(pair.mask[layout.mask].all())
            adapted = adapter(QwenContext(pair.layer12_tokens, pair.mask))
        for name, value in (('layer12', pair.layer12_tokens), ('layer24', pair.layer24_tokens), ('adapter12', adapted.tokens)):
            features[name].append(value[layout.mask].detach().cpu().to(torch.bfloat16))
        grid = inputs['image_grid_thw'][0].tolist()
        effective_hw = [grid[1]*processor.config.patch_size, grid[2]*processor.config.patch_size]
        resample = getattr(processor.processor.image_processor, 'resample', Image.Resampling.BICUBIC)
        resized = Image.fromarray(external).resize((effective_hw[1], effective_hw[0]), resample=resample)
        rgb_predictions.append(red_centroid(resized))
        if index in (0, 31, 63, 127, 191):
            print(json.dumps(dict(samples=index+1, tokens=len(centers), effective_hw=effective_hw,
                                 elapsed_s=time.monotonic()-started)), flush=True)
    cached = {name:torch.stack(values) for name,values in features.items()}
    assert not any(p.requires_grad for p in encoder.parameters())
    assert not any(p.requires_grad for p in adapter.parameters())
    assert digest(args.data/'samples.jsonl') == manifest_sha, '采样清单在特征提取期间变化'
    torch.save(dict(features=cached, centers=centers, **identity), args.output/'features.pt')
    (args.output/'rgb_predictions.json').write_text(json.dumps(dict(predictions=rgb_predictions, **identity))+'\n')
    info = dict(
        samples=len(rows), tokens=len(centers), effective_hw=effective_hw,
        feature_shapes={k:list(v.shape) for k,v in cached.items()},
        adapter_checkpoint_sha256=ADAPTER_CHECKPOINT_SHA, qwen_model=QWEN_MODEL_ID,
        qwen_revision=QWEN_REVISION, adapter_layer_identity=metadata['qwen_context_hidden_state'],
        processor_config=asdict(processor.config), features_sha256=digest(args.output/'features.pt'),
        sample_manifest_sha256=manifest_sha,
        rgb_predictions_sha256=digest(args.output/'rgb_predictions.json'),
        scope='all current external visual tokens from actual dual-image V1 prompt; not all text/wrist tokens',
        no_gt_selection=True, elapsed_s=time.monotonic()-started,
    )
    (args.output/'summary.json').write_text(json.dumps(info,indent=2)+'\n')


if __name__ == '__main__':
    main()
