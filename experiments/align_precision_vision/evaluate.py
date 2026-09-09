"""完整分母上的定位/位姿development评估；GT关键点仅作独立诊断。"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from experiments.align_precision_vision.geometry import yaw_symmetries, corner_permutations
from experiments.align_precision_vision.pose_recovery import recover_pose,METHODS
from experiments.align_precision_vision.train import digest
from experiments.align_precision_vision.vision import AlignLocalizer


def quantiles(values):
    return None if not values else dict(zip(['median','p90','max'],np.quantile(values,[.5,.9,1]).tolist()))


def evaluate(manifest, output, *, checkpoint=None, split='development', device='cpu',method='pnp-only'):
    manifest=Path(manifest);output=Path(output)
    if output.exists():raise FileExistsError(output)
    data=json.loads(manifest.read_text())
    if (data.get('schema')!='align-precision-keypoints/v1' or data.get('object_model')!='upright-cube-4cm'
        or data.get('pixel_convention')!='zero-based-pixel-centers' or split not in ('train','development','all')):
        raise ValueError('评估仅接受声明的训练/development几何数据')
    seen=set();scenes={};paths=set()
    for row in data['records']:
        if row['split'] not in ('train','development'):raise ValueError('不消费final test')
        if row['id'] in seen:raise ValueError('重复样本')
        seen.add(row['id'])
        if row['scene'] in scenes and scenes[row['scene']]!=row['split']:raise ValueError('跨split场景')
        scenes[row['scene']]=row['split']
        file=(manifest.parent/row['file']).resolve()
        if file in paths:raise ValueError('重复文件')
        paths.add(file)
    rows=[r for r in data['records'] if split=='all' or r['split']==split]
    if not rows:raise ValueError('评估集合为空')
    model=None
    if checkpoint is not None:
        payload=torch.load(checkpoint,map_location=device,weights_only=True)
        if payload.get('format')!='align-precision-localizer/v1' or not payload.get('completed'):
            raise ValueError('定位checkpoint身份不合法')
        if payload['config'].get('manifest_sha256')!=digest(manifest):
            raise ValueError('checkpoint与评估数据manifest身份不一致')
        model=AlignLocalizer(channels=tuple(payload['config']['channels'])).to(device)
        model.load_state_dict(payload['model'],strict=True);model.eval()
    samples=[];groups=defaultdict(list)
    for row in rows:
        loaded=[]
        for key,hash_key in [('file','sha256'),('audit_file','audit_sha256')]:
            path=(manifest.parent/row[key]).resolve()
            if not path.is_relative_to(manifest.parent.resolve()) or digest(path)!=row[hash_key]:
                raise ValueError('样本路径或hash不符')
            with np.load(path,allow_pickle=False) as x:loaded.append({k:x[k] for k in x.files})
        sample,audit=loaded
        if not np.array_equal(sample['visible'],audit['visible']) or not np.array_equal(
            sample['pixel_uv'],audit['pixel_uv'],equal_nan=True):raise ValueError('监督与审计不一致')
        truth=sample['pixel_uv'];visible=sample['visible']
        if model is None:
            uv=truth;prob=visible.astype(float)
        else:
            image=torch.tensor(sample['rgb'].transpose(2,0,1).copy(),dtype=torch.float32,device=device)[None]/255
            with torch.no_grad():out=model(image);p,v=out.decode()
            uv=p[0].cpu().numpy();prob=v[0].cpu().numpy()
        measurement=recover_pose(uv,prob,audit['intrinsic'],audit['base_from_camera_cv'],
            image_shape=sample['rgb'].shape[:2],episode=row['scene'],target='cube',timestamp_s=float(audit['timestamp_s']),
            method=method,rgb=sample['rgb'],depth_m=audit['depth_m'])
        record=dict(id=row['id'],scene=row['scene'],split=row['split'],camera=row['camera'],phase=row['phase'],
            visible_count=int(visible.sum()),predicted_visible_count=int((prob>=.8).sum()),
            reason=measurement.reason,position_error_mm=None,rotation_error_deg=None,pixel_error_px=None)
        # 评估也只允许整组对称，不能逐点各选一个对应关系。
        best=None
        for perm in corner_permutations():
            mask=visible[perm]
            error=float(np.linalg.norm(uv[mask]-truth[perm][mask],axis=-1).mean()) if mask.any() else None
            if error is not None and np.isfinite(error) and (best is None or error<best[0]):
                best=(error,mask)
        if best is not None:
            record['pixel_error_px']=best[0]
            predicted=prob>=.8;mask=best[1]
            record.update(visibility_tp=int((predicted&mask).sum()),visibility_fp=int((predicted&~mask).sum()),
                          visibility_fn=int((~predicted&mask).sum()))
        elif not visible.any():
            record.update(visibility_tp=0,visibility_fp=int((prob>=.8).sum()),visibility_fn=0)
        if measurement.base_from_object is not None:
            est=measurement.base_from_object;gt=audit['base_from_object']
            record['position_error_mm']=float(np.linalg.norm(est[:3,3]-gt[:3,3])*1000)
            record['rotation_error_deg']=float(min(np.degrees(Rotation.from_matrix(
                est[:3,:3].T@(gt@s)[:3,:3]).magnitude()) for s in yaw_symmetries()))
        record['within_2mm_5deg']=bool(record['position_error_mm'] is not None and
            record['position_error_mm']<=2 and record['rotation_error_deg']<=5)
        samples.append(record)
        for key in ('all','camera:'+row['camera'],'phase:'+row['phase'],'split:'+row['split']):groups[key].append(record)
    def summarize(items):
        valid=[x for x in items if x['reason']=='valid']
        tp=sum(x.get('visibility_tp',0) for x in items);fp=sum(x.get('visibility_fp',0) for x in items)
        fn=sum(x.get('visibility_fn',0) for x in items)
        return dict(frames=len(items),valid=len(valid),valid_fraction=len(valid)/len(items),
            within_2mm_5deg_fraction=sum(x['within_2mm_5deg'] for x in items)/len(items),
            reasons=dict(Counter(x['reason'] for x in items)),
            position_error_mm=quantiles([x['position_error_mm'] for x in valid]),
            rotation_error_deg=quantiles([x['rotation_error_deg'] for x in valid]),
            pixel_error_px=quantiles([x['pixel_error_px'] for x in items if x['pixel_error_px'] is not None]),
            visible_count=dict(Counter(x['visible_count'] for x in items)),
            visibility_precision=tp/(tp+fp) if tp+fp else None,visibility_recall=tp/(tp+fn) if tp+fn else None)
    result=dict(schema='align-precision-evaluation/v2',method=method,mode='gt-keypoints-diagnostic' if model is None else 'learned-keypoints',
        split=split,manifest_sha256=digest(manifest),checkpoint_sha256=digest(checkpoint) if checkpoint else None,
        threshold_note='2mm/5deg为本轮感知诊断，不修改技能验收',groups={k:summarize(v) for k,v in groups.items()},samples=samples)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['groups']['all']),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--split',choices=['train','development','all'],default='development');p.add_argument('--device',default='cpu')
    p.add_argument('--method',choices=METHODS,default='pnp-only')
    evaluate(**vars(p.parse_args()))
