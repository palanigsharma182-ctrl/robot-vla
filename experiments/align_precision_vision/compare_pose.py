"""同一帧/同一次角点预测的三组离线配对比较，不修改已有结果。"""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from experiments.align_precision_vision.geometry import GeometrySpec,yaw_symmetries
from experiments.align_precision_vision.pose_recovery import METHODS,PnPSpec,recover_pose,project_cube,visible_indices
from experiments.align_precision_vision.train import digest
from experiments.align_precision_vision.vision import AlignLocalizer


def stats(values):
    return dict(n=len(values),median=None,p90=None,mean=None) if not values else dict(
        n=len(values),median=float(np.median(values)),p90=float(np.quantile(values,.9)),mean=float(np.mean(values)))


def summarize(rows):
    valid=[r for r in rows if r['valid']]
    return dict(frames=len(rows),valid=len(valid),coverage=len(valid)/len(rows) if rows else None,
        pnp_eligible_frames=sum(r['visible_points']>=4 for r in rows),
        reprojection_error_px=stats([r['reprojection_error_px'] for r in valid]),
        position_error_mm=stats([r['position_error_mm'] for r in valid]),
        orientation_error_deg=stats([r['orientation_error_deg'] for r in valid]),
        accurate_8mm_5deg=sum(r['accurate_8mm_5deg'] for r in rows),
        accuracy_coverage_8mm_5deg=sum(r['accurate_8mm_5deg'] for r in rows)/len(rows) if rows else None,
        reasons=dict(Counter(r['reason'] for r in rows)),
        depth_status=dict(Counter(r['diagnostics'].get('depth_status','not_used') for r in rows)))


def compare(manifest,checkpoint,output,*,split='development',keypoints='predicted',device='cuda',wall_seconds=600):
    manifest=Path(manifest);output=Path(output)
    if output.exists():raise FileExistsError(output)
    if split not in ('train','development') or keypoints not in ('predicted','gt-diagnostic'):
        raise ValueError('只允许train/development和显式GT诊断')
    if wall_seconds<=0:raise ValueError('时间预算必须为正')
    data=json.loads(manifest.read_text());identity=digest(manifest)
    if (data.get('schema')!='align-precision-keypoints/v1' or data.get('object_model')!='upright-cube-4cm'
        or data.get('pixel_convention')!='zero-based-pixel-centers'):raise ValueError('数据合同不匹配')
    scenes={};seen=set();paths=set()
    for row in data['records']:
        if row['split'] not in ('train','development'):raise ValueError('不读取受保护test')
        if row['id'] in seen:raise ValueError('重复id')
        seen.add(row['id'])
        if row['scene'] in scenes and scenes[row['scene']]!=row['split']:raise ValueError('场景跨split')
        scenes[row['scene']]=row['split']
        path=(manifest.parent/row['file']).resolve()
        if path in paths:raise ValueError('重复文件')
        paths.add(path)
    selected=[r for r in data['records'] if r['split']==split]
    if not selected:raise ValueError('样本为空')
    model=None
    if keypoints=='predicted':
        if checkpoint is None:raise ValueError('预测对照必须提供同一checkpoint')
        payload=torch.load(checkpoint,map_location=device,weights_only=True)
        if (payload.get('format')!='align-precision-localizer/v1' or not payload.get('completed')
            or payload['config']['manifest_sha256']!=identity):raise ValueError('checkpoint身份不匹配')
        model=AlignLocalizer(channels=tuple(payload['config']['channels'])).to(device)
        model.load_state_dict(payload['model'],strict=True);model.eval()
    spec=GeometrySpec();pnp_spec=PnPSpec();started=time.monotonic()
    protocol=dict(schema='align-pose-three-arm/v1',keypoints=keypoints,split=split,
        manifest_sha256=identity,checkpoint_sha256=digest(checkpoint) if model is not None else None,
        geometry_spec=asdict(spec),pnp_spec=asdict(pnp_spec),methods=list(METHODS),
        foreground='fixed RGB red threshold intersect PnP silhouette then erode; no GT mask',
        reprojection='RMS pixel distance over all visibility-filtered correspondences, including RANSAC outliers',
        orientation='minimum SO(3) angle over four whole-object yaw symmetries',
        selection='same frozen 1000-step checkpoint; no retraining/threshold tuning',
        coverage_denominator='all selected manifest frames; report valid-only errors and common-support comparisons',
        wall_seconds=wall_seconds,planned_frames=len(selected),source_sha256={name:digest(Path(__file__).parent/name)
            for name in ('compare_pose.py','pose_recovery.py','geometry.py','vision.py')})
    output.parent.mkdir(parents=True,exist_ok=True)
    protocol_path=output.with_suffix('.protocol.json')
    if protocol_path.exists():raise FileExistsError(protocol_path)
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    records={method:[] for method in METHODS}
    for index,row in enumerate(selected):
        if time.monotonic()-started>wall_seconds:raise TimeoutError('三组对照达到预算，未生成完整结果')
        for file_key,hash_key in [('file','sha256'),('audit_file','audit_sha256')]:
            path=(manifest.parent/row[file_key]).resolve()
            if not path.is_relative_to(manifest.parent.resolve()) or digest(path)!=row[hash_key]:raise ValueError('路径或hash错误')
        with np.load(manifest.parent/row['file'],allow_pickle=False) as x:
            rgb=x['rgb'];truth=x['pixel_uv'];visible=x['visible']
        with np.load(manifest.parent/row['audit_file'],allow_pickle=False) as x:
            # 刻意不加载object_mask；GT位姿仅在求解完成后计算误差。
            audit={k:x[k] for k in ('depth_m','intrinsic','base_from_camera_cv','base_from_object','timestamp_s')}
        if model is not None:
            image=torch.tensor(rgb.transpose(2,0,1).copy(),dtype=torch.float32,device=device)[None]/255
            with torch.no_grad():prediction=model(image);uv,prob=prediction.decode()
            uv=uv[0].cpu().numpy();prob=prob[0].cpu().numpy()
        else:uv=truth;prob=visible.astype(float)
        _,ids=visible_indices(uv,prob,rgb.shape[:2],spec.visibility_threshold)
        for method in METHODS:
            tick=time.perf_counter()
            measurement=recover_pose(uv,prob,audit['intrinsic'],audit['base_from_camera_cv'],
                image_shape=rgb.shape[:2],episode=row['scene'],target='cube',timestamp_s=float(audit['timestamp_s']),
                method=method,rgb=rgb,depth_m=audit['depth_m'],spec=spec,pnp_spec=pnp_spec)
            record=dict(id=row['id'],scene=row['scene'],camera=row['camera'],phase=row['phase'],
                visible_points=len(ids),valid=measurement.reason=='valid',reason=measurement.reason,
                reprojection_error_px=None,position_error_mm=None,orientation_error_deg=None,
                accurate_8mm_5deg=False,diagnostics=measurement.diagnostics,elapsed_ms=(time.perf_counter()-tick)*1000)
            if record['valid']:
                estimate=measurement.base_from_object;gt=audit['base_from_object']
                projected,_=project_cube(np.linalg.inv(audit['base_from_camera_cv'])@estimate,audit['intrinsic'])
                record['reprojection_error_px']=float(np.sqrt(np.mean(np.sum((projected[ids]-uv[ids])**2,axis=1))))
                record['position_error_mm']=float(np.linalg.norm(estimate[:3,3]-gt[:3,3])*1000)
                record['orientation_error_deg']=float(min(np.degrees(Rotation.from_matrix(
                    estimate[:3,:3].T@(gt@s)[:3,:3]).magnitude()) for s in yaw_symmetries()))
                record['accurate_8mm_5deg']=record['position_error_mm']<=8 and record['orientation_error_deg']<=5
            records[method].append(record)
        if (index+1)%50==0:print(json.dumps(dict(processed=index+1,planned=len(selected))),flush=True)
    groups={}
    for method,rows in records.items():
        grouped=defaultdict(list)
        for row in rows:
            for name in ('all','camera:'+row['camera'],'phase:'+row['phase'],'scene:'+row['scene']):grouped[name].append(row)
        groups[method]={name:summarize(values) for name,values in grouped.items()}
    paired={}
    for left,right in [('pnp-only','pnp-depth-refine'),('pnp-only','old-corner-depth'),('pnp-depth-refine','old-corner-depth')]:
        pairs=[(a,b) for a,b in zip(records[left],records[right]) if a['valid'] and b['valid']]
        paired[left+' vs '+right]=dict(common_frames=len(pairs),left=summarize([a for a,b in pairs]),
            right=summarize([b for a,b in pairs]),position_delta_right_minus_left_mm=stats([
                b['position_error_mm']-a['position_error_mm'] for a,b in pairs]))
    result=dict(protocol=protocol,status='completed',elapsed_s=time.monotonic()-started,groups=groups,paired=paired,records=records)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({m:g['all'] for m,g in groups.items()}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['train','development'],default='development')
    p.add_argument('--keypoints',choices=['predicted','gt-diagnostic'],default='predicted')
    p.add_argument('--device',default='cuda');p.add_argument('--wall-seconds',type=int,default=600)
    compare(**vars(p.parse_args()))
