"""共享一次角点预测的单帧/局部/因果多视角development比较。"""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from experiments.align_precision_vision.geometry import GeometrySpec, PoseMeasurement, corner_permutations, yaw_symmetries
from experiments.align_precision_vision.pose_recovery import PnPSpec, recover_pose, project_cube, visible_indices
from experiments.align_precision_vision.train import digest
from experiments.align_precision_vision.vision import AlignLocalizer


def write_json(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def stats(values):
    values=[v for v in values if v is not None]
    return dict(n=len(values),median=float(np.median(values)) if values else None,
                p90=float(np.quantile(values,.9)) if values else None,
                mean=float(np.mean(values)) if values else None)


@dataclass
class Frame:
    row: dict
    rgb: np.ndarray
    truth_uv: np.ndarray
    truth_visible: np.ndarray
    audit: dict


def load_frames(manifest,split='development'):
    """按场景划分且检查hash，只打开指定split；GT mask不在读取字段中。"""
    path=Path(manifest);data=json.loads(path.read_text());seen=set();files=set();scenes={};times=set();frames=[]
    if (data.get('schema')!='align-precision-keypoints/v1' or data.get('object_model')!='upright-cube-4cm'
        or data.get('pixel_convention')!='zero-based-pixel-centers' or split not in ('train','development')):
        raise ValueError('数据合同不符')
    for row in data['records']:
        if row['split'] not in ('train','development'):raise ValueError('不消费受保护test')
        if row['id'] in seen:raise ValueError('重复id')
        seen.add(row['id'])
        if row['scene'] in scenes and scenes[row['scene']]!=row['split']:raise ValueError('场景跨split')
        scenes[row['scene']]=row['split']
        file=(path.parent/row['file']).resolve()
        if file in files:raise ValueError('重复样本路径')
        files.add(file)
        if row['split']!=split:continue
        loaded=[]
        for key,hash_key,names in [('file','sha256',('rgb','pixel_uv','visible')),
            ('audit_file','audit_sha256',('depth_m','intrinsic','base_from_camera_cv','base_from_object','timestamp_s'))]:
            source=(path.parent/row[key]).resolve()
            if not source.is_relative_to(path.parent.resolve()) or digest(source)!=row[hash_key]:raise ValueError('样本路径/hash错误')
            with np.load(source,allow_pickle=False) as x:loaded.append({name:x[name] for name in names})
        sample,audit=loaded;stamp=float(audit['timestamp_s']);identity=(row['scene'],row['camera'],stamp)
        if not np.isfinite(stamp) or identity in times:raise ValueError('时间非有限或相机重复时间')
        times.add(identity)
        if (sample['rgb'].dtype!=np.uint8 or sample['rgb'].ndim!=3 or sample['rgb'].shape[-1]!=3
            or sample['pixel_uv'].shape!=(8,2) or sample['visible'].shape!=(8,) or sample['visible'].dtype!=bool):
            raise ValueError('监督shape/dtype错误')
        visible_indices(sample['pixel_uv'],sample['visible'].astype(float),sample['rgb'].shape[:2],.8)
        frames.append(Frame(row,sample['rgb'],sample['pixel_uv'],sample['visible'],audit))
    if not frames:raise ValueError('无评估样本')
    return sorted(frames,key=lambda f:(f.row['scene'],f.row['camera'],float(f.audit['timestamp_s'])))


def motion_audit(frames):
    """仅报告静态假设的偏离，不向求解器传GT、不按GT删除帧。"""
    groups=defaultdict(list)
    for f in frames:groups[f.row['scene']].append(f.audit['base_from_object'])
    result={}
    for scene,poses in groups.items():
        ref=poses[0]
        result[scene]=dict(max_translation_from_first_mm=max(float(np.linalg.norm(p[:3,3]-ref[:3,3])*1000) for p in poses),
            max_rotation_from_first_deg=max(float(np.degrees(Rotation.from_matrix(ref[:3,:3].T@p[:3,:3]).magnitude())) for p in poses))
    return result


def load_model(checkpoint,manifest,device):
    payload=torch.load(checkpoint,map_location=device,weights_only=True)
    if (payload.get('format')!='align-precision-localizer/v1' or not payload.get('completed')
        or payload['config']['manifest_sha256']!=digest(manifest)):raise ValueError('权重/数据身份不符')
    model=AlignLocalizer(channels=tuple(payload['config']['channels'])).to(device)
    model.load_state_dict(payload['model'],strict=True);return model.eval()


def predictions(frames,model,check=lambda:None):
    values=[]
    if model is None:
        for f in frames:
            check();values.append((f.truth_uv,f.truth_visible.astype(float)))
        return values
    was_training=model.training;model.eval();device=next(model.parameters()).device
    try:
        with torch.no_grad():
            for f in frames:
                check()
                rgb=torch.as_tensor(f.rgb.transpose(2,0,1).copy(),device=device,dtype=torch.float32)[None]/255
                uv,visibility=model(rgb).decode();values.append((uv[0].cpu().numpy(),visibility[0].cpu().numpy()))
    finally:model.train(was_training)
    check()
    return values


def prediction_metrics(frames,values):
    groups=defaultdict(list)
    for f,(uv,prob) in zip(frames,values,strict=True):
        # 评估对应关系仅允许一整组yaw排列；不进入任何求解输入。
        candidates=[]
        for permutation in corner_permutations():
            mask=f.truth_visible[permutation]
            error=float(np.linalg.norm(uv[mask]-f.truth_uv[permutation][mask],axis=1).mean()) if mask.any() else 0.
            candidates.append((error,permutation))
        error,permutation=min(candidates,key=lambda x:x[0]);truth=f.truth_visible[permutation];pred=prob>=.8
        _,ids=visible_indices(uv,prob,f.rgb.shape[:2],.8)
        record=dict(pixel_error_px=error if truth.any() else None,tp=int((pred&truth).sum()),fp=int((pred&~truth).sum()),
            fn=int((~pred&truth).sum()),gt_visible=int(truth.sum()),pred_visible=len(ids))
        groups['all'].append(record);groups['camera:'+f.row['camera']].append(record)
    result={}
    for key,rows in groups.items():
        tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
        result[key]=dict(frames=len(rows),pixel_error_px=stats([r['pixel_error_px'] for r in rows]),tp=tp,fp=fp,fn=fn,
            precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
            gt_visible_histogram=dict(Counter(r['gt_visible'] for r in rows)),pred_visible_histogram=dict(Counter(r['pred_visible'] for r in rows)))
    return result


def measure_record(frame,uv,prob,measurement,elapsed_ms):
    _,ids=visible_indices(uv,prob,frame.rgb.shape[:2],.8);a=frame.audit
    r=dict(id=frame.row['id'],scene=frame.row['scene'],camera=frame.row['camera'],phase=frame.row['phase'],
        timestamp_s=float(a['timestamp_s']),visible_points=len(ids),valid=measurement.reason=='valid',reason=measurement.reason,
        source=measurement.source,measurement_age_s=float(a['timestamp_s'])-measurement.timestamp_s,
        reprojection_error_px=None,position_error_mm=None,orientation_error_deg=None,accurate_8mm_5deg=False,
        elapsed_ms=elapsed_ms,diagnostics=measurement.diagnostics)
    if r['valid']:
        estimate=measurement.base_from_object;gt=a['base_from_object']
        projected,_=project_cube(np.linalg.inv(a['base_from_camera_cv'])@estimate,a['intrinsic'])
        if len(ids):
            # 多帧输出可能等价于过去帧yaw身份，用当前整组对应的最小重投影报告。
            permutations=corner_permutations() if measurement.source.startswith(('precision-multiview','precision-static-hold')) else [np.arange(8)]
            r['reprojection_error_px']=float(min(np.sqrt(np.mean(np.sum((projected[p][ids]-uv[ids])**2,axis=1))) for p in permutations))
        r['position_error_mm']=float(np.linalg.norm(estimate[:3,3]-gt[:3,3])*1000)
        r['orientation_error_deg']=float(min(np.degrees(Rotation.from_matrix(estimate[:3,:3].T@(gt@s)[:3,:3]).magnitude()) for s in yaw_symmetries()))
        r['accurate_8mm_5deg']=r['position_error_mm']<=8 and r['orientation_error_deg']<=5
    return r


def summarize(rows):
    good=[r for r in rows if r['valid']];n=len(rows);accurate=sum(r['accurate_8mm_5deg'] for r in good)
    return dict(frames=n,valid=len(good),coverage=len(good)/n if n else None,accurate_8mm_5deg=accurate,
        accuracy_coverage_8mm_5deg=accurate/n if n else None,false_accept_count=len(good)-accurate,
        false_accept_fraction=(len(good)-accurate)/len(good) if good else None,
        **{name:stats([r[name] for r in good]) for name in ('reprojection_error_px','position_error_mm','orientation_error_deg','measurement_age_s')},
        elapsed_ms=stats([r['elapsed_ms'] for r in rows]),reasons=dict(Counter(r['reason'] for r in rows)))


def evaluate_frames(frames,model=None,*,routes=False,wall_seconds=600,stop_check=lambda:None):
    started=time.monotonic()
    def check():
        stop_check()
        if time.monotonic()-started>=wall_seconds:raise TimeoutError('评估墙钟上限')
    check();values=predictions(frames,model,check);methods=['pnp-only']
    if routes:
        from experiments.align_precision_vision.local_pose import recover_local_pose
        from experiments.align_precision_vision.multiview_pose import ViewObservation,recover_multiview_pose
        methods+=['local-p3p','static-hold','multiview-rgb','multiview']
    records={m:[] for m in methods};history=defaultdict(list);last={}
    for f,(uv,prob) in zip(frames,values,strict=True):
        check()
        a=f.audit;stamp=float(a['timestamp_s']);key=(f.row['scene'],f.row['camera'])
        kwargs=dict(image_shape=f.rgb.shape[:2],episode=f.row['scene'],target='cube',timestamp_s=stamp,depth_m=a['depth_m'],rgb=f.rgb)
        tick=time.perf_counter();single=recover_pose(uv,prob,a['intrinsic'],a['base_from_camera_cv'],**kwargs)
        records['pnp-only'].append(measure_record(f,uv,prob,single,(time.perf_counter()-tick)*1000))
        if not routes:continue
        tick=time.perf_counter();local=recover_local_pose(uv,prob,a['intrinsic'],a['base_from_camera_cv'],**kwargs)
        records['local-p3p'].append(measure_record(f,uv,prob,local,(time.perf_counter()-tick)*1000))
        if single.reason=='valid':last[key]=single
        tick=time.perf_counter();previous=last.get(key)
        if previous is not None and stamp-previous.timestamp_s<=1.5:
            hold=PoseMeasurement(previous.episode,previous.target,previous.timestamp_s,previous.base_from_object,'valid',
                'precision-static-hold/v1',dict(assumption='stationary object',max_hold_age_s=1.5))
        else:hold=PoseMeasurement(f.row['scene'],'cube',stamp,None,'hold_unavailable','precision-static-hold/v1')
        records['static-hold'].append(measure_record(f,uv,prob,hold,(time.perf_counter()-tick)*1000))
        current=ViewObservation(uv=uv,visibility=prob,intrinsic=a['intrinsic'],base_from_camera_cv=a['base_from_camera_cv'],**kwargs)
        recent=[v for v in history[key] if stamp-v.timestamp_s<=1.5][-3:]
        for method,use_depth in [('multiview-rgb',False),('multiview',True)]:
            check();tick=time.perf_counter();fusion=recover_multiview_pose(recent,current=current,use_depth=use_depth)
            records[method].append(measure_record(f,uv,prob,fusion,(time.perf_counter()-tick)*1000))
        history[key]=(recent+[current])[-4:]
    groups={}
    for method,rows in records.items():
        check()
        bins=defaultdict(list)
        for r in rows:
            for key in ('all','camera:'+r['camera'],'scene:'+r['scene'],'phase:'+r['phase'],'visible:'+str(r['visible_points'])):bins[key].append(r)
        groups[method]={key:summarize(rows) for key,rows in bins.items()}
    paired={}
    for method in methods[1:]:
        common=[(a,b) for a,b in zip(records['pnp-only'],records[method],strict=True) if a['valid'] and b['valid']]
        paired[method]=dict(common_frames=len(common),position_delta_method_minus_pnp_mm=stats([b['position_error_mm']-a['position_error_mm'] for a,b in common]))
    result=dict(status='completed',keypoints='gt-diagnostic' if model is None else 'predicted',groups=groups,paired_vs_pnp=paired,
                prediction_metrics=prediction_metrics(frames,values),object_motion_audit=motion_audit(frames),records=records,elapsed_s=time.monotonic()-started)
    check();return result


def compare_routes(manifest,checkpoint,output,*,keypoints='predicted',device='cuda',wall_seconds=600,
                   deadline_utc='2026-09-10T02:00:00+00:00'):
    started=time.monotonic();deadline=datetime.fromisoformat(deadline_utc)
    if deadline.tzinfo is None:raise ValueError('截止时间必须带时区')
    def check():
        if datetime.now(timezone.utc)>=deadline:raise TimeoutError('已到批次截止时间')
        if time.monotonic()-started>=wall_seconds:raise TimeoutError('比较总墙钟上限')
    check()
    output=Path(output)
    if output.exists() or output.with_suffix('.protocol.json').exists():raise FileExistsError(output)
    frames=load_frames(manifest);model=load_model(checkpoint,manifest,device) if keypoints=='predicted' else None
    protocol=dict(schema='precision-two-route-development/v1',manifest_sha256=digest(manifest),
        checkpoint_sha256=digest(checkpoint) if model is not None else None,keypoints=keypoints,split='development',
        geometry=asdict(GeometrySpec()),pnp=asdict(PnPSpec()),
        selection='frozen thresholds; no GT/future in solvers',hold_seconds=1.5,window_frames=4,window_seconds=1.5,
        depth_source='eroded RGB foreground only',reprojection='current observed correspondences, whole-yaw-aligned for temporal methods; null if none',
        orientation='four whole-object yaw symmetries',wall_seconds=wall_seconds,deadline_utc=deadline_utc,
        multiview_methods={'multiview-rgb':'2D only','multiview':'2D plus eroded interior depth consistency check'},
        source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')})
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output.with_suffix('.protocol.json'),protocol)
    check();result=evaluate_frames(frames,model,routes=True,wall_seconds=wall_seconds-(time.monotonic()-started),stop_check=check)
    result['protocol']=protocol;result['total_elapsed_s']=time.monotonic()-started
    check();write_json(output,result);print(json.dumps({m:g['all'] for m,g in result['groups'].items()}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--keypoints',choices=['predicted','gt-diagnostic'],default='predicted')
    p.add_argument('--device',default='cuda');p.add_argument('--wall-seconds',type=int,default=600)
    p.add_argument('--deadline-utc',default='2026-09-10T02:00:00+00:00')
    compare_routes(**vars(p.parse_args()))
