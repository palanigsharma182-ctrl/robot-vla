"""只读GT角点深度诊断；真实深度与privileged替换结果明确分开。"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.align_precision_vision.geometry import measure_keypoints
from experiments.align_precision_vision.train import digest


def diagnose(manifest, output):
    manifest=Path(manifest);output=Path(output)
    if output.exists():raise FileExistsError(output)
    data=json.loads(manifest.read_text())
    if data.get('schema')!='align-precision-keypoints/v1':raise ValueError('manifest schema错误')
    counts=dict(visible=0,same_object=0,over3mm=0,outside_depth_range=0)
    errors=[];eligible=0;upper=0;groups={}
    for row in data['records']:
        if row['split'] not in ('train','development'):raise ValueError('不消费final test')
        path=(manifest.parent/row['audit_file']).resolve()
        if not path.is_relative_to(manifest.parent.resolve()) or digest(path)!=row['audit_sha256']:
            raise ValueError('audit路径/hash错误')
        with np.load(path,allow_pickle=False) as x:d={k:x[k] for k in x.files}
        uv=d['pixel_uv'];vis=d['visible'];depth=d['depth_m'];oracle_depth=depth.copy()
        group=groups.setdefault(row['camera'],dict(visible=0,wrong_actor=0,over3mm=0))
        for i in np.flatnonzero(vis):
            px,py=np.floor(uv[i]+.5).astype(int)
            actual=float(depth[py,px]);expected=float(d['expected_depth_m'][i]);error=abs(actual-expected)*1000
            on_object=bool(d['object_mask'][py,px])
            counts['visible']+=1;counts['same_object']+=int(on_object);counts['over3mm']+=int(error>3)
            counts['outside_depth_range']+=int(not .05<=expected<=2)
            group['visible']+=1;group['wrong_actor']+=int(not on_object);group['over3mm']+=int(error>3)
            errors.append(error);oracle_depth[py,px]=expected
        if vis.sum()>=3:
            eligible+=1
            measurement=measure_keypoints(uv,vis.astype(float),oracle_depth,d['intrinsic'],d['base_from_camera_cv'],
                episode=row['scene'],target='cube',timestamp_s=float(d['timestamp_s']))
            upper+=int(measurement.reason=='valid')
    result=dict(mode='diagnostic-only privileged expected-depth substitution; never deployment',
        manifest_sha256=digest(manifest),frames=len(data['records']),counts=counts,by_camera=groups,
        visible_corner_abs_depth_error_mm_quantiles=np.quantile(errors,[.5,.9,1]).tolist() if errors else None,
        frames_with_at_least_3_visible=eligible,substituted_depth_valid=upper)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);diagnose(**vars(p.parse_args()))
