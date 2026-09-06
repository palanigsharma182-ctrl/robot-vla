"""RGB-D已知立方体中心候选：不使用固定高度、分割GT或物体真实位姿。"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path

import cv2
import numpy as np

HALF_SIZE_M = .02
PROVIDER_ID = 'front-rgbd-known-cube-three-planes-m0/v1:' + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass
class Estimate:
    position: np.ndarray | None = None
    covariance: np.ndarray | None = None
    reason: str = 'not_evaluated'
    diagnostics: dict = field(default_factory=dict)

    @property
    def valid(self):
        return self.position is not None and self.reason == 'accepted_candidate'


def target_mask(rgb):
    """冻结的红色单实例规则；第二大红区域过大时拒绝身份歧义。"""
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:
        raise ValueError('RGB必须为uint8[H,W,3]')
    red,green,blue = np.moveaxis(rgb.astype(np.float64),-1,0)
    mask = ((red>80)&(red>2.5*green)&(red>2.5*blue)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask,8)
    if count<=1:
        return np.zeros(mask.shape,dtype=bool), 'target_missing'
    order = np.argsort(stats[1:,cv2.CC_STAT_AREA])[::-1]+1
    if len(order)>1 and stats[order[1],cv2.CC_STAT_AREA]>.3*stats[order[0],cv2.CC_STAT_AREA]:
        return np.zeros(mask.shape,dtype=bool), 'target_ambiguous'
    selected = (labels==order[0]).astype(np.uint8)
    selected = cv2.erode(selected,np.ones((3,3),np.uint8),iterations=1).astype(bool)
    return selected, 'ok' if selected.sum()>=80 else 'target_too_small'


def backproject(depth_mm, mask, intrinsic, base_from_camera_cv):
    """深度按光轴Z解释，毫米转米，再以同帧外参变换到机器人基坐标系。"""
    depth = np.asarray(depth_mm)
    if depth.shape!=mask.shape or intrinsic.shape!=(3,3) or base_from_camera_cv.shape!=(4,4):
        raise ValueError('RGB-D/标定shape不匹配')
    if not np.isfinite(intrinsic).all() or not np.isfinite(base_from_camera_cv).all():
        raise ValueError('标定非有限')
    if intrinsic[0,0]<=0 or intrinsic[1,1]<=0:
        raise ValueError('焦距必须为正')
    if not np.allclose(base_from_camera_cv[3],[0,0,0,1],atol=1e-8):
        raise ValueError('齐次外参末行无效')
    rotation = base_from_camera_cv[:3,:3]
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(rotation),1,atol=1e-5):
        raise ValueError('外参必须是刚体变换')
    good = mask & np.isfinite(depth) & (depth>50) & (depth<2000)
    v,u = np.where(good)
    rays = np.c_[u,v,np.ones(len(u))]@np.linalg.inv(intrinsic).T
    camera_points = rays*(depth[v,u,None]*.001)
    return camera_points@rotation.T+base_from_camera_cv[:3,3]


def estimate_center(points, camera_origin, *, half_size=HALF_SIZE_M):
    """拟合三个独立可见面，以已知半边长求中心；不足三面时明确拒绝。"""
    points = np.asarray(points,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError('点云必须是有限[N,3]')
    info = dict(points=len(points),shape_prior='known 4cm cube',covariance_status='heuristic, not calibrated')
    if len(points)<120:
        return Estimate(reason='insufficient_depth_points',diagnostics=info)
    rng = np.random.default_rng(20260907)
    if len(points)>2500:
        points = points[rng.choice(len(points),2500,replace=False)]
    remaining = points.copy()
    normals, distances, residuals, counts = [], [], [], []
    for _ in range(3):
        if len(remaining)<40:
            break
        triples = remaining[rng.integers(0,len(remaining),size=(384,3))]
        ns = np.cross(triples[:,1]-triples[:,0],triples[:,2]-triples[:,0])
        lengths = np.linalg.norm(ns,axis=1)
        good = lengths>1e-7
        ns, triples = ns[good]/lengths[good,None], triples[good]
        if normals:
            good = (np.abs(ns@np.asarray(normals).T)<.20).all(axis=1)
            ns, triples = ns[good],triples[good]
        if len(ns)==0:
            break
        ds = np.sum(ns*triples[:,0],axis=1)
        errors = np.abs(remaining@ns.T-ds)
        best = int((errors<.0012).sum(axis=0).argmax())
        inlier = errors[:,best]<.0012
        if inlier.sum()<max(40,.04*len(points)):
            break
        face = remaining[inlier]
        centroid = face.mean(axis=0)
        _,_,vectors = np.linalg.svd(face-centroid,full_matrices=False)
        normal = vectors[-1]
        if normal@(camera_origin-centroid)<0:
            normal=-normal
        distance = float(normal@centroid)
        if normals and max(abs(normal@n) for n in normals)>.20:
            break
        normals.append(normal);distances.append(distance)
        residuals.append(float(np.sqrt(np.mean((face@normal-distance)**2))))
        counts.append(len(face))
        remaining = remaining[np.abs(remaining@normal-distance)>.0015]
    info.update(faces=len(normals),face_points=counts,plane_rms_m=residuals)
    if len(normals)!=3:
        return Estimate(reason='three_independent_faces_required',diagnostics=info)
    a = np.asarray(normals)
    if np.linalg.cond(a)>1.5:
        return Estimate(reason='ill_conditioned_planes',diagnostics=info)
    position = np.linalg.solve(a,np.asarray(distances)-half_size)
    # 独立面法向略有噪声；用最近正交基检查整片点云是否符合已知形状。
    u,_,vt = np.linalg.svd(a)
    basis = u@vt
    surface_errors = np.abs(np.max(np.abs((points-position)@basis.T),axis=1)-half_size)
    p90 = float(np.percentile(surface_errors,90))
    support = float(np.mean(surface_errors<.0025))
    info.update(surface_p90_m=p90,surface_support=support)
    if p90>.0025 or support<.9:
        return Estimate(reason='cube_shape_inconsistent',diagnostics=info)
    # 未经实物校准的保守工程占位；不能宣称统计置信度或正式provider资格。
    inverse = np.linalg.inv(a)
    sigma = max(.002,p90)
    covariance = inverse@np.diag(np.full(3,sigma**2))@inverse.T + np.eye(3)*.001**2
    return Estimate(position,covariance,'accepted_candidate',info)


def measure(rgb, depth_mm, intrinsic, base_from_camera_cv):
    mask,reason = target_mask(rgb)
    if reason!='ok':
        return Estimate(reason=reason,diagnostics=dict(mask_pixels=int(mask.sum()))),mask
    points = backproject(depth_mm,mask,intrinsic,base_from_camera_cv)
    estimate = estimate_center(points,base_from_camera_cv[:3,3])
    estimate.diagnostics['mask_pixels'] = int(mask.sum())
    if estimate.valid:
        camera_point = np.linalg.inv(base_from_camera_cv)@np.r_[estimate.position,1.]
        pixel = intrinsic@camera_point[:3]
        uv = pixel[:2]/pixel[2] if pixel[2]>0 else np.array([np.inf,np.inf])
        if not (np.isfinite(uv).all() and 0<=uv[0]<rgb.shape[1] and 0<=uv[1]<rgb.shape[0]):
            return Estimate(reason='estimated_center_out_of_view',diagnostics=estimate.diagnostics),mask
    return estimate,mask
