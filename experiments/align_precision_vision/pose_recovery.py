"""4cm方块PnP；深度仅从RGB前景腐蚀后的内部区域读取。"""
from dataclasses import dataclass, asdict
import math
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from experiments.align_precision_vision.geometry import CUBE_POINTS, GeometrySpec, PoseMeasurement, measure_keypoints
from experiments.tcp_memory_control.geometry import pose

METHODS=('pnp-only','pnp-depth-refine','old-corner-depth')
SOURCES={'pnp-only':'precision-rgb-keypoints-pnp/v1',
         'pnp-depth-refine':'precision-rgbd-keypoints-pnp-refine/v1',
         'old-corner-depth':'precision-rgbd-keypoints/v1'}


@dataclass(frozen=True)
class PnPSpec:
    ransac_threshold_px: float = 3.
    max_reprojection_rms_px: float = 3.
    min_inlier_fraction: float = .6
    iterations: int = 100
    seed: int = 17
    erosion_radius_px: int = 2
    min_interior_points: int = 32
    max_interior_points: int = 256
    depth_median_limit_m: float = .005
    depth_p90_limit_m: float = .010
    refinement_translation_bound_m: float = .010
    refinement_rotation_bound_rad: float = .15
    refinement_max_evaluations: int = 40

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):raise ValueError('PnP参数必须有限')
        if any(v<=0 for k,v in asdict(self).items() if k!='seed'):raise ValueError('PnP阈值必须为正')
        if self.min_inlier_fraction>1 or self.min_interior_points>self.max_interior_points:
            raise ValueError('PnP点数/比例错误')
        for name in ('iterations','seed','erosion_radius_px','min_interior_points','max_interior_points','refinement_max_evaluations'):
            if not isinstance(getattr(self,name),int):raise ValueError('计数参数必须为整数')


def project_cube(camera_from_object, intrinsic):
    xyz=CUBE_POINTS@camera_from_object[:3,:3].T+camera_from_object[:3,3]
    projected=xyz@intrinsic.T
    with np.errstate(divide='ignore',invalid='ignore'):
        return projected[:,:2]/projected[:,2:3],xyz[:,2]


def visible_indices(pixel_uv,visibility,image_shape,threshold):
    uv=np.asarray(pixel_uv,dtype=np.float64);vis=np.asarray(visibility,dtype=np.float64)
    if uv.shape!=(8,2) or vis.shape!=(8,) or not np.isfinite(vis).all() or np.any((vis<0)|(vis>1)):
        raise ValueError('PnP输入必须为8个2D角点和[0,1]可见性')
    h,w=image_shape
    if min(h,w)<=0:raise ValueError('图像尺寸错误')
    selected=(vis>=threshold)&np.isfinite(uv).all(1)
    selected&=(uv[:,0]>=0)&(uv[:,0]<=w-1)&(uv[:,1]>=0)&(uv[:,1]<=h-1)
    return uv,np.flatnonzero(selected)


def solve_pnp(uv,ids,intrinsic,config):
    """四点以上；平面点用IPPE保留候选，非平面用SQPnP，另加RANSAC。

    只按观测重投影与正深度选解，不读取GT或为每个角点单独选对称。
    """
    if len(ids)<4:return None,dict(reason='insufficient_visible_pnp',keypoints=len(ids))
    points=np.ascontiguousarray(CUBE_POINTS[ids]);pixels=np.ascontiguousarray(uv[ids])
    rank=np.linalg.matrix_rank(points-points.mean(0),tol=1e-8)
    if rank<2 or np.linalg.matrix_rank(pixels-pixels.mean(0),tol=1e-4)<2:
        return None,dict(reason='degenerate_pnp',keypoints=len(ids))
    candidates=[];errors=[]
    try:
        flag=cv2.SOLVEPNP_IPPE if rank==2 else cv2.SOLVEPNP_SQPNP
        result=cv2.solvePnPGeneric(points,pixels,intrinsic,None,flags=flag)
        if result[0]:candidates.extend(zip(result[1],result[2]))
    except cv2.error as exc:errors.append(str(exc).splitlines()[-1])
    try:
        cv2.setRNGSeed(config.seed)
        ok,rvec,tvec,inliers=cv2.solvePnPRansac(points,pixels,intrinsic,None,
            iterationsCount=config.iterations,reprojectionError=config.ransac_threshold_px,
            confidence=.999,flags=cv2.SOLVEPNP_EPNP)
        if ok and inliers is not None and len(inliers)>=4:
            chosen=inliers.reshape(-1)
            rvec,tvec=cv2.solvePnPRefineLM(points[chosen],pixels[chosen],intrinsic,None,rvec,tvec)
            candidates.append((rvec,tvec))
    except cv2.error as exc:errors.append(str(exc).splitlines()[-1])
    scored=[]
    for rvec,tvec in candidates:
        transform=np.eye(4);transform[:3,:3]=cv2.Rodrigues(rvec)[0];transform[:3,3]=np.asarray(tvec).reshape(3)
        pred,z=project_cube(transform,intrinsic)
        if not np.isfinite(transform).all() or np.min(z)<=.001:continue
        residual=np.linalg.norm(pred[ids]-pixels,axis=1)
        inliers=residual<=config.ransac_threshold_px
        if inliers.sum()<max(4,math.ceil(len(ids)*config.min_inlier_fraction)):continue
        rms=float(np.sqrt(np.mean(residual[inliers]**2)))
        if rms>config.max_reprojection_rms_px:continue
        scored.append((int(inliers.sum()),rms,transform,ids[inliers],residual))
    if not scored:return None,dict(reason='pnp_no_valid_solution',keypoints=len(ids),solver_errors=errors)
    scored.sort(key=lambda x:(-x[0],x[1]));best=scored[0]
    info=dict(reason='valid',keypoints=len(ids),pnp_inliers=best[3].tolist(),candidate_count=len(scored),
              pnp_reprojection_inlier_rms_px=best[1],reprojection_all_rms_px=float(np.sqrt(np.mean(best[4]**2))),
              planar=bool(rank==2),selection='max inliers then minimum inlier reprojection; no GT',solver_errors=errors)
    if len(scored)>1:info['second_candidate_rms_gap_px']=float(scored[1][1]-best[1])
    return best[2],info


def interior_mask(rgb,camera_from_object,intrinsic,config):
    """当前红色目标的RGB前景∩PnP投影轮廓，再腐蚀；不接受GT mask参数。"""
    image=np.asarray(rgb)
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:raise ValueError('前景需要原始RGB uint8')
    red,green,blue=np.moveaxis(image.astype(np.float32),-1,0)
    foreground=(red>=60)&(red>1.5*green)&(red>1.5*blue)
    uv,z=project_cube(camera_from_object,intrinsic)
    silhouette=np.zeros(image.shape[:2],np.uint8)
    if np.min(z)>.001 and np.isfinite(uv).all():
        hull=cv2.convexHull(np.clip(np.rint(uv),-100000,100000).astype(np.int32))
        cv2.fillConvexPoly(silhouette,hull,1)
    kernel=np.ones((2*config.erosion_radius_px+1,)*2,np.uint8)
    return cv2.erode((foreground&(silhouette!=0)).astype(np.uint8),kernel,
                     borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)


def cube_ray_depth(camera_from_object,rays):
    """相机射线与已知cube表面的最近正交点，返回光轴Z深度。"""
    rotation=camera_from_object[:3,:3];origin=-rotation.T@camera_from_object[:3,3]
    direction=rays@rotation
    parallel=np.abs(direction)<1e-12
    safe=np.where(parallel,1.,direction)
    first=(-.02-origin)/safe;second=(.02-origin)/safe
    lo=np.where(parallel,-np.inf,np.minimum(first,second))
    hi=np.where(parallel,np.inf,np.maximum(first,second))
    outside=np.any(parallel&(np.abs(origin)>.02),axis=1)
    near=lo.max(1);far=hi.min(1);hit=(far>=np.maximum(near,0))&~outside
    depth=np.where(near>0,near,far)
    return np.where(hit,depth,np.nan)


def refine_depth(initial,uv,ids,intrinsic,rgb,depth_m,geometry,config):
    """固定腐蚀内部采样集，优化6D局部增量；无有效深度时显式回退PnP。"""
    if rgb is None or depth_m is None:
        return initial,dict(depth_status='unavailable_fallback_pnp',refined=False)
    mask=interior_mask(rgb,initial,intrinsic,config)
    depth=np.asarray(depth_m)
    if depth.shape!=mask.shape or not np.issubdtype(depth.dtype,np.floating):raise ValueError('深度必须为同分辨率浮点米')
    ys,xs=np.nonzero(mask)
    # 只在eroded foreground interior取depth，绝不回退到角点或邻接轮廓。
    observed=depth[ys,xs]
    valid=np.isfinite(observed)&(observed>=geometry.min_depth_m)&(observed<=geometry.max_depth_m)
    ys,xs,observed=ys[valid],xs[valid],observed[valid]
    if len(xs)>config.max_interior_points:
        selected=np.linspace(0,len(xs)-1,config.max_interior_points,dtype=int)
        ys,xs,observed=ys[selected],xs[selected],observed[selected]
    info=dict(interior_points=len(xs),interior_mask_pixels=int(mask.sum()),refined=False,
              foreground_source='rgb-red-threshold-and-pnp-silhouette/v1',erosion_radius_px=config.erosion_radius_px)
    if len(xs)<config.min_interior_points:
        return initial,dict(info,depth_status='insufficient_interior_fallback_pnp')
    rays=np.c_[xs,ys,np.ones(len(xs))]@np.linalg.inv(intrinsic).T
    predicted=cube_ray_depth(initial,rays)
    hit=np.isfinite(predicted)
    rays,observed=rays[hit],observed[hit]
    info['interior_points']=len(rays)
    if len(rays)<config.min_interior_points:
        return initial,dict(info,depth_status='insufficient_intersection_fallback_pnp')
    before=np.abs(cube_ray_depth(initial,rays)-observed)
    info.update(depth_before_median_m=float(np.median(before)),depth_before_p90_m=float(np.quantile(before,.9)))
    def updated(delta):
        value=initial.copy();value[:3,:3]=Rotation.from_rotvec(delta[:3]).as_matrix()@initial[:3,:3]
        value[:3,3]+=delta[3:];return value
    def residual(delta):
        value=updated(delta);pixels,_=project_cube(value,intrinsic)
        reproj=(pixels[ids]-uv[ids]).ravel()/(1.5*np.sqrt(len(ids)))
        predicted=cube_ray_depth(value,rays)
        difference=np.where(np.isfinite(predicted),predicted-observed,.05)
        return np.r_[reproj,difference/(.002*np.sqrt(len(rays)))]
    bound=np.array([config.refinement_rotation_bound_rad]*3+[config.refinement_translation_bound_m]*3)
    try:
        opt=least_squares(residual,np.zeros(6),bounds=(-bound,bound),loss='soft_l1',
                          max_nfev=config.refinement_max_evaluations)
    except (ValueError,np.linalg.LinAlgError) as exc:
        info['refinement_error']=f'{type(exc).__name__}: {exc}'
        consistent=np.median(before)<=config.depth_median_limit_m and np.quantile(before,.9)<=config.depth_p90_limit_m
        if consistent:return initial,dict(info,depth_status='optimizer_failed_consistent_fallback_pnp')
        return None,dict(info,depth_status='depth_refinement_failed')
    candidate=updated(opt.x);pixels,z=project_cube(candidate,intrinsic)
    after=np.abs(cube_ray_depth(candidate,rays)-observed)
    initial_pixels,_=project_cube(initial,intrinsic)
    before_rms=float(np.sqrt(np.mean(np.sum((initial_pixels[ids]-uv[ids])**2,axis=1))))
    after_rms=float(np.sqrt(np.mean(np.sum((pixels[ids]-uv[ids])**2,axis=1))))
    acceptable=(opt.success and np.isfinite(after).all() and np.min(z)>.001
        and np.median(after)<=config.depth_median_limit_m and np.quantile(after,.9)<=config.depth_p90_limit_m
        and after_rms<=min(config.max_reprojection_rms_px,before_rms+.5)
        and np.linalg.norm(residual(opt.x))<=np.linalg.norm(residual(np.zeros(6)))+1e-8)
    info.update(optimizer_status=int(opt.status),optimizer_nfev=int(opt.nfev),
        depth_after_median_m=float(np.median(after)) if np.isfinite(after).all() else None,
        depth_after_p90_m=float(np.quantile(after,.9)) if np.isfinite(after).all() else None,
        refined_reprojection_rms_px=after_rms)
    if acceptable:return candidate,dict(info,depth_status='refined',refined=True)
    consistent=np.median(before)<=config.depth_median_limit_m and np.quantile(before,.9)<=config.depth_p90_limit_m
    if consistent:return initial,dict(info,depth_status='consistent_unrefined_pnp')
    return None,dict(info,depth_status='depth_inconsistent')


def recover_pose(pixel_uv,visibility,intrinsic,base_from_camera_cv,*,image_shape,episode,target,timestamp_s,
                 method='pnp-only',depth_m=None,rgb=None,spec=GeometrySpec(),pnp_spec=PnPSpec()):
    if method not in METHODS:raise ValueError('未知位姿恢复方法')
    if method=='old-corner-depth':
        if depth_m is None:raise ValueError('旧对照需要depth')
        return measure_keypoints(pixel_uv,visibility,depth_m,intrinsic,base_from_camera_cv,
                                 episode=episode,target=target,timestamp_s=timestamp_s,spec=spec)
    camera=pose(base_from_camera_cv);k=np.asarray(intrinsic,dtype=float)
    if (k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0
        or not np.allclose(k[2],[0,0,1]) or not np.allclose([k[0,1],k[1,0]],[0,0])):
        raise ValueError('PnP需要已去畸变、零skew OpenCV内参')
    uv,ids=visible_indices(pixel_uv,visibility,image_shape,spec.visibility_threshold)
    value,info=solve_pnp(uv,ids,k,pnp_spec)
    info.update(method=method,visibility_threshold=spec.visibility_threshold,distortion='rectified-zero')
    if value is None:
        return PoseMeasurement(episode,target,timestamp_s,None,info['reason'],SOURCES[method],info)
    if method=='pnp-depth-refine':
        value,depth_info=refine_depth(value,uv,np.array(info['pnp_inliers']),k,rgb,depth_m,spec,pnp_spec)
        info.update(depth_info)
        if value is None:
            info['reason']=depth_info['depth_status']
            return PoseMeasurement(episode,target,timestamp_s,None,info['reason'],SOURCES[method],info)
    projected,_=project_cube(value,k)
    info['reprojection_all_rms_px']=float(np.sqrt(np.mean(np.sum((projected[ids]-uv[ids])**2,axis=1))))
    return PoseMeasurement(episode,target,timestamp_s,camera@value,'valid',SOURCES[method],info)
