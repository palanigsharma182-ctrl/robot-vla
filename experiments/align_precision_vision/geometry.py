"""已知物体关键点的 RGB-D 位姿与 Align 相对误差；不读取模拟器 GT。"""
from dataclasses import dataclass, field
import math
import numpy as np

from experiments.tcp_memory_control.geometry import pose, pose_delta

KEYPOINT_NAMES = tuple(f'corner_{i}' for i in range(8))
CUBE_POINTS = np.array([(x, y, z) for x in (-.02, .02)
                        for y in (-.02, .02) for z in (-.02, .02)])


def yaw_symmetries():
    values=[]
    for k in range(4):
        a=k*math.pi/2; t=np.eye(4)
        t[:3,:3]=[[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]]
        values.append(t)
    return tuple(values)


def corner_permutations():
    """直立方块四重对称；用于整组关键点监督，不能逐点各选一个对称。"""
    return np.array([np.linalg.norm((CUBE_POINTS@s[:3,:3].T)[:,None]-CUBE_POINTS[None],axis=-1).argmin(1)
                     for s in yaw_symmetries()])


@dataclass(frozen=True)
class PoseMeasurement:
    episode: str
    target: str
    timestamp_s: float
    base_from_object: np.ndarray | None
    reason: str
    source: str = 'precision-rgbd-keypoints/v1'
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.episode or not self.target or not math.isfinite(self.timestamp_s):
            raise ValueError('测量必须有 episode、目标身份和有限时间')
        if (self.reason == 'valid') != (self.base_from_object is not None):
            raise ValueError('位姿与有效状态矛盾')
        if self.base_from_object is not None:
            value=pose(self.base_from_object);value.setflags(write=False)
            object.__setattr__(self,'base_from_object',value)


@dataclass(frozen=True)
class AlignCondition:
    features: tuple[float, ...]  # dx/dy/dz / 5cm，rotvec / 0.5rad，valid，age / max_age
    source: str
    reason: str
    symmetry_index: int | None = None

    def __post_init__(self):
        if len(self.features)!=8 or not np.isfinite(self.features).all():
            raise ValueError('条件必须是有限8维')
        if self.features[6] not in (0.,1.) or not 0<=self.features[7]<=1:
            raise ValueError('有效性或年龄范围错误')
        if (self.reason=='valid') != bool(self.features[6]):
            raise ValueError('原因与有效性矛盾')


@dataclass(frozen=True)
class GeometrySpec:
    visibility_threshold: float = .8
    min_depth_m: float = .05
    max_depth_m: float = 2.
    max_fit_rms_m: float = .002
    max_pair_error_m: float = .004
    max_age_s: float = .05
    max_sensor_skew_s: float = .025

    def __post_init__(self):
        vals=tuple(self.__dict__.values())
        if not all(math.isfinite(v) and v>0 for v in vals):raise ValueError('阈值必须有限且为正')
        if self.visibility_threshold>1 or self.min_depth_m>=self.max_depth_m:
            raise ValueError('可见性或深度范围无效')


def measure_keypoints(pixel_uv, visibility, depth_m, intrinsic, base_from_camera_cv,
                      *, episode, target, timestamp_s, object_points=CUBE_POINTS, spec=GeometrySpec()):
    """UV 为原始深度图的零基像素中心坐标；深度是 OpenCV 光轴 Z，单位米。"""
    uv=np.asarray(pixel_uv,dtype=float);vis=np.asarray(visibility,dtype=float)
    model=np.asarray(object_points,dtype=float);depth=np.asarray(depth_m)
    k=np.asarray(intrinsic,dtype=float);camera=pose(base_from_camera_cv)
    if (model.ndim!=2 or model.shape[1]!=3 or uv.shape!=(len(model),2) or vis.shape!=(len(model),)
        or depth.ndim!=2 or not np.isfinite(model).all() or not np.isfinite(vis).all()
        or np.any((vis<0)|(vis>1))):raise ValueError('关键点、可见性或深度 shape/数值错误')
    if (k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0
        or not np.allclose(k[2],[0,0,1]) or abs(np.linalg.det(k))<1e-12):
        raise ValueError('相机内参无效')
    def result(value,reason,**info):
        return PoseMeasurement(episode,target,timestamp_s,value,reason,diagnostics=info)
    ids=[];points=[]
    for i,(u,v) in enumerate(uv):
        if vis[i]<spec.visibility_threshold:continue
        if not np.isfinite([u,v]).all() or not (0<=u<=depth.shape[1]-1 and 0<=v<=depth.shape[0]-1):
            continue
        # 不从邻近背景或缺失深度补造点；表面边缘造成的错误由刚体一致性检查拒绝。
        z=float(depth[int(np.floor(v+.5)),int(np.floor(u+.5))])
        if not math.isfinite(z) or not spec.min_depth_m<=z<=spec.max_depth_m:continue
        ray=np.linalg.solve(k,[u,v,1.]);point=ray*z
        ids.append(i);points.append(camera[:3,:3]@point+camera[:3,3])
    if len(ids)<3:return result(None,'insufficient_visible_depth',keypoints=len(ids))
    a=model[ids];b=np.asarray(points)
    if np.linalg.matrix_rank(a-a.mean(0),tol=1e-6)<2 or np.linalg.matrix_rank(b-b.mean(0),tol=1e-6)<2:
        return result(None,'degenerate_keypoints',keypoints=len(ids))
    pair_error=float(np.max(np.abs(np.linalg.norm(a[:,None]-a[None],axis=-1)-np.linalg.norm(b[:,None]-b[None],axis=-1))))
    u,_,vt=np.linalg.svd((a-a.mean(0)).T@(b-b.mean(0)))
    correction=np.eye(3);correction[2,2]=np.linalg.det(vt.T@u.T)
    rotation=vt.T@correction@u.T;translation=b.mean(0)-rotation@a.mean(0)
    rms=float(np.sqrt(np.mean(np.sum((a@rotation.T+translation-b)**2,axis=1))))
    info=dict(keypoints=len(ids),fit_rms_m=rms,pair_error_m=pair_error,uncertainty='not calibrated')
    if rms>spec.max_fit_rms_m or pair_error>spec.max_pair_error_m:
        return result(None,'rigid_fit_rejected',**info)
    transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=translation
    return result(transform,'valid',**info)


def align_condition(measurement, base_from_tcp, object_from_pregrasp, *, episode, target,
                    now_s, tcp_timestamp_s, spec=GeometrySpec(), symmetries=None):
    """技能目标来自固定抓取变换；相对误差在当前 TCP 轴向，缺测不伪装成零误差。"""
    tcp=pose(base_from_tcp);grasp=pose(object_from_pregrasp)
    if not np.isfinite([now_s,tcp_timestamp_s]).all():raise ValueError('控制时间非有限')
    age=now_s-measurement.timestamp_s
    reason=measurement.reason
    if measurement.episode!=episode:reason='episode_mismatch'
    elif measurement.target!=target:reason='target_mismatch'
    elif age < -1e-9 or tcp_timestamp_s>now_s+1e-9:reason='future_observation'
    elif age>spec.max_age_s or now_s-tcp_timestamp_s>spec.max_age_s:reason='stale_observation'
    elif abs(tcp_timestamp_s-measurement.timestamp_s)>spec.max_sensor_skew_s:reason='sensor_skew'
    normalized_age=float(np.clip(age/spec.max_age_s,0,1))
    if reason!='valid':
        return AlignCondition((0.,)*6+(0.,normalized_age),measurement.source,reason)
    goal=measurement.base_from_object@grasp
    choices=(np.eye(4),) if symmetries is None else tuple(symmetries)
    if not choices:raise ValueError('目标对称集合不能为空')
    errors=[]
    for s in choices:
        s=pose(s)
        if not np.allclose(s[:3,3],0):raise ValueError('只允许绕目标原点的旋转对称')
        errors.append(pose_delta(tcp,goal@s,tcp))
    best=int(np.argmin([np.linalg.norm(e[3:]) for e in errors]))
    features=errors[best]/np.array([.05]*3+[.5]*3)
    return AlignCondition(tuple(features)+(1.,normalized_age),measurement.source,'valid',best)


def oracle_measurement(base_from_object, *, episode, target, timestamp_s):
    """只能由独立 Oracle 对照调用；measured 模式拒绝此来源。"""
    return PoseMeasurement(episode,target,timestamp_s,base_from_object,'valid','oracle-diagnostic/v1')
