"""世界Memory只读；每个chunk的控制信息在实际TCP固定轴向中表达。"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA = 'tcp-anchor-command-delta-rotvec-gripper/v1'
FEATURE_SCHEMA = 'target-relative-to-actual-tcp-memory/12d/v1'


def pose(value):
    t = np.asarray(value, dtype=np.float64)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
        or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-7)
        or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5)):
        raise ValueError('必须提供有效base_from_tcp SE(3)')
    return t.copy()


def relative_features(snapshot, base_from_tcp, offset_base=(0., 0., .08)):
    """位置转换到TCP；协方差只旋转；不刷新Memory时间或就地改写快照。

    TCP位姿作为确定性输入，本候选不声称已传播机器人的位姿估计不确定性。
    offset在base中定义；不是TCP自身Z轴。
    """
    t = pose(base_from_tcp)
    if not snapshot['available']:
        return np.zeros(12, dtype=np.float32)
    f = np.asarray(snapshot['features'], dtype=np.float64).copy()
    offset = np.asarray(offset_base, dtype=np.float64)
    if f.shape != (12,) or offset.shape != (3,) or not np.isfinite(np.r_[f, offset]).all():
        raise ValueError('Memory特征/任务偏移非法')
    r = t[:3, :3]
    f[:3] = r.T @ (f[:3] + offset - t[:3, 3])
    cov = np.zeros((3, 3));indices = np.triu_indices(3)
    cov[indices] = f[3:9];cov = cov + cov.T - np.diag(cov.diagonal())
    if np.linalg.eigvalsh(cov).min() < -1e-7:
        raise ValueError('Memory协方差非半正定')
    f[3:9] = (r.T @ cov @ r)[indices]
    return f.astype(np.float32)


def pose_delta(previous, target, anchor):
    """相邻commanded目标差分，全部在chunk开始时固定的TCP轴向表达。"""
    previous, target, anchor = pose(previous), pose(target), pose(anchor)
    r = anchor[:3, :3]
    return np.r_[r.T @ (target[:3, 3] - previous[:3, 3]),
        Rotation.from_matrix(r.T @ target[:3, :3] @ previous[:3, :3].T @ r).as_rotvec()]


def apply_delta(previous, delta, anchor):
    previous, anchor = pose(previous), pose(anchor)
    d = np.asarray(delta, dtype=np.float64)
    if d.shape != (6,) or not np.isfinite(d).all():
        raise ValueError('TCP增量必须为有限6维')
    r = anchor[:3, :3];target = previous.copy()
    target[:3, 3] += r @ d[:3]
    target[:3, :3] = r @ Rotation.from_rotvec(d[3:]).as_matrix() @ r.T @ previous[:3, :3]
    return pose(target)


@dataclass(frozen=True)
class TCPActionSpec:
    translation_limit_m: float = .01
    rotation_component_limit_rad: float = .10
    horizon: int = 16
    execute_steps: int = 4
    schema: str = SCHEMA

    def __post_init__(self):
        if (not np.isfinite([self.translation_limit_m, self.rotation_component_limit_rad]).all()
            or min(self.translation_limit_m, self.rotation_component_limit_rad) <= 0
            or not 1 <= self.execute_steps <= self.horizon or self.schema != SCHEMA):
            raise ValueError('TCP动作合同非法')

    @property
    def limits(self):
        return np.array([self.translation_limit_m]*3 + [self.rotation_component_limit_rad]*3)

    def normalize(self, physical):
        a = np.asarray(physical, dtype=np.float64)
        if a.ndim < 1 or a.shape[-1] != 7 or not np.isfinite(a).all():
            raise ValueError('TCP动作必须为有限[...,7]')
        if np.any(a[..., 6] < 0) or np.any(a[..., 6] > 1):
            raise ValueError('夹爪目标开口必须在[0,1]')
        result = np.concatenate([a[..., :6] / self.limits, 2*a[..., 6:7]-1], axis=-1)
        if np.any(np.abs(result) > 1+1e-6):
            raise ValueError('TCP标签超过预定范围，禁止静默clamp')
        return result.astype(np.float32)

    def denormalize(self, normalized):
        a = np.asarray(normalized, dtype=np.float64)
        if a.ndim < 1 or a.shape[-1] != 7 or not np.isfinite(a).all() or np.any(np.abs(a)>1+1e-6):
            raise ValueError('归一化TCP动作必须为有限[-1,1]的[...,7]')
        return np.concatenate([a[..., :6]*self.limits, (a[..., 6:7]+1)/2], axis=-1).astype(np.float32)
