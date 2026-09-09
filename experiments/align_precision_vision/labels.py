"""仿真专用角点监督：投影与可见性分开，不进入部署测量路径。"""
import numpy as np
from experiments.align_precision_vision.geometry import CUBE_POINTS
from experiments.tcp_memory_control.geometry import pose


def project(points, intrinsic, camera_from_object):
    transform = pose(camera_from_object)
    xyz = np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]
    image = xyz @ np.asarray(intrinsic).T
    uv = np.full((len(xyz), 2), np.nan)
    forward = xyz[:, 2] > 0
    uv[forward] = image[forward, :2] / image[forward, 2:3]
    return uv, xyz[:, 2]


def corner_labels(intrinsic, camera_from_object, depth_m, object_mask, *, tolerance_m=.003):
    """用朝向、轮廓邻域的同物体深度及邻接面探针核验遮挡。

    角点处深度离散不稳定，允许3x3轮廓邻域；再用距角点1mm的面内探针
    检查真实表面。此为合成监督近似，原始证据保留，不能当实测定位器。
    """
    depth = np.asarray(depth_m)
    mask = np.asarray(object_mask)
    if depth.ndim != 2 or mask.shape != depth.shape or mask.dtype != np.bool_:
        raise ValueError('深度与物体mask须同分辨率')
    camera = pose(camera_from_object)
    origin = np.linalg.inv(camera)[:3, 3]
    uv, z = project(CUBE_POINTS, intrinsic, camera)
    h, w = depth.shape
    projected = (z > 0) & np.isfinite(uv).all(1)
    projected &= (uv[:, 0] >= 0) & (uv[:, 0] <= w-1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h-1)

    def supported(pixel, expected, radius):
        if not np.isfinite(pixel).all() or expected <= 0:
            return False
        x, y = np.floor(pixel+.5).astype(int)
        if not 0 <= x < w or not 0 <= y < h:
            return False
        yy = slice(max(0,y-radius), min(h,y+radius+1))
        xx = slice(max(0,x-radius), min(w,x+radius+1))
        values = depth[yy,xx]
        return bool(np.any(mask[yy,xx] & np.isfinite(values) & (values > 0)
                           & (np.abs(values-expected) <= tolerance_m)))

    visible = np.zeros(8, bool)
    for i, corner in enumerate(CUBE_POINTS):
        if not projected[i] or not supported(uv[i], z[i], 1):
            continue
        for axis in range(3):
            sign = np.sign(corner[axis])
            if sign*(origin[axis]-corner[axis]) <= 0:
                continue
            probe = corner * .95
            probe[axis] = corner[axis]
            p, d = project(probe[None], intrinsic, camera)
            if supported(p[0], d[0], 0):
                visible[i] = True
                break
    labels = uv.astype(np.float32)
    labels[~visible] = np.nan
    return dict(pixel_uv=labels, visible=visible, projected_uv=uv.astype(np.float32),
                projected=projected, expected_depth_m=z.astype(np.float32))
