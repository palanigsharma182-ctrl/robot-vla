"""OpenCV 相机坐标到机器人 base frame 的显式平面几何。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from robot_vla.observation import invert_se3, validate_se3


def _intrinsic_matrix(value: np.ndarray) -> np.ndarray:
    intrinsic = np.asarray(value, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("intrinsic_cv 必须是有限 [3,3] 矩阵")
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise ValueError("intrinsic_cv 的 fx/fy 必须为正数")
    if not np.allclose(intrinsic[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-7):
        raise ValueError("intrinsic_cv 最后一行必须是 [0,0,1]")
    return intrinsic


def _image_size(image_size_hw: tuple[int, int]) -> tuple[int, int]:
    if len(image_size_hw) != 2:
        raise ValueError("image_size_hw 必须是 (height,width)")
    height, width = image_size_hw
    if (
        not isinstance(height, Integral)
        or isinstance(height, bool)
        or not isinstance(width, Integral)
        or isinstance(width, bool)
        or height <= 0
        or width <= 0
    ):
        raise ValueError("image_size_hw 必须包含两个正整数")
    return int(height), int(width)


def normalized_uv_to_pixel(
    normalized_uv: np.ndarray,
    image_size_hw: tuple[int, int],
) -> np.ndarray:
    """把像素中心归一化坐标还原为 OpenCV intrinsic 使用的 ``(u,v)``。

    项目约定 ``normalized = ((u + 0.5) / W, (v + 0.5) / H)``，与现有
    Qwen spatial probe 的像素中心语义一致。
    """

    uv = np.asarray(normalized_uv, dtype=np.float64)
    if uv.shape != (2,) or not np.isfinite(uv).all():
        raise ValueError("normalized_uv 必须是有限 [2]")
    if np.any(uv < 0.0) or np.any(uv > 1.0):
        raise ValueError("normalized_uv 必须位于 [0,1]")
    height, width = _image_size(image_size_hw)
    return np.asarray(
        (uv[0] * width - 0.5, uv[1] * height - 0.5),
        dtype=np.float64,
    )


def pixel_to_normalized_uv(
    pixel_uv: np.ndarray,
    image_size_hw: tuple[int, int],
) -> np.ndarray:
    pixel = np.asarray(pixel_uv, dtype=np.float64)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError("pixel_uv 必须是有限 [2]")
    height, width = _image_size(image_size_hw)
    return np.asarray(
        ((pixel[0] + 0.5) / width, (pixel[1] + 0.5) / height),
        dtype=np.float32,
    )


@dataclass(frozen=True)
class PlaneIntersection:
    point_base_m: np.ndarray
    ray_parameter: float


def pixel_ray_to_base_plane(
    pixel_uv: np.ndarray,
    intrinsic_cv: np.ndarray,
    base_from_camera_cv: np.ndarray,
    *,
    plane_normal_base: np.ndarray,
    plane_offset_base_m: float,
) -> PlaneIntersection:
    """把 OpenCV 像素射线与 base-frame 平面 ``n·X+d=0`` 求交。

    输入图像必须已经按 ``intrinsic_cv`` 对应的畸变模型校正；本函数不会静默忽略
    非零畸变参数。
    """

    pixel = np.asarray(pixel_uv, dtype=np.float64)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError("pixel_uv 必须是有限 [2]")
    intrinsic = _intrinsic_matrix(intrinsic_cv)
    transform = validate_se3(base_from_camera_cv, "base_from_camera_cv")
    normal = np.asarray(plane_normal_base, dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        raise ValueError("plane_normal_base 必须是有限 [3]")
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-12:
        raise ValueError("plane_normal_base 不能退化为零向量")
    if not math.isfinite(plane_offset_base_m):
        raise ValueError("plane_offset_base_m 必须是有限数值")
    normal = normal / normal_norm
    offset = float(plane_offset_base_m) / normal_norm

    direction_camera = np.linalg.solve(
        intrinsic,
        np.asarray((pixel[0], pixel[1], 1.0), dtype=np.float64),
    )
    origin_base = transform[:3, 3]
    direction_base = transform[:3, :3] @ direction_camera
    denominator = float(np.dot(normal, direction_base))
    if abs(denominator) <= 1e-12:
        raise ValueError("相机射线与目标平面平行")
    scale = -(float(np.dot(normal, origin_base)) + offset) / denominator
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("目标平面位于相机射线后方")
    point_base = origin_base + scale * direction_base
    if not np.isfinite(point_base).all():
        raise ValueError("平面反投影产生 NaN 或 Inf")
    return PlaneIntersection(
        point_base_m=point_base.astype(np.float32),
        ray_parameter=float(scale),
    )


def normalized_uv_to_base_z_plane(
    normalized_uv: np.ndarray,
    intrinsic_cv: np.ndarray,
    base_from_camera_cv: np.ndarray,
    image_size_hw: tuple[int, int],
    *,
    plane_base_z_m: float,
) -> PlaneIntersection:
    """把归一化像素射线与 ``base z = plane_base_z_m`` 求交。"""

    if not math.isfinite(plane_base_z_m):
        raise ValueError("plane_base_z_m 必须是有限数值")
    pixel = normalized_uv_to_pixel(normalized_uv, image_size_hw)
    return pixel_ray_to_base_plane(
        pixel,
        intrinsic_cv,
        base_from_camera_cv,
        plane_normal_base=np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        plane_offset_base_m=-float(plane_base_z_m),
    )


def project_base_point_to_normalized_uv(
    point_base_m: np.ndarray,
    intrinsic_cv: np.ndarray,
    base_from_camera_cv: np.ndarray,
    image_size_hw: tuple[int, int],
) -> np.ndarray:
    """把 base-frame 点投影到与模型 heatmap 一致的归一化像素坐标。"""

    point = np.asarray(point_base_m, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("point_base_m 必须是有限 [3]")
    intrinsic = _intrinsic_matrix(intrinsic_cv)
    camera_from_base = invert_se3(base_from_camera_cv, "base_from_camera_cv")
    point_camera = camera_from_base @ np.concatenate((point, np.ones(1, dtype=np.float64)))
    depth = float(point_camera[2])
    if not math.isfinite(depth) or depth <= 0.0:
        raise ValueError("base-frame 点位于相机后方或相机平面上")
    projected = intrinsic @ (point_camera[:3] / depth)
    return pixel_to_normalized_uv(projected[:2], image_size_hw)


def planar_tcp_delta_from_normalized_uv(
    normalized_uv: np.ndarray,
    intrinsic_cv: np.ndarray,
    base_from_camera_cv: np.ndarray,
    base_from_tcp: np.ndarray,
    image_size_hw: tuple[int, int],
    *,
    target_plane_base_z_m: float,
    desired_tcp_offset_from_target_base_m: np.ndarray,
    desired_delta_yaw_base_rad: float = 0.0,
) -> np.ndarray:
    """从目标像素生成 base-frame TCP commanded-target delta ``[dx,dy,dz,dyaw]``。"""

    base_from_tcp_value = validate_se3(base_from_tcp, "base_from_tcp")
    offset = np.asarray(desired_tcp_offset_from_target_base_m)
    if offset.shape != (3,) or offset.dtype != np.float32 or not np.isfinite(offset).all():
        raise ValueError("desired_tcp_offset_from_target_base_m 必须是有限 float32 [3]")
    if not math.isfinite(desired_delta_yaw_base_rad):
        raise ValueError("desired_delta_yaw_base_rad 必须是有限数值")
    intersection = normalized_uv_to_base_z_plane(
        normalized_uv,
        intrinsic_cv,
        base_from_camera_cv,
        image_size_hw,
        plane_base_z_m=target_plane_base_z_m,
    )
    desired_tcp_position = intersection.point_base_m + offset
    translation_delta = desired_tcp_position - base_from_tcp_value[:3, 3]
    return np.asarray(
        (*translation_delta.tolist(), float(desired_delta_yaw_base_rad)),
        dtype=np.float32,
    )


__all__ = [
    "PlaneIntersection",
    "normalized_uv_to_base_z_plane",
    "normalized_uv_to_pixel",
    "pixel_ray_to_base_plane",
    "pixel_to_normalized_uv",
    "planar_tcp_delta_from_normalized_uv",
    "project_base_point_to_normalized_uv",
]
