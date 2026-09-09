"""三点局部几何位姿恢复；只用 RGB 前景内部深度消歧 P3P 候选。"""

from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from experiments.align_precision_vision.geometry import (
    CUBE_POINTS,
    GeometrySpec,
    PoseMeasurement,
    yaw_symmetries,
)
from experiments.align_precision_vision.pose_recovery import (
    PnPSpec,
    cube_ray_depth,
    project_cube,
    recover_pose,
    visible_indices,
)
from experiments.tcp_memory_control.geometry import pose


LOCAL_SOURCE = "precision-rgbd-keypoints-local-p3p/v2"


@dataclass(frozen=True)
class LocalPoseSpec:
    """P3P 消歧的预先固定阈值；深度单位均为米。"""

    erosion_radius_px: int = 2
    min_interior_points: int = 32
    max_interior_points: int = 256
    depth_median_limit_m: float = 0.005
    depth_p90_limit_m: float = 0.010
    min_depth_score_gap_m: float = 0.001
    equivalent_translation_m: float = 0.001
    equivalent_rotation_rad: float = math.radians(2.0)
    foreground_secondary_ratio: float = 0.25

    def __post_init__(self):
        values = asdict(self)
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("局部位姿参数必须有限")
        if any(value <= 0 for value in values.values()):
            raise ValueError("局部位姿阈值必须为正")
        for name in ("erosion_radius_px", "min_interior_points", "max_interior_points"):
            if not isinstance(getattr(self, name), int):
                raise ValueError("局部位姿计数参数必须为整数")
        if self.min_interior_points > self.max_interior_points:
            raise ValueError("局部位姿内部点数范围错误")
        if self.foreground_secondary_ratio >= 1:
            raise ValueError("红色前景次通道比例必须位于(0,1)")


def _intrinsic(value):
    intrinsic = np.asarray(value, dtype=np.float64)
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or intrinsic[0, 0] <= 0
        or intrinsic[1, 1] <= 0
        or not np.allclose(intrinsic[2], [0, 0, 1])
        or not np.allclose([intrinsic[0, 1], intrinsic[1, 0]], [0, 0])
    ):
        raise ValueError("P3P需要已去畸变、零skew OpenCV内参")
    return intrinsic


def _shared_interior_samples(rgb, depth_m, intrinsic, image_shape, geometry, config):
    """固定 RGB 腐蚀前景样本；候选位姿不参与 mask 或抽样。"""
    image = np.asarray(rgb)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or image.shape[:2] != tuple(image_shape)
    ):
        raise ValueError("前景需要原始RGB uint8")
    depth = np.asarray(depth_m)
    if depth.shape != image.shape[:2] or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError("深度必须为同分辨率浮点米")

    red, green, blue = np.moveaxis(image.astype(np.float32), -1, 0)
    # 当前目标的红色通道远高于次通道；更宽的相对阈值会把木桌当成目标。
    secondary = np.maximum(green, blue)
    foreground = (
        (red >= 60) & (secondary <= config.foreground_secondary_ratio * red)
    )
    kernel = np.ones((2 * config.erosion_radius_px + 1,) * 2, np.uint8)
    interior = cv2.erode(
        foreground.astype(np.uint8),
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    ys, xs = np.nonzero(interior)
    observed = depth[ys, xs]
    valid = (
        np.isfinite(observed)
        & (observed >= geometry.min_depth_m)
        & (observed <= geometry.max_depth_m)
    )
    ys, xs, observed = ys[valid], xs[valid], observed[valid]
    if len(xs) > config.max_interior_points:
        selected = np.linspace(0, len(xs) - 1, config.max_interior_points, dtype=int)
        ys, xs, observed = ys[selected], xs[selected], observed[selected]
    rays = np.c_[xs, ys, np.ones(len(xs))] @ np.linalg.inv(intrinsic).T
    info = {
        "foreground_source": "rgb-red-secondary-ratio-eroded/v2",
        "foreground_secondary_ratio": config.foreground_secondary_ratio,
        "erosion_radius_px": config.erosion_radius_px,
        "interior_mask_pixels": int(interior.sum()),
        "interior_points": int(len(xs)),
        "shared_candidate_samples": True,
    }
    return rays, observed, info


def _pose_difference(first, second):
    """按方块整体四重 yaw 对称计算两个候选的最小姿态差。"""
    translation = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    rotations = []
    for symmetry in yaw_symmetries():
        relative = first[:3, :3].T @ (second @ symmetry)[:3, :3]
        rotations.append(float(Rotation.from_matrix(relative).magnitude()))
    return translation, min(rotations)


def _solve_p3p(uv, ids, intrinsic, rays, observed, pnp_config, local_config):
    points = np.ascontiguousarray(CUBE_POINTS[ids], dtype=np.float64)
    pixels = np.ascontiguousarray(uv[ids], dtype=np.float64)
    if (
        np.linalg.matrix_rank(points - points.mean(0), tol=1e-8) < 2
        or np.linalg.matrix_rank(pixels - pixels.mean(0), tol=1e-4) < 2
    ):
        return None, {"reason": "degenerate_p3p", "keypoints": 3}

    solver_errors = []
    raw_candidates = []
    try:
        count, rvecs, tvecs = cv2.solveP3P(
            points, pixels, intrinsic, None, flags=cv2.SOLVEPNP_AP3P
        )
        if count:
            raw_candidates.extend(zip(rvecs, tvecs))
    except cv2.error as exc:
        solver_errors.append(str(exc).splitlines()[-1])

    scored = []
    diagnostics = []
    for index, (rvec, tvec) in enumerate(raw_candidates):
        rotation_vector = np.asarray(rvec, dtype=np.float64).reshape(-1)
        translation_vector = np.asarray(tvec, dtype=np.float64).reshape(-1)
        if (
            rotation_vector.shape != (3,)
            or translation_vector.shape != (3,)
            or not np.isfinite(rotation_vector).all()
            or not np.isfinite(translation_vector).all()
        ):
            diagnostics.append(
                {
                    "index": index,
                    "camera_translation_m": None,
                    "camera_rotation_vector_rad": None,
                    "reprojection_rms_px": None,
                    "positive_cube_depth": False,
                    "intersection_points": 0,
                    "uses_all_shared_samples": False,
                    "depth_median_m": None,
                    "depth_p90_m": None,
                    "depth_score_m": None,
                    "valid": False,
                    "rejection": "nonfinite_solver_candidate",
                }
            )
            continue
        candidate = np.eye(4)
        candidate[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
        candidate[:3, 3] = translation_vector
        projected, z = project_cube(candidate, intrinsic)
        residual_px = np.linalg.norm(projected[ids] - pixels, axis=1)
        rms_px = float(np.sqrt(np.mean(residual_px**2)))
        finite_rms = math.isfinite(rms_px)
        predicted = cube_ray_depth(candidate, rays)
        intersections = np.isfinite(predicted)
        row = {
            "index": index,
            "camera_translation_m": candidate[:3, 3].tolist(),
            "camera_rotation_vector_rad": Rotation.from_matrix(
                candidate[:3, :3]
            ).as_rotvec().tolist(),
            "reprojection_rms_px": rms_px if finite_rms else None,
            "positive_cube_depth": bool(np.isfinite(z).all() and np.min(z) > 0.001),
            "intersection_points": int(intersections.sum()),
            "uses_all_shared_samples": bool(intersections.all()),
            "depth_median_m": None,
            "depth_p90_m": None,
            "depth_score_m": None,
            "valid": False,
            "rejection": None,
        }
        # 错解缺少射线交点时也必须失败，不能丢掉不利样本后评分。
        if (
            row["positive_cube_depth"]
            and finite_rms
            and rms_px <= pnp_config.max_reprojection_rms_px
            and intersections.all()
        ):
            difference = np.abs(predicted - observed)
            median = float(np.median(difference))
            p90 = float(np.quantile(difference, 0.9))
            score = float((median + p90) / 2.0)
            row.update(
                depth_median_m=median,
                depth_p90_m=p90,
                depth_score_m=score,
                valid=(
                    median <= local_config.depth_median_limit_m
                    and p90 <= local_config.depth_p90_limit_m
                ),
            )
            if row["valid"]:
                scored.append((score, candidate, row))
            else:
                row["rejection"] = "depth_limits"
        elif not row["positive_cube_depth"]:
            row["rejection"] = "nonpositive_cube_depth"
        elif not finite_rms or rms_px > pnp_config.max_reprojection_rms_px:
            row["rejection"] = "reprojection_limit"
        else:
            row["rejection"] = "missing_shared_ray_intersection"
        diagnostics.append(row)

    info = {
        "reason": "valid",
        "keypoints": 3,
        "candidate_count": len(raw_candidates),
        "valid_candidate_count": len(scored),
        "candidate_diagnostics": diagnostics,
        "solver": "OpenCV AP3P",
        "solver_errors": solver_errors,
        "selection": "shared eroded RGB foreground depth; no GT",
        "depth_median_limit_m": local_config.depth_median_limit_m,
        "depth_p90_limit_m": local_config.depth_p90_limit_m,
        "min_depth_score_gap_m": local_config.min_depth_score_gap_m,
        "equivalent_translation_m": local_config.equivalent_translation_m,
        "equivalent_rotation_rad": local_config.equivalent_rotation_rad,
    }
    if not raw_candidates:
        info["reason"] = "p3p_no_solution"
        return None, info
    if not scored:
        info["reason"] = "p3p_depth_inconsistent"
        return None, info

    scored.sort(key=lambda item: item[0])
    best_score, best, best_row = scored[0]
    info.update(selected_candidate=best_row["index"], selected_depth_score_m=best_score)
    distinct = []
    for score, candidate, row in scored[1:]:
        translation, rotation = _pose_difference(best, candidate)
        equivalent = (
            translation <= local_config.equivalent_translation_m
            and rotation <= local_config.equivalent_rotation_rad
        )
        distinct.append(
            {
                "index": row["index"],
                "depth_score_gap_m": float(score - best_score),
                "translation_gap_m": translation,
                "rotation_gap_rad_mod_yaw": rotation,
                "equivalent_mod_cube_yaw": equivalent,
            }
        )
        if not equivalent and score - best_score < local_config.min_depth_score_gap_m:
            info.update(reason="ambiguous_p3p", competing_candidates=distinct)
            return None, info
    if distinct:
        info["competing_candidates"] = distinct
    return best, info


def recover_local_pose(
    pixel_uv,
    visibility,
    intrinsic,
    base_from_camera_cv,
    *,
    image_shape,
    episode,
    target,
    timestamp_s,
    depth_m=None,
    rgb=None,
    spec=GeometrySpec(),
    pnp_spec=PnPSpec(),
    local_spec=LocalPoseSpec(),
):
    """四点沿用原 PnP；三点仅在共享 RGB 内部深度可消歧时返回。"""
    uv, ids = visible_indices(
        pixel_uv, visibility, image_shape, spec.visibility_threshold
    )
    if len(ids) >= 4:
        return recover_pose(
            pixel_uv,
            visibility,
            intrinsic,
            base_from_camera_cv,
            image_shape=image_shape,
            episode=episode,
            target=target,
            timestamp_s=timestamp_s,
            method="pnp-only",
            depth_m=depth_m,
            rgb=rgb,
            spec=spec,
            pnp_spec=pnp_spec,
        )

    common = {
        "method": "local-p3p-depth",
        "visibility_threshold": spec.visibility_threshold,
        "keypoints": int(len(ids)),
        "distortion": "rectified-zero",
    }
    camera = pose(base_from_camera_cv)
    k = _intrinsic(intrinsic)
    if len(ids) < 3:
        return PoseMeasurement(
            episode,
            target,
            timestamp_s,
            None,
            "insufficient_visible_local_pose",
            LOCAL_SOURCE,
            common,
        )
    if rgb is None or depth_m is None:
        return PoseMeasurement(
            episode,
            target,
            timestamp_s,
            None,
            "p3p_depth_unavailable",
            LOCAL_SOURCE,
            common,
        )
    rays, observed, depth_info = _shared_interior_samples(
        rgb, depth_m, k, image_shape, spec, local_spec
    )
    common.update(depth_info)
    if len(rays) < local_spec.min_interior_points:
        return PoseMeasurement(
            episode,
            target,
            timestamp_s,
            None,
            "insufficient_interior_depth_p3p",
            LOCAL_SOURCE,
            common,
        )
    value, info = _solve_p3p(
        uv, ids, k, rays, observed, pnp_spec, local_spec
    )
    common.update(info)
    if value is None:
        return PoseMeasurement(
            episode,
            target,
            timestamp_s,
            None,
            common["reason"],
            LOCAL_SOURCE,
            common,
        )
    return PoseMeasurement(
        episode,
        target,
        timestamp_s,
        camera @ value,
        "valid",
        LOCAL_SOURCE,
        common,
    )


__all__ = ["LOCAL_SOURCE", "LocalPoseSpec", "recover_local_pose"]
