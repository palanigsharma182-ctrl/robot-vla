"""静止桌面方块的因果多视角2D重投影融合。"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from experiments.align_precision_vision.geometry import (
    GeometrySpec,
    PoseMeasurement,
    corner_permutations,
)
from experiments.align_precision_vision.pose_recovery import (
    PnPSpec,
    cube_ray_depth,
    interior_mask,
    project_cube,
    solve_pnp,
    visible_indices,
)
from experiments.tcp_memory_control.geometry import pose


RGB_SOURCE = 'precision-multiview-rgb/v1'
RGBD_SOURCE = 'precision-multiview-rgbd/v1'
MAX_VIEWS = 4
MAX_WINDOW_S = 1.5
# 纯旋转不会改变相机中心，不能把重复视线称为新的三角测量信息。
MIN_CAMERA_BASELINE_M = .002
_TIME_TOLERANCE_S = 1e-9


@dataclass(frozen=True)
class ViewObservation:
    """一帧在线可用输入；外参为 OpenCV 相机坐标到共同 base 坐标。"""

    uv: np.ndarray
    visibility: np.ndarray
    intrinsic: np.ndarray
    base_from_camera_cv: np.ndarray
    image_shape: tuple[int, int]
    episode: str
    target: str
    timestamp_s: float
    rgb: np.ndarray | None = None
    depth_m: np.ndarray | None = None

    def __post_init__(self):
        uv = np.array(self.uv, dtype=np.float64, copy=True)
        visibility = np.array(self.visibility, dtype=np.float64, copy=True)
        intrinsic = np.array(self.intrinsic, dtype=np.float64, copy=True)
        camera = pose(self.base_from_camera_cv)
        if uv.shape != (8, 2) or visibility.shape != (8,):
            raise ValueError('多视角输入必须为8个2D角点和8个可见性')
        if not np.isfinite(visibility).all() or np.any((visibility < 0) | (visibility > 1)):
            raise ValueError('多视角可见性必须为有限[0,1]')
        if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
                or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                or not np.allclose(intrinsic[2], [0, 0, 1])
                or not np.allclose([intrinsic[0, 1], intrinsic[1, 0]], [0, 0])):
            raise ValueError('多视角需要已去畸变、零skew OpenCV内参')
        if (not isinstance(self.image_shape, tuple) or len(self.image_shape) != 2
                or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
                       for v in self.image_shape)):
            raise ValueError('image_shape必须是正整数(h,w)')
        if not self.episode or not self.target or not math.isfinite(self.timestamp_s):
            raise ValueError('多视角帧必须有scene、target和有限时间')
        rgb = None if self.rgb is None else np.array(self.rgb, copy=True)
        # depth是否读取由recover_multiview_pose(use_depth=...)决定；RGB对照不能在构造时偷读。
        depth = self.depth_m
        if rgb is not None and (rgb.shape != self.image_shape + (3,) or rgb.dtype != np.uint8):
            raise ValueError('RGB必须为同分辨率uint8三通道图像')
        for value in (uv, visibility, intrinsic, camera, rgb):
            if value is not None:
                value.setflags(write=False)
        object.__setattr__(self, 'uv', uv)
        object.__setattr__(self, 'visibility', visibility)
        object.__setattr__(self, 'intrinsic', intrinsic)
        object.__setattr__(self, 'base_from_camera_cv', camera)
        object.__setattr__(self, 'rgb', rgb)
        object.__setattr__(self, 'depth_m', depth)


def _reject(current, reason, diagnostics, source):
    info = dict(diagnostics)
    info.update(reason=reason, causal=True, static_object_assumption=True,
                max_views=MAX_VIEWS, max_window_s=MAX_WINDOW_S)
    return PoseMeasurement(current.episode, current.target, current.timestamp_s,
                           None, reason, source, info)


def _validate_history(history, current):
    """先审计全部输入，再裁窗口；不能用裁窗掩盖跨scene或未来帧。"""
    if not isinstance(history, list):
        raise TypeError('history必须是仅含过去帧的list')
    if any(not isinstance(view, ViewObservation) for view in history):
        raise TypeError('history只能包含ViewObservation')
    times = [view.timestamp_s for view in history]
    if len(set(times)) != len(times):
        return 'duplicate_history_timestamp'
    if any(b < a for a, b in zip(times, times[1:])):
        return 'history_not_time_sorted'
    if any(view.episode != current.episode for view in history):
        return 'episode_mismatch'
    if any(view.target != current.target for view in history):
        return 'target_mismatch'
    if any(view.timestamp_s > current.timestamp_s + _TIME_TOLERANCE_S for view in history):
        return 'future_observation'
    if any(abs(view.timestamp_s - current.timestamp_s) <= _TIME_TOLERANCE_S for view in history):
        return 'duplicate_current_timestamp'
    return None


def _rotation_angle(matrix):
    return float(Rotation.from_matrix(matrix).magnitude())


def _camera_baselines(views):
    translation = []
    rotation = []
    for i, first in enumerate(views):
        for second in views[i + 1:]:
            translation.append(float(np.linalg.norm(
                first.base_from_camera_cv[:3, 3] - second.base_from_camera_cv[:3, 3])))
            rotation.append(_rotation_angle(
                first.base_from_camera_cv[:3, :3].T @ second.base_from_camera_cv[:3, :3]))
    return translation, rotation


def _assign_symmetries(base_from_object, prepared):
    permutations = corner_permutations()
    chosen = []
    for item in prepared:
        camera_from_object = np.linalg.inv(item['view'].base_from_camera_cv) @ base_from_object
        projected, depth = project_cube(camera_from_object, item['view'].intrinsic)
        if not np.isfinite(projected).all() or np.min(depth) <= .001:
            return None
        scores = []
        for permutation in permutations:
            error = projected[permutation[item['ids']]] - item['uv'][item['ids']]
            scores.append(float(np.mean(np.sum(error * error, axis=1))))
        chosen.append(int(np.argmin(scores)))
    return tuple(chosen)


def _optimize_seed(seed, prepared):
    """交替固定每帧整组yaw对应并优化共同base_from_object。"""
    permutations = corner_permutations()
    initial = seed.copy()

    def updated(delta):
        value = initial.copy()
        value[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix() @ initial[:3, :3]
        value[:3, 3] += delta[3:]
        return value

    assignments = _assign_symmetries(initial, prepared)
    if assignments is None:
        return None, 'initial_projection_invalid'
    solution = np.zeros(6)
    optimizer = None
    for _ in range(4):
        def residual(delta):
            base_from_object = updated(delta)
            parts = []
            for symmetry_index, item in zip(assignments, prepared):
                camera_from_object = np.linalg.inv(item['view'].base_from_camera_cv) @ base_from_object
                projected, depth = project_cube(camera_from_object, item['view'].intrinsic)
                if not np.isfinite(projected).all() or np.min(depth) <= .001:
                    return np.full(sum(2 * len(x['ids']) for x in prepared), 1e4)
                model_ids = permutations[symmetry_index, item['ids']]
                parts.append((projected[model_ids] - item['uv'][item['ids']]).ravel())
            return np.concatenate(parts)

        try:
            optimizer = least_squares(residual, solution, loss='soft_l1', f_scale=2.,
                                      max_nfev=80)
        except (ValueError, np.linalg.LinAlgError) as exc:
            return None, f'{type(exc).__name__}: {exc}'
        solution = optimizer.x
        candidate = updated(solution)
        revised = _assign_symmetries(candidate, prepared)
        if revised is None:
            return None, 'optimized_projection_invalid'
        if revised == assignments:
            break
        assignments = revised
    candidate = updated(solution)
    residuals = []
    per_view = []
    for symmetry_index, item in zip(assignments, prepared):
        camera_from_object = np.linalg.inv(item['view'].base_from_camera_cv) @ candidate
        projected, depth = project_cube(camera_from_object, item['view'].intrinsic)
        if not np.isfinite(projected).all() or np.min(depth) <= .001:
            return None, 'final_projection_invalid'
        model_ids = permutations[symmetry_index, item['ids']]
        error = projected[model_ids] - item['uv'][item['ids']]
        squared = np.sum(error * error, axis=1)
        residuals.extend(squared.tolist())
        per_view.append(float(np.sqrt(np.mean(squared))))
    return dict(value=candidate, assignments=assignments,
                reprojection_rms_px=float(np.sqrt(np.mean(residuals))),
                per_view_reprojection_rms_px=per_view,
                optimizer_success=bool(optimizer is not None and optimizer.success),
                optimizer_nfev=int(optimizer.nfev) if optimizer is not None else 0), None


def _depth_consistency(base_from_object, views, spec, pnp_spec):
    """仅从RGB腐蚀内部取深度；无数据或内部点不足时保留2D结果。"""
    per_view = []
    checked = 0
    inconsistent = False
    for view in views:
        info = dict(timestamp_s=view.timestamp_s, status='unavailable_fallback_2d',
                    interior_mask_pixels=0, sampled_points=0)
        if view.rgb is None or view.depth_m is None:
            per_view.append(info)
            continue
        depth = np.asarray(view.depth_m)
        if (depth.shape != view.image_shape
                or not np.issubdtype(depth.dtype, np.floating)):
            raise ValueError('深度必须为同分辨率浮点米')
        camera_from_object = np.linalg.inv(view.base_from_camera_cv) @ base_from_object
        mask = interior_mask(view.rgb, camera_from_object, view.intrinsic, pnp_spec)
        ys, xs = np.nonzero(mask)
        observed = depth[ys, xs]
        valid = (np.isfinite(observed) & (observed >= spec.min_depth_m)
                 & (observed <= spec.max_depth_m))
        ys, xs, observed = ys[valid], xs[valid], observed[valid]
        if len(xs) > pnp_spec.max_interior_points:
            selected = np.linspace(0, len(xs) - 1, pnp_spec.max_interior_points,
                                   dtype=int)
            ys, xs, observed = ys[selected], xs[selected], observed[selected]
        info.update(interior_mask_pixels=int(mask.sum()), sampled_points=int(len(xs)))
        if len(xs) < pnp_spec.min_interior_points:
            info['status'] = 'insufficient_interior_fallback_2d'
            per_view.append(info)
            continue
        rays = np.c_[xs, ys, np.ones(len(xs))] @ np.linalg.inv(view.intrinsic).T
        predicted = cube_ray_depth(camera_from_object, rays)
        hit = np.isfinite(predicted)
        predicted, observed = predicted[hit], observed[hit]
        info['sampled_points'] = int(len(observed))
        if len(observed) < pnp_spec.min_interior_points:
            info['status'] = 'insufficient_intersection_fallback_2d'
            per_view.append(info)
            continue
        error = np.abs(predicted - observed)
        median = float(np.median(error))
        p90 = float(np.quantile(error, .9))
        consistent = (median <= pnp_spec.depth_median_limit_m
                      and p90 <= pnp_spec.depth_p90_limit_m)
        info.update(status='consistent' if consistent else 'inconsistent',
                    depth_error_median_m=median, depth_error_p90_m=p90)
        checked += 1
        inconsistent |= not consistent
        per_view.append(info)
    return dict(depth_views=per_view, depth_checked_views=checked,
                depth_fallback_2d=checked < len(views),
                depth_consistent=not inconsistent)


def recover_multiview_pose(history: list[ViewObservation], *, current: ViewObservation,
                           spec=GeometrySpec(), pnp_spec=PnPSpec(), use_depth: bool = True):
    """融合最多三帧过去观测和当前帧，只适用于同一静止物体。"""
    if not isinstance(current, ViewObservation):
        raise TypeError('current必须是ViewObservation')
    if not isinstance(use_depth, bool):
        raise TypeError('use_depth必须是bool')
    source = RGBD_SOURCE if use_depth else RGB_SOURCE
    problem = _validate_history(history, current)
    diagnostics = dict(history_input_frames=len(history), current_timestamp_s=current.timestamp_s)
    if problem is not None:
        return _reject(current, problem, diagnostics, source)

    earliest = current.timestamp_s - MAX_WINDOW_S
    eligible = [view for view in history if view.timestamp_s >= earliest - _TIME_TOLERANCE_S]
    views = eligible[-(MAX_VIEWS - 1):] + [current]
    prepared = []
    pnp_initializations = []
    for view in views:
        uv, ids = visible_indices(view.uv, view.visibility, view.image_shape,
                                  spec.visibility_threshold)
        if len(ids):
            prepared.append(dict(view=view, uv=uv, ids=ids))
        value, info = solve_pnp(uv, ids, view.intrinsic, pnp_spec)
        if value is not None:
            pnp_initializations.append((view, view.base_from_camera_cv @ value, info))

    current_ids = next((item['ids'] for item in prepared if item['view'] is current),
                       np.empty(0, dtype=int))
    diagnostics.update(
        eligible_history_frames=len(eligible),
        selected_window_frames=len(views),
        observed_views=len(prepared),
        current_keypoints=int(len(current_ids)),
        current_evidence=bool(len(current_ids)),
        initialization_candidates=len(pnp_initializations),
        used_timestamps_s=[item['view'].timestamp_s for item in prepared],
        used_ages_s=[current.timestamp_s - item['view'].timestamp_s for item in prepared],
        window_span_s=(current.timestamp_s - prepared[0]['view'].timestamp_s) if prepared else 0.,
        depth_status='pending_consistency_check' if use_depth else 'disabled_rgb_only',
    )
    if not len(current_ids):
        return _reject(current, 'missing_current_evidence', diagnostics, source)
    if not pnp_initializations:
        diagnostics['initialization_requirement'] = '至少一帧有四个PnP可见角点'
        return _reject(current, 'multiview_initialization_wait', diagnostics, source)
    if len(prepared) < 2:
        return _reject(current, 'insufficient_multiview_evidence', diagnostics, source)

    translation_baselines, rotation_baselines = _camera_baselines(
        [item['view'] for item in prepared])
    diagnostics.update(
        camera_translation_baselines_m=translation_baselines,
        camera_rotation_baselines_rad=rotation_baselines,
        max_camera_translation_baseline_m=max(translation_baselines, default=0.),
        max_camera_rotation_baseline_rad=max(rotation_baselines, default=0.),
        independent_camera_baseline=bool(translation_baselines
                                         and max(translation_baselines) >= MIN_CAMERA_BASELINE_M),
        min_camera_baseline_m=MIN_CAMERA_BASELINE_M,
    )
    if not diagnostics['independent_camera_baseline']:
        return _reject(current, 'insufficient_camera_baseline', diagnostics, source)

    candidates = []
    optimization_failures = []
    for _, seed, _ in pnp_initializations:
        result, failure = _optimize_seed(seed, prepared)
        if result is not None:
            candidates.append(result)
        elif failure is not None:
            optimization_failures.append(failure)
    diagnostics['optimization_failures'] = optimization_failures
    if not candidates:
        return _reject(current, 'multiview_optimization_failed', diagnostics, source)
    candidates.sort(key=lambda item: item['reprojection_rms_px'])
    best = candidates[0]
    diagnostics.update(
        reprojection_rms_px=best['reprojection_rms_px'],
        per_view_reprojection_rms_px=best['per_view_reprojection_rms_px'],
        yaw_symmetry_indices=list(best['assignments']),
        optimizer_success=best['optimizer_success'],
        optimizer_nfev=best['optimizer_nfev'],
        optimized_initializations=len(candidates),
        initialization_timestamps_s=[view.timestamp_s for view, _, _ in pnp_initializations],
        initialization_from_current=any(view is current for view, _, _ in pnp_initializations),
        uncertainty='not calibrated',
    )
    if (not best['optimizer_success']
            or best['reprojection_rms_px'] > pnp_spec.max_reprojection_rms_px
            or max(best['per_view_reprojection_rms_px']) > pnp_spec.max_reprojection_rms_px):
        return _reject(current, 'multiview_reprojection_conflict', diagnostics, source)
    if use_depth:
        depth_info = _depth_consistency(best['value'], views, spec, pnp_spec)
        diagnostics.update(depth_info)
        diagnostics['depth_status'] = ('consistent_or_fallback_2d'
                                       if depth_info['depth_consistent'] else 'inconsistent')
        if not depth_info['depth_consistent']:
            return _reject(current, 'multiview_depth_inconsistent', diagnostics, source)
    else:
        diagnostics.update(depth_views=[], depth_checked_views=0,
                           depth_fallback_2d=False)
    diagnostics['reason'] = 'valid'
    return PoseMeasurement(current.episode, current.target, current.timestamp_s,
                           best['value'], 'valid', source, diagnostics)
