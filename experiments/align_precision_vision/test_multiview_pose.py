"""静止方块因果多视角融合的合成反例。"""
from dataclasses import replace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from experiments.align_precision_vision.geometry import corner_permutations, yaw_symmetries
from experiments.align_precision_vision.multiview_pose import (
    RGBD_SOURCE,
    RGB_SOURCE,
    ViewObservation,
    recover_multiview_pose,
)
from experiments.align_precision_vision.pose_recovery import cube_ray_depth, project_cube


K = np.array([[400., 0., 160.], [0., 400., 120.], [0., 0., 1.]])


def camera_pose(x, yaw=0.):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    value[:3, 3] = [x, 0., 0.]
    return value


def truth_pose(x=0.):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler('xyz', [.2, -.15, .3]).as_matrix()
    value[:3, 3] = [x, .01, .35]
    return value


def observation(timestamp, camera, truth=None, *, visibility=None, episode='scene', target='cube'):
    truth = truth_pose() if truth is None else truth
    uv, _ = project_cube(np.linalg.inv(camera) @ truth, K)
    return ViewObservation(uv, np.ones(8) if visibility is None else visibility, K, camera,
                           (240, 320), episode, target, timestamp)


def rendered_observation(timestamp, camera, truth=None):
    truth = truth_pose() if truth is None else truth
    base = observation(timestamp, camera, truth=truth)
    camera_from_object = np.linalg.inv(camera) @ truth
    yy, xx = np.indices(base.image_shape)
    rays = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.size)] @ np.linalg.inv(K).T
    depth = cube_ray_depth(camera_from_object, rays).reshape(base.image_shape)
    mask = np.isfinite(depth)
    rgb = np.zeros(base.image_shape + (3,), np.uint8)
    rgb[mask] = [200, 10, 10]
    depth = np.where(mask, depth, .8).astype(np.float32)
    return replace(base, rgb=rgb, depth_m=depth), mask


def symmetry_error(estimate, truth):
    return min(
        np.linalg.norm(Rotation.from_matrix(
            estimate[:3, :3].T @ (truth @ symmetry)[:3, :3]).as_rotvec())
        for symmetry in yaw_symmetries()
    )


def test_fuses_each_camera_extrinsic_and_whole_yaw_relabeling():
    truth = truth_pose()
    first = observation(0., camera_pose(-.03), truth=truth)
    current = observation(.1, camera_pose(.03, .08), truth=truth)
    # 模拟网络在第二帧选择了另一个整体yaw等价标签。
    current = replace(current, uv=current.uv[corner_permutations()[1]],
                      visibility=current.visibility[corner_permutations()[1]])
    result = recover_multiview_pose([first], current=current)
    assert result.reason == 'valid'
    np.testing.assert_allclose(result.base_from_object[:3, 3], truth[:3, 3], atol=1e-6)
    assert symmetry_error(result.base_from_object, truth) < 1e-6
    assert result.diagnostics['used_timestamps_s'] == [0., .1]
    assert result.diagnostics['current_evidence']
    assert result.diagnostics['max_camera_translation_baseline_m'] > .05


def test_history_is_strictly_causal_ordered_and_same_identity():
    current = observation(1., camera_pose(.03))
    past = observation(.9, camera_pose(-.03))
    assert recover_multiview_pose([replace(past, timestamp_s=1.1)], current=current).reason == 'future_observation'
    assert recover_multiview_pose([replace(past, timestamp_s=1.)], current=current).reason == 'duplicate_current_timestamp'
    assert recover_multiview_pose([past, replace(past, timestamp_s=.8)], current=current).reason == 'history_not_time_sorted'
    assert recover_multiview_pose([replace(past, episode='other')], current=current).reason == 'episode_mismatch'
    assert recover_multiview_pose([replace(past, target='other')], current=current).reason == 'target_mismatch'


def test_window_uses_current_plus_three_recent_frames_within_1p5_seconds():
    current = observation(2., camera_pose(.04))
    history = [observation(t, camera_pose(x)) for t, x in
               [(0., -.05), (.6, -.04), (1., -.03), (1.4, -.02), (1.8, 0.)]]
    result = recover_multiview_pose(history, current=current)
    assert result.reason == 'valid'
    assert result.diagnostics['selected_window_frames'] == 4
    assert result.diagnostics['used_timestamps_s'] == [1., 1.4, 1.8, 2.]
    assert result.diagnostics['window_span_s'] == 1.


def test_missing_observation_duplicate_and_low_baseline_are_rejected():
    invisible = np.zeros(8)
    first = observation(0., camera_pose(-.03))
    current = observation(.1, camera_pose(.03), visibility=invisible)
    assert recover_multiview_pose([first], current=current).reason == 'missing_current_evidence'
    normal = observation(.1, camera_pose(.03))
    duplicate = replace(first, timestamp_s=.05)
    assert recover_multiview_pose([first, replace(duplicate, timestamp_s=0.)], current=normal).reason == 'duplicate_history_timestamp'
    repeated_view = observation(0., camera_pose(.03))
    assert recover_multiview_pose([repeated_view], current=normal).reason == 'insufficient_camera_baseline'


def test_no_single_frame_pnp_initialization_waits_explicitly():
    visibility = np.array([1., 1., 1., 0., 0., 0., 0., 0.])
    first = observation(0., camera_pose(-.03), visibility=visibility)
    current = observation(.1, camera_pose(.03), visibility=visibility)
    result = recover_multiview_pose([first], current=current)
    assert result.reason == 'multiview_initialization_wait'
    assert result.diagnostics['current_keypoints'] == 3
    assert result.diagnostics['initialization_candidates'] == 0


def test_conflicting_static_object_observations_are_rejected():
    first = observation(0., camera_pose(-.03), truth=truth_pose())
    current = observation(.1, camera_pose(.03), truth=truth_pose(.06))
    result = recover_multiview_pose([first], current=current)
    assert result.reason == 'multiview_reprojection_conflict'
    assert max(result.diagnostics['per_view_reprojection_rms_px']) > 3.


def test_rgbd_ignores_contaminated_foreground_boundary_depth():
    first, first_mask = rendered_observation(0., camera_pose(-.03))
    current, current_mask = rendered_observation(.1, camera_pose(.03))
    clean = recover_multiview_pose([first], current=current, use_depth=True)
    contaminated = []
    for view, mask in ((first, first_mask), (current, current_mask)):
        edge = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        bad = view.depth_m.copy()
        bad[edge] = 1.7
        contaminated.append(replace(view, depth_m=bad))
    result = recover_multiview_pose([contaminated[0]], current=contaminated[1], use_depth=True)
    assert clean.reason == result.reason == 'valid'
    assert clean.source == result.source == RGBD_SOURCE
    np.testing.assert_allclose(clean.base_from_object, result.base_from_object, atol=1e-10)
    assert clean.diagnostics['depth_checked_views'] == result.diagnostics['depth_checked_views'] == 2


def test_rgbd_rejects_wrong_eroded_interior_depth():
    first, first_mask = rendered_observation(0., camera_pose(-.03))
    current, current_mask = rendered_observation(.1, camera_pose(.03))
    bad_views = []
    for view, mask in ((first, first_mask), (current, current_mask)):
        bad = view.depth_m.copy()
        bad[mask] += .10
        bad_views.append(replace(view, depth_m=bad))
    result = recover_multiview_pose([bad_views[0]], current=bad_views[1], use_depth=True)
    assert result.reason == 'multiview_depth_inconsistent'
    assert result.diagnostics['depth_checked_views'] == 2
    assert all(item['status'] == 'inconsistent' for item in result.diagnostics['depth_views'])


def test_missing_depth_explicitly_falls_back_to_2d():
    first = observation(0., camera_pose(-.03))
    current = observation(.1, camera_pose(.03))
    result = recover_multiview_pose([first], current=current, use_depth=True)
    assert result.reason == 'valid' and result.source == RGBD_SOURCE
    assert result.diagnostics['depth_checked_views'] == 0
    assert result.diagnostics['depth_fallback_2d']
    assert [item['status'] for item in result.diagnostics['depth_views']] == [
        'unavailable_fallback_2d', 'unavailable_fallback_2d']


def test_rgb_mode_never_reads_depth():
    class ForbiddenDepth:
        def __array__(self, *args, **kwargs):
            raise AssertionError('RGB多视角对照不应访问depth')

    first = replace(observation(0., camera_pose(-.03)), depth_m=ForbiddenDepth())
    current = replace(observation(.1, camera_pose(.03)), depth_m=ForbiddenDepth())
    result = recover_multiview_pose([first], current=current, use_depth=False)
    assert result.reason == 'valid' and result.source == RGB_SOURCE
    assert result.diagnostics['depth_status'] == 'disabled_rgb_only'
    assert result.diagnostics['depth_checked_views'] == 0


def test_optimizer_failure_records_exception_type(monkeypatch):
    import experiments.align_precision_vision.multiview_pose as module

    def fail(*args, **kwargs):
        raise ValueError('synthetic optimizer failure')

    monkeypatch.setattr(module, 'least_squares', fail)
    first = observation(0., camera_pose(-.03))
    current = observation(.1, camera_pose(.03))
    result = recover_multiview_pose([first], current=current, use_depth=False)
    assert result.reason == 'multiview_optimization_failed'
    assert result.diagnostics['optimization_failures']
    assert all(value.startswith('ValueError: synthetic optimizer failure')
               for value in result.diagnostics['optimization_failures'])
