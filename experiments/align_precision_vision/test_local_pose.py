"""局部三点几何的真实 P3P、深度消歧和拒绝反例。"""

import json
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.align_precision_vision.geometry import (
    CUBE_POINTS,
    corner_permutations,
    yaw_symmetries,
)
from experiments.align_precision_vision.local_pose import (
    LOCAL_SOURCE,
    LocalPoseSpec,
    recover_local_pose,
)
from experiments.align_precision_vision.pose_recovery import (
    cube_ray_depth,
    project_cube,
    recover_pose,
)


def rendered(transform=None):
    intrinsic = np.array([[400.0, 0, 160], [0, 400.0, 120], [0, 0, 1]])
    if transform is None:
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.15, 0.3]).as_matrix()
        transform[:3, 3] = [0.01, -0.005, 0.30]
    yy, xx = np.indices((240, 320))
    rays = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.size)] @ np.linalg.inv(intrinsic).T
    depth = cube_ray_depth(transform, rays).reshape(240, 320)
    mask = np.isfinite(depth)
    rgb = np.zeros((240, 320, 3), np.uint8)
    rgb[mask] = [200, 10, 10]
    depth = np.where(mask, depth, 0.8).astype(np.float32)
    uv, _ = project_cube(transform, intrinsic)
    return intrinsic, transform, uv, rgb, depth, mask


def recover(uv, intrinsic, *, visibility, rgb=None, depth_m=None, camera=None, **kwargs):
    return recover_local_pose(
        uv,
        visibility,
        intrinsic,
        np.eye(4) if camera is None else camera,
        image_shape=(240, 320),
        episode="e",
        target="cube",
        timestamp_s=0.0,
        rgb=rgb,
        depth_m=depth_m,
        **kwargs,
    )


def three_visible():
    # 三个非共线、非同一面的已知对应，确实产生 P3P 多候选。
    visibility = np.zeros(8)
    visibility[[0, 3, 5]] = 1.0
    return visibility


def test_three_points_choose_pose_with_shared_interior_depth():
    intrinsic, truth, uv, rgb, depth, _ = rendered()
    result = recover(
        uv, intrinsic, visibility=three_visible(), rgb=rgb, depth_m=depth
    )
    assert result.reason == "valid" and result.source == LOCAL_SOURCE
    assert result.diagnostics["keypoints"] == 3
    assert result.diagnostics["shared_candidate_samples"]
    assert result.diagnostics["candidate_count"] >= 1
    np.testing.assert_allclose(result.base_from_object, truth, atol=1e-5)


def test_three_points_without_depth_rejects():
    intrinsic, _, uv, rgb, depth, _ = rendered()
    visibility = three_visible()
    assert recover(uv, intrinsic, visibility=visibility, rgb=rgb).reason == "p3p_depth_unavailable"


def test_real_p3p_candidates_with_equal_shared_depth_are_ambiguous():
    intrinsic, _, uv, _, _, _ = rendered()
    ids = np.flatnonzero(three_visible())
    count, rvecs, tvecs = cv2.solveP3P(
        np.ascontiguousarray(CUBE_POINTS[ids]),
        np.ascontiguousarray(uv[ids]),
        intrinsic,
        None,
        flags=cv2.SOLVEPNP_AP3P,
    )
    assert count >= 2
    candidates = []
    for rvec, tvec in zip(rvecs, tvecs):
        value = np.eye(4)
        value[:3, :3] = cv2.Rodrigues(rvec)[0]
        value[:3, 3] = np.asarray(tvec).reshape(3)
        candidates.append(value)

    # 两个真实 P3P 解投影出的凸方块都包含三点三角形；只保留共同内部，
    # 并把观测深度置于两解中点，构造无法由观测消歧而不能由 GT 代选的反例。
    yy, xx = np.indices((240, 320))
    rays = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.size)] @ np.linalg.inv(intrinsic).T
    first = cube_ray_depth(candidates[0], rays).reshape(240, 320)
    second = cube_ray_depth(candidates[1], rays).reshape(240, 320)
    common = np.isfinite(first) & np.isfinite(second)
    assert cv2.erode(common.astype(np.uint8), np.ones((5, 5), np.uint8)).sum() >= 32
    rgb = np.zeros((240, 320, 3), np.uint8)
    rgb[common] = [200, 10, 10]
    depth = np.full((240, 320), 0.8, np.float32)
    depth[common] = ((first[common] + second[common]) / 2).astype(np.float32)
    result = recover(
        uv,
        intrinsic,
        visibility=three_visible(),
        rgb=rgb,
        depth_m=depth,
        local_spec=LocalPoseSpec(
            depth_median_limit_m=1.0,
            depth_p90_limit_m=1.0,
            min_depth_score_gap_m=1.0,
        ),
    )
    assert result.reason == "ambiguous_p3p"
    assert result.diagnostics["valid_candidate_count"] >= 2


def test_contour_depth_pollution_never_changes_three_point_result():
    intrinsic, _, uv, rgb, depth, mask = rendered()
    visibility = three_visible()
    kernel = np.ones((5, 5), np.uint8)
    edge = mask & ~cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    assert edge.any()
    contaminated = depth.copy()
    contaminated[edge] = 1.7
    original = recover(uv, intrinsic, visibility=visibility, rgb=rgb, depth_m=depth)
    changed = recover(
        uv, intrinsic, visibility=visibility, rgb=rgb, depth_m=contaminated
    )
    assert original.reason == changed.reason == "valid"
    np.testing.assert_allclose(
        original.base_from_object, changed.base_from_object, atol=1e-10
    )


def test_wood_background_depth_is_not_sampled_as_red_foreground():
    intrinsic, truth, uv, _, depth, mask = rendered()
    rgb = np.empty((240, 320, 3), np.uint8)
    rgb[:] = [168, 102, 63]
    rgb[mask] = [231, 4, 4]
    visibility = three_visible()
    original = recover(
        uv, intrinsic, visibility=visibility, rgb=rgb, depth_m=depth
    )
    contaminated = depth.copy()
    contaminated[~mask] = 1.7
    changed = recover(
        uv, intrinsic, visibility=visibility, rgb=rgb, depth_m=contaminated
    )
    assert original.reason == changed.reason == "valid"
    assert original.diagnostics["foreground_secondary_ratio"] == 0.25
    assert 32 <= original.diagnostics["interior_mask_pixels"] < int(mask.sum())
    np.testing.assert_allclose(original.base_from_object, truth, atol=1e-5)
    np.testing.assert_allclose(
        original.base_from_object, changed.base_from_object, atol=1e-10
    )


def test_four_points_are_exact_pnp_only_parity():
    intrinsic, _, uv, rgb, depth, _ = rendered()
    visibility = np.zeros(8)
    visibility[[0, 1, 2, 3]] = 1.0
    expected = recover_pose(
        uv,
        visibility,
        intrinsic,
        np.eye(4),
        image_shape=(240, 320),
        episode="e",
        target="cube",
        timestamp_s=0.0,
        method="pnp-only",
        rgb=rgb,
        depth_m=depth,
    )
    actual = recover(
        uv, intrinsic, visibility=visibility, rgb=rgb, depth_m=depth
    )
    assert actual.reason == expected.reason
    assert actual.source == expected.source
    assert actual.diagnostics == expected.diagnostics
    np.testing.assert_array_equal(actual.base_from_object, expected.base_from_object)


def test_three_point_extrinsic_and_whole_cube_yaw_symmetry():
    intrinsic, truth, uv, rgb, depth, _ = rendered()
    camera = np.eye(4)
    camera[:3, :3] = Rotation.from_euler("z", 0.4).as_matrix()
    camera[:3, 3] = [0.1, -0.2, 0.05]
    result = recover(
        uv,
        intrinsic,
        visibility=three_visible(),
        rgb=rgb,
        depth_m=depth,
        camera=camera,
    )
    assert result.reason == "valid"
    np.testing.assert_allclose(result.base_from_object, camera @ truth, atol=1e-5)

    for permutation, symmetry in zip(corner_permutations(), yaw_symmetries()):
        symmetric = recover(
            uv[permutation],
            intrinsic,
            visibility=three_visible()[permutation],
            rgb=rgb,
            depth_m=depth,
        )
        assert symmetric.reason == "valid"
        np.testing.assert_allclose(symmetric.base_from_object, truth @ symmetry, atol=1e-5)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_fewer_than_three_visible_points_reject_without_reading_depth(count):
    class Forbidden:
        def __array__(self, *args, **kwargs):
            raise AssertionError("不足三点时不应读取RGB或深度")

    intrinsic, _, uv, _, _, _ = rendered()
    visibility = np.zeros(8)
    visibility[:count] = 1.0
    result = recover(
        uv,
        intrinsic,
        visibility=visibility,
        rgb=Forbidden(),
        depth_m=Forbidden(),
    )
    assert result.reason == "insufficient_visible_local_pose"
    assert result.diagnostics["keypoints"] == count


def test_shared_interior_minimum_and_spec_validation():
    intrinsic, _, uv, rgb, depth, _ = rendered()
    tiny = np.zeros_like(rgb)
    tiny[100:104, 100:104] = [200, 10, 10]
    result = recover(
        uv, intrinsic, visibility=three_visible(), rgb=tiny, depth_m=depth
    )
    assert result.reason == "insufficient_interior_depth_p3p"
    with pytest.raises(ValueError, match="点数"):
        LocalPoseSpec(min_interior_points=33, max_interior_points=32)
    with pytest.raises(ValueError, match="正"):
        LocalPoseSpec(min_depth_score_gap_m=0)
    with pytest.raises(ValueError, match="比例"):
        LocalPoseSpec(foreground_secondary_ratio=1)


def test_nonfinite_solver_candidate_has_strict_json_diagnostics(monkeypatch):
    import experiments.align_precision_vision.local_pose as module

    def nonfinite(*args, **kwargs):
        return 1, [np.full((3, 1), np.nan)], [np.zeros((3, 1))]

    monkeypatch.setattr(module.cv2, "solveP3P", nonfinite)
    intrinsic, _, uv, rgb, depth, _ = rendered()
    result = recover(
        uv, intrinsic, visibility=three_visible(), rgb=rgb, depth_m=depth
    )
    assert result.reason == "p3p_depth_inconsistent"
    row = result.diagnostics["candidate_diagnostics"][0]
    assert row["rejection"] == "nonfinite_solver_candidate"
    json.dumps(result.diagnostics, allow_nan=False)


@pytest.mark.parametrize("bad_input", ["rgb", "depth"])
def test_rgb_and_depth_must_both_match_declared_image_shape(bad_input):
    intrinsic, _, uv, rgb, depth, _ = rendered()
    if bad_input == "rgb":
        rgb = rgb[:-1]
        depth = depth[:-1]
    else:
        depth = depth[:-1]
    with pytest.raises(ValueError):
        recover(
            uv, intrinsic, visibility=three_visible(), rgb=rgb, depth_m=depth
        )
