"""E014 Precision long-tail 诊断的纯函数与冻结 taxonomy 规则。"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_vla.precision.geometry import (
    normalized_uv_to_base_z_plane,
    normalized_uv_to_pixel,
    pixel_to_normalized_uv,
)

E014_DIAGNOSTIC_VERSION = "e014-precision-long-tail-diagnostic/v1"
E014_RULES_VERSION = "e014-precision-long-tail-rules/v1"
FAILURE_TAXONOMY_PRIORITY = (
    "label_or_channel_contract_failure",
    "temporal_alignment_failure",
    "semantic_swap_failure",
    "geometry_conditioning_failure",
    "multimodal_softargmax_failure",
    "visibility_or_ood_failure",
    "generic_correspondence_failure",
    "unclear_or_mixed",
)
WORLD_ERROR_THRESHOLDS_MM = (5.0, 10.0, 20.0, 50.0, 100.0)


@dataclass(frozen=True)
class LocalPeak:
    pixel_uv: tuple[float, float]
    value: float

    def to_dict(self) -> dict[str, Any]:
        return {"pixel_uv": list(self.pixel_uv), "value": self.value}


@dataclass(frozen=True)
class LocalPeakDiagnostics:
    peaks: tuple[LocalPeak, ...]
    local_maxima_count: int
    separated_local_maxima_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "peaks": [peak.to_dict() for peak in self.peaks],
            "local_maxima_count": self.local_maxima_count,
            "separated_local_maxima_count": self.separated_local_maxima_count,
        }


def _finite_map(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) <= 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须是有限非空二维数组")
    return array


def probability_from_logits(logits: np.ndarray, *, temperature: float) -> np.ndarray:
    """以 float64 稳定复算冻结 temperature 下的二维 softmax。"""

    values = _finite_map(logits, "logits")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature 必须是有限正数")
    shifted = values / float(temperature)
    shifted -= float(np.max(shifted))
    probability = np.exp(shifted)
    probability /= float(np.sum(probability))
    return probability


def local_peak_nms(
    score_map: np.ndarray,
    *,
    radius_px: int,
    max_reported_peaks: int = 3,
) -> LocalPeakDiagnostics:
    """先找 8-neighbour local maxima，再做固定半径 spatial NMS。

    这避免把同一宽峰上相邻的两个高像素错误解释为两个独立模式。平坦平台会
    产生多个候选，但 deterministic NMS 只保留平台上的一个代表点。
    """

    scores = _finite_map(score_map, "score_map")
    if not isinstance(radius_px, int) or isinstance(radius_px, bool) or radius_px < 1:
        raise ValueError("radius_px 必须是正整数")
    if (
        not isinstance(max_reported_peaks, int)
        or isinstance(max_reported_peaks, bool)
        or max_reported_peaks < 1
    ):
        raise ValueError("max_reported_peaks 必须是正整数")

    height, width = scores.shape
    padded = np.pad(scores, 1, mode="constant", constant_values=-np.inf)
    local = np.ones((height, width), dtype=np.bool_)
    for row_offset in range(3):
        for column_offset in range(3):
            if row_offset == 1 and column_offset == 1:
                continue
            neighbour = padded[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
            local &= scores >= neighbour

    candidate_y, candidate_x = np.nonzero(local)
    candidate_values = scores[candidate_y, candidate_x]
    order = np.lexsort((candidate_x, candidate_y, -candidate_values))
    suppressed = np.zeros((height, width), dtype=np.bool_)
    retained: list[LocalPeak] = []
    separated_count = 0
    radius_squared = radius_px * radius_px
    for candidate_index in order:
        y = int(candidate_y[candidate_index])
        x = int(candidate_x[candidate_index])
        if suppressed[y, x]:
            continue
        separated_count += 1
        if len(retained) < max_reported_peaks:
            retained.append(LocalPeak(pixel_uv=(float(x), float(y)), value=float(scores[y, x])))
        y0 = max(0, y - radius_px)
        y1 = min(height, y + radius_px + 1)
        x0 = max(0, x - radius_px)
        x1 = min(width, x + radius_px + 1)
        rows = np.arange(y0, y1, dtype=np.int32)[:, None]
        columns = np.arange(x0, x1, dtype=np.int32)[None, :]
        disk = (rows - y) ** 2 + (columns - x) ** 2 <= radius_squared
        suppressed[y0:y1, x0:x1] |= disk

    return LocalPeakDiagnostics(
        peaks=tuple(retained),
        local_maxima_count=int(local.sum()),
        separated_local_maxima_count=separated_count,
    )


def point_distance(first: Sequence[float], second: Sequence[float]) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != (2,) or right.shape != (2,):
        raise ValueError("point_distance 只接受两个 [2] 点")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("point_distance 输入必须有限")
    return float(np.linalg.norm(left - right))


def point_inside_mask(point_pixel_uv: Sequence[float], mask: np.ndarray) -> bool:
    point = np.asarray(point_pixel_uv, dtype=np.float64)
    value = np.asarray(mask)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("point_pixel_uv 必须是有限 [2]")
    if value.ndim != 2 or value.dtype != np.bool_:
        raise ValueError("mask 必须是 bool [H,W]")
    x = int(np.rint(point[0]))
    y = int(np.rint(point[1]))
    return bool(0 <= y < value.shape[0] and 0 <= x < value.shape[1] and value[y, x])


def semantic_distance_features(
    *,
    predicted_object_px: Sequence[float],
    predicted_goal_px: Sequence[float],
    gt_object_px: Sequence[float] | None,
    gt_goal_px: Sequence[float] | None,
) -> dict[str, Any]:
    """计算 object/goal 条件性换位的对称距离与 margin。

    margin 定义为 ``d(pred, correct) - d(pred, other)``；正数表示预测更接近
    另一个语义目标。
    """

    predicted_object = np.asarray(predicted_object_px, dtype=np.float64)
    predicted_goal = np.asarray(predicted_goal_px, dtype=np.float64)
    if predicted_object.shape != (2,) or predicted_goal.shape != (2,):
        raise ValueError("predicted object/goal 必须是 [2]")

    result: dict[str, Any] = {
        "d_pred_object_to_gt_object_px": None,
        "d_pred_object_to_gt_goal_px": None,
        "d_pred_goal_to_gt_goal_px": None,
        "d_pred_goal_to_gt_object_px": None,
        "object_semantic_margin_px": None,
        "goal_semantic_margin_px": None,
        "paired_object_goal_swap": False,
        "object_goal_distance_px": None,
    }
    object_gt = None if gt_object_px is None else np.asarray(gt_object_px, dtype=np.float64)
    goal_gt = None if gt_goal_px is None else np.asarray(gt_goal_px, dtype=np.float64)
    if object_gt is not None:
        result["d_pred_object_to_gt_object_px"] = point_distance(predicted_object, object_gt)
        result["d_pred_goal_to_gt_object_px"] = point_distance(predicted_goal, object_gt)
    if goal_gt is not None:
        result["d_pred_object_to_gt_goal_px"] = point_distance(predicted_object, goal_gt)
        result["d_pred_goal_to_gt_goal_px"] = point_distance(predicted_goal, goal_gt)
    if object_gt is not None and goal_gt is not None:
        object_margin = (
            result["d_pred_object_to_gt_object_px"] - result["d_pred_object_to_gt_goal_px"]
        )
        goal_margin = result["d_pred_goal_to_gt_goal_px"] - result["d_pred_goal_to_gt_object_px"]
        result["object_semantic_margin_px"] = float(object_margin)
        result["goal_semantic_margin_px"] = float(goal_margin)
        result["paired_object_goal_swap"] = bool(object_margin > 0.0 and goal_margin > 0.0)
        result["object_goal_distance_px"] = point_distance(object_gt, goal_gt)
    return result


def temporal_alignment_features(
    *,
    prediction_px: Sequence[float],
    current_gt_px: Sequence[float] | None,
    previous_gt_px: Sequence[float] | None,
    next_gt_px: Sequence[float] | None,
) -> dict[str, Any]:
    prediction = np.asarray(prediction_px, dtype=np.float64)
    if prediction.shape != (2,) or not np.isfinite(prediction).all():
        raise ValueError("prediction_px 必须是有限 [2]")

    def distance(value: Sequence[float] | None) -> float | None:
        return None if value is None else point_distance(prediction, value)

    distances = {
        -1: distance(previous_gt_px),
        0: distance(current_gt_px),
        1: distance(next_gt_px),
    }
    neighbour = [
        (offset, value) for offset, value in distances.items() if offset != 0 and value is not None
    ]
    best_offset: int | None = None
    best_distance: float | None = None
    improvement: float | None = None
    if neighbour:
        best_offset, best_distance = min(neighbour, key=lambda item: (float(item[1]), item[0]))
        if distances[0] is not None:
            improvement = float(distances[0] - best_distance)
    return {
        "prediction_to_gt_t_minus_1_px": distances[-1],
        "prediction_to_gt_t_px": distances[0],
        "prediction_to_gt_t_plus_1_px": distances[1],
        "best_adjacent_offset": best_offset,
        "best_adjacent_distance_px": best_distance,
        "adjacent_improvement_px": improvement,
    }


def geometry_conditioning(
    *,
    normalized_uv: np.ndarray,
    intrinsic_cv: np.ndarray,
    base_from_camera_cv: np.ndarray,
    image_size_hw: tuple[int, int],
    plane_base_z_m: float,
) -> dict[str, Any]:
    """保存 unit-ray 条件、物理射线距离与局部 ``dXY/duv`` Jacobian。"""

    normalized = np.asarray(normalized_uv, dtype=np.float64)
    intrinsic = np.asarray(intrinsic_cv, dtype=np.float64)
    transform = np.asarray(base_from_camera_cv, dtype=np.float64)
    height, width = image_size_hw
    pixel = normalized_uv_to_pixel(normalized, image_size_hw)
    direction_camera = np.linalg.solve(
        intrinsic,
        np.asarray((pixel[0], pixel[1], 1.0), dtype=np.float64),
    )
    direction_base = transform[:3, :3] @ direction_camera
    direction_norm = float(np.linalg.norm(direction_base))
    if direction_norm <= 1e-12:
        raise ValueError("相机射线方向退化")
    unit_ray = direction_base / direction_norm
    intersection = normalized_uv_to_base_z_plane(
        normalized,
        intrinsic,
        transform,
        image_size_hw,
        plane_base_z_m=plane_base_z_m,
    )

    def xy_at(pixel_uv: np.ndarray) -> np.ndarray:
        projected = normalized_uv_to_base_z_plane(
            pixel_to_normalized_uv(pixel_uv, image_size_hw),
            intrinsic,
            transform,
            image_size_hw,
            plane_base_z_m=plane_base_z_m,
        )
        return projected.point_base_m[:2].astype(np.float64)

    lower = np.asarray((-0.5, -0.5), dtype=np.float64)
    upper = np.asarray((width - 0.5, height - 0.5), dtype=np.float64)
    derivatives: list[np.ndarray] = []
    for axis in range(2):
        negative = pixel.copy()
        positive = pixel.copy()
        negative[axis] = max(lower[axis], pixel[axis] - 1.0)
        positive[axis] = min(upper[axis], pixel[axis] + 1.0)
        span = float(positive[axis] - negative[axis])
        if span <= 1e-12:
            raise ValueError("局部几何 Jacobian 无法取有限差分")
        derivatives.append((xy_at(positive) - xy_at(negative)) / span)
    jacobian = np.column_stack(derivatives)
    sigma_max = float(np.linalg.svd(jacobian, compute_uv=False)[0])
    return {
        "camera_position_base_m": transform[:3, 3].astype(float).tolist(),
        "camera_viewing_direction_base": transform[:3, 2].astype(float).tolist(),
        "ray_direction_base": direction_base.astype(float).tolist(),
        "unit_ray_direction_base": unit_ray.astype(float).tolist(),
        "n_dot_ray": float(direction_base[2]),
        "abs_n_dot_unit_ray": abs(float(unit_ray[2])),
        "ray_parameter": float(intersection.ray_parameter),
        "physical_ray_distance_m": float(intersection.ray_parameter * direction_norm),
        "predicted_world_point_base_m": intersection.point_base_m.astype(float).tolist(),
        "local_jacobian_xy_m_per_px": jacobian.astype(float).tolist(),
        "jacobian_sigma_max_mm_per_px": sigma_max * 1000.0,
    }


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values = [row.get(key) for row in rows]
    return np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )


def _quantile(values: np.ndarray, q: float, *, name: str) -> float:
    if values.size == 0:
        raise RuntimeError(f"{name} 没有有限 validation 样本")
    return float(np.quantile(values, q))


def derive_validation_rules(
    rows: Sequence[Mapping[str, Any]],
    *,
    heatmap_sigma_px: float,
) -> dict[str, Any]:
    """只用 frozen validation rows 封存 E014 taxonomy 数值规则。"""

    if not math.isfinite(heatmap_sigma_px) or heatmap_sigma_px <= 0.0:
        raise ValueError("heatmap_sigma_px 必须是有限正数")
    if not rows:
        raise ValueError("validation rows 不能为空")
    nms_radius = math.ceil(3.0 * heatmap_sigma_px)
    valid = [
        row
        for row in rows
        if bool(row.get("gt_keypoint_valid")) and row.get("world_xy_error_m") is not None
    ]
    if not valid:
        raise RuntimeError("validation 没有有效 keypoint")

    pixel = _finite_values(valid, "pixel_error_px")
    temporal_advantage = np.maximum(
        _finite_values(valid, "adjacent_improvement_px"),
        0.0,
    )
    semantic_margins = np.concatenate(
        (
            np.maximum(_finite_values(valid, "object_semantic_margin_px"), 0.0),
            np.maximum(_finite_values(valid, "goal_semantic_margin_px"), 0.0),
        )
    )
    top2_ratio = _finite_values(valid, "top2_top1_probability_ratio")
    soft_to_top1 = _finite_values(valid, "softargmax_to_top1_distance_px")
    top1_gain = np.maximum(
        _finite_values(valid, "softargmax_minus_top1_error_px"),
        0.0,
    )
    dot = _finite_values(valid, "abs_n_dot_unit_ray")
    jacobian = _finite_values(valid, "jacobian_sigma_max_mm_per_px")
    edge = _finite_values(valid, "gt_edge_distance_px")
    mask_fraction = _finite_values(valid, "gt_mask_area_fraction")
    oracle = _finite_values(valid, "oracle_roundtrip_error_px")

    thresholds = {
        "nms_radius_px": nms_radius,
        "catastrophic_world_error_m": 0.020,
        "large_pixel_error_px": max(
            float(2 * nms_radius),
            _quantile(pixel, 0.99, name="validation pixel error"),
        ),
        "small_pixel_error_px": _quantile(pixel, 0.90, name="validation pixel error"),
        "temporal_adjacent_improvement_px": max(
            float(nms_radius),
            _quantile(temporal_advantage, 0.995, name="validation temporal advantage"),
        ),
        "semantic_swap_margin_px": max(
            float(nms_radius),
            _quantile(semantic_margins, 0.99, name="validation semantic margin"),
        ),
        "multimodal_top2_top1_ratio_min": min(
            0.95,
            max(0.10, _quantile(top2_ratio, 0.99, name="validation top2/top1 ratio")),
        ),
        "multimodal_softargmax_to_top1_px": max(
            float(nms_radius) / 2.0,
            _quantile(soft_to_top1, 0.99, name="validation softargmax displacement"),
        ),
        "multimodal_top1_error_improvement_px": max(
            2.0,
            _quantile(top1_gain, 0.99, name="validation top1 improvement"),
        ),
        "geometry_abs_n_dot_unit_ray_max": _quantile(
            dot,
            0.01,
            name="validation abs(n dot unit ray)",
        ),
        "geometry_jacobian_sigma_min_mm_per_px": _quantile(
            jacobian,
            0.99,
            name="validation geometry Jacobian",
        ),
        "visibility_edge_distance_max_px": max(
            2.0,
            _quantile(edge, 0.01, name="validation edge distance"),
        ),
        "visibility_mask_area_fraction_max": _quantile(
            mask_fraction,
            0.01,
            name="validation mask area fraction",
        ),
        "label_oracle_roundtrip_error_max_px": max(
            1e-3,
            float(np.max(oracle)) * 10.0,
        ),
    }
    return {
        "version": E014_RULES_VERSION,
        "derivation_split": "val",
        "derivation_policy": "fixed-quantiles-before-test/v1",
        "heatmap_sigma_px": float(heatmap_sigma_px),
        "taxonomy_priority": list(FAILURE_TAXONOMY_PRIORITY),
        "thresholds": thresholds,
        "validation_population": {
            "prediction_rows": len(rows),
            "valid_keypoint_rows": len(valid),
        },
    }


def classify_outlier(row: Mapping[str, Any], rules: Mapping[str, Any]) -> str:
    """按冻结互斥优先级给一行预测分配且只分配一个 taxonomy。"""

    thresholds = rules.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise TypeError("rules.thresholds 缺失")
    expected_priority = list(FAILURE_TAXONOMY_PRIORITY)
    if list(rules.get("taxonomy_priority", ())) != expected_priority:
        raise ValueError("taxonomy priority 漂移")

    oracle = row.get("oracle_roundtrip_error_px")
    label_failure = bool(row.get("label_source_timestep_mismatch"))
    label_failure |= bool(row.get("gt_inside_other_mask") and not row.get("gt_inside_own_mask"))
    if oracle is not None:
        label_failure |= float(oracle) > float(thresholds["label_oracle_roundtrip_error_max_px"])
    if label_failure:
        return "label_or_channel_contract_failure"

    temporal = row.get("adjacent_improvement_px")
    adjacent_distance = row.get("best_adjacent_distance_px")
    if (
        temporal is not None
        and adjacent_distance is not None
        and float(temporal) >= float(thresholds["temporal_adjacent_improvement_px"])
        and float(adjacent_distance) <= float(thresholds["large_pixel_error_px"])
    ):
        return "temporal_alignment_failure"

    keypoint_name = row.get("keypoint_type")
    margin_key = (
        "object_semantic_margin_px"
        if keypoint_name == "object_center"
        else "goal_semantic_margin_px"
    )
    margin = row.get(margin_key)
    strong_other_mask = bool(
        (row.get("peak_inside_other_gt_mask") or row.get("softargmax_inside_other_gt_mask"))
        and not row.get("softargmax_inside_own_gt_mask")
    )
    if margin is not None and (
        float(margin) >= float(thresholds["semantic_swap_margin_px"])
        or strong_other_mask
        or bool(row.get("paired_object_goal_swap"))
    ):
        return "semantic_swap_failure"

    world_error = row.get("world_xy_error_m")
    pixel_error = row.get("pixel_error_px")
    catastrophic = world_error is not None and float(world_error) >= float(
        thresholds["catastrophic_world_error_m"]
    )
    bad_geometry = (
        row.get("abs_n_dot_unit_ray") is not None
        and float(row["abs_n_dot_unit_ray"]) <= float(thresholds["geometry_abs_n_dot_unit_ray_max"])
    ) or (
        row.get("jacobian_sigma_max_mm_per_px") is not None
        and float(row["jacobian_sigma_max_mm_per_px"])
        >= float(thresholds["geometry_jacobian_sigma_min_mm_per_px"])
    )
    if (
        catastrophic
        and pixel_error is not None
        and float(pixel_error) <= float(thresholds["small_pixel_error_px"])
        and bad_geometry
    ):
        return "geometry_conditioning_failure"

    if (
        row.get("softargmax_to_top1_distance_px") is not None
        and float(row["softargmax_to_top1_distance_px"])
        >= float(thresholds["multimodal_softargmax_to_top1_px"])
        and row.get("softargmax_minus_top1_error_px") is not None
        and float(row["softargmax_minus_top1_error_px"])
        >= float(thresholds["multimodal_top1_error_improvement_px"])
        and row.get("top2_top1_probability_ratio") is not None
        and float(row["top2_top1_probability_ratio"])
        >= float(thresholds["multimodal_top2_top1_ratio_min"])
        and int(row.get("separated_local_maxima_count", 0)) >= 2
    ):
        return "multimodal_softargmax_failure"

    visibility_signature = bool(
        row.get("confidence_accepted")
        and (
            not row.get("gt_visible")
            or not row.get("gt_projection_valid")
            or (
                row.get("gt_edge_distance_px") is not None
                and float(row["gt_edge_distance_px"])
                <= float(thresholds["visibility_edge_distance_max_px"])
            )
            or (
                row.get("gt_mask_area_fraction") is not None
                and float(row["gt_mask_area_fraction"])
                <= float(thresholds["visibility_mask_area_fraction_max"])
            )
        )
    )
    if visibility_signature:
        return "visibility_or_ood_failure"

    if bool(row.get("gt_keypoint_valid")) and (
        catastrophic
        or (
            pixel_error is not None
            and float(pixel_error) >= float(thresholds["large_pixel_error_px"])
        )
    ):
        return "generic_correspondence_failure"
    return "unclear_or_mixed"


def _distribution(values_m: np.ndarray) -> dict[str, Any]:
    if values_m.size == 0:
        return {
            "count": 0,
            "p50_mm": None,
            "p90_mm": None,
            "p95_mm": None,
            "p99_mm": None,
            "max_mm": None,
            "threshold_counts": {f"over_{int(mm)}mm": 0 for mm in WORLD_ERROR_THRESHOLDS_MM},
        }
    values_mm = values_m * 1000.0
    return {
        "count": int(values_mm.size),
        "p50_mm": float(np.quantile(values_mm, 0.50)),
        "p90_mm": float(np.quantile(values_mm, 0.90)),
        "p95_mm": float(np.quantile(values_mm, 0.95)),
        "p99_mm": float(np.quantile(values_mm, 0.99)),
        "max_mm": float(np.max(values_mm)),
        "threshold_counts": {
            f"over_{int(mm)}mm": int(np.sum(values_mm > mm)) for mm in WORLD_ERROR_THRESHOLDS_MM
        },
    }


def aggregate_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("prediction rows 不能为空")
    valid = [
        row
        for row in rows
        if bool(row.get("gt_keypoint_valid")) and row.get("world_xy_error_m") is not None
    ]
    object_rows = [row for row in valid if row.get("keypoint_type") == "object_center"]
    goal_rows = [row for row in valid if row.get("keypoint_type") == "goal_center"]
    accepted_all = [row for row in rows if bool(row.get("confidence_accepted"))]
    accepted_valid = [row for row in valid if bool(row.get("confidence_accepted"))]

    def world(items: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.asarray([float(row["world_xy_error_m"]) for row in items], dtype=np.float64)

    accepted_world = world(accepted_valid)
    accepted_count = len(accepted_valid)
    accuracy = {
        f"accepted_accuracy_rate_at_{int(mm)}mm": (
            None if accepted_count == 0 else float(np.mean(accepted_world <= mm / 1000.0))
        )
        for mm in (5.0, 10.0, 20.0)
    }
    catastrophic_accepted = int(np.sum(accepted_world > 0.020))
    catastrophic_all = int(np.sum(world(valid) > 0.020))
    return {
        "prediction_row_count": len(rows),
        "valid_keypoint_count": len(valid),
        "invalid_keypoint_count": len(rows) - len(valid),
        "world_error": {
            "all": _distribution(world(valid)),
            "object_center": _distribution(world(object_rows)),
            "goal_center": _distribution(world(goal_rows)),
            "confidence_accepted": _distribution(accepted_world),
        },
        "confidence": {
            "semantics": "accepted-validity-and-localization-audit/v1",
            "accepted_row_count": len(accepted_all),
            "accepted_valid_keypoint_count": accepted_count,
            "accepted_invalid_keypoint_count": len(accepted_all) - accepted_count,
            "accepted_validity_precision": (
                None if not accepted_all else float(accepted_count / len(accepted_all))
            ),
            **accuracy,
            "accepted_over_20mm_count": catastrophic_accepted,
            "accepted_over_50mm_count": int(np.sum(accepted_world > 0.050)),
            "accepted_over_100mm_count": int(np.sum(accepted_world > 0.100)),
            "accepted_max_error_mm": (
                None if accepted_count == 0 else float(np.max(accepted_world) * 1000.0)
            ),
            "false_safe_catastrophic_rate": (
                None if accepted_count == 0 else float(catastrophic_accepted / accepted_count)
            ),
            "catastrophic_acceptance_rate": (
                None if catastrophic_all == 0 else float(catastrophic_accepted / catastrophic_all)
            ),
        },
    }


def taxonomy_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter = Counter(str(row.get("failure_taxonomy")) for row in rows)
    unknown = set(counter) - set(FAILURE_TAXONOMY_PRIORITY)
    if unknown:
        raise ValueError(f"未知 failure taxonomy: {sorted(unknown)}")
    result = {name: int(counter.get(name, 0)) for name in FAILURE_TAXONOMY_PRIORITY}
    if sum(result.values()) != len(rows):
        raise RuntimeError("taxonomy 计数未精确覆盖输入 rows")
    return result


def sample_fingerprint(
    *,
    trajectory_id: str,
    dataset_index: int,
    timestep: int,
    keypoint_type: str,
) -> str:
    payload = (
        f"{E014_DIAGNOSTIC_VERSION}\0{trajectory_id}\0{dataset_index}\0{timestep}\0{keypoint_type}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def assert_public_payload_safe(value: Any) -> None:
    """拒绝把原始身份、路径或私有数组引用写入 GitHub 聚合 JSON。"""

    banned_keys = {
        "trajectory_id",
        "dataset_index",
        "timestep",
        "camera_position_base_m",
        "camera_viewing_direction_base",
        "ray_direction_base",
        "unit_ray_direction_base",
        "predicted_world_point_base_m",
        "gt_world_xy_m",
        "predicted_world_xy_m",
    }
    path_prefixes = ("/home/", "/mnt/", "C:\\", "D:\\")

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            intersection = banned_keys & set(item)
            if intersection:
                raise ValueError(f"public payload 含私有 key: {sorted(intersection)}")
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, str):
            if item.startswith(path_prefixes):
                raise ValueError("public payload 含敏感绝对路径")

    visit(value)


__all__ = [
    "E014_DIAGNOSTIC_VERSION",
    "E014_RULES_VERSION",
    "FAILURE_TAXONOMY_PRIORITY",
    "LocalPeak",
    "LocalPeakDiagnostics",
    "aggregate_prediction_rows",
    "assert_public_payload_safe",
    "classify_outlier",
    "derive_validation_rules",
    "geometry_conditioning",
    "local_peak_nms",
    "point_distance",
    "point_inside_mask",
    "probability_from_logits",
    "sample_fingerprint",
    "semantic_distance_features",
    "taxonomy_counts",
    "temporal_alignment_features",
]
