from __future__ import annotations

import numpy as np
import pytest

from robot_vla.precision.outliers import (
    FAILURE_FAMILY_ORDER,
    FAILURE_TAXONOMY_PRIORITY,
    aggregate_prediction_rows,
    assert_public_payload_safe,
    classify_outlier,
    derive_validation_rules,
    failure_family,
    failure_family_counts,
    geometry_conditioning,
    local_peak_nms,
    semantic_distance_features,
    taxonomy_counts,
    temporal_alignment_features,
)


def _rules() -> dict[str, object]:
    return {
        "taxonomy_priority": list(FAILURE_TAXONOMY_PRIORITY),
        "thresholds": {
            "label_oracle_roundtrip_error_max_px": 0.01,
            "temporal_adjacent_improvement_px": 5.0,
            "large_pixel_error_px": 10.0,
            "semantic_swap_margin_px": 5.0,
            "catastrophic_world_error_m": 0.020,
            "small_pixel_error_px": 1.0,
            "geometry_abs_n_dot_unit_ray_max": 0.2,
            "geometry_jacobian_sigma_min_mm_per_px": 5.0,
            "multimodal_softargmax_to_top1_px": 3.0,
            "multimodal_top1_error_improvement_px": 2.0,
            "multimodal_top2_top1_ratio_min": 0.5,
            "visibility_edge_distance_max_px": 2.0,
            "visibility_mask_area_fraction_max": 0.01,
        },
    }


def _row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "keypoint_type": "object_center",
        "gt_keypoint_valid": True,
        "gt_visible": True,
        "gt_projection_valid": True,
        "world_xy_error_m": 0.001,
        "pixel_error_px": 0.5,
        "confidence_accepted": False,
        "label_source_timestep_mismatch": False,
        "gt_inside_other_mask": False,
        "gt_inside_own_mask": True,
        "oracle_roundtrip_error_px": 0.0,
        "adjacent_improvement_px": 0.0,
        "best_adjacent_distance_px": 0.5,
        "object_semantic_margin_px": -10.0,
        "goal_semantic_margin_px": -10.0,
        "peak_inside_other_gt_mask": False,
        "softargmax_inside_other_gt_mask": False,
        "softargmax_inside_own_gt_mask": True,
        "paired_object_goal_swap": False,
        "abs_n_dot_unit_ray": 0.8,
        "jacobian_sigma_max_mm_per_px": 1.0,
        "softargmax_to_top1_distance_px": 0.2,
        "softargmax_minus_top1_error_px": 0.0,
        "top2_top1_probability_ratio": 0.1,
        "separated_local_maxima_count": 2,
        "gt_edge_distance_px": 20.0,
        "gt_mask_area_fraction": 0.1,
    }
    row.update(updates)
    return row


def test_local_peak_nms_does_not_treat_adjacent_pixels_as_two_modes() -> None:
    scores = np.full((12, 12), -10.0, dtype=np.float32)
    scores[3, 3] = 10.0
    scores[3, 4] = 9.0
    scores[9, 9] = 8.0

    result = local_peak_nms(scores, radius_px=3)

    assert result.peaks[0].pixel_uv == (3.0, 3.0)
    assert result.peaks[1].pixel_uv == (9.0, 9.0)
    assert result.local_maxima_count >= 2
    assert result.separated_local_maxima_count >= 2


def test_semantic_and_temporal_counterfactual_features_have_explicit_sign() -> None:
    semantic = semantic_distance_features(
        predicted_object_px=(10.0, 0.0),
        predicted_goal_px=(0.0, 0.0),
        gt_object_px=(0.0, 0.0),
        gt_goal_px=(10.0, 0.0),
    )
    assert semantic["object_semantic_margin_px"] == 10.0
    assert semantic["goal_semantic_margin_px"] == 10.0
    assert semantic["paired_object_goal_swap"]

    temporal = temporal_alignment_features(
        prediction_px=(10.0, 0.0),
        current_gt_px=(0.0, 0.0),
        previous_gt_px=(9.0, 0.0),
        next_gt_px=(20.0, 0.0),
    )
    assert temporal["best_adjacent_offset"] == -1
    assert temporal["adjacent_improvement_px"] == 9.0


def test_geometry_conditioning_reports_unit_ray_and_metric_jacobian() -> None:
    intrinsic = np.asarray(((100.0, 0.0, 50.0), (0.0, 100.0, 50.0), (0.0, 0.0, 1.0)))
    transform = np.eye(4, dtype=np.float64)
    transform[2, 3] = 1.0

    result = geometry_conditioning(
        normalized_uv=np.asarray((0.505, 0.505), dtype=np.float32),
        intrinsic_cv=intrinsic,
        base_from_camera_cv=transform,
        image_size_hw=(100, 100),
        plane_base_z_m=2.0,
    )

    assert result["abs_n_dot_unit_ray"] == pytest.approx(1.0)
    assert result["physical_ray_distance_m"] == pytest.approx(1.0)
    assert result["jacobian_sigma_max_mm_per_px"] == pytest.approx(10.0, rel=1e-4)
    assert result["gt_plane_z_m"] == pytest.approx(2.0)


def test_taxonomy_is_mutually_exclusive_and_honours_priority() -> None:
    rules = _rules()
    label_and_swap = _row(
        oracle_roundtrip_error_px=1.0,
        object_semantic_margin_px=20.0,
    )
    assert classify_outlier(label_and_swap, rules) == "label_or_channel_contract_failure"
    assert (
        classify_outlier(
            _row(adjacent_improvement_px=8.0, best_adjacent_distance_px=1.0),
            rules,
        )
        == "temporal_alignment_failure"
    )
    assert classify_outlier(_row(object_semantic_margin_px=20.0), rules) == "semantic_swap_failure"
    assert (
        classify_outlier(
            _row(
                world_xy_error_m=0.1,
                pixel_error_px=0.5,
                abs_n_dot_unit_ray=0.1,
            ),
            rules,
        )
        == "geometry_conditioning_failure"
    )
    assert (
        classify_outlier(
            _row(
                softargmax_to_top1_distance_px=5.0,
                softargmax_minus_top1_error_px=4.0,
                top2_top1_probability_ratio=0.8,
            ),
            rules,
        )
        == "multimodal_softargmax_failure"
    )
    assert (
        classify_outlier(
            _row(
                gt_keypoint_valid=False,
                gt_visible=False,
                confidence_accepted=True,
            ),
            rules,
        )
        == "visibility_or_ood_failure"
    )
    assert (
        classify_outlier(
            _row(world_xy_error_m=0.1, pixel_error_px=30.0),
            rules,
        )
        == "generic_correspondence_failure"
    )


def test_validation_rule_derivation_and_aggregate_keep_confidence_semantics_separate() -> None:
    validation_rows = []
    for index in range(100):
        validation_rows.append(
            _row(
                world_xy_error_m=0.001 + index * 1e-6,
                pixel_error_px=0.2 + index * 0.001,
                adjacent_improvement_px=float(index % 3) * 0.1,
                object_semantic_margin_px=-20.0 + index * 0.01,
                goal_semantic_margin_px=-18.0 + index * 0.01,
                top2_top1_probability_ratio=0.1 + index * 0.001,
                softargmax_to_top1_distance_px=0.2 + index * 0.001,
                softargmax_minus_top1_error_px=0.1,
                abs_n_dot_unit_ray=0.6 + index * 0.001,
                jacobian_sigma_max_mm_per_px=1.0 + index * 0.01,
                gt_edge_distance_px=10.0 + index * 0.1,
                gt_mask_area_fraction=0.05 + index * 0.0001,
                oracle_roundtrip_error_px=1e-8,
            )
        )
    rules = derive_validation_rules(validation_rows, heatmap_sigma_px=1.5)
    assert rules["thresholds"]["nms_radius_px"] == 5
    assert rules["derivation_split"] == "val"

    rows = [
        {
            "keypoint_type": "object_center",
            "gt_keypoint_valid": True,
            "world_xy_error_m": 0.004,
            "confidence_accepted": True,
            "failure_taxonomy": "unclear_or_mixed",
        },
        {
            "keypoint_type": "goal_center",
            "gt_keypoint_valid": True,
            "world_xy_error_m": 0.060,
            "confidence_accepted": True,
            "failure_taxonomy": "generic_correspondence_failure",
        },
        {
            "keypoint_type": "goal_center",
            "gt_keypoint_valid": False,
            "world_xy_error_m": None,
            "confidence_accepted": True,
            "failure_taxonomy": "visibility_or_ood_failure",
        },
    ]
    aggregate = aggregate_prediction_rows(rows)
    assert aggregate["confidence"]["accepted_validity_precision"] == pytest.approx(2 / 3)
    assert aggregate["confidence"]["accepted_accuracy_rate_at_5mm"] == pytest.approx(0.5)
    assert aggregate["confidence"]["accepted_over_50mm_count"] == 1
    assert sum(taxonomy_counts(rows).values()) == 3
    assert failure_family("semantic_swap_failure") == "correspondence_failure"
    family_counts = failure_family_counts(rows)
    assert set(family_counts) == set(FAILURE_FAMILY_ORDER)
    assert family_counts["correspondence_failure"] == 1
    assert family_counts["visibility_or_ood_failure"] == 1
    assert family_counts["unclear_or_mixed"] == 1
    assert sum(family_counts.values()) == 3


def test_failure_family_rejects_unknown_taxonomy() -> None:
    with pytest.raises(ValueError, match="未知 failure taxonomy"):
        failure_family("invented_failure")


def test_public_payload_rejects_raw_identity_or_absolute_path() -> None:
    assert_public_payload_safe({"sample_fingerprint": "a" * 20, "count": 1})
    with pytest.raises(ValueError, match="私有 key"):
        assert_public_payload_safe({"trajectory_id": "secret"})
    with pytest.raises(ValueError, match="绝对路径"):
        assert_public_payload_safe({"source": "/home/user/private"})
