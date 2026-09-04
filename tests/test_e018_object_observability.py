from __future__ import annotations

import numpy as np
import pytest

from robot_vla.precision.object_observability import (
    OBJECT_WRITE_SCORE_SEMANTICS,
    ObjectWriteEvidence,
    derive_object_observability,
)


def test_object_center_on_own_mask_is_observable() -> None:
    object_mask = np.zeros((8, 8), dtype=np.bool_)
    goal_mask = np.zeros_like(object_mask)
    object_mask[4, 4] = True

    label = derive_object_observability(
        object_exists=True,
        projection_valid=True,
        projected_normalized_uv=np.asarray(((4.5 / 8.0), (4.5 / 8.0))),
        object_mask=object_mask,
        goal_mask=goal_mask,
        legacy_visible=True,
        support_radius_px=1,
    )

    assert label.observable
    assert label.in_fov
    assert label.center_inside_object_mask
    assert label.occlusion_type == "observable"
    assert not label.legacy_contract_mismatch


def test_other_instance_at_object_center_is_goal_occlusion() -> None:
    object_mask = np.zeros((8, 8), dtype=np.bool_)
    goal_mask = np.zeros_like(object_mask)
    goal_mask[4, 4] = True

    label = derive_object_observability(
        object_exists=True,
        projection_valid=True,
        projected_normalized_uv=(4.5 / 8.0, 4.5 / 8.0),
        object_mask=object_mask,
        goal_mask=goal_mask,
        legacy_visible=True,
    )

    assert not label.observable
    assert label.center_inside_goal_mask
    assert label.occlusion_type == "goal_occlusion"
    assert label.legacy_contract_mismatch


def test_object_write_evidence_uses_only_deployable_predictions() -> None:
    evidence = ObjectWriteEvidence(
        visibility_probability=0.9,
        projection_validity_probability=0.8,
        object_mask_probability=0.95,
        goal_mask_probability=0.1,
        normalized_entropy=0.2,
        radial_sigma_px=0.25,
        geometry_valid=True,
    )

    assert evidence.observable
    assert evidence.structurally_eligible
    assert evidence.score == pytest.approx(0.8)
    assert evidence.accepted(threshold=0.79)
    assert not evidence.accepted(threshold=0.81)
    assert evidence.to_dict()["score_semantics"] == OBJECT_WRITE_SCORE_SEMANTICS


def test_object_write_requires_own_mask_and_valid_geometry() -> None:
    no_mask = ObjectWriteEvidence(
        visibility_probability=1.0,
        projection_validity_probability=1.0,
        object_mask_probability=0.49,
        goal_mask_probability=0.0,
        normalized_entropy=0.0,
        radial_sigma_px=0.0,
        geometry_valid=True,
    )
    bad_geometry = ObjectWriteEvidence(
        visibility_probability=1.0,
        projection_validity_probability=1.0,
        object_mask_probability=1.0,
        goal_mask_probability=0.0,
        normalized_entropy=0.0,
        radial_sigma_px=0.0,
        geometry_valid=False,
    )

    assert not no_mask.structurally_eligible
    assert not bad_geometry.structurally_eligible


def test_projection_contract_rejects_missing_uv() -> None:
    mask = np.zeros((8, 8), dtype=np.bool_)
    with pytest.raises(ValueError, match="projected_normalized_uv"):
        derive_object_observability(
            object_exists=True,
            projection_valid=True,
            projected_normalized_uv=None,
            object_mask=mask,
            goal_mask=mask,
            legacy_visible=False,
        )
