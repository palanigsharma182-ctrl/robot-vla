import pytest

from robot_vla.sim.local_dagger_risk import (
    RISK_COMPONENT_UNITS,
    compute_paired_risk_components,
    midrank_empirical_percentiles,
    score_and_select_risk_candidates,
)


def _boundary(boundary_type: str, **overrides):
    value = {
        "boundary_type": boundary_type,
        "tcp_object_relative_xyz_m": (0.01, -0.02, 0.03),
        "tcp_object_xy_error_m": 0.04,
        "tcp_object_relative_z_m": 0.03,
        "tcp_linear_speed_m_s": 0.05,
        "joint_velocity_rms_rad_s": 0.06,
        "gripper_opening": 0.7,
        "object_linear_speed_m_s": 0.08,
        "object_angular_speed_rad_s": 0.09,
        "robot_object_contact_force_n": 2.0,
        "is_grasped": True,
        "arm_mean_pairwise_disagreement": 0.1,
        "gripper_mean_pairwise_disagreement": 0.2,
    }
    value.update(overrides)
    return value


def test_paired_risk_components_follow_preregistered_boundary_formulas() -> None:
    rg = compute_paired_risk_components(
        "reach_grasp",
        _boundary("reach_grasp"),
        _boundary(
            "reach_grasp",
            tcp_object_relative_z_m=0.01,
            gripper_opening=0.5,
        ),
    )
    assert set(rg) == set(RISK_COMPONENT_UNITS["reach_grasp"])
    assert rg["relative_z_deviation_m"] == pytest.approx(0.02)
    assert rg["gripper_opening_deviation"] == pytest.approx(0.2)

    gl = compute_paired_risk_components(
        "grasp_lift",
        _boundary("grasp_lift"),
        _boundary(
            "grasp_lift",
            tcp_object_relative_xyz_m=(0.01, -0.02, 0.01),
            gripper_opening=0.5,
        ),
    )
    assert set(gl) == set(RISK_COMPONENT_UNITS["grasp_lift"])
    assert gl["object_tcp_relative_position_deviation_m"] == pytest.approx(0.02)
    assert gl["contact_grasp_instability"] == 0.0


def test_midrank_percentiles_handle_ties_without_seed_dependent_jitter() -> None:
    assert midrank_empirical_percentiles([3.0, 1.0, 1.0, 5.0]) == pytest.approx(
        (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0)
    )


def test_risk_selection_is_unique_and_deterministic_under_complete_ties() -> None:
    names = RISK_COMPONENT_UNITS["reach_grasp"]
    candidates = [
        {
            "environment_seed": seed,
            "risk_components": {name: 1.0 for name in names},
        }
        for seed in range(30_000, 30_025)
    ]

    result = score_and_select_risk_candidates("reach_grasp", candidates)

    assert result.high_risk_seeds == tuple(range(30_000, 30_014))
    assert result.low_risk_seeds == tuple(range(30_014, 30_020))
    assert len(set(result.high_risk_seeds + result.low_risk_seeds)) == 20
    assert {row["risk_score"] for row in result.scored_candidates} == {0.5}
