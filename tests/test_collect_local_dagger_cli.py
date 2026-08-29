import pytest

from robot_vla.cli.collect_local_dagger import derive_collection_sampling_seed


def test_collection_sampling_seed_is_paired_and_boundary_specific() -> None:
    first = derive_collection_sampling_seed(
        52_012,
        environment_seed=29_990,
        boundary_type="reach_grasp",
    )
    repeated = derive_collection_sampling_seed(
        52_012,
        environment_seed=29_990,
        boundary_type="reach_grasp",
    )
    other_boundary = derive_collection_sampling_seed(
        52_012,
        environment_seed=29_990,
        boundary_type="grasp_lift",
    )

    assert first == repeated
    assert first != other_boundary


def test_collection_sampling_seed_rejects_invalid_identity() -> None:
    with pytest.raises(ValueError, match="未知 boundary"):
        derive_collection_sampling_seed(
            52_012,
            environment_seed=29_990,
            boundary_type="lift_transport",
        )
