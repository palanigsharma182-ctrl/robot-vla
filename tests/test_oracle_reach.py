from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from robot_vla.adapters import ProprioNormalizer, ProprioStats
from robot_vla.contracts import RobotSpec
from robot_vla.data.events import EVENT_STATE_CONTRACT_VERSION
from robot_vla.diagnostics.oracle_reach import (
    OracleGeometryContextEncoder,
    OracleGeometryPolicy,
    OracleReachCollator,
    OracleReachDataset,
    oracle_case,
    parameter_state_sha256,
)
from robot_vla.model.expert import ExpertConfig, StandaloneActionExpert


def _normalizer(spec: RobotSpec) -> ProprioNormalizer:
    return ProprioNormalizer(
        ProprioStats(
            mean=(0.0,) * spec.proprio_dim,
            std=(1.0,) * spec.proprio_dim,
            count=5,
        ),
        spec,
    )


def _event_state_arrays(steps: int, object_position_m: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "robot_object_contact_force_n": np.zeros(steps, dtype=np.float32),
        "support_contact_force_n": np.ones(steps, dtype=np.float32),
        "is_grasped": np.zeros(steps, dtype=np.bool_),
        "object_position_m": object_position_m,
        "object_linear_velocity_m_s": np.zeros((steps, 3), dtype=np.float32),
        "object_angular_velocity_rad_s": np.zeros((steps, 3), dtype=np.float32),
        "commanded_joint_target_rad": np.zeros((steps, 7), dtype=np.float32),
        "applied_joint_correction_rad": np.zeros((steps, 7), dtype=np.float32),
    }


def test_oracle_encoder_returns_single_720d_context_and_validates_distance() -> None:
    encoder = OracleGeometryContextEncoder()
    geometry = torch.tensor(
        [[0.03, -0.04, 0.0, 0.05], [0.0, 0.0, 0.1, 0.1]],
        dtype=torch.float32,
    )
    context = encoder(geometry)
    assert context.tokens.shape == (2, 1, 720)
    assert context.mask.shape == (2, 1)
    assert context.mask.dtype == torch.bool
    assert context.mask.all()
    assert torch.isfinite(context.tokens).all()

    invalid = geometry.clone()
    invalid[0, 3] = 0.5
    with pytest.raises(ValueError, match="distance"):
        encoder(invalid)


def test_oracle_policy_reuses_existing_flow_loss_and_expert() -> None:
    config = ExpertConfig(
        hidden_size=60,
        state_hidden_size=16,
        num_layers=2,
        intermediate_size=64,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=20,
    )
    policy = OracleGeometryPolicy(StandaloneActionExpert(config))
    geometry = torch.tensor([[0.03, 0.04, 0.0, 0.05]], dtype=torch.float32)
    action = torch.zeros(1, config.action_horizon, config.action_dim)
    mask = torch.ones(1, config.action_horizon, dtype=torch.bool)
    output = policy.flow_matching_loss(
        {"relative_geometry": geometry},
        torch.zeros(1, config.proprio_dim),
        action,
        mask,
        event_loss_weight=0.0,
        generator=torch.Generator().manual_seed(42),
    )
    assert output.prediction.shape == action.shape
    assert output.loss.item() == output.base_loss.item()
    assert output.critical_mask.shape == mask.shape


def test_reach_dataset_uses_current_geometry_only_and_filters_anchor_skill(
    tmp_path,
    meta_factory,
    arrays_factory,
    write_dataset,
) -> None:
    spec = RobotSpec()
    meta = meta_factory(
        randomization={
            "seed": 7,
            "event_state_contract_version": EVENT_STATE_CONTRACT_VERSION,
        }
    )
    object_position = np.asarray(
        [
            [0.10, 0.20, 0.30],
            [0.11, 0.21, 0.31],
            [0.12, 0.22, 0.32],
            [0.13, 0.23, 0.33],
            [0.14, 0.24, 0.34],
        ],
        dtype=np.float32,
    )
    arrays = arrays_factory(
        skill_id=np.asarray([0, 1, 0, 2, 3], dtype=np.int16),
        **_event_state_arrays(5, object_position),
    )
    write_dataset(meta, arrays)

    def fake_fk(arm_q: np.ndarray) -> np.ndarray:
        return np.asarray([arm_q[0], arm_q[1], arm_q[2]], dtype=np.float32)

    dataset = OracleReachDataset(
        str(tmp_path), [meta], spec, _normalizer(spec), fake_fk
    )
    assert len(dataset) == 2
    assert [dataset[index]["timestep"] for index in range(len(dataset))] == [0, 2]
    assert all(dataset[index]["skill_id"] == 0 for index in range(len(dataset)))
    first = dataset[0]["relative_geometry"]
    expected_delta = np.asarray([0.10, 0.70, 0.30], dtype=np.float32)
    np.testing.assert_allclose(first[:3], expected_delta)
    assert first[3] == pytest.approx(float(np.linalg.norm(expected_delta)))

    changed_future = object_position.copy()
    changed_future[1:] = 0.8
    write_dataset(
        meta,
        replace(
            arrays,
            **_event_state_arrays(5, changed_future),
        ),
    )
    rebuilt = OracleReachDataset(
        str(tmp_path), [meta], spec, _normalizer(spec), fake_fk
    )
    np.testing.assert_array_equal(rebuilt[0]["relative_geometry"], first)
    assert rebuilt.window_sha256 == dataset.window_sha256


def test_oracle_collator_keeps_full_action_chunk_and_masks(
    tmp_path,
    meta_factory,
    arrays_factory,
    write_dataset,
) -> None:
    spec = RobotSpec()
    meta = meta_factory(
        randomization={
            "seed": 7,
            "event_state_contract_version": EVENT_STATE_CONTRACT_VERSION,
        }
    )
    object_position = np.full((5, 3), 0.1, dtype=np.float32)
    arrays = arrays_factory(
        skill_id=np.zeros(5, dtype=np.int16),
        **_event_state_arrays(5, object_position),
    )
    write_dataset(meta, arrays)
    dataset = OracleReachDataset(
        str(tmp_path),
        [meta],
        spec,
        _normalizer(spec),
        lambda _: np.zeros(3, dtype=np.float32),
    )
    batch = OracleReachCollator("oracle", spec)([dataset[0], dataset[4]])
    assert batch["action"].shape == (2, 16, 8)
    assert batch["action_mask"].sum(dim=1).tolist() == [5, 1]
    assert batch["qwen_inputs"]["relative_geometry"].shape == (2, 4)
    assert batch["event_mask"].dtype == torch.bool
    assert batch["skill_id"].tolist() == [0, 0]


def test_expert_initialization_hash_and_predefined_case_boundaries() -> None:
    torch.manual_seed(42)
    first = StandaloneActionExpert(
        ExpertConfig(
            hidden_size=60,
            state_hidden_size=16,
            num_layers=2,
            intermediate_size=64,
            num_attention_heads=3,
            num_key_value_heads=1,
            head_dim=20,
        )
    )
    torch.manual_seed(42)
    second = StandaloneActionExpert(first.config)
    assert parameter_state_sha256(first) == parameter_state_sha256(second)
    assert oracle_case(5) == "case_1"
    assert oracle_case(4) == "case_1"
    assert oracle_case(3) == "case_3"
    assert oracle_case(2) == "case_3"
    assert oracle_case(1) == "case_2"
    assert oracle_case(0) == "case_2"
