from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robot_vla.precision.checkpoint import (
    PRECISION_CHECKPOINT_FORMAT,
    PrecisionCheckpointProvenance,
    PrecisionCheckpointRole,
    load_precision_checkpoint,
    load_torch_precision_frame_predictor,
    save_precision_checkpoint,
)
from robot_vla.precision.model import PrecisionThreeHeadUNet, PrecisionUNetConfig
from robot_vla.precision.provider import TorchPrecisionFramePredictorConfig


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> PrecisionUNetConfig:
    return PrecisionUNetConfig(
        encoder_channels=(8, 16, 32),
        structured_state_dim=6,
        state_hidden_size=8,
        head_hidden_size=16,
    )


def _provenance() -> PrecisionCheckpointProvenance:
    return PrecisionCheckpointProvenance(
        role=PrecisionCheckpointRole.SYNTHETIC_DEBUG,
        data_identity_sha256="1" * 64,
        training_config_sha256="2" * 64,
        source_tree_sha256="3" * 64,
        seed=13013,
        examples_seen=32,
        optimizer_steps=4,
    )


def test_precision_checkpoint_roundtrip_loads_verified_frozen_predictor(tmp_path) -> None:
    torch.manual_seed(13)
    model = PrecisionThreeHeadUNet(_config())
    checkpoint_path = tmp_path / "precision.pt"

    saved = save_precision_checkpoint(checkpoint_path, model, _provenance())

    assert saved.format_version == PRECISION_CHECKPOINT_FORMAT
    assert saved.checkpoint_sha256 == _sha256(checkpoint_path)
    assert os.stat(checkpoint_path).st_mode & 0o777 == 0o600
    loaded = load_precision_checkpoint(
        checkpoint_path,
        expected_checkpoint_sha256=saved.checkpoint_sha256,
        expected_provenance_sha256=saved.provenance_sha256,
    )
    assert loaded.receipt == saved
    assert loaded.provenance == _provenance()
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[name], expected)

    loaded_predictor = load_torch_precision_frame_predictor(
        checkpoint_path,
        expected_checkpoint_sha256=saved.checkpoint_sha256,
        expected_provenance_sha256=saved.provenance_sha256,
        predictor_config=TorchPrecisionFramePredictorConfig(device="cpu"),
    )
    prediction = loaded_predictor.predictor.predict(
        np.full((32, 40, 3), 127, dtype=np.uint8),
        np.zeros(_config().structured_state_dim, dtype=np.float32),
        np.zeros(_config().motion_spec.motion_dim, dtype=np.float32),
    )
    assert prediction.keypoints.normalized_uv.shape == (1, 2, 2)
    assert loaded_predictor.predictor.identity.checkpoint_sha256 == saved.checkpoint_sha256
    assert (
        loaded_predictor.predictor.identity.parameter_state_sha256 == saved.parameter_state_sha256
    )
    loaded_predictor.predictor.verify_identity()


def test_precision_checkpoint_refuses_overwrite_or_external_identity_drift(tmp_path) -> None:
    checkpoint_path = tmp_path / "precision.pt"
    receipt = save_precision_checkpoint(
        checkpoint_path,
        PrecisionThreeHeadUNet(_config()),
        _provenance(),
    )

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        save_precision_checkpoint(
            checkpoint_path,
            PrecisionThreeHeadUNet(_config()),
            _provenance(),
        )
    with pytest.raises(RuntimeError, match="文件 SHA-256"):
        load_precision_checkpoint(
            checkpoint_path,
            expected_checkpoint_sha256="f" * 64,
        )
    with pytest.raises(RuntimeError, match="provenance 与预期"):
        load_precision_checkpoint(
            checkpoint_path,
            expected_checkpoint_sha256=receipt.checkpoint_sha256,
            expected_provenance_sha256="e" * 64,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload["model_state"]["motion_head.2.bias"].add_(1.0),
            "parameter state SHA-256",
        ),
        (
            lambda payload: payload["model_config"].__setitem__("head_hidden_size", 99),
            "model config SHA-256",
        ),
        (
            lambda payload: payload.__setitem__("unexpected", True),
            "checkpoint keys 漂移",
        ),
    ),
)
def test_precision_checkpoint_rejects_internal_payload_drift(
    tmp_path,
    mutation,
    message: str,
) -> None:
    original = tmp_path / "original.pt"
    save_precision_checkpoint(
        original,
        PrecisionThreeHeadUNet(_config()),
        _provenance(),
    )
    payload = torch.load(original, map_location="cpu", weights_only=True)
    mutation(payload)
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)

    with pytest.raises((RuntimeError, ValueError), match=message):
        load_precision_checkpoint(
            tampered,
            expected_checkpoint_sha256=_sha256(tampered),
        )


def test_precision_checkpoint_provenance_requires_real_training_progress() -> None:
    with pytest.raises(ValueError, match="至少一个"):
        PrecisionCheckpointProvenance(
            role=PrecisionCheckpointRole.FORMAL_TRAINING,
            data_identity_sha256="1" * 64,
            training_config_sha256="2" * 64,
            source_tree_sha256="3" * 64,
            seed=13013,
            examples_seen=0,
            optimizer_steps=0,
        )
