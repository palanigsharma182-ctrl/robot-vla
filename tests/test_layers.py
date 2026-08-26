import pytest

torch = pytest.importorskip("torch")

from robot_vla.model.layers import FP32RMSNorm


def test_fp32_rms_norm_preserves_bfloat16_boundary_and_weight_gradient() -> None:
    layer = FP32RMSNorm(8)
    value = torch.randn(2, 3, 8, dtype=torch.bfloat16, requires_grad=True)

    output = layer(value)
    output.float().square().mean().backward()

    assert output.dtype == torch.bfloat16
    assert value.grad is not None and value.grad.dtype == torch.bfloat16
    assert layer.weight.grad is not None and layer.weight.grad.dtype == torch.float32


def test_fp32_rms_norm_matches_torch_reference_for_float32() -> None:
    layer = FP32RMSNorm(8)
    value = torch.randn(2, 3, 8)

    expected = torch.nn.functional.rms_norm(
        value,
        layer.normalized_shape,
        layer.weight,
        layer.eps,
    )

    torch.testing.assert_close(layer(value), expected)
