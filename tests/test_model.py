from __future__ import annotations

import torch

from oisst_fno.model import FNO2d, TruncatedFourierMix2d


def test_fourier_mixing_shape_and_gradient() -> None:
    layer = TruncatedFourierMix2d(3, 5, modes_y=4, modes_x=4)
    x = torch.randn(2, 3, 16, 20, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 5, 16, 20)
    y.mean().backward()
    assert x.grad is not None
    assert layer.kernel_components.grad is not None


def test_fourier_mixing_rejects_overlapping_vertical_modes() -> None:
    layer = TruncatedFourierMix2d(2, 2, modes_y=9, modes_x=2)
    x = torch.randn(1, 2, 16, 20)
    try:
        layer(x)
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("Expected overlapping vertical Fourier bands to be rejected.")


def test_fno_output_shape() -> None:
    model = FNO2d(in_channels=15, width=16, modes_y=4, modes_x=4, depth=2, padding=2)
    x = torch.randn(2, 15, 24, 32)
    y = model(x)
    assert y.shape == (2, 1, 24, 32)
