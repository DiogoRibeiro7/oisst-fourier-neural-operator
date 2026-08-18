from __future__ import annotations

import numpy as np
import torch

from oisst_fno.metrics import (
    daily_rmse,
    mae,
    masked_mse_loss,
    moving_block_bootstrap_mean_ci,
    rmse,
    skill_score,
    spectral_error_energy_by_band,
)


def test_masked_metrics_ignore_land() -> None:
    pred = np.array([[1.0, 100.0], [3.0, 4.0]])
    obs = np.array([[1.0, 0.0], [1.0, 4.0]])
    mask = np.array([[1, 0], [1, 1]], dtype=bool)
    assert np.isclose(mae(pred, obs, mask), 2.0 / 3.0)
    assert np.isclose(rmse(pred, obs, mask), np.sqrt(4.0 / 3.0))


def test_masked_torch_loss() -> None:
    pred = torch.tensor([[[[1.0, 9.0]]]])
    obs = torch.tensor([[[[3.0, 0.0]]]])
    mask = torch.tensor([[[[1.0, 0.0]]]])
    assert torch.isclose(masked_mse_loss(pred, obs, mask), torch.tensor(4.0))


def test_skill_score() -> None:
    assert np.isclose(skill_score(0.8, 1.0), 0.2)


def test_daily_rmse_returns_one_value_per_case() -> None:
    pred = np.zeros((2, 2, 2), dtype=float)
    obs = np.array([np.ones((2, 2)), np.full((2, 2), 2.0)])
    result = daily_rmse(pred, obs)
    assert np.allclose(result, [1.0, 2.0])


def test_block_bootstrap_constant_series_has_degenerate_interval() -> None:
    values = np.full(30, 2.5)
    mean, lower, upper = moving_block_bootstrap_mean_ci(
        values, block_length=5, n_bootstrap=200, seed=2
    )
    assert np.isclose(mean, 2.5)
    assert np.isclose(lower, 2.5)
    assert np.isclose(upper, 2.5)


def test_spectral_error_energy_detects_high_frequency_error() -> None:
    height = width = 16
    checkerboard = np.indices((height, width)).sum(axis=0) % 2
    target = np.zeros((1, height, width), dtype=float)
    prediction = checkerboard.astype(float)[None, :, :]
    mask = np.ones((height, width), dtype=bool)

    energies = spectral_error_energy_by_band(
        prediction,
        target,
        mask,
        band_edges=(0.0, 0.25, 0.75, 1.01),
    )

    assert energies[-1] > energies[0]
