"""Masked and scale-aware forecast metrics for ocean fields."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

FloatArray = NDArray[np.floating]


def masked_mse_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean squared error over valid ocean cells only."""
    weights = mask.to(dtype=prediction.dtype).expand_as(prediction)
    denominator = weights.sum().clamp_min(1.0)
    return (((prediction - target) ** 2) * weights).sum() / denominator


def _valid_values(
    prediction: FloatArray,
    target: FloatArray,
    mask: FloatArray | None,
) -> tuple[FloatArray, FloatArray]:
    pred = np.asarray(prediction, dtype=float)
    obs = np.asarray(target, dtype=float)
    if pred.shape != obs.shape:
        raise ValueError("prediction and target must have identical shapes.")
    valid = np.isfinite(pred) & np.isfinite(obs)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not np.any(valid):
        raise ValueError("No valid cells are available for metric computation.")
    return pred[valid], obs[valid]


def rmse(prediction: FloatArray, target: FloatArray, mask: FloatArray | None = None) -> float:
    """Root mean squared error on valid cells."""
    pred, obs = _valid_values(prediction, target, mask)
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def mae(prediction: FloatArray, target: FloatArray, mask: FloatArray | None = None) -> float:
    """Mean absolute error on valid cells."""
    pred, obs = _valid_values(prediction, target, mask)
    return float(np.mean(np.abs(pred - obs)))


def daily_rmse(
    prediction: FloatArray,
    target: FloatArray,
    mask: FloatArray | None = None,
) -> FloatArray:
    """RMSE for each forecast case in arrays shaped [sample, height, width]."""
    pred = np.asarray(prediction, dtype=float)
    obs = np.asarray(target, dtype=float)
    if pred.shape != obs.shape or pred.ndim != 3:
        raise ValueError("prediction and target must share shape [sample, height, width].")
    valid = np.isfinite(pred) & np.isfinite(obs)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    count = valid.sum(axis=(1, 2))
    if np.any(count == 0):
        raise ValueError("Every sample must contain at least one valid cell.")
    squared = np.where(valid, (pred - obs) ** 2, 0.0)
    rmse: FloatArray = np.sqrt(squared.sum(axis=(1, 2)) / count)
    return rmse


def anomaly_correlation(
    prediction: FloatArray,
    target: FloatArray,
    climatology: FloatArray,
    mask: FloatArray | None = None,
) -> float:
    """Pearson correlation between predicted and observed anomalies."""
    pred_anom = np.asarray(prediction, dtype=float) - np.asarray(climatology, dtype=float)
    obs_anom = np.asarray(target, dtype=float) - np.asarray(climatology, dtype=float)
    pred, obs = _valid_values(pred_anom, obs_anom, mask)
    if np.std(pred) == 0 or np.std(obs) == 0:
        return float("nan")
    return float(np.corrcoef(pred, obs)[0, 1])


def skill_score(model_rmse: float, reference_rmse: float) -> float:
    """RMSE skill relative to a reference; positive is better than reference."""
    if reference_rmse <= 0 or not np.isfinite(reference_rmse):
        raise ValueError("reference_rmse must be finite and positive.")
    return float(1.0 - model_rmse / reference_rmse)


def moving_block_bootstrap_mean_ci(
    values: FloatArray,
    *,
    block_length: int,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Moving-block bootstrap CI for the mean of an autocorrelated series."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("values must be a finite one-dimensional array.")
    if not 1 <= block_length <= len(array):
        raise ValueError("block_length must lie between 1 and len(values).")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1.")

    rng = np.random.default_rng(seed)
    n = len(array)
    possible_starts = np.arange(n - block_length + 1)
    n_blocks = int(np.ceil(n / block_length))
    boot_means = np.empty(n_bootstrap, dtype=float)
    offsets = np.arange(block_length)

    for index in range(n_bootstrap):
        starts = rng.choice(possible_starts, size=n_blocks, replace=True)
        sampled = np.concatenate([array[start + offsets] for start in starts])[:n]
        boot_means[index] = float(np.mean(sampled))

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return float(np.mean(array)), float(lower), float(upper)


def spectral_error_energy_by_band(
    prediction: FloatArray,
    target: FloatArray,
    mask: FloatArray,
    *,
    band_edges: Sequence[float] = (0.0, 0.15, 0.35, 0.65, 1.01),
) -> FloatArray:
    """Mean tapered Fourier error energy in normalized radial-frequency bands.

    The radial frequency is normalized so 0 is the spatial mean and 1 is the
    corner Nyquist radius. A common mask and Hann taper are applied to every
    forecast before the FFT. The result is intended for *relative scale
    attribution between models*, not as an absolute physical spectrum because
    masking and tapering alter spectral amplitudes.
    """
    pred = np.asarray(prediction, dtype=float)
    obs = np.asarray(target, dtype=float)
    if pred.shape != obs.shape or pred.ndim != 3:
        raise ValueError("prediction and target must share shape [sample, height, width].")

    valid = np.asarray(mask, dtype=bool)
    if valid.ndim == 2:
        valid = np.broadcast_to(valid, pred.shape)
    if valid.shape != pred.shape:
        raise ValueError("mask must be spatial or have the same shape as prediction.")

    edges = np.asarray(tuple(band_edges), dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("band_edges must be a strictly increasing one-dimensional sequence.")
    if edges[0] < 0 or edges[-1] <= 1.0:
        raise ValueError("band_edges must start at >=0 and end above 1.0.")

    height, width = pred.shape[-2:]
    taper = np.outer(np.hanning(height), np.hanning(width))
    fy = np.fft.fftfreq(height) / 0.5
    fx = np.fft.rfftfreq(width) / 0.5
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) / np.sqrt(2.0)

    accumulated = np.zeros(len(edges) - 1, dtype=float)
    counts = np.zeros(len(edges) - 1, dtype=int)

    for sample in range(pred.shape[0]):
        sample_valid = valid[sample] & np.isfinite(pred[sample]) & np.isfinite(obs[sample])
        if not np.any(sample_valid):
            raise ValueError("Every sample must contain at least one valid cell.")
        error = np.where(sample_valid, pred[sample] - obs[sample], 0.0)
        error -= float(np.sum(error) / np.sum(sample_valid)) * sample_valid
        transformed = np.fft.rfft2(error * sample_valid * taper, norm="ortho")
        power = np.abs(transformed) ** 2

        for band in range(len(edges) - 1):
            selected = (radius >= edges[band]) & (radius < edges[band + 1])
            if np.any(selected):
                accumulated[band] += float(np.mean(power[selected]))
                counts[band] += 1

    if np.any(counts == 0):
        raise ValueError("At least one spectral band contains no Fourier coefficients.")
    band_energy: FloatArray = accumulated / counts
    return band_energy


def temporal_increment_correlation(
    prediction: FloatArray,
    target: FloatArray,
    mask: FloatArray | None = None,
) -> float:
    """Correlation between predicted and observed day-to-day changes.

    Pointwise error says whether a forecast sequence is close to the truth. This says
    whether it *moves* like the truth. A forecast can have low RMSE while being nearly
    constant in time — a plausible failure mode for a model that predicts a smoothed
    climatology — and that shows up here as a low increment correlation even when RMSE
    looks acceptable.

    Both arrays are ``[lead, height, width]``; at least two lead times are required.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape.")
    if prediction.ndim != 3:
        raise ValueError("expected [lead, height, width] arrays.")
    if prediction.shape[0] < 2:
        raise ValueError("At least two lead times are required to form increments.")

    predicted_change = np.diff(prediction, axis=0)
    observed_change = np.diff(target, axis=0)

    if mask is not None:
        keep = np.broadcast_to(np.asarray(mask, dtype=bool), predicted_change.shape)
    else:
        keep = np.ones_like(predicted_change, dtype=bool)
    keep = keep & np.isfinite(predicted_change) & np.isfinite(observed_change)
    if not keep.any():
        raise ValueError("No valid cells remain after masking.")

    a = predicted_change[keep]
    b = observed_change[keep]
    a_centred = a - a.mean()
    b_centred = b - b.mean()
    denominator = float(np.sqrt((a_centred**2).sum() * (b_centred**2).sum()))
    if denominator == 0.0:
        return 0.0
    return float((a_centred * b_centred).sum() / denominator)


def temporal_variability_ratio(
    prediction: FloatArray,
    target: FloatArray,
    mask: FloatArray | None = None,
) -> float:
    """Ratio of predicted to observed day-to-day variability.

    Values below one mean the forecast evolves more slowly than reality — temporal
    over-smoothing. Above one means it is jumpier. Reported alongside RMSE because
    minimising squared error rewards smoothing, so a good RMSE and a ratio near 0.3 is a
    result that needs saying out loud rather than burying.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape.")
    if prediction.ndim != 3:
        raise ValueError("expected [lead, height, width] arrays.")
    if prediction.shape[0] < 2:
        raise ValueError("At least two lead times are required to form increments.")

    predicted_change = np.diff(prediction, axis=0)
    observed_change = np.diff(target, axis=0)
    if mask is not None:
        keep = np.broadcast_to(np.asarray(mask, dtype=bool), predicted_change.shape)
    else:
        keep = np.ones_like(predicted_change, dtype=bool)
    keep = keep & np.isfinite(predicted_change) & np.isfinite(observed_change)
    if not keep.any():
        raise ValueError("No valid cells remain after masking.")

    observed_std = float(np.std(observed_change[keep]))
    if observed_std == 0.0:
        raise ValueError("Observed increments have zero variability.")
    return float(np.std(predicted_change[keep]) / observed_std)


def parameter_count(model: torch.nn.Module) -> int:
    """Return the number of trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
