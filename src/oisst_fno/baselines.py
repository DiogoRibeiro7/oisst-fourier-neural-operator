"""Strong, reusable baselines for SST field forecasting.

The project remains notebook-first. This module contains only baselines that are
used in more than one notebook and therefore should not be duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from oisst_fno.data import ForecastSpec, forecast_target_indices

FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]


def _climatology_day(times: NDArray[np.datetime64]) -> NDArray[np.int_]:
    """Map dates to a fixed 366-day calendar using month/day only."""
    index = pd.DatetimeIndex(times)
    reference = pd.to_datetime("2000-" + index.strftime("%m-%d"))
    return np.asarray(reference.dayofyear, dtype=int)


def seasonal_climatology(
    train_values: FloatArray,
    train_times: NDArray[np.datetime64],
    target_times: NDArray[np.datetime64],
    *,
    half_window_days: int = 15,
) -> FloatArray:
    """Predict fields from a train-only circular day-of-year climatology.

    For each target date, average training fields whose calendar day lies within
    ``half_window_days`` on a fixed 366-day circular calendar. This provides a
    substantially stronger seasonal reference than a single annual mean field
    without using validation or test observations.
    """
    values = np.asarray(train_values, dtype=float)
    if values.ndim != 3:
        raise ValueError("train_values must have shape [time, height, width].")
    train_dates = np.asarray(train_times)
    target_dates = np.asarray(target_times)
    if train_dates.shape[0] != values.shape[0]:
        raise ValueError("train_times length must match train_values time dimension.")
    if half_window_days < 0 or half_window_days > 182:
        raise ValueError("half_window_days must be between 0 and 182.")

    train_day = _climatology_day(train_dates)
    target_day = _climatology_day(target_dates)
    forecasts: list[FloatArray] = []

    for day in target_day:
        direct = np.abs(train_day - day)
        circular = np.minimum(direct, 366 - direct)
        selected = circular <= half_window_days
        if not np.any(selected):
            raise ValueError(f"No training dates available near climatological day {day}.")
        forecasts.append(np.nanmean(values[selected], axis=0))

    return np.asarray(forecasts, dtype=float)


@dataclass(slots=True)
class EOFRidgeForecaster:
    """Low-rank linear spatiotemporal baseline using EOF/PCA scores plus ridge.

    PCA/EOFs compress each daily SST field on valid ocean cells. Ridge regression
    then maps the concatenated EOF scores from the lookback window to the target
    day's EOF scores. This tests whether a simple global linear dynamical model
    explains any apparent advantage of the FNO.
    """

    n_components: int = 32
    alpha: float = 1.0
    _pca: PCA | None = None
    _ridge: Ridge | None = None
    _mask: BoolArray | None = None
    _shape: tuple[int, int] | None = None
    _spec: ForecastSpec | None = None

    def __post_init__(self) -> None:
        if self.n_components < 1:
            raise ValueError("n_components must be positive.")
        if self.alpha < 0 or not np.isfinite(self.alpha):
            raise ValueError("alpha must be finite and non-negative.")

    @staticmethod
    def _ocean_matrix(values: FloatArray, mask: BoolArray) -> FloatArray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 3:
            raise ValueError("values must have shape [time, height, width].")
        if mask.shape != array.shape[1:]:
            raise ValueError("mask must match the spatial dimensions of values.")
        matrix = array[:, mask]
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Valid ocean cells must be finite for the EOF baseline.")
        return matrix

    def fit(self, values: FloatArray, spec: ForecastSpec) -> "EOFRidgeForecaster":
        """Fit EOFs and ridge dynamics using one chronological training series."""
        array = np.asarray(values, dtype=float)
        if array.ndim != 3:
            raise ValueError("values must have shape [time, height, width].")
        mask = np.all(np.isfinite(array), axis=0)
        matrix = self._ocean_matrix(array, mask)
        max_components = min(matrix.shape[0], matrix.shape[1])
        if self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} exceeds available rank {max_components}."
            )

        pca = PCA(n_components=self.n_components, svd_solver="randomized", random_state=0)
        scores = pca.fit_transform(matrix)
        target_indices = forecast_target_indices(len(scores), spec)
        starts = np.arange(len(target_indices), dtype=int)
        histories = np.stack(
            [scores[start : start + spec.lookback_days].reshape(-1) for start in starts]
        )
        targets = scores[target_indices]

        ridge = Ridge(alpha=self.alpha)
        ridge.fit(histories, targets)

        self._pca = pca
        self._ridge = ridge
        self._mask = mask
        self._shape = array.shape[1:]
        self._spec = spec
        return self

    def predict_series(self, values: FloatArray) -> FloatArray:
        """Forecast every valid target window in a contiguous input series."""
        if self._pca is None or self._ridge is None or self._mask is None:
            raise RuntimeError("The EOF-ridge forecaster must be fitted before prediction.")
        if self._shape is None or self._spec is None:
            raise RuntimeError("Missing fitted model metadata.")

        array = np.asarray(values, dtype=float)
        if array.shape[1:] != self._shape:
            raise ValueError("Prediction fields must match the fitted spatial grid.")
        matrix = self._ocean_matrix(array, self._mask)
        scores = self._pca.transform(matrix)
        target_indices = forecast_target_indices(len(scores), self._spec)
        starts = np.arange(len(target_indices), dtype=int)
        histories = np.stack(
            [scores[start : start + self._spec.lookback_days].reshape(-1) for start in starts]
        )
        predicted_scores = self._ridge.predict(histories)
        ocean_values = self._pca.inverse_transform(predicted_scores)

        output = np.full((len(target_indices), *self._shape), np.nan, dtype=float)
        output[:, self._mask] = ocean_values
        return output
