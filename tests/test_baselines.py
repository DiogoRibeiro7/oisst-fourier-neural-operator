from __future__ import annotations

import numpy as np

from oisst_fno.baselines import EOFRidgeForecaster, seasonal_climatology
from oisst_fno.data import ForecastSpec


def test_seasonal_climatology_uses_matching_calendar_window() -> None:
    times = np.array(
        ["2020-01-01", "2020-07-01", "2021-01-01", "2021-07-01"],
        dtype="datetime64[D]",
    )
    values = np.array([1.0, 10.0, 3.0, 14.0], dtype=float)[:, None, None]
    target = np.array(["2022-01-01", "2022-07-01"], dtype="datetime64[D]")

    result = seasonal_climatology(values, times, target, half_window_days=0)

    assert np.allclose(result[:, 0, 0], [2.0, 12.0])


def test_eof_ridge_forecaster_returns_expected_shape() -> None:
    rng = np.random.default_rng(4)
    time = np.arange(40, dtype=float)
    spatial_pattern = np.array([[1.0, 0.5], [-0.5, -1.0]])
    values = time[:, None, None] * 0.05 * spatial_pattern[None, :, :]
    values += 0.001 * rng.standard_normal(values.shape)
    spec = ForecastSpec(lookback_days=4, horizon_days=2)

    model = EOFRidgeForecaster(n_components=2, alpha=1e-4).fit(values[:30], spec)
    prediction = model.predict_series(values[30:])

    assert prediction.shape == (5, 2, 2)
    assert np.all(np.isfinite(prediction))
