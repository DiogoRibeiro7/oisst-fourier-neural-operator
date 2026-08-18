from __future__ import annotations

import numpy as np

from oisst_fno.data import (
    ForecastSpec,
    Region,
    SSTWindowDataset,
    Standardizer,
    build_erddap_url,
    forecast_target_indices,
)


def test_build_erddap_url_contains_requested_subset() -> None:
    url = build_erddap_url("2024-01-01", "2024-01-31", Region(), ("sst", "err"))
    assert "ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon.nc" in url
    assert "sst" in url
    assert "err" in url
    assert "330.125" in url


def test_standardizer_roundtrip() -> None:
    values = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32)
    scaler = Standardizer.fit(values)
    restored = scaler.inverse_transform(scaler.transform(values))
    assert np.allclose(restored[np.isfinite(values)], values[np.isfinite(values)])


def test_window_dataset_shapes_and_target_offset() -> None:
    values = np.arange(10 * 3 * 4, dtype=np.float32).reshape(10, 3, 4)
    dataset = SSTWindowDataset(values, ForecastSpec(lookback_days=3, horizon_days=2))
    x, y, mask = dataset[0]
    assert x.shape == (3, 3, 4)
    assert y.shape == (1, 3, 4)
    assert mask.shape == (1, 3, 4)
    assert np.allclose(y.numpy()[0], values[4])


def test_forecast_target_indices() -> None:
    spec = ForecastSpec(lookback_days=3, horizon_days=2)
    assert np.array_equal(forecast_target_indices(8, spec), np.array([4, 5, 6, 7]))
