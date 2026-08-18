from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from oisst_fno.data import (
    ForecastSpec,
    Region,
    SSTWindowDataset,
    Standardizer,
    build_erddap_url,
    forecast_target_indices,
    open_oisst,
)
from oisst_fno.provenance import DownloadManifest, ProvenanceError, manifest_path_for


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


def _write_erddap_like(path: Path, lon_start: float, n_lon: int = 8) -> np.ndarray:
    """Write a NetCDF file shaped like an ERDDAP OISST subset.

    Mirrors the real response: a singleton ``depth`` dimension and longitudes on the
    NOAA [0, 360) convention.
    """
    times = np.arange(
        np.datetime64("2024-01-01T12:00:00"),
        np.datetime64("2024-01-04T12:00:00"),
        np.timedelta64(1, "D"),
    )
    lat = 30.125 + 0.25 * np.arange(5)
    lon = (lon_start + 0.25 * np.arange(n_lon)) % 360.0
    values = np.arange(len(times) * 1 * len(lat) * n_lon, dtype=np.float32).reshape(
        len(times), 1, len(lat), n_lon
    )
    ds = xr.Dataset(
        {"sst": (("time", "depth", "latitude", "longitude"), values)},
        coords={"time": times, "depth": [0.0], "latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(path, engine="scipy")
    return values[:, 0]


def test_open_oisst_normalizes_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "atlantic.nc"
    expected = _write_erddap_like(path, lon_start=330.125)

    ds = open_oisst(path)

    assert "depth" not in ds.dims
    assert ds["sst"].dims == ("time", "lat", "lon")
    lon = ds["lon"].values
    assert np.all(np.diff(lon) > 0), "longitude must stay ascending"
    assert lon.min() == pytest.approx(-29.875)
    # Values must still line up with their coordinates after conversion.
    assert np.allclose(ds["sst"].values, expected)


def test_open_oisst_reorders_a_region_crossing_the_antimeridian(tmp_path: Path) -> None:
    """Regression test: reordering used to index the wrong axis after the depth squeeze.

    Longitudes spanning 180 degrees convert to a non-monotonic axis, which is the only
    case that needs reordering. On xarray's scipy backend the lazy reorder previously
    raised IndexError; the values below must survive the sort intact.
    """
    path = tmp_path / "pacific.nc"
    expected = _write_erddap_like(path, lon_start=179.125, n_lon=8)

    ds = open_oisst(path)

    lon = ds["lon"].values
    assert np.all(np.diff(lon) > 0), "longitude must be sorted ascending"
    # Sorting the axis must permute the data the same way, not just the coordinate.
    order = np.argsort(((np.arange(8) * 0.25 + 179.125 + 180.0) % 360.0) - 180.0)
    assert np.allclose(ds["sst"].values, expected[:, :, order])


def test_open_oisst_verifies_a_sidecar_manifest(tmp_path: Path) -> None:
    path = tmp_path / "atlantic.nc"
    _write_erddap_like(path, lon_start=330.125)
    DownloadManifest.build(
        data_path=path,
        source_url="https://example.invalid/subset.nc",
        dataset_id="ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon",
        dataset_doi="10.25921/RE9P-PT57",
        product_version="Version v02r01",
        start_date="2024-01-01",
        end_date="2024-01-03",
        variables=("sst",),
        lat_min=30.125,
        lat_max=31.125,
        lon_min=330.125,
        lon_max=332.125,
    ).write(manifest_path_for(path))

    open_oisst(path)  # manifest matches, so this succeeds

    with path.open("ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(ProvenanceError):
        open_oisst(path)

    # The check can be skipped deliberately.
    open_oisst(path, verify_manifest=False)
