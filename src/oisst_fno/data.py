"""Data access and temporal-window utilities for NOAA OISST.

The module stays deliberately small. Dataset exploration, plotting, model fitting,
and scientific interpretation belong in the notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence
from urllib.parse import quote

import numpy as np
import requests
import torch
import xarray as xr
from torch import Tensor
from torch.utils.data import Dataset

ERDDAP_BASE: Final[str] = (
    "https://www.ncei.noaa.gov/erddap/griddap/"
    "ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon.nc"
)


@dataclass(frozen=True, slots=True)
class Region:
    """Spatial bounds on the native OISST 0.25-degree grid.

    NOAA ERDDAP longitudes use the [0, 360) convention. Defaults correspond
    approximately to 30–50°N and 30–5°W.
    """

    lat_min: float = 30.125
    lat_max: float = 50.125
    lon_min: float = 330.125
    lon_max: float = 355.125

    def __post_init__(self) -> None:
        if not -89.875 <= self.lat_min < self.lat_max <= 89.875:
            raise ValueError("Latitude bounds are outside the OISST grid.")
        if not 0.125 <= self.lon_min < self.lon_max <= 359.875:
            raise ValueError("Longitude bounds must use NOAA's [0, 360) convention.")


@dataclass(frozen=True, slots=True)
class ForecastSpec:
    """Temporal supervised-learning specification."""

    lookback_days: int = 14
    horizon_days: int = 7

    def __post_init__(self) -> None:
        if self.lookback_days < 1:
            raise ValueError("lookback_days must be positive.")
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be positive.")


@dataclass(frozen=True, slots=True)
class Standardizer:
    """Global train-only standardization parameters."""

    mean: float
    std: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError("mean must be finite.")
        if not np.isfinite(self.std) or self.std <= 0:
            raise ValueError("std must be finite and positive.")

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        """Fit a global standardizer while ignoring missing land cells."""
        array = np.asarray(values, dtype=np.float32)
        mean = float(np.nanmean(array))
        std = float(np.nanstd(array))
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Standardize values without changing NaN locations."""
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Return standardized values to degrees Celsius."""
        return np.asarray(values, dtype=np.float32) * self.std + self.mean


def build_erddap_url(
    start_date: str,
    end_date: str,
    region: Region,
    variables: Sequence[str] = ("sst",),
) -> str:
    """Build an NCEI ERDDAP NetCDF subset URL.

    Dates are interpreted at OISST's daily 12:00 UTC timestamp. Bounds should be
    native-grid values ending in .125/.375/.625/.875 to avoid ERDDAP ambiguity.
    """
    allowed = {"sst", "anom", "err", "ice"}
    requested = tuple(variables)
    if not requested or any(variable not in allowed for variable in requested):
        raise ValueError(f"variables must be a non-empty subset of {sorted(allowed)}")

    slices = (
        f"[({start_date}T12:00:00Z):1:({end_date}T12:00:00Z)]"
        "[(0.0)]"
        f"[({region.lat_min}):1:({region.lat_max})]"
        f"[({region.lon_min}):1:({region.lon_max})]"
    )
    query = ",".join(f"{variable}{slices}" for variable in requested)
    return f"{ERDDAP_BASE}?{quote(query, safe='[](),:.-=')}"


def download_subset(
    destination: Path,
    start_date: str,
    end_date: str,
    region: Region,
    variables: Sequence[str] = ("sst",),
    *,
    overwrite: bool = False,
    timeout_seconds: int = 300,
) -> Path:
    """Download a regional OISST NetCDF file with streamed I/O."""
    destination = Path(destination)
    if destination.exists() and not overwrite:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = build_erddap_url(start_date, end_date, region, variables)
    partial = destination.with_suffix(destination.suffix + ".part")

    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(destination)
    return destination


def open_oisst(path: Path) -> xr.Dataset:
    """Open an OISST subset and normalize coordinate names/conventions."""
    ds = xr.open_dataset(Path(path))
    if "depth" in ds.dims:
        ds = ds.squeeze("depth", drop=True)
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})

    # Convert NOAA [0, 360) longitude to conventional [-180, 180) for plots.
    converted_lon = ((ds["lon"] + 180.0) % 360.0) - 180.0
    ds = ds.assign_coords(lon=converted_lon).sortby("lon")
    return ds


def validate_daily_time_axis(ds: xr.Dataset) -> None:
    """Raise if the dataset does not have a contiguous daily time index."""
    if "time" not in ds.coords:
        raise ValueError("Dataset has no time coordinate.")
    time_values = np.asarray(ds["time"].values)
    if time_values.size < 2:
        raise ValueError("At least two daily observations are required.")
    diffs = np.diff(time_values).astype("timedelta64[D]").astype(int)
    if not np.all(diffs == 1):
        bad = np.flatnonzero(diffs != 1)[:5]
        raise ValueError(f"Non-daily gaps detected near positions {bad.tolist()}.")


def forecast_target_indices(n_steps: int, spec: ForecastSpec) -> np.ndarray:
    """Return target indices for all valid lookback/horizon windows."""
    minimum = spec.lookback_days + spec.horizon_days
    if n_steps < minimum:
        raise ValueError(f"Need at least {minimum} time steps, got {n_steps}.")
    starts = np.arange(n_steps - minimum + 1, dtype=int)
    return starts + spec.lookback_days + spec.horizon_days - 1


def forecast_target_times(
    times: np.ndarray,
    spec: ForecastSpec,
) -> np.ndarray:
    """Return timestamps corresponding to supervised forecast targets."""
    array = np.asarray(times)
    if array.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    return array[forecast_target_indices(len(array), spec)]


def temporal_split(
    data: xr.DataArray,
    train_end: str,
    validation_end: str,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Create chronological train/validation/test splits without overlap."""
    train = data.sel(time=slice(None, train_end))
    validation_start = np.datetime64(train_end) + np.timedelta64(1, "D")
    validation = data.sel(time=slice(validation_start, validation_end))
    test = data.sel(time=slice(np.datetime64(validation_end) + np.timedelta64(1, "D"), None))
    if train.sizes.get("time", 0) == 0 or validation.sizes.get("time", 0) == 0:
        raise ValueError("Train and validation splits must be non-empty.")
    return train, validation, test


class SSTWindowDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Lazy temporal windows over a standardized SST array.

    Input array shape is ``[time, height, width]``. NaNs are interpreted as
    land/missing cells, filled with zero after standardization, and accompanied
    by a static ocean mask.
    """

    def __init__(
        self,
        values: np.ndarray,
        spec: ForecastSpec,
    ) -> None:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError("values must have shape [time, height, width].")
        minimum = spec.lookback_days + spec.horizon_days
        if array.shape[0] < minimum:
            raise ValueError(f"Need at least {minimum} time steps, got {array.shape[0]}.")

        self._values = array
        self._spec = spec
        self._mask = np.isfinite(array).any(axis=0).astype(np.float32)
        self._filled = np.nan_to_num(array, nan=0.0, copy=True)

    def __len__(self) -> int:
        return self._values.shape[0] - self._spec.lookback_days - self._spec.horizon_days + 1

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index
        stop = start + self._spec.lookback_days
        target_index = stop + self._spec.horizon_days - 1

        x = torch.from_numpy(self._filled[start:stop].copy())
        y = torch.from_numpy(self._filled[target_index : target_index + 1].copy())
        mask = torch.from_numpy(self._mask[None, ...].copy())
        return x, y, mask
