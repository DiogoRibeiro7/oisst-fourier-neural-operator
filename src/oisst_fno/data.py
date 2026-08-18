"""Data access and temporal-window utilities for NOAA OISST.

The module stays deliberately small. Dataset exploration, plotting, model fitting,
and scientific interpretation belong in the notebooks.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Final
from urllib.parse import quote

import numpy as np
import requests
import torch
import xarray as xr
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from .provenance import DownloadManifest, manifest_path_for
from .validation import validate_time_axis

FloatArray = NDArray[np.float32]

OISST_DATASET_ID: Final[str] = "ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon"
OISST_DOI: Final[str] = "10.25921/RE9P-PT57"

# Verified against the live NCEI ERDDAP metadata for this dataset.
OISST_PRODUCT_VERSION: Final[str] = "Version v02r01"

ERDDAP_BASE: Final[str] = f"https://www.ncei.noaa.gov/erddap/griddap/{OISST_DATASET_ID}.nc"

# NetCDF-3 classic files start with "CDF", NetCDF-4/HDF5 with the HDF signature.
# ERDDAP reports failures as HTML or plain text, which these prefixes reject.
_NETCDF_MAGIC: Final[tuple[bytes, ...]] = (b"CDF", b"\x89HDF")

_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 425, 429, 500, 502, 503, 504})

SMOKE_TEST_DAYS: Final[int] = 3


class DownloadError(RuntimeError):
    """Raised when an OISST download cannot be completed or is not trustworthy."""


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
    def fit(cls, values: np.ndarray) -> Standardizer:
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


def smoke_test_end_date(start_date: str, days: int = SMOKE_TEST_DAYS) -> str:
    """Return the end date of a short smoke-test window starting at ``start_date``.

    Used to exercise the whole acquisition path — URL, network, write, validation —
    against a few megabytes rather than several years.
    """
    if days < 1:
        raise ValueError("days must be positive.")
    start = date.fromisoformat(start_date)
    return (start + timedelta(days=days - 1)).isoformat()


def _looks_like_netcdf(path: Path) -> bool:
    """Return True when the file begins with a NetCDF or HDF signature."""
    with path.open("rb") as handle:
        head = handle.read(8)
    return any(head.startswith(magic) for magic in _NETCDF_MAGIC)


def _stream_to_file(response: requests.Response, target: Path) -> int:
    """Write a streamed response body to ``target`` and return the bytes written."""
    written = 0
    with target.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
                written += len(chunk)
    return written


def _fetch_once(url: str, partial: Path, timeout_seconds: int) -> None:
    """Perform one download attempt into ``partial``.

    Raises ``requests`` exceptions for transport problems so the caller can decide
    whether the failure is worth retrying, and :class:`DownloadError` for responses
    that arrived intact but are not usable data.
    """
    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        declared = response.headers.get("Content-Length")
        written = _stream_to_file(response, partial)

    if written == 0:
        raise DownloadError(f"The server returned an empty body for {url}")

    if declared is not None:
        expected = int(declared)
        if written != expected:
            raise DownloadError(
                f"Truncated download: received {written} bytes but the server declared "
                f"{expected}. The partial file was discarded."
            )

    if not _looks_like_netcdf(partial):
        preview = partial.read_bytes()[:200].decode("utf-8", errors="replace").strip()
        raise DownloadError(
            "The response is not a NetCDF file. ERDDAP usually returns an error "
            f"message here. First bytes:\n{preview}"
        )


def download_subset(
    destination: Path,
    start_date: str,
    end_date: str,
    region: Region,
    variables: Sequence[str] = ("sst",),
    *,
    overwrite: bool = False,
    timeout_seconds: int = 300,
    max_attempts: int = 4,
    backoff_seconds: float = 2.0,
    smoke_test: bool = False,
    write_manifest: bool = True,
) -> Path:
    """Download a regional OISST NetCDF subset, with provenance and bounded retries.

    The download is written to a ``.part`` file and only moved into place once it is
    complete, the declared length matches, and the payload is really NetCDF. A partial
    or error response therefore never leaves a file that later looks like valid data.

    Transient failures (connection resets, timeouts, and 5xx/429 responses) are retried
    up to ``max_attempts`` times with exponential backoff. Client errors such as a bad
    date range fail immediately, because retrying cannot fix them.

    Set ``smoke_test`` to fetch only the first few days of the requested period.

    Unless ``write_manifest`` is False, a sidecar ``<name>.manifest.json`` records the
    source URL, timestamp, request parameters, file size, SHA-256 hash, and dataset DOI.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")

    if smoke_test:
        end_date = smoke_test_end_date(start_date)

    destination = Path(destination)
    if destination.exists() and not overwrite:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = build_erddap_url(start_date, end_date, region, variables)
    partial = destination.with_suffix(destination.suffix + ".part")

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            _fetch_once(url, partial, timeout_seconds)
            break
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else None
            if status not in _RETRYABLE_STATUS:
                partial.unlink(missing_ok=True)
                raise DownloadError(
                    f"ERDDAP rejected the request with HTTP {status}. This will not "
                    f"succeed on retry; check the dates, bounds, and variables.\n{url}"
                ) from error
            last_error = error
        except (requests.ConnectionError, requests.Timeout, DownloadError) as error:
            last_error = error

        partial.unlink(missing_ok=True)
        if attempt == max_attempts:
            raise DownloadError(
                f"Download failed after {max_attempts} attempt(s): {last_error}\n{url}"
            ) from last_error
        time.sleep(backoff_seconds * 2 ** (attempt - 1))

    partial.replace(destination)

    if write_manifest:
        DownloadManifest.build(
            data_path=destination,
            source_url=url,
            dataset_id=OISST_DATASET_ID,
            dataset_doi=OISST_DOI,
            product_version=OISST_PRODUCT_VERSION,
            start_date=start_date,
            end_date=end_date,
            variables=tuple(variables),
            lat_min=region.lat_min,
            lat_max=region.lat_max,
            lon_min=region.lon_min,
            lon_max=region.lon_max,
            smoke_test=smoke_test,
        ).write(manifest_path_for(destination))

    return destination


def open_oisst(path: Path, *, verify_manifest: bool = True) -> xr.Dataset:
    """Open an OISST subset and normalize coordinate names/conventions.

    When a sidecar manifest exists it is checked first, so a truncated or modified file
    raises :class:`~oisst_fno.provenance.ProvenanceError` instead of being opened as if
    it were the data originally downloaded.
    """
    path = Path(path)
    if verify_manifest:
        manifest_path = manifest_path_for(path)
        if manifest_path.exists():
            DownloadManifest.read(manifest_path).verify(path)

    ds = xr.open_dataset(path)
    if "depth" in ds.dims:
        ds = ds.squeeze("depth", drop=True)
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})

    # Convert NOAA [0, 360) longitude to conventional [-180, 180) for plots.
    converted_lon = ((np.asarray(ds["lon"].values, dtype=float) + 180.0) % 360.0) - 180.0
    ds = ds.assign_coords(lon=converted_lon)

    if converted_lon.size > 1 and not np.all(np.diff(converted_lon) > 0):
        # The region straddles the 0/360 seam, so the converted axis is out of order.
        # Reordering a lazily loaded array after the squeeze above indexes the wrong
        # axis on xarray's scipy backend, so materialise the values before sorting.
        ds = ds.load().sortby("lon")
    return ds


def validate_daily_time_axis(ds: xr.Dataset) -> None:
    """Raise if the dataset does not have a contiguous daily time index.

    Kept as a focused helper for notebooks that only care about the time axis. Use
    :func:`oisst_fno.validation.validate_oisst_dataset` for the full structural check.
    """
    validate_time_axis(ds).raise_for_status()


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
    targets: np.ndarray = array[forecast_target_indices(len(array), spec)]
    return targets


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


class SSTSequenceDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Temporal windows whose target is a *sequence* of future fields.

    The single-field :class:`SSTWindowDataset` answers "what is the field at one lead
    time". This answers "what is the trajectory", which is what a spatiotemporal operator
    predicts and what makes temporal coherence measurable at all.

    Inputs keep the ``[time, height, width]`` layout so a 3-D model can consume time as a
    dimension; a channel-stacked 2-D model reads the same tensor with time as channels, so
    both architectures train on identical windows and identical targets.
    """

    def __init__(
        self,
        values: np.ndarray,
        lookback_days: int,
        horizon_days: int,
    ) -> None:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError("values must have shape [time, height, width].")
        if lookback_days < 1 or horizon_days < 1:
            raise ValueError("lookback_days and horizon_days must be positive.")
        minimum = lookback_days + horizon_days
        if array.shape[0] < minimum:
            raise ValueError(f"Need at least {minimum} time steps, got {array.shape[0]}.")

        self._lookback = lookback_days
        self._horizon = horizon_days
        self._mask: FloatArray = np.asarray(np.isfinite(array).any(axis=0), dtype=np.float32)
        self._filled: FloatArray = np.asarray(
            np.nan_to_num(array, nan=0.0, copy=True), dtype=np.float32
        )
        self._steps = int(array.shape[0])

    def __len__(self) -> int:
        return self._steps - self._lookback - self._horizon + 1

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index
        stop = start + self._lookback
        target_stop = stop + self._horizon

        x = torch.from_numpy(self._filled[start:stop].copy())
        y = torch.from_numpy(self._filled[stop:target_stop].copy())
        mask = torch.from_numpy(self._mask[None, ...].copy())
        return x, y, mask

    def target_offsets(self) -> np.ndarray:
        """Lead times, in days, of each element of the target sequence."""
        return np.arange(1, self._horizon + 1, dtype=int)


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

        self._values: FloatArray = array
        self._spec = spec
        self._mask: FloatArray = np.asarray(np.isfinite(array).any(axis=0), dtype=np.float32)
        self._filled: FloatArray = np.asarray(
            np.nan_to_num(array, nan=0.0, copy=True), dtype=np.float32
        )

    def __len__(self) -> int:
        available = int(self._values.shape[0])
        return available - self._spec.lookback_days - self._spec.horizon_days + 1

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
