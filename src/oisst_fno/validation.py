"""Structural validation for downloaded NOAA OISST subsets.

The checks here are deliberately mechanical: coordinates, spacing, continuity, value
ranges, and land-mask stability. They exist so that a broken or silently truncated
download fails loudly instead of flowing into the supervised dataset.

Interpretation of *what the data mean* stays in notebooks ``01`` and ``02``. This module
only decides whether a file is structurally what OISST v2.1 claims to be.

Constants below were verified against the live NCEI ERDDAP metadata for
``ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon``: 0.25-degree resolution, ``sst`` in
Celsius, and stored ``valid_min``/``valid_max`` of -300/4500 in hundredths of a degree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

GRID_SPACING_DEG = 0.25
SST_VALID_MIN_C = -3.0
SST_VALID_MAX_C = 45.0

# Coordinate spacing is float arithmetic on a published grid; allow a small tolerance.
_SPACING_TOLERANCE_DEG = 1e-6


class DataValidationError(ValueError):
    """Raised when an OISST subset fails structural validation."""


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A single failed check."""

    check: str
    detail: str

    def __str__(self) -> str:
        return f"{self.check}: {self.detail}"


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Outcome of validating a dataset.

    A report is returned rather than raised immediately so that a notebook can display
    every problem at once instead of fixing them one traceback at a time.
    """

    issues: tuple[ValidationIssue, ...] = ()
    summary: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when no check failed."""
        return not self.issues

    def raise_for_status(self) -> None:
        """Raise :class:`DataValidationError` listing every issue, if any."""
        if self.issues:
            listed = "\n".join(f"  - {issue}" for issue in self.issues)
            raise DataValidationError(f"OISST subset failed {len(self.issues)} check(s):\n{listed}")


def _check_monotonic(values: np.ndarray, name: str, issues: list[ValidationIssue]) -> None:
    if values.size < 2:
        return
    diffs = np.diff(values)
    if not np.all(diffs > 0):
        bad = int(np.flatnonzero(diffs <= 0)[0])
        issues.append(
            ValidationIssue(
                f"{name}_monotonic",
                f"not strictly increasing; first violation at index {bad}",
            )
        )


def _check_spacing(values: np.ndarray, name: str, issues: list[ValidationIssue]) -> None:
    if values.size < 2:
        return
    diffs = np.diff(values.astype(float))
    off_grid = np.abs(diffs - GRID_SPACING_DEG) > _SPACING_TOLERANCE_DEG
    if np.any(off_grid):
        first = int(np.flatnonzero(off_grid)[0])
        issues.append(
            ValidationIssue(
                f"{name}_spacing",
                f"expected {GRID_SPACING_DEG} degree steps; index {first} steps by "
                f"{diffs[first]:.6f}",
            )
        )


def _check_time_axis(ds: xr.Dataset, issues: list[ValidationIssue]) -> None:
    times = np.asarray(ds["time"].values)
    if times.size < 2:
        issues.append(ValidationIssue("time_length", "at least two time steps required"))
        return

    unique, counts = np.unique(times, return_counts=True)
    if unique.size != times.size:
        duplicated = unique[counts > 1][:5]
        issues.append(
            ValidationIssue(
                "time_duplicates",
                f"{times.size - unique.size} duplicated timestamp(s), e.g. "
                f"{[str(value) for value in duplicated]}",
            )
        )

    if not np.all(np.diff(times) > np.timedelta64(0, "s")):
        issues.append(ValidationIssue("time_monotonic", "timestamps are not increasing"))
        return

    steps = np.diff(times).astype("timedelta64[s]").astype(np.int64)
    one_day = 24 * 60 * 60
    gaps = np.flatnonzero(steps != one_day)
    if gaps.size:
        first = int(gaps[0])
        issues.append(
            ValidationIssue(
                "time_continuity",
                f"{gaps.size} non-daily step(s); first between {times[first]} and "
                f"{times[first + 1]}",
            )
        )


def _check_values(sst: xr.DataArray, issues: list[ValidationIssue]) -> None:
    finite = np.isfinite(np.asarray(sst.values))
    if not finite.any():
        issues.append(ValidationIssue("sst_all_missing", "no finite SST values present"))
        return

    values = np.asarray(sst.values)[finite]
    low = float(values.min())
    high = float(values.max())
    if low < SST_VALID_MIN_C or high > SST_VALID_MAX_C:
        issues.append(
            ValidationIssue(
                "sst_range",
                f"values span [{low:.2f}, {high:.2f}] degrees Celsius, outside the "
                f"published valid range [{SST_VALID_MIN_C}, {SST_VALID_MAX_C}]",
            )
        )


def _check_land_mask_stability(sst: xr.DataArray, issues: list[ValidationIssue]) -> None:
    """Land cells must be missing on every day, not on some days only.

    A mask that moves between time steps usually means an incomplete download or a
    regridding problem, and it would silently change which cells the masked metrics
    average over.
    """
    missing = ~np.isfinite(np.asarray(sst.values))
    if missing.ndim != 3:
        issues.append(ValidationIssue("sst_dims", f"expected 3 dimensions, got {missing.ndim}"))
        return

    always_missing = missing.all(axis=0)
    ever_missing = missing.any(axis=0)
    unstable = int(np.count_nonzero(ever_missing & ~always_missing))
    if unstable:
        issues.append(
            ValidationIssue(
                "land_mask_stability",
                f"{unstable} cell(s) are missing on some days but not all; the land "
                "mask is not constant in time",
            )
        )


def validate_time_axis(ds: xr.Dataset) -> ValidationReport:
    """Validate only the time coordinate: present, unique, increasing, and daily."""
    issues: list[ValidationIssue] = []
    if "time" not in ds.coords:
        issues.append(ValidationIssue("coords_present", "missing coordinate: time"))
        return ValidationReport(issues=tuple(issues))
    _check_time_axis(ds, issues)
    return ValidationReport(issues=tuple(issues))


def validate_oisst_dataset(
    ds: xr.Dataset,
    *,
    variable: str = "sst",
    require_daily: bool = True,
) -> ValidationReport:
    """Validate an opened OISST subset and return every problem found.

    Checks dimensions, coordinate monotonicity and 0.25-degree spacing, daily
    continuity, duplicated timestamps, physical value range, and land-mask stability.

    Set ``require_daily`` to False for deliberately subsampled datasets, where gaps are
    expected rather than a symptom of a broken download.
    """
    issues: list[ValidationIssue] = []

    missing_coords = [name for name in ("time", "lat", "lon") if name not in ds.coords]
    if missing_coords:
        issues.append(ValidationIssue("coords_present", f"missing coordinate(s): {missing_coords}"))
        return ValidationReport(issues=tuple(issues))

    if variable not in ds.data_vars:
        issues.append(
            ValidationIssue(
                "variable_present",
                f"{variable!r} not among data variables "
                f"{sorted(str(name) for name in ds.data_vars)}",
            )
        )
        return ValidationReport(issues=tuple(issues))

    sst = ds[variable]
    expected_dims = ("time", "lat", "lon")
    if tuple(sst.dims) != expected_dims:
        issues.append(
            ValidationIssue("dim_order", f"expected dims {expected_dims}, got {tuple(sst.dims)}")
        )
        return ValidationReport(issues=tuple(issues))

    lat = np.asarray(ds["lat"].values, dtype=float)
    lon = np.asarray(ds["lon"].values, dtype=float)

    _check_monotonic(lat, "lat", issues)
    _check_monotonic(lon, "lon", issues)
    _check_spacing(lat, "lat", issues)
    _check_spacing(lon, "lon", issues)

    if require_daily:
        _check_time_axis(ds, issues)

    _check_values(sst, issues)
    _check_land_mask_stability(sst, issues)

    times = np.asarray(ds["time"].values)
    summary = {
        "time_steps": str(times.size),
        "time_start": str(times[0]) if times.size else "n/a",
        "time_end": str(times[-1]) if times.size else "n/a",
        "grid": f"{lat.size} x {lon.size}",
        "lat_range": f"{lat.min():.3f} to {lat.max():.3f}" if lat.size else "n/a",
        "lon_range": f"{lon.min():.3f} to {lon.max():.3f}" if lon.size else "n/a",
        "ocean_cells": str(int(np.count_nonzero(np.isfinite(np.asarray(sst.values)).any(axis=0)))),
    }
    return ValidationReport(issues=tuple(issues), summary=summary)
