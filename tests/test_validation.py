from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from oisst_fno.validation import (
    DataValidationError,
    validate_oisst_dataset,
    validate_time_axis,
)


def make_dataset(
    *,
    days: int = 5,
    n_lat: int = 4,
    n_lon: int = 6,
    lat_start: float = 30.125,
    lon_start: float = 330.125,
    spacing: float = 0.25,
    times: np.ndarray | None = None,
) -> xr.Dataset:
    """Build a small, structurally valid OISST-like dataset."""
    if times is None:
        times = np.arange(
            np.datetime64("2024-01-01T12:00:00"),
            np.datetime64("2024-01-01T12:00:00") + np.timedelta64(days, "D"),
            np.timedelta64(1, "D"),
        )
    lat = lat_start + spacing * np.arange(n_lat)
    lon = lon_start + spacing * np.arange(n_lon)

    rng = np.random.default_rng(0)
    values = 15.0 + rng.normal(0.0, 0.5, size=(len(times), n_lat, n_lon)).astype(np.float32)
    # A stable land mask: the same two cells are missing on every day.
    values[:, 0, 0] = np.nan
    values[:, 1, 2] = np.nan

    return xr.Dataset(
        {"sst": (("time", "lat", "lon"), values)},
        coords={"time": times, "lat": lat, "lon": lon},
    )


def test_valid_dataset_reports_no_issues() -> None:
    report = validate_oisst_dataset(make_dataset())

    assert report.ok
    assert report.issues == ()
    report.raise_for_status()


def test_summary_describes_the_subset() -> None:
    report = validate_oisst_dataset(make_dataset(days=5, n_lat=4, n_lon=6))

    assert report.summary["time_steps"] == "5"
    assert report.summary["grid"] == "4 x 6"
    # 24 cells minus the two permanently masked ones.
    assert report.summary["ocean_cells"] == "22"


def test_missing_coordinate_is_reported() -> None:
    ds = make_dataset().drop_vars("lat")

    report = validate_oisst_dataset(ds)

    assert not report.ok
    assert report.issues[0].check == "coords_present"


def test_missing_variable_is_reported() -> None:
    ds = make_dataset().rename({"sst": "anom"})

    report = validate_oisst_dataset(ds)

    assert [issue.check for issue in report.issues] == ["variable_present"]


def test_time_gap_is_reported() -> None:
    times = np.array(
        [
            np.datetime64("2024-01-01T12:00:00"),
            np.datetime64("2024-01-02T12:00:00"),
            np.datetime64("2024-01-04T12:00:00"),  # one day missing
            np.datetime64("2024-01-05T12:00:00"),
        ]
    )
    report = validate_oisst_dataset(make_dataset(times=times))

    checks = [issue.check for issue in report.issues]
    assert "time_continuity" in checks


def test_time_gap_is_allowed_when_daily_continuity_is_not_required() -> None:
    times = np.array(
        [
            np.datetime64("2024-01-01T12:00:00"),
            np.datetime64("2024-01-03T12:00:00"),
            np.datetime64("2024-01-05T12:00:00"),
        ]
    )
    report = validate_oisst_dataset(make_dataset(times=times), require_daily=False)

    assert report.ok


def test_duplicate_timestamps_are_reported() -> None:
    times = np.array(
        [
            np.datetime64("2024-01-01T12:00:00"),
            np.datetime64("2024-01-02T12:00:00"),
            np.datetime64("2024-01-02T12:00:00"),
            np.datetime64("2024-01-03T12:00:00"),
        ]
    )
    report = validate_oisst_dataset(make_dataset(times=times))

    checks = [issue.check for issue in report.issues]
    assert "time_duplicates" in checks


def test_non_monotonic_latitude_is_reported() -> None:
    ds = make_dataset()
    ds = ds.assign_coords(lat=ds["lat"].values[::-1])

    report = validate_oisst_dataset(ds)

    assert "lat_monotonic" in [issue.check for issue in report.issues]


def test_wrong_grid_spacing_is_reported() -> None:
    report = validate_oisst_dataset(make_dataset(spacing=0.5))

    checks = [issue.check for issue in report.issues]
    assert "lat_spacing" in checks
    assert "lon_spacing" in checks


def test_out_of_range_values_are_reported() -> None:
    ds = make_dataset()
    ds["sst"][0, 2, 2] = 120.0

    report = validate_oisst_dataset(ds)

    assert "sst_range" in [issue.check for issue in report.issues]


def test_unstable_land_mask_is_reported() -> None:
    ds = make_dataset()
    # Missing on one day only: a moving mask, not land.
    ds["sst"][2, 3, 3] = np.nan

    report = validate_oisst_dataset(ds)

    issue = next(i for i in report.issues if i.check == "land_mask_stability")
    assert "1 cell(s)" in issue.detail


def test_raise_for_status_lists_every_issue() -> None:
    report = validate_oisst_dataset(make_dataset(spacing=0.5))

    with pytest.raises(DataValidationError) as excinfo:
        report.raise_for_status()

    message = str(excinfo.value)
    assert "lat_spacing" in message
    assert "lon_spacing" in message


def test_validate_time_axis_only_checks_time() -> None:
    # Wrong spacing would fail the full check, but not the time-only one.
    assert validate_time_axis(make_dataset(spacing=0.5)).ok

    ds = make_dataset().drop_vars("time")
    assert not validate_time_axis(ds).ok
