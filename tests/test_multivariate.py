"""Multivariate-forcing tests. No network access: the ERA5 request is built, never sent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from oisst_fno.data import ForecastSpec, Region
from oisst_fno.multivariate import (
    ALIGNMENT_DECISIONS,
    ERA5_DOI,
    ERA5_LICENSE,
    ERA5_VARIABLES,
    OISST_AUX_VARIABLES,
    AblationSpec,
    MultivariateWindowDataset,
    PerVariableStandardizer,
    VariableSpec,
    align_exogenous,
    build_era5_request,
    channel_layout,
    daily_from_hourly,
    era5_area_from_region,
    regrid_to_reference,
    standard_ablations,
    variable_spec,
)

# --------------------------------------------------------------------------- specs


def test_known_variables_cover_the_prompt_candidates() -> None:
    assert {"u10", "v10"} <= set(ERA5_VARIABLES), "10 m wind components"
    assert "t2m" in ERA5_VARIABLES, "2 m air temperature"
    assert {"sshf", "slhf", "ssr"} <= set(ERA5_VARIABLES), "surface heat fluxes"
    assert "msl" in ERA5_VARIABLES, "mean sea-level pressure"
    assert "err" in OISST_AUX_VARIABLES, "OISST analysis error as a confidence feature"


def test_licensing_is_recorded() -> None:
    assert ERA5_LICENSE == "CC-BY-4.0"
    assert ERA5_DOI == "10.24381/cds.adbb2d47"


def test_alignment_decisions_are_documented() -> None:
    joined = " ".join(ALIGNMENT_DECISIONS).lower()
    for topic in ("grid", "latitude", "longitude", "hourly", "mask", "training split"):
        assert topic in joined


def test_fluxes_accumulate_and_ice_uses_nearest_neighbour() -> None:
    assert ERA5_VARIABLES["sshf"].daily_reduction == "sum"
    assert ERA5_VARIABLES["u10"].daily_reduction == "mean"
    assert OISST_AUX_VARIABLES["ice"].regrid_method == "nearest"


def test_variable_spec_lookup_and_rejection() -> None:
    assert variable_spec("t2m").units == "K"
    with pytest.raises(KeyError, match="Unknown variable"):
        variable_spec("not_a_variable")


def test_variable_spec_validates_its_fields() -> None:
    with pytest.raises(ValueError, match="Unknown source"):
        VariableSpec(name="x", source="somewhere", source_name="x", units="1", description="")
    with pytest.raises(ValueError, match="regrid method"):
        VariableSpec(
            name="x",
            source="era5",
            source_name="x",
            units="1",
            description="",
            regrid_method="cubic",
        )
    with pytest.raises(ValueError, match="daily reduction"):
        VariableSpec(
            name="x",
            source="era5",
            source_name="x",
            units="1",
            description="",
            daily_reduction="median",
        )


# ----------------------------------------------------------------------- ablations


def test_standard_ablations_are_prespecified_and_cover_the_prompt() -> None:
    names = [arm.name for arm in standard_ablations()]

    assert names[0] == "sst-only", "the interpretable baseline comes first"
    assert names == [
        "sst-only",
        "sst+wind",
        "sst+air-temperature",
        "sst+heat-flux",
        "sst+all",
    ]
    assert standard_ablations()[0].exogenous == ()
    assert all(arm.rationale for arm in standard_ablations())


def test_ablation_serializes() -> None:
    arm = AblationSpec(name="x", exogenous=("u10",), rationale="because")
    assert arm.to_dict()["exogenous"] == ("u10",)


# --------------------------------------------------------------------- ERA5 request


def test_era5_area_converts_region_to_cds_box() -> None:
    area = era5_area_from_region(Region())

    north, west, south, east = area
    assert (north, south) == (50.125, 30.125)
    # NOAA 330.125/355.125 in [0, 360) become negative longitudes.
    assert west == pytest.approx(-29.875)
    assert east == pytest.approx(-4.875)
    assert north > south and west < east


def test_era5_request_is_well_formed() -> None:
    request = build_era5_request(("u10", "v10"), "2024-01-30", "2024-02-02", Region())

    assert request["dataset"] == "reanalysis-era5-single-levels"
    body = request["request"]
    assert body["variable"] == ["10m_u_component_of_wind", "10m_v_component_of_wind"]
    assert body["year"] == ["2024"]
    assert body["month"] == ["01", "02"]
    assert body["day"] == ["01", "02", "30", "31"]
    assert len(body["time"]) == 24, "all hours, because fluxes are summed over the day"
    assert body["grid"] == [0.25, 0.25]
    assert body["data_format"] == "netcdf"


def test_era5_request_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="At least one variable"):
        build_era5_request((), "2024-01-01", "2024-01-02", Region())
    with pytest.raises(ValueError, match="Not ERA5 variables"):
        build_era5_request(("err",), "2024-01-01", "2024-01-02", Region())
    with pytest.raises(ValueError, match="must not precede"):
        build_era5_request(("u10",), "2024-01-05", "2024-01-01", Region())


# ------------------------------------------------------------------ time alignment


def _hourly(days: int = 3, value_per_hour: float = 1.0) -> xr.Dataset:
    times = np.arange(
        np.datetime64("2024-01-01T00:00:00"),
        np.datetime64("2024-01-01T00:00:00") + np.timedelta64(24 * days, "h"),
        np.timedelta64(1, "h"),
    )
    shape = (len(times), 2, 2)
    return xr.Dataset(
        {
            "u10": (("time", "lat", "lon"), np.full(shape, value_per_hour, dtype=np.float32)),
            "sshf": (("time", "lat", "lon"), np.full(shape, value_per_hour, dtype=np.float32)),
        },
        coords={"time": times, "lat": [30.0, 30.25], "lon": [-30.0, -29.75]},
    )


def test_daily_reduction_averages_and_accumulates_correctly() -> None:
    specs = (ERA5_VARIABLES["u10"], ERA5_VARIABLES["sshf"])

    daily = daily_from_hourly(_hourly(days=3, value_per_hour=2.0), specs)

    # A full 24-hour window: mean stays 2, the accumulated flux sums to 48.
    interior = daily.isel(time=1)
    assert float(interior["u10"][0, 0]) == pytest.approx(2.0)
    assert float(interior["sshf"][0, 0]) == pytest.approx(48.0)


def test_daily_values_carry_the_oisst_analysis_timestamp() -> None:
    daily = daily_from_hourly(_hourly(days=3), (ERA5_VARIABLES["u10"],))

    hours = [int(str(value)[11:13]) for value in np.asarray(daily["time"].values)]
    assert set(hours) == {12}, "daily values are stamped at the 12:00 UTC analysis time"


def test_daily_window_ends_at_the_analysis_time() -> None:
    """The value at 12:00 UTC must not summarise hours after that timestamp."""
    hourly = _hourly(days=3, value_per_hour=0.0)
    # Mark only the hours strictly after noon on the second day.
    times = np.asarray(hourly["time"].values)
    after_noon = times > np.datetime64("2024-01-02T12:00:00")
    values = hourly["u10"].values.copy()
    values[after_noon] = 100.0
    hourly["u10"].values = values

    daily = daily_from_hourly(hourly, (ERA5_VARIABLES["u10"],))
    at_noon = daily.sel(time=np.datetime64("2024-01-02T12:00:00"))

    assert float(at_noon["u10"][0, 0]) == pytest.approx(0.0), (
        "hours after the analysis timestamp must not leak into its daily value"
    )


def test_daily_reduction_requires_the_variable() -> None:
    with pytest.raises(KeyError, match="t2m"):
        daily_from_hourly(_hourly(), (ERA5_VARIABLES["t2m"],))


# ---------------------------------------------------------------------- regridding


def _reference(days: int = 3) -> xr.DataArray:
    times = np.arange(
        np.datetime64("2024-01-01T12:00:00"),
        np.datetime64("2024-01-01T12:00:00") + np.timedelta64(days, "D"),
        np.timedelta64(1, "D"),
    )
    lat = 30.125 + 0.25 * np.arange(4)
    lon = -29.875 + 0.25 * np.arange(5)
    values = np.full((days, 4, 5), 15.0, dtype=np.float32)
    values[:, 0, 0] = np.nan  # land
    return xr.DataArray(
        values, dims=("time", "lat", "lon"), coords={"time": times, "lat": lat, "lon": lon}
    )


def test_regrid_lands_on_the_reference_grid() -> None:
    reference = _reference()
    # ERA5-style grid: whole/quarter degrees, latitude descending.
    source = xr.DataArray(
        np.ones((3, 5, 6), dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={
            "time": reference["time"].values,
            "lat": np.array([31.0, 30.75, 30.5, 30.25, 30.0]),
            "lon": np.array([-30.0, -29.75, -29.5, -29.25, -29.0, -28.75]),
        },
    )

    out = regrid_to_reference(source, reference)

    assert np.array_equal(out["lat"].values, reference["lat"].values)
    assert np.array_equal(out["lon"].values, reference["lon"].values)


def test_regrid_preserves_a_north_south_gradient_despite_flipped_latitude() -> None:
    """ERA5 latitudes descend. Sorting must happen before interpolation, or the field flips."""
    reference = _reference(days=1)
    descending_lat = np.array([31.0, 30.75, 30.5, 30.25, 30.0])
    # Value increases with latitude.
    values = np.tile(descending_lat.reshape(1, -1, 1), (1, 1, 6)).astype(np.float32)
    source = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={
            "time": reference["time"].values,
            "lat": descending_lat,
            "lon": np.linspace(-30.0, -28.75, 6),
        },
    )

    out = regrid_to_reference(source, reference)

    column = out.isel(time=0, lon=0).values
    assert np.all(np.diff(column) > 0), "value must still increase northward"
    assert column[0] == pytest.approx(reference["lat"].values[0], abs=1e-4)


def test_regrid_rejects_unknown_method_and_missing_coords() -> None:
    reference = _reference()
    with pytest.raises(ValueError, match="Unknown method"):
        regrid_to_reference(reference, reference, method="cubic")
    with pytest.raises(ValueError, match="missing the 'lat'"):
        regrid_to_reference(reference.rename({"lat": "y"}), reference)


def test_align_applies_the_reference_land_mask() -> None:
    reference = _reference()
    source = xr.Dataset(
        {
            "u10": xr.DataArray(
                np.ones((3, 5, 6), dtype=np.float32),
                dims=("time", "lat", "lon"),
                coords={
                    "time": reference["time"].values,
                    "lat": np.linspace(30.0, 31.0, 5),
                    "lon": np.linspace(-30.0, -28.75, 6),
                },
            )
        }
    )

    aligned = align_exogenous(source, reference, (ERA5_VARIABLES["u10"],))

    assert aligned["u10"].shape == reference.shape
    # Land in the reference must be missing in the exogenous channel too.
    assert bool(np.isnan(aligned["u10"].values[:, 0, 0]).all())
    assert not np.isnan(aligned["u10"].values[:, 1, 1]).any()


def test_align_reports_a_missing_variable() -> None:
    reference = _reference()
    with pytest.raises(KeyError, match="u10"):
        align_exogenous(xr.Dataset(), reference, (ERA5_VARIABLES["u10"],))


# -------------------------------------------------------------------- normalization


def _training_ds() -> xr.Dataset:
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "u10": (("time", "lat", "lon"), rng.normal(5.0, 2.0, (20, 3, 3)).astype(np.float32)),
            "msl": (
                ("time", "lat", "lon"),
                rng.normal(101_325.0, 500.0, (20, 3, 3)).astype(np.float32),
            ),
        }
    )


def test_per_variable_standardization_puts_channels_on_one_scale() -> None:
    ds = _training_ds()

    scaler = PerVariableStandardizer.fit(ds, ("u10", "msl"))
    out = scaler.transform(ds)

    for name in ("u10", "msl"):
        assert float(np.nanmean(out[name].values)) == pytest.approx(0.0, abs=1e-4)
        assert float(np.nanstd(out[name].values)) == pytest.approx(1.0, abs=1e-4)

    # Pressure in pascals must not dominate wind in m/s after standardization.
    assert abs(float(np.nanstd(out["msl"].values)) - float(np.nanstd(out["u10"].values))) < 1e-3


def test_standardizer_statistics_come_only_from_what_it_was_fitted_on() -> None:
    train = _training_ds()
    scaler = PerVariableStandardizer.fit(train, ("u10",))
    fitted_mean = scaler.stats["u10"][0]

    shifted = train.copy(deep=True)
    shifted["u10"] = shifted["u10"] + 50.0
    scaler.transform(shifted)

    assert scaler.stats["u10"][0] == fitted_mean, "transform must never refit"


def test_standardizer_rejects_a_channel_that_is_empty_over_the_region() -> None:
    """Auditing the real domain found `ice` is 100% missing at 30-50N."""
    ds = xr.Dataset({"ice": (("time", "lat", "lon"), np.full((4, 2, 2), np.nan, dtype=np.float32))})

    with pytest.raises(ValueError, match="entirely missing over this region"):
        PerVariableStandardizer.fit(ds, ("ice",))


def test_standardizer_rejects_a_constant_channel() -> None:
    ds = xr.Dataset({"flat": (("time", "lat", "lon"), np.ones((5, 2, 2), dtype=np.float32))})

    with pytest.raises(ValueError, match="zero variance"):
        PerVariableStandardizer.fit(ds, ("flat",))


def test_standardizer_roundtrips(tmp_path: Path) -> None:
    scaler = PerVariableStandardizer.fit(_training_ds(), ("u10", "msl"))

    restored = PerVariableStandardizer.load(scaler.save(tmp_path / "scaler.json"))

    assert restored.stats == scaler.stats


# ------------------------------------------------------------------------- dataset


def test_channel_layout_is_explicit() -> None:
    layout = channel_layout(3, ("u10", "t2m"))

    assert layout == ("sst_t-2", "sst_t-1", "sst_t-0", "u10_t0", "t2m_t0", "ocean_mask")


def _sst(steps: int = 10) -> np.ndarray:
    values = np.arange(steps * 3 * 4, dtype=np.float32).reshape(steps, 3, 4)
    values[:, 0, 0] = np.nan
    return values


def test_multivariate_dataset_stacks_history_then_forcing() -> None:
    spec = ForecastSpec(lookback_days=3, horizon_days=2)
    sst = _sst()
    forcing = np.arange(10 * 2 * 3 * 4, dtype=np.float32).reshape(10, 2, 3, 4)

    dataset = MultivariateWindowDataset(sst, spec, forcing, ("u10", "t2m"))
    x, y, mask = dataset[0]

    assert x.shape == (5, 3, 4), "3 SST history channels + 2 exogenous"
    assert y.shape == (1, 3, 4)
    assert mask.shape == (1, 3, 4)
    assert dataset.channel_names == (
        "sst_t-2",
        "sst_t-1",
        "sst_t-0",
        "u10_t0",
        "t2m_t0",
        "ocean_mask",
    )


def test_forcing_is_taken_at_the_last_input_day_not_the_target() -> None:
    """Using forcing from the target day would forecast with unavailable information."""
    spec = ForecastSpec(lookback_days=3, horizon_days=2)
    sst = _sst()
    # Channel value equals its time index, so the source day is identifiable.
    forcing = np.zeros((10, 1, 3, 4), dtype=np.float32)
    for step in range(10):
        forcing[step] = step

    dataset = MultivariateWindowDataset(sst, spec, forcing, ("u10",))
    x, _, _ = dataset[0]

    # Window 0 uses SST days 0..2 and targets day 4.
    assert float(x[3, 1, 1]) == 2.0, "forcing must come from the last observed day"


def test_dataset_without_forcing_matches_the_sst_only_path() -> None:
    from oisst_fno.data import SSTWindowDataset

    spec = ForecastSpec(lookback_days=3, horizon_days=2)
    sst = _sst()

    multivariate = MultivariateWindowDataset(sst, spec)
    baseline = SSTWindowDataset(sst, spec)

    assert len(multivariate) == len(baseline)
    for index in range(len(baseline)):
        mx, my, mm = multivariate[index]
        bx, by, bm = baseline[index]
        assert torch.equal(mx, bx)
        assert torch.equal(my, by)
        assert torch.equal(mm, bm)


def test_dataset_rejects_mismatched_forcing() -> None:
    spec = ForecastSpec(lookback_days=3, horizon_days=2)
    sst = _sst()

    with pytest.raises(ValueError, match="same time axis"):
        MultivariateWindowDataset(sst, spec, np.zeros((9, 1, 3, 4), dtype=np.float32), ("u10",))
    with pytest.raises(ValueError, match="same spatial grid"):
        MultivariateWindowDataset(sst, spec, np.zeros((10, 1, 5, 5), dtype=np.float32), ("u10",))
    with pytest.raises(ValueError, match="names given"):
        MultivariateWindowDataset(sst, spec, np.zeros((10, 2, 3, 4), dtype=np.float32), ("u10",))
    with pytest.raises(ValueError, match="shape"):
        MultivariateWindowDataset(sst, spec, np.zeros((10, 3, 4), dtype=np.float32), ("u10",))
