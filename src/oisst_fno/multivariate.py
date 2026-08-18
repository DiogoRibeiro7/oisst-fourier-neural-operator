"""Exogenous atmospheric and ocean forcing for the SST forecasting experiment.

The SST-only experiment stays the interpretable baseline. This module adds the pieces
needed to ask a separate question: does exogenous information improve 7-day SST field
forecasting beyond what SST history alone supports?

Two sources are handled, and they differ in what they cost to obtain:

- **OISST auxiliary variables** (``err``, ``ice``, ``anom``) come from the same NCEI
  ERDDAP request as ``sst``. No credentials, no regridding, and already on the study
  grid — so they are the immediately runnable exogenous channels.
- **ERA5** atmospheric fields require a free Copernicus CDS account and a personal
  access token. The request builder here is offline and testable; actually fetching the
  data needs those credentials, so the ERA5 arm of the experiment cannot run in CI.

Alignment decisions are recorded in :data:`ALIGNMENT_DECISIONS` and documented in
``docs/MULTIVARIATE.md``, because they are choices rather than facts.

Nothing here fits preprocessing on anything but a training split, and nothing here
licenses a causal reading: a channel that improves forecasts is predictive, which is not
the same as physically driving the SST evolution.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from .data import ForecastSpec, Region

FloatArray = NDArray[np.float32]

# Verified against the live Copernicus CDS catalogue entry for this collection.
ERA5_DATASET_ID = "reanalysis-era5-single-levels"
ERA5_DOI = "10.24381/cds.adbb2d47"
ERA5_LICENSE = "CC-BY-4.0"
ERA5_PROVIDER = "ECMWF / Copernicus Climate Change Service (C3S)"
ERA5_GRID_DEG = 0.25
# ERA5 is updated daily with roughly five days of latency. The recent end of the record
# is ERA5T, which can be revised two to three months later.
ERA5_LATENCY_DAYS = 5

OISST_ANALYSIS_TIME_UTC = 12

ALIGNMENT_DECISIONS: tuple[str, ...] = (
    "ERA5 and OISST are both on a 0.25-degree grid, but they are not the same grid: "
    "OISST cells are centred on .125/.375/.625/.875 while ERA5 sits on whole and "
    "quarter degrees, a half-cell offset. Exogenous fields are therefore interpolated "
    "onto the OISST grid, never assumed to match.",
    "ERA5 latitude descends from north to south; OISST ascends. Latitude is sorted "
    "ascending before interpolation so the two orientations cannot be silently mixed.",
    "OISST longitude uses [0, 360) at source but open_oisst converts to [-180, 180). "
    "ERA5 is requested in [-180, 180) so both share one convention before regridding.",
    "ERA5 is hourly; OISST is a daily analysis timestamped 12:00 UTC. Hourly fields are "
    "reduced over the 24 hours *ending* at that timestamp, never a centred or forward "
    "window, so an exogenous value never summarises hours after the SST field it "
    "accompanies. OISST's own analysis does use observations from the whole day, so "
    "this is the conservative choice rather than the matching one.",
    "Accumulated ERA5 fluxes are stored over the preceding hour; they are summed rather "
    "than averaged over the day, and reported per day.",
    "Interpolation is bilinear for continuous fields and nearest-neighbour for fields "
    "with sharp boundaries such as sea-ice concentration.",
    "The OISST land mask is reapplied after regridding, so exogenous channels are "
    "missing exactly where SST is missing and the masked metrics keep comparing the "
    "same cells.",
    "Normalisation statistics for every channel are fitted on the training split only.",
)


@dataclass(frozen=True, slots=True)
class VariableSpec:
    """One exogenous input channel.

    ``daily_reduction`` says how sub-daily data becomes a daily value, and
    ``regrid_method`` how it reaches the OISST grid. Both are recorded per variable
    because the right choice differs: fluxes accumulate, sea ice must not be smeared
    across its boundary.
    """

    name: str
    source: str
    source_name: str
    units: str
    description: str
    regrid_method: str = "linear"
    daily_reduction: str = "mean"

    def __post_init__(self) -> None:
        if self.source not in {"oisst", "era5"}:
            raise ValueError(f"Unknown source {self.source!r}; expected 'oisst' or 'era5'.")
        if self.regrid_method not in {"linear", "nearest"}:
            raise ValueError(f"Unknown regrid method {self.regrid_method!r}.")
        if self.daily_reduction not in {"mean", "sum", "max", "min", "instant"}:
            raise ValueError(f"Unknown daily reduction {self.daily_reduction!r}.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


# Atmospheric drivers with a plausible mechanism for surface temperature evolution:
# wind mixing and evaporative cooling, air-sea temperature contrast, radiative and
# turbulent heat exchange, and synoptic forcing.
ERA5_VARIABLES: dict[str, VariableSpec] = {
    "u10": VariableSpec(
        name="u10",
        source="era5",
        source_name="10m_u_component_of_wind",
        units="m s-1",
        description="Eastward 10 m wind; wind mixing deepens the surface layer.",
    ),
    "v10": VariableSpec(
        name="v10",
        source="era5",
        source_name="10m_v_component_of_wind",
        units="m s-1",
        description="Northward 10 m wind.",
    ),
    "t2m": VariableSpec(
        name="t2m",
        source="era5",
        source_name="2m_temperature",
        units="K",
        description="2 m air temperature; the air-sea contrast drives sensible heat flux.",
    ),
    "msl": VariableSpec(
        name="msl",
        source="era5",
        source_name="mean_sea_level_pressure",
        units="Pa",
        description="Mean sea-level pressure, a proxy for synoptic weather regime.",
    ),
    "sshf": VariableSpec(
        name="sshf",
        source="era5",
        source_name="surface_sensible_heat_flux",
        units="J m-2",
        description="Surface sensible heat flux, accumulated hourly.",
        daily_reduction="sum",
    ),
    "slhf": VariableSpec(
        name="slhf",
        source="era5",
        source_name="surface_latent_heat_flux",
        units="J m-2",
        description="Surface latent heat flux, accumulated hourly.",
        daily_reduction="sum",
    ),
    "ssr": VariableSpec(
        name="ssr",
        source="era5",
        source_name="surface_net_solar_radiation",
        units="J m-2",
        description="Surface net solar radiation, accumulated hourly.",
        daily_reduction="sum",
    ),
}

# Available from the same ERDDAP request as sst, so these need no credentials and no
# regridding. err is an analysis-uncertainty field, not an observation.
OISST_AUX_VARIABLES: dict[str, VariableSpec] = {
    "err": VariableSpec(
        name="err",
        source="oisst",
        source_name="err",
        units="Celsius",
        description=(
            "OISST analysis error estimate. A confidence channel: it says where the "
            "analysis itself is uncertain, not where the ocean is."
        ),
    ),
    "ice": VariableSpec(
        name="ice",
        source="oisst",
        source_name="ice",
        units="fraction",
        description="Sea-ice concentration; sharp boundaries, so nearest-neighbour.",
        regrid_method="nearest",
    ),
    "anom": VariableSpec(
        name="anom",
        source="oisst",
        source_name="anom",
        units="Celsius",
        description=(
            "OISST anomaly against its own climatology. Derived from sst, so it adds "
            "no independent information and exists for diagnostics only."
        ),
    ),
}

ALL_VARIABLES: dict[str, VariableSpec] = {**ERA5_VARIABLES, **OISST_AUX_VARIABLES}


def variable_spec(name: str) -> VariableSpec:
    """Look up a known variable specification by canonical name."""
    try:
        return ALL_VARIABLES[name]
    except KeyError:
        raise KeyError(f"Unknown variable {name!r}. Known: {sorted(ALL_VARIABLES)}") from None


@dataclass(frozen=True, slots=True)
class AblationSpec:
    """One arm of the multivariate comparison.

    Every arm predicts the same target dates from the same SST history; they differ
    only in which exogenous channels are appended.
    """

    name: str
    exogenous: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def standard_ablations() -> tuple[AblationSpec, ...]:
    """The prespecified comparison arms.

    Fixed in advance so that the reported comparison is confirmatory rather than the
    best of however many combinations were tried. Anything beyond this list is
    exploratory and must be labelled as such.
    """
    return (
        AblationSpec(
            name="sst-only",
            exogenous=(),
            rationale="Interpretable baseline; the original experiment, unchanged.",
        ),
        AblationSpec(
            name="sst+wind",
            exogenous=("u10", "v10"),
            rationale="Wind mixing is the most direct mechanical driver of surface change.",
        ),
        AblationSpec(
            name="sst+air-temperature",
            exogenous=("t2m",),
            rationale="Air-sea temperature contrast drives sensible heat exchange.",
        ),
        AblationSpec(
            name="sst+heat-flux",
            exogenous=("sshf", "slhf", "ssr"),
            rationale="Direct surface energy budget terms.",
        ),
        AblationSpec(
            name="sst+all",
            exogenous=("u10", "v10", "t2m", "msl", "sshf", "slhf", "ssr"),
            rationale="Everything together; more capacity, so it must beat the others on merit.",
        ),
    )


def era5_area_from_region(region: Region) -> list[float]:
    """Convert a study :class:`~oisst_fno.data.Region` to a CDS ``area`` box.

    CDS expects ``[North, West, South, East]`` on [-180, 180), while the study region
    carries NOAA's [0, 360) longitudes.
    """

    def to_signed(longitude: float) -> float:
        return ((longitude + 180.0) % 360.0) - 180.0

    return [region.lat_max, to_signed(region.lon_min), region.lat_min, to_signed(region.lon_max)]


def build_era5_request(
    variables: tuple[str, ...],
    start_date: str,
    end_date: str,
    region: Region,
    *,
    grid_deg: float = ERA5_GRID_DEG,
) -> dict[str, Any]:
    """Build a ``cdsapi`` request for hourly ERA5 single-level fields.

    Returns the request dictionary only; submitting it requires a Copernicus CDS account
    and a personal access token, so this function performs no network access and is
    testable offline.

    All 24 hours are requested because the daily reduction happens afterwards, and
    accumulated fluxes need every hour to be summed correctly.
    """
    if not variables:
        raise ValueError("At least one variable is required.")

    unknown = [name for name in variables if name not in ERA5_VARIABLES]
    if unknown:
        raise ValueError(f"Not ERA5 variables: {unknown}. Known: {sorted(ERA5_VARIABLES)}")

    start = np.datetime64(start_date, "D")
    end = np.datetime64(end_date, "D")
    if end < start:
        raise ValueError("end_date must not precede start_date.")

    days = np.arange(start, end + np.timedelta64(1, "D"))
    years = sorted({str(day.astype("datetime64[Y]").astype(int) + 1970) for day in days})
    months = sorted({f"{day.astype(object).month:02d}" for day in days})
    day_numbers = sorted({f"{day.astype(object).day:02d}" for day in days})

    return {
        "dataset": ERA5_DATASET_ID,
        "request": {
            "product_type": ["reanalysis"],
            "variable": [ERA5_VARIABLES[name].source_name for name in variables],
            "year": years,
            "month": months,
            "day": day_numbers,
            "time": [f"{hour:02d}:00" for hour in range(24)],
            "area": era5_area_from_region(region),
            "grid": [grid_deg, grid_deg],
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
    }


def daily_from_hourly(
    ds: xr.Dataset,
    specs: tuple[VariableSpec, ...],
    *,
    analysis_hour: int = OISST_ANALYSIS_TIME_UTC,
) -> xr.Dataset:
    """Reduce hourly fields to daily values aligned with the OISST analysis timestamp.

    Each daily value summarises the 24 hours **ending** at ``analysis_hour``, so an
    exogenous channel never contains information from after the SST field it accompanies.
    A centred or forward window would put future hours into a forecasting input.

    Accumulated fluxes are summed; everything else is averaged, or sampled at the start
    of the window when the reduction is ``instant``.
    """
    if "time" not in ds.coords:
        raise ValueError("Hourly dataset needs a time coordinate.")

    # Shifting forward by the analysis hour makes each calendar-day bin cover the
    # 24 hours ending at that hour; the label is shifted back afterwards.
    offset = np.timedelta64(analysis_hour, "h")
    shifted = ds.assign_coords(time=ds["time"] + offset)

    reduced: dict[str, xr.DataArray] = {}
    for spec in specs:
        if spec.source_name not in shifted and spec.name not in shifted:
            raise KeyError(f"{spec.name!r} is not present in the hourly dataset.")
        key = spec.name if spec.name in shifted else spec.source_name
        grouped = shifted[key].resample(time="1D")

        if spec.daily_reduction == "sum":
            values = grouped.sum()
        elif spec.daily_reduction == "max":
            values = grouped.max()
        elif spec.daily_reduction == "min":
            values = grouped.min()
        elif spec.daily_reduction == "instant":
            values = grouped.first()
        else:
            values = grouped.mean()
        reduced[spec.name] = values

    daily = xr.Dataset(reduced)
    # Label each window by the analysis timestamp it ends at.
    return daily.assign_coords(time=daily["time"] + offset)


def regrid_to_reference(
    source: xr.DataArray,
    reference: xr.DataArray,
    *,
    method: str = "linear",
) -> xr.DataArray:
    """Interpolate ``source`` onto the latitude/longitude grid of ``reference``.

    Both grids are sorted ascending first, because ERA5 latitudes descend while OISST
    latitudes ascend and silently mixing the two flips the field.
    """
    if method not in {"linear", "nearest"}:
        raise ValueError(f"Unknown method {method!r}; expected 'linear' or 'nearest'.")
    for name in ("lat", "lon"):
        if name not in source.coords:
            raise ValueError(f"Source is missing the {name!r} coordinate.")
        if name not in reference.coords:
            raise ValueError(f"Reference is missing the {name!r} coordinate.")

    ordered = source.sortby("lat").sortby("lon")
    return ordered.interp(
        lat=reference["lat"],
        lon=reference["lon"],
        method=method,  # type: ignore[arg-type]
        kwargs={"fill_value": None},
    )


def align_exogenous(
    exogenous: xr.Dataset,
    reference: xr.DataArray,
    specs: tuple[VariableSpec, ...],
    *,
    apply_reference_mask: bool = True,
) -> xr.Dataset:
    """Regrid exogenous variables onto the SST grid and match its time axis.

    The reference land mask is reapplied afterwards so that exogenous channels are
    missing exactly where SST is, which keeps every masked metric averaging over the
    same cells regardless of how many channels a model uses.
    """
    aligned: dict[str, xr.DataArray] = {}
    land = ~np.isfinite(np.asarray(reference.values)).any(axis=0)

    for spec in specs:
        if spec.name not in exogenous:
            raise KeyError(f"{spec.name!r} is missing from the exogenous dataset.")
        regridded = regrid_to_reference(exogenous[spec.name], reference, method=spec.regrid_method)
        matched = regridded.sel(time=reference["time"])
        if apply_reference_mask:
            matched = matched.where(~xr.DataArray(land, dims=("lat", "lon")))
        aligned[spec.name] = matched

    return xr.Dataset(aligned)


@dataclass
class PerVariableStandardizer:
    """Per-channel mean and standard deviation, fitted on a training split only.

    Kept separate from :class:`~oisst_fno.data.Standardizer` because exogenous channels
    have wildly different units — pressure in pascals beside wind in m/s — so one global
    statistic would let a single channel dominate the input scale.
    """

    stats: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def fit(cls, ds: xr.Dataset, variables: tuple[str, ...]) -> PerVariableStandardizer:
        """Fit statistics on the given dataset, which must be the training split."""
        stats: dict[str, tuple[float, float]] = {}
        for name in variables:
            if name not in ds:
                raise KeyError(f"{name!r} is missing from the dataset.")
            values = np.asarray(ds[name].values, dtype=np.float64)
            finite = np.isfinite(values)
            if not finite.any():
                raise ValueError(
                    f"{name!r} is entirely missing over this region, so it carries no "
                    "information. Sea ice, for example, is absent from the Northeast "
                    "Atlantic domain. Audit the variable in notebook 12 before "
                    "including it."
                )
            observed = values[finite]
            mean = float(observed.mean())
            std = float(observed.std())
            if not np.isfinite(mean) or not np.isfinite(std):
                raise ValueError(f"{name!r} produced non-finite statistics.")
            if std <= 0:
                raise ValueError(
                    f"{name!r} has zero variance in the training split, so it carries no "
                    "information and cannot be standardized."
                )
            stats[name] = (mean, std)
        return cls(stats=stats)

    def transform(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize every fitted channel, leaving missing values missing."""
        out: dict[str, xr.DataArray] = {}
        for name, (mean, std) in self.stats.items():
            if name not in ds:
                raise KeyError(f"{name!r} is missing from the dataset.")
            out[name] = (ds[name] - mean) / std
        return xr.Dataset(out)

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Return a JSON-serializable representation."""
        return {name: {"mean": m, "std": s} for name, (m, s) in self.stats.items()}

    def save(self, path: Path) -> Path:
        """Write the statistics as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> PerVariableStandardizer:
        """Read statistics written by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(stats={name: (row["mean"], row["std"]) for name, row in payload.items()})


def channel_layout(lookback_days: int, exogenous: tuple[str, ...]) -> tuple[str, ...]:
    """Name every input channel in order, for the record and for debugging.

    The layout is: ``lookback_days`` SST history channels, then one channel per
    exogenous variable taken at the most recent input day, then the ocean mask.
    """
    sst = tuple(f"sst_t-{lookback_days - 1 - index}" for index in range(lookback_days))
    return sst + tuple(f"{name}_t0" for name in exogenous) + ("ocean_mask",)


class MultivariateWindowDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """SST history plus exogenous channels, over leakage-safe temporal windows.

    Exogenous fields are taken at the **last input day** of each window, never at the
    target date: using forcing from the target day would be forecasting with information
    that does not exist at prediction time.

    With no exogenous variables this is equivalent to
    :class:`~oisst_fno.data.SSTWindowDataset`, so the SST-only arm of the comparison runs
    through exactly the same code path as the multivariate arms.
    """

    def __init__(
        self,
        sst: np.ndarray,
        spec: ForecastSpec,
        exogenous: np.ndarray | None = None,
        exogenous_names: tuple[str, ...] = (),
    ) -> None:
        array = np.asarray(sst, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError("sst must have shape [time, height, width].")
        minimum = spec.lookback_days + spec.horizon_days
        if array.shape[0] < minimum:
            raise ValueError(f"Need at least {minimum} time steps, got {array.shape[0]}.")

        if exogenous is None:
            forcing = np.zeros((array.shape[0], 0, *array.shape[1:]), dtype=np.float32)
        else:
            forcing = np.asarray(exogenous, dtype=np.float32)
            if forcing.ndim != 4:
                raise ValueError("exogenous must have shape [time, channel, height, width].")
            if forcing.shape[0] != array.shape[0]:
                raise ValueError("exogenous and sst must share the same time axis.")
            if forcing.shape[2:] != array.shape[1:]:
                raise ValueError("exogenous and sst must share the same spatial grid.")
        if len(exogenous_names) != forcing.shape[1]:
            raise ValueError(
                f"{len(exogenous_names)} names given for {forcing.shape[1]} exogenous channels."
            )

        self._spec = spec
        self._names = tuple(exogenous_names)
        self._mask: FloatArray = np.asarray(np.isfinite(array).any(axis=0), dtype=np.float32)
        self._sst: FloatArray = np.asarray(np.nan_to_num(array, nan=0.0), dtype=np.float32)
        self._forcing: FloatArray = np.asarray(np.nan_to_num(forcing, nan=0.0), dtype=np.float32)
        self._steps = int(array.shape[0])

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Ordered names of the channels this dataset yields."""
        return channel_layout(self._spec.lookback_days, self._names)

    def __len__(self) -> int:
        return self._steps - self._spec.lookback_days - self._spec.horizon_days + 1

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index
        stop = start + self._spec.lookback_days
        last_input = stop - 1
        target_index = stop + self._spec.horizon_days - 1

        history = self._sst[start:stop]
        forcing = self._forcing[last_input]  # exogenous at the last observed day only
        x = torch.from_numpy(np.concatenate((history, forcing), axis=0).copy())
        y = torch.from_numpy(self._sst[target_index : target_index + 1].copy())
        mask = torch.from_numpy(self._mask[None, ...].copy())
        return x, y, mask
