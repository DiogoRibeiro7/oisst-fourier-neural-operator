# Multivariate forcing — sources, alignment, and interpretation limits

The SST-only experiment in notebooks `00`–`11` remains the interpretable baseline. This
document covers the separate question posed by notebooks `12` and `13`:

> Does exogenous atmospheric or ocean information improve 7-day SST field forecasting
> beyond what SST history alone supports?

The bar is deliberately high. A multivariate model that fits better, or that beats the
SST-only model while both lose to persistence, has **not** answered it. The question is
whether exogenous fields add *out-of-sample information beyond SST persistence*.

## Sources

### NOAA OISST auxiliary variables

`err`, `ice`, and `anom` arrive in the same NCEI ERDDAP request as `sst`. No credentials,
no regridding, and the same time axis and provenance manifest.

| Variable | Units | What it is |
|---|---|---|
| `err` | Celsius | OISST's own analysis-error estimate — a confidence channel |
| `ice` | fraction | Sea-ice concentration |
| `anom` | Celsius | Anomaly against OISST's climatology; derived from `sst`, so it adds no independent information |

**Audit finding:** `ice` is **100% missing over the Northeast Atlantic study region**
(30–50°N) — there is no sea ice at those latitudes. It is a reasonable candidate in
general and useless here, which is exactly the kind of thing notebook `12` exists to catch
before modelling rather than after. `PerVariableStandardizer.fit` refuses such a channel
with an explicit error.

`err` deserves care of a different kind. It describes the **analysis**, not the ocean: its
structure follows observation coverage. A model that improves when given `err` may be
learning where the product is reliable, which is a legitimate confidence feature but not a
physical driver.

### ERA5 (Copernicus Climate Change Service)

Verified against the live CDS catalogue entry:

| Property | Value |
|---|---|
| Dataset | `reanalysis-era5-single-levels` |
| Provider | ECMWF / Copernicus Climate Change Service (C3S) |
| Licence | CC-BY-4.0, and the Copernicus licence must be accepted on the CDS account |
| DOI | [10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47) |
| Grid | regular 0.25° latitude–longitude for the reanalysis |
| Cadence | hourly |
| Latency | about 5 days |

**Access requires a free Copernicus CDS account and a personal access token.** The request
builder in `oisst_fno.multivariate` is offline and unit-tested, but downloading needs those
credentials, so the ERA5 arm of the experiment cannot run in CI and has not been run here.

**ERA5T caveat.** The recent end of the record is ERA5T, which may be revised two to three
months later. An ERA5T value used as a forecast input today is not guaranteed to be the
value stored at that date afterwards, so a multivariate result over the most recent months
is not exactly reproducible. This is a threat to validity for the test period specifically,
since the test split is the most recent data.

## Alignment decisions

These are **choices**, not facts, and each one could reasonably have gone another way. They
are recorded in `oisst_fno.multivariate.ALIGNMENT_DECISIONS` and asserted in the tests.

### Grids are not the same grid

ERA5 and OISST are both 0.25°, which invites the assumption that they align. They do not:
OISST cells are centred on `.125/.375/.625/.875` while ERA5 sits on whole and quarter
degrees — a half-cell offset. Exogenous fields are always interpolated onto the OISST grid.

### Orientation

ERA5 latitude descends north to south; OISST ascends. Latitude is sorted ascending before
interpolation, so the two orientations cannot be silently mixed — a mismatch would flip a
field north-for-south with no error raised.

Longitude: OISST is `[0, 360)` at source and `open_oisst` converts to `[-180, 180)`. ERA5
is requested directly in `[-180, 180)`, so both share one convention before regridding.

### Time

ERA5 is hourly; OISST is a daily analysis stamped 12:00 UTC. Hourly fields are reduced over
the **24 hours ending at that timestamp** — not a centred window and certainly not a forward
one, so an exogenous value never summarises hours occurring after the SST field it
accompanies.

OISST's own analysis does use observations across the whole day, so a centred window would
arguably *match* OISST better. The trailing window is the conservative choice, chosen
because this project treats leakage as the more serious error.

### Reductions and interpolation

- Accumulated fluxes (`sshf`, `slhf`, `ssr`) are stored over the preceding hour and are
  **summed** over the day, not averaged.
- Interpolation is bilinear for continuous fields, nearest-neighbour for fields with sharp
  boundaries such as sea-ice concentration.
- The OISST land mask is reapplied after regridding, so exogenous channels are missing
  exactly where SST is. Every masked metric therefore averages over the same cells no
  matter how many channels a model uses.

### Forcing is taken at the last observed day

Exogenous channels enter at the **last input day** of each window, never at the target
date. Using forcing from the target day would be forecasting with information that does not
exist at prediction time.

### Normalisation

Per-variable, fitted on the **training split only**. Per-variable rather than global because
the units differ by orders of magnitude — pressure in pascals beside wind in m/s — and one
shared statistic would let a single channel dominate the input scale.

## Comparison design

Prespecified arms, fixed in advance so the reported comparison is confirmatory rather than
the best of however many combinations were tried:

| Arm | Exogenous channels |
|---|---|
| `sst-only` | none — the interpretable baseline, unchanged |
| `sst+wind` | `u10`, `v10` |
| `sst+air-temperature` | `t2m` |
| `sst+heat-flux` | `sshf`, `slhf`, `ssr` |
| `sst+all` | all seven |

Every arm predicts identical target dates from the same SST history under a matched
architecture, optimizer, schedule, and epoch budget. Notebook `13` asserts the identical
targets rather than assuming them.

Adding channels does add parameters in the lifting layer. That difference is reported
rather than hidden, because "the bigger model won" is the alternative explanation this
comparison has to rule out.

## Interpretation limits

1. **Predictive is not causal.** If wind channels improve the forecast, wind carries
   information about future SST in this dataset. That is not an attribution of SST change
   to wind forcing, and nothing in this design supports one.
2. **ERA5 is a reanalysis, not observation.** Like OISST it is a model product. Agreement
   between them can partly reflect shared assimilated inputs rather than independent
   physical signal.
3. **Regime-specific gains are regime-specific findings.** A gain concentrated in one season
   is a claim about that season. Subgroup splits chosen after seeing test results are
   exploratory and must be labelled as such; confirmatory thresholds come from training and
   validation only.
4. **A negative result is a valid outcome.** If exogenous forcing adds nothing beyond SST
   history, that is the finding.

## Status

The pipeline, the audit, and the comparison harness are implemented and tested. The OISST
auxiliary path has been run against the live service. **The empirical comparison has not
been run**: the ERA5 arms need CDS credentials, and every arm is a full training run.
Notebook `13` raises rather than emitting placeholder numbers, so no result is reported
that was not actually produced.
