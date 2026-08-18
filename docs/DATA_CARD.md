# Data Card — NOAA OISST v2.1

## Dataset

NOAA/NCEI Optimum Interpolation Sea Surface Temperature (OISST), version 2.1.

- Spatial resolution: 0.25°.
- Temporal resolution: daily.
- Product type: Level-4 gridded climate analysis.
- Inputs include satellite SST plus in-situ ship, buoy, and Argo observations, with sea-ice information used in the analysis.
- Official DOI: https://doi.org/10.25921/RE9P-PT57

NOAA describes OISST as a spatially complete field constructed by bias-adjusting and blending observations from multiple platforms and filling spatial gaps using optimum interpolation.

## Study subset

Default project bounds use NOAA's native longitude convention:

- latitude: 30.125 to 50.125°N;
- longitude: 330.125 to 355.125°E, approximately 29.875 to 4.875°W.

The subset covers much of the Northeast Atlantic west of Iberia and France.

## Variables

Primary v0.1 model:

- `sst`: daily sea-surface temperature, °C.

Available for later extensions:

- `anom`: NOAA daily SST anomaly;
- `err`: estimated analysis error standard deviation;
- `ice`: sea-ice concentration.

## Central validity limitation

OISST is **not raw sensor data** and is not an independent measurement at every grid point. It is an analyzed/reconstructed product.

Therefore:

\[
\text{model skill on OISST}
\neq
\text{model skill on raw independent ocean observations}.
\]

The optimum-interpolation pipeline can contribute to spatial smoothness and temporal predictability. An FNO may partly learn regularities of the analysis product rather than only physical ocean evolution.

This does not invalidate the forecasting task, but it constrains interpretation.

## Version and processing changes

OISST v2.1 superseded v2 in 2020. NOAA reports important input/processing improvements for 2016 onward, including additional buoy data, Argo observations, changed satellite inputs, revised ship-buoy corrections, and revised sea-ice-to-SST conversion.

Any historical extension should check whether conclusions are stable across processing eras.

## Spectral-analysis limitation

Land masking creates discontinuities that inject Fourier power at multiple frequencies. Project spectral comparisons therefore:

1. apply the same mask to every model and truth field;
2. apply a common spatial taper;
3. interpret frequency-band metrics relatively between models;
4. avoid treating the masked-domain spectrum as a pure physical ocean spectrum.

A stronger extension should repeat the analysis on fully oceanic subdomains.

## External-validity limitation

The Northeast Atlantic is one region. Results may not transfer to:

- western boundary currents;
- equatorial regions;
- semi-enclosed seas;
- coastal domains;
- other SST analysis/reanalysis products.

## Acquisition provenance and validation

Every download through `oisst_fno.data.download_subset` writes a sidecar manifest,
`<file>.nc.manifest.json`, so a processed dataset can always be traced to the exact
source that produced it:

| Field | Purpose |
|---|---|
| `source_url` | The full ERDDAP request, including dates, bounds, and variables |
| `downloaded_at` | UTC timestamp of the download |
| `start_date`, `end_date`, `variables` | What was requested |
| `lat_min`/`lat_max`, `lon_min`/`lon_max` | Spatial bounds, on NOAA's [0, 360) convention |
| `file_bytes`, `sha256` | Detects later truncation or modification |
| `dataset_id`, `dataset_doi`, `product_version` | Which NOAA product version this is |
| `smoke_test` | Whether this is a short pipeline-check download rather than study data |

`open_oisst` verifies the manifest before opening, so a corrupted file raises rather than
being silently analysed.

The following dataset properties were verified against the live NCEI ERDDAP metadata and
are encoded as constants in `oisst_fno`:

- product version `Version v02r01`, NOAA Level 4;
- variables `sst`, `anom`, `err`, `ice`; `sst` in degrees Celsius;
- stored `valid_min`/`valid_max` of -300/4500 in hundredths of a degree, that is -3 to
  45 °C;
- 0.25-degree grid spacing in both latitude and longitude;
- longitude on [0, 360), latitude within -89.875 to 89.875.

`oisst_fno.validation.validate_oisst_dataset` checks a downloaded subset against those
properties: coordinate presence and dimension order, strictly increasing coordinates,
0.25-degree spacing, daily continuity, duplicated timestamps, values inside the published
range, and land-mask stability over time.

Failures are reported, never repaired. A land mask that changes between time steps, for
example, would silently alter which cells the masked metrics average over, so it is
raised as an error rather than filled in.

Downloads are written to a `.part` file and moved into place only after the body is
complete, the declared `Content-Length` matches, and the payload really is NetCDF —
ERDDAP reports failures as HTML, which would otherwise be saved with a `.nc` extension.
Transient failures are retried with bounded exponential backoff; client errors such as an
out-of-range date fail immediately.

## Leakage policy

- Chronological split only.
- Fit normalization using training period only.
- Fit EOF/PCA bases and learned baselines on training only.
- Do not compute climatology from validation/test observations when evaluating a forecast.
- Hyperparameter selection may use validation data but not the test period.
- Test subgroup thresholds must be prespecified or estimated from training/validation, not optimized after seeing test results.
- Test metrics should be produced once per finalized experiment specification where practical.
