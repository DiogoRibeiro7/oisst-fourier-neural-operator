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

## Leakage policy

- Chronological split only.
- Fit normalization using training period only.
- Fit EOF/PCA bases and learned baselines on training only.
- Do not compute climatology from validation/test observations when evaluating a forecast.
- Hyperparameter selection may use validation data but not the test period.
- Test subgroup thresholds must be prespecified or estimated from training/validation, not optimized after seeing test results.
- Test metrics should be produced once per finalized experiment specification where practical.
