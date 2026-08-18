# Threats to Validity

## 1. OISST is an analyzed product, not raw ocean truth

NOAA OISST v2.1 combines satellite and in-situ observations, applies bias adjustments, and uses optimum interpolation to construct a complete daily 0.25° field.

Consequences:

- spatial smoothness partly reflects the analysis procedure;
- apparent forecastability can partly reflect temporal/spatial regularization in the product;
- performance should not be interpreted as performance on raw independent observations;
- comparison with a second SST product would strengthen external validity.

## 2. Persistence is intrinsically strong

SST has substantial temporal memory. A low absolute RMSE does not establish useful learned dynamics.

Primary comparison:

\[
Skill=1-\frac{RMSE_{model}}{RMSE_{persistence}}.
\]

Daily paired error differences should also be reported.

## 3. Forecast cases are temporally dependent

Overlapping lookback windows and ocean persistence make daily forecast errors non-IID.

Do not use an IID standard error or naive bootstrap across individual days. The project uses a moving-block bootstrap for paired forecast-date differences.

The block length itself is a sensitivity parameter and should be varied in a research-grade analysis.

## 4. Application novelty is limited

FNOs have already been used for ocean forecasting, and 2026 work includes Atlantic SST as a real-data application. This project cannot claim novelty from the combination “FNO + Atlantic SST”.

See `RESEARCH_POSITIONING.md`.

## 5. Missing physical and atmospheric predictors

The v0.1 model uses SST history only. Winds, surface fluxes, currents, SSH, mixed-layer structure, and other forcing variables are absent.

Therefore poor skill does not imply that neural operators are unsuitable for ocean forecasting. Conversely, good short-horizon skill does not imply that SST history is a sufficient state representation.

## 6. Spectral diagnostics are affected by masking

Land masks introduce sharp discontinuities and therefore high-frequency Fourier energy. The same mask and taper are used for all models so scale comparisons are meaningful **relatively**, but the resulting spectra should not be interpreted as uncontaminated physical ocean spectra.

A stronger extension should repeat spectral diagnostics on one or more fully oceanic subdomains.

## 7. One region and one product limit external validity

A result over the Northeast Atlantic does not automatically generalize to western boundary currents, equatorial regions, enclosed seas, or other SST products.

A paper-level claim should test multiple regions or datasets if it asserts general properties of FNOs.

## 8. Hyperparameter budget can bias model comparisons

If FNO receives substantially more tuning than CNN/ConvLSTM/U-Net or EOF-ridge baselines, the comparison is not a clean architecture test.

Record tuning budgets and use matched or explicitly justified search budgets.

## 9. Resolution generalization is easy to overclaim

The mathematical neural-operator formulation is not the same as empirical invariance to arbitrary discretization changes. Any resolution-transfer experiment must define interpolation/regridding and train/test resolutions precisely.

## 10. Post-hoc subgroup discovery can exaggerate findings

Seasons and frequency bands should be prespecified. “Hard forecast” thresholds used for test stratification should be defined from validation data, not chosen after inspecting test errors.

Exploratory post-hoc subgroups must be labeled exploratory.
