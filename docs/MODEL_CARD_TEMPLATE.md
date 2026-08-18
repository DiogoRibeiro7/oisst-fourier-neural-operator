# Model Card — OISST FNO

## Model purpose

Describe the precise forecast task and horizon.

State explicitly that this is supervised forecasting of an analyzed SST product, not PDE discovery.

## Research question

Report the falsifiable question:

> Does the FNO add forecast information beyond persistence and simpler low-rank/local models, and at which spatial scales?

## Training data

Record exact dates, variables, spatial bounds, exclusions, OISST version, and acquisition hash/provenance.

Include the limitation that OISST is bias-adjusted and optimum-interpolated rather than raw gridded observations.

## Inputs and outputs

Document tensor shapes, normalization, coordinate channels, lookback, forecast horizon, and mask handling.

## Architecture

Record width, number of Fourier modes, depth, padding, parameter count, optimizer, scheduler, seed, and tuning budget.

Most of this is written automatically by notebook `07` and should be copied from the
artifacts rather than retyped:

- `artifacts/metrics/fno_experiment.json` — the full `ExperimentConfig` (data dates,
  region, lookback, horizon, scaler, model shape, optimizer, learning rate, weight decay,
  batch size, epochs, early-stopping rule, seed) plus the environment it ran in: Python
  and package versions, platform, git commit and whether the tree was dirty, torch and
  CUDA versions, and the GPU model.
- `artifacts/metrics/fno_training_history.json` — per-epoch train and validation loss,
  learning rate, wall-clock seconds, gradient norm, and peak GPU memory.
- `artifacts/models/fno_best.pt` — the best-validation checkpoint, not the last epoch.

State the learning-curve verdict (`converged`, `underfit`, `overfit`, or `unstable`) that
notebook `07` reports, and do not present results from an unstable run.

### Reproducibility caveat

Seeding fixes Python, NumPy, and PyTorch, and makes CPU runs reproducible. It does **not**
guarantee bitwise-identical GPU results: cuDNN algorithm selection, atomic float
accumulation, cuFFT planning, mixed precision, and driver or hardware differences all
introduce variation. `oisst_fno.experiment.GPU_NONDETERMINISM_NOTES` lists these and
notebook `07` prints them on every CUDA run. Two runs of the same configuration should
therefore be *statistically consistent*, not identical, and any claimed difference between
configurations must be larger than that run-to-run spread.

## Baselines

At minimum report on identical forecast cases:

- train-only seasonal climatology;
- persistence;
- linear trend extrapolation;
- EOF+ridge global linear dynamics;
- local CNN if trained.

For a research-grade benchmark add ConvLSTM/U-Net-style models and the official NeuralOperator FNO under a documented tuning budget.

## Evaluation

At minimum report:

- masked RMSE;
- MAE;
- anomaly correlation;
- persistence skill;
- paired daily RMSE difference FNO minus persistence;
- moving-block-bootstrap confidence interval for that paired difference;
- spatial error maps;
- scale-resolved Fourier error energy and skill.

## Regime analysis

Report predefined meteorological seasons.

If using a “hard persistence” subgroup, define the threshold from validation data and freeze it before test evaluation.

## Spectral interpretation

State that land masking and tapering alter absolute spectra. Frequency-band results are comparative diagnostics, not uncontaminated physical spectra.

Report whether any aggregate FNO gain is accompanied by high-frequency degradation.

## Stress tests

Include season, validation-defined difficult cases, extreme-SST regimes defined without test leakage, resolution changes, random seeds, and plausible distribution shifts.

## Statistical uncertainty

Explain the temporal dependence of forecast errors and the block-bootstrap design, including block length and sensitivity checks.

## Novelty boundary

Do not describe the FNO method, ocean FNO application, or Atlantic SST FNO forecasting as novel. Reference `docs/RESEARCH_POSITIONING.md`.

## Limitations

State where the model fails and what alternative explanations remain. Specifically discuss:

- OISST processing;
- persistence/smoothness;
- missing atmospheric/ocean forcing variables;
- one-region external validity;
- tuning-budget fairness;
- mask-induced spectral artifacts.

## Intended use

Research and portfolio demonstration unless separately validated for operational forecasting.

## Out-of-scope use

Operational marine safety, navigation, fisheries decisions, or climate attribution without additional validation.
