# OISST Fourier Neural Operator

[![ci](https://github.com/DiogoRibeiro7/oisst-fourier-neural-operator/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/oisst-fourier-neural-operator/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21995656.svg)](https://doi.org/10.5281/zenodo.21995656)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)
[![checked: mypy strict](https://img.shields.io/badge/mypy-strict-2a6db2)](https://mypy-lang.org)

A notebook-first data science project asking a deliberately narrow question:

> **When does a Fourier Neural Operator (FNO) provide forecast information beyond persistence for real sea-surface temperature fields, and at which spatial scales does that gain occur?**

The project uses NOAA/NCEI **Optimum Interpolation Sea Surface Temperature (OISST) v2.1** over the Northeast Atlantic and studies 7-day-ahead field forecasting:

\[
(SST_{t-13},\ldots,SST_t)\longmapsto SST_{t+7}.
\]

This is an empirical forecasting study. It does **not** claim that the FNO discovers or solves the governing ocean equations.

## Research positioning

The repository is original as a software/data-science project, but neither the FNO method nor the broad application of FNOs to ocean/SST forecasting is novel. Existing work includes FNO-based regional ocean modeling, Atlantic-ocean neural-operator studies, and a 2026 FNO spatiotemporal study with Atlantic SST as a real-data application.

Accordingly, the repo does **not** claim:

- a novel neural-operator architecture;
- the first use of FNOs for ocean forecasting;
- the first use of FNOs for Atlantic SST forecasting;
- that OISST forecasting is equivalent to learning an ocean PDE solution operator.

The research contribution tested here is narrower: **diagnose whether, where, and at what spatial frequencies the spectral inductive bias of an FNO improves on persistence and simpler statistical/learned baselines.** See [`docs/RESEARCH_POSITIONING.md`](docs/RESEARCH_POSITIONING.md).

## Primary research questions

1. **Aggregate skill:** does the FNO beat persistence on the untouched test period?
2. **Scale attribution:** if it does, in which spatial-frequency bands does the error reduction occur?
3. **Regime dependence:** are gains stable by season and on forecast cases that were difficult for persistence according to a validation-defined threshold?
4. **Model necessity:** does the FNO outperform a low-rank global linear EOF+ridge model and a local CNN, or can simpler dynamics explain the gain?
5. **Spectral fidelity:** does improved RMSE come at the cost of smoothing high-frequency structure?

The null result is scientifically acceptable:

\[
H_0:\quad Skill_{FNO\mid persistence}\leq 0.
\]

If the FNO does not reliably beat persistence, the repository should say so.

## Data and an important validity caveat

NOAA OISST v2.1 is a daily 0.25° Level-4 analysis produced from satellite and in-situ observations including ships, buoys, and Argo floats. NOAA applies bias adjustments and optimum interpolation to generate a spatially complete field.

Therefore:

\[
\text{forecasting OISST}
\neq
\text{forecasting unprocessed ocean observations}.
\]

Some smoothness and spatial predictability can reflect the OISST analysis pipeline itself. This is treated as a threat to validity, not hidden as a footnote. See [`docs/DATA_CARD.md`](docs/DATA_CARD.md) and [`docs/THREATS_TO_VALIDITY.md`](docs/THREATS_TO_VALIDITY.md).

## Primary experiment

- **Data:** NOAA OISST v2.1 daily SST.
- **Region:** Northeast Atlantic, roughly 30–50°N and 30–5°W.
- **Input:** previous 14 daily SST fields.
- **Target:** SST field 7 days ahead.
- **Training:** 2020–2024.
- **Validation:** 2025.
- **Test:** 2026 data available at execution time, or a user-selected final holdout period.
- **Primary null baseline:** persistence.
- **Additional baselines:** train-only seasonal climatology, linear trend, EOF+ridge global linear dynamics, and a small local CNN.
- **Main model:** compact FNO2d implemented in PyTorch.
- **Metrics:** masked RMSE, MAE, anomaly correlation, persistence skill, paired daily RMSE difference, block-bootstrap uncertainty, spatial error maps, and scale-resolved spectral error energy.

The date split is configurable. Do not silently use future observations when executing this project later.

## Why these baselines matter

SST evolves slowly, so persistence can be difficult to beat. A complex model should not receive credit for predicting a slowly changing field.

The EOF+ridge baseline adds a second challenge. It compresses the SST field into empirical orthogonal functions and learns linear dynamics in that global low-rank space. If FNO does not beat this model, an apparent advantage may not require nonlinear Fourier operator layers.

The small CNN asks a different question: whether learned **local** spatial filtering is already sufficient.

The intended hierarchy is therefore:

\[
\text{climatology}
\rightarrow
\text{persistence}
\rightarrow
\text{linear trend}
\rightarrow
\text{EOF+ridge}
\rightarrow
\text{local CNN}
\rightarrow
\text{FNO}.
\]

A research-grade extension should also add ConvLSTM/U-Net-style baselines and the official NeuralOperator implementation under a matched tuning budget.

## Notebook sequence

| Notebook | Purpose |
|---|---|
| `00_problem_framing.ipynb` | Define the falsifiable question, literature overlap, leakage rules, and success criteria. |
| `01_data_acquisition.ipynb` | Download a reproducible OISST regional subset through NOAA ERDDAP. |
| `02_data_audit_eda.ipynb` | Audit coordinates, missingness, OISST processing implications, temporal continuity, and seasonality. |
| `03_fourier_spectral_analysis.ipynb` | Examine spatial spectra while explicitly accounting for mask/taper limitations. |
| `04_supervised_dataset.ipynb` | Build leakage-safe temporal splits, normalization, masks, windows, and target dates. |
| `05_baselines.ipynb` | Establish seasonal climatology, persistence, trend, EOF+ridge, and local-CNN baselines before FNO tuning. |
| `06_fno_architecture.ipynb` | Explain and sanity-check the spectral layer and compact FNO architecture. |
| `07_train_fno.ipynb` | Train the FNO with checkpointing and validation monitoring. |
| `08_evaluation.ipynb` | Frozen test evaluation, paired FNO-vs-persistence uncertainty, and prediction export. |
| `09_spectral_skill_attribution.ipynb` | Attribute skill by spatial-frequency band, season, and validation-defined forecast-difficulty regime. |
| `10_robustness_ablation.ipynb` | Modes, width, lookback, horizon, resolution, seed, and regime sensitivity. |
| `11_conclusions_model_card.ipynb` | State only conclusions supported by stored evidence and document threats to validity. |
| `12_multivariate_audit.ipynb` | Audit candidate exogenous variables: resolution, units, grid orientation, missingness, provenance, licensing. |
| `13_multivariate_experiment.ipynb` | Compare SST-only against multivariate arms on identical targets under a matched budget. |
| `14_spatiotemporal_fno.ipynb` | Compare the channel-stacked 2-D FNO against a space-time 3-D operator at matched capacity. |

## Scientific success criteria

The FNO is not considered justified by one lower aggregate RMSE. Evidence should include:

- positive skill relative to persistence;
- a paired block-bootstrap interval for daily RMSE differences;
- comparison with EOF+ridge and learned local baselines;
- stable or interpretable results across seasons;
- scale-resolved evidence showing where forecast error is reduced;
- no severe high-frequency degradation hidden by aggregate RMSE;
- explicit negative findings where any of these conditions fail.

The block bootstrap is used because daily forecast errors are serially dependent; treating every forecast date as IID would overstate precision.

## OISST and spectral interpretation

A land mask creates sharp boundaries, which themselves contain high-frequency Fourier energy. Spectral diagnostics therefore use the same mask and taper for all compared forecasts and are interpreted **relatively**, as scale attribution between models, not as a pure physical ocean spectrum.

This caveat is part of the analysis design.

## Quick start

```bash
poetry install
poetry run python -m ipykernel install --user --name oisst-fno --display-name "OISST FNO"
poetry run jupyter lab
```

Run notebooks in numeric order. Notebooks `00`–`11` are the core SST-only study;
`12`–`13` are a multivariate extension that asks a separate question and is documented in
[`docs/MULTIVARIATE.md`](docs/MULTIVARIATE.md); `14` compares architectures.

Contributors should also install the git hooks, which strip notebook outputs and run
ruff and mypy before each commit:

```bash
poetry run pre-commit install
```

## Data source

Official NOAA resources:

- OISST product page: https://www.ncei.noaa.gov/products/optimum-interpolation-sst
- OISST v2.1 metadata and DOI: https://doi.org/10.25921/RE9P-PT57
- NCEI ERDDAP: https://www.ncei.noaa.gov/erddap/

The code uses ERDDAP dataset identifier `ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon`.

## Repository philosophy

This is intentionally not a large application framework. The scientific narrative belongs in notebooks. `src/oisst_fno/` contains only components that are reused or need unit tests:

- data acquisition and temporal-window utilities;
- download provenance manifests and structural dataset validation;
- experiment configuration, environment capture, and training history;
- exogenous-forcing alignment, per-variable normalization, and ablation specifications;
- strong reusable baselines;
- the FNO implementation, and a spatiotemporal variant for architecture comparison;
- forecast and scale-aware metrics.

## Implementation provenance

The forecasting workflow and research design are project-specific. The Fourier Neural Operator is a published method, not a novel architecture introduced here. The compact PyTorch implementation is independently structured and was audited against public reference implementations before publication.

See:

- [`docs/PROVENANCE.md`](docs/PROVENANCE.md)
- [`docs/PROVENANCE_AUDIT.md`](docs/PROVENANCE_AUDIT.md)

## Scientific guardrails

1. **Temporal split before normalization.** Fit preprocessing on training data only.
2. **No random train/test split.** Neighboring daily SST fields are dependent.
3. **Persistence is the primary null model.** Complexity alone is not evidence.
4. **Strong alternative baselines are mandatory.** FNO must justify nonlinear global spectral mixing.
5. **Use paired uncertainty.** Forecast-date errors are compared on identical targets with temporal dependence respected.
6. **Inspect space and frequency.** Scalar metrics can hide regional or high-frequency failures.
7. **Treat OISST as an analyzed product.** Do not interpret its predictability as pure ocean predictability.
8. **Do not call this PDE discovery.** This is supervised field forecasting.
9. **Do not claim application novelty.** Ocean/SST FNO forecasting already exists in the literature.
10. **Negative results are valid.** If FNO adds no robust skill, that is the conclusion.


## Quality checks

```bash
make quality
```

which runs the same checks as CI:

```bash
poetry run ruff check src tests
poetry run ruff format --check src tests
poetry run mypy src
poetry run pytest
```

CI additionally verifies that notebooks carry no stored outputs and that the citation
metadata stays valid and version-consistent.

## Contributing

Contributions are welcome, including ones that report negative results. Please read
[`CONTRIBUTING.md`](CONTRIBUTING.md) first — it sets out the development workflow and
the scientific rules that changes to data handling, baselines, or evaluation must
respect. Methodological problems such as leakage or a mis-specified baseline should be
filed with the "Scientific issue" template.

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). Security
matters are covered by the [security policy](SECURITY.md); release history is in
[`CHANGELOG.md`](CHANGELOG.md).

## Citation and archiving

Citation metadata lives in [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this
repository" button, BibTeX/APA export) and [`.zenodo.json`](.zenodo.json) (deposition
metadata used when a tagged release is archived on Zenodo).

Releases are archived on Zenodo through the GitHub integration. Two DOIs exist:

| DOI | Resolves to |
|---|---|
| [`10.5281/zenodo.21995656`](https://doi.org/10.5281/zenodo.21995656) | **Concept DOI** — always the latest version. Use this to cite the project. |
| [`10.5281/zenodo.21995657`](https://doi.org/10.5281/zenodo.21995657) | **Version DOI** — pinned to v0.1.0. Use this when reproducing a specific result. |

To cite this software:

> Ribeiro, D. (2026). *OISST Fourier Neural Operator: scale-resolved evaluation of FNO
> forecast skill against persistence* (Version 0.1.0) [Computer software].
> https://doi.org/10.5281/zenodo.21995656

Cite the OISST dataset separately — [10.25921/RE9P-PT57](https://doi.org/10.25921/RE9P-PT57)
— since NOAA's attribution requirements travel with the data.

The archive is a source snapshot: OISST data, model checkpoints, metrics, and figures
are gitignored and are not included. Reproduction re-downloads OISST v2.1 from NOAA
NCEI ERDDAP and re-runs the notebooks in order.

Full procedure, including the fields to complete before the first deposit and the
version-bump checklist: [`docs/ZENODO_RELEASE.md`](docs/ZENODO_RELEASE.md).

## License

Code is released under the MIT License. NOAA data remain governed by their source terms and attribution requirements.
