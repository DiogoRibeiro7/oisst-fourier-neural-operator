# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because this is a research repository, the changelog also records changes that affect
**reported results**, not only code. Any release that changes a published metric,
figure, or conclusion says so explicitly under "Results".

## [Unreleased]

### Added

- Hardened OISST acquisition (prompt `01`): bounded retries with exponential backoff on
  transient failures, atomic writes so a partial download is never mistaken for complete
  data, `Content-Length` and NetCDF-payload verification, and an optional smoke-test mode.
- `oisst_fno.provenance`: a sidecar manifest recording source URL, timestamp, requested
  dates, bounds, variables, file size, SHA-256, and the dataset DOI. `open_oisst` verifies
  it before opening.
- `oisst_fno.validation`: structural checks for coordinates, 0.25-degree spacing, daily
  continuity, duplicated timestamps, value range, and land-mask stability, reported as a
  list of issues rather than silently repaired.
- Dataset constants verified against the live NCEI ERDDAP metadata (product version
  `v02r01`, Celsius units, valid range, grid spacing).
- 38 offline tests covering the download, provenance, and validation paths, with the
  network mocked.

- DOI badge and citation snippet in the README, and the concept and version DOIs
  recorded in `CITATION.cff` and `docs/ZENODO_RELEASE.md`, following the Zenodo deposit
  of v0.1.0.

## [0.1.0] - 2026-08-18

First public release.

### Added

- Eleven sequenced notebooks from problem framing through spectral skill attribution,
  robustness ablation, and conclusions.
- `oisst_fno` package: NOAA ERDDAP acquisition and temporal windowing, seasonal
  climatology / persistence / trend / EOF+ridge / local-CNN baselines, a compact FNO2d,
  and masked forecast and scale-aware metrics.
- Documentation: research positioning, data card, provenance and provenance audit,
  threats to validity, and references.
- Citation metadata (`CITATION.cff`, `.zenodo.json`) and the Zenodo archiving procedure
  in `docs/ZENODO_RELEASE.md`.
- Repository infrastructure: contribution guide carrying the project's scientific rules,
  code of conduct, security policy, issue templates including a "Scientific issue" form,
  pull-request template with a scientific-review checklist, Dependabot, `.editorconfig`,
  and `.gitattributes`.
- `poetry.lock` for reproducible dependency resolution, and a `py.typed` marker.
- CI jobs validating notebook hygiene (no stored outputs) and citation metadata
  (`CITATION.cff` schema, `.zenodo.json`, and version agreement with `pyproject.toml`).

### Changed

- `pyproject.toml` uses PEP 621 `[project]` metadata with classifiers, keywords, and
  project URLs, and an SPDX license expression.
- CI enforces `ruff format --check`, caches the Poetry environment, cancels superseded
  runs, and runs on `actions/checkout@v7` and `actions/setup-python@v7`.
- Notebooks carry the cell ids their declared nbformat 4.5 requires.

### Fixed

- `open_oisst` reordered longitudes on a lazily loaded array, which indexed the wrong
  axis after the singleton `depth` squeeze and raised `IndexError` on xarray's scipy
  backend. Reordering now happens only for regions that actually need it, on
  materialised values.
- Ruff and strict-mypy errors that prevented CI from passing. Types are pinned at the
  numpy and torch stub boundary rather than suppressed.
- `pandas-stubs` added as a dev dependency, and mypy targets Python 3.12 semantics so
  numpy 2.4's PEP 695 stubs parse.

### Results

- No results are published in this release. The notebooks define the study and its
  evaluation design; no metric, figure, or conclusion is claimed here.

[Unreleased]: https://github.com/DiogoRibeiro7/oisst-fourier-neural-operator/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/oisst-fourier-neural-operator/releases/tag/v0.1.0
