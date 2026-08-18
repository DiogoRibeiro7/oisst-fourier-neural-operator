# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because this is a research repository, the changelog also records changes that affect
**reported results**, not only code. Any release that changes a published metric,
figure, or conclusion says so explicitly under "Results".

## [Unreleased]

### Added

- Repository infrastructure: contribution guide, code of conduct, security policy,
  issue and pull-request templates, Dependabot, `.editorconfig`, and `.gitattributes`.
- `poetry.lock` for reproducible dependency resolution.
- CI jobs validating notebook hygiene (no stored outputs) and citation metadata
  (`CITATION.cff`, `.zenodo.json`, and version agreement with `pyproject.toml`).
- `py.typed` marker so downstream users get the package's type information.

### Changed

- `pyproject.toml` migrated to PEP 621 `[project]` metadata with classifiers, keywords,
  and project URLs.
- CI now also enforces `ruff format --check`, caches the Poetry environment, and cancels
  superseded runs.

### Fixed

- Ruff and strict-mypy errors that prevented CI from passing. Types are pinned at the
  numpy/torch stub boundary rather than suppressed.

### Results

- No change to reported results. All changes are infrastructure or typing.

## [0.1.0] - 2026-08-18

### Added

- Initial public release: eleven sequenced notebooks from problem framing through
  spectral skill attribution, robustness ablation, and conclusions.
- `oisst_fno` package: NOAA ERDDAP acquisition and temporal windowing, seasonal
  climatology / persistence / trend / EOF+ridge / local-CNN baselines, a compact FNO2d,
  and masked forecast and scale-aware metrics.
- Documentation: research positioning, data card, provenance and provenance audit,
  threats to validity, and references.
- Citation metadata (`CITATION.cff`, `.zenodo.json`) and the Zenodo archiving procedure.

[Unreleased]: https://github.com/DiogoRibeiro7/oisst-fourier-neural-operator/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/oisst-fourier-neural-operator/releases/tag/v0.1.0
