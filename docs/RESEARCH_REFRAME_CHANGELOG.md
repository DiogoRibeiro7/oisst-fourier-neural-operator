# Research Reframe Changelog — 2026-08-18

The repository was reframed after an originality/novelty audit.

## Main change

Previous broad question:

> Can an FNO forecast Northeast Atlantic SST better than simple baselines?

Current question:

> **When, where, and at what spatial scales does an FNO add forecast information beyond persistence and simpler global/local baselines?**

## Changes made

- Added explicit literature/novelty positioning.
- Added threats-to-validity documentation.
- Made persistence the primary scientific null model.
- Added train-only seasonal climatology.
- Added reusable EOF+ridge global low-rank baseline with validation selection.
- Added daily paired RMSE contrasts.
- Added moving-block-bootstrap uncertainty for FNO versus persistence.
- Added target-date preservation and prediction artifact export.
- Added a dedicated spectral-skill-attribution notebook.
- Added predefined frequency bands and validation-defined “hard persistence” cases.
- Added mask/taper caveats for Fourier analysis.
- Expanded robustness requirements to seeds, block length, mode count, resolution, and fully oceanic subdomains.
- Updated model/data cards with novelty and validity constraints.
- Strengthened research-grade benchmarking and paper-assessment prompts.
- Added spectral attribution, novelty red-team, and cross-product/cross-region replication prompts.

## New evidence standard

A lower aggregate FNO RMSE is no longer enough.

The preferred evidence chain is:

\[
\text{FNO beats persistence}
\rightarrow
\text{paired uncertainty supports the gain}
\rightarrow
\text{FNO beats EOF+ridge}
\rightarrow
\text{skill is attributable by scale/regime}
\rightarrow
\text{result survives robustness tests}.
\]

Failure at any stage should be reported rather than hidden.
