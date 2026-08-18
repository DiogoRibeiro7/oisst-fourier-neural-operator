# Provenance audit

Date: 2026-08-18

## Objective

Assess whether the repository contains implementation details that are too close in wording, naming, or code structure to public Fourier Neural Operator reference implementations.

This is a provenance/style audit, not a legal opinion and not a plagiarism detector.

## Public references inspected

1. NeuralOperator official repository: `neuraloperator/neuraloperator`.
2. NeuralOperator official FNO theory guide and its pedagogical `SpectralConv2d` implementation.
3. Early public `fourier_2d.py` implementation associated with the original FNO work.
4. Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*.

## Findings

| Area | Assessment before remediation | Reason | Status |
|---|---|---|---|
| Low-level spectral convolution | **Too close stylistically** | Used the conventional `SpectralConv2d` name, two complex weight tensors with the same channel ordering, the same two retained FFT corners, and the same `bixy,ioxy->boxy` contraction used in the official teaching implementation | **Rewritten** |
| Overall FNO data flow | Expected conceptual similarity | Coordinate embedding/lifting → Fourier blocks → projection is part of the published architecture. The local implementation uses different modules and control flow | Retained with attribution |
| Local Fourier block | Low concern | Project version adds GroupNorm and a project-specific block abstraction | Retained |
| Coordinate encoding | Low concern | Generated internally with project-specific normalized coordinates rather than copied external grid-building code | Retained |
| Padding | Low concern | Simple PyTorch right/bottom padding; concept is common and implementation is independent | Retained |
| SST data pipeline | Project-specific | NOAA OISST acquisition, chronological windows, masks, train-only normalization, and forecast horizon are specific to this repository | Retained |
| Training loop | Low concern | Standard PyTorch optimizer/scheduler/checkpoint workflow; no distinctive reference-code structure found | Retained |
| Evaluation and baselines | Project-specific | Persistence, climatology, SST metrics, robustness, and ablation workflow were designed for this forecasting problem | Retained |
| Notebook narrative | Project-specific | Organized around observational SST forecasting rather than the Darcy/Navier–Stokes examples in the reference code | Retained |

## Remediation applied

The original local `SpectralConv2d` draft was replaced by `TruncatedFourierMix2d`.

The rewritten implementation differs structurally in several ways:

- complex parameters are stored as real/imaginary components and converted with `torch.view_as_complex`;
- kernel layout is `[band, out_channels, in_channels, modes_y, modes_x, real_imag]`;
- channel contraction uses `bihw,oihw->bohw`;
- retained frequency regions are represented with explicit slices;
- vertical-mode overlap is validated explicitly;
- the block is named `OperatorBlock2d` with `global_path`, `local_path`, and normalization;
- the top-level FNO uses project-specific encoder/decoder terminology and coordinate generation;
- source documentation points readers to this audit and clearly attributes the FNO algorithm.

## What remains similar by necessity

A correct truncated 2-D FNO implementation will still contain the mathematical sequence

1. real FFT;
2. selection of low Fourier modes;
3. learned complex mixing across channels;
4. inverse real FFT;
5. a local linear path and non-linearity in an FNO block.

Similarity at this level is algorithmic rather than evidence of copied source code.

## Publication language

Recommended:

> This repository presents an original applied data-science workflow for sea-surface-temperature forecasting using a compact, independently structured PyTorch implementation of the published Fourier Neural Operator architecture.

Avoid:

> We developed a new Fourier Neural Operator architecture.

Also avoid implying that the model learns the governing ocean PDEs. It is trained as a supervised spatiotemporal field forecaster.

## Attribution

Cite at minimum:

- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. *Fourier Neural Operator for Parametric Partial Differential Equations*. ICLR 2021.
- NeuralOperator official implementation: https://github.com/neuraloperator/neuraloperator

The repository's own MIT license applies to its project-specific code. The NeuralOperator repository is also MIT-licensed, but this project does not rely on license compatibility as a substitute for attribution.

## Originality versus research novelty

This audit now distinguishes two separate questions:

1. **Was the repository/source code independently produced?**  
   Current assessment: yes, after remediation of the original low-level spectral-layer draft.

2. **Is the scientific application novel?**  
   Not by application alone. FNO-based ocean forecasting already exists, and current literature includes Atlantic SST forecasting with FNOs. The repository therefore does not claim novelty from `FNO + Atlantic SST`.

The updated empirical contribution is the evaluation design around persistence, EOF+ridge, temporally valid paired uncertainty, and spatial-frequency attribution. Whether that becomes a research-novel contribution depends on the completed results and a fresh literature comparison; code originality does not establish paper novelty.

See `docs/RESEARCH_POSITIONING.md` for the current novelty boundary.
