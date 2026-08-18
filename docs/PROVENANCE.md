# Implementation provenance

## Scope

This repository applies Fourier Neural Operators to NOAA OISST sea-surface-temperature forecasting. The repository-specific formulation, notebook workflow, data preparation, baselines, diagnostics, evaluation design, and development prompts were developed for this project. This does **not** imply that FNO-based ocean or Atlantic SST forecasting is a novel application; see `RESEARCH_POSITIONING.md`.

The **Fourier Neural Operator algorithm is not original to this repository**. It follows:

- Li, Z. et al. *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021.
- Kovachki, N. et al. *Neural Operator: Learning Maps Between Function Spaces*.

The official `neuraloperator/neuraloperator` project is the reference implementation and should be used as an external benchmark when validating this repository.

## Code provenance audit

Before publication, the local Fourier layer was compared against:

1. the official NeuralOperator FNO implementation;
2. the official NeuralOperator "from scratch" FNO tutorial; and
3. the early public FNO reference implementation associated with the original paper.

An earlier draft used the conventional `SpectralConv2d` teaching pattern: two retained Fourier bands, a complex `einsum`, and an inverse real FFT. Although that pattern is mathematically natural, its structure and naming were too close to the public tutorial for this repository's goal of presenting an independently written educational implementation.

The current implementation was therefore rewritten. In particular it:

- uses the project-specific class name `TruncatedFourierMix2d`;
- stores complex kernels as real-valued components with an explicit final real/imaginary axis;
- uses `[out_channels, in_channels, ...]` kernel ordering rather than the common `[in_channels, out_channels, ...]` teaching layout;
- constructs complex kernels with `torch.view_as_complex` at execution time;
- uses a different channel-contraction expression and explicit frequency-band slices;
- validates overlapping vertical Fourier bands;
- uses project-specific `OperatorBlock2d`, encoder/decoder naming, normalization, and coordinate encoding;
- keeps the real-data SST forecasting workflow independent of the PDE examples used in the original FNO repositories.

## Attribution policy

When publishing or discussing this repository:

- describe the **application and experimental workflow** as original project work;
- describe the model as an implementation/application of the published Fourier Neural Operator method;
- cite Li et al. for FNO and NeuralOperator for the official implementation;
- do not describe the Fourier layer itself as a novel architecture;
- do not imply that SST forecasting here learns the governing ocean PDEs.

## License note

This repository uses the MIT license. The official NeuralOperator repository also uses the MIT license. Algorithmic attribution is nevertheless retained because licensing permission and scholarly attribution are separate concerns.
