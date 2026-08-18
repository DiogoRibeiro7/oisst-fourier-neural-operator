# Research Positioning and Novelty Boundary

## What this repository is

This repository is an original, notebook-first empirical study of whether a Fourier Neural Operator provides forecast skill beyond persistence for NOAA OISST sea-surface-temperature fields, with particular attention to **where** that skill occurs in space, season, and spatial frequency.

It is also a portfolio project demonstrating scientific data engineering, Fourier analysis, neural operators, baseline construction, leakage-safe validation, uncertainty estimation, and adversarial model evaluation.

## What is already known

The project must not imply that FNO-based ocean forecasting or Atlantic SST forecasting is new.

Relevant prior work includes:

1. **Li et al. — Fourier Neural Operator for Parametric Partial Differential Equations**  
   Foundational FNO method.  
   https://arxiv.org/abs/2010.08895

2. **Chattopadhyay et al. — OceanNet: A principled neural operator-based digital twin for regional oceans**  
   FNO-based regional ocean forecasting in the northwest Atlantic/Gulf Stream setting.  
   https://arxiv.org/abs/2310.00813

3. **Sun et al. — Streamlining Ocean Dynamics Modeling with Fourier Neural Operators: A Multiobjective Hyperparameter and Architecture Optimization Approach**  
   FNO optimization for ocean dynamics and long-horizon forecasting.  
   https://arxiv.org/abs/2404.05768

4. **Nag, Zammit-Mangion, Singh & Cressie — Spatio-temporal modeling and forecasting with Fourier neural operators**  
   2026 work using FNO-based dynamical spatiotemporal forecasting, including Atlantic sea-surface-temperature data and uncertainty quantification.  
   https://arxiv.org/abs/2601.01813

5. **Jahanmard et al. — Principled Fourier Neural Operators for High-Resolution Regional Ocean Modeling**  
   2026 regional-ocean work explicitly studying temporal structure, long-horizon stability, and spectral fidelity.  
   https://doi.org/10.1029/2025JH001131

This is not intended to be an exhaustive literature review. Before any paper submission, repeat a current systematic search.

## Claims this repository must not make

The following claims are unsupported:

- “We introduce Fourier Neural Operators.”
- “We are the first to use FNOs for ocean forecasting.”
- “We are the first to use FNOs for Atlantic SST forecasting.”
- “The model learns the governing ocean equations.”
- “The model learns a universal ocean solution operator.”
- “Resolution transfer is guaranteed because FNOs are mesh independent.”
- “A lower test RMSE proves the FNO architecture is superior.”

## The narrower empirical contribution

The project asks:

\[
\boxed{\text{When, where, and at what spatial scales does FNO beat persistence?}}
\]

The intended contribution is therefore **diagnostic rather than architectural**.

Specific evidence sought:

1. paired forecast skill beyond persistence;
2. uncertainty that respects temporal dependence;
3. comparison with low-rank global linear dynamics (EOF+ridge);
4. comparison with learned local spatial dynamics (CNN, later ConvLSTM/U-Net);
5. scale-resolved error reduction rather than only aggregate RMSE;
6. season/regime dependence;
7. explicit identification of cases where FNO is not justified.

## Why this can still be useful

A model class can be established while the conditions under which it is useful remain poorly understood for a particular data product and forecast design.

The project is strongest when framed as an **evaluation question**:

> Does the FNO's global spectral mixing provide information beyond the temporal persistence and low-rank spatial structure already present in an optimum-interpolated SST product?

That question can produce a valuable negative result.

## Publication novelty threshold

The current repository should be described as a portfolio/reproducibility/benchmark study unless the completed experiments reveal a sufficiently general finding.

A paper claim would need to go beyond “FNO forecasts SST well.” Potentially defensible contributions could include:

- a robust empirical law relating FNO skill to spatial-frequency bands and forecast regimes;
- evidence that FNO gains concentrate in specific scales not captured by persistence/EOF dynamics;
- a general evaluation methodology for separating persistence skill from operator-model skill across multiple SST products or regions;
- a new, justified method motivated by observed failure modes rather than architecture novelty for its own sake.

Whether any of these is novel must be checked against the literature **after** the result exists.

## Public description

Safe description:

> A reproducible study of when Fourier Neural Operators add forecast skill beyond persistence and low-rank linear dynamics for NOAA OISST fields, with spatial- and frequency-resolved diagnostics.

Avoid:

> A novel FNO method for Atlantic SST forecasting.
