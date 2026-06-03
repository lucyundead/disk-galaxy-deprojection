# Baseline-Residual Deprojection Design

## Purpose

Build a simulation-calibrated, probabilistic deprojection workflow for local
barred galaxies using TNG50 as the first validation arena. The method should
infer stellar mass structure from S4G-like projected images by combining a
classical deprojection baseline with a learned residual posterior.

The first milestone focuses on physical summaries. A later milestone extends the
same residual framework to a coarse 3D stellar-mass density grid.

## Approved Direction

Use a sequential residual ladder:

1. Build a classical deprojection baseline from the mock image and known
   geometry.
2. Measure residuals between baseline-derived summaries and true TNG50 summaries.
3. Train a probabilistic model for those residuals.
4. Validate posterior calibration on held-out TNG50 galaxies.
5. Extend to coarse 3D density residuals only after summary residuals work.

This design treats deprojection as uncertain inference, not a unique inverse
mapping.

## Scientific Scope

Primary target:

- Stellar mass distribution, not total mass or dark matter.

Initial image domain:

- S4G-like 3.6 micron or stellar surface-density images generated from TNG50.
- Inclinations from face-on to moderately inclined, initially `i < 60 deg`.

Initial learned target:

- Residual correction to physical summaries derived from the baseline.

Second learned target:

- Residual correction to a coarse 3D stellar-mass density representation.

## Milestone 1: Summary Residual Benchmark

Milestone 1 should produce a complete, reproducible benchmark with these parts:

1. TNG50 barred-galaxy manifest.
2. Coordinate and orientation convention.
3. S4G-like mock image generation.
4. Classical baseline deprojection.
5. True physical summary extraction.
6. Baseline summary extraction.
7. Probabilistic residual model.
8. Held-out-galaxy evaluation.
9. Posterior calibration and reprojection checks.

The core inference target is:

```text
p(delta_summaries | image_mock, baseline_summaries, inclination, PA, bar_angle, metadata)
```

The final corrected estimate is:

```text
summary_corrected = summary_baseline + delta_summary_sample
```

## Milestone 1 Summary Targets

Use a compact target vector:

- radial enclosed stellar mass profile,
- radial stellar surface-density profile,
- disk vertical scale-height summary,
- bar thickness summary,
- bar strength or bar amplitude summary,
- central stellar mass concentration summary.

The exact bin count can be chosen during implementation, but bins must be coarse
enough for TNG50 resolution and stable under small changes in binning.

## Baseline Model

The baseline should be deliberately simple and interpretable:

- deproject the 2D image using known inclination and PA,
- align to the disk plane and bar frame,
- assume a simple vertical profile such as exponential or sech-squared,
- estimate baseline summaries from this reconstructed structure,
- keep all baseline assumptions recorded in metadata.

The baseline is not a strawman. It is the reference method that the residual
model must improve or calibrate.

## Residual Model

Start with a small probabilistic model rather than a high-dimensional generator.
Acceptable first models:

- conditional normalizing flow,
- mixture density network,
- conditional VAE.

Inputs:

- S4G-like image,
- baseline summaries,
- inclination,
- disk PA,
- bar angle,
- stellar mass or normalization metadata,
- image quality metadata such as noise level and mask fraction.

Outputs:

- posterior samples of summary residuals,
- posterior mean and credible intervals,
- optional covariance or correlation diagnostics.

## Milestone 2: Coarse 3D Residual Benchmark

Milestone 2 begins only after Milestone 1 has demonstrated calibrated residual
posteriors on held-out galaxies.

The core inference target becomes:

```text
p(delta_rho_3d | image_mock, rho_baseline_3d, geometry, metadata)
```

The corrected density sample is:

```text
rho_corrected_sample = rho_baseline_3d + delta_rho_3d_sample
```

The coarse grid should use a physically meaningful coordinate system, preferably
cylindrical or bar-aligned coordinates, and must preserve mass within a stated
tolerance.

Candidate models:

- baseline-plus-residual conditional diffusion,
- baseline-plus-residual flow matching,
- lower-dimensional latent model decoded into a coarse density grid.

## Data Flow

1. Select z~0 TNG50 barred galaxies.
2. Convert stellar particles into a true 3D stellar-mass representation.
3. Extract true physical summaries.
4. Project each galaxy into multiple S4G-like mock images.
5. Run baseline deprojection on each mock image.
6. Extract baseline summaries and, later, baseline coarse 3D density.
7. Train residual model on train galaxies.
8. Tune on validation galaxies.
9. Report final metrics on test galaxies never used for training or tuning.

Train, validation, and test splits must be by galaxy before generating multiple
projections.

## Evaluation

Milestone 1 metrics:

- radial mass profile error,
- surface-density profile error,
- vertical scale-height error,
- bar thickness error,
- central concentration error,
- posterior coverage,
- posterior sharpness,
- reprojection consistency.

Milestone 2 metrics:

- voxel or grid-cell mass error at resolved scales,
- enclosed mass profile error,
- vertical profile error,
- total mass conservation,
- reprojection chi-square,
- posterior calibration for derived summaries.

All results should be stratified by inclination, bar viewing angle, bar strength,
stellar mass, image noise, and morphology.

## Testing Strategy

Use verification checks before model complexity grows:

- synthetic toy galaxy projection and deprojection sanity check,
- mass conservation from particles to grids,
- coordinate alignment visual check,
- train/test leakage check by subhalo ID,
- baseline output reproducibility with fixed config,
- residual target calculation test,
- posterior coverage test on held-out galaxies,
- reprojection test for posterior samples.

## Boundaries

Do not include real S4G inference in Milestone 1. Real S4G transfer is a later
stage after the simulation benchmark is credible.

Do not claim recovery of unique 3D structure. Report posterior distributions and
failure cases.

Do not begin with a full 3D generator. The summary residual benchmark is the
gate that decides whether the richer 3D model is worth building.

## Expected Deliverables

Milestone 1 deliverables:

- reproducible TNG50 barred-galaxy manifest,
- mock image generation script or notebook plus config,
- baseline deprojection implementation,
- summary target extraction,
- residual model training pipeline,
- held-out benchmark report with calibration plots and failure cases.

Milestone 2 deliverables:

- coarse 3D target representation,
- baseline 3D reconstruction,
- probabilistic 3D residual model,
- 3D and reprojection benchmark report.
