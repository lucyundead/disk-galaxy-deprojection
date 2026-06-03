# Milestone 1 Synthetic Summary Benchmark

## Scope

This benchmark validates the baseline-plus-residual workflow on synthetic barred
galaxies before using TNG50 particle data.

## Artifacts

- Manifest: `outputs/milestone1_synthetic/manifest.csv`
- Residual table: `outputs/milestone1_synthetic/residual_table.npz`
- Model checkpoint: `outputs/milestone1_synthetic/summary_residual_mdn.pt`
- Metrics: `outputs/milestone1_synthetic/metrics.json`

## Default Run Metrics

- Held-out test projections: 12
- Baseline MAE: 589892416.0
- Corrected MAE: 425393184.0
- 68 percent interval coverage: 0.7458333333333333

## Required Checks

- Train, validation, and test splits are by galaxy ID.
- Baseline summaries and true summaries share identical names.
- Posterior samples produce finite corrected summaries.
- `coverage_68` is between 0 and 1.
- `baseline_mae` and `corrected_mae` are reported.

## Interpretation

This synthetic run is a software and methodology check. It is not evidence that
the method works on TNG50 or S4G. The next scientific step is to replace the
synthetic particle generator with local TNG50 stellar particle files and repeat
the same evaluation protocol.
