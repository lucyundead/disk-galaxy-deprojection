# Milestone 2b Density Residual PCA Evaluation

Date: 2026-06-09

This report evaluates the deterministic PCA residual model trained for the
Milestone 2b 3D density target:

```text
delta_rho_3d = rho_true_cylindrical - rho_baseline_cylindrical
```

Inputs were the fetched local Milestone 2b artifacts under
`outputs/tng50_milestone2b/`. No TNG data were downloaded locally. The compact
figure payload was rendered with the remote Python environment for matplotlib
compatibility, but no scheduler job was needed.

## Artifacts

- Metrics:
  `outputs/tng50_milestone2b/density_residual_pca_report/density_residual_pca_report_metrics.json`
- Compact report:
  `outputs/tng50_milestone2b/density_residual_pca_report/density_residual_pca_report.md`
- Example held-out mass-grid slices:
  `outputs/tng50_milestone2b/density_residual_pca_report/figures/`

The PNG diagnostics tile truth, geometric baseline, PCA-corrected density, and
mean-train-residual correction in that order. Panels show log cylindrical
mass summed over vertical bins in the bar-aligned `R-phi` grid. The baseline
now uses smooth deprojected surface-density deposition and a normalized
`sech^2` vertical profile rather than one pseudo-particle per image pixel.

Each labeled figure has two rows. The top row compares mass maps. The bottom
row shows residual maps, defined as candidate minus truth. A useful correction
should make the PCA-corrected residual panel less structured and lower contrast
than the baseline residual panel. The mean-train-residual panel is the sanity
check: it shows what can be gained by applying an average train-set correction
without using the held-out image or geometry.

## Held-Out Test Summary

The test split contains 117 projections from 13 held-out galaxies. The PCA basis
uses 32 components and captures 92.52 percent of train residual variance.

| Metric | Baseline | Mean train residual | PCA model |
| --- | ---: | ---: | ---: |
| Cell-mass MAE (Msun) | 3.986e6 | 2.685e6 | 1.150e6 |
| Improvement over baseline | 0.00% | 32.66% | 71.14% |
| Total-mass fractional MAE | 0.0453 | 0.0453 | 0.0453 |
| Radial profile MAE (Msun) | 6.936e8 | 8.807e8 | 5.650e8 |
| Vertical profile MAE (Msun) | 5.891e9 | 3.693e9 | 8.337e8 |
| Bar-frame `m=2` MAE | 0.0429 | 0.0387 | 0.0386 |

The unchanged total-mass error is expected: the correction is rescaled to
preserve the baseline grid mass, so this model only redistributes mass within
the coarse cylindrical target.

## Stratification

The PCA model improves over both the geometric baseline and the mean-train
residual sanity baseline in every requested bin.

| Inclination (deg) | Baseline MAE | Mean train residual MAE | PCA model MAE |
| ---: | ---: | ---: | ---: |
| 20 | 4.011e6 | 2.717e6 | 1.218e6 |
| 40 | 3.983e6 | 2.680e6 | 1.083e6 |
| 60 | 3.965e6 | 2.656e6 | 1.149e6 |

| Bar viewing angle (deg) | Baseline MAE | Mean train residual MAE | PCA model MAE |
| ---: | ---: | ---: | ---: |
| 0 | 4.029e6 | 2.713e6 | 1.164e6 |
| 45 | 3.985e6 | 2.681e6 | 1.129e6 |
| 90 | 3.946e6 | 2.660e6 | 1.158e6 |

## Interpretation

The revised smooth baseline removes the artificial inner `R-phi` holes produced
by the earlier point-particle deposition. The mean train residual remains a
useful sanity baseline: it improves cell-mass MAE by 32.66 percent, so part of
the gain is still a systematic correction to the geometric baseline. The
deterministic PCA model improves further to 71.14 percent and wins especially in
vertical profile error, indicating that the image and geometry features are
contributing projection-specific information beyond the average residual.

The weakest scientific limitation remains total mass. Because the current
correction preserves baseline grid mass, it cannot improve the 4.5 percent
held-out total-mass fractional MAE. That is a target-definition choice, not a
training failure.

## Recommendation

Proceed next to a probabilistic residual model over the existing PCA
coefficients, or a small deterministic ensemble as a calibration bridge, before
scaling up the network or moving to the cluster. Add a separate scalar
total-mass correction head only if total stellar mass improvement becomes part
of the Milestone 2b target.
