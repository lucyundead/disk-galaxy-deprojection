# Milestone 2b Probabilistic PCA Residual MDN Sweep

Date: 2026-06-09

## Goal

Run a small local calibration sweep for the probabilistic PCA residual model using
the fetched Milestone 2b clean 3D artifacts. This sweep tests whether the MDN
posterior uncertainty is mainly conservatively scaled, or whether it is unstable
across random seeds and mixture counts.

No TNG data were downloaded locally, and no remote cluster job was submitted.

## Inputs

- PCA payload:
  `outputs/tng50_milestone2b/density_residual_diagnostics/density_residual_pca.npz`
- Density residual table:
  `outputs/tng50_milestone2b/density_residual_table.npz`
- Sweep output directory:
  `outputs/tng50_milestone2b/density_residual_pca_mdn_sweep/`

Command:

```bash
.venv/bin/python scripts/sweep_density_residual_pca_mdn.py \
  --pca outputs/tng50_milestone2b/density_residual_diagnostics/density_residual_pca.npz \
  --density-table outputs/tng50_milestone2b/density_residual_table.npz \
  --output-dir outputs/tng50_milestone2b/density_residual_pca_mdn_sweep \
  --seeds 20260608 20260609 20260610 \
  --n-components 1 3 5 \
  --epochs 200 \
  --hidden-dim 128 \
  --image-feature-size 24 \
  --batch-size 64 \
  --n-samples 128 \
  --patience 50 \
  --device cpu
```

The sweep fits a simple post-processing temperature scale on validation-set PCA
coefficient coverage. The scale multiplies posterior sample deviations around
their posterior mean, then evaluates the resulting 68 percent interval coverage
on the held-out test galaxies.

## Aggregate Results

Nine runs were completed: 3 random seeds crossed with 1, 3, and 5 mixture
components.

| group | posterior-mean cell MAE (Msun) | raw coeff. coverage | temp scale | temp-scaled test coverage |
|---|---:|---:|---:|---:|
| 1 component, mean | 1.067e6 | 0.741 | 0.817 | 0.665 |
| 3 components, mean | 1.121e6 | 0.768 | 0.800 | 0.679 |
| 5 components, mean | 1.129e6 | 0.786 | 0.767 | 0.680 |
| all runs, mean | 1.106e6 | 0.765 | 0.794 | 0.675 |

Across all runs:

- posterior-mean cell-mass MAE range: `1.042e6` to `1.187e6 Msun`;
- raw 68 percent PCA coefficient coverage range: `0.703` to `0.807`;
- validation-fitted temperature scale range: `0.75` to `0.90`;
- temperature-scaled test coverage range: `0.658` to `0.694`.

Best posterior-mean MAE:

- run: `components_1_seed_20260610`
- posterior-mean cell-mass MAE: `1.042e6 Msun`
- raw coefficient coverage: `0.768`
- temperature scale from validation: `0.75`
- temperature-scaled test coverage: `0.658`
- best validation NLL epoch: `4`

## Geometry-Stratified Coverage

After validation-fitted temperature scaling, mean test coverage by inclination:

| inclination | mean coverage |
|---:|---:|
| 20 deg | 0.682 |
| 40 deg | 0.694 |
| 60 deg | 0.648 |

Mean test coverage by bar viewing angle:

| bar angle | mean coverage |
|---:|---:|
| 0 deg | 0.681 |
| 45 deg | 0.668 |
| 90 deg | 0.675 |

The bar-angle stratification is acceptably close to the 0.68 target for this
small benchmark. The 60 degree inclination slice is the main remaining concern:
the same global validation temperature that fixes aggregate coverage leaves
moderately inclined projections undercovered.

## Inclination-Aware Temperature Check

The sweep aggregation was rerun with `--skip-existing`, reusing the same nine
trained MDN runs and adding one validation-fitted temperature scale per
inclination bin. This does not change posterior means or cell-mass MAE; it only
rescales posterior samples around their posterior mean before computing
coefficient coverage.

Aggregate test coverage improved slightly:

| calibration | mean test coverage | run-to-run std |
|---|---:|---:|
| global validation temperature | 0.675 | 0.013 |
| inclination-aware validation temperature | 0.678 | 0.012 |

Mean test coverage by inclination:

| inclination | global temp coverage | inclination-aware coverage | mean inclination scale |
|---:|---:|---:|---:|
| 20 deg | 0.682 | 0.665 | 0.767 |
| 40 deg | 0.694 | 0.672 | 0.756 |
| 60 deg | 0.648 | 0.697 | 0.894 |

This confirms that the MDN uncertainty is geometry-dependent. The 60 degree bin
needs wider intervals than the lower-inclination bins after validation
calibration. However, the three-bin scalar correction is not a perfect solution:
it fixes the 60 degree undercoverage but pushes the 20 and 40 degree bins a bit
low on test coverage.

For the recommended 1-component MDN family, the inclination-aware check is more
balanced:

| inclination | 1-component inclination-aware coverage | mean scale |
|---:|---:|---:|
| 20 deg | 0.669 | 0.833 |
| 40 deg | 0.677 | 0.800 |
| 60 deg | 0.687 | 0.900 |

## Recommendation

Temperature calibration is enough for the current aggregate coefficient-level 68
percent coverage target, and the new inclination-aware check supports the
scientific expectation that uncertainty depends on viewing geometry. A single
global temperature is too blunt for inclination-stratified claims because it
leaves 60 degree projections undercovered. Separate inclination temperatures fix
that slice, especially for the 1-component model family.

The uncertainty model is not too unstable to use as a Milestone 2b probabilistic
baseline, but it should remain provisional. The current evidence supports
reporting both global and inclination-aware calibration diagnostics; it does not
yet support claiming a fully calibrated posterior for every geometry slice.

For the current model family, the 1-component MDN is the best default. It has the
lowest posterior-mean cell-mass MAE and the best validation NLL in this sweep,
while additional mixture components mainly increase conservative raw coverage
without improving the posterior mean. The practical baseline should therefore be
the 1-component MDN with inclination-aware temperature diagnostics carried
forward into the next Milestone 2b evaluation.
