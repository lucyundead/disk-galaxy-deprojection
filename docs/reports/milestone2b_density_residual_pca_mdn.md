# Milestone 2b Probabilistic PCA Residual Evaluation

Date: 2026-06-09

This report evaluates the first probabilistic residual model over the existing
32-dimensional PCA coefficients for the Milestone 2b 3D density residual target.
The model is a compact mixture density network trained on the same pooled image
features, geometry metadata, and mass features used by the deterministic PCA
residual MLP.

No TNG data were downloaded locally. The deterministic report figures were
rendered with the remote Python environment for matplotlib compatibility, but no
scheduler job was needed.

## Artifacts

- Script: `scripts/train_density_residual_pca_mdn.py`
- Metrics:
  `outputs/tng50_milestone2b/density_residual_pca_mdn/density_residual_pca_mdn_metrics.json`
- Checkpoint:
  `outputs/tng50_milestone2b/density_residual_pca_mdn/density_residual_pca_mdn.pt`
- Posterior samples and reconstructed posterior mean:
  `outputs/tng50_milestone2b/density_residual_pca_mdn/density_residual_pca_mdn_predictions.npz`

## Run Configuration

```bash
.venv/bin/python scripts/train_density_residual_pca_mdn.py \
  --pca outputs/tng50_milestone2b/density_residual_diagnostics/density_residual_pca.npz \
  --density-table outputs/tng50_milestone2b/density_residual_table.npz \
  --output-dir outputs/tng50_milestone2b/density_residual_pca_mdn \
  --epochs 250 \
  --hidden-dim 128 \
  --image-feature-size 24 \
  --batch-size 64 \
  --n-components 3 \
  --n-samples 256 \
  --weight-decay 0.0001 \
  --patience 50 \
  --device cpu
```

Best validation negative log likelihood occurred at epoch 4.

## Held-Out Metrics

The test split contains 117 projections from 13 held-out galaxies.

| Metric | Geometric baseline | Deterministic PCA MLP | Probabilistic PCA MDN |
| --- | ---: | ---: | ---: |
| Cell-mass MAE (Msun) | 3.986e6 | 1.150e6 | 1.099e6 |
| Improvement over baseline | 0.00% | 71.14% | 72.44% |
| Total-mass fractional MAE | 0.0453 | 0.0453 | 0.0453 |
| PCA coefficient RMSE | 0.001132 mean-coeff baseline | 0.000734 | 0.000740 |
| PCA coefficient 68% coverage | not applicable | not applicable | 0.752 |

After revising the baseline to use smooth surface-density deposition and a
normalized `sech^2` vertical profile, the MDN posterior mean is slightly better
than the deterministic MLP in cell-mass MAE. Its held-out coefficient coverage
is wider than nominal at 0.752, so the uncertainty branch is useful but may be a
little conservative.

## Interpretation

This is now both a useful calibration bridge and a competitive point predictor.
The deterministic MLP remains the simpler point-prediction baseline, but the MDN
provides uncertainty intervals over the compressed 3D residual coordinates and
slightly improves posterior-mean mass error on held-out galaxies.

The unchanged total-mass fractional MAE is expected. The reconstructed correction
preserves the baseline grid mass, so this model redistributes stellar mass
inside the cylindrical grid rather than correcting total mass.

## Recommendation

Keep the MDN as the uncertainty baseline and tune calibration before scaling up:

- rerun with a few seeds to check whether the 0.752 coverage is stable;
- compare 1, 3, and 5 mixture components for coverage and posterior-mean MAE;
- consider temperature or variance calibration only after seed sensitivity is
  known;
- keep the deterministic MLP as the point-prediction baseline.

Do not move to a larger full-3D generator yet. The next useful step is a small
seed-and-mixture sweep over PCA-coefficient uncertainty, still using the fetched
Milestone 2b outputs.
