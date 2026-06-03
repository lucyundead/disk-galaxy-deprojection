# Disk Galaxy Deprojection

This repository builds a baseline-plus-residual probabilistic benchmark for
deprojecting barred-galaxy stellar mass structure from S4G-like images.

Milestone 1 predicts posterior residuals for physical summaries:

- radial stellar mass profile
- radial surface-density profile
- vertical scale-height summary
- bar thickness summary
- bar amplitude summary
- central concentration summary

Start with:

```bash
python -m pytest -q
python scripts/build_synthetic_benchmark.py --config configs/milestone1.synthetic.toml
python scripts/train_summary_residual_mdn.py --data outputs/milestone1_synthetic/residual_table.npz
python scripts/evaluate_summary_residual.py --run-dir outputs/milestone1_synthetic
```

## Synthetic Milestone 1 Benchmark

Run the synthetic benchmark:

```bash
python scripts/build_synthetic_benchmark.py --config configs/milestone1.synthetic.toml
python scripts/train_summary_residual_mdn.py --data outputs/milestone1_synthetic/residual_table.npz --output-dir outputs/milestone1_synthetic
python scripts/evaluate_summary_residual.py --run-dir outputs/milestone1_synthetic
```

The benchmark writes:

- `outputs/milestone1_synthetic/manifest.csv`
- `outputs/milestone1_synthetic/residual_table.npz`
- `outputs/milestone1_synthetic/summary_residual_mdn.pt`
- `outputs/milestone1_synthetic/normalization.npz`
- `outputs/milestone1_synthetic/metrics.json`
