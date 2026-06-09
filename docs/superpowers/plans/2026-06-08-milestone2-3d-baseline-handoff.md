# Milestone 2 3D Baseline Handoff

> **For the next thread:** Start here before editing code. This handoff captures
> the current Milestone 2 state after TNG50 ingestion, clean Milestone 2b
> regeneration, matching baseline 3D density-grid construction, the first
> `delta_rho_3d` residual dataset, PCA compression, and the first trained
> deterministic residual model.

**Date:** 2026-06-08

**Current branch:** `codex/milestone2-tng-ingestion-design`

**Latest relevant commit:** `06d6e60 Add Milestone 2 3D baseline handoff`

**Workspace note:** the current workspace contains uncommitted Milestone 2b
implementation changes and fetched output artifacts. Check `git status` before
editing.

**Workspace:** `/home/lucyundead/projects/disk-galaxy-deprojection`

**Remote project root:** `/home/zli/disk-galaxy-deprojection`

**Cluster wrapper:** `/home/lucyundead/codex/hpc-agent/hpc`

---

## Project Direction

The project is following the residual ladder:

1. Classical deprojection baseline.
2. Residual posterior on physical summary quantities.
3. Coarse 3D residual around a baseline 3D density field.

Do not jump directly to a large end-to-end 3D generator. The Milestone 2b target
now exists as a deterministic first pass:

```text
delta_rho_3d = rho_true_cylindrical - rho_baseline_cylindrical
```

where both fields use the same disk/bar-aligned cylindrical grid.

The immediate next scientific target is not a larger model. First produce a
held-out evaluation report for the trained PCA residual model and compare it
against simple sanity baselines.

---

## Completed State

### Sample

The current TNG50 sample is the visually approved stricter barred sample:

- TNG50-1 snapshot `99`
- central subhalos only
- `stellar_mass_msun >= 10^9.5`
- `Barred == True`
- `BarStrength[0] >= 0.2`
- `BarSize[0] >= 2.0 kpc`
- top 64 by stellar mass
- train/validation/test split by galaxy: `38/13/13`

Key local files:

- `outputs/tng50_milestone2/barred_sample_m9p5_rbar2_64.csv`
- `outputs/tng50_milestone2/manifest.csv`

The projection manifest has 576 rows:

- 64 galaxies
- 9 projections per galaxy
- inclinations: `20`, `40`, `60` degrees
- face-on bar viewing angles: `0`, `45`, `90` degrees
- disk PA: `0` degrees

### Summary Residual Benchmark

Remote benchmark was run through:

```bash
python scripts/cluster_dgdp.py run-tng50
python scripts/cluster_dgdp.py fetch-tng50
```

Current benchmark artifacts:

- `outputs/tng50_milestone2/residual_table.npz`
- `outputs/tng50_milestone2/summary_residual_mdn.pt`
- `outputs/tng50_milestone2/normalization.npz`
- `outputs/tng50_milestone2/metrics.json`
- `outputs/tng50_milestone2/diagnostics/`

Projection-grid metrics:

- `baseline_mae`: `188152672.0`
- `corrected_mae`: `69017608.0`
- `coverage_68`: `0.7004273504273504`
- `n_test`: `117`

Diagnostic rerun with 256 posterior samples:

- `baseline_mae`: `188152672.0`
- `corrected_mae`: `66601940.0`
- `coverage_68`: `0.7568376068376068`

Interpretation: the residual model reduces aggregate MAE, especially baseline
bias with inclination, but summary-level results are not uniformly robust.
Weak spots include the inner `2 kpc` enclosed-mass target, vertical scale
height, outer low-density annuli in fractional error, and central mass fraction.

### True Cylindrical 3D Density Products

Implemented in:

- `src/dgdp/density3d.py`
- `scripts/build_tng50_density_grid.py`
- `scripts/cluster_dgdp.py` subcommand `run-tng50-density`

Run commands:

```bash
python scripts/cluster_dgdp.py sync
python scripts/cluster_dgdp.py run-tng50-density
python scripts/cluster_dgdp.py fetch-tng50
```

Local fetched outputs:

- `outputs/tng50_milestone2/density_grids_logr_cyl/`
- `outputs/tng50_milestone2/density_grids_logr_cyl/density_grid_catalog.csv`
- `outputs/tng50_milestone2/density_grids_logr_cyl/density_grid_diagnostics.json`
- 64 per-galaxy HDF5 files named
  `subhalo_<id>_density_cylindrical.hdf5`

Grid definition:

- coordinate frame: disk plane aligned to `z = 0`, bar major axis aligned to
  intrinsic `x`
- shape: `(n_R, n_phi, n_z) = (32, 48, 32)`
- radial edges: explicit `R = 0`, then logarithmic from `0.05` to `30.0 kpc`
- azimuth edges: linear from `-pi` to `pi`
- vertical edges: linear from `-10.0` to `10.0 kpc`
- density unit: `Msun/kpc^3`
- mass unit: `Msun`
- orientation source: compact 80000-particle files
- density source: all formed stellar particles streamed directly from TNG50
  snapshot chunks

Completed run diagnostics:

- density grids written: `64`
- local fetched size: about `31 MB`
- median mass fraction inside grid: `0.8812214944172503`
- minimum mass fraction inside grid: `0.5249403366441652`
- maximum mass fraction inside grid: `0.9745345221702302`

Caveat: the current grid is an inner bar/disk target. It conserves mass inside
the stated cylindrical volume, but it intentionally does not capture all stellar
halo or far-outer disk mass. The lowest mass fractions are in the most massive,
extended systems.

### Clean Milestone 2b Dataset

The earlier 96x96 mock images with PSF/noise were judged too coarse and too
strongly blurred/noisy for this first 3D residual step. The current clean
Milestone 2b config is:

- `configs/milestone2b.clean3d.toml`
- image size: `192 x 192`
- pixel scale: `0.35 kpc/pixel`
- field of view: `67.2 kpc`
- `psf_sigma_pixels = 0.0`
- `noise_sigma_fraction = 0.0`

The mock images were regenerated from the same all-formed-star TNG chunk source
used for the true 3D density grids. This fixed the earlier source mismatch where
compact sampled images were being compared to all-particle true grids.

Cluster commands used:

```bash
python scripts/cluster_dgdp.py sync
python scripts/cluster_dgdp.py run-tng50-milestone2b-clean
python scripts/cluster_dgdp.py fetch-tng50-milestone2b
```

Key local artifacts:

- `outputs/tng50_milestone2b/residual_table.npz`
- `outputs/tng50_milestone2b/all_particle_image_catalog.csv`
- `outputs/tng50_milestone2b/all_particle_image_diagnostics.json`
- `outputs/tng50_milestone2b/density_grids_logr_cyl/`

Verified artifact facts:

- images: `(576, 192, 192)`, `float32`
- source particles: `tng_chunks_all_formed_stars`
- PSF/noise: none
- image mass range: about `7.08e10` to `1.14e12 Msun`

### Baseline Density And Residual Table

Implemented in:

- `src/dgdp/density3d.py`
- `scripts/build_tng50_all_particle_images.py`
- `scripts/build_tng50_baseline_density_grid.py`
- `scripts/build_tng50_density_residual_table.py`
- `scripts/analyze_tng50_density_residuals.py`

Current local artifacts:

- `outputs/tng50_milestone2b/baseline_density_grids_logr_cyl/`
- `outputs/tng50_milestone2b/baseline_density_grids_logr_cyl/baseline_density_grid_diagnostics.json`
- `outputs/tng50_milestone2b/density_residual_table.npz`
- `outputs/tng50_milestone2b/density_residual_table_diagnostics.json`
- `outputs/tng50_milestone2b/density_residual_diagnostics/density_residual_diagnostics.json`
- `outputs/tng50_milestone2b/density_residual_diagnostics/density_residual_pca.npz`

The residual table contains:

- `images`: `(576, 192, 192)`, `float32`
- `truth_density`: `(576, 32, 48, 32)`, `float32`
- `baseline_density`: `(576, 32, 48, 32)`, `float32`
- `delta_density`: `(576, 32, 48, 32)`, `float32`

Diagnostics:

- baseline median mass fraction inside grid: `0.9021112253471084`
- baseline median reprojection L1 fraction: `0.4320973845405449`
- residual table median absolute delta density: `542835.0`
- baseline total-mass fractional MAE: `0.07923948764801025`
- PCA components: `32`
- PCA cumulative explained variance: `0.9024961590766907`
- PCA reconstruction relative RMSE: `0.3420652449131012`

Scientific caution: the baseline is a useful geometric comparison, not a unique
physical deprojection. Its reprojection mismatch is still substantial, and the
PCA target is a compressed residual representation, not the full posterior over
all plausible 3D stellar structures.

### First Deterministic PCA Residual Model

Implemented in:

- `scripts/train_density_residual_pca.py`

Training command used locally:

```bash
.venv/bin/python scripts/train_density_residual_pca.py \
  --pca outputs/tng50_milestone2b/density_residual_diagnostics/density_residual_pca.npz \
  --density-table outputs/tng50_milestone2b/density_residual_table.npz \
  --output-dir outputs/tng50_milestone2b/density_residual_pca_model \
  --epochs 500 \
  --hidden-dim 128 \
  --image-feature-size 24 \
  --batch-size 64 \
  --weight-decay 0.0001 \
  --patience 60 \
  --device auto
```

Local artifacts:

- `outputs/tng50_milestone2b/density_residual_pca_model/density_residual_pca_mlp.pt`
- `outputs/tng50_milestone2b/density_residual_pca_model/density_residual_pca_metrics.json`
- `outputs/tng50_milestone2b/density_residual_pca_model/density_residual_pca_normalization.npz`
- `outputs/tng50_milestone2b/density_residual_pca_model/density_residual_pca_predictions.npz`

Model summary:

- deterministic MLP on pooled log-image features plus geometry/mass metadata
- target: 32 PCA coefficients for `delta_mass / truth_grid_mass`
- train/validation/test split remains by galaxy
- trained on CPU; V100s were not needed for this compressed first model
- best epoch: `28`
- best validation MSE: `1.0564786195755005`

Held-out test metrics:

- baseline cell-mass MAE: `6247315.5 Msun`
- corrected cell-mass MAE: `3382101.25 Msun`
- relative cell-mass MAE improvement: about `46%`
- baseline total-mass fractional MAE: `0.08416559547185898`
- corrected total-mass fractional MAE: `0.08416559547185898`
- mean-coefficient RMSE: `0.008716241456568241`
- model-coefficient RMSE: `0.008345530368387699`

Important interpretation: the corrected and baseline total-mass fractional MAE
are identical by construction. The trainer preserves the baseline total mass
after applying the residual correction, so the model is only learning spatial
redistribution of mass. Add a separate scalar mass-correction head only if total
stellar mass improvement becomes a target.

---

## 2026-06-09 Update: Smooth Baseline Repair And PCA/MDN Evaluation

The held-out diagnostic figures exposed an important baseline artifact: the
left side of the baseline `R-phi` panel could appear black because the previous
baseline deposited one pseudo-particle per image pixel into a finer cylindrical
grid. For subhalo `96762`, projection `0`, this created empty inner `R-phi`
cells even though the input image had nonzero mass.

This was a data-construction artifact, not a matplotlib issue. The baseline was
revised in `scripts/build_tng50_baseline_density_grid.py` to:

1. deproject the mock image into disk-plane surface density
   `Sigma(R, phi)`;
2. integrate that surface density over cylindrical `R-phi` cell area;
3. distribute the resulting surface mass vertically with a normalized finite-bin
   `sech^2(z / h)` profile.

In formula form, the baseline now uses:

```text
rho(R, phi, z) = Sigma(R, phi) * sech^2(z / h) / integral sech^2(z / h) dz
```

where the integral is evaluated over the vertical grid edges. The default
vertical scale height remains `0.4 kpc`.

Tests added or updated:

- `test_sech2_vertical_bin_weights_are_symmetric_and_normalized`
- `test_smooth_baseline_grid_does_not_leave_inner_phi_holes_for_smooth_image`

After the repair, the problematic held-out example no longer has zero-mass
baseline holes. The refreshed labeled PNG is:

- `outputs/tng50_milestone2b/density_residual_pca_report/figures/test_example_subhalo_96762_projection_0.png`

Regenerated local Milestone 2b artifacts:

- `outputs/tng50_milestone2b/baseline_density_grids_logr_cyl/`
- `outputs/tng50_milestone2b/density_residual_table.npz`
- `outputs/tng50_milestone2b/density_residual_diagnostics/`
- `outputs/tng50_milestone2b/density_residual_pca_model/`
- `outputs/tng50_milestone2b/density_residual_pca_report/`
- `outputs/tng50_milestone2b/density_residual_pca_mdn/`

No TNG data were downloaded locally. The compact figure payload was rendered
with the remote Python environment for matplotlib compatibility, but no
scheduler job was submitted.

Updated deterministic PCA report:

- `docs/reports/milestone2b_density_residual_pca.md`
- test baseline cell-mass MAE: `3.986e6 Msun`
- mean-train-residual cell-mass MAE: `2.685e6 Msun`
- deterministic PCA corrected cell-mass MAE: `1.150e6 Msun`
- deterministic PCA improvement over baseline: `71.14%`
- baseline total-mass fractional MAE: `0.0453`
- PCA cumulative explained variance: `92.52%`

Updated probabilistic PCA MDN report:

- `docs/reports/milestone2b_density_residual_pca_mdn.md`
- MDN posterior-mean cell-mass MAE: `1.099e6 Msun`
- MDN posterior-mean improvement over baseline: `72.44%`
- MDN PCA coefficient 68 percent coverage: `0.752`

Interpretation:

- The black/empty baseline panel was fixed by using surface-density deposition
  plus the normalized `sech^2` vertical structure.
- The mean-train-residual baseline still improves substantially, so part of the
  residual is a systematic correction to the geometric baseline.
- Both deterministic PCA and MDN use image/geometry features to improve beyond
  that average residual.
- Total mass still does not improve because the current correction preserves
  baseline grid mass by construction.
- The old baseline reprojection L1 diagnostic is less physically meaningful
  after this change because it reprojects coarse cylindrical cell centers rather
  than the smooth deprojected surface-density field.

Fresh verification after the repair:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest
```

Result:

- `ruff`: all checks passed
- `pytest`: `87 passed in 30.94s`

---

## What Not To Redo

- Do not rebuild the barred sample unless the user explicitly asks.
- Do not regenerate face-on galleries unless changing sample cuts.
- Do not rerun `run-tng50` unless changing the projection grid, sample, or
  summary benchmark logic.
- Do not regenerate Milestone 2b unless changing image size, pixel scale,
  PSF/noise, or the all-particle source contract.
- Do not download TNG data locally. TNG data should remain on the cluster.
- Do not add delete or cleanup behavior to the cluster wrapper.
- Do not scale up to a larger GPU model before reviewing held-out scientific
  diagnostics for the current deterministic PCA model.

---

## Next Thread Goal

Continue from the repaired smooth `sech^2` baseline and completed deterministic
plus MDN PCA reports. The next useful Milestone 2b step is a small calibration
sweep over the probabilistic PCA residual model, not a larger full-3D generator.

The next implementation should compare:

1. random seeds for the current 3-component MDN;
2. mixture counts such as `1`, `3`, and `5`;
3. posterior-mean cell-mass MAE;
4. PCA coefficient 68 percent coverage;
5. coverage stratified by inclination and bar viewing angle if sample sizes are
   adequate;
6. whether simple variance temperature calibration is needed.

---

## Suggested Next Implementation Plan

### Task 1: Add Sweep Driver

Create a focused local script that trains `train_density_residual_pca_mdn.py`
across a small grid of seeds and mixture counts, writes one subdirectory per
run, and aggregates a CSV/JSON summary.

Likely file:

- `scripts/sweep_density_residual_pca_mdn.py`

Keep the script small. It can call the existing trainer as a subprocess rather
than duplicating training code.

### Task 2: Add Calibration Summary

Aggregate at least:

- posterior-mean cell-mass MAE;
- coefficient RMSE;
- coefficient 68 percent coverage;
- optional coverage by inclination and bar viewing angle;
- best validation NLL and epoch.

Likely report:

- `docs/reports/milestone2b_density_residual_pca_mdn_sweep.md`

### Task 3: Decide Whether Calibration Is Enough

If coverage remains consistently conservative near `0.75`, try a simple
variance-temperature post-processing check. If coverage is unstable across
seeds, keep the MDN as a provisional uncertainty baseline and avoid claiming
calibration.

### Task 4: Defer Larger Models

Do not move to flow matching, diffusion, or a full 3D generator until the PCA
coefficient uncertainty baseline is stable and clearly documented.

---

## Verification Baseline

The last verified local state after the smooth-baseline repair and PCA/MDN
evaluation:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest
```

Result:

- `ruff`: all checks passed
- `pytest`: `87 passed in 30.94s`

---

## Good Opening Prompt For A New Thread

```text
Continue disk-galaxy-deprojection Milestone 2b from commit 06d6e60 on branch
codex/milestone2-tng-ingestion-design. The workspace contains uncommitted
Milestone 2b code and output artifacts, so inspect git status first. Read
docs/superpowers/plans/2026-06-08-milestone2-3d-baseline-handoff.md first.
The deterministic PCA report, MDN report, and smooth `sech^2` baseline repair
are already done. Next task: run a small probabilistic PCA residual calibration
sweep over MDN seeds and mixture counts, aggregate posterior-mean MAE and 68
percent coefficient coverage, and write a short report recommending whether
temperature calibration is enough or whether the uncertainty model is still too
unstable. Do not download TNG data locally. Use the existing cluster wrapper
only if a remote job is actually needed; this sweep should likely run locally
from fetched Milestone 2b outputs.
```
