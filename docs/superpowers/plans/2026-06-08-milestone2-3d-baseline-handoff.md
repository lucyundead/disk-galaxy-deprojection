# Milestone 2 3D Baseline Handoff

> **For the next thread:** Start here before editing code. This handoff captures
> the current Milestone 2 state after TNG50 ingestion, projection diagnostics,
> and true cylindrical 3D density-grid generation.

**Date:** 2026-06-08

**Current branch:** `codex/milestone2-tng-ingestion-design`

**Latest relevant commit:** `8e029c0 Add TNG50 cylindrical density grids`

**Workspace:** `/home/lucyundead/projects/disk-galaxy-deprojection`

**Remote project root:** `/home/zli/disk-galaxy-deprojection`

**Cluster wrapper:** `/home/lucyundead/codex/hpc-agent/hpc`

---

## Project Direction

The project is following the residual ladder:

1. Classical deprojection baseline.
2. Residual posterior on physical summary quantities.
3. Coarse 3D residual around a baseline 3D density field.

Do not jump directly to a large end-to-end 3D generator. The immediate next
target is:

```text
delta_rho_3d = rho_true_cylindrical - rho_baseline_cylindrical
```

where both fields use the same disk/bar-aligned cylindrical grid.

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

---

## What Not To Redo

- Do not rebuild the barred sample unless the user explicitly asks.
- Do not regenerate face-on galleries unless changing sample cuts.
- Do not rerun `run-tng50` unless changing the projection grid, sample, or
  summary benchmark logic.
- Do not download TNG data locally. TNG data should remain on the cluster.
- Do not add delete or cleanup behavior to the cluster wrapper.

---

## Next Thread Goal

Build matching baseline cylindrical 3D density grids in the same coordinate
system as the true density products.

The next implementation should produce:

1. A baseline particle or density construction from each projection image using
   known geometry.
2. A cylindrical baseline grid with the same `(R, phi, z)` bin edges as the
   true grid.
3. Per-projection baseline grid files or a compact baseline tensor dataset.
4. A first residual dataset:

```text
rho_true_cylindrical[galaxy] - rho_baseline_cylindrical[projection]
```

5. Verification checks:
   - baseline and truth use identical grid edges;
   - baseline mass is finite and nonnegative;
   - truth-grid mass is conserved within the stored grid volume;
   - baseline reprojection roughly matches the input mock image;
   - train/validation/test split remains by galaxy.

---

## Suggested Next Implementation Plan

### Task 1: Baseline 3D Grid Spec

Create a small reusable helper that reads true-grid HDF5 edges and constructs a
`CylindricalGridSpec`. This prevents drift between true and baseline grids.

Likely files:

- Modify: `src/dgdp/density3d.py`
- Test: `tests/test_density3d.py`

### Task 2: Baseline Particles To Cylindrical Grid

Use the existing `baseline_particles_from_image(...)` as the first simple
baseline. Convert those particles into a cylindrical grid using the same grid
edges as the true density product.

Likely files:

- Create: `scripts/build_tng50_baseline_density_grid.py`
- Reuse: `src/dgdp/baseline.py`
- Reuse: `src/dgdp/density3d.py`
- Test: `tests/test_build_tng50_baseline_density_grid.py`

Important: baseline particles are already in the deprojected frame generated
from the image baseline. Confirm whether additional bar-angle rotation is
needed before binning.

### Task 3: Residual Dataset

Build a compact dataset that aligns:

- projection metadata from `outputs/tng50_milestone2/manifest.csv`
- mock images from `outputs/tng50_milestone2/residual_table.npz`
- true density grids from
  `outputs/tng50_milestone2/density_grids_logr_cyl/`
- baseline density grids from the new baseline output directory

Start with a downsampled or summary projection of the 3D residual before
training a full high-dimensional model.

Likely files:

- Create: `scripts/build_tng50_density_residual_table.py`
- Test: `tests/test_build_tng50_density_residual_table.py`

### Task 4: Cluster Wiring

Add a new cluster subcommand only after the local script has tests:

```bash
python scripts/cluster_dgdp.py run-tng50-baseline-density
```

Keep the wrapper thin and avoid remote deletion behavior.

Likely files:

- Modify: `scripts/cluster_dgdp.py`
- Test: `tests/test_cluster_dgdp.py`

---

## Verification Baseline

The last verified local state after commit `8e029c0`:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest
```

Result:

- `ruff`: all checks passed
- `pytest`: `70 passed`

---

## Good Opening Prompt For A New Thread

```text
Continue disk-galaxy-deprojection Milestone 2 from commit 8e029c0 on branch
codex/milestone2-tng-ingestion-design. Read
docs/superpowers/plans/2026-06-08-milestone2-3d-baseline-handoff.md first.
Next task: implement matching baseline cylindrical 3D density grids for
outputs/tng50_milestone2/density_grids_logr_cyl, then define the first
delta_rho_3d residual dataset. Use the existing cluster wrapper and do not
download TNG data locally.
```
