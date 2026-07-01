# Mixture q_m retrain: fixed sech² height dictionary (not per-galaxy-scaled)

**Date:** 2026-06-30
**Outcome:** the free-knot vertical q_m is replaced by a **fixed K=7 sech² height dictionary**
(geomspace 0.2–3.5 kpc). Thickness is carried by the predicted weight *vector* (which heights
get weight), not a per-galaxy scalar. This gives z-symmetry + positivity + ∫q dz=1 **by
construction** (so the NGC post-hoc anchor-renorm / z-symmetrise / taper are deleted) **and**
generalises to OOD real galaxies (NGC 4321/4371), which the per-galaxy-scaled variant did not.

## What changed this session
- **Data (cluster):** merged the richer-grid milestone-2d shards → `residual_table.npz`
  (36,630 rows / 185 galaxies) and built `density_residual_table.npz` (R=64, ~31 GB, stays on
  the cluster). The login-node concat OOM'd (a login-node watchdog kills ~>8 GB) → `merge_cmd`
  rewritten frugal (pre-allocate images, fill shard-by-shard, ~5.4 GB) + atomic (tmp→os.replace).
- **Mixture wiring:** `src/dgdp/vertical_mixture.py` gained `weights_target` + `reconstruct`
  (fixed-dictionary), wired into `deproject_fourier_rz_conserving.py` (target build +
  `reconstruct_smooth`) and mirrored into `deproject_real_image_ngc4321_learned.py` (post-hoc
  anchor-renorm **deleted**). New config `configs/fourier_rz_mixture.json` (frozen allocation
  untouched).
- **Latent bug fixed:** `vol` was built from a *defaulted* 32×48 grid spec, not the table edges
  — it would have crashed the R=64 retrain on line 1. Now `vol` uses the table's own edges.
- **grid_mdn skipped on the cluster:** `fit_pca` uses an exact `np.linalg.svd`; on the full grid
  residual (21978×98304) that is ~5e16 flops (infeasible) and runs before the mixture variants.
  Added `--skip-grid-mdn` (the non-conserving PCA-on-grid baseline isn't needed; the free-knot
  JSON is the reference).

## The parametrization journey (3 iterations)
1. **Per-galaxy-scaled K=4** (heights = s·{0.3,0.7,1.5,3.0}, s predicted per galaxy). Exact
   conservation; on matched R=32 data it **matched/beat** free-knot (image_anchor relL2 0.434 vs
   0.470, oracle 0.330 vs 0.337, massAcc better everywhere) at 160 vs 496 coeffs. **But** on the
   real galaxies it produced disks ~15–30 % too thin (NGC 4321 1.30 vs free-knot 1.54; NGC 4371
   0.955 vs MGE ~1.4).
2. **Diagnosis** (`scripts/_diag_shead.py`, since deleted): the scalar-s MDN head **regresses to
   the mean** — thick-tercile truth s 0.583 → predicted 0.462 (×0.78), predicted range half the
   truth range, corr 0.64. Thick galaxies (incl. OOD reals) get a too-small s → too thin.
3. **Calibration (variance-match log s) FAILED:** NGC 4371 went 0.955 → 0.880 kpc (thinner). The
   s-head predicts NGC 4371 *below the median*; expanding the deviation pushes a below-median
   point further down. Calibration fixes spread, not a wrong location.
4. **Fixed dictionary (adopted):** drop per-galaxy s; predict weights on fixed heights. Thickness
   lives in the weight vector (like the free-knot z-profile did), so no scalar bottleneck.

## Results — fixed K=7 dictionary
Real galaxies (train on R=32 TNG, apply OOD; post-hoc patches removed):

| galaxy | free-knot | per-gal s | **fixed dict** | conservation (mass frac R<3) | v_c |
|---|---|---|---|---|---|
| NGC 4321 RMS\|z\| | 1.54 | 1.30 | **1.44** | 0.257 → 0.257 | tracks baseline |
| NGC 4371 RMS\|z\| | ~1.5 (MGE ~1.4) | 0.955 | **1.16** | 0.529 → 0.536 | ~178, tracks baseline |

In-distribution recovery (R=32 held-out, `--skip-grid-mdn`):

| variant | free-knot | per-gal s | **fixed dict** |
|---|---|---|---|
| image_anchor relL2 | 0.470 | 0.434 | 0.488 |
| radial_all relL2 | 0.375 | 0.376 | 0.394 |
| oracle relL2 | 0.337 | 0.330 | 0.362 |
| massAcc (radial) | 0.061 | 0.040 | 0.040 |

**Tradeoff:** fixed dict is ~5 % worse on in-distribution relL2 (the per-galaxy s adapts its
basis per galaxy — the very adaptivity that breaks OOD) but keeps exact conservation (massAcc
0.040) and is **right on the real galaxies**, which is the application. Target 280 coeffs
(vs free-knot 496). K=7 target representation rel-L2 0.0115.

## R=64 cluster retrain
Job 404397 (PBS, mem=160gb, ppn=8, `--skip-grid-mdn`, fixed-dict config) on
`density_residual_table.npz`, 26 min. Output:
`outputs/tng50_milestone2d_rich/mixture_retrain/` (metrics json + figure, stays on cluster).
Held-out R=64 recovery, n_target 280:

| variant | relL2 | massAcc | mass_cons_vs_image |
|---|---|---|---|
| geom_baseline | 0.530 | 0.079 | 0 |
| cons. image_anchor | 0.522 | 0.079 | 4.1e-8 |
| cons. +radial m=0 | 0.457 | 0.033 | 0.061 |
| cons. +radial all | 0.455 | 0.033 | 0.061 |
| cons. oracle_anchor | 0.433 | 5e-8 | 0.071 |

Same pattern as R=32: **exact mass conservation** (4e-8), massAcc better than free-knot
(0.033 vs 0.061), relL2 ~7 % above the per-galaxy R=64 run (0.455 vs 0.426 radial_all) — the
OOD-robustness tradeoff. relL2 sits above R=32 purely from the finer grid (geom_baseline 0.530
vs 0.495, parametrization-independent).

## Files (all UNCOMMITTED)
- `src/dgdp/vertical_mixture.py`: `weights_target`/`reconstruct` now take an absolute `heights`
  dictionary (removed the per-galaxy `vertical_scales`/`calibrate_scale` experiments).
- `scripts/deproject_fourier_rz_conserving.py`: mixture target + `reconstruct_smooth`; `vol`
  from table edges; `--skip-grid-mdn`; mixture output filenames (free-knot baseline preserved).
- `scripts/deproject_real_image_ngc4321_learned.py`: fixed-dict mirror; post-hoc anchor-renorm
  deleted.
- `configs/fourier_rz_mixture.json`: allocation (12/8/6 radial knots × K=7) + heights.
- `scripts/_milestone2d_richgrid_cluster.py`: frugal `merge`, `retrain` mode, `pbs_single(ppn=)`.
- Verify: `ruff` clean; `pytest -q` 111 passed / 1 skipped; `python -m dgdp.vertical_mixture`
  self-check green.

## Open / next
- Fill in the R=64 retrain numbers (404397).
- NGC 4371 (1.16) is ~17 % under the MGE (1.4) — the hardest OOD case (SB0); within the report's
  15–20 % MGE tolerance but the thinnest of the reals. K or the height range could be tuned if a
  closer MGE match is wanted.
- Commit is pending an explicit go-ahead.
