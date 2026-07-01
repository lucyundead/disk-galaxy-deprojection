# Session handoff — 2026-06-30

**State:** a richer-grid milestone-2d dataset is being generated on the cluster for the q_m
**mixture retrain**. Two Torque/PBS jobs were launched 2026-06-29 (~8 h image array + a
parallel truth job) — by now they are likely **done**. Pick up at the resume sequence below.

## Environment / rules
- WSL `openSUSE-Tumbleweed`; project `/home/lucyundead/projects/disk-galaxy-deprojection`;
  run with `.venv/bin/python`; branch `codex/milestone2-tng-ingestion-design`.
- Cluster: **Torque/PBS (Maui), NOT SLURM** — `qsub`/`qstat`, queue `normal`; driven by
  `scripts/_milestone2d_richgrid_cluster.py` (uses the tested `cluster_dgdp.run_remote` +
  the `hpc` wrapper). Remote root `/home/zli/disk-galaxy-deprojection`; conda `paicos-conda`
  **has torch** → we train on the cluster and fetch only small outputs.
- Keep `ruff check .` and `pytest -q` clean (baseline 111 passed/1 skipped; new scripts
  aren't imported by tests). Commit only when asked. **Do NOT** regenerate the frozen
  `outputs/nbody_shen2010/fourier_rz_allocation.json`. AGAMA via
  `PYTHONPATH=/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313`.

## RESUME SEQUENCE (do this first)
All via `.venv/bin/python scripts/_milestone2d_richgrid_cluster.py <mode>`:
1. **`qstat`** — is `404003[]` (image array, 185 tasks) and `404008` (truth R=64) still running?
   Also `shards` (count per-galaxy shards) and `check`. If the array failed any tasks, the
   per-galaxy builder is idempotent — re-`qsub` the failed indices (`-t i,j,...`).
2. When **`404003[]` done** → **`merge`** (login-node concat of 185 shards →
   `outputs/tng50_milestone2d_rich/{residual_table.npz, manifest.csv}`). Expect **36,630 rows /
   185 galaxies**.
3. When **merge + `404008` both done** → **`table`** (PBS: baseline grids +
   `build_tng50_density_residual_table` → `outputs/tng50_milestone2d_rich/density_residual_table.npz`,
   **~36 GB**, truth+baseline+images per row at R=64). **Stays on the cluster** (too big to fetch).
4. **Wire the mixture** (the one remaining code step, not yet done): replace the free-knot q_m
   in `scripts/deproject_fourier_rz_conserving.py` (target build ~l.163-176 + `reconstruct_smooth`
   ~l.184-216) using `src/dgdp/vertical_mixture.py` (per-galaxy-scaled **K=4 sech² mixture**;
   positive weights + **signed fallback for m=2**; **new** mixture config — leave the frozen
   allocation alone; keep PCA + clip/renorm). Full design: `docs/reports/2026-06-25-mixture-qm-retrain-scope.md`.
5. **Retrain ON THE CLUSTER** on the new table (sync code, run remotely, fetch model/metrics/figures).
   Then mirror the mixture into `deproject_real_image_ngc4321_learned.py` (delete the post-hoc
   anchor-renorm / z-symmetrise / taper — now structural) and re-run NGC 4321 + NGC 4371 to
   confirm v_c & RMS|z| hold.

## Rebuild design (the running jobs)
Grid: inclination 10–60° step 5 (11) × bar −80…90° step 10 (18) = **198 proj/galaxy × 185 =
36,630 rows**; R=64 log, n_phi 48, n_z 32, z_max 5. **No negative inclinations** (degenerate
with +i & flipped bar for a midplane-symmetric disk, already spanned; the mixture q_m is
z-symmetric anyway); −90 dropped (=+90 for an m=2 bar). The image builder re-projects all
particles per view (~15 s/proj → ~6 days serial) → parallelized as a **PBS job array**
(`build_tng50_all_particle_images.py --galaxy-index`, one task/galaxy → `shards/<i>/`).
Validated on a 2-task array (correct per-galaxy shards). Per-task: `nodes=1:ppn=1, 16gb, 2h`,
array cap `%20`. `base_manifest.csv` (185 gal; split 111 train / 37 test / 37 val) already written.

## Why we're doing this (key findings — pointers for depth)
- **Thickness is largely NOT the OOD** for total-light deprojection in the TNG mass range:
  sample-weighted RMS|z| ratio ≈ 1.07 vs Comerón+2018. The deprojection target **includes the
  stellar halo** (a face-on image contains halo light) → memory `deprojection-target-includes-halo`;
  pure-disc N-body is a poor retrain target. Report: `docs/reports/2026-06-25-tng-vertical-scale-vs-comeron.md`.
- **NGC 4371 (SB0) vs Behzad Tahmasebzadeh's independent MGE deprojection**: the TNG-learned
  q_m vertical structure matches the MGE to ~15–20 %; after an **anchor-conservation fix** the
  rotation curve matches too (175 vs 179 km/s); the image-anchored in-plane keeps the bar A2
  better than the MGE. Data + notebooks in repo `NGC4371/`. Report:
  `docs/reports/2026-06-25-ngc4371-mge-validation.md`.
- **Parametrization pre-check**: a per-galaxy-scaled K=4 sech² mixture represents the truth
  vertical profiles *better* than the free-knot q_m (RMS|z| err median 5 % vs 18 %) **and**
  gives z-symmetry + positivity + ∫q dz=1 by construction → bake into the TRAINING
  parametrization. Module `src/dgdp/vertical_mixture.py`; de-risk `scripts/test_mixture_pergalaxy.py`.
- The long running-log handoff `docs/reports/2026-06-10-project-handoff.md` has dated blocks
  for each of the above (incl. the full "RICHER-GRID REBUILD RUNNING ON CLUSTER 2026-06-29" block).
  Memory (`disk-galaxy-deprojection-workflow`, `ngc4371-mge-validation`,
  `deprojection-target-includes-halo`) loads automatically.

## New/changed files this session (all UNCOMMITTED)
- New scripts: `measure_tng_vertical_scale.py`, `compare_tng_comeron_vertical.py`,
  `ngc4371_mge_benchmark.py`, `compare_ngc4371_learned_vs_mge.py`,
  `plot_ngc4371_density_comparison.py`, `compare_ngc4371_vc_and_bar.py`,
  `test_mixture_vs_freeknot.py`, `test_mixture_pergalaxy.py`, `_milestone2d_richgrid_cluster.py`.
- New module: `src/dgdp/vertical_mixture.py`.
- Edits: `deproject_real_image_ngc4321_learned.py` (WCS-bypass flags `--pix-arcsec/--pa-pix-deg/
  --center-x/-y/--mask/--galaxy-name` + the per-column anchor-conservation fix in `predict_learned`);
  `build_tng50_all_particle_images.py` (`--galaxy-index`); `pyproject.toml` (ruff `extend-exclude=["NGC4371"]`).
- Outputs: `outputs/tng50_vertical_scale/`, `outputs/real_images/ngc4371_*`. Reports listed above.
- Cluster (remote): `outputs/tng50_milestone2d_rich/` (shards + base_manifest now; the ~36 GB table after `table`).

## Gotchas
- Inline python sent to the cluster via `wsl bash -lc '...'`: **no single quotes** in the python
  (Windows quoting eats them); shell loop vars also get eaten — use the orchestration script's modes.
- `build_tng50_all_particle_images.py --bar-angles-deg=-80,...` must use the `=` form (leading `-`).
- `qstat <id>` needs the array form; use the `qstat` mode (parses `dgdp_img` task states).
- The 36 GB table won't load in 15 GB local RAM — that's why training is on the cluster.
