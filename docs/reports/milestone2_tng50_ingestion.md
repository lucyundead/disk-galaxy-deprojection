# Milestone 2: TNG50 Ingestion

## Status

TNG50 ingestion pipeline implemented and executed on the remote cluster through
`/home/lucyundead/codex/hpc-agent/hpc shell`.

## Pipeline Overview

1. Build TNG50 candidate manifest from group catalog
2. Extract compact stellar particle files using offset-based reading
3. Build residual benchmark table from TNG50 particles
4. Train summary residual MDN on TNG50 data
5. Evaluate metrics and calibrate uncertainties

## TNG50 Tiny Benchmark Metrics

- Manifest: outputs/tng50_milestone2/manifest.csv
- Residual table: outputs/tng50_milestone2/residual_table.npz
- Metrics: outputs/tng50_milestone2/metrics.json

- Candidate rows: 24
- Compact stellar particle files written remotely: 24
- `baseline_mae`: 1914679168.0
- `corrected_mae`: 368521838592.0
- `coverage_68`: 0.31
- `n_test`: 5

This first TNG50 run validates ingestion and benchmark wiring only. It is not a
scientific performance claim: the sample is selected from massive central
subhalos by group-catalog cuts, not yet from a barred-galaxy catalog; each
galaxy is truncated to at most 80000 stellar particles; and the MDN was trained
for only 10 epochs.

## Cluster Verification

- `python scripts/cluster_dgdp.py sync`: passed using archive fallback because
  local `rsync` is unavailable in WSL.
- `python scripts/cluster_dgdp.py check-env`: passed in `paicos-conda` for
  `numpy`, `pandas`, `h5py`, and `torch`.
- `python scripts/cluster_dgdp.py reproduce-milestone1`: passed.
  - `baseline_mae`: 589892416.0
  - `corrected_mae`: 431299584.0
  - `coverage_68`: 0.7291666666666666
  - `n_test`: 12
- `python scripts/cluster_dgdp.py run-tng50`: passed.
- `python scripts/cluster_dgdp.py fetch-tng50`: passed and fetched compact
  outputs locally, excluding remote particle files.

The remote TNG50-1 group catalog was inspected directly on 2026-06-03 under
`/home/cossim/IllustrisTNG/TNG50-1/groups_099`. Empty group-catalog chunks omit
some datasets, so the reader skips empty chunks.

## Barred TNG50 Sample Definition

The downloaded morphology/bar catalog is available on the cluster at
`/home/zli/disk-galaxy-deprojection/morphs_kinematic_bars.hdf5`. Its HDF5 header
identifies it as the IllustrisTNG supplementary data catalog
`morphs_kinematic_bars`, reference `Zana et al. (2022)`, for `TNG50-1`.

For the first real barred-galaxy sample, use:

- snapshot: `99`
- group catalog root: `/home/cossim/IllustrisTNG/TNG50-1`
- central subhalos only, via `GroupFirstSub`
- `star_particles >= 50000`
- `stellar_mass_msun >= 1.0e9`
- catalog `Snapshot_99/Barred == True`
- primary `Snapshot_99/BarStrength[0] >= 0.2`
- primary `Snapshot_99/BarSize[0] >= 1.0`
- deterministic galaxy-level split seed: `20260604`

This cut produces 266 available barred central candidates. The first runnable
subset is the top 64 by stellar mass:

- remote manifest: `outputs/tng50_milestone2/barred_sample_64.csv`
- local fetched manifest: `outputs/tng50_milestone2/barred_sample_64.csv`
- split counts: 38 train, 13 validation, 13 test
- stellar mass range: `8.66431488834513e10` to `1.6546997800139434e12` Msun
- primary bar-strength range: `0.2064` to `0.5684`
- primary bar-size range: `1.2421` to `7.4258`

## Remote Commands

```bash
python scripts/cluster_dgdp.py sync
python scripts/cluster_dgdp.py check-env
python scripts/cluster_dgdp.py reproduce-milestone1
python scripts/cluster_dgdp.py run-tng50
python scripts/cluster_dgdp.py fetch-tng50
```

## Files Created

- src/dgdp/tng50_catalog.py - Group catalog reader with TNG50GroupCatalog dataclass
- src/dgdp/bar_catalog.py - Optional bar catalog inspection and join
- src/dgdp/tng50.py - Offset-based stellar particle extraction (expanded)
- scripts/build_tng50_manifest.py - Remote manifest builder
- scripts/extract_tng50_particles.py - Remote particle extractor
- scripts/build_tng50_benchmark.py - Remote benchmark builder
- scripts/cluster_dgdp.py - Cluster CLI (expanded with run-tng50)
- configs/milestone2.cluster.toml - Cluster configuration

## Next Decisions

- Replace group-catalog-only candidates with the downloaded kinematic
  morphology/bar catalog once its schema is inspected.
- Reduce the mismatch between TNG particles and Milestone 1 synthetic summaries
  before interpreting corrected MAE.
- Move from this summary residual smoke test toward the Milestone 2 coarse 3D
  target only after the barred sample and target definition are fixed.
