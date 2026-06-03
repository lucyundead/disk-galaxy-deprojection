# TNG50 Ingestion Cluster Design

## Purpose

Milestone 2 replaces the Milestone 1 synthetic particle source with selected
TNG50-1 stellar particles while preserving the existing summary-residual
benchmark target:

```text
p(delta_summaries | image_mock, baseline_summaries, geometry, metadata)
```

This milestone runs TNG-facing work on the remote cluster because the full
TNG50-1 snapshot data are too large to move locally. Coarse 3D residual modeling
is deferred until this TNG ingestion workflow is reproducible.

## Approved Direction

Use a thin project-local cluster bridge. The bridge calls the existing local HPC
wrapper at:

```text
/home/lucyundead/codex/hpc-agent/hpc
```

The bridge copies lightweight DGDP project files to:

```text
/home/zli/disk-galaxy-deprojection/
```

and runs DGDP-specific scripts on the cluster in the `paicos-conda`
environment. It must not copy Codex/OpenAI files, local virtual environments,
large generated outputs, or full TNG50 data.

## Remote Data Sources

Remote TNG50-1 data live at:

```text
/home/cossim/IllustrisTNG/TNG50-1/
```

Milestone 2 uses:

- `groups_099/fof_subhalo_tab_099.*.hdf5` for z=0 subhalo metadata,
- `postprocessing/offsets/offsets_099.hdf5` for particle-range lookup if the
  required subhalo offsets are present,
- `snapdir_099/` only for selected `PartType4` stellar particles,
- the downloaded Galaxy Morphologies and Bar Properties catalog if it proves to
  contain both a usable subhalo identifier and a z=0 bar label or bar property
  field.

Earlier snapshots, real S4G data, and full snapshot transfer remain out of
scope.

## Cluster Reproduction Gate

Before reading TNG50 data, reproduce Milestone 1 on the cluster:

1. Sync the lightweight DGDP project files to
   `/home/zli/disk-galaxy-deprojection/`.
2. Activate or otherwise use `paicos-conda`.
3. Verify the remote environment can import `numpy`, `pandas`, `h5py`, and
   `torch`.
4. Run the synthetic benchmark build, training, and evaluation scripts on the
   cluster.
5. Confirm remote `metrics.json` reports finite `baseline_mae`,
   `corrected_mae`, `coverage_68`, and `n_test`.

This gate proves the wrapper, remote project copy, Python environment, PyTorch,
and artifact paths work before the TNG-specific scripts are introduced.

## TNG50 Ingestion Workflow

After the cluster reproduction gate passes:

1. Build a z=0 candidate manifest from `groups_099`.
2. If the downloaded morphology/bar catalog contains a subhalo identifier plus
   a bar label or quantitative bar field at z=0, join it to the manifest and
   select barred disk candidates from those catalog fields.
3. If the catalog lacks those fields, fall back to conservative group catalog
   filters first and postpone custom bar classification to a later slice.
4. Extract only selected subhalos' stellar particles from `snapdir_099`.
5. Store compact per-galaxy artifacts and a manifest under:

```text
/home/zli/disk-galaxy-deprojection/outputs/tng50_milestone2/
```

6. Build the same residual table shape used by Milestone 1.
7. Train and evaluate the existing MDN residual model on the cluster.
8. Fetch only compact artifacts, metrics, and reports back to the local
   workspace.

## Particle Fields

The first TNG particle extraction should read only the fields needed for the
existing benchmark and audit trail:

- `PartType4/Coordinates`,
- `PartType4/Masses`,
- `PartType4/Velocities`,
- `PartType4/GFM_StellarFormationTime`,
- `PartType4/ParticleIDs` when available.

The extraction should exclude non-stellar wind particles by keeping only
particles with positive `GFM_StellarFormationTime` when that field is present.

## Outputs

Remote outputs:

- cluster Milestone 1 reproduction metrics,
- TNG50 candidate manifest,
- selected-galaxy compact particle artifacts,
- TNG50 residual table,
- TNG50 trained model checkpoint,
- TNG50 evaluation metrics.

Local fetched outputs:

- manifest CSV,
- metrics JSON,
- compact reports,
- model checkpoint if it is below 500 MB; otherwise fetch only metrics and
  checkpoint metadata.

## Testing And Verification

Milestone 2 tests should cover:

- cluster config parsing,
- remote command construction without executing SSH in unit tests,
- TNG group-catalog field reading from small HDF5 fixtures,
- morphology/bar catalog inspection from small HDF5 fixtures,
- selected particle extraction from small HDF5 fixtures,
- manifest split validation by subhalo ID,
- residual table compatibility with Milestone 1 scripts.

Remote verification should cover:

- `paicos-conda` import check,
- remote Milestone 1 synthetic reproduction,
- TNG candidate manifest creation,
- extraction of a tiny TNG sample,
- training/evaluation on that tiny TNG sample.

## Boundaries

Do not download or copy full TNG50 snapshots locally. Do not run Codex or install
OpenAI packages on the cluster. Do not train on real S4G data. Do not implement
coarse 3D residual modeling in this milestone. Do not delete remote files.

## External References

- TNG data specifications, especially snapshot and group-catalog structure:
  https://www.tng-project.org/data/docs/specifications/ accessed 2026-06-03.
- TNG example scripts for local snapshot and group-catalog file organization:
  https://www.tng-project.org/data/docs/scripts/ accessed 2026-06-03.

## Success Criteria

Milestone 2 is successful when:

1. The cluster wrapper can sync and run DGDP scripts through the existing HPC
   agent.
2. Milestone 1 synthetic build, training, and evaluation are reproduced on the
   cluster.
3. A documented z=0 TNG50 candidate manifest is generated from `groups_099` and
   optional bar catalog fields.
4. A tiny selected TNG50 stellar-particle sample is extracted without moving
   full snapshots.
5. The existing summary-residual benchmark trains and evaluates on that TNG50
   sample.
6. Metrics and reports clearly state that this is the first TNG ingestion
   benchmark, not yet the coarse 3D residual milestone.
