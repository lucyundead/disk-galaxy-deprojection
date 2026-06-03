# TNG50 Ingestion Cluster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cluster-run Milestone 2 workflow that first reproduces Milestone 1 on the remote cluster, then ingests a tiny selected TNG50-1 z=0 stellar-particle sample into the existing summary-residual benchmark.

**Architecture:** Keep local DGDP as the source of truth, add a thin cluster bridge that syncs lightweight files to `/home/zli/disk-galaxy-deprojection/`, and run remote commands through `/home/lucyundead/codex/hpc-agent/hpc shell`. TNG50 data stay on the cluster; local fetches bring back only compact manifests, metrics, reports, and small checkpoints.

**Tech Stack:** Python 3.11+, NumPy, pandas, h5py, PyTorch, pytest, ruff, rsync, existing HPC wrapper.

---

## Scope

This plan implements the approved TNG50 ingestion milestone:

1. Add local cluster configuration and command construction.
2. Add a project-local cluster CLI that syncs code and runs remote DGDP commands through the existing HPC wrapper.
3. Reproduce the Milestone 1 synthetic benchmark on the cluster using `paicos-conda`.
4. Read TNG50 z=0 group catalogs and optional bar catalog files.
5. Extract only selected `PartType4` stellar particles from `snapdir_099` using offset files.
6. Build, train, and evaluate the existing summary-residual benchmark on a tiny TNG50 sample.

This plan does not implement coarse 3D residuals.

## File Structure

- Create: `configs/milestone2.cluster.toml` - local/remote cluster paths and run defaults.
- Create: `src/dgdp/cluster.py` - cluster config loading, sync manifest, remote command construction.
- Create: `scripts/cluster_dgdp.py` - project-local bridge CLI for sync, env check, Milestone 1 reproduction, remote run, and fetch.
- Create: `src/dgdp/tng50_catalog.py` - group-catalog chunk reading and z=0 candidate manifest construction.
- Create: `src/dgdp/bar_catalog.py` - HDF5 catalog inspection and optional bar-catalog join.
- Modify: `src/dgdp/tng50.py` - add offset-based selected stellar-particle extraction and compact particle writing.
- Create: `scripts/build_tng50_manifest.py` - remote script to write a selected TNG50 candidate manifest.
- Create: `scripts/extract_tng50_particles.py` - remote script to write compact per-galaxy particle files.
- Create: `scripts/build_tng50_benchmark.py` - remote script to build a residual table from compact TNG50 particles.
- Create: `docs/reports/milestone2_tng50_ingestion.md` - report template populated after remote verification.
- Modify: `README.md` - document cluster Milestone 2 commands.
- Create tests under `tests/` for every new module and script smoke path.

Generated remote outputs:

- `/home/zli/disk-galaxy-deprojection/outputs/cluster_milestone1/`
- `/home/zli/disk-galaxy-deprojection/outputs/tng50_milestone2/`

Fetched local outputs:

- `outputs/cluster_milestone1/`
- `outputs/tng50_milestone2/`

## Assumptions

- Remote TNG50 base path is `/home/cossim/IllustrisTNG/TNG50-1`.
- Remote DGDP path is `/home/zli/disk-galaxy-deprojection`.
- Local HPC wrapper path is `/home/lucyundead/codex/hpc-agent/hpc`.
- Remote commands can activate the existing `paicos-conda` environment by running `source /home/zli/.bashrc` first.
- The downloaded bar/morphology catalog path is optional. If unset or unusable, the first TNG sample uses conservative group-catalog cuts.
- TNG particle coordinates and masses are converted to physical kpc and solar masses using `HubbleParam` from file headers.

## Task 1: Cluster Config And Command Builder

**Files:**
- Create: `configs/milestone2.cluster.toml`
- Create: `src/dgdp/cluster.py`
- Create: `tests/test_cluster.py`

- [ ] **Step 1: Write the failing cluster config tests**

Create `tests/test_cluster.py`:

```python
from pathlib import Path

from dgdp.cluster import build_remote_script, load_cluster_config, sync_include_paths


def test_load_cluster_config_reads_defaults():
    cfg = load_cluster_config(Path("configs/milestone2.cluster.toml"))

    assert cfg.hpc_wrapper == Path("/home/lucyundead/codex/hpc-agent/hpc")
    assert cfg.remote_project_root == "/home/zli/disk-galaxy-deprojection"
    assert cfg.remote_tng50_root == "/home/cossim/IllustrisTNG/TNG50-1"
    assert cfg.conda_env == "paicos-conda"
    assert cfg.remote_output_dir == "outputs/tng50_milestone2"


def test_sync_include_paths_are_lightweight():
    paths = sync_include_paths()

    assert "src" in paths
    assert "scripts" in paths
    assert "configs" in paths
    assert ".venv" not in paths
    assert "outputs" not in paths


def test_build_remote_script_activates_environment_and_project():
    script = build_remote_script(
        remote_project_root="/home/zli/disk-galaxy-deprojection",
        conda_env="paicos-conda",
        commands=["python -c \"print('ok')\""],
    )

    assert "source /home/zli/.bashrc" in script
    assert "conda activate paicos-conda" in script
    assert "cd /home/zli/disk-galaxy-deprojection" in script
    assert "python -c \"print('ok')\"" in script
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_cluster.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'dgdp.cluster'`.

- [ ] **Step 3: Create the cluster config**

Create `configs/milestone2.cluster.toml`:

```toml
hpc_wrapper = "/home/lucyundead/codex/hpc-agent/hpc"
cluster_host = "gravity-login01"
remote_project_root = "/home/zli/disk-galaxy-deprojection"
remote_tng50_root = "/home/cossim/IllustrisTNG/TNG50-1"
conda_env = "paicos-conda"
remote_milestone1_output_dir = "outputs/cluster_milestone1"
remote_output_dir = "outputs/tng50_milestone2"
local_fetch_dir = "outputs/tng50_milestone2"
bar_catalog_path = ""
snapshot = 99
hubble_param = 0.6774
max_candidate_galaxies = 24
max_particles_per_galaxy = 80000
```

- [ ] **Step 4: Implement the config and command builder**

Create `src/dgdp/cluster.py`:

```python
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClusterConfig:
    hpc_wrapper: Path
    cluster_host: str
    remote_project_root: str
    remote_tng50_root: str
    conda_env: str
    remote_milestone1_output_dir: str
    remote_output_dir: str
    local_fetch_dir: str
    bar_catalog_path: str
    snapshot: int
    hubble_param: float
    max_candidate_galaxies: int
    max_particles_per_galaxy: int


def load_cluster_config(path: Path) -> ClusterConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return ClusterConfig(
        hpc_wrapper=Path(data["hpc_wrapper"]),
        cluster_host=str(data["cluster_host"]),
        remote_project_root=str(data["remote_project_root"]),
        remote_tng50_root=str(data["remote_tng50_root"]),
        conda_env=str(data["conda_env"]),
        remote_milestone1_output_dir=str(data["remote_milestone1_output_dir"]),
        remote_output_dir=str(data["remote_output_dir"]),
        local_fetch_dir=str(data["local_fetch_dir"]),
        bar_catalog_path=str(data["bar_catalog_path"]),
        snapshot=int(data["snapshot"]),
        hubble_param=float(data["hubble_param"]),
        max_candidate_galaxies=int(data["max_candidate_galaxies"]),
        max_particles_per_galaxy=int(data["max_particles_per_galaxy"]),
    )


def sync_include_paths() -> tuple[str, ...]:
    return (
        "configs",
        "docs",
        "pyproject.toml",
        "README.md",
        "scripts",
        "src",
        "tests",
    )


def build_remote_script(
    *,
    remote_project_root: str,
    conda_env: str,
    commands: list[str],
) -> str:
    lines = [
        "set -e",
        "source /home/zli/.bashrc",
        f"conda activate {conda_env}",
        f"cd {remote_project_root}",
        *commands,
        "exit",
    ]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 5: Run the cluster config tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_cluster.py -q
```

Expected: PASS with `3 passed`.

- [ ] **Step 6: Commit**

Run:

```bash
git add configs/milestone2.cluster.toml src/dgdp/cluster.py tests/test_cluster.py
git commit -m "Add cluster configuration helpers"
```

## Task 2: Project-Local Cluster Bridge CLI

**Files:**
- Create: `scripts/cluster_dgdp.py`
- Create: `tests/test_cluster_dgdp.py`

- [ ] **Step 1: Write CLI command-construction tests**

Create `tests/test_cluster_dgdp.py`:

```python
from pathlib import Path

from scripts.cluster_dgdp import build_rsync_push_command, build_rsync_fetch_command


def test_build_rsync_push_command_excludes_large_paths():
    cmd = build_rsync_push_command(
        project_root=Path("/local/dgdp"),
        cluster_host="gravity-login01",
        remote_project_root="/home/zli/disk-galaxy-deprojection",
    )

    joined = " ".join(cmd)
    assert "rsync" in cmd[0]
    assert "--exclude=.git" in cmd
    assert "--exclude=.venv" in cmd
    assert "--exclude=outputs" in cmd
    assert "/local/dgdp/" in joined
    assert "gravity-login01:/home/zli/disk-galaxy-deprojection/" in joined


def test_build_rsync_fetch_command_fetches_only_outputs():
    cmd = build_rsync_fetch_command(
        cluster_host="gravity-login01",
        remote_project_root="/home/zli/disk-galaxy-deprojection",
        remote_output_dir="outputs/tng50_milestone2",
        local_fetch_dir=Path("outputs/tng50_milestone2"),
    )

    joined = " ".join(cmd)
    assert "gravity-login01:/home/zli/disk-galaxy-deprojection/outputs/tng50_milestone2/" in joined
    assert "outputs/tng50_milestone2/" in joined
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_cluster_dgdp.py -q
```

Expected: FAIL because `scripts.cluster_dgdp` does not exist.

- [ ] **Step 3: Implement the cluster bridge CLI**

Create `scripts/cluster_dgdp.py`:

```python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dgdp.cluster import build_remote_script, load_cluster_config


def build_rsync_push_command(
    *,
    project_root: Path,
    cluster_host: str,
    remote_project_root: str,
) -> list[str]:
    return [
        "rsync",
        "-av",
        "--delete",
        "--exclude=.git",
        "--exclude=.pytest_cache",
        "--exclude=.ruff_cache",
        "--exclude=.venv",
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "--exclude=outputs",
        f"{project_root}/",
        f"{cluster_host}:{remote_project_root}/",
    ]


def build_rsync_fetch_command(
    *,
    cluster_host: str,
    remote_project_root: str,
    remote_output_dir: str,
    local_fetch_dir: Path,
) -> list[str]:
    return [
        "rsync",
        "-av",
        f"{cluster_host}:{remote_project_root}/{remote_output_dir}/",
        f"{local_fetch_dir}/",
    ]


def run_remote(config_path: Path, commands: list[str]) -> int:
    cfg = load_cluster_config(config_path)
    script = build_remote_script(
        remote_project_root=cfg.remote_project_root,
        conda_env=cfg.conda_env,
        commands=commands,
    )
    proc = subprocess.run([str(cfg.hpc_wrapper), "shell"], input=script, text=True, check=False)
    return int(proc.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/milestone2.cluster.toml"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sync")
    sub.add_parser("check-env")
    sub.add_parser("reproduce-milestone1")
    sub.add_parser("fetch-tng50")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("remote_command", nargs="+")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_cluster_config(args.config)
    project_root = Path(__file__).resolve().parents[1]

    if args.command == "sync":
        cmd = build_rsync_push_command(
            project_root=project_root,
            cluster_host=cfg.cluster_host,
            remote_project_root=cfg.remote_project_root,
        )
        return subprocess.run(cmd, check=False).returncode

    if args.command == "check-env":
        return run_remote(
            args.config,
            ["python -c \"import numpy, pandas, h5py, torch; print('cluster python ok')\""],
        )

    if args.command == "reproduce-milestone1":
        return run_remote(
            args.config,
            [
                f"python scripts/build_synthetic_benchmark.py --config configs/milestone1.synthetic.toml --output-dir {cfg.remote_milestone1_output_dir}",
                f"python scripts/train_summary_residual_mdn.py --data {cfg.remote_milestone1_output_dir}/residual_table.npz --output-dir {cfg.remote_milestone1_output_dir}",
                f"python scripts/evaluate_summary_residual.py --run-dir {cfg.remote_milestone1_output_dir}",
                f"python -m json.tool {cfg.remote_milestone1_output_dir}/metrics.json",
            ],
        )

    if args.command == "fetch-tng50":
        Path(cfg.local_fetch_dir).mkdir(parents=True, exist_ok=True)
        cmd = build_rsync_fetch_command(
            cluster_host=cfg.cluster_host,
            remote_project_root=cfg.remote_project_root,
            remote_output_dir=cfg.remote_output_dir,
            local_fetch_dir=Path(cfg.local_fetch_dir),
        )
        return subprocess.run(cmd, check=False).returncode

    if args.command == "run":
        return run_remote(args.config, [" ".join(args.remote_command)])

    raise RuntimeError(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the CLI tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_cluster_dgdp.py -q
```

Expected: PASS with `2 passed`.

- [ ] **Step 5: Run CLI help smoke test**

Run:

```bash
.venv/bin/python scripts/cluster_dgdp.py --help
```

Expected: exit code 0 and help text listing `sync`, `check-env`, `reproduce-milestone1`, `fetch-tng50`, and `run`.

- [ ] **Step 6: Commit**

Run:

```bash
git add scripts/cluster_dgdp.py tests/test_cluster_dgdp.py
git commit -m "Add DGDP cluster bridge CLI"
```

## Task 3: Reproduce Milestone 1 On The Cluster

**Files:**
- Modify: `docs/reports/milestone2_tng50_ingestion.md`

- [ ] **Step 1: Sync DGDP to the cluster**

Run:

```bash
.venv/bin/python scripts/cluster_dgdp.py sync
```

Expected: rsync completes without copying `.venv`, `.git`, or `outputs`.

- [ ] **Step 2: Check the remote Python environment**

Run:

```bash
.venv/bin/python scripts/cluster_dgdp.py check-env
```

Expected: prints `cluster python ok`.

If this fails because `conda activate paicos-conda` is not available in non-interactive shells, update `build_remote_script()` in `src/dgdp/cluster.py` to run:

```bash
source /home/zli/.bashrc
conda activate paicos-conda
```

and re-run `tests/test_cluster.py`.

- [ ] **Step 3: Reproduce Milestone 1 remotely**

Run:

```bash
.venv/bin/python scripts/cluster_dgdp.py reproduce-milestone1
```

Expected: remote synthetic benchmark writes:

```text
/home/zli/disk-galaxy-deprojection/outputs/cluster_milestone1/metrics.json
```

and prints finite values for `baseline_mae`, `corrected_mae`, `coverage_68`, and `n_test`.

- [ ] **Step 4: Fetch the cluster Milestone 1 output**

Run:

```bash
mkdir -p outputs/cluster_milestone1
rsync -av gravity-login01:/home/zli/disk-galaxy-deprojection/outputs/cluster_milestone1/ outputs/cluster_milestone1/
```

Expected: local `outputs/cluster_milestone1/metrics.json` exists.

- [ ] **Step 5: Create the report skeleton**

Create `docs/reports/milestone2_tng50_ingestion.md`:

```markdown
# Milestone 2 TNG50 Ingestion

## Scope

This milestone runs DGDP on the remote cluster, reproduces the Milestone 1
synthetic benchmark there, and then ingests a tiny selected TNG50-1 z=0 stellar
particle sample into the existing summary-residual benchmark.

## Cluster Milestone 1 Gate

- Remote project root: `/home/zli/disk-galaxy-deprojection`
- Remote environment: `paicos-conda`
- Metrics file: `outputs/cluster_milestone1/metrics.json`

The exact metric values are recorded after the remote run.

## TNG50 Ingestion

The first TNG50 ingestion uses `groups_099`, optional bar-catalog fields, and
selected `PartType4` particles from `snapdir_099`. Full TNG50 snapshots are not
copied locally.
```

- [ ] **Step 6: Commit**

Run:

```bash
git add docs/reports/milestone2_tng50_ingestion.md
git commit -m "Record cluster milestone gate"
```

## Task 4: TNG50 Group Catalog Reader And Candidate Manifest

**Files:**
- Create: `src/dgdp/tng50_catalog.py`
- Create: `tests/test_tng50_catalog.py`

- [ ] **Step 1: Write group-catalog fixture tests**

Create `tests/test_tng50_catalog.py`:

```python
from pathlib import Path

import h5py
import numpy as np

from dgdp.tng50_catalog import build_candidate_manifest, read_group_catalog


def _write_group_chunk(path: Path, first_subs: list[int], subhalo_offset: int) -> None:
    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        group = handle.create_group("Group")
        subhalo = handle.create_group("Subhalo")
        header.attrs["HubbleParam"] = 0.6774
        group["GroupFirstSub"] = np.asarray(first_subs, dtype=np.int64)
        n_sub = 3
        lens = np.zeros((n_sub, 6), dtype=np.int64)
        lens[:, 4] = np.asarray([5000, 100, 7000], dtype=np.int64)
        masses = np.zeros((n_sub, 6), dtype=np.float32)
        masses[:, 4] = np.asarray([1.0, 0.01, 2.0], dtype=np.float32)
        subhalo["SubhaloLenType"] = lens
        subhalo["SubhaloMassType"] = masses
        subhalo["SubhaloPos"] = np.arange(n_sub * 3, dtype=np.float32).reshape(n_sub, 3)
        subhalo["SubhaloVel"] = np.zeros((n_sub, 3), dtype=np.float32)
        subhalo["SubhaloHalfmassRadType"] = np.ones((n_sub, 6), dtype=np.float32)
        subhalo["SubhaloGrNr"] = np.arange(subhalo_offset, subhalo_offset + n_sub)


def test_read_group_catalog_concatenates_subhalos(tmp_path):
    group_dir = tmp_path / "groups_099"
    group_dir.mkdir()
    _write_group_chunk(group_dir / "fof_subhalo_tab_099.0.hdf5", [0], 0)
    _write_group_chunk(group_dir / "fof_subhalo_tab_099.1.hdf5", [3], 3)

    catalog = read_group_catalog(group_dir, snapshot=99)

    assert catalog.hubble_param == 0.6774
    assert catalog.subhalo_len_type.shape == (6, 6)
    assert catalog.subhalo_mass_type.shape == (6, 6)
    assert catalog.central_subhalo_ids.tolist() == [0, 3]


def test_build_candidate_manifest_filters_centrals_and_particle_count(tmp_path):
    group_dir = tmp_path / "groups_099"
    group_dir.mkdir()
    _write_group_chunk(group_dir / "fof_subhalo_tab_099.0.hdf5", [0], 0)

    catalog = read_group_catalog(group_dir, snapshot=99)
    manifest = build_candidate_manifest(
        catalog,
        min_star_particles=1000,
        min_stellar_mass_msun=1.0e9,
        max_candidates=2,
    )

    assert manifest["subhalo_id"].tolist() == [2, 0]
    assert manifest["split"].tolist() == ["train", "val"]
    assert manifest["star_particles"].tolist() == [7000, 5000]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_tng50_catalog.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'dgdp.tng50_catalog'`.

- [ ] **Step 3: Implement the catalog reader**

Create `src/dgdp/tng50_catalog.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TNG50GroupCatalog:
    hubble_param: float
    subhalo_len_type: np.ndarray
    subhalo_mass_type: np.ndarray
    subhalo_pos: np.ndarray
    subhalo_vel: np.ndarray
    subhalo_halfmass_rad_type: np.ndarray
    subhalo_grnr: np.ndarray
    central_subhalo_ids: np.ndarray


def _group_files(group_dir: Path, snapshot: int) -> list[Path]:
    return sorted(group_dir.glob(f"fof_subhalo_tab_{snapshot:03d}.*.hdf5"))


def read_group_catalog(group_dir: Path, *, snapshot: int = 99) -> TNG50GroupCatalog:
    files = _group_files(group_dir, snapshot)
    if not files:
        raise FileNotFoundError(f"no group catalog files found in {group_dir}")

    len_parts = []
    mass_parts = []
    pos_parts = []
    vel_parts = []
    halfmass_parts = []
    grnr_parts = []
    central_ids = []
    subhalo_offset = 0
    hubble = 0.6774

    for path in files:
        with h5py.File(path, "r") as handle:
            hubble = float(handle["Header"].attrs.get("HubbleParam", hubble))
            group_first_sub = np.asarray(handle["Group/GroupFirstSub"], dtype=np.int64)
            valid = group_first_sub[group_first_sub >= 0] + subhalo_offset
            central_ids.append(valid)

            subhalo_len = np.asarray(handle["Subhalo/SubhaloLenType"], dtype=np.int64)
            len_parts.append(subhalo_len)
            mass_parts.append(np.asarray(handle["Subhalo/SubhaloMassType"], dtype=np.float64))
            pos_parts.append(np.asarray(handle["Subhalo/SubhaloPos"], dtype=np.float64))
            vel_parts.append(np.asarray(handle["Subhalo/SubhaloVel"], dtype=np.float64))
            halfmass_parts.append(
                np.asarray(handle["Subhalo/SubhaloHalfmassRadType"], dtype=np.float64)
            )
            grnr_parts.append(np.asarray(handle["Subhalo/SubhaloGrNr"], dtype=np.int64))
            subhalo_offset += subhalo_len.shape[0]

    return TNG50GroupCatalog(
        hubble_param=hubble,
        subhalo_len_type=np.vstack(len_parts),
        subhalo_mass_type=np.vstack(mass_parts),
        subhalo_pos=np.vstack(pos_parts),
        subhalo_vel=np.vstack(vel_parts),
        subhalo_halfmass_rad_type=np.vstack(halfmass_parts),
        subhalo_grnr=np.concatenate(grnr_parts),
        central_subhalo_ids=np.concatenate(central_ids) if central_ids else np.array([], dtype=int),
    )


def _split_for_rank(rank: int, n_rows: int) -> str:
    train_end = int(round(0.60 * n_rows))
    val_end = int(round(0.80 * n_rows))
    if rank < train_end:
        return "train"
    if rank < val_end:
        return "val"
    return "test"


def build_candidate_manifest(
    catalog: TNG50GroupCatalog,
    *,
    min_star_particles: int,
    min_stellar_mass_msun: float,
    max_candidates: int,
) -> pd.DataFrame:
    subhalo_id = np.arange(catalog.subhalo_len_type.shape[0], dtype=int)
    star_particles = catalog.subhalo_len_type[:, 4].astype(int)
    stellar_mass_msun = catalog.subhalo_mass_type[:, 4] * 1.0e10 / catalog.hubble_param
    central = np.isin(subhalo_id, catalog.central_subhalo_ids)
    keep = (
        central
        & (star_particles >= min_star_particles)
        & (stellar_mass_msun >= min_stellar_mass_msun)
    )
    rows = pd.DataFrame(
        {
            "subhalo_id": subhalo_id[keep],
            "snapshot": 99,
            "star_particles": star_particles[keep],
            "stellar_mass_msun": stellar_mass_msun[keep],
            "subhalo_grnr": catalog.subhalo_grnr[keep],
            "subhalo_pos_x_ckpc_h": catalog.subhalo_pos[keep, 0],
            "subhalo_pos_y_ckpc_h": catalog.subhalo_pos[keep, 1],
            "subhalo_pos_z_ckpc_h": catalog.subhalo_pos[keep, 2],
        }
    )
    rows = rows.sort_values("stellar_mass_msun", ascending=False).head(max_candidates)
    rows = rows.reset_index(drop=True)
    rows["split"] = [_split_for_rank(i, len(rows)) for i in range(len(rows))]
    return rows
```

- [ ] **Step 4: Run the catalog tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_tng50_catalog.py -q
```

Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/dgdp/tng50_catalog.py tests/test_tng50_catalog.py
git commit -m "Add TNG50 group catalog reader"
```

## Task 5: Optional Bar Catalog Inspection And Join

**Files:**
- Create: `src/dgdp/bar_catalog.py`
- Create: `tests/test_bar_catalog.py`

- [ ] **Step 1: Write bar catalog tests**

Create `tests/test_bar_catalog.py`:

```python
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from dgdp.bar_catalog import inspect_hdf5_datasets, join_bar_catalog


def test_inspect_hdf5_datasets_lists_nested_data(tmp_path):
    path = tmp_path / "bar.hdf5"
    with h5py.File(path, "w") as handle:
        handle["Catalog/SubfindID"] = np.array([1, 2])
        handle["Catalog/Bartype"] = np.array([1, 0])

    rows = inspect_hdf5_datasets(path)

    assert ("Catalog/SubfindID", (2,), "int64") in rows
    assert ("Catalog/Bartype", (2,), "int64") in rows


def test_join_bar_catalog_adds_bar_fields(tmp_path):
    path = tmp_path / "bar.hdf5"
    with h5py.File(path, "w") as handle:
        handle["SubfindID"] = np.array([10, 20])
        handle["Bartype"] = np.array([1, 0])
        handle["A2max"] = np.array([0.45, 0.10])

    manifest = pd.DataFrame({"subhalo_id": [10, 20], "split": ["train", "test"]})
    joined = join_bar_catalog(manifest, path)

    assert joined["bar_type"].tolist() == [1, 0]
    assert joined["bar_a2_catalog"].tolist() == [0.45, 0.10]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_bar_catalog.py -q
```

Expected: FAIL with missing `dgdp.bar_catalog`.

- [ ] **Step 3: Implement bar catalog helpers**

Create `src/dgdp/bar_catalog.py`:

```python
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def inspect_hdf5_datasets(path: Path) -> list[tuple[str, tuple[int, ...], str]]:
    rows: list[tuple[str, tuple[int, ...], str]] = []

    def visit(name: str, obj: h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            rows.append((name, tuple(obj.shape), str(obj.dtype)))

    with h5py.File(path, "r") as handle:
        handle.visititems(visit)
    return rows


def _read_dataset_by_candidates(handle: h5py.File, names: tuple[str, ...]) -> np.ndarray | None:
    datasets = {name.lower().split("/")[-1]: name for name, _, _ in inspect_hdf5_datasets(Path(handle.filename))}
    for candidate in names:
        key = candidate.lower()
        if key in datasets:
            return np.asarray(handle[datasets[key]])
    return None


def join_bar_catalog(manifest: pd.DataFrame, path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as handle:
        subfind_id = _read_dataset_by_candidates(
            handle, ("SubfindID", "SubhaloID", "SubhaloIndex", "subhalo_id")
        )
        bar_type = _read_dataset_by_candidates(handle, ("Bartype", "BarType", "bar_type"))
        a2max = _read_dataset_by_candidates(handle, ("A2max", "A2Max", "bar_a2", "BarA2"))

    if subfind_id is None:
        raise ValueError("bar catalog does not contain a recognized subhalo identifier field")

    bar_df = pd.DataFrame({"subhalo_id": np.asarray(subfind_id, dtype=int)})
    if bar_type is not None:
        bar_df["bar_type"] = np.asarray(bar_type, dtype=int)
    if a2max is not None:
        bar_df["bar_a2_catalog"] = np.asarray(a2max, dtype=float)
    if "bar_type" not in bar_df and "bar_a2_catalog" not in bar_df:
        raise ValueError("bar catalog does not contain a recognized bar label or bar strength field")

    return manifest.merge(bar_df, on="subhalo_id", how="left")
```

- [ ] **Step 4: Run the bar catalog tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_bar_catalog.py -q
```

Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/dgdp/bar_catalog.py tests/test_bar_catalog.py
git commit -m "Add optional bar catalog helpers"
```

## Task 6: Offset-Based Stellar Particle Extraction

**Files:**
- Modify: `src/dgdp/tng50.py`
- Modify: `tests/test_tng50.py`

- [ ] **Step 1: Extend TNG particle tests**

Append to `tests/test_tng50.py`:

```python
from dgdp.tng50 import load_subhalo_stars_from_chunks, write_particle_set_hdf5
from dgdp.types import ParticleSet


def test_load_subhalo_stars_from_chunks_reads_across_files(tmp_path):
    snap_dir = tmp_path / "snapdir_099"
    snap_dir.mkdir()
    offsets = tmp_path / "offsets_099.hdf5"

    with h5py.File(offsets, "w") as handle:
        subhalo = handle.create_group("Subhalo")
        data = np.zeros((2, 6), dtype=np.int64)
        data[1, 4] = 1
        subhalo["SnapByType"] = data

    coords0 = np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]])
    coords1 = np.array([[12.0, 0.0, 0.0], [13.0, 0.0, 0.0]])
    for chunk, coords in enumerate([coords0, coords1]):
        with h5py.File(snap_dir / f"snap_099.{chunk}.hdf5", "w") as handle:
            header = handle.create_group("Header")
            stars = handle.create_group("PartType4")
            header.attrs["NumPart_ThisFile"] = np.array([0, 0, 0, 0, len(coords), 0])
            header.attrs["HubbleParam"] = 0.5
            stars["Coordinates"] = coords
            stars["Masses"] = np.ones(len(coords))
            stars["Velocities"] = np.zeros((len(coords), 3))
            stars["GFM_StellarFormationTime"] = np.ones(len(coords))
            stars["ParticleIDs"] = np.arange(chunk * 10, chunk * 10 + len(coords))

    particles = load_subhalo_stars_from_chunks(
        snap_dir=snap_dir,
        offsets_path=offsets,
        subhalo_id=1,
        star_particle_count=3,
        subhalo_center_ckpc_h=np.array([10.0, 0.0, 0.0]),
        snapshot=99,
        hubble_param=0.5,
        max_particles=10,
    )

    assert particles.positions_kpc.shape == (3, 3)
    assert particles.masses_msun.tolist() == [2.0e10, 2.0e10, 2.0e10]
    assert np.allclose(particles.positions_kpc[:, 0], [2.0, 4.0, 6.0])


def test_write_particle_set_hdf5_round_trips_velocities(tmp_path):
    path = tmp_path / "compact.hdf5"
    particles = ParticleSet(
        positions_kpc=np.ones((2, 3)),
        masses_msun=np.array([1.0, 2.0]),
        velocities_kms=np.zeros((2, 3)),
    )

    write_particle_set_hdf5(path, particles, subhalo_id=42)
    loaded = load_particle_set_hdf5(path, length_unit_kpc=1.0, mass_unit_msun=1.0)

    assert loaded.positions_kpc.shape == (2, 3)
    assert loaded.velocities_kms.shape == (2, 3)
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_tng50.py -q
```

Expected: FAIL because the new functions are missing.

- [ ] **Step 3: Implement extraction and compact writing**

Modify `src/dgdp/tng50.py` to add:

```python
def _snapshot_files(snap_dir: Path, snapshot: int) -> list[Path]:
    return sorted(snap_dir.glob(f"snap_{snapshot:03d}.*.hdf5"))


def _star_counts_by_file(files: list[Path]) -> list[int]:
    counts = []
    for path in files:
        with h5py.File(path, "r") as handle:
            counts.append(int(handle["Header"].attrs["NumPart_ThisFile"][4]))
    return counts


def _read_optional_dataset(group: h5py.Group, name: str, selection: slice) -> np.ndarray | None:
    if name not in group:
        return None
    return np.asarray(group[name][selection])


def load_subhalo_stars_from_chunks(
    *,
    snap_dir: Path,
    offsets_path: Path,
    subhalo_id: int,
    star_particle_count: int,
    subhalo_center_ckpc_h: np.ndarray,
    snapshot: int,
    hubble_param: float,
    max_particles: int,
) -> ParticleSet:
    with h5py.File(offsets_path, "r") as handle:
        start = int(handle["Subhalo/SnapByType"][subhalo_id, 4])
    length = int(star_particle_count)
    length = min(length, int(max_particles))

    files = _snapshot_files(snap_dir, snapshot)
    counts = _star_counts_by_file(files)
    remaining_start = start
    remaining_length = length
    coords_parts = []
    mass_parts = []
    vel_parts = []
    formation_parts = []

    for path, count in zip(files, counts):
        if remaining_start >= count:
            remaining_start -= count
            continue
        if remaining_length <= 0:
            break
        local_start = remaining_start
        take = min(count - local_start, remaining_length)
        selection = slice(local_start, local_start + take)
        with h5py.File(path, "r") as handle:
            stars = handle["PartType4"]
            coords_parts.append(np.asarray(stars["Coordinates"][selection], dtype=float))
            mass_parts.append(np.asarray(stars["Masses"][selection], dtype=float))
            vel = _read_optional_dataset(stars, "Velocities", selection)
            form = _read_optional_dataset(stars, "GFM_StellarFormationTime", selection)
            if vel is not None:
                vel_parts.append(vel.astype(float))
            if form is not None:
                formation_parts.append(form.astype(float))
        remaining_length -= take
        remaining_start = 0

    if not coords_parts:
        return ParticleSet(positions_kpc=np.empty((0, 3)), masses_msun=np.empty((0,)))

    coords = np.vstack(coords_parts)
    masses = np.concatenate(mass_parts)
    velocities = np.vstack(vel_parts) if vel_parts else None
    if formation_parts:
        formed = np.concatenate(formation_parts) > 0.0
        coords = coords[formed]
        masses = masses[formed]
        if velocities is not None:
            velocities = velocities[formed]

    positions_kpc = (coords - np.asarray(subhalo_center_ckpc_h, dtype=float)) / hubble_param
    masses_msun = masses * 1.0e10 / hubble_param
    return ParticleSet(positions_kpc=positions_kpc, masses_msun=masses_msun, velocities_kms=velocities)


def write_particle_set_hdf5(path: Path, particles: ParticleSet, *, subhalo_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["subhalo_id"] = int(subhalo_id)
        stars = handle.create_group("PartType4")
        stars["Coordinates"] = particles.positions_kpc
        stars["Masses"] = particles.masses_msun
        if particles.velocities_kms is not None:
            stars["Velocities"] = particles.velocities_kms
```

Also update `load_particle_set_hdf5()` so it reads optional velocities:

```python
velocities = (
    np.asarray(handle["PartType4/Velocities"], dtype=float)
    if "PartType4/Velocities" in handle
    else None
)
return ParticleSet(positions_kpc=coords, masses_msun=masses, velocities_kms=velocities)
```

- [ ] **Step 4: Run the TNG particle tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_tng50.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/dgdp/tng50.py tests/test_tng50.py
git commit -m "Add TNG50 stellar particle extraction"
```

## Task 7: TNG50 Manifest Script

**Files:**
- Create: `scripts/build_tng50_manifest.py`
- Create: `tests/test_build_tng50_manifest.py`

- [ ] **Step 1: Write script smoke test**

Create `tests/test_build_tng50_manifest.py`:

```python
import subprocess
import sys

import h5py
import numpy as np

def test_build_tng50_manifest_writes_csv(tmp_path):
    tng_root = tmp_path / "TNG50-1"
    group_dir = tng_root / "groups_099"
    group_dir.mkdir(parents=True)

    def write_group_chunk(path):
        with h5py.File(path, "w") as handle:
            header = handle.create_group("Header")
            group = handle.create_group("Group")
            subhalo = handle.create_group("Subhalo")
            header.attrs["HubbleParam"] = 0.6774
            group["GroupFirstSub"] = np.array([0])
            lens = np.zeros((3, 6), dtype=np.int64)
            lens[:, 4] = np.array([5000, 100, 7000])
            masses = np.zeros((3, 6), dtype=np.float32)
            masses[:, 4] = np.array([1.0, 0.01, 2.0])
            subhalo["SubhaloLenType"] = lens
            subhalo["SubhaloMassType"] = masses
            subhalo["SubhaloPos"] = np.zeros((3, 3), dtype=np.float32)
            subhalo["SubhaloVel"] = np.zeros((3, 3), dtype=np.float32)
            subhalo["SubhaloHalfmassRadType"] = np.ones((3, 6), dtype=np.float32)
            subhalo["SubhaloGrNr"] = np.arange(3)

    write_group_chunk(group_dir / "fof_subhalo_tab_099.0.hdf5")
    output = tmp_path / "manifest.csv"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_tng50_manifest.py",
            "--tng-root",
            str(tng_root),
            "--output",
            str(output),
            "--max-candidates",
            "2",
            "--min-star-particles",
            "1000",
            "--min-stellar-mass-msun",
            "1e9",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote TNG50 manifest" in result.stdout
    assert output.exists()
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_build_tng50_manifest.py -q
```

Expected: FAIL because the script does not exist.

- [ ] **Step 3: Implement manifest script**

Create `scripts/build_tng50_manifest.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from dgdp.bar_catalog import join_bar_catalog
from dgdp.tng50_catalog import build_candidate_manifest, read_group_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tng-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=int, default=99)
    parser.add_argument("--bar-catalog", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--min-star-particles", type=int, default=5000)
    parser.add_argument("--min-stellar-mass-msun", type=float, default=1.0e9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = read_group_catalog(args.tng_root / f"groups_{args.snapshot:03d}", snapshot=args.snapshot)
    manifest = build_candidate_manifest(
        catalog,
        min_star_particles=args.min_star_particles,
        min_stellar_mass_msun=args.min_stellar_mass_msun,
        max_candidates=args.max_candidates,
    )
    if args.bar_catalog is not None and args.bar_catalog.exists():
        manifest = join_bar_catalog(manifest, args.bar_catalog)
        if "bar_type" in manifest:
            manifest = manifest[manifest["bar_type"].fillna(0).astype(int) > 0]
        elif "bar_a2_catalog" in manifest:
            manifest = manifest[manifest["bar_a2_catalog"].fillna(0.0).astype(float) > 0.2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"wrote TNG50 manifest to {args.output} with {len(manifest)} rows")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the smoke test**

Run:

```bash
.venv/bin/python -m pytest tests/test_build_tng50_manifest.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add scripts/build_tng50_manifest.py tests/test_build_tng50_manifest.py
git commit -m "Add TNG50 manifest script"
```

## Task 8: TNG50 Particle Extraction Script

**Files:**
- Create: `scripts/extract_tng50_particles.py`
- Create: `tests/test_extract_tng50_particles.py`

- [ ] **Step 1: Write extraction script smoke test**

Create `tests/test_extract_tng50_particles.py`:

```python
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd


def test_extract_tng50_particles_writes_compact_files(tmp_path):
    tng_root = tmp_path / "TNG50-1"
    snap_dir = tng_root / "snapdir_099"
    offsets_dir = tng_root / "postprocessing" / "offsets"
    snap_dir.mkdir(parents=True)
    offsets_dir.mkdir(parents=True)
    with h5py.File(offsets_dir / "offsets_099.hdf5", "w") as handle:
        subhalo = handle.create_group("Subhalo")
        offsets = np.zeros((1, 6), dtype=np.int64)
        subhalo["SnapByType"] = offsets
    with h5py.File(snap_dir / "snap_099.0.hdf5", "w") as handle:
        header = handle.create_group("Header")
        stars = handle.create_group("PartType4")
        header.attrs["NumPart_ThisFile"] = np.array([0, 0, 0, 0, 2, 0])
        stars["Coordinates"] = np.ones((2, 3))
        stars["Masses"] = np.ones(2)
        stars["Velocities"] = np.zeros((2, 3))
        stars["GFM_StellarFormationTime"] = np.ones(2)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "subhalo_id": [0],
            "subhalo_pos_x_ckpc_h": [0.0],
            "subhalo_pos_y_ckpc_h": [0.0],
            "subhalo_pos_z_ckpc_h": [0.0],
            "star_particles": [2],
            "split": ["train"],
        }
    ).to_csv(manifest, index=False)
    output_dir = tmp_path / "particles"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/extract_tng50_particles.py",
            "--tng-root",
            str(tng_root),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--max-particles-per-galaxy",
            "10",
            "--hubble-param",
            "1.0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote 1 compact particle files" in result.stdout
    assert (output_dir / "subhalo_0.hdf5").exists()
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_extract_tng50_particles.py -q
```

Expected: FAIL because the script does not exist.

- [ ] **Step 3: Implement extraction script**

Create `scripts/extract_tng50_particles.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dgdp.tng50 import load_subhalo_stars_from_chunks, write_particle_set_hdf5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tng-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=int, default=99)
    parser.add_argument("--max-particles-per-galaxy", type=int, default=80000)
    parser.add_argument("--hubble-param", type=float, default=0.6774)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    wrote = 0
    for row in manifest.itertuples(index=False):
        subhalo_id = int(row.subhalo_id)
        center = np.array(
            [
                float(row.subhalo_pos_x_ckpc_h),
                float(row.subhalo_pos_y_ckpc_h),
                float(row.subhalo_pos_z_ckpc_h),
            ]
        )
        particles = load_subhalo_stars_from_chunks(
            snap_dir=args.tng_root / f"snapdir_{args.snapshot:03d}",
            offsets_path=args.tng_root / "postprocessing" / "offsets" / f"offsets_{args.snapshot:03d}.hdf5",
            subhalo_id=subhalo_id,
            star_particle_count=int(row.star_particles),
            subhalo_center_ckpc_h=center,
            snapshot=args.snapshot,
            hubble_param=args.hubble_param,
            max_particles=args.max_particles_per_galaxy,
        )
        write_particle_set_hdf5(output_dir / f"subhalo_{subhalo_id}.hdf5", particles, subhalo_id=subhalo_id)
        wrote += 1

    print(f"wrote {wrote} compact particle files to {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run extraction script tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_extract_tng50_particles.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add scripts/extract_tng50_particles.py tests/test_extract_tng50_particles.py
git commit -m "Add TNG50 particle extraction script"
```

## Task 9: TNG50 Benchmark Builder

**Files:**
- Create: `scripts/build_tng50_benchmark.py`
- Create: `tests/test_build_tng50_benchmark.py`

- [ ] **Step 1: Write benchmark builder smoke test**

Create `tests/test_build_tng50_benchmark.py`:

```python
import subprocess
import sys

import numpy as np
import pandas as pd

from dgdp.tng50 import write_particle_set_hdf5
from dgdp.types import ParticleSet


def test_build_tng50_benchmark_writes_residual_table(tmp_path):
    particle_dir = tmp_path / "particles"
    particle_dir.mkdir()
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "subhalo_id": [0],
            "projection_id": [0],
            "split": ["train"],
            "inclination_deg": [0.0],
            "disk_pa_deg": [0.0],
            "bar_angle_deg": [0.0],
        }
    ).to_csv(manifest, index=False)
    particles = ParticleSet(
        positions_kpc=np.random.default_rng(1).normal(size=(2000, 3)),
        masses_msun=np.ones(2000) * 1.0e6,
    )
    write_particle_set_hdf5(particle_dir / "subhalo_0.hdf5", particles, subhalo_id=0)
    output_dir = tmp_path / "run"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_tng50_benchmark.py",
            "--config",
            "configs/milestone1.synthetic.toml",
            "--manifest",
            str(manifest),
            "--particle-dir",
            str(particle_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote TNG50 residual table" in result.stdout
    assert (output_dir / "residual_table.npz").exists()
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_build_tng50_benchmark.py -q
```

Expected: FAIL because the script does not exist.

- [ ] **Step 3: Implement TNG50 benchmark builder**

Create `scripts/build_tng50_benchmark.py` by adapting `scripts/build_synthetic_benchmark.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dgdp.baseline import baseline_particles_from_image
from dgdp.config import load_config
from dgdp.dataset import build_residual_row
from dgdp.projection import project_to_mock_image
from dgdp.summaries import extract_summary_vector
from dgdp.tng50 import load_particle_set_hdf5
from dgdp.types import Geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--particle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    manifest = pd.read_csv(args.manifest)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in manifest.itertuples(index=False):
        subhalo_id = int(row.subhalo_id)
        projection_id = int(getattr(row, "projection_id", 0))
        geometry = Geometry(
            inclination_deg=float(getattr(row, "inclination_deg", 0.0)),
            disk_pa_deg=float(getattr(row, "disk_pa_deg", 0.0)),
            bar_angle_deg=float(getattr(row, "bar_angle_deg", 0.0)),
        )
        particles = load_particle_set_hdf5(
            args.particle_dir / f"subhalo_{subhalo_id}.hdf5",
            length_unit_kpc=1.0,
            mass_unit_msun=1.0,
        )
        mock = project_to_mock_image(
            particles,
            geometry,
            cfg.image_size,
            cfg.pixel_scale_kpc,
            cfg.psf_sigma_pixels,
            cfg.noise_sigma_fraction,
            seed=cfg.seed + subhalo_id * 100 + projection_id,
        )
        baseline_particles = baseline_particles_from_image(mock, vertical_scale_height_kpc=0.4)
        true_summary = extract_summary_vector(
            particles,
            cfg.radial_bins_kpc,
            cfg.vertical_bins_kpc,
            bar_angle_deg=geometry.bar_angle_deg,
        )
        baseline_summary = extract_summary_vector(
            baseline_particles,
            cfg.radial_bins_kpc,
            cfg.vertical_bins_kpc,
            bar_angle_deg=geometry.bar_angle_deg,
        )
        rows.append(
            build_residual_row(
                galaxy_id=subhalo_id,
                projection_id=projection_id,
                split=str(row.split),
                image=mock.image,
                true_summary=true_summary,
                baseline_summary=baseline_summary,
                geometry=geometry,
            )
        )

    np.savez_compressed(
        output_dir / "residual_table.npz",
        images=np.stack([r["image"] for r in rows]),
        baseline=np.stack([r["baseline"] for r in rows]),
        truth=np.stack([r["truth"] for r in rows]),
        delta=np.stack([r["delta"] for r in rows]),
        metadata=np.stack([r["metadata"] for r in rows]),
        split=np.array([r["split"] for r in rows]),
        galaxy_id=np.array([r["galaxy_id"] for r in rows], dtype=int),
        projection_id=np.array([r["projection_id"] for r in rows], dtype=int),
        summary_names=np.array(rows[0]["summary_names"], dtype=str),
    )
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    print(f"wrote TNG50 residual table to {output_dir / 'residual_table.npz'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run benchmark builder tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_build_tng50_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add scripts/build_tng50_benchmark.py tests/test_build_tng50_benchmark.py
git commit -m "Add TNG50 benchmark builder"
```

## Task 10: Remote TNG50 Tiny Benchmark Run

**Files:**
- Modify: `scripts/cluster_dgdp.py`
- Modify: `tests/test_cluster_dgdp.py`
- Modify: `docs/reports/milestone2_tng50_ingestion.md`

- [ ] **Step 1: Extend cluster CLI tests for remote TNG command text**

Append to `tests/test_cluster_dgdp.py`:

```python
from scripts.cluster_dgdp import build_tng50_remote_commands


def test_build_tng50_remote_commands_include_manifest_extract_train_eval():
    commands = build_tng50_remote_commands(
        remote_tng50_root="/home/cossim/IllustrisTNG/TNG50-1",
        remote_output_dir="outputs/tng50_milestone2",
        bar_catalog_path="",
        max_candidate_galaxies=4,
        max_particles_per_galaxy=1000,
        hubble_param=0.6774,
    )
    text = "\n".join(commands)

    assert "scripts/build_tng50_manifest.py" in text
    assert "scripts/extract_tng50_particles.py" in text
    assert "scripts/build_tng50_benchmark.py" in text
    assert "scripts/train_summary_residual_mdn.py" in text
    assert "scripts/evaluate_summary_residual.py" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_cluster_dgdp.py -q
```

Expected: FAIL because `build_tng50_remote_commands` is missing.

- [ ] **Step 3: Add TNG remote command construction and CLI subcommand**

Modify `scripts/cluster_dgdp.py`:

```python
def build_tng50_remote_commands(
    *,
    remote_tng50_root: str,
    remote_output_dir: str,
    bar_catalog_path: str,
    max_candidate_galaxies: int,
    max_particles_per_galaxy: int,
    hubble_param: float,
) -> list[str]:
    manifest = f"{remote_output_dir}/manifest.csv"
    particles = f"{remote_output_dir}/particles"
    bar_arg = f" --bar-catalog {bar_catalog_path}" if bar_catalog_path else ""
    return [
        f"mkdir -p {remote_output_dir}",
        (
            "python scripts/build_tng50_manifest.py "
            f"--tng-root {remote_tng50_root} "
            f"--output {manifest} "
            f"--max-candidates {max_candidate_galaxies}"
            f"{bar_arg}"
        ),
        (
            "python scripts/extract_tng50_particles.py "
            f"--tng-root {remote_tng50_root} "
            f"--manifest {manifest} "
            f"--output-dir {particles} "
            f"--max-particles-per-galaxy {max_particles_per_galaxy} "
            f"--hubble-param {hubble_param}"
        ),
        (
            "python scripts/build_tng50_benchmark.py "
            "--config configs/milestone1.synthetic.toml "
            f"--manifest {manifest} "
            f"--particle-dir {particles} "
            f"--output-dir {remote_output_dir}"
        ),
        (
            "python scripts/train_summary_residual_mdn.py "
            f"--data {remote_output_dir}/residual_table.npz "
            f"--output-dir {remote_output_dir} "
            "--epochs 10 --hidden-dim 32 --n-components 2"
        ),
        f"python scripts/evaluate_summary_residual.py --run-dir {remote_output_dir} --n-samples 32",
        f"python -m json.tool {remote_output_dir}/metrics.json",
    ]
```

Add `run-tng50` to `parse_args()`:

```python
sub.add_parser("run-tng50")
```

Add handler in `main()`:

```python
if args.command == "run-tng50":
    return run_remote(
        args.config,
        build_tng50_remote_commands(
            remote_tng50_root=cfg.remote_tng50_root,
            remote_output_dir=cfg.remote_output_dir,
            bar_catalog_path=cfg.bar_catalog_path,
            max_candidate_galaxies=cfg.max_candidate_galaxies,
            max_particles_per_galaxy=cfg.max_particles_per_galaxy,
            hubble_param=cfg.hubble_param,
        ),
    )
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_cluster_dgdp.py -q
```

Expected: PASS.

- [ ] **Step 5: Sync and run the tiny TNG benchmark remotely**

Run:

```bash
.venv/bin/python scripts/cluster_dgdp.py sync
.venv/bin/python scripts/cluster_dgdp.py run-tng50
```

Expected: remote `outputs/tng50_milestone2/metrics.json` exists and prints finite metrics.

- [ ] **Step 6: Fetch compact outputs**

Run:

```bash
.venv/bin/python scripts/cluster_dgdp.py fetch-tng50
```

Expected: local `outputs/tng50_milestone2/metrics.json` exists.

- [ ] **Step 7: Update report with observed metrics**

Modify `docs/reports/milestone2_tng50_ingestion.md` to add:

```markdown
## TNG50 Tiny Benchmark Metrics

- Manifest: `outputs/tng50_milestone2/manifest.csv`
- Residual table: `outputs/tng50_milestone2/residual_table.npz`
- Metrics: `outputs/tng50_milestone2/metrics.json`

The first run uses a tiny selected sample and is intended to validate ingestion,
not to make a scientific performance claim.
```

Include the exact metric values from `outputs/tng50_milestone2/metrics.json`.

- [ ] **Step 8: Commit**

Run:

```bash
git add scripts/cluster_dgdp.py tests/test_cluster_dgdp.py docs/reports/milestone2_tng50_ingestion.md
git commit -m "Run tiny TNG50 ingestion benchmark"
```

## Task 11: README And Final Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README with Milestone 2 cluster commands**

Append to `README.md`:

```markdown
## Milestone 2 Cluster TNG50 Ingestion

Milestone 2 runs TNG-facing work on the remote cluster through the existing HPC
wrapper at `/home/lucyundead/codex/hpc-agent/hpc`.

```bash
python scripts/cluster_dgdp.py sync
python scripts/cluster_dgdp.py check-env
python scripts/cluster_dgdp.py reproduce-milestone1
python scripts/cluster_dgdp.py run-tng50
python scripts/cluster_dgdp.py fetch-tng50
```

The cluster workflow keeps full TNG50 snapshots remote and fetches only compact
artifacts under `outputs/tng50_milestone2/`.
```

- [ ] **Step 2: Run full local tests**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Run lint**

Run:

```bash
.venv/bin/python -m ruff check .
```

Expected: no lint errors.

- [ ] **Step 4: Verify local git status**

Run:

```bash
git status --short
```

Expected: only intentional README/report changes, generated `outputs/` ignored.

- [ ] **Step 5: Commit final docs**

Run:

```bash
git add README.md
git commit -m "Document Milestone 2 cluster workflow"
```

## Completion Criteria

Milestone 2 implementation is complete when:

- Local tests pass.
- Local lint passes.
- `scripts/cluster_dgdp.py check-env` succeeds on the remote cluster.
- Milestone 1 synthetic benchmark is reproduced on the cluster.
- A TNG50 z=0 candidate manifest is created from `groups_099`.
- Compact per-subhalo stellar-particle files are extracted from `snapdir_099` using offsets.
- The existing summary-residual benchmark builds, trains, and evaluates on the tiny TNG50 sample.
- Local fetched `outputs/tng50_milestone2/metrics.json` exists.
- `docs/reports/milestone2_tng50_ingestion.md` states the run is an ingestion validation, not a final scientific claim.

## Follow-On Plan After This Milestone

After Milestone 2 passes:

1. Inspect the downloaded bar/morphology catalog fields and strengthen barred-galaxy selection.
2. Increase the TNG50 sample size only after tiny-sample extraction is stable.
3. Add disk/bar alignment from stellar angular momentum and Fourier `m=2` diagnostics.
4. Only then plan the coarse 3D residual milestone:

```text
p(delta_rho_3d | image_mock, rho_baseline_3d, geometry, metadata)
```
