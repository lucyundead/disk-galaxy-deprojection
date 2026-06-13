"""Cluster orchestration for the scoped milestone-2d z-rebin validation.

Re-bins ALL galaxies' truth density at |z|<5 kpc / 0.3125 kpc into a new remote
output dir (leaving the production 0.625 kpc data untouched), rebuilds the
baseline grids and the density residual table, so the adopted MDN config can be
retrained at the finer resolution to test recovery of the boxy/peanut X.

Drives the tested cluster_dgdp.run_remote path with explicit command lists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cluster_dgdp import run_remote  # noqa: E402

CONFIG = Path("configs/milestone2c.cluster.toml")
SOURCE = "outputs/tng50_milestone2c"  # manifest + particles
CLEAN = "outputs/tng50_milestone2c_clean3d"  # images live in its residual_table.npz
Z5 = "outputs/tng50_milestone2c_z5"  # new finer-z output
CONFIG_TOML = "configs/milestone2b.clean3d.toml"
TNG_ROOT = "/home/cossim/IllustrisTNG/TNG50-1"


def probe() -> list[str]:
    return [
        "echo STATE_START",
        f"ls -la {SOURCE}/manifest.csv {SOURCE}/residual_table.npz {CLEAN}/residual_table.npz 2>&1 | cat",
        f"(ls {SOURCE}/particles/*.hdf5 | wc -l)",
        (
            "python - <<'PY'\n"
            "import numpy as np\n"
            "for p in ['%s/residual_table.npz','%s/residual_table.npz']:\n"
            "    try:\n"
            "        t=np.load(p)\n"
            "        print(p, 'images' in t.files, t['images'].shape if 'images' in t.files else None, len(t['split']))\n"
            "    except Exception as e:\n"
            "        print(p, 'ERR', e)\n"
            "PY" % (SOURCE, CLEAN)
        ),
        "echo STATE_END",
    ]


def rebin_all(z_max: float, n_z: int) -> list[str]:
    out = f"{Z5}/density_grids_logr_cyl"
    return [
        f"mkdir -p {out}",
        (
            "python scripts/build_tng50_density_grid.py "
            f"--manifest {SOURCE}/manifest.csv "
            f"--particle-dir {SOURCE}/particles "
            f"--tng-root {TNG_ROOT} "
            f"--output-dir {out} "
            "--snapshot 99 --hubble-param 0.6774 "
            "--r-min-kpc 0.05 --r-max-kpc 30.0 --n-r 32 --n-phi 48 "
            f"--z-max-kpc {z_max} --n-z {n_z}"
        ),
        f"ls {out}/*.hdf5 | wc -l",
        "echo REBIN_ALL_DONE",
    ]


def baseline_and_table(images_table: str, holdout_subhalo: int) -> list[str]:
    truth = f"{Z5}/density_grids_logr_cyl"
    baseline = f"{Z5}/baseline_density_grids_logr_cyl"
    table = f"{Z5}/density_residual_table.npz"
    manifest = f"{Z5}/manifest_holdout_{holdout_subhalo}.csv"
    # Force the X galaxy into the held-out test split so its recovery is a
    # genuine out-of-sample test. All 9 projections move together, so the
    # galaxy-level split stays leak-free.
    holdout = (
        "python - <<'PY'\n"
        "import pandas as pd\n"
        f"m = pd.read_csv('{SOURCE}/manifest.csv')\n"
        f"m.loc[m['subhalo_id'] == {holdout_subhalo}, 'split'] = 'test'\n"
        f"m.to_csv('{manifest}', index=False)\n"
        f"print('holdout split counts by galaxy:', "
        "m.drop_duplicates('subhalo_id')['split'].value_counts().to_dict())\n"
        "PY"
    )
    return [
        f"mkdir -p {Z5}",
        holdout,
        (
            "python scripts/build_tng50_baseline_density_grid.py "
            f"--config {CONFIG_TOML} "
            f"--manifest {manifest} "
            f"--residual-table {images_table} "
            f"--truth-grid-dir {truth} "
            f"--output-dir {baseline}"
        ),
        (
            "python scripts/build_tng50_density_residual_table.py "
            f"--manifest {manifest} "
            f"--residual-table {images_table} "
            f"--truth-grid-dir {truth} "
            f"--baseline-grid-dir {baseline} "
            f"--output {table}"
        ),
        f"ls -la {table}",
        "echo BASELINE_TABLE_DONE",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["probe", "rebin-all", "baseline-table"])
    parser.add_argument("--z-max-kpc", type=float, default=5.0)
    parser.add_argument("--n-z", type=int, default=32)
    parser.add_argument("--images-table", type=str, default=f"{CLEAN}/residual_table.npz")
    parser.add_argument("--holdout-subhalo", type=int, default=392276)
    args = parser.parse_args()
    if args.mode == "probe":
        commands = probe()
    elif args.mode == "rebin-all":
        commands = rebin_all(args.z_max_kpc, args.n_z)
    else:
        commands = baseline_and_table(args.images_table, args.holdout_subhalo)
    return run_remote(CONFIG, commands)


if __name__ == "__main__":
    raise SystemExit(main())
