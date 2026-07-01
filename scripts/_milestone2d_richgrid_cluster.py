"""Cluster orchestration: rebuild the milestone-2d sample at a RICHER projection grid and R=64.

Grid: inclination 10..60 deg step 5 (11), bar angle -80..90 deg step 10 (18) -> 198 proj/galaxy
(negative inclinations dropped -- degenerate with +i & flipped bar angle for a midplane-symmetric
disk, which we already span; -90 dropped -- duplicate of +90 for an m=2 bar). R=64 log bins.

Pipeline (no builder code changes -- all CLI args):
  base manifest (galaxy-level) -> build_tng50_all_particle_images.py (new grid) -> images + expanded
  manifest -> build_tng50_density_grid.py (--n-r 64) -> truth grids -> baseline + density residual table.

`validate` runs the whole chain for N_VAL galaxies into a *_val dir and inspects the output; `full`
runs all galaxies into the production dir. Writes to NEW remote dirs; existing data untouched.
Drives the tested cluster_dgdp.run_remote path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cluster_dgdp import run_remote  # noqa: E402

CONFIG = Path("configs/milestone2c.cluster.toml")
CONFIG_TOML = "configs/milestone2b.clean3d.toml"
SOURCE = "outputs/tng50_milestone2c"            # galaxy manifest + per-galaxy particle hdf5s
TNG_ROOT = "/home/cossim/IllustrisTNG/TNG50-1"
OUT_FULL = "outputs/tng50_milestone2d_rich"
OUT_VAL = "outputs/tng50_milestone2d_rich_val"
INCL = "10,15,20,25,30,35,40,45,50,55,60"        # 11
BAR = "-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90"  # 18
N_R, N_PHI, N_Z, Z_MAX, R_MIN, R_MAX = 64, 48, 32, 5.0, 0.05, 30.0
N_VAL = 2
REMOTE_ROOT = "/home/zli/disk-galaxy-deprojection"   # cfg.remote_project_root (login01, Torque/PBS)
SMALL_INCL, SMALL_BAR = "20,40", "0,45"              # tiny grid to validate the PBS-array mechanism fast


def probe() -> list[str]:
    return [
        "echo STATE_START",
        f"ls -la {SOURCE}/manifest.csv 2>&1 | cat",
        f"head -1 {SOURCE}/manifest.csv",
        f"(ls {SOURCE}/particles/*.hdf5 2>/dev/null | wc -l)",
        "echo STATE_END",
    ]


def base_manifest_cmd(out: str, n_gal: int) -> str:
    sub = f"g = g.head({n_gal})\n" if n_gal else ""
    return (
        "python - <<'PY'\n"
        "import pandas as pd, os\n"
        f"m = pd.read_csv('{SOURCE}/manifest.csv')\n"
        "drop = [c for c in ['projection_id','inclination_deg','disk_pa_deg','bar_angle_deg'] if c in m.columns]\n"
        "g = m.drop(columns=drop).drop_duplicates('subhalo_id').reset_index(drop=True)\n"
        f"{sub}"
        f"os.makedirs('{out}', exist_ok=True)\n"
        f"g.to_csv('{out}/base_manifest.csv', index=False)\n"
        "print('base galaxies:', len(g), '| split:', g['split'].value_counts().to_dict())\n"
        "PY"
    )


def images_cmd(out: str) -> str:
    return (
        "python scripts/build_tng50_all_particle_images.py "
        f"--config {CONFIG_TOML} --manifest {out}/base_manifest.csv "
        f"--particle-dir {SOURCE}/particles --tng-root {TNG_ROOT} --output-dir {out} "
        f"--snapshot 99 --hubble-param 0.6774 "
        # =value form: the bar-angle list starts with '-80', which argparse would otherwise read as a flag
        f"--inclinations-deg={INCL} --bar-angles-deg={BAR}"
    )


def truth_cmd(out: str) -> str:
    return (
        "python scripts/build_tng50_density_grid.py "
        f"--manifest {out}/base_manifest.csv --particle-dir {SOURCE}/particles --tng-root {TNG_ROOT} "
        f"--output-dir {out}/density_grids_logr_cyl --snapshot 99 --hubble-param 0.6774 "
        f"--r-min-kpc {R_MIN} --r-max-kpc {R_MAX} --n-r {N_R} --n-phi {N_PHI} "
        f"--z-max-kpc {Z_MAX} --n-z {N_Z}"
    )


def table_cmds(out: str) -> list[str]:
    truth = f"{out}/density_grids_logr_cyl"
    baseline = f"{out}/baseline_density_grids_logr_cyl"
    return [
        (
            "python scripts/build_tng50_baseline_density_grid.py "
            f"--config {CONFIG_TOML} --manifest {out}/manifest.csv "
            f"--residual-table {out}/residual_table.npz --truth-grid-dir {truth} --output-dir {baseline}"
        ),
        (
            "python scripts/build_tng50_density_residual_table.py "
            f"--manifest {out}/manifest.csv --residual-table {out}/residual_table.npz "
            f"--truth-grid-dir {truth} --baseline-grid-dir {baseline} "
            f"--output {out}/density_residual_table.npz"
        ),
    ]


def inspect_cmd(out: str) -> str:
    return (
        f"ls -la {out}/density_residual_table.npz\n"
        "python - <<'PY'\n"
        "import numpy as np\n"
        f"t = np.load('{out}/density_residual_table.npz')\n"
        "print('keys:', t.files)\n"
        "for k in ('truth_density','images','galaxy_id','projection_id','split'):\n"
        "    if k in t.files: print(k, t[k].shape, t[k].dtype)\n"
        "print('unique galaxies:', len(np.unique(t['galaxy_id'])), '| rows:', len(t['galaxy_id']))\n"
        "print('projections/galaxy:', len(t['galaxy_id'])//max(len(np.unique(t['galaxy_id'])),1))\n"
        "PY"
    )


def imgtest(out: str) -> list[str]:
    return [
        "echo IMGTEST_START; date",
        (
            "python -u scripts/build_tng50_all_particle_images.py "
            f"--config {CONFIG_TOML} --manifest {out}/base_manifest.csv "
            f"--particle-dir {SOURCE}/particles --tng-root {TNG_ROOT} --output-dir {out}/imgtest "
            "--snapshot 99 --hubble-param 0.6774 --inclinations-deg=20 --bar-angles-deg=0"
        ),
        "date",
        f"ls -la {out}/imgtest/residual_table.npz 2>&1 | cat",
        "echo IMGTEST_END",
    ]


def check(out: str) -> list[str]:
    return [
        f"echo CHECK_START {out}",
        f"ls -la {out} 2>&1 | cat",
        f"echo '--- truth grids (count) ---'; ls {out}/density_grids_logr_cyl/*.hdf5 2>/dev/null | wc -l",
        f"echo '--- tables ---'; ls -la {out}/residual_table.npz {out}/density_residual_table.npz 2>&1 | cat",
        "echo CHECK_END",
    ]


def image_array(out: str, incl: str, bar: str, arr_range: str, n_gal: int) -> list[str]:
    """Build base manifest + a Torque PBS array .pbs (one task per galaxy via --galaxy-index) + qsub it."""
    pbs = (
        f"cat > {out}/gen_images.pbs <<'PBS_EOF'\n"
        "#!/bin/bash\n#PBS -N dgdp_img\n#PBS -q normal\n#PBS -l nodes=1:ppn=1\n"
        "#PBS -l walltime=02:00:00\n#PBS -l mem=16gb\n#PBS -j oe\n"
        f"#PBS -o {REMOTE_ROOT}/{out}/logs/\n"
        f"cd {REMOTE_ROOT}\nsource /home/zli/.bashrc\nconda activate paicos-conda\n"
        "export PYTHONPATH=$PWD/src:$PYTHONPATH\n"
        f"python scripts/build_tng50_all_particle_images.py --config {CONFIG_TOML} "
        f"--manifest {out}/base_manifest.csv --particle-dir {SOURCE}/particles --tng-root {TNG_ROOT} "
        f"--output-dir {out}/shards/$PBS_ARRAYID --snapshot 99 --hubble-param 0.6774 "
        f"--inclinations-deg={incl} --bar-angles-deg={bar} --galaxy-index $PBS_ARRAYID\n"
        "PBS_EOF"
    )
    return [
        base_manifest_cmd(out, n_gal),
        f"mkdir -p {out}/shards {out}/logs",
        pbs,
        f"qsub -t {arr_range} {out}/gen_images.pbs",
    ]


def pbs_single(out: str, name: str, body: str, walltime: str, mem: str, ppn: int = 1) -> str:
    return (
        f"cat > {out}/{name}.pbs <<'PBS_EOF'\n"
        f"#!/bin/bash\n#PBS -N {name}\n#PBS -q normal\n#PBS -l nodes=1:ppn={ppn}\n"
        f"#PBS -l walltime={walltime}\n#PBS -l mem={mem}\n#PBS -j oe\n#PBS -o {REMOTE_ROOT}/{out}/logs/\n"
        f"cd {REMOTE_ROOT}\nsource /home/zli/.bashrc\nconda activate paicos-conda\n"
        "export PYTHONPATH=$PWD/src:$PYTHONPATH\n"
        f"{body}\n"
        "PBS_EOF"
    )


def truth_pbs(out: str) -> list[str]:
    return [f"mkdir -p {out}/logs",
            pbs_single(out, "dgdp_truth", truth_cmd(out), "06:00:00", "16gb"),
            f"qsub {out}/dgdp_truth.pbs"]


def merge_cmd(out: str) -> str:
    # Frugal: the only big array is images (~5.4 GB total); preallocate once and fill shard
    # by shard so peak memory is ~one output copy, not 185 inputs + a concatenated copy.
    return (
        "python - <<'PY'\n"
        f"OUT = '{out}'\n"
        "import os, glob, numpy as np, pandas as pd\n"
        "dirs = sorted(glob.glob(OUT + '/shards/*/'), key=lambda p: int(p.rstrip('/').rsplit('/', 1)[-1]))\n"
        "t0 = np.load(dirs[0] + 'residual_table.npz')\n"
        "per = t0['images'].shape[0]\n"
        "# ponytail: fixed incl*bar grid => every shard has `per` rows; preallocate upper bound, trim\n"
        "images = np.empty((len(dirs) * per,) + t0['images'].shape[1:], dtype=t0['images'].dtype)\n"
        "small_keys = [k for k in ['metadata', 'split', 'galaxy_id', 'projection_id', 'baseline', 'truth', 'delta'] if k in t0.files]\n"
        "small = {k: [] for k in small_keys}\n"
        "mans = []\n"
        "row = 0\n"
        "for d in dirs:\n"
        "    t = np.load(d + 'residual_table.npz')\n"
        "    im = t['images']\n"
        "    images[row:row + im.shape[0]] = im\n"
        "    row += im.shape[0]\n"
        "    for k in small_keys: small[k].append(t[k])\n"
        "    mans.append(pd.read_csv(d + 'manifest.csv'))\n"
        "merged = {'images': images[:row]}\n"
        "for k in small_keys: merged[k] = np.concatenate(small[k])\n"
        "merged['summary_names'] = t0['summary_names']\n"
        "tmp = OUT + '/.residual_table.tmp'\n"
        "np.savez_compressed(tmp, **merged)\n"
        "os.replace(tmp + '.npz', OUT + '/residual_table.npz')\n"
        "pd.concat(mans, ignore_index=True).to_csv(OUT + '/manifest.csv', index=False)\n"
        "print('MERGED rows', len(merged['galaxy_id']), 'galaxies', len(np.unique(merged['galaxy_id'])),"
        " 'shards', len(dirs))\n"
        "PY"
    )


def table_pbs(out: str) -> list[str]:
    truth = f"{out}/density_grids_logr_cyl"
    baseline = f"{out}/baseline_density_grids_logr_cyl"
    body = (
        f"python scripts/build_tng50_baseline_density_grid.py --config {CONFIG_TOML} --manifest {out}/manifest.csv "
        f"--residual-table {out}/residual_table.npz --truth-grid-dir {truth} --output-dir {baseline} && "
        f"python scripts/build_tng50_density_residual_table.py --manifest {out}/manifest.csv "
        f"--residual-table {out}/residual_table.npz --truth-grid-dir {truth} --baseline-grid-dir {baseline} "
        f"--output {out}/density_residual_table.npz"
    )
    return [pbs_single(out, "dgdp_table", body, "12:00:00", "48gb"), f"qsub {out}/dgdp_table.pbs"]


def retrain_pbs(out: str) -> list[str]:
    """PBS: retrain the conserving deprojection (sech^2 mixture q_m) on the R=64 table.

    Trains on the cluster (paicos-conda has torch); writes only small outputs (metrics json +
    figure) to {out}/mixture_retrain. The table loads truth+baseline+images at R=64 -> hi mem.
    """
    od = f"{out}/mixture_retrain"
    body = (
        "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8\n"
        "python scripts/deproject_fourier_rz_conserving.py "
        f"--table {out}/density_residual_table.npz "
        f"--allocation configs/fourier_rz_mixture.json --output-dir {od} --skip-grid-mdn"
    )
    return [f"mkdir -p {od}/figures",
            pbs_single(out, "dgdp_retrain", body, "10:00:00", "160gb", ppn=8),
            f"qsub {out}/dgdp_retrain.pbs"]


def train_bundle_pbs(out: str) -> list[str]:
    """PBS: train the fixed-dict q_m head on R=64 and export the ~1 MB torch-free package bundle
    to src/dgdp/models/dgdp_fixed_dict.npz (fetched back and committed into the package)."""
    body = (
        "export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8\n"
        "python scripts/train_deprojection_model.py "
        f"--table {out}/density_residual_table.npz --out src/dgdp/models/dgdp_fixed_dict.npz"
    )
    return [pbs_single(out, "dgdp_bundle", body, "06:00:00", "160gb", ppn=8),
            f"qsub {out}/dgdp_bundle.pbs"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "validate", "full", "check", "inspect", "imgtest",
                                     "array-test", "array-full", "qstat", "shards",
                                     "truth", "merge", "table", "retrain", "train-bundle"])
    args = ap.parse_args()
    if args.mode == "probe":
        return run_remote(CONFIG, probe())
    if args.mode == "array-test":
        return run_remote(CONFIG, image_array(OUT_VAL, SMALL_INCL, SMALL_BAR, "0-1", N_VAL))
    if args.mode == "array-full":
        return run_remote(CONFIG, image_array(OUT_FULL, INCL, BAR, "0-184%20", 0))
    if args.mode == "truth":      # R=64 truth grids; independent of images -> run in PARALLEL with the array
        return run_remote(CONFIG, truth_pbs(OUT_FULL))
    if args.mode == "merge":      # after the image array: concat shards -> residual_table.npz + manifest.csv
        return run_remote(CONFIG, [merge_cmd(OUT_FULL)])
    if args.mode == "table":      # after merge + truth: baseline grids + final density_residual_table.npz
        return run_remote(CONFIG, table_pbs(OUT_FULL))
    if args.mode == "retrain":    # after table: train the sech^2-mixture conserving deprojection on R=64
        return run_remote(CONFIG, retrain_pbs(OUT_FULL))
    if args.mode == "train-bundle":  # after table: export the torch-free package model bundle
        return run_remote(CONFIG, train_bundle_pbs(OUT_FULL))
    if args.mode == "qstat":
        return run_remote(CONFIG, [
            "qstat -u zli 2>&1 | tail -25",
            "python - <<PY\nimport subprocess as s\no=s.run(['qstat','-t'],capture_output=True,text=True).stdout\n"
            "rows=[l.split()[-2] for l in o.splitlines() if 'dgdp_img' in l]\n"
            "from collections import Counter\nprint('dgdp_img task states:', dict(Counter(rows)))\nPY",
        ])
    if args.mode == "shards":
        out = OUT_FULL
        return run_remote(CONFIG, [
            f"python -c \"import glob; print('shards with table:', len(glob.glob('{out}/shards/*/residual_table.npz')))\"",
            f"echo failed-logs:; grep -lE 'Error|Traceback|exceeded' {out}/logs/* 2>/dev/null | cat",
        ])
    if args.mode == "imgtest":
        return run_remote(CONFIG, imgtest(OUT_VAL))
    if args.mode == "check":
        return run_remote(CONFIG, check(OUT_VAL))
    if args.mode == "inspect":
        return run_remote(CONFIG, [inspect_cmd(OUT_VAL)])
    out, n_gal = (OUT_VAL, N_VAL) if args.mode == "validate" else (OUT_FULL, 0)
    cmds = [
        f"mkdir -p {out}",
        base_manifest_cmd(out, n_gal),
        images_cmd(out),
        truth_cmd(out),
        *table_cmds(out),
    ]
    if args.mode == "validate":
        cmds.append(inspect_cmd(out))
    return run_remote(CONFIG, cmds)


if __name__ == "__main__":
    raise SystemExit(main())
