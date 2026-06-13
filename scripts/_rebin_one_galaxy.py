"""One-off helper: re-bin a single galaxy's truth density at a finer z-grid on
the cluster, to test whether the boxy/peanut X resolves at 0.3125 kpc.

Drives the tested cluster_dgdp.run_remote path with an explicit command list
(each list element becomes its own line in the remote script, so shell
operators survive without shlex.join mangling). Read-only on the remote except
for writing into a scratch dir under the remote project root.

Subcommands:
  probe   - connectivity + verify manifest/particle file for the subhalo exist
  rebin   - build a single-galaxy truth density grid at the requested z-grid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cluster_dgdp import run_remote  # noqa: E402

CONFIG = Path("configs/milestone2c.cluster.toml")
SOURCE = "outputs/tng50_milestone2c"  # manifest + particles live here
SCRATCH = "outputs/tng50_milestone2c_zrebin"


def probe(subhalo: int) -> list[str]:
    manifest = f"{SOURCE}/manifest.csv"
    particle = f"{SOURCE}/particles/subhalo_{subhalo}.hdf5"
    return [
        "echo PROBE_START",
        "hostname",
        "whoami",
        "pwd",
        f"(test -f {manifest} && echo MANIFEST_OK) || echo MANIFEST_MISSING",
        f"(grep -c ^{subhalo}, {manifest} || true)",
        f"(ls -la {particle} && echo PARTICLE_OK) || echo PARTICLE_MISSING",
        "echo PROBE_END",
    ]


def rebin(subhalos: list[int], z_max: float, n_z: int) -> list[str]:
    manifest = f"{SOURCE}/manifest.csv"
    tag = "_".join(str(s) for s in subhalos[:3]) + (f"_plus{len(subhalos) - 3}" if len(subhalos) > 3 else "")
    one_manifest = f"{SCRATCH}/manifest_{tag}.csv"
    outdir = f"{SCRATCH}/zmax{z_max:g}_nz{n_z}"
    commands = [
        f"mkdir -p {SCRATCH}",
        f"head -n 1 {manifest} > {one_manifest}",
    ]
    for sid in subhalos:
        commands.append(f"grep ^{sid}, {manifest} >> {one_manifest}")
    commands += [
        f"wc -l {one_manifest}",
        f"mkdir -p {outdir}",
        (
            "python scripts/build_tng50_density_grid.py "
            f"--manifest {one_manifest} "
            f"--particle-dir {SOURCE}/particles "
            "--tng-root /home/cossim/IllustrisTNG/TNG50-1 "
            f"--output-dir {outdir} "
            "--snapshot 99 --hubble-param 0.6774 "
            "--r-min-kpc 0.05 --r-max-kpc 30.0 --n-r 32 --n-phi 48 "
            f"--z-max-kpc {z_max} --n-z {n_z}"
        ),
        f"ls -la {outdir}",
        "echo REBIN_DONE",
    ]
    return commands


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["probe", "rebin"])
    parser.add_argument("--subhalo", type=int, help="single subhalo (probe)")
    parser.add_argument("--subhalos", type=str, help="comma-separated subhalos (rebin)")
    parser.add_argument("--z-max-kpc", type=float, default=5.0)
    parser.add_argument("--n-z", type=int, default=32)
    args = parser.parse_args()
    if args.mode == "probe":
        commands = probe(args.subhalo)
    else:
        ids = [int(s) for s in (args.subhalos or str(args.subhalo)).split(",")]
        commands = rebin(ids, args.z_max_kpc, args.n_z)
    return run_remote(CONFIG, commands)


if __name__ == "__main__":
    raise SystemExit(main())
