from __future__ import annotations

import argparse
import base64
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from dgdp.cluster import build_remote_script, load_cluster_config, sync_include_paths


def build_tng50_remote_commands(
    *,
    remote_tng50_root: str,
    remote_output_dir: str,
    bar_catalog_path: str,
    snapshot: int,
    max_candidate_galaxies: int,
    max_particles_per_galaxy: int,
    hubble_param: float,
) -> list[str]:
    output_dir = shlex.quote(remote_output_dir)
    tng_root = shlex.quote(remote_tng50_root)
    manifest = shlex.quote(f"{remote_output_dir}/manifest.csv")
    particles = shlex.quote(f"{remote_output_dir}/particles")
    metrics = shlex.quote(f"{remote_output_dir}/metrics.json")
    residual_table = shlex.quote(f"{remote_output_dir}/residual_table.npz")
    bar_arg = f" --bar-catalog {shlex.quote(bar_catalog_path)}" if bar_catalog_path else ""
    return [
        f"mkdir -p {output_dir}",
        (
            "python scripts/build_tng50_manifest.py "
            f"--tng-root {tng_root} "
            f"--output {manifest} "
            f"--snapshot {snapshot} "
            f"--max-candidates {max_candidate_galaxies}"
            f"{bar_arg}"
        ),
        (
            "python scripts/extract_tng50_particles.py "
            f"--tng-root {tng_root} "
            f"--manifest {manifest} "
            f"--output-dir {particles} "
            f"--snapshot {snapshot} "
            f"--max-particles-per-galaxy {max_particles_per_galaxy} "
            f"--hubble-param {hubble_param}"
        ),
        (
            "python scripts/build_tng50_benchmark.py "
            "--config configs/milestone1.synthetic.toml "
            f"--manifest {manifest} "
            f"--particle-dir {particles} "
            f"--output-dir {output_dir}"
        ),
        (
            "python scripts/train_summary_residual_mdn.py "
            f"--data {residual_table} "
            f"--output-dir {output_dir} "
            "--epochs 10 --hidden-dim 32 --n-components 2"
        ),
        f"python scripts/evaluate_summary_residual.py --run-dir {output_dir} --n-samples 32",
        f"python -m json.tool {metrics}",
    ]


def build_rsync_push_command(
    *,
    project_root: Path,
    cluster_host: str,
    remote_project_root: str,
) -> list[str]:
    return [
        "rsync",
        "-av",
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


def _sync_tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    path = Path(info.name)
    if any(part in {".pytest_cache", ".ruff_cache", "__pycache__"} for part in path.parts):
        return None
    if path.suffix == ".pyc":
        return None
    return info


def write_sync_archive(project_root: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as archive:
        for rel_path in sync_include_paths():
            path = project_root / rel_path
            if path.exists():
                archive.add(path, arcname=rel_path, filter=_sync_tar_filter)


def run_archive_sync(config_path: Path, project_root: Path) -> int:
    cfg = load_cluster_config(config_path)
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "dgdp_sync.tar.gz"
        write_sync_archive(project_root, archive_path)
        encoded = base64.b64encode(archive_path.read_bytes()).decode("ascii")

    remote_archive = f"{cfg.remote_project_root}/.dgdp_sync_archive.tar.gz"
    script = "\n".join(
        [
            "set -e",
            f"mkdir -p {shlex.quote(cfg.remote_project_root)}",
            f"base64 -d > {shlex.quote(remote_archive)} <<'DGDP_ARCHIVE'",
            encoded,
            "DGDP_ARCHIVE",
            f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(cfg.remote_project_root)}",
            "exit",
        ]
    )
    proc = subprocess.run([str(cfg.hpc_wrapper), "shell"], input=script + "\n", text=True, check=False)
    return int(proc.returncode)


def build_remote_command(tokens: list[str]) -> str:
    return shlex.join(tokens)


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
    sub.add_parser("run-tng50")
    sub.add_parser("fetch-tng50")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("remote_command", nargs="+")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_cluster_config(args.config)
    project_root = Path(__file__).resolve().parents[1]

    if args.command == "sync":
        if shutil.which("rsync") is None:
            return run_archive_sync(args.config, project_root)
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

    if args.command == "run-tng50":
        return run_remote(
            args.config,
            build_tng50_remote_commands(
                remote_tng50_root=cfg.remote_tng50_root,
                remote_output_dir=cfg.remote_output_dir,
                bar_catalog_path=cfg.bar_catalog_path,
                snapshot=cfg.snapshot,
                max_candidate_galaxies=cfg.max_candidate_galaxies,
                max_particles_per_galaxy=cfg.max_particles_per_galaxy,
                hubble_param=cfg.hubble_param,
            ),
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
        return run_remote(args.config, [build_remote_command(args.remote_command)])

    raise RuntimeError(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
