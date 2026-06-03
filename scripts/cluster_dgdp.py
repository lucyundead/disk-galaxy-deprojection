from __future__ import annotations

import argparse
import subprocess
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
