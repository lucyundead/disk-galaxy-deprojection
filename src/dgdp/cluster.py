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
