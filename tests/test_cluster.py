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
    assert 'export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"' in script
    assert "python -c \"print('ok')\"" in script
