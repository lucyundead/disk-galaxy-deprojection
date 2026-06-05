import tarfile
from pathlib import Path

from scripts.cluster_dgdp import (
    _extract_encoded_archive,
    build_remote_command,
    build_rsync_fetch_command,
    build_rsync_push_command,
    build_tng50_density_remote_commands,
    build_tng50_remote_commands,
    write_sync_archive,
)


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
    assert "--delete" not in cmd
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


def test_write_sync_archive_includes_lightweight_files(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "README.md").write_text("hello", encoding="utf-8")
    (project_root / "src").mkdir()
    (project_root / "src" / "module.py").write_text("x = 1\n", encoding="utf-8")
    (project_root / "outputs").mkdir()
    (project_root / "outputs" / "large.dat").write_text("skip", encoding="utf-8")
    (project_root / "src" / "__pycache__").mkdir()
    (project_root / "src" / "__pycache__" / "module.pyc").write_bytes(b"skip")
    archive = tmp_path / "sync.tar.gz"

    write_sync_archive(project_root, archive)

    with tarfile.open(archive, "r:gz") as handle:
        names = set(handle.getnames())

    assert "README.md" in names
    assert "src/module.py" in names
    assert "outputs/large.dat" not in names
    assert "src/__pycache__/module.pyc" not in names


def test_extract_encoded_archive_ignores_wrapper_noise():
    payload = "aGVsbG8="
    output = f"warning before\nDGDP_FETCH_BEGIN\n{payload}\nDGDP_FETCH_END\nwarning after\n"

    assert _extract_encoded_archive(output) == b"hello"


def test_build_remote_command_preserves_semicolon_token_boundary():
    command = build_remote_command(["echo", "ok; rm -rf outputs/tmp"])

    assert command == "echo 'ok; rm -rf outputs/tmp'"


def test_build_tng50_remote_commands_include_manifest_extract_train_eval():
    commands = build_tng50_remote_commands(
        remote_tng50_root="/home/cossim/IllustrisTNG/TNG50-1",
        remote_output_dir="outputs/tng50_milestone2",
        bar_catalog_path="",
        snapshot=99,
        max_candidate_galaxies=4,
        min_star_particles=50000,
        min_stellar_mass_msun=3162277660.1683793,
        min_bar_strength=0.2,
        min_bar_size_kpc=2.0,
        split_seed=20260604,
        max_particles_per_galaxy=1000,
        hubble_param=0.6774,
    )
    text = "\n".join(commands)

    assert "scripts/build_tng50_manifest.py" in text
    assert "scripts/extract_tng50_particles.py" in text
    assert "scripts/build_tng50_benchmark.py" in text
    assert "scripts/train_summary_residual_mdn.py" in text
    assert "scripts/evaluate_summary_residual.py" in text
    assert "--snapshot 99" in text
    assert "--min-stellar-mass-msun 3162277660.1683793" in text
    assert "--min-bar-size-kpc 2.0" in text


def test_build_tng50_density_remote_commands_use_existing_manifest_and_particles():
    commands = build_tng50_density_remote_commands(
        remote_tng50_root="/home/cossim/IllustrisTNG/TNG50-1",
        remote_output_dir="outputs/tng50_milestone2",
        snapshot=99,
        hubble_param=0.6774,
    )
    text = "\n".join(commands)

    assert "scripts/build_tng50_density_grid.py" in text
    assert "--manifest outputs/tng50_milestone2/manifest.csv" in text
    assert "--particle-dir outputs/tng50_milestone2/particles" in text
    assert "--tng-root /home/cossim/IllustrisTNG/TNG50-1" in text
    assert "--output-dir outputs/tng50_milestone2/density_grids_logr_cyl" in text
    assert "--r-min-kpc 0.05" in text
    assert "--n-r 32" in text
    assert "--n-phi 48" in text
    assert "--n-z 32" in text
