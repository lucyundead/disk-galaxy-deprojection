from pathlib import Path

from scripts.cluster_dgdp import build_rsync_fetch_command, build_rsync_push_command


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
