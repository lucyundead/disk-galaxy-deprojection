import subprocess
import sys


def test_evaluation_script_writes_metrics(tmp_path):
    run_dir = tmp_path / "run"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_synthetic_benchmark.py",
            "--config",
            "configs/milestone1.synthetic.toml",
            "--output-dir",
            str(run_dir),
            "--n-galaxies",
            "5",
            "--projections-per-galaxy",
            "2",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/train_summary_residual_mdn.py",
            "--data",
            str(run_dir / "residual_table.npz"),
            "--output-dir",
            str(run_dir),
            "--epochs",
            "2",
            "--hidden-dim",
            "16",
            "--n-components",
            "2",
        ],
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_summary_residual.py",
            "--run-dir",
            str(run_dir),
            "--n-samples",
            "8",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "wrote metrics" in result.stdout
    assert (run_dir / "metrics.json").exists()
