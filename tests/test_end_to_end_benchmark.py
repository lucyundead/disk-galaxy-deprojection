import json
import subprocess
import sys


def test_end_to_end_synthetic_benchmark(tmp_path):
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
            "6",
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
            "3",
            "--hidden-dim",
            "16",
            "--n-components",
            "2",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_summary_residual.py",
            "--run-dir",
            str(run_dir),
            "--n-samples",
            "8",
        ],
        check=True,
    )
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    assert metrics["n_test"] > 0
    assert metrics["baseline_mae"] >= 0.0
    assert metrics["corrected_mae"] >= 0.0
    assert 0.0 <= metrics["coverage_68"] <= 1.0
