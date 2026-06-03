import subprocess
import sys


def test_build_synthetic_benchmark_creates_residual_table(tmp_path):
    output_dir = tmp_path / "run"
    cmd = [
        sys.executable,
        "scripts/build_synthetic_benchmark.py",
        "--config",
        "configs/milestone1.synthetic.toml",
        "--output-dir",
        str(output_dir),
        "--n-galaxies",
        "4",
        "--projections-per-galaxy",
        "2",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    assert "wrote residual table" in result.stdout
    assert (output_dir / "residual_table.npz").exists()
    assert (output_dir / "manifest.csv").exists()
