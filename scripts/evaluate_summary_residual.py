from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dgdp.metrics import interval_coverage, mean_absolute_error
from dgdp.models.mdn import SummaryResidualMDN
from dgdp.train import make_feature_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = np.load(args.run_dir / "residual_table.npz", allow_pickle=True)
    norm = np.load(args.run_dir / "normalization.npz")
    checkpoint = torch.load(args.run_dir / "summary_residual_mdn.pt", map_location="cpu")

    model = SummaryResidualMDN(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        n_components=int(checkpoint["n_components"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    x_all = make_feature_matrix(table["images"], table["baseline"], table["metadata"])
    x_all = (x_all - norm["x_mean"]) / norm["x_scale"]
    split = table["split"].astype(str)
    test_mask = split == "test"
    x_test = torch.tensor(x_all[test_mask], dtype=torch.float32)

    samples_z = model.sample(x_test, n_samples=args.n_samples).numpy()
    samples = samples_z * norm["y_scale"][None, None, :] + norm["y_mean"][None, None, :]
    baseline = table["baseline"][test_mask]
    truth = table["truth"][test_mask]
    corrected_samples = baseline[:, None, :] + samples
    corrected_mean = corrected_samples.mean(axis=1)
    lower = np.quantile(corrected_samples, 0.16, axis=1)
    upper = np.quantile(corrected_samples, 0.84, axis=1)

    metrics = {
        "baseline_mae": mean_absolute_error(baseline, truth),
        "corrected_mae": mean_absolute_error(corrected_mean, truth),
        "coverage_68": interval_coverage(lower, upper, truth),
        "n_test": int(test_mask.sum()),
    }
    (args.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote metrics to {args.run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
