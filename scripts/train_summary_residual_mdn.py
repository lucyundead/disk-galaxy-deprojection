from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dgdp.models.mdn import SummaryResidualMDN
from dgdp.train import make_feature_matrix, standardize_train_apply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260602)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.data.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    table = np.load(args.data, allow_pickle=True)
    x_all = make_feature_matrix(table["images"], table["baseline"], table["metadata"])
    y_all = table["delta"].astype(np.float32)
    split = table["split"].astype(str)
    train_mask = split == "train"
    val_mask = split == "val"

    x_train, _x_val, x_mean, x_scale = standardize_train_apply(x_all[train_mask], x_all[val_mask])
    y_train, _y_val, y_mean, y_scale = standardize_train_apply(y_all[train_mask], y_all[val_mask])

    model = SummaryResidualMDN(
        input_dim=x_train.shape[1],
        output_dim=y_train.shape[1],
        hidden_dim=args.hidden_dim,
        n_components=args.n_components,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        loss = model.negative_log_likelihood(x_tensor, y_tensor)
        loss.backward()
        optimizer.step()
        if epoch == 0 or epoch == args.epochs - 1:
            print(f"epoch={epoch + 1} train_nll={loss.item():.6f}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": x_train.shape[1],
            "output_dim": y_train.shape[1],
            "hidden_dim": args.hidden_dim,
            "n_components": args.n_components,
        },
        output_dir / "summary_residual_mdn.pt",
    )
    np.savez_compressed(
        output_dir / "normalization.npz",
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
    )
    print(f"wrote model to {output_dir / 'summary_residual_mdn.pt'}")


if __name__ == "__main__":
    main()
