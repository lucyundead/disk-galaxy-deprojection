from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from dgdp.models.mdn import SummaryResidualMDN
from train_density_residual_pca import (
    _choose_device,
    _make_batches,
    make_density_residual_features,
    standardize_with_train,
)

try:
    from report_milestone2b_physical_summaries import _radial_band_ids
except ModuleNotFoundError:
    from scripts.report_milestone2b_physical_summaries import _radial_band_ids


DEFAULT_DENSITY_TABLE = Path("outputs/tng50_milestone2b/density_residual_table.npz")
DEFAULT_PREDICTIONS = Path(
    "outputs/tng50_milestone2b/density_residual_pca_mdn_sweep_central/"
    "components_1_seed_20260609/density_residual_pca_mdn_predictions.npz"
)
DEFAULT_OUTPUT_DIR = Path("outputs/tng50_milestone2b/m2_amplitude_correction")

TARGET_CLIP = 2.0
AMPLITUDE_FLOOR_MSUN = 1.0
PROFILE_FLOOR = 1.0e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density-table", type=Path, default=DEFAULT_DENSITY_TABLE)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-radial-bands", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--image-feature-size", type=int, default=24)
    parser.add_argument("--central-pixel-scale-kpc", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-components", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _m2_band_amplitudes(
    mass: np.ndarray,
    phi_edges_rad: np.ndarray,
    band_ids: np.ndarray,
    n_bands: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-band m=2 amplitude (Msun) and band total mass (Msun) per row."""
    phi_centers = 0.5 * (phi_edges_rad[:-1] + phi_edges_rad[1:])
    radial_phi_mass = mass.sum(axis=3, dtype=np.float64)
    c2_radial = np.abs(
        (radial_phi_mass * np.exp(2j * phi_centers)[None, None, :]).sum(axis=2)
    )
    total_radial = radial_phi_mass.sum(axis=2)
    amplitudes = np.zeros((mass.shape[0], n_bands))
    totals = np.zeros((mass.shape[0], n_bands))
    for band in range(n_bands):
        members = band_ids == band
        amplitudes[:, band] = c2_radial[:, members].sum(axis=1)
        totals[:, band] = total_radial[:, members].sum(axis=1)
    return amplitudes, totals


def _m2_profiles(mass: np.ndarray, phi_edges_rad: np.ndarray) -> np.ndarray:
    phi_centers = 0.5 * (phi_edges_rad[:-1] + phi_edges_rad[1:])
    radial_phi_mass = mass.sum(axis=3, dtype=np.float64)
    c2 = np.abs((radial_phi_mass * np.exp(2j * phi_centers)[None, None, :]).sum(axis=2))
    total = radial_phi_mass.sum(axis=2)
    return np.divide(c2, total, out=np.zeros_like(total), where=total > 0.0)


def _coverage_68(samples: np.ndarray, truth: np.ndarray) -> float:
    lower = np.quantile(samples, 0.16, axis=1)
    upper = np.quantile(samples, 0.84, axis=1)
    return float(np.mean((truth >= lower) & (truth <= upper)))


def _split_metrics(
    *,
    mask: np.ndarray,
    true_delta: np.ndarray,
    mean_delta: np.ndarray,
    sampled_delta: np.ndarray,
    profile_true: np.ndarray,
    profile_posterior: np.ndarray,
    band_ids: np.ndarray,
) -> dict[str, float | int]:
    corrected = profile_posterior[mask] * np.exp(mean_delta[mask][:, band_ids])
    truth = profile_true[mask]
    return {
        "n_rows": int(np.sum(mask)),
        "log_delta_mae": float(np.mean(np.abs(mean_delta[mask] - true_delta[mask]))),
        "log_delta_coverage_68": _coverage_68(sampled_delta[mask], true_delta[mask]),
        "uncorrected_m2_profile_bias": float(np.mean(profile_posterior[mask] - truth)),
        "corrected_m2_profile_bias": float(np.mean(corrected - truth)),
        "uncorrected_m2_profile_mae": float(np.mean(np.abs(profile_posterior[mask] - truth))),
        "corrected_m2_profile_mae": float(np.mean(np.abs(corrected - truth))),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = _choose_device(args.device)

    with np.load(args.density_table) as table, np.load(args.predictions) as predictions:
        split = table["split"].astype(str)
        train_mask = split == "train"
        val_mask = split == "val"
        if int(np.sum(train_mask)) < 1:
            raise ValueError("training requires at least one train row")
        r_edges = table["r_edges_kpc"].astype(np.float32)
        phi_edges = table["phi_edges_rad"].astype(np.float64)
        n_radial_bins = r_edges.shape[0] - 1
        band_ids = _radial_band_ids(n_radial_bins, args.n_radial_bands)
        radial_area = 0.5 * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
        dphi = np.diff(phi_edges.astype(np.float32))
        dz = np.diff(table["z_edges_kpc"].astype(np.float32))
        volumes = (
            radial_area[:, None, None] * dphi[None, :, None] * dz[None, None, :]
        ).astype(np.float32)
        truth_mass = table["truth_density"].astype(np.float32) * volumes[None]
        baseline_mass = table["baseline_density"].astype(np.float32) * volumes[None]
        posterior_mean_mass = predictions["posterior_mean_mass"].astype(np.float32)

        baseline_amp, baseline_total = _m2_band_amplitudes(
            baseline_mass,
            phi_edges,
            band_ids,
            args.n_radial_bands,
        )
        posterior_amp, posterior_total = _m2_band_amplitudes(
            posterior_mean_mass,
            phi_edges,
            band_ids,
            args.n_radial_bands,
        )
        profile_true = _m2_profiles(truth_mass, phi_edges)
        profile_posterior = _m2_profiles(posterior_mean_mass, phi_edges)
        # target: per-band ratio of band-summed normalized m=2 profile values, so a
        # perfect prediction makes the corrected band-mean profile match truth exactly
        band_true = np.zeros((profile_true.shape[0], args.n_radial_bands))
        band_posterior = np.zeros_like(band_true)
        for band in range(args.n_radial_bands):
            members = band_ids == band
            band_true[:, band] = profile_true[:, members].sum(axis=1)
            band_posterior[:, band] = profile_posterior[:, members].sum(axis=1)
        true_delta = np.clip(
            np.log((band_true + PROFILE_FLOOR) / (band_posterior + PROFILE_FLOOR)),
            -TARGET_CLIP,
            TARGET_CLIP,
        ).astype(np.float32)

        features = make_density_residual_features(
            table["images"].astype(np.float32),
            table["metadata"].astype(np.float32),
            baseline_grid_mass_msun=table["baseline_grid_mass_msun"].astype(np.float32),
            image_feature_size=args.image_feature_size,
            central_pixel_scale_kpc=args.central_pixel_scale_kpc,
        )
        bar_features = np.column_stack(
            (
                np.log10(
                    (baseline_amp + AMPLITUDE_FLOOR_MSUN)
                    / (baseline_total + AMPLITUDE_FLOOR_MSUN)
                ),
                np.log10(
                    (posterior_amp + AMPLITUDE_FLOOR_MSUN)
                    / (posterior_total + AMPLITUDE_FLOOR_MSUN)
                ),
            )
        ).astype(np.float32)
        features = np.column_stack((features, bar_features)).astype(np.float32)
        x_all, x_mean, x_scale = standardize_with_train(features, train_mask)
        y_all, y_mean, y_scale = standardize_with_train(true_delta, train_mask)

        model = SummaryResidualMDN(
            input_dim=x_all.shape[1],
            output_dim=args.n_radial_bands,
            hidden_dim=args.hidden_dim,
            n_components=args.n_components,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        x_tensor = torch.tensor(x_all, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y_all, dtype=torch.float32, device=device)
        train_indices = np.flatnonzero(train_mask)
        val_indices = np.flatnonzero(val_mask)
        best_epoch = 0
        best_val = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for batch in _make_batches(train_indices, batch_size=args.batch_size, rng=rng):
                optimizer.zero_grad()
                loss = model.negative_log_likelihood(x_tensor[batch], y_tensor[batch])
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            model.eval()
            with torch.no_grad():
                eval_indices = val_indices if len(val_indices) else train_indices
                val_nll = float(
                    model.negative_log_likelihood(
                        x_tensor[eval_indices],
                        y_tensor[eval_indices],
                    ).item()
                )
            if val_nll < best_val:
                best_val = val_nll
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epoch == 0 or epoch == args.epochs - 1:
                print(
                    f"epoch={epoch + 1} "
                    f"train_nll={float(np.mean(losses)):.6f} "
                    f"val_nll={val_nll:.6f}"
                )
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            sampled_z = model.sample(x_tensor, n_samples=args.n_samples).cpu().numpy()
        sampled_log_m2_delta = sampled_z * y_scale[None, None, :] + y_mean[None, None, :]
        mean_log_m2_delta = sampled_log_m2_delta.mean(axis=1).astype(np.float32)

        metrics: dict[str, float | int | str | dict[str, float | int]] = {
            "device": str(device),
            "predictions": str(args.predictions),
            "n_radial_bands": int(args.n_radial_bands),
            "n_radial_bins": int(n_radial_bins),
            "image_feature_size": int(args.image_feature_size),
            "central_pixel_scale_kpc": float(args.central_pixel_scale_kpc),
            "hidden_dim": int(args.hidden_dim),
            "epochs": int(args.epochs),
            "best_epoch": int(best_epoch),
            "best_val_nll": float(best_val),
            "n_components": int(args.n_components),
            "n_samples": int(args.n_samples),
            "seed": int(args.seed),
        }
        for label in ("train", "val", "test"):
            mask = split == label
            if not np.any(mask):
                metrics[label] = {"n_rows": 0}
                continue
            metrics[label] = _split_metrics(
                mask=mask,
                true_delta=true_delta,
                mean_delta=mean_log_m2_delta,
                sampled_delta=sampled_log_m2_delta,
                profile_true=profile_true,
                profile_posterior=profile_posterior,
                band_ids=band_ids,
            )

        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": x_all.shape[1],
                "output_dim": args.n_radial_bands,
                "hidden_dim": args.hidden_dim,
                "n_components": args.n_components,
                "image_feature_size": args.image_feature_size,
                "central_pixel_scale_kpc": args.central_pixel_scale_kpc,
                "n_radial_bands": args.n_radial_bands,
                "n_radial_bins": n_radial_bins,
            },
            args.output_dir / "m2_amplitude_correction.pt",
        )
        np.savez_compressed(
            args.output_dir / "m2_amplitude_correction_normalization.npz",
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
            y_scale=y_scale,
        )
        np.savez_compressed(
            args.output_dir / "m2_amplitude_correction_predictions.npz",
            sampled_log_m2_delta=sampled_log_m2_delta.astype(np.float32),
            mean_log_m2_delta=mean_log_m2_delta,
            true_log_m2_delta=true_delta,
            n_radial_bands=np.array(args.n_radial_bands, dtype=np.int32),
            n_radial_bins=np.array(n_radial_bins, dtype=np.int32),
            split=split,
            galaxy_id=table["galaxy_id"],
            projection_id=table["projection_id"],
            metadata=table["metadata"],
        )

    (args.output_dir / "m2_amplitude_correction_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"best_epoch={best_epoch} best_val_nll={best_val:.6f}")
    print(
        "wrote m2 amplitude correction model to "
        f"{args.output_dir / 'm2_amplitude_correction.pt'}"
    )


if __name__ == "__main__":
    main()
