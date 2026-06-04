from __future__ import annotations

import argparse
from pathlib import Path

from dgdp.bar_catalog import join_bar_catalog
from dgdp.tng50_catalog import assign_galaxy_splits, build_candidate_manifest, read_group_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tng-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=int, default=99)
    parser.add_argument("--bar-catalog", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--min-star-particles", type=int, default=5000)
    parser.add_argument("--min-stellar-mass-msun", type=float, default=1.0e9)
    parser.add_argument("--min-bar-strength", type=float, default=0.2)
    parser.add_argument("--min-bar-size-kpc", type=float, default=1.0)
    parser.add_argument("--split-seed", type=int, default=20260604)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = read_group_catalog(args.tng_root / f"groups_{args.snapshot:03d}", snapshot=args.snapshot)
    has_bar_catalog = args.bar_catalog is not None and args.bar_catalog.exists()
    prefilter_max = catalog.subhalo_len_type.shape[0] if has_bar_catalog else args.max_candidates
    manifest = build_candidate_manifest(
        catalog,
        min_star_particles=args.min_star_particles,
        min_stellar_mass_msun=args.min_stellar_mass_msun,
        max_candidates=prefilter_max,
        snapshot=args.snapshot,
    )
    if has_bar_catalog:
        manifest = join_bar_catalog(manifest, args.bar_catalog, snapshot=args.snapshot)
        if "barred_catalog" in manifest:
            manifest = manifest[manifest["barred_catalog"].fillna(False).astype(bool)]
        elif "bar_type" in manifest:
            manifest = manifest[manifest["bar_type"].fillna(0).astype(int) > 0]
        if "bar_a2_catalog" in manifest:
            manifest = manifest[
                manifest["bar_a2_catalog"].fillna(0.0).astype(float) >= args.min_bar_strength
            ]
        if "bar_length_catalog" in manifest:
            manifest = manifest[
                manifest["bar_length_catalog"].fillna(0.0).astype(float) >= args.min_bar_size_kpc
            ]
        manifest = manifest.sort_values("stellar_mass_msun", ascending=False).head(args.max_candidates)
        manifest = assign_galaxy_splits(manifest.reset_index(drop=True), seed=args.split_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"wrote TNG50 manifest to {args.output} with {len(manifest)} rows")


if __name__ == "__main__":
    main()
