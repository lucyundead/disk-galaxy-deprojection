from __future__ import annotations

import argparse
from pathlib import Path

from dgdp.bar_catalog import join_bar_catalog
from dgdp.tng50_catalog import build_candidate_manifest, read_group_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tng-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=int, default=99)
    parser.add_argument("--bar-catalog", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--min-star-particles", type=int, default=5000)
    parser.add_argument("--min-stellar-mass-msun", type=float, default=1.0e9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = read_group_catalog(args.tng_root / f"groups_{args.snapshot:03d}", snapshot=args.snapshot)
    manifest = build_candidate_manifest(
        catalog,
        min_star_particles=args.min_star_particles,
        min_stellar_mass_msun=args.min_stellar_mass_msun,
        max_candidates=args.max_candidates,
        snapshot=args.snapshot,
    )
    if args.bar_catalog is not None and args.bar_catalog.exists():
        manifest = join_bar_catalog(manifest, args.bar_catalog)
        if "bar_type" in manifest:
            manifest = manifest[manifest["bar_type"].fillna(0).astype(int) > 0]
        elif "bar_a2_catalog" in manifest:
            manifest = manifest[manifest["bar_a2_catalog"].fillna(0.0).astype(float) > 0.2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"wrote TNG50 manifest to {args.output} with {len(manifest)} rows")


if __name__ == "__main__":
    main()
