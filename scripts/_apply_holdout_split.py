"""Force a galaxy into the held-out test split in a density residual table.

The cluster table build copies the split from the summary residual_table, so a
manifest-level holdout does not propagate. This rewrites the table with the
chosen subhalo's rows reassigned to 'test' (all its projections move together,
so the galaxy-level split stays leak-free), which is what the PCA fit and MDN
training read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-subhalo", type=int, required=True)
    args = parser.parse_args()

    src = np.load(args.input)
    data = {k: src[k] for k in src.files}
    split = data["split"].astype("<U5").copy()
    mask = data["galaxy_id"] == args.holdout_subhalo
    if mask.sum() == 0:
        raise SystemExit(f"subhalo {args.holdout_subhalo} not in table")
    split[mask] = "test"
    data["split"] = split
    np.savez(args.output, **data)

    counts = {s: int((split == s).sum()) for s in ("train", "val", "test")}
    print(f"reassigned {int(mask.sum())} rows of subhalo {args.holdout_subhalo} to test")
    print(f"row split counts: {counts}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
