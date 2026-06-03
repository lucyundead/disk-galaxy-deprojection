from __future__ import annotations

import numpy as np
import pandas as pd


def _split_for_index(index: int, n_galaxies: int) -> str:
    train_end = int(round(0.60 * n_galaxies))
    val_end = int(round(0.80 * n_galaxies))
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def make_synthetic_manifest(
    *,
    seed: int,
    n_galaxies: int,
    projections_per_galaxy: int,
    max_inclination_deg: float = 60.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for galaxy_index in range(n_galaxies):
        galaxy_id = int(10_000 + galaxy_index)
        split = _split_for_index(galaxy_index, n_galaxies)
        for projection_id in range(projections_per_galaxy):
            rows.append(
                {
                    "galaxy_id": galaxy_id,
                    "projection_id": int(projection_id),
                    "split": split,
                    "seed": int(seed + galaxy_index * 100 + projection_id),
                    "inclination_deg": float(rng.uniform(0.0, max_inclination_deg)),
                    "disk_pa_deg": float(rng.uniform(0.0, 180.0)),
                    "bar_angle_deg": float(rng.uniform(0.0, 180.0)),
                }
            )
    return pd.DataFrame(rows)


def validate_split_by_galaxy(manifest: pd.DataFrame) -> bool:
    split_counts = manifest.groupby("galaxy_id")["split"].nunique()
    return bool((split_counts == 1).all())
