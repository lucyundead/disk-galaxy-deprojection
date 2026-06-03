from __future__ import annotations

import numpy as np

from dgdp.types import Geometry, SummaryVector


def build_residual_row(
    *,
    galaxy_id: int,
    projection_id: int,
    split: str,
    image: np.ndarray,
    true_summary: SummaryVector,
    baseline_summary: SummaryVector,
    geometry: Geometry,
) -> dict[str, object]:
    if true_summary.names != baseline_summary.names:
        raise ValueError("true_summary and baseline_summary must use the same names")
    return {
        "galaxy_id": int(galaxy_id),
        "projection_id": int(projection_id),
        "split": str(split),
        "image": np.asarray(image, dtype=np.float32),
        "baseline": np.asarray(baseline_summary.values, dtype=np.float32),
        "truth": np.asarray(true_summary.values, dtype=np.float32),
        "delta": np.asarray(true_summary.values - baseline_summary.values, dtype=np.float32),
        "metadata": np.asarray(
            [geometry.inclination_deg, geometry.disk_pa_deg, geometry.bar_angle_deg],
            dtype=np.float32,
        ),
        "summary_names": true_summary.names,
    }
