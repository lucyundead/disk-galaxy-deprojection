import numpy as np

from dgdp.dataset import build_residual_row
from dgdp.types import Geometry, SummaryVector


def test_residual_row_contains_baseline_truth_and_delta():
    true = SummaryVector(names=("a", "b"), values=np.array([3.0, 7.0]))
    base = SummaryVector(names=("a", "b"), values=np.array([1.0, 10.0]))
    image = np.ones((4, 4), dtype=float)
    geom = Geometry(inclination_deg=30.0, disk_pa_deg=20.0, bar_angle_deg=10.0)

    row = build_residual_row(
        galaxy_id=1,
        projection_id=2,
        split="train",
        image=image,
        true_summary=true,
        baseline_summary=base,
        geometry=geom,
    )

    assert row["delta"].tolist() == [2.0, -3.0]
    assert row["summary_names"] == ("a", "b")
    assert row["metadata"].tolist() == [30.0, 20.0, 10.0]
