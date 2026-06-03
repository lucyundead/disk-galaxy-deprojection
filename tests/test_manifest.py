import pandas as pd

from dgdp.manifest import make_synthetic_manifest, validate_split_by_galaxy


def test_manifest_has_projection_rows():
    manifest = make_synthetic_manifest(seed=51, n_galaxies=5, projections_per_galaxy=4)

    assert len(manifest) == 20
    assert set(manifest["split"]) == {"train", "val", "test"}
    assert {
        "galaxy_id",
        "projection_id",
        "inclination_deg",
        "disk_pa_deg",
        "bar_angle_deg",
    }.issubset(manifest.columns)


def test_validate_split_by_galaxy_rejects_leakage():
    bad = pd.DataFrame(
        {
            "galaxy_id": [1, 1],
            "projection_id": [0, 1],
            "split": ["train", "test"],
        }
    )

    assert validate_split_by_galaxy(bad) is False
