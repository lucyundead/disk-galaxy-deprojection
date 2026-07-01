import numpy as np

from dgdp.features import make_features


def test_features_match_golden():
    g = np.load("tests/data/features_golden.npz")
    feat = make_features(g["img"], g["meta"], baseline_grid_mass_msun=g["bg"],
                         image_feature_size=24, central_pixel_scale_kpc=0.35)
    assert feat.shape == g["feat"].shape
    np.testing.assert_allclose(feat, g["feat"], rtol=1e-6, atol=1e-6)
