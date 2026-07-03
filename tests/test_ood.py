import numpy as np

from dgdp.ood import _reference, ood_check


def test_ood_check_center_far_and_mismatch():
    ref = _reference()
    fm, fs = ref["feat_mean"], ref["feat_scale"]
    center = (ref["mu"] * fs + fm)[None, :]                  # the cloud mean itself
    got = ood_check(center, fm, fs)
    assert got["percentile"] <= 5.0 and got["d2"] < 1e-6
    assert len(got["nearest_train_ids"]) == 5
    assert len(set(got["nearest_train_ids"])) == 5           # deduplicated by galaxy

    far = ((ref["mu"] + 50.0 * ref["lam"][0] * ref["comps"][0]) * fs + fm)[None, :]
    assert ood_check(far, fm, fs)["percentile"] == 100.0     # 50 sigma along PC1

    assert ood_check(np.zeros((1, 26)), np.zeros(26), np.ones(26)) is None   # wrong dim
    assert ood_check(center, fm + 1.0, fs) is None           # standardisation mismatch
