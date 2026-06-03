import numpy as np

from dgdp.train import make_feature_matrix, standardize_train_apply


def test_feature_matrix_combines_image_baseline_and_metadata():
    images = np.ones((2, 4, 4), dtype=np.float32)
    baseline = np.ones((2, 3), dtype=np.float32) * 2.0
    metadata = np.ones((2, 3), dtype=np.float32) * 3.0

    x = make_feature_matrix(images, baseline, metadata)

    assert x.shape == (2, 22)
    assert np.allclose(x[:, :16], 1.0)
    assert np.allclose(x[:, 16:19], 2.0)
    assert np.allclose(x[:, 19:], 3.0)


def test_standardize_train_apply_returns_finite_arrays():
    train = np.array([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32)
    test = np.array([[5.0, 10.0]], dtype=np.float32)

    train_z, test_z, mean, scale = standardize_train_apply(train, test)

    assert np.all(np.isfinite(train_z))
    assert np.all(np.isfinite(test_z))
    assert mean.shape == (2,)
    assert scale.shape == (2,)
