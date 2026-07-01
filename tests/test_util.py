import numpy as np

from dgdp.util import interp_matrix


def test_interp_matrix_linear():
    src = np.array([0.0, 1.0, 2.0])
    dst = np.array([0.5, 1.5])
    w = interp_matrix(src, dst)
    assert w.shape == (2, 3)
    np.testing.assert_allclose(w @ src, dst)          # reproduces linear values
    assert np.allclose(w.sum(axis=1), 1.0)            # partition of unity


def test_interp_matrix_clamps():
    src = np.array([0.0, 1.0, 2.0])
    w = interp_matrix(src, np.array([-1.0, 5.0]))     # outside range
    np.testing.assert_allclose(w @ src, [0.0, 2.0])   # clamped to ends
