import numpy as np

from dgdp.model import DeprojectionModel, save_bundle


def test_bundle_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    cfg = dict(heights=np.geomspace(0.2, 3.5, 7),
               alloc={0: (12, 7), 2: (8, 7), 4: (6, 7)},
               r_min=0.12, r_max=15.0,
               grid=dict(r_min=0.05, r_max=30.0, n_r=64, n_phi=48, z_max=5.0, n_z=32),
               image_feature_size=24, central_pixel_scale_kpc=0.35,
               img_mass_median=4.75e10, base_mass_median=3.0e10)
    din, hid, ncomp, tgt = 586, 128, 32, 280
    mlp = dict(W1=rng.normal(size=(hid, din)), b1=np.zeros(hid),
               W2=rng.normal(size=(hid, hid)), b2=np.zeros(hid),
               Wm=rng.normal(size=(ncomp, hid)), bm=np.zeros(ncomp))   # means head -> PCA scores
    pca = dict(vec=rng.normal(size=(ncomp, tgt)), mean=np.zeros(tgt),
               y_mean=np.zeros(ncomp), y_std=np.ones(ncomp))
    feat = dict(mean=np.zeros(din), scale=np.ones(din))
    p = tmp_path / "m.npz"
    save_bundle(str(p), cfg=cfg, mlp=mlp, pca=pca, feat=feat)
    m = DeprojectionModel.load(str(p))
    w = m.predict_weights(rng.normal(size=(1, din)).astype(np.float32))
    assert w.shape == (1, tgt)
    assert m.heights.shape == (7,) and m.k_by_m == {0: 7, 2: 7, 4: 7}
