# dgdp End-to-End Deprojection Package — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a pip-installable `dgdp` package that turns a galaxy image (FITS/array) + geometry + M/L into a deprojected 3D stellar-mass cube, rotation curve, renderings, and (optional AGAMA) potential — using a bundled torch-free pretrained model.

**Architecture:** Move the inference pipeline out of `scripts/` into the installed `dgdp/` package (`image`→`features`→`model`→`harmonics`/`vertical_mixture`→`deproject`/`rotation`). The trained MDN's mean head is exported to numpy so inference needs only numpy+astropy+matplotlib; the ~1 MB R=64 model bundle ships in the wheel.

**Tech Stack:** Python ≥3.11, numpy, astropy (FITS), matplotlib; setuptools src-layout. Training-only: torch, scipy (cluster PBS). Optional: AGAMA (manual build).

**Spec:** `docs/superpowers/specs/2026-06-30-dgdp-package-design.md`. Read it first.

**Conventions for this plan:**
- Run everything in WSL `openSUSE-Tumbleweed`, project `/home/lucyundead/projects/disk-galaxy-deprojection`, interpreter `.venv/bin/python`, `PYTHONPATH=src`.
- "Port verbatim" = copy the named function(s) unchanged except imports; it is pure and already tested in situ. Keep the research scripts untouched.
- After each task: `ruff check .` clean and the named tests green. Commit at the end of each task.
- Golden tests pin the ported inference to the current script's numbers so the OOD behaviour cannot silently drift.

---

## Task 1: `dgdp.util` — the interpolation matrix (shared by inference)

`r_resample` needs `_interp_matrix`, which today lives in `scripts/deproject_fourier_rz_compare.py:52-62`. Move it into the package so inference has no `scripts/` dependency.

**Files:**
- Create: `src/dgdp/util.py`
- Test: `tests/test_util.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/test_util.py
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
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError: dgdp.util`)
Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_util.py -q`

- [ ] **Step 3: Implement** — copy `_interp_matrix` verbatim, rename public:
```python
# src/dgdp/util.py
from __future__ import annotations
import numpy as np

def interp_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """1-D linear-interpolation weight matrix (len(dst), len(src)); dst is clamped."""
    src = np.asarray(src, dtype=float)
    dst = np.clip(np.asarray(dst, dtype=float), src[0], src[-1])
    idx = np.clip(np.searchsorted(src, dst) - 1, 0, len(src) - 2)
    frac = (dst - src[idx]) / (src[idx + 1] - src[idx])
    weights = np.zeros((len(dst), len(src)))
    rows = np.arange(len(dst))
    weights[rows, idx] = 1.0 - frac
    weights[rows, idx + 1] = frac
    return weights
```

- [ ] **Step 4: Run — expect PASS.** `PYTHONPATH=src .venv/bin/python -m pytest tests/test_util.py -q`
- [ ] **Step 5: Commit** — `git add src/dgdp/util.py tests/test_util.py && git commit -m "feat(dgdp): interp_matrix util for inference"`

---

## Task 2: `dgdp.harmonics` — even-m harmonics, R-resample, density reconstruct

Port the azimuthal-harmonic + reconstruct logic (currently `scripts/deproject_fourier_rz_conserving.py`: `EVEN_M` l.42, `harmonics` l.45-50, `r_resample` l.53-56; and the NGC `reconstruct_smooth` `scripts/deproject_real_image_ngc4321_learned.py:71-101`). Inference-only, numpy.

**Files:**
- Create: `src/dgdp/harmonics.py`
- Test: `tests/test_harmonics.py`

- [ ] **Step 1: Write the failing test** — conservation + shape on a synthetic axisym+bar field:
```python
# tests/test_harmonics.py
import numpy as np
from dgdp.harmonics import EVEN_M, harmonics, r_resample, reconstruct_density
from dgdp import vertical_mixture as vm

def test_harmonics_sigma_is_zintegral():
    nR, nphi, nz = 6, 8, 12
    z = np.linspace(-5, 5, nz); dz = z[1] - z[0]
    field = np.abs(np.random.default_rng(0).normal(1, .1, (1, nR, nphi, nz)))
    a, sigma = harmonics(field, dz)
    np.testing.assert_allclose(sigma[0], field.mean(axis=2).sum(axis=2) * dz * nphi / nphi, rtol=1e-5)
    # m=0 Sigma equals the phi-summed z-integral / nphi * nphi -> just the axisym z-integral
    assert set(a) == set(EVEN_M)

def test_reconstruct_density_conserves_column():
    # one row, image anchor = flat Sigma; mixture weights on a fixed dictionary
    r_grid = np.linspace(0.5, 20, 16); z_grid = np.linspace(-5, 5, 32)
    phi = np.linspace(-np.pi, np.pi, 48, endpoint=False)
    heights = np.geomspace(0.2, 3.5, 7)
    rk = {0: np.geomspace(0.12, 15, 12), 2: np.geomspace(0.12, 15, 8), 4: np.geomspace(0.12, 15, 6)}
    K = len(heights)
    # weight vector: m=0 uniform positive, m>0 zero -> pure axisym disk
    vec = []
    for m in EVEN_M:
        w = np.zeros((1, len(rk[m]), K)); w[..., 1] = 1.0        # all weight on 2nd height
        vec.append(w.reshape(1, -1))
        if m != 0: vec.append(np.zeros((1, len(rk[m]) * K)))
    vec = np.concatenate(vec, axis=1)
    anchor = {0: np.ones((1, len(r_grid)), complex), 2: np.zeros((1, len(r_grid)), complex),
              4: np.zeros((1, len(r_grid)), complex)}
    rho = reconstruct_density(vec, anchor, rk, {m: K for m in EVEN_M}, heights,
                              r_grid, z_grid, phi)              # (1,nR,nphi,nz) cell density
    col = rho[0].sum(axis=(1, 2)) * (z_grid[1] - z_grid[0])     # int over phi,z ~ nphi*Sigma
    assert np.all(rho >= 0)
    assert np.allclose(col / col.mean(), 1.0, atol=1e-6)        # flat anchor -> flat column
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError: dgdp.harmonics`).
Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_harmonics.py -q`

- [ ] **Step 3: Implement** — port + adapt (`reconstruct_density` = the NGC `reconstruct_smooth` body, returning cell **density** not mass so the caller multiplies by vol once):
```python
# src/dgdp/harmonics.py
from __future__ import annotations
import numpy as np
from dgdp import vertical_mixture as vm
from dgdp.util import interp_matrix

EVEN_M = (0, 2, 4)

def harmonics(field: np.ndarray, dz: float):
    """Even-m azimuthal harmonics a_m(R,z) and z-integral Sigma_m(R). field: (rows,nR,nphi,nz)."""
    coeff = np.fft.rfft(field, axis=2) / field.shape[2]
    a = {m: coeff[:, :, m, :].astype(np.complex64) for m in EVEN_M}
    sigma = {m: (a[m].sum(axis=2) * dz) for m in EVEN_M}
    return a, sigma

def r_resample(cmap: np.ndarray, r_src: np.ndarray, r_dst: np.ndarray) -> np.ndarray:
    """Resample only the R axis of (rows,R,z) maps (z handled by the mixture)."""
    return np.einsum("Rr,nrz->nRz", interp_matrix(r_src, r_dst), cmap)

def reconstruct_density(vec_rows, anchor, rk_by_m, k_by_m, heights,
                        r_grid, z_grid, phi_centers) -> np.ndarray:
    """Predicted mixture weights -> q_m(z;R) -> anchor Sigma_m(R) -> cell DENSITY (rows,nR,nphi,nz).

    q is z-symmetric, >=0 (m=0), int q dz=1 by construction; linear R-interp preserves the unit
    integral so anchoring by Sigma_m(R) conserves the column mass exactly. Non-negative clipped.
    """
    rows = vec_rows.shape[0]
    rho = np.zeros((rows, len(r_grid), len(phi_centers), len(z_grid)))
    i = 0
    for m in EVEN_M:
        rk, kk = rk_by_m[m], k_by_m[m]
        size = len(rk) * kk
        wre = vec_rows[:, i:i + size].reshape(rows, len(rk), kk); i += size
        if m == 0:
            w = wre.astype(np.complex128)
        else:
            wim = vec_rows[:, i:i + size].reshape(rows, len(rk), kk); i += size
            w = wre + 1j * wim
        q_rk = vm.reconstruct(w, z_grid, heights, signed=(m != 0))
        q_grid = r_resample(q_rk, rk, r_grid)
        a_m = anchor[m][:, :, None] * q_grid
        if m == 0:
            rho += a_m.real[:, :, None, :]
        else:
            cos_m, sin_m = np.cos(m * phi_centers), np.sin(m * phi_centers)
            rho += 2.0 * (a_m.real[:, :, None, :] * cos_m[None, None, :, None]
                          - a_m.imag[:, :, None, :] * sin_m[None, None, :, None])
    return np.clip(rho, 0.0, None)
```

- [ ] **Step 4: Run — expect PASS.** `PYTHONPATH=src .venv/bin/python -m pytest tests/test_harmonics.py -q`
- [ ] **Step 5: Commit** — `git add src/dgdp/harmonics.py tests/test_harmonics.py && git commit -m "feat(dgdp): harmonics + density reconstruct for inference"`

---

## Task 3: `dgdp.features` — image → feature vector (golden-pinned)

Port `make_density_residual_features`, `average_pool_images`, `make_central_image_features` from `scripts/train_density_residual_pca.py` (features l.107-141; helpers `average_pool_images` l.~45-54 and `make_central_image_features` — read the file for exact ranges). Predictions are OOD-sensitive to these, so pin with a golden test against the current script output.

**Files:**
- Create: `src/dgdp/features.py`
- Test: `tests/test_features_golden.py`

- [ ] **Step 1: Golden fixture** — generate once from the CURRENT script functions and save:
```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import numpy as np, sys; sys.path.insert(0, "scripts")
from train_density_residual_pca import make_density_residual_features
rng = np.random.default_rng(1)
img = np.abs(rng.normal(1, .2, (3, 96, 96))).astype(np.float32)
meta = rng.uniform(10, 60, (3, 3)).astype(np.float32)
bg = np.abs(rng.normal(1e6, 1e5, 3)).astype(np.float32)
feat = make_density_residual_features(img, meta, baseline_grid_mass_msun=bg,
                                      image_feature_size=24, central_pixel_scale_kpc=0.35)
np.savez("tests/data/features_golden.npz", img=img, meta=meta, bg=bg, feat=feat)
print("feat shape", feat.shape)
PY
```

- [ ] **Step 2: Write the failing test**
```python
# tests/test_features_golden.py
import numpy as np
from dgdp.features import make_features

def test_features_match_golden():
    g = np.load("tests/data/features_golden.npz")
    feat = make_features(g["img"], g["meta"], baseline_grid_mass_msun=g["bg"],
                         image_feature_size=24, central_pixel_scale_kpc=0.35)
    assert feat.shape == g["feat"].shape
    np.testing.assert_allclose(feat, g["feat"], rtol=1e-6, atol=1e-6)
```

- [ ] **Step 3: Run — expect FAIL** (`ModuleNotFoundError: dgdp.features`).
- [ ] **Step 4: Implement** — copy the three functions verbatim into `src/dgdp/features.py` (rename `make_density_residual_features`→`make_features`, keep helpers private `_average_pool_images`, `_make_central_image_features`). No other change; they are pure numpy. Read `scripts/train_density_residual_pca.py` for the exact bodies and copy them.
- [ ] **Step 5: Run — expect PASS.** `PYTHONPATH=src .venv/bin/python -m pytest tests/test_features_golden.py -q`
- [ ] **Step 6: Commit** — `git add src/dgdp/features.py tests/test_features_golden.py tests/data/features_golden.npz && git commit -m "feat(dgdp): image feature builder (golden-pinned)"`

---

## Task 4: `dgdp.image` — FITS/array load + geometry + geometric baseline

Port `deproject_ngc4321` + `onsky_pa` + `pixel_pa_from_onsky` from `scripts/deproject_real_image_ngc4321_learned.py:57-188`. Split into `load_image` (I/O + geometry) and `geometric_baseline` (deprojection → baseline density + TNG-format image). Reuse `dgdp.density3d`.

**Files:**
- Create: `src/dgdp/image.py`
- Test: `tests/test_image.py`

- [ ] **Step 1: Write the failing test** — synthetic exponential disk FITS, explicit geometry, assert baseline conserves the requested mass and the TNG image is 192²:
```python
# tests/test_image.py
import numpy as np
from astropy.io import fits
from dgdp.image import load_image, geometric_baseline
from dgdp.density3d import make_cylindrical_grid_spec, cylindrical_bin_volumes

def test_geometric_baseline_mass_and_shape(tmp_path):
    ny = nx = 200
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - 100, yy - 100)
    data = 100.0 * np.exp(-r / 20.0)                       # face-on exp disk
    p = tmp_path / "disk.fits"; fits.PrimaryHDU(data.astype(np.float32)).writeto(p)
    spec = make_cylindrical_grid_spec(z_max_kpc=5.0, n_z=32)
    gi = load_image(str(p), pix_arcsec=1.0, pa_pix_deg=0.0, center=(100, 100),
                    inclination_deg=0.0, distance_mpc=15.0)
    base = geometric_baseline(gi, spec, scale_height_kpc=0.3, stellar_mass=5e10)
    vol = cylindrical_bin_volumes(spec)
    assert np.isclose((base["baseline_density"] * vol).sum(), 5e10, rtol=1e-6)
    assert base["image_tng"].shape == (192, 192)
    assert base["baseline_density"].shape == (spec.r_edges_kpc.size - 1, 48, 32)
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** — `load_image(source, *, pix_arcsec=None, pa_pix_deg=None, pa_onsky_deg=None, center=None, inclination_deg, distance_mpc, mask=None)` returns a `GalaxyImage` dataclass (`light`, `cx`, `cy`, `pix_kpc`, `pa_pix`, `incl`, `bkg`). `geometric_baseline(gi, spec, *, scale_height_kpc, stellar_mass)` returns the dict `{baseline_density, image_tng, M_star}`. Copy the body of `deproject_ngc4321` (l.125-188) split across the two functions; copy `onsky_pa`/`pixel_pa_from_onsky` for the WCS PA path (used only when `pa_pix_deg is None and pa_onsky_deg is not None`). `source` may be a FITS path or a 2-D array (skip `fits.open` for arrays; require `pix_arcsec` then).
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** — `git add src/dgdp/image.py tests/test_image.py && git commit -m "feat(dgdp): image load + geometric baseline"`

---

## Task 5: `dgdp.model` — bundle format + torch-free MLP forward + save

`DeprojectionModel.load(path)` loads the `.npz` bundle and predicts the mixture weight vector in pure numpy. `save_bundle(...)` is used by the trainer (Task 8). Verify the numpy forward against a tiny torch MDN (parity), gated behind the `[train]` extra.

**Files:**
- Create: `src/dgdp/model.py`
- Test: `tests/test_model_export.py` (parity, needs torch), `tests/test_model_roundtrip.py` (save/load, numpy only)

- [ ] **Step 1: Write the numpy-only round-trip test first**
```python
# tests/test_model_roundtrip.py
import numpy as np
from dgdp.model import DeprojectionModel, save_bundle

def test_bundle_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    cfg = dict(heights=np.geomspace(0.2, 3.5, 7),
               alloc={0: (12, 7), 2: (8, 7), 4: (6, 7)},
               r_min=0.12, r_max=15.0, grid=dict(r_min=0.05, r_max=30.0, n_r=64, n_phi=48,
                                                 z_max=5.0, n_z=32),
               image_feature_size=24, central_pixel_scale_kpc=0.35,
               img_mass_median=4.75e10, base_mass_median=3.0e10)
    din, hid, tgt = 586, 128, 280
    mlp = dict(W1=rng.normal(size=(hid, din)), b1=np.zeros(hid),
               W2=rng.normal(size=(hid, hid)), b2=np.zeros(hid),
               Wm=rng.normal(size=(tgt, hid)), bm=np.zeros(tgt))
    pca = dict(vec=rng.normal(size=(32, tgt)), mean=np.zeros(tgt),
               y_mean=np.zeros(32), y_std=np.ones(32))
    feat = dict(mean=np.zeros(din), scale=np.ones(din))
    p = tmp_path / "m.npz"
    save_bundle(str(p), cfg=cfg, mlp=mlp, pca=pca, feat=feat)
    m = DeprojectionModel.load(str(p))
    w = m.predict_weights(rng.normal(size=(1, din)).astype(np.float32))
    assert w.shape == (1, tgt)
    assert m.heights.shape == (7,) and m.k_by_m == {0: 7, 2: 7, 4: 7}
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement**
```python
# src/dgdp/model.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from dgdp.fourier_rz import r_knots

def _relu(x): return np.maximum(x, 0.0)

def save_bundle(path, *, cfg, mlp, pca, feat):
    """Persist a trained bundle. cfg is JSON-able; arrays saved by key."""
    import json
    np.savez(path, config=json.dumps(cfg, default=lambda a: np.asarray(a).tolist()),
             W1=mlp["W1"], b1=mlp["b1"], W2=mlp["W2"], b2=mlp["b2"],
             Wm=mlp["Wm"], bm=mlp["bm"],
             pca_vec=pca["vec"], pca_mean=pca["mean"], y_mean=pca["y_mean"], y_std=pca["y_std"],
             feat_mean=feat["mean"], feat_scale=feat["scale"])

@dataclass
class DeprojectionModel:
    W1: np.ndarray; b1: np.ndarray; W2: np.ndarray; b2: np.ndarray
    Wm: np.ndarray; bm: np.ndarray
    pca_vec: np.ndarray; pca_mean: np.ndarray; y_mean: np.ndarray; y_std: np.ndarray
    feat_mean: np.ndarray; feat_scale: np.ndarray
    heights: np.ndarray; rk_by_m: dict; k_by_m: dict
    grid: dict; image_feature_size: int; central_pixel_scale_kpc: float
    img_mass_median: float; base_mass_median: float

    @classmethod
    def load(cls, path: str) -> "DeprojectionModel":
        import json
        z = np.load(path, allow_pickle=False)
        cfg = json.loads(str(z["config"]))
        heights = np.asarray(cfg["heights"], float)
        alloc = {int(m): tuple(v) for m, v in cfg["alloc"].items()}
        rk_by_m = {m: r_knots(nr, cfg["r_max"], cfg["r_min"]) for m, (nr, _k) in alloc.items()}
        k_by_m = {m: len(heights) for m in alloc}
        return cls(z["W1"], z["b1"], z["W2"], z["b2"], z["Wm"], z["bm"],
                   z["pca_vec"], z["pca_mean"], z["y_mean"], z["y_std"],
                   z["feat_mean"], z["feat_scale"], heights, rk_by_m, k_by_m,
                   cfg["grid"], int(cfg["image_feature_size"]),
                   float(cfg["central_pixel_scale_kpc"]),
                   float(cfg["img_mass_median"]), float(cfg["base_mass_median"]))

    def predict_weights(self, features: np.ndarray) -> np.ndarray:
        """features (rows, D) raw -> standardize -> MLP mean head -> PCA inverse -> weight vec."""
        x = (np.asarray(features, float) - self.feat_mean) / self.feat_scale
        h = _relu(x @ self.W1.T + self.b1)
        h = _relu(h @ self.W2.T + self.b2)
        scores = h @ self.Wm.T + self.bm                      # (rows, n_comp)  (means head)
        return (scores * self.y_std + self.y_mean) @ self.pca_vec + self.pca_mean
```
Note: `Wm/bm` are the MDN `means` layer sliced to the single component; `n_comp` here = PCA components (32). See Task 8 for how these are extracted from torch.

- [ ] **Step 4: Run round-trip — expect PASS.** `PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_roundtrip.py -q`
- [ ] **Step 5: Write parity test** (torch, `[train]` extra):
```python
# tests/test_model_export.py
import numpy as np, pytest
torch = pytest.importorskip("torch")
from dgdp.models.mdn import SummaryResidualMDN
from dgdp.model import _relu

def test_numpy_meanhead_matches_torch():
    d_in, hid, out = 20, 128, 8
    mdn = SummaryResidualMDN(d_in, out, hidden_dim=hid, n_components=1).eval()
    x = torch.randn(5, d_in)
    with torch.no_grad():
        _, means, _ = mdn.forward(x)          # (5,1,out) deterministic mean head
    W1 = mdn.net[0].weight.detach().numpy(); b1 = mdn.net[0].bias.detach().numpy()
    W2 = mdn.net[2].weight.detach().numpy(); b2 = mdn.net[2].bias.detach().numpy()
    Wm = mdn.means.weight.detach().numpy(); bm = mdn.means.bias.detach().numpy()
    xn = x.numpy()
    h = _relu(_relu(xn @ W1.T + b1) @ W2.T + b2)
    got = h @ Wm.T + bm
    np.testing.assert_allclose(got, means[:, 0, :].numpy(), rtol=1e-5, atol=1e-5)
```
- [ ] **Step 6: Run parity — expect PASS.** `PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_export.py -q`
- [ ] **Step 7: Commit** — `git add src/dgdp/model.py tests/test_model_roundtrip.py tests/test_model_export.py && git commit -m "feat(dgdp): torch-free model bundle load + MLP mean head"`

---

## Task 6: `dgdp.rotation` — direct-sum v_c and optional AGAMA potential

Port `direct_vc` from `scripts/deproject_real_image_ngc4321_learned.py:106-122` → `v_circ`. Add `potential(...)` that lazily imports agama via the existing `dgdp.agama_density` + `dgdp.fourier_rz.fit_fourier_rz_from_grid`.

**Files:**
- Create: `src/dgdp/rotation.py`
- Test: `tests/test_rotation.py`

- [ ] **Step 1: Write the failing test** — point-mass sanity + AGAMA-absent behaviour:
```python
# tests/test_rotation.py
import numpy as np, pytest
from dgdp.rotation import v_circ, potential

def test_v_circ_point_mass_scaling():
    # a compact central blob -> v_c ~ sqrt(GM/R) falling outward
    r = np.linspace(0.5, 20, 32); phi = np.linspace(-np.pi, np.pi, 24, endpoint=False)
    z = np.linspace(-1, 1, 8)
    mass = np.zeros((r.size, phi.size, z.size)); mass[0] += 1e10   # all mass at inner R
    vc = v_circ(mass, r, phi, z, np.array([2.0, 8.0]))
    assert vc[0] > vc[1] > 0                                       # declining curve

def test_potential_without_agama_raises_clearly(monkeypatch):
    import builtins, importlib
    real = builtins.__import__
    def fake(name, *a, **k):
        if name == "agama": raise ImportError("no agama")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(RuntimeError, match="AGAMA"):
        potential(np.ones((4, 4, 4)), np.linspace(1, 4, 4), np.linspace(-np.pi, np.pi, 4, endpoint=False),
                  np.linspace(-1, 1, 4), total_mass=1e10, query_R=np.array([1.0]), query_z=np.array([0.0]))
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement**
```python
# src/dgdp/rotation.py
from __future__ import annotations
import numpy as np

def v_circ(mass_grid, r_grid, phi_centers, z_grid, radii, *, eps=0.15) -> np.ndarray:
    """Midplane v_c(R) [km/s] by direct softened summation over cells as point masses."""
    grav = 4.300917270e-6  # kpc (km/s)^2 / Msun
    rr, pp, zz = np.meshgrid(r_grid, phi_centers, z_grid, indexing="ij")
    cx, cy, cz = (rr * np.cos(pp)).ravel(), (rr * np.sin(pp)).ravel(), zz.ravel()
    m = mass_grid.ravel(); keep = m > 0
    cx, cy, cz, m = cx[keep], cy[keep], cz[keep], m[keep]
    out = np.empty(len(radii))
    for i, radius in enumerate(radii):
        dx = cx - radius
        inv = (dx * dx + cy * cy + cz * cz + eps * eps) ** -1.5
        accel_x = grav * np.sum(m * inv * dx)
        out[i] = np.sqrt(max(-accel_x * radius, 0.0))
    return out

def potential(mass_grid, r_grid, phi_centers, z_grid, *, total_mass, query_R, query_z):
    """AGAMA CylSpline potential value at (query_R, query_z, phi=0). Requires agama."""
    try:
        import agama  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "AGAMA is required for the potential field but is not importable. Build it from "
            "source (https://github.com/GalacticDynamics-Oxford/Agama) and add it to PYTHONPATH."
        ) from exc
    from dgdp.agama_density import fourier_rz_to_agama_density
    from dgdp.fourier_rz import fit_fourier_rz_from_grid
    vol = None  # caller passes mass; density = mass/vol handled by fit input below
    model = fit_fourier_rz_from_grid(mass_grid, r_grid, z_grid, n_r=25, n_z_half=12,
                                     r_max=float(r_grid[-1]), z_max=float(abs(z_grid).max()))
    dens = fourier_rz_to_agama_density(model, total_mass=float(total_mass),
                                       r_max=float(r_grid[-1]), z_max=float(abs(z_grid).max()))
    import agama
    pot = agama.Potential(type="CylSpline", density=dens)
    pts = np.column_stack([query_R, np.zeros_like(query_R), query_z])
    return pot.potential(pts)
```
(If `fit_fourier_rz_from_grid` expects density not mass, divide `mass_grid` by the cell volumes before the call — check its docstring during implementation and adjust; the golden e2e test in Task 7 will catch a units error via v_c.)

- [ ] **Step 4: Run — expect PASS** (the agama test passes via the raise; skip the value check if agama installed).
- [ ] **Step 5: Commit** — `git add src/dgdp/rotation.py tests/test_rotation.py && git commit -m "feat(dgdp): direct-sum v_c + optional AGAMA potential"`

---

## Task 7: `dgdp.deproject` — orchestration + DeprojectionResult

Tie it together: image → baseline → features (with the 2 absolute features set to the bundle medians for OOD) → model → reconstruct → scale to absolute mass. This mirrors `predict_learned` + the (C)/(rms/vc) blocks of the current NGC script, minus training.

**Files:**
- Create: `src/dgdp/deproject.py`
- Test: `tests/test_deproject_e2e.py`

- [ ] **Step 1: Write the end-to-end test** (uses the bundled model + repo NGC 4321 FITS):
```python
# tests/test_deproject_e2e.py
import numpy as np, pytest
from pathlib import Path
from dgdp import deproject

FITS = Path("NGC4321_m_c_r_f.fits")
pytestmark = pytest.mark.skipif(not FITS.exists(), reason="NGC4321 fits not present")

def test_ngc4321_end_to_end():
    r = deproject(str(FITS), distance_mpc=15.2, inclination_deg=30.0, pa_onsky_deg=153.0,
                  ml=1.0, stellar_mass=6.0e10)
    assert r.density_3d.ndim == 3 and np.all(r.density_3d >= 0) and np.isfinite(r.density_3d).all()
    np.testing.assert_allclose(r.total_mass, 6.0e10, rtol=1e-3)
    radii = np.linspace(1, 15, 40)
    vc = r.v_circ(radii)
    assert 150.0 < vc.max() < 190.0                          # NGC4321 params -> peak ~155-180
    rms = r.rms_z(radii)
    assert rms[radii < 3].mean() > 0.6                        # thick/flaring, not the 0.3 baseline
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError: cannot import name 'deproject'`).
- [ ] **Step 3: Implement `deproject` + `DeprojectionResult`**
```python
# src/dgdp/deproject.py
from __future__ import annotations
from dataclasses import dataclass
from importlib.resources import files
import numpy as np
from dgdp.density3d import make_cylindrical_grid_spec, cylindrical_bin_volumes
from dgdp.image import load_image, geometric_baseline
from dgdp.features import make_features
from dgdp.model import DeprojectionModel
from dgdp.harmonics import EVEN_M, harmonics, reconstruct_density
from dgdp import rotation

_BUNDLED = files("dgdp.models").joinpath("dgdp_fixed_dict.npz")

def _resolve_mass(*, stellar_mass, luminosity, ml, image_light, zeropoint, band_solar_mag,
                  distance_mpc):
    if stellar_mass is not None:
        return float(stellar_mass), False
    if luminosity is not None:
        return float(ml) * float(luminosity), False
    if zeropoint is not None and band_solar_mag is not None:
        # counts -> apparent mag -> absolute -> luminosity (Lsun); documented convention
        flux = float(np.nansum(image_light))
        app_mag = -2.5 * np.log10(max(flux, 1e-30)) + float(zeropoint)
        dist_mod = 5.0 * np.log10(distance_mpc * 1e6) - 5.0
        abs_mag = app_mag - dist_mod
        lum = 10.0 ** (-0.4 * (abs_mag - float(band_solar_mag)))
        return float(ml) * lum, False
    return 1.0, True                                          # relative-only scale

@dataclass
class DeprojectionResult:
    density_3d: np.ndarray                                   # (nR,nphi,nz) Msun (or relative)
    grid: dict
    total_mass: float
    relative: bool
    _vol: np.ndarray

    def v_circ(self, radii_kpc):
        return rotation.v_circ(self.density_3d * self._vol, self.grid["r"], self.grid["phi"],
                               self.grid["z"], np.asarray(radii_kpc, float))
    def rms_z(self, radii_kpc):
        m_rz = (self.density_3d * self._vol).sum(axis=1)      # (R,z)
        tot = np.maximum(m_rz.sum(axis=1), 1e-30)
        rms = np.sqrt((m_rz * self.grid["z"][None, :] ** 2).sum(axis=1) / tot)
        return np.interp(np.asarray(radii_kpc, float), self.grid["r"], rms)
    @property
    def face_on(self):
        return (self.density_3d * self._vol).sum(axis=2)     # (R,phi) column mass
    @property
    def edge_on(self):
        return (self.density_3d * self._vol).sum(axis=1)     # (R,z)
    def potential(self, R, z):
        return rotation.potential(self.density_3d * self._vol, self.grid["r"], self.grid["phi"],
                                  self.grid["z"], total_mass=self.total_mass,
                                  query_R=np.atleast_1d(R), query_z=np.atleast_1d(z))
    def save(self, out_dir):
        import os, csv
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, "density.npz"), density_3d=self.density_3d,
                 r=self.grid["r"], phi=self.grid["phi"], z=self.grid["z"], total_mass=self.total_mass)
        radii = np.linspace(self.grid["r"][0], min(self.grid["r"][-1], 18.0), 80)
        vc = self.v_circ(radii)
        with open(os.path.join(out_dir, "rotation_curve.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["R_kpc", "v_c_kms"]); w.writerows(zip(radii, vc))

def deproject(image, *, distance_mpc, inclination_deg, pa_pix_deg=None, pa_onsky_deg=None,
              center=None, pix_arcsec=None, mask=None, ml=1.0, stellar_mass=None,
              luminosity=None, zeropoint=None, band_solar_mag=None, bar_angle_deg=0.0,
              scale_height_kpc=0.3, model=None) -> DeprojectionResult:
    m = model if isinstance(model, DeprojectionModel) else DeprojectionModel.load(str(model or _BUNDLED))
    g = m.grid
    spec = make_cylindrical_grid_spec(r_min_kpc=g["r_min"], r_max_kpc=g["r_max"], n_r=g["n_r"],
                                      n_phi=g["n_phi"], z_max_kpc=g["z_max"], n_z=g["n_z"])
    r_grid = 0.5 * (spec.r_edges_kpc[:-1] + spec.r_edges_kpc[1:])
    z_grid = 0.5 * (spec.z_edges_kpc[:-1] + spec.z_edges_kpc[1:])
    phi = 0.5 * (spec.phi_edges_rad[:-1] + spec.phi_edges_rad[1:])
    dz = float(np.diff(spec.z_edges_kpc)[0]); vol = cylindrical_bin_volumes(spec).astype(float)

    total_mass, relative = _resolve_mass(stellar_mass=stellar_mass, luminosity=luminosity, ml=ml,
                                         image_light=None, zeropoint=zeropoint,
                                         band_solar_mag=band_solar_mag, distance_mpc=distance_mpc)
    gi = load_image(image, pix_arcsec=pix_arcsec, pa_pix_deg=pa_pix_deg, pa_onsky_deg=pa_onsky_deg,
                    center=center, inclination_deg=inclination_deg, distance_mpc=distance_mpc,
                    mask=mask)
    if relative and zeropoint is not None and band_solar_mag is not None:
        total_mass, relative = _resolve_mass(stellar_mass=None, luminosity=None, ml=ml,
                                             image_light=gi.light, zeropoint=zeropoint,
                                             band_solar_mag=band_solar_mag, distance_mpc=distance_mpc)
    base = geometric_baseline(gi, spec, scale_height_kpc=scale_height_kpc,
                              stellar_mass=max(total_mass, 1.0))
    _, sigma_img = harmonics(base["baseline_density"][None], dz)
    anchor = {mm: sigma_img[mm] for mm in EVEN_M}

    img = base["image_tng"]
    feat = make_features(img[None], np.array([[inclination_deg, 0.0, bar_angle_deg]], np.float32),
                         baseline_grid_mass_msun=np.array([m.base_mass_median], np.float32),
                         image_feature_size=m.image_feature_size,
                         central_pixel_scale_kpc=m.central_pixel_scale_kpc)
    # OOD: substitute the two absolute mass features with the TNG-train medians (see spec §4)
    feat[0, -2:] = [np.log10(max(m.img_mass_median, 1.0)), np.log10(max(m.base_mass_median, 1.0))]
    vec = m.predict_weights(feat)
    rho = reconstruct_density(vec, anchor, m.rk_by_m, m.k_by_m, m.heights, r_grid, z_grid, phi)[0]
    mass = rho * vol
    if not relative:
        mass *= total_mass / max(mass.sum(), 1e-30)
    density = mass / vol
    return DeprojectionResult(density, {"r": r_grid, "phi": phi, "z": z_grid},
                              float(mass.sum()), relative, vol)
```
(During implementation, confirm the feature layout puts the two absolute-mass features last; if not, set them by the indices `make_features` uses — check Task 3's source. The golden test in Task 3 fixes the layout.)

- [ ] **Step 4: Run — expect PASS** once Task 8 has produced the bundle. Until then, expect skip/known-fail; re-run after Task 8.
- [ ] **Step 5: Export `deproject` in `src/dgdp/__init__.py`** — add `from dgdp.deproject import deproject, DeprojectionResult`.
- [ ] **Step 6: Commit** — `git add src/dgdp/deproject.py src/dgdp/__init__.py tests/test_deproject_e2e.py && git commit -m "feat(dgdp): deproject() orchestration + result"`

---

## Task 8: Trainer + R=64 bundle (cluster)

`scripts/train_deprojection_model.py` trains the fixed-dict q_m head on the R=64 table (reusing `make_features` from `dgdp`, `vertical_mixture.weights_target`, and `fit_pca`/`train_mdn` from `scripts/deproject_fourier_rz_compare.py`), extracts the MDN mean-head weights + PCA + featnorm + config, and writes the bundle via `dgdp.model.save_bundle`. Runs on the cluster (31 GB table); fetch the ~1 MB `.npz`.

**Files:**
- Create: `scripts/train_deprojection_model.py`
- Modify: `scripts/_milestone2d_richgrid_cluster.py` (add a `train-bundle` PBS mode mirroring `retrain`)
- Add (generated): `src/dgdp/models/dgdp_fixed_dict.npz`

- [ ] **Step 1: Write the trainer** — it must reproduce the NGC training path exactly (features, target = `weights_target(a_rk, z_grid, heights, signed)`, PCA(32), MDN `train_mdn`), then export:
```python
# key export block (after training `mdn`, PCA vec/mean, y_mean/std, feat_mean/scale on the R=64 table)
import numpy as np
from dgdp.model import save_bundle
mlp = dict(W1=mdn.net[0].weight.detach().numpy(), b1=mdn.net[0].bias.detach().numpy(),
           W2=mdn.net[2].weight.detach().numpy(), b2=mdn.net[2].bias.detach().numpy(),
           Wm=mdn.means.weight.detach().numpy(), bm=mdn.means.bias.detach().numpy())
pca = dict(vec=vec, mean=mean, y_mean=y_mean, y_std=y_std)
feat = dict(mean=feat_mean, scale=feat_scale)
cfg = dict(heights=heights.tolist(), alloc={0:[12,7],2:[8,7],4:[6,7]}, r_min=0.12, r_max=15.0,
           grid=dict(r_min=0.05, r_max=30.0, n_r=64, n_phi=48, z_max=5.0, n_z=32),
           image_feature_size=24, central_pixel_scale_kpc=0.35,
           img_mass_median=float(np.median(table["images"].astype(np.float64).sum(axis=(1,2))[train])),
           base_mass_median=float(np.median(table["baseline_grid_mass_msun"].astype(np.float64)[train])))
save_bundle("src/dgdp/models/dgdp_fixed_dict.npz", cfg=cfg, mlp=mlp, pca=pca, feat=feat)
```
Read `scripts/deproject_real_image_ngc4321_learned.py:231-260` for the exact feature/target/PCA/MDN training lines to reuse (drop the NGC application part).

- [ ] **Step 2: Add cluster mode** — in `scripts/_milestone2d_richgrid_cluster.py`, add a `train-bundle` mode: a `pbs_single` job (mem=160gb, ppn=8, `export OMP_NUM_THREADS=8 …`) running `python scripts/train_deprojection_model.py --table {OUT_FULL}/density_residual_table.npz --out src/dgdp/models/dgdp_fixed_dict.npz`, plus a dispatch branch. Mirror the existing `retrain` function.
- [ ] **Step 3: Sync + run on cluster**
```bash
.venv/bin/python scripts/cluster_dgdp.py --config configs/milestone2c.cluster.toml sync
.venv/bin/python scripts/_milestone2d_richgrid_cluster.py train-bundle   # qsub; wait for it
```
- [ ] **Step 4: Fetch the bundle** — the `.npz` is written into the synced tree on the cluster; copy it back:
```bash
.venv/bin/python scripts/cluster_dgdp.py --config configs/milestone2c.cluster.toml run \
  bash -c 'base64 src/dgdp/models/dgdp_fixed_dict.npz' | tail -n +1 > /tmp/b64
# decode into src/dgdp/models/dgdp_fixed_dict.npz  (or rsync if a direct host is reachable)
```
(Use whatever fetch path the repo already uses; the file is ~1 MB.)
- [ ] **Step 5: Verify + run the parity/e2e tests** — `PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_export.py tests/test_deproject_e2e.py -q` (e2e now has the bundle → expect PASS; v_c peak in range, rms flaring).
- [ ] **Step 6: Commit** — `git add scripts/train_deprojection_model.py scripts/_milestone2d_richgrid_cluster.py src/dgdp/models/dgdp_fixed_dict.npz && git commit -m "feat(dgdp): R=64 trainer + bundled model"`

---

## Task 9: NGC 4321/4371 re-check with the R=64 bundle (validation gate)

Confirm the packaged R=64 model reproduces the validated physics (the earlier NGC checks used the R=32 model).

**Files:** Test: `tests/test_ngc_validation.py` (marked `slow`, skips if fits absent)

- [ ] **Step 1: Write assertions** vs the reports' numbers (tolerances generous — this is a regression gate, not a re-derivation):
```python
# tests/test_ngc_validation.py
import numpy as np, pytest
from pathlib import Path
from dgdp import deproject

@pytest.mark.slow
@pytest.mark.skipif(not Path("NGC4371/final/NGC4371_S4G_cut_EL_Fil.fits").exists(), reason="data")
def test_ngc4371_thick_and_conserving():
    r = deproject("NGC4371/final/NGC4371_S4G_cut_EL_Fil.fits", distance_mpc=16.194,
                  inclination_deg=58.0, pa_pix_deg=1.8, center=(254.6, 152.8),
                  pix_arcsec=0.75, mask="NGC4371/final/NGC4371_mask.fits",
                  ml=1.0, stellar_mass=3.53e10)
    radii = np.linspace(1, 15, 40)
    eff = float(np.average(r.rms_z(radii[radii < 12]),
                           weights=r.face_on.sum(axis=1)[:1]*0 + 1))  # inner-weighted mean
    assert 0.9 < r.rms_z(np.array([1.5]))[0] < 2.0                    # flaring, ~MGE regime
    assert 150.0 < r.v_circ(np.array([2.0]))[0] < 200.0              # v_c not depressed
```
- [ ] **Step 2: Run — expect PASS.** `PYTHONPATH=src .venv/bin/python -m pytest tests/test_ngc_validation.py -q -m slow`
- [ ] **Step 3: Commit** — `git add tests/test_ngc_validation.py && git commit -m "test(dgdp): NGC R=64 bundle validation gate"`

---

## Task 10: `dgdp.cli` + entry point

**Files:**
- Create: `src/dgdp/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/test_cli.py
import subprocess, sys, os, numpy as np
from pathlib import Path
import pytest
FITS = Path("NGC4321_m_c_r_f.fits")
pytestmark = pytest.mark.skipif(not FITS.exists(), reason="data")

def test_cli_writes_outputs(tmp_path):
    out = tmp_path / "o"
    env = {**os.environ, "PYTHONPATH": "src"}
    rc = subprocess.run([sys.executable, "-m", "dgdp.cli", str(FITS),
                         "--distance-mpc", "15.2", "--inclination-deg", "30",
                         "--pa-onsky-deg", "153", "--ml", "1.0", "--stellar-mass", "6e10",
                         "--no-figures", "-o", str(out)], env=env).returncode
    assert rc == 0
    assert (out / "density.npz").exists() and (out / "rotation_curve.csv").exists()
    d = np.load(out / "density.npz"); assert d["density_3d"].ndim == 3
```
- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement**
```python
# src/dgdp/cli.py
from __future__ import annotations
import argparse
from dgdp import deproject

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dgdp-deproject",
                                 description="Deproject a galaxy image -> 3D density + rotation curve.")
    ap.add_argument("image")
    ap.add_argument("--distance-mpc", type=float, required=True)
    ap.add_argument("--inclination-deg", type=float, required=True)
    ap.add_argument("--pa-pix-deg", type=float, default=None)
    ap.add_argument("--pa-onsky-deg", type=float, default=None)
    ap.add_argument("--center", type=float, nargs=2, default=None)
    ap.add_argument("--pix-arcsec", type=float, default=None)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--ml", type=float, default=1.0)
    ap.add_argument("--stellar-mass", type=float, default=None)
    ap.add_argument("--luminosity", type=float, default=None)
    ap.add_argument("--zeropoint", type=float, default=None)
    ap.add_argument("--band-solar-mag", type=float, default=None)
    ap.add_argument("--bar-angle-deg", type=float, default=0.0)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--potential", action="store_true")
    ap.add_argument("-o", "--output-dir", default="dgdp_out")
    a = ap.parse_args(argv)
    r = deproject(a.image, distance_mpc=a.distance_mpc, inclination_deg=a.inclination_deg,
                  pa_pix_deg=a.pa_pix_deg, pa_onsky_deg=a.pa_onsky_deg,
                  center=tuple(a.center) if a.center else None, pix_arcsec=a.pix_arcsec,
                  mask=a.mask, ml=a.ml, stellar_mass=a.stellar_mass, luminosity=a.luminosity,
                  zeropoint=a.zeropoint, band_solar_mag=a.band_solar_mag,
                  bar_angle_deg=a.bar_angle_deg)
    r.save(a.output_dir)
    if not a.no_figures:
        from dgdp.figures import save_summary_figure   # Task 10b (matplotlib)
        save_summary_figure(r, a.output_dir)
    if a.potential:
        import numpy as np
        np.save(f"{a.output_dir}/potential_R.npy",
                r.potential(np.linspace(1, 15, 40), np.zeros(40)))
    print(f"wrote outputs to {a.output_dir} (total mass {r.total_mass:.3e} Msun)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```
- [ ] **Step 3b: `src/dgdp/figures.py`** — a small `save_summary_figure(result, out_dir)` (matplotlib) drawing edge-on | face-on | v_c | RMS|z| (adapt the 4 relevant panels from `deproject_real_image_ngc4321_learned.py:373-421`, dropping the AGAMA panels). Keep it import-guarded so core install without matplotlib still runs `--no-figures`.
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** — `git add src/dgdp/cli.py src/dgdp/figures.py tests/test_cli.py && git commit -m "feat(dgdp): CLI + summary figure"`

---

## Task 11: Packaging + README

**Files:**
- Modify: `pyproject.toml`
- Create: `README.md` (or update)

- [ ] **Step 1: Update `pyproject.toml`** — slim core, add extras, entry point, package-data:
```toml
[project]
name = "disk-galaxy-deprojection"
version = "0.2.0"
description = "Learned mass-conserving deprojection of galaxy images -> 3D density, rotation curve, potential."
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "astropy>=6.0", "matplotlib>=3.8"]

[project.optional-dependencies]
train = ["torch>=2.1", "scipy>=1.11", "pandas>=2.1", "h5py>=3.10"]
dev = ["pytest>=7.4", "ruff>=0.1"]

[project.scripts]
dgdp-deproject = "dgdp.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
dgdp = ["models/*.npz"]
```
(Keep `[tool.pytest.ini_options]`, `[tool.ruff]` as they are.)
- [ ] **Step 2: Verify install into a scratch venv**
```bash
python -m venv /tmp/dgdp-test && /tmp/dgdp-test/bin/pip install -e . && \
  /tmp/dgdp-test/bin/python -c "from dgdp import deproject; import dgdp.cli; print('import ok')" && \
  /tmp/dgdp-test/bin/dgdp-deproject --help
```
Expected: imports with only core deps; `--help` prints. (torch NOT installed → confirms torch-free inference.)
- [ ] **Step 3: Write `README.md`** — sections: install (`pip install git+https://github.com/lucyundead/disk-galaxy-deprojection`), 8-line quickstart (API + CLI), the M/L two paths with a worked example, the AGAMA note (manual build for `--potential`), a one-paragraph method summary (image-anchored in-plane deprojection + learned fixed-dict sech² vertical q_m trained on TNG) linking `docs/reports/2026-06-30-mixture-qm-fixed-dictionary.md`, and outputs description.
- [ ] **Step 4: Full suite + ruff** — `PYTHONPATH=src .venv/bin/python -m pytest -q && .venv/bin/ruff check .` (expect all green, incl. the prior 111).
- [ ] **Step 5: Commit** — `git add pyproject.toml README.md && git commit -m "build(dgdp): installable package metadata + README"`

---

## Self-Review

**Spec coverage:** §1 API/CLI → Tasks 7,10; §3 modules → Tasks 1-7,10; §4 model/bundle/torch-free → Tasks 5,8; §5 API signature → Task 7; §6 CLI → Task 10; §7 M/L (both paths, precedence, relative fallback) → `_resolve_mass` Task 7; §8 deps/packaging → Task 11; §9 tests (parity/e2e/CLI) → Tasks 5,7,10; §10 risks (feature golden, agama-optional, R=64 NGC recheck) → Tasks 3,6,9. No gaps.

**Placeholder scan:** no TBD/TODO; the two "confirm during implementation" notes (rotation units in Task 6, absolute-feature indices in Task 7) name the exact check + the test that catches a mistake — not open-ended.

**Type consistency:** `DeprojectionModel` fields/`predict_weights` (Task 5) match usage in Task 7; `reconstruct_density` signature (Task 2) matches its Task 7 call; bundle keys in `save_bundle`/`load` (Task 5) match the trainer export (Task 8); `deproject(...)` kwargs (Task 7) match the CLI (Task 10) and e2e/NGC tests (Tasks 7,9).
