# Spec: `dgdp` end-to-end deprojection package

**Date:** 2026-06-30
**Goal:** turn the research pipeline into a pip-installable package that takes a galaxy
image and returns the deprojected 3D stellar density, rotation curve, renderings, and
(optionally) the gravitational potential — with a **bundled pretrained model** so no TNG
training data or GPU is needed at inference.

## 1. What it does

```python
from dgdp import deproject
r = deproject("galaxy.fits", distance_mpc=15.2, inclination_deg=30, pa_deg=153,
              pix_arcsec=0.75, center=(cx, cy),      # or read from the FITS WCS
              ml=1.0, stellar_mass=6e10)             # OR zeropoint=…, band_solar_mag=…
r.density_3d            # deprojected cylindrical stellar-mass cube (nR, nphi, nz) [Msun]
r.grid                  # r_kpc, phi_rad, z_kpc bin centres/edges
r.v_circ(radii_kpc)     # rotation curve [km/s]
r.rms_z(radii_kpc)      # vertical thickness RMS|z|(R) [kpc]
r.edge_on, r.face_on    # 2-D renderings (arrays)
r.potential(R, z)       # only if AGAMA importable; else raises with an install hint
r.save("out/")          # density.npz + rotation_curve.csv + figures.png
```
CLI: `dgdp-deproject galaxy.fits --distance-mpc 15.2 --inclination-deg 30 --pa-deg 153
--pix-arcsec 0.75 --center 512 480 --ml 1.0 --stellar-mass 6e10 -o out/`.

## 2. Non-goals (v1)

Single image, single **constant scalar** M/L, **point-estimate** outputs, one galaxy per
call. Not in v1: spatially-varying (multi-band/SED) M/L, uncertainty bands, multi-galaxy
batching, GUI.

## 3. Architecture

The inference path moves *into* the installed package (today it is scattered across
`scripts/`). Each module has one job:

| module | responsibility | key deps |
|---|---|---|
| `dgdp/image.py` | load FITS/array, background-subtract, resolve geometry (WCS or explicit PA/centre/pixel scale), build the geometric sech² baseline density + the TNG-format observed image | numpy, astropy |
| `dgdp/features.py` | image + metadata → the 586-D feature vector (pooled log image, central-flux features, 2 absolute mass features) | numpy |
| `dgdp/harmonics.py` | `harmonics` (even-m azimuthal FFT → a_m, Sigma_m), `r_resample` (R-only knot↔grid), `reconstruct_density` (predicted weights + image anchor → cell mass) | numpy |
| `dgdp/model.py` | load the bundled model; `predict_weights(features) → mixture weight vector` as a **pure-numpy MLP mean head + PCA inverse**; `save_bundle(...)` for the trainer | numpy |
| `dgdp/deproject.py` | orchestrate: image → baseline → features → weights → `vertical_mixture.reconstruct` + anchor → density cube, scaled to absolute mass; returns `DeprojectionResult` | numpy |
| `dgdp/rotation.py` | `v_circ` (direct softened summation, numpy); `potential` (AGAMA CylSpline, optional) | numpy; agama? |
| `dgdp/cli.py` | argparse → `deproject` → `result.save` | — |
| `dgdp/vertical_mixture.py`, `fourier_rz.py`, `density3d.py`, `agama_density.py` | **existing**, reused as-is | — |
| `dgdp/models/dgdp_fixed_dict.npz` | bundled pretrained weights (~1 MB) | — |
| `scripts/train_deprojection_model.py` | **maintainer-only**: train the fixed-dict mixture on the TNG table, export the numpy bundle | torch, scipy |

Research scripts (`deproject_real_image_ngc4321_learned.py`, etc.) stay as-is; the package
does not import from `scripts/`.

## 4. Pretrained model

**Why bundled:** the current script trains at runtime on the 808 MB TNG table. We train once
and ship the tiny result.

**Inference is torch-free.** `SummaryResidualMDN` (`src/dgdp/models/mdn.py`) is a 2-layer MLP
trunk + `logits`/`means`/`log_scales` heads; trained with `n_components=1`. The scripts use
`sample(x,128).mean(axis=1)`, which for one component converges to the deterministic mean head
`means(net(x))[:,0,:]`. We export **only** `net` (W₁b₁, W₂b₂) + `means` (W,b) and compute the
forward in numpy: `h = relu(W₂·relu(W₁x+b₁)+b₂); weights = W_means·h + b_means`. logits/
log_scales are unused at inference. (This is the *expected* prediction — cleaner than the
128-sample average the reports used; a parity test pins them together.)

**Bundle (`.npz`) contents:** `feat_mean, feat_scale`; `pca_vec, pca_mean, y_mean, y_std`
(target PCA + score standardization); `mlp_W1,b1,W2,b2,Wmeans,bmeans`; config = `heights`,
knot params (`r_min,r_max`, per-mode `n_r`, `K=len(heights)`, `EVEN_M`), grid spec
(`r_min,r_max,n_r,n_phi,z_max,n_z`), `image_feature_size`, `central_pixel_scale_kpc`, and the
two TNG-train-median absolute mass features (for the OOD in-distribution substitution).

**Provenance:** `scripts/train_deprojection_model.py` trains on the **R=64** TNG milestone-2d
table (`density_residual_table.npz`) and writes the bundle. Because that table is ~31 GB (cluster
only), the trainer runs as a cluster PBS job (like the retrain) and only the ~1 MB bundle is
fetched + committed. Retraining needs the table + `[train]` extra.

## 5. Public API

`deproject(image, *, distance_mpc, inclination_deg, pa_deg=None, pa_onsky_deg=None,
center=None, pix_arcsec=None, mask=None, ml=1.0, stellar_mass=None, luminosity=None,
zeropoint=None, band_solar_mag=None, bar_angle_deg=0.0, model=None, grid=None) ->
DeprojectionResult`

- `image`: FITS path or 2-D numpy array.
- geometry: WCS-derived if `pix_arcsec`/`center`/`pa_*` omitted and a FITS with WCS is given;
  else explicit (the existing WCS-bypass path).
- absolute scale: see §7.
- `model`: override the bundled model path.

`DeprojectionResult`: `.density_3d`, `.grid`, `.total_mass`, `.v_circ(radii)`,
`.rms_z(radii)`, `.edge_on`, `.face_on`, `.potential(R,z)`, `.save(dir)`. `.potential` imports
agama lazily and raises `RuntimeError` with a build hint if absent.

## 6. CLI

`dgdp-deproject IMAGE.fits [geometry flags] [--ml, mass/zeropoint flags] [-o OUTDIR]
[--no-figures] [--potential]`. Writes `density.npz`, `rotation_curve.csv`, and (default)
`deprojection.png` (edge-on | face-on | v_c | RMS|z|). `--potential` adds the AGAMA field
(errors clearly if agama missing). Exit non-zero on bad geometry/units.

## 7. M/L & absolute mass scaling

The deprojected **shape** is calibration-free; M/L sets only the v_c amplitude. Two entry paths
(both accepted):
1. **Photometric:** `zeropoint` + `band_solar_mag` + `distance_mpc` → luminosity from the
   background-subtracted image counts → `M* = ml × L`.
2. **Direct:** `stellar_mass` (total M*, M/L already folded in) → used as the absolute scale
   directly; `ml` is then **ignored for scaling** (no double-count). `luminosity` (total L) →
   `M* = ml × luminosity`.
Precedence: `stellar_mass` > `ml × luminosity` > photometric (`ml × L_image`). If none resolve
to a mass, the density shape and a **relative** v_c(R) are still returned (with a warning);
absolute km/s requires one of the three.

## 8. Dependencies & packaging

- setuptools + src-layout (unchanged). `name = disk-galaxy-deprojection`, import `dgdp`.
- **core**: `numpy>=1.26`, `astropy>=6.0`, `matplotlib>=3.8`.
- **extras**: `[train] = torch, scipy, pandas, h5py`; `[dev] = pytest, ruff`.
- `[project.scripts] dgdp-deproject = "dgdp.cli:main"`.
- `[tool.setuptools.package-data] dgdp = ["models/*.npz"]`.
- AGAMA: **not** a pip extra (not on PyPI); documented manual build, detected at runtime.
- Install: `pip install git+https://github.com/lucyundead/disk-galaxy-deprojection`.

## 9. Testing

- **parity** (`tests/test_model_export.py`, needs `[train]`): numpy mean head vs torch
  `means(net(x))` on random inputs, < 1e-5.
- **end-to-end** (`tests/test_deproject_e2e.py`): `deproject()` on the repo NGC 4321 FITS with
  its known geometry → assert density finite & ≥0, correct shape; `v_circ` peak ∈ ~[150,190]
  km/s; `rms_z` flares above the thin baseline; `∫ρ dV ≈ stellar_mass` (1%).
- **CLI** (`tests/test_cli.py`): run on the test FITS → assert `density.npz`,
  `rotation_curve.csv`, `deprojection.png` exist and load.
- **units** where cheap: image geometry (WCS vs explicit) centre/PA resolution; feature-vector
  length; `reconstruct` conservation (already covered by `vertical_mixture` self-check).
- The existing 111 tests stay green; `ruff` clean.

## 10. Risks / open questions

- **Feature reproduction OOD:** the packaged features must exactly match training (pooling,
  central-flux, the 2 absolute-mass substitutions) or predictions drift — covered by porting
  `make_density_residual_features` verbatim + a golden-value test against the current script.
- **Photometric path correctness:** counts→luminosity needs the right zero-point convention;
  documented + a numeric example; direct-mass path is the safe default.
- **AGAMA optional import** must never break core import — lazy import inside `potential` only.
- **Model = R=64 fixed-dict.** The R=64 conserving retrain is validated (exact conservation,
  relL2 0.455); the R=64-trained bundle must be re-checked end-to-end on NGC 4321/4371 (the
  earlier thickness validation used the R=32 model) — Phase 3.

## 11. Implementation phases (the plan will expand these)

1. Port inference helpers into `dgdp/` (`image`, `features`, `harmonics`) with golden tests vs
   the current script's intermediate arrays.
2. `model.py` + `scripts/train_deprojection_model.py`; run the trainer on the cluster (R=64),
   fetch + commit the ~1 MB bundle; numpy↔torch parity test.
3. `deproject.py` + `DeprojectionResult` + `rotation.py`; end-to-end test + NGC 4321/4371
   re-check with the R=64 bundle (thickness + conservation hold).
4. `cli.py` + entry point; CLI test.
5. `pyproject.toml` deps/extras/package-data; README (install, quickstart, M/L, AGAMA, method).
