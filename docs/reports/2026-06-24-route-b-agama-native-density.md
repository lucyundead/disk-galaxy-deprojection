# Route B: AGAMA's native density expansion as the deprojection representation

Date: 2026-06-24

Two questions about using AGAMA's own density expansion in the deprojection backbone,
following `2026-06-23-fourier-rz-target-deprojection.md` and
`2026-06-23-potential-force-validation.md` (and its 2026-06-24 "Finer (R,z) grid"
section, which bumped `fit_fourier_rz` to a 25x25 default so the Fourier x (R,z) rep
matches the AGAMA CylSpline gold on forces):

1. **Does predicting AGAMA's native `DensityAzimuthalHarmonic` / `CylSpline`
   coefficients (finer quintic spline) raise the deprojection oracle ceiling** over
   our even-m Fourier x (R,z) target? Prototyped cheaply on the three reference
   galaxies first; the full milestone-2d MDN/flow retrain was gated on a clear win.
2. **A `to_agama_density()` helper** that wraps a predicted Fourier x (R,z) /
   conserving density as a native `agama.Density` (and `agama.Potential`) with no
   retraining, demonstrated end to end on one held-out galaxy.

> **Headline.** (1) The AGAMA-native expansion does **not** raise the representation
> ceiling: at matched resolution `DensityAzimuthalHarmonic` **ties** the Fourier x
> (R,z) target on the smooth SPH-KDE truth and is **worse** than the 25x25 Fourier
> full; AGAMA and our Fourier rep are the *same representation class* (even-m
> azimuthal harmonics x smooth (R,z) maps), so the finer spline cannot recover the
> odd-m / fine-cell structure that caps the deprojection oracle. **Gate not passed ->
> no full retrain.** This confirms and sharpens the standing expectation (prediction
> is PCA-32- and image->coefficient-limited, not representation-limited). (2) The
> `to_agama_density()` helper preserves the predicted total mass **exactly** through
> the AGAMA wrap (residual ~1e-15) and the predicted potential reproduces the
> truth-density rotation curve to **~3 km/s** on a held-out galaxy.

## 1. AGAMA-native coefficient oracle prototype

`scripts/agama_density_oracle_prototype.py`. For each reference galaxy (Shen2010 +
TNG 554189 / 392276, full particles, smooth SPH-KDE truth, k=32) we build an AGAMA
`DensityAzimuthalHarmonic` on a *fixed* grid + mmax + symmetry (`gridsizeR=gridsizez=25`,
`symmetry='triaxial'` = even m only, bar on x), verify its coefficients round-trip
through `export()` / re-import, and score the *oracle* (true-coefficient)
reconstruction's 3D rel-L2 against the same SPH-KDE truth on the standard 0.2 kpc eval
grid (`truth > 1e-3 max` mask) - head to head with the Fourier x (R,z) target and the
cylindrical grid.

**Coefficient round-trip is exact.** `export()` writes the per-m (R-row x z-col)
coefficient tables; `agama.Density(file=...)` reads them back and the density agrees
to **3-5e-14** (median rel diff) on all three galaxies. So AGAMA coefficients are a
usable fixed-layout target. (AGAMA auto-drops harmonics with no power; on the real
galaxies all requested even m are retained.)

### Representation ceiling: 3D rel-L2 vs the smooth SPH-KDE truth (lower better)

| representation | n_coeff | Shen2010 | TNG 554189 | TNG 392276 |
| - | -: | -: | -: | -: |
| Fourier x (R,z) full 25x25 | 6875 | **0.351** | **0.293** | **0.345** |
| Fourier x (R,z) full 14x13 | 2002 | 0.385 | 0.307 | 0.370 |
| Fourier x (R,z) compressed | 496 | 0.377 | 0.331 | 0.380 |
| AGAMA AZH (mmax=6) | 2500 | 0.387 | 0.337 | 0.372 |
| AGAMA AZH (mmax=10) | 3750 | 0.410 | 0.321 | 0.416 |
| cylindrical grid 0.3125 | 49152 | 0.465 | 0.600 | 0.444 |

- **AGAMA `DensityAzimuthalHarmonic` (mmax=6, 2500 coeff) ties the Fourier x (R,z)
  full 14x13 (2002 coeff)** to within ~0.01-0.03 rel-L2 and is **worse than the
  Fourier full 25x25** on every galaxy. It does **not** raise the smooth-truth
  ceiling. Our 496-coeff compressed target matches AGAMA's 2500-coeff expansion - ~5x
  more compact for the same fidelity.
- **More harmonics hurt**: `mmax=10` is worse than `mmax=6` on Shen2010 and 392276,
  because m=6,8,10 of an equal-mass N-body / SPH-KDE density are mostly shot noise
  (the Task-1 compression finding) - the quintic spline faithfully fits that noise.
- Both smooth reps beat the histogram grid on a smooth truth, as established.

### Forces: AGAMA CylSpline median fractional force error vs the particle gold

| representation | Shen2010 | TNG 554189 | TNG 392276 |
| - | -: | -: | -: |
| AGAMA AZH (mmax=6) -> CylSpline | 0.011 | 0.078 | 0.016 |
| Fourier x (R,z) full 25x25 | 0.019 | 0.028 | 0.029 |
| cylindrical grid 0.3125 | 0.014 | 0.025 | 0.037 |

AGAMA AZH is **erratic on forces** - best on the two concentrated high-N galaxies
(Shen2010, 392276) but markedly worst on the extended low-N disk 554189 (0.078 vs
~0.03), the same N-dependent bias-variance behaviour seen throughout. No systematic
advantage over the Fourier rep or the grid.

### Why a finer spline does not help, and the gate decision

The deprojection oracle (`deproject_fourier_rz_compare.py`) is the PCA-32
reconstruction of the *true* target, scored on the noisy TNG grid: grid oracle resid
rel-L2 **0.281** (the grid target is the exact residual, so its only loss is PCA-32),
Fourier oracle **0.379** (the even-m + (R,z) projection additionally discards ~25% of
the grid residual - odd-m lopsidedness + fine-cell structure - and that prior study
found the Fourier oracle *flat above ~600 coefficients*, i.e. capped by the
**projection**, not the (R,z) resolution).

AGAMA `DensityAzimuthalHarmonic` is the **same representation class** as our Fourier x
(R,z): even-m azimuthal harmonics times smooth (R,z) maps. It differs only in
interpolation order (quintic vs linear) and knot placement - second-order effects that
the smooth-truth tie above confirms are negligible. A quintic (R,z) spline **cannot**
recover the odd-m asymmetry or the sub-knot cell structure that the even-m+(R,z)
projection throws away, so the AGAMA PCA-32 oracle would land at ~the Fourier oracle
(~0.379), still well above the grid oracle (0.281) and far above where the
predictability-limited MDN sits (the MDN/oracle cell-mass gap is ~1.4x). Since the raw
(no-PCA) representation oracle is an *upper bound* on any PCA-32 oracle, and that upper
bound already does not beat the Fourier target, no held-out recovery gain is possible.

**Decision: the AGAMA-coeff oracle is not clearly better -> do NOT proceed to the full
MDN/flow retrain on the milestone-2d table.** This was the explicit gate. The cheap
3-galaxy prototype settles it without the expensive 1665-row per-row AGAMA fit (which
would also require fitting `DensityAzimuthalHarmonic` to *signed* residual grids); the
representation-class equivalence plus the "oracle flat above ~600 coeff" finding make
that run redundant. Net: route B (1) **refutes** a representation gain; the bottleneck
remains image->coefficient predictability, exactly as the expectation stated.

## 2. `to_agama_density()` helper + end-to-end demo

`src/dgdp/agama_density.py` (unit-tested in `tests/test_agama_density.py`, skipped
unless the prebuilt AGAMA is importable). The even-m Fourier x (R,z) / conserving
density is already in the AGAMA CylSpline form, so wrapping it needs **no retraining** -
purely a post-hoc representation wrap:

- `to_agama_density(density_callable, total_mass=..., mmax=6, symmetry='triaxial', ...)`
  -> `agama.DensityAzimuthalHarmonic`. The azimuthal-harmonic fit is *linear* in the
  input density, so a single global rescale makes `totalMass()` equal the requested
  `total_mass` **exactly** - a mass-conserving deprojection stays mass-conserving
  through the wrap.
- `fourier_rz_to_agama_density(model, total_mass=..., ...)` - convenience over a
  `dgdp.fourier_rz` model via `reconstruct_fourier_rz`.
- `to_agama_potential(density, ...)` -> `agama.Potential(type='CylSpline', ...)` from an
  `agama.Density` or a callable.
- `circular_velocity(potential, radii)` - azimuthally-averaged v_c(R) for a barred
  (non-axisymmetric) potential.

### Demo (`scripts/agama_predicted_potential_demo.py`)

End to end on **TNG 554189** (held out, test split; full particles available), with no
retraining: the *predicted* 3D density is the adopted grid PCA-MDN posterior mean from
the saved milestone-2d predictions (`mdn_z5/...predictions.npz` - a genuine
image->density prediction, loaded not recomputed). Row: inclination 40 deg, bar angle
0.

`observed image -> predicted 3D density -> agama.Potential -> v_c(R) + orbits`
(`figures/agama_predicted_potential_demo.png`).

- **Mass conservation is exact through the wrap.** Predicted total 3.19e10 Msun
  (within the R<15 / |z|<4 kpc representation domain: 2.63e10, 82% - 554189 is an
  extended disk); the AGAMA densities reproduce these to **4.8e-16 / 3.1e-15** relative
  (machine precision). Without the helper's rescale the harmonic fit would drift; with
  it, a mass-conserving deprojection is preserved.
- **The predicted potential reproduces the dynamics.** v_c(R) from the predicted
  CylSpline matches the truth-density potential's v_c to **rms 3.4 km/s** (Fourier-x
  (R,z) wrap) / 3.3 km/s (raw-grid wrap) over R<15 kpc.
- **Orbits** integrated in the predicted potential (~5.9 Gyr) at R0 = 1.5 / 3 / 6 kpc
  (v_c 180 / 142 / 108 km/s) are well behaved face-on and edge-on.

So a predicted, mass-conserving deprojection drops straight into AGAMA for orbits /
actions / rotation curves with no extra fitting and no mass leakage - the practical
payoff of the Fourier x (R,z) rep being CylSpline-form.

## Verification

`.venv/bin/python -m ruff check .` clean. `pytest -q` **111 passed, 1 skipped** (the
AGAMA test module skips without AGAMA); with
`PYTHONPATH=/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313`, **115 passed**
(the 4 AGAMA-density tests run). The frozen deprojection target
`outputs/nbody_shen2010/fourier_rz_allocation.json` was **not** regenerated.

## Code and artifacts

- `src/dgdp/agama_density.py` - `to_agama_density`, `fourier_rz_to_agama_density`,
  `to_agama_potential`, `circular_velocity` (lazy AGAMA import; exact mass preservation).
- `tests/test_agama_density.py` - mass-preservation, density-match, potential/v_c,
  Fourier-model wrap (`pytest.importorskip('agama')`).
- `scripts/agama_density_oracle_prototype.py` - route-B(1) oracle prototype; writes
  `outputs/nbody_shen2010/agama_density_oracle_metrics.json` and
  `figures/agama_density_oracle_prototype.png`.
- `scripts/agama_predicted_potential_demo.py` - route-B(2) end-to-end demo; writes
  `outputs/nbody_shen2010/agama_predicted_potential_demo_metrics.json` and
  `figures/agama_predicted_potential_demo.png`.
- Run both with `PYTHONPATH=/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313`.
