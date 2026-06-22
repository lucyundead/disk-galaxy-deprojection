# N-body Cross-Check: Deprojecting the Shen2010 Milky Way Bar

Date: 2026-06-13

This is the first **out-of-distribution** test of the adopted Milestone 2c MDN
deprojection model on a real N-body barred galaxy that was never in training:
the Shen et al. (2010) pure-disk Milky Way model (snapshot `t800`). It executes
the N-body validation step flagged on the presentation roadmap (slide 16), with
one important twist found here: *this snapshot is boxy/thick, not X-shaped*.

## Data and setup

- Source: `gravity-login01:/home/zli/Shen2010MW/t800info.dat`, 982,889 particles,
  ASCII columns `x y z vx vy vz`. Pure-disk, equal-mass (no mass column).
- **Units are already physical** (kpc, km/s): R50 = 2.1 kpc, R90 = 10.3 kpc,
  |z|90 = 1.2 kpc, |v| ~ 175 km/s. No rescaling. The model drops straight into
  the production grid (R 0.05-30 kpc, |z| < 10 kpc).
- Total mass normalized to a MW-like 4.5e10 Msun (equal-mass particles). This
  only sets the two absolute log-mass features; it keeps them inside the TNG
  training range. Everything else (image, central features) is scale-free.
- Aligned to the disk/bar frame with the standard pipeline helper
  (`align_particles_to_disk_bar_frame`): disk normal from angular momentum
  (came out along z, as expected for a disk), bar major axis to intrinsic x via
  the m=2 phase (residual bar angle 7e-16 deg). Bar half-length (m=2 half-max)
  = 4.4 kpc; peak m=2 amplitude 0.20-0.27 (a strong bar). 100% of the mass
  lands inside the grid.
- Model: adopted 2c run
  `density_residual_pca_mdn_sweep_central/components_1_seed_20260608`,
  **no retraining, core MDN only (no correction heads)**, 128 posterior samples.
- Geometries: inclination {20, 40, 60} deg x bar viewing angle {20, 40, 60} deg,
  disk PA = 0 (9 projections). Clean 192x192 images at 0.35 kpc/pixel, no
  PSF/noise (the Milestone 2b/2c image convention).

Reproduce:

```bash
.venv/bin/python scripts/eval_nbody_shen2010_deprojection.py
.venv/bin/python scripts/_nbody_shen2010_check_xshape.py
```

Artifacts: `outputs/nbody_shen2010/` (metrics JSON, predicted grids, the
auto-generated table `nbody_shen2010_deprojection.md`, and `figures/`).

## Headline

Applied with **zero retraining** to a genuinely out-of-distribution N-body bar,
the MDN **recovers the in-plane 3D structure well and roughly halves the
geometric baseline's 3D cell-mass error at every one of the 9 geometries**, but
it **over-thickens the disk vertically** because it imposes the (thicker) TNG
vertical-structure prior on an intrinsically thin disk.

## Quantitative results (median over 9 geometries unless noted)

| Quantity | Truth | Geometric baseline | MDN |
| - | -: | -: | -: |
| 3D cell-mass MAE [Msun] | - | 5.09e5 | **2.75e5** (47% better) |
| total mass fractional error | - | ~0.004 | 0.004 |
| central mass fraction (R<2 kpc) | 0.490 | 0.444-0.489 | 0.456-0.492 |
| bar m=2 amplitude (R<=L_bar) | 0.199 | 0.10-0.23 | 0.13-0.28 |
| global vertical RMS height [kpc] | 0.77 | 0.41 | 1.25 |

- **Cell-mass MAE**: MDN 2.6e5-3.1e5 Msun vs baseline 5.0e5-6.0e5; the 47%
  improvement is remarkably flat across inclination and bar angle (range
  43-50%). The baseline degrades with inclination (worst at i=60); the MDN does
  not. (Note: this is below the 66% improvement on in-distribution TNG test
  galaxies - the out-of-distribution penalty.)
- **Central concentration**: the thin-disk baseline loses central mass as
  inclination rises (0.489 -> 0.444 at i=60 vs truth 0.490); the MDN corrects
  most of that deficit (up to 0.46-0.49), the same inclination-dependent fix it
  learned on TNG.
- **Bar m=2**: at unfavorable geometry the baseline bar nearly vanishes
  (0.097 at i=60, bar=20 vs truth 0.199); the MDN restores it toward truth,
  occasionally over-predicting (e.g. 0.28 at i=60, bar=60).
- **Vertical structure (the failure mode)**: the thin-disk baseline is far too
  thin (RMS z 0.41 vs truth 0.77 kpc). The MDN correctly recognizes the bar
  must be vertically extended and thickens it, but **overshoots to 1.25 kpc
  (~1.6x truth)**. Its vertical error is comparable in magnitude to the
  baseline's but opposite in sign. This is a clean out-of-distribution
  signature: the Shen2010 disk is thinner than typical mass-selected TNG50
  barred galaxies, so the TNG-learned thickening is too large. It is consistent
  with the Milestone 2c finding that the thin-disk baseline overstretches thick
  disks, just running the other way here.

Figures: `figures/nbody_recovery_i40_bar40.png` (mock image + face-on and
edge-on truth/baseline/MDN + bar-end vertical profile) and
`figures/nbody_recovery_vs_geometry.png` (cell-mass MAE, vertical thickness,
central concentration across all 9 geometries).

## Boxy/peanut/X: confirmed, and the MDN reproduces the morphology

CORRECTION (a first pass wrongly called this snapshot "boxy, not X"). The model
**does have a real boxy/peanut/X bulge**. The earlier mistake was the criterion:
the `dip` / `peak |z|` tests looked for an off-plane density *maximum* (a literal
local minimum in rho(z) at the midplane), which is only the strongest
"X-with-off-plane-peaks" sub-type. A boxy/peanut/X is fundamentally a
**contour-shape** feature - pinched iso-density contours in a side-on log edge-on
map - and in a slab the thin disk fills the midplane, so rho(z) stays
single-peaked even when the contours are unmistakably peanut-shaped. "dip=1.000"
was a true number, but it does not mean "no X".

Measured the right way (iso-density contour height z(x), bar along x;
`_nbody_shen2010_check_xshape.py`):

- viewed at fine resolution in a thin slab (|y|<0.5 kpc, 0.05-0.1 kpc pixels;
  `_nbody_shen2010_fourier_basis_demo.py`) the truth is a **strong X** -
  pinched waist, off-plane arms reaching |z| ~ 1.5 kpc - comparable to strong
  observed peanuts (confirmed by the model author). The scalar pinch metric is
  noise- and slab-sensitive and UNDER-reports strength (it read ~1.0 on the
  noisy full fine grid and ~0.89 on the thick-slab production render); the
  visual / contour-height z(x) is the reliable readout;
- the **production 0.625 kpc grid smears the X away** (flat-topped contour
  height z(x)): resolving the off-plane arms needs <= 0.3 kpc vertical pixels.
  This is the milestone-2d resolution point, now sharper - the grid the adopted
  MDN operates on cannot represent this X at all;
- the geometric thin-disk baseline is flat (no pinch): it cannot represent the
  peanut at all;
- the **MDN reproduces the boxy/peanut morphology** at the production
  resolution - a vertically-extended, pinched central structure, dramatically
  unlike the flat baseline - but at that resolution the peanut is essentially
  central vertical thickening, so this is coarse boxy/peanut reproduction, not
  recovery of the resolved off-plane X arms (which the grid cannot hold).

Two caveats keep this honest:

- the MDN over-thickens the vertical scale (see above: RMS z 1.25 vs 0.77 kpc),
  so it gets the *shape* but not the *size*;
- at the production grid's 0.625 kpc vertical resolution the peanut is mostly
  expressed as central vertical thickening, which the MDN already learned to do
  on TNG. So this demonstrates coarse boxy/peanut reproduction, not necessarily
  recovery of the fine off-plane X arms. The latter needs the finer-z grid
  (0.3125 kpc) and is where the Milestone 2d concern (peanut-poor training) still
  applies. Figure: `figures/nbody_xshape_check.png`.

## Implications

1. **Positive**: the physics-first core model (geometric baseline + PCA-residual
   MDN) generalizes to a real N-body bar with no retraining for all in-plane
   structure - central concentration, bar m=2, radial profile, total mass - and
   halves the classical-deprojection 3D error. This is strong external
   validation of the method's in-plane behavior.
2. **Actionable failure**: vertical over-thickening from the TNG thickness prior.
   This reinforces the handoff's current top priority (vertical spread
   mis-calibration). Candidate fixes: a thickness-aware / heteroscedastic
   vertical head conditioned on an image thickness proxy and inclination, or
   adding N-body disks spanning a range of thicknesses to training.
3. **Peanut/X**: the MDN reproduces the boxy/peanut morphology (shape, not
   scale) on a real out-of-distribution bar - a genuinely positive result for
   slide 14/16. The remaining open question is fine off-plane X-arm recovery at
   0.3125 kpc resolution, which this snapshot *can* now be used to test (it has
   the structure); that is where the peanut-poor-training concern still bites and
   N-body training data would help.

## Representation idea: structured azimuthal-Fourier basis (enables flow matching)

Motivated by the existing face-on image decomposition (keep azimuthal harmonics
m=0,2,4,6,8,10), extend it to the 3D field: at each (R,z) keep only the even,
low-order azimuthal harmonics of the cylindrical density,
rho(R,phi,z) ~ sum_{m in 0,2,4,6,8,10} a_m(R,z) cos(m phi) (bar frame). Measured
on t800 (`_nbody_shen2010_fourier_basis_demo.py`):

- odd-m power is 0.65% of the total -> dropping odd m enforces bar symmetry for
  free (the current PCA must learn that symmetry from limited data);
- even m<=10 preserves the X while removing ~22% high-m *local* noise (the
  smoothing wanted for a clean potential in downstream dynamics);
- 10.7x dimension reduction from the azimuthal truncation alone (before any
  (R,z) compression).

Proposed target representation (replaces/augments the raw-cell PCA):
1. azimuthal Fourier, even m<=10, in the bar-aligned frame -> 6 real amplitude
   maps a_m(R,z);
2. compress each a_m(R,z) with data-driven (PCA-per-m) or smooth analytic
   (radial spline x vertical) modes - the vertical basis must be X-capable
   (Gauss-Hermite even orders or empirical vertical modes; a single sech^2
   cannot represent the peanut), on a <= 0.3 kpc vertical grid;
3. predict the resulting O(50-150) smooth, symmetry-respecting coefficients.

Why this helps a flow-matching posterior: the current 32-PCA target is already
low-dim but is built on raw cells (no enforced symmetry, mixes in local/odd
noise) and a 1-Gaussian MDN already fits it (slide 15), so a flow had no clear
opening. A structured, smoother target that resolves the X is where genuine
posterior non-Gaussianity (off-plane-arm degeneracies) could appear and where
flow matching would earn its keep; and the smoothing + symmetry are valuable for
the dynamical modeling regardless of the generator.

### Foundation built (2026-06-13): `build_nbody_structured_basis.py`

Per the chosen scope (residual target, hybrid PCA-per-m, foundation only). A fine
0.125 kpc truth grid (32 x 48 x 64) was built for t800 - fine enough to resolve
the X - with the 9 per-geometry baselines and residuals
delta_rho = rho_true - rho_baseline. The residual was expressed in the structured
basis (azimuthal even m<=10, then per-m low-rank (R,z) compression). Results
(`outputs/nbody_shen2010/nbody_structured_basis_metrics.json`,
`figures/nbody_structured_basis.png`):

- odd-m power is 3.8% of the residual -> dropping it enforces bar symmetry; the
  even-m<=10 truncation captures 82% of the residual variance (the discarded 18%
  is odd-m lopsidedness + high-m local noise = the smoothing wanted for dynamics);
- per even harmonic the (R,z) map needs ~8 modes for 99% energy (m=0 needs 3);
- in the bar/peanut region (R<6 kpc, |z|<2.5) the structured reconstruction cuts
  the truth density error from 0.365 (geometric baseline, no residual) to 0.155
  (full even-m<=10), already ~0.18 at 48 coefficients (rank k=4);
- vertical thickness RMS_z(x): the full even-m<=10 reconstruction recovers it
  close to truth; a low rank (k=4) UNDER-restores the high-|z| tails (vertically
  thin) even though its in-plane density error is small - the (R,z) low-rank
  compression trades vertical-tail fidelity, so vertical recovery needs higher
  rank than density recovery. (Reconstructions are positivity-clipped, as the
  pipeline does.)

CAVEAT on reading the figure: RMS_z(x) is overall thickness (a z^2 moment), NOT
the X shape. The coarse 0.625 kpc production grid PRESERVES RMS_z (it tracks
truth) yet cannot resolve the X CONTOUR (~3 z-bins inside |z|<1 kpc) - so
"the production grid smears the X" refers to the contour/waist shape, which needs
<=0.3 kpc z and is seen in the contour diagnostic, not in RMS_z(x).

Honest caveat: ~50-100 single-galaxy capacity coefficients is NOT fewer than the
current 32 cross-galaxy PCA modes - the win is the enforced symmetry + smoothing
+ fine-z X-capability, not raw compactness. The cross-galaxy PCA-per-m dimension
(the real flow-matching target size, fit across many galaxies) is still TBD and is
the next step.

### Cross-galaxy basis, peanut census, and X span-check (2026-06-14)

Done on the existing fine-z (0.3125 kpc) milestone-2d table
(`/mnt/e/dgdp-milestone2d/density_residual_table.npz`, 185 galaxies, 999 train).

**Peanut census (provisional / unreliable).** Built and validated a
tilt-invariant boxy/peanut metric (`scripts/peanut_strength.py`, m=4 isophote b4
in the inertia principal frame): it passes synthetic ground truth (thin disk
-0.069 disky, ellipsoid 0.000 neutral, Shen2010 +0.042 boxy; tilted versions
unchanged). Census of the 185 (disk frame, no de-tilt, bar-scaled bulge region):
median ~0, ~9-11% clearly boxy/peanut, ~1% as strong as Shen2010. BUT an
independent cross-check (`scripts/_tng_peanut_crosscheck.py`) shows the metric
does NOT correlate with bar vertical thickening (Pearson r=-0.07) and only weakly
with catalog A2 (r=+0.15) - so on real multi-component galaxies it conflates
disk/classical-bulge/noise and the ~10% is NOT trustworthy. Conclusion: "is TNG
peanut-poor" stays open; a reliable answer needs a dedicated B/P pipeline
(disk+bulge decomposition or unsharp-mask + visual labels). Tilt aside: 392276
(milestone-2d's "X") is the single most-tilted galaxy in the sample (12.3 deg vs
median 1.0; `scripts/_tng392276_xz_check.py`, `_tng_tilt_scan.py`), and a tilt
mimics off-plane peanut mass - so the old dip-based "1-in-185" was confounded.

**Structured basis dimension (`scripts/build_structured_basis_tng.py`).**
Literal PCA-per-m is REJECTED: it needs 294 dims for 95% var (m=0:7 but m=4-10:
46-78 each, carrying ~0.1% the power) and at dim 294 reconstructs (test rel-L2
0.279) no better than the existing global flattened PCA at dim 32 (0.275); the
global PCA captures cross-harmonic correlations the per-m split discards.
CONSTRUCTIVE alternative: Fourier-FILTER (even m<=10) THEN global PCA. The filter
discards 24.9% of the raw residual (odd/high-m local noise = the wanted
smoothing), the smooth target needs only 43 comps for 95% var (vs 125 raw), and a
32-dim basis represents it at rel-L2 0.155. So symmetry + smoothing + low dim are
achievable as a one-line filter before the existing PCA - the right flow-matching
target - NOT a per-m rebuild.

**X span-check (`scripts/_nbody_shen2010_span_check.py`).** Projecting the
Shen2010 residual onto a TNG-fit basis only PARTIALLY represents the strong X:
b4 truth +0.034, thin-disk baseline -0.037, raw global PCA-32 recon +0.010,
filtered+global PCA-32 +0.004, filtered+global PCA-64 +0.013. Recon closes
~60-70% of the baseline->truth boxiness gap and recovers the vertical thickening
(edge-on far broader than baseline) but not the full strong X; more components
help. So for a clean strongly-buckled OOD X, REPRESENTATION is a partial
bottleneck at 32-64 dims, not only prediction - N-body X examples in the basis fit
(or an X-capable vertical basis / more components) would improve it; the common
weaker boxy thickening is already well represented by TNG.

Next: a flow-matching prototype on the ~32-43 filter+global-PCA coefficients
(vs the 1-Gaussian MDN), and/or enrich the basis with N-body X examples to lift
the strong-X representation ceiling.

### Flow-matching prototype (2026-06-14)

`scripts/flow_matching_prototype.py`: conditional flow matching vs the 1-Gaussian
MDN, identical 586 features and identical filter+global-PCA-32 target, TNG fine-z.
Accuracy is IDENTICAL (coeff posterior-mean RMSE ~0.0006 both; residual recon
rel-L2 MDN 0.331 vs flow 0.313). The flow is better calibrated raw (68% coverage
flow 0.693 on-target vs MDN 0.774 over-conservative) and learns a tighter, mildly
non-Gaussian posterior. But both sit far from the oracle/basis ceiling
(recon 0.31-0.33 vs 0.155), so the bottleneck is image->coefficient predictability,
NOT the generator. On TNG, flow matching is a marginal (calibration-only) win,
consistent with the slide-15 deferral; its real potential is genuinely
non-Gaussian regimes (strong-X off-plane-arm degeneracies), to be tested on the
N-body models. Figure: `figures/flow_vs_mdn.png`.

## Representation backbone: grid-free SPH-KDE + superellipsoid (2026-06-14)

Motivated by gal3d (superellipsoid iso-density shape modeling from particles) and
Tahmasebzadeh, Zhu, Shen, Gerhard & Qin 2021 (MGE deprojection of barred galaxies,
validated on a Shen-type peanut). Key reframes from those: (a) **MGE cannot
represent the B/P; the superellipsoid generalizes it** (squareness exponent
`S<1` = pinched peanut, `S=1` = ellipsoidal/MGE-like shell, `S>1` = boxy), so the
superellipsoid is the better core representation while MGE stays a strong bulk
baseline; (b) for the downstream dynamics the right metric is the
**potential/orbits**, not cell-mass MAE - and the paper shows MGE recovers the
potential to <10% and 85% of orbit morphologies *despite missing the peanut
density*, so the X is dynamically second-order (the bulk dominates the potential);
(c) the deprojection degeneracy and the image->coefficient bottleneck are
basis-independent and worst at low inclination - a superellipsoid fixes
*representation*, not *constraint*.

These point to dropping the coarse cylindrical grid entirely. Proof of concept
(`scripts/_nbody_shen2010_gridfree.py`, `scripts/reconstruct_superellipsoid_3d.py`):

- **mesh-free SPH-KDE** (k=32, cubic spline) resolves the boxy/peanut X at any
  resolution, where the 0.625 kpc grid smears it;
- a nested **superellipsoid** fit (semi-axes pinned to iso-surface extents,
  squareness fitted; degenerate free-`a` fit avoided; global bulge-principal-axis
  rotation absorbs the 392276 tilt) reconstructs `rho(x)` by interpolating shell
  levels, and **beats the grid in 3D** against the SPH-KDE truth (rel-L2,
  truth>1e-3 max, 0.2 kpc eval):

| galaxy (full particles) | N stars | superellipsoid (144 params) | grid 0.3125 kpc (49,152 cells) | grid 0.625 kpc |
| - | -: | -: | -: | -: |
| Shen2010 (N-body bar+disk) | 0.98M | **0.381** (win) | 0.465 | 0.569 |
| TNG 392276 (massive bar/peanut) | 1.50M | 0.437 (~tie) | 0.444 | 0.507 |
| TNG 554189 (strong bar, extended disk) | 0.56M | 0.619 (slight loss) | 0.600 | 0.712 |

FULL-PARTICLE CORRECTION (2026-06-22): the first TNG numbers here used 80k-capped
subsamples (the milestone-2c `max_particles_per_galaxy=80000` extraction cap),
which starved the grid (~1.6 particles/cell) and inflated the superellipsoid's
apparent margin. Re-extracted UNCAPPED from the cluster (392276 1.50M, 554189
0.56M stars; true counts 1.56M / 0.56M) and re-ran: the grid's rel-L2 drops sharply
(392276 0.579 -> 0.444, 554189 0.674 -> 0.600) and the superellipsoid's "win"
largely vanishes - it now **ties** the fine grid on 392276 (0.437 vs 0.444) and
**slightly loses** on 554189 (0.619 vs 0.600). It still wins on the clean Shen2010
N-body bar+disk (0.381 vs 0.465), where the stratified model is a good fit.

So the rel-L2 "win" was mostly a particle-noise artifact; on a fair, full-particle
comparison the superellipsoid is **competitive with, not better than,** the fine
grid on real TNG galaxies. The real, **N-independent** advantages are the point:
~150 parameters vs 49,152 cells (~340x compression) at comparable 3D fidelity,
smooth and evaluable at any resolution (the X is not resolution-limited), and
potential-ready (AGAMA `CylSpline`/`Multipole`). Two further honest points: (i) a
MONOLITHIC superellipsoid over-thickens an extended thin disk (554189, figure row
2) - a single shape per iso-density level cannot do thin disk + rounder bulge - so
the backbone should be a DISK + superellipsoid-BULGE decomposition (the GALFIT/MGE
split of Tahmasebzadeh+21), or the even-m Fourier x (R,z) basis (no stratification
assumption); (ii) rel-L2 ~0.4-0.6 is the smooth-model floor (it drops
spiral/asymmetry/clumps). Full particles live in `/mnt/e/dgdp-fullparticles/`.

### Even-m Fourier x (R,z) fixes the disk and beats the grid (2026-06-22)

Why the monolithic superellipsoid over-thickens disks: a single iso-density shape
per level cannot be thin-at-large-R (disk) and round-at-small-R (bulge) at once -
an R-z composite problem. Per-shell orientation (helps tilt only) and an m=4
azimuthal term (helps in-plane bar boxiness only) do NOT address it. The fix, with
NO image decomposition, is to drop stratification and represent the density as
even-m azimuthal Fourier x smooth (R,z) maps (one component;
`fourier_rz_fit`/`fourier_rz_reconstruct`): rho(R,phi,z) = sum_{m=0,2,4,6,8,10}
a_m(R,z) cos(m phi) + b_m(R,z) sin(m phi), each a_m(R,z) a FREE 2D map (control
knots: log R, dense-near-plane z; smooth interpolation). A flat disk and a rounder
bulge coexist natively; the X lives in the z-structure of a_0,a_2,a_4. This is the
AGAMA `CylSpline` form (potential-ready). Full-particle 3D rel-L2 vs SPH-KDE truth:

| galaxy | superellipsoid (144) | Fourier x (R,z) (2002) | grid 0.3125 (49,152) |
| - | -: | -: | -: |
| Shen2010 | 0.381 | 0.385 | 0.465 |
| TNG 554189 (extended disk) | 0.619 | **0.307** | 0.600 |
| TNG 392276 | 0.437 | **0.370** | 0.444 |

The disk over-thickening is FIXED (554189 0.619 -> 0.307; figure col 4 vs col 3),
and the Fourier x (R,z) basis matches/beats the fine grid on all three at ~25x
fewer (smooth, potential-ready) coefficients. The cleanest apples-to-apples win is
vs the superellipsoid (both smooth fits to the same truth): the non-stratified
basis is a far better disk+bulge representation. Caveats: (i) against a smooth
SPH-KDE truth, smooth representations are naturally favored over the histogram grid,
so part of the grid margin is its residual noise/blockiness; (ii) the 2002 count
is reducible - it uses a uniform (R,z) allocation, but m=6,8,10 carry ~0.1% the
power and need far fewer (R,z) DOF, so a power-weighted allocation reaches a few
hundred.

**Recommended backbone: SPH-KDE truth + even-m Fourier x smooth (R,z)** as the
deprojection target (single component, NO image/disk-bulge decomposition; handles
disk+bulge, X-capable, potential-ready via AGAMA `CylSpline`/`Multipole`), grid
demoted to an eval scaffold. The superellipsoid remains a useful compact *bulge*
descriptor (and a clean B/P squareness metric), but is not the universal density
backbone. NOTE: gal3d itself was not installed (sandbox blocked the external
package); the method is reimplemented in numpy/scipy.

## Code added

- `scripts/eval_nbody_shen2010_deprojection.py` - end-to-end OOD deprojection +
  metrics + figures (reuses the validated forward pass from
  `scan_inclination_sensitivity.py` and the X diagnostic from
  `analyze_mdn_x_recovery.py`).
- `scripts/_nbody_shen2010_characterize.py` - raw-snapshot characterization.
- `scripts/_nbody_shen2010_check_xshape.py` - contour-shape boxy/peanut/X test
  (iso-density contour-height z(x) pinch; truth vs baseline vs MDN).
- `scripts/_nbody_shen2010_fourier_basis_demo.py` - confirms the strong X at
  fine resolution, the production-grid smearing, and that an even-m<=10
  azimuthal-Fourier basis preserves the X while smoothing local noise.
- `scripts/build_nbody_structured_basis.py` - builds the fine 0.125 kpc truth
  grid and the structured residual basis (even-m<=10 x PCA-per-m), with the
  X-reconstruction truncation study (foundation).
- `scripts/peanut_strength.py` - tilt-invariant m=4 isophote-b4 boxy/peanut
  metric (`--mode validate` ground-truth controls; `--mode census` on the 185).
- `scripts/_tng392276_xz_check.py`, `scripts/_tng_tilt_scan.py` - 392276 tilt
  diagnosis and the sample-wide x-z tilt scan.
- `scripts/_tng_peanut_crosscheck.py` - external cross-check of the census vs bar
  thickening and catalog A2 (showed the per-galaxy metric is disk/bulge-confounded).
- `scripts/build_structured_basis_tng.py` - cross-galaxy basis dimension study
  (per-m rejected; Fourier-filter + global PCA recommended).
- `scripts/_nbody_shen2010_span_check.py` - does a TNG-fit basis represent the
  Shen2010 X (partial, ~60-70%).
- `scripts/flow_matching_prototype.py` - conditional flow matching vs the
  1-Gaussian MDN on the filter+global-PCA target (TNG).
- `scripts/_nbody_shen2010_gridfree.py` - mesh-free SPH-KDE + superellipsoid
  iso-density shape fit (grid-free proof of concept on Shen2010).
- `scripts/reconstruct_superellipsoid_3d.py` - reconstruct smooth 3D density from
  superellipsoid shells AND from an even-m Fourier x (R,z) basis, and compare 3D
  fidelity to the cylindrical grid vs an SPH-KDE truth (Shen2010 + TNG 554189 +
  392276; `--tng-ids`, `--tng-particle-dir`; uses full uncapped particles).

`ruff check .` clean; `pytest -q` 102 passed (additive scripts, no code changes
to the package).
