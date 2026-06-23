# Potential/force validation of the 3D density representations

Date: 2026-06-23

Does the adopted even-m Fourier x (R,z) representation (and its power-weighted
compression) reproduce the *dynamics* - potential, forces, rotation curve - of the
truth, and how does it compare to the cylindrical grid and the superellipsoid? This
is the potential/force validation flagged as next-step (3); it follows
`2026-06-23-fourier-rz-target-deprojection.md`.

> **Headline (after the softening/method robustness check below):** under a *fair,
> self-consistent* comparison - every representation, and the particle reference,
> run through the SAME Poisson solver - the Fourier x (R,z) rep is **competitive with
> the cylindrical grid** for forces (full Fourier ~6-12% vs grid ~6-23%), and is
> *better* on the extended lower-N disk. An earlier draft of this report used only a
> direct-sum reference at a single softening and overstated the grid's advantage
> (~3-4x); that gap was largely an artifact of the reference, not a real deficiency.

## Setup

AGAMA's `CylSpline` (the native solver the representation targets) needs a C++
compiler the sandbox lacks, so this uses a dependency-free **isolated FFT
Poisson/force solver** (`src/dgdp/poisson_fft.py`, Hockney-Eastwood zero-padded
convolution; unit-tested against the point-mass Kepler solution to <3% potential,
<5% force). The maps ARE in CylSpline form; this measures the same density->force
fidelity.

Per galaxy (Shen2010 + TNG 554189 + 392276, full particles) each density is sampled
on a common fine Cartesian box (+-10 kpc x,y / +-3.5 kpc z at 0.25 kpc - fully
inside the Fourier r_max=15 / z_max=4 domain, so no edge clamping; the bar +
inner-disk region where the bar/peanut dynamics live) and run through the solver.
Representations: SPH-KDE truth, Fourier x (R,z) full (2002 coeff), Fourier x (R,z)
compressed (the Task-1 power-weighted 496-coeff allocation), cylindrical grid
(0.3125 kpc), nested superellipsoid. Reproduce:
`.venv/bin/python scripts/validate_fourier_rz_potential.py` and
`.venv/bin/python scripts/check_potential_softening.py`.

## The reference matters: softening sweep + a self-consistent comparison

The "truth" force is not unique. The real simulation solves Poisson (TreePM); a
direct particle sum is one choice and it depends on the softening eps. Worse, the
representations' forces come from the FFT Poisson solver (effective softening ~ the
0.25 kpc grid), so comparing them to a direct sum at a different eps is inconsistent.
`scripts/check_potential_softening.py` tests both:

**(a) Softening sweep** (direct sum, eps = 0.05..1.0 kpc): the median force error
swings ~10x (e.g. Fourier-full on Shen2010: 0.12 at eps=0.05 -> 1.29 at eps=1.0) and
the two references (direct-sum at eps=0.3 vs the self-consistent PM below) disagree
by 8-31%. Much of the large-eps growth is a softening *mismatch* - the reps'
grid-Poisson forces are fixed at ~0.25 kpc softening, so once eps exceeds that the
reference is softer than the reps and *everyone's* error inflates. So the direct-sum
numbers are only interpretable near eps ~ the grid scale, and even there the choice
tilts the result.

**(b) Self-consistent PM reference** (the fair test): deposit the particles on the
same box and run them through the SAME FFT Poisson solver as every representation -
no direct sum, no eps choice; everyone solves Poisson identically. Median fractional
force error over the body vs this PM reference:

| representation | Shen2010 | TNG 554189 | TNG 392276 |
| - | -: | -: | -: |
| cylindrical grid 0.3125 | **0.095** | 0.227 | **0.064** |
| Fourier x (R,z) full | 0.124 | **0.064** | 0.088 |
| Fourier x (R,z) compressed | 0.147 | 0.133 | 0.111 |
| superellipsoid | 0.124 | 0.278 | 0.135 |
| SPH-KDE | 0.028 | 0.161 | 0.019 |

For contrast, the direct-sum-at-eps=0.3 reference gave grid 0.06-0.09 vs Fourier
0.18-0.35 (the "3-4x" of the first draft).

Rotation curve v_c(R) rms error and potential rms error (vs the SPH-KDE-truth field;
curves peak ~150-230 km/s):

| representation | pot-err (Shen / 554189 / 392276) | v_c rms [km/s] (Shen / 554189 / 392276) |
| - | -: | -: |
| cylindrical grid | 0.05 / 0.06 / 0.05 | 5.8 / 3.6 / 7.1 |
| Fourier full | 0.15 / 0.20 / 0.10 | 12.2 / 11.9 / 10.5 |
| Fourier compressed | 0.20 / 0.35 / 0.16 | 17.0 / 20.0 / 16.2 |
| superellipsoid | 0.04 / 0.21 / 0.07 | 11.4 / 13.0 / 17.3 |

(`figures/potential_softening_check.png`, `figures/fourier_rz_potential_validation.png`,
`fourier_rz_potential_metrics.json`.)

## Findings

1. **Under a consistent Poisson comparison the Fourier x (R,z) rep is competitive
   with the grid** (full Fourier ~6-12% vs grid ~6-23%), not 3-4x worse. Neither
   dominates: the grid is marginally better on the two high-N, centrally-concentrated
   galaxies (Shen2010, 392276); the smooth Fourier is markedly better on the
   extended, lower-N disk (554189), where the cylindrical grid is noisy.

2. **The earlier large gap was an artifact of the direct-sum reference.** At the
   matched softening the direct sum still carries small-scale force fluctuations from
   local particle clumps / N-body discreteness. The smooth Fourier rep cannot (and
   should not) reproduce those - Shen2010 is a Monte-Carlo sampling of a *smooth*
   disk, so they are noise, not signal - whereas the grid partly does (its density
   carries the same clumps), so it scored well *specifically against the direct sum*.
   The PM reference smooths them consistently for everyone and removes the penalty.
   At low N, smoothing genuinely helps (bias-variance) - hence Fourier winning 554189.

3. **The rotation curve v_c(R) is competitive for full Fourier; only the compressed
   rep lags.** vs the PARTICLE rotation curve (`scripts/plot_rotation_curves.py`,
   `figures/rotation_curve_comparison.png`) the v_c rms error is: grid 9.0 / 14.5 /
   8.4, Fourier full 11.3 / **7.6** / 10.0 (best on 554189), Fourier compressed 16.4 /
   13.7 / 16.1, superellipsoid 10.2 / 21.5 / 17.0 km/s (Shen / 554189 / 392276). So
   full Fourier ~ grid; the *compressed* rep's coarse log-spaced R-knots (12-14 over
   0.12-15 kpc) under-resolve the radial mass profile. An n_R sweep (full rep) shows
   the outer v_c overshoot is two effects: (i) linear interpolation of the convex
   (exponential) outer profile between sparse log-knots overestimates it - n_R 14->24
   removes most of it (Shen outer-R rms 12->8 km/s); (ii) a residual floor that more
   knots do NOT remove (n_R=48 still overshoots v_c(R~8) by ~7 km/s) because the maps
   are sampled from adaptive SPH-KDE, which over-smooths / does not conserve mass in
   the steep sparse outer disk - the histogram grid avoids this by construction. To
   match the grid's outer v_c, build a_0(R) mass-conservingly (fit_fourier_rz_from_grid
   or renormalize to the enclosed-mass profile), not just add knots. (The
   v_c-vs-SPH-KDE-field table above inflates the gap; the particle reference is fairer.)

4. **Compression cost is modest** (full -> compressed adds ~0.02-0.07 to the force
   error) and stays competitive with the grid; the superellipsoid is erratic
   (best on 554189, worse on the bars).

5. **"Potential-ready" is supported once measured fairly.** The Fourier x (R,z) form
   gives forces comparable to the fine grid (better at low N), is smooth / analytic /
   cheap (~500 params, CylSpline-ready), and is the natural object for a smooth
   potential. The fine grid remains (marginally) the most accurate for the
   centrally-concentrated, well-sampled cases and for v_c. The *density rel-L2 vs a
   smooth truth* (Task 1) and the *image->density prediction* (companion report) are
   separate questions and are unchanged by this.

## Caveats and next steps

- The fairest reference is the self-consistent PM force (everyone solves Poisson
  identically). Direct summation is softening-dependent and penalizes smooth reps
  for sub-resolution / discreteness force fluctuations; quote it only near eps ~ the
  grid scale.
- AGAMA `CylSpline` (the native solver) was substituted by a tested FFT Poisson
  solver for this report. UPDATE (2026-06-23): AGAMA IS now available (the user
  pre-built `agama 1.0.159`; import into `.venv` via
  `PYTHONPATH=/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313`), so a
  CylSpline cross-check of these FFT-Poisson numbers is now possible and is the
  natural follow-up. The PM reference is at the box resolution (0.25 kpc); a finer
  truth would add a common resolution floor to all reps but should not change the
  ranking.
- v_c(R): add R-knots to the Fourier (R,z) maps if the rotation curve is the target;
  for the full 3D force field the full Fourier already matches the grid.
- This report corrects an earlier draft that used only the direct-sum reference and
  concluded "grid 3-4x better"; the fair comparison shows competitiveness.

## AGAMA CylSpline cross-check (`scripts/agama_potential_crosscheck.py`)

AGAMA became available (user pre-built `agama 1.0.159`; imported into the `.venv` via
the Agama build dir). Redone with the *real* CylSpline solver: gold = AGAMA CylSpline
from the particles; each representation = AGAMA CylSpline from its density; median
fractional force error over the body, plus an AGAMA-vs-FFT-Poisson agreement check on
the same density.

| representation | Shen2010 | TNG 554189 | TNG 392276 |
| - | -: | -: | -: |
| SPH-KDE truth | 0.009 | 0.072 | 0.023 |
| **cylindrical grid** | **0.014** | **0.025** | **0.037** |
| Fourier full | 0.085 | 0.063 | 0.103 |
| Fourier compressed (496) | 0.130 | 0.165 | 0.140 |
| superellipsoid | 0.125 | 0.195 | 0.130 |

AGAMA-vs-FFT-Poisson solver agreement (median frac force diff on the same density):
Fourier full **0.026-0.053** (the solvers agree on smooth fields - the in-house
FFT-Poisson is validated), but grid **0.106-0.226** (they differ because AGAMA's
CylSpline *re-fits/smooths* the blocky grid density onto its own 25x25 spline, while
FFT-Poisson uses it raw).

Findings: (1) the in-house FFT-Poisson solver is **validated** against AGAMA on smooth
densities (~3-5%). (2) With the standard CylSpline (which denoises the input onto its
spline basis), the **cylindrical grid is the most force-accurate representation (1-4%)**
- *more clearly* than the raw-FFT-Poisson comparison showed, because AGAMA removes the
grid's Poisson noise (the thing that hurt it raw, esp. low-N 554189), leaving its
unbiased mass-conserving enclosed mass. (3) The **Fourier rep is ~6-10% in BOTH solvers**
(solver-independent) - the cost of its pre-smoothing/compression; the 496-coeff
compression adds more (13-17%). (4) Practical reading: for the most accurate *potential*,
feed CylSpline the most faithful density (grid/particles -> 1-4%); the Fourier x (R,z) rep
is the compact, smooth, predict-from-image *density/deprojection* object (~6-10%
force-faithful), not the way to squeeze the last few % out of the potential. This
refines the FFT-Poisson "Fourier competitive with grid" reading: that came from the raw
grid's noise; with the denoising CylSpline the grid is clearly best.

### What the ~10% Fourier-vs-particle gap is (`scripts/_agama_smoothing_decomposition.py`)

Both the Fourier rep and the (triaxial) particle gold drop the odd-m asymmetry, so the
gap is NOT the asymmetry. Decomposed against the FULL particle potential (`gold_full` =
CylSpline particles, symmetry='none', mmax=8, which KEEPS odd-m), median force error:

| contribution | Shen | 554189 | 392276 |
| - | -: | -: | -: |
| drop odd-m asymmetry + m>6 (gold_even vs gold_full) | 0.017 | 0.014 | 0.008 |
| drop all non-axisymmetry, m=0 only (gold_axi) | 0.094 | 0.098 | 0.074 |
| Fourier full (total) | 0.098 | 0.067 | 0.105 |
| Fourier full vs gold_even (same even-m<=6; (R,z)+SPH only) | 0.094 | 0.065 | 0.105 |

So the odd-m asymmetry costs only **~1-2%**; the Fourier error is **almost entirely the
(R,z)+SPH-KDE smoothing** (fourier_full vs gold_even ~= fourier_full vs gold_full). The
two smoothings differ in the radial/vertical plane (SPH-KDE k=32 + coarse 14x13 knots vs
a 25x25 spline fit straight to the particles), not azimuthally. Implication: closing the
gap means refining the (R,z) representation (more knots / less pre-smoothing), not adding
asymmetric harmonics - consistent with the v_c finding and "the radial anchor is the lever".

### Finer (R,z) grid closes the force gap (2026-06-24)

The (R,z)-smoothing gap is just knot resolution. `fit_fourier_rz`'s default was bumped
from 14x13 to **25x25** (n_r=25, n_z_half=12); `fourier_full` then **matches the gold**.
AGAMA force error vs the particle gold: fourier_full **0.019 / 0.028 / 0.029**
(Shen / 554189 / 392276) vs grid 0.014 / 0.025 / 0.037 - a match (was 0.085 / 0.063 /
0.103 at 14x13). Re-run validate: pot-err vs SPH 0.02-0.10 (was 0.10-0.20), v_c-rms
2.3-5.7 km/s (was 10-12, now <= grid on Shen and 392276). This is a **representation-only**
change - NO retrain - because it is a direct fit to a known density; the deprojection
*predicted* target stays the compact ~500 coeff (prediction is PCA-32-limited, so a finer
predicted grid would not help recovery). The compressed (496) stays ~0.13-0.16 (AGAMA),
as expected. (Note: validate's *direct-sum* gold still shows fourier_full ~0.10-0.26
because that gold is sharp + the rep carries SPH-KDE smoothing; the AGAMA-Poisson gold -
the right reference per the no-direct-sum discussion - shows the match.) Historical
studies (`compress_fourier_rz_target.py`, `reconstruct_superellipsoid_3d.py`) pin
n_r=14,n_z_half=6 so their numbers and the frozen deprojection allocation are unchanged.

## Code and artifacts

- `src/dgdp/poisson_fft.py` - isolated FFT Poisson/force solver + v_c(R) helper
  (`tests/test_poisson_fft.py`, point-mass Kepler check).
- `scripts/validate_fourier_rz_potential.py` - per-galaxy validation; writes
  `fourier_rz_potential_metrics.json` and `figures/fourier_rz_potential_validation.png`.
- `scripts/check_potential_softening.py` - the softening sweep + self-consistent PM
  reference; writes `figures/potential_softening_check.png`.
- `scripts/plot_rotation_curves.py` - stellar v_c(R) per representation vs the
  particle rotation curve; writes `figures/rotation_curve_comparison.png`.
- `scripts/agama_potential_crosscheck.py` - AGAMA CylSpline cross-check (real solver)
  + AGAMA-vs-FFT-Poisson agreement; run with
  `PYTHONPATH=/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313` (or it
  self-inserts that path); writes `agama_crosscheck_metrics.json` and
  `figures/agama_potential_crosscheck.png`.
- `scripts/_agama_smoothing_decomposition.py` - decomposes the Fourier-vs-particle force
  gap into odd-m asymmetry (~1-2%) vs (R,z)+SPH smoothing (the rest).
