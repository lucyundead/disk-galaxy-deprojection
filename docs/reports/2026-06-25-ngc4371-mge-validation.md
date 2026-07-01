# NGC 4371 (SB0): TNG-learned q_m vs an independent MGE deprojection

**Date:** 2026-06-25
**Goal:** test the TNG-learned vertical prior `q_m` on a real galaxy that is *in* the TNG
barred-morphology class (an SB0), by comparing our deprojection to Behzad Tahmasebzadeh's
**independent MGE deprojection** of the same galaxy. This is a galaxy-specific benchmark,
stronger than the statistical Comerón comparison.
**Branch:** `codex/milestone2-tng-ingestion-design` (not committed).
**Scripts:** `scripts/ngc4371_mge_benchmark.py`, `scripts/compare_ngc4371_learned_vs_mge.py`,
`scripts/deproject_real_image_ngc4321_learned.py` (now takes `--galaxy-name/--pix-arcsec/
--pa-pix-deg/--center-x/-y/--mask` to run on a WCS-less S4G cutout).
**Data (from Behzad, staged in `NGC4371/`):** S4G 3.6 μm image + PSF + mask + GALFIT model
(disk + inner disk + bar + nucleus, **no classical bulge**) + the MGE notebook.

## TL;DR

**The TNG-learned vertical structure is independently validated.** Our learned RMS|z|(R)
matches the MGE deprojection to **~15–20 % over R=0.5–8 kpc**, both flaring 0.5→~1.6–2 kpc;
the naive thin sech²(h=0.3) baseline is 3–6× too thin. The thick, flaring q_m that the
2026-06-24 alarm flagged as "too thick" is **corroborated by an independent method on a
real galaxy** — it is physical, not a TNG artifact.

**A reconstruction bug was found and fixed.** The *conserving reconstruction* did not
preserve the in-plane Σ(R) on this strong-barred, centrally-peaked SB0 (max rel. diff
198 %; central mass fraction R<3 kpc dropped 0.52→0.24), which depressed the learned v_c
(peak 131 vs the MGE's 179). Root cause: `reconstruct_smooth` normalises ∫q dz=1 on the
*knot* grid, but resampling to the fine z grid let Σ(R) drift for an OOD q. Fix: re-impose
the image anchor per column (keep the vertical shape, set ∫ρ dz = Σ_image(R,φ)). After the
fix Σ(R) is preserved to 0.2 % and the learned v_c peak is **175 km/s ≈ MGE 179**, with the
RMS|z| validation unchanged. The vertical prior itself needed no change.

## Setup

GALFIT geometry → pipeline inputs: i=58° (= arccos disk axis ratio 0.536), disk PA(pixel)
≈1.8° (=90°+GALFIT −88.2°), centre (254.6,152.8), 0.75″/pix, D=16.194 Mpc, mask applied,
M\*=3.53×10¹⁰ (M/L=1, to match the MGE total). **i=58° is in-distribution** — the TNG mocks
span inclinations 20/40/60°.

**MGE benchmark (Behzad):** triaxial-bar + oblate-disk deprojection, viewing angle
(θ,φ,ψ)=(59°,−11°,89°); intrinsic disk flattening q≈0.28 (round inner disk 0.95), thin
bar (q≈0.12–0.22); total L=3.53×10¹⁰ (M/L=1, self-consistent at D=16.194; the notebook's
2.42×10¹⁰ used a stale 13.4 Mpc in one cell). AGAMA `CylSpline` potential.

## Results

RMS|z|(R) [kpc]:

| R | ours baseline | ours learned | MGE | learned/MGE |
|---|---|---|---|---|
| 0.5 | 0.27 | 0.48 | 0.42 | 1.16 |
| 1 | 0.27 | 0.69 | 0.57 | 1.21 |
| 2 | 0.27 | 1.03 | 0.88 | 1.17 |
| 3 | 0.27 | 1.30 | 1.11 | 1.17 |
| 5 | 0.27 | 1.58 | 1.39 | 1.14 |
| 8 | 0.27 | 1.64 | 1.68 | 0.98 |
| 12 | 0.27 | 1.61* | 2.00 | 0.80 |

\* beyond R≈14 the learned reconstruction has an edge artifact (RMS|z|→0); ignore R>12.

Stellar v_c (M/L=1, after the anchor fix): **ours baseline 182, ours learned 175, MGE
179 km/s** — all three track each other across the curve (the thick vertical trims v_c by
only 1–4 % vs the thin baseline). Before the fix the learned peak was 131 (the Σ(R) drift).
Comerón disk equivalent at v_c≈180: RMS|z|≈1.40 kpc (consistent, mid-disk).

## Interpretation

- **Vertical prior validated.** Two independent deprojections — a TNG-trained ML prior and
  a geometric MGE — agree on NGC 4371's vertical profile to ~15–20 %, including the
  flaring. Combined with the earlier sample-weighted Comerón result, this closes the
  "is the learned q_m too thick?" question: **no, it is realistic.**
- **The v_c gap was in-plane, not vertical — now fixed.** v_c is set by M(<R)/R; RMS|z|(R)
  is a per-radius vertical moment, so the two are orthogonal (one passed while the other
  failed). The learned reconstruction had de-concentrated the disk — half-mass radius
  5.0 kpc vs the MGE/image 2.6–2.9 kpc; 25 % of mass within R<3 kpc vs 51 % — dropping v_c
  by ~√2. Re-imposing the image anchor per column restored R_half and v_c (175≈179) while
  leaving the validated vertical structure intact.

## Face-on / edge-on density comparison

`scripts/plot_ngc4371_density_comparison.py` (fig
`outputs/real_images/ngc4371_density_faceon_edgeon.png`) renders both reconstructions as
line-of-sight surface density with the bar aligned to x. **Face-on:** both show a bar
along x + an extended disk of similar central concentration (learned slightly boxier, MGE
outer disk rounder). **Edge-on:** both show a central concentration + a thick, flaring disk
of similar vertical extent — the visual counterpart of the RMS|z| agreement. The learned
render is z-symmetrised (a face-on deprojection carries no up/down information), tapered
beyond R≈11 kpc (where q_m is unconstrained), offset off the x=0 axis, and percentile-
scaled — removing the earlier rendering artifacts (central hot pixel, x=0 interpolation
seam, faint large-R features, and the top/bottom edge-on asymmetry).

## Rotation curve and bar strength

`scripts/compare_ngc4371_vc_and_bar.py` (fig `outputs/real_images/ngc4371_vc_and_bar.png`),
both via AGAMA CylSpline with an **azimuthally-averaged** v_c (orientation-independent; the
MGE's compact central Gaussians make direct-summing on the coarse grid unreliable, so we
build potentials instead):

- **Rotation curves agree** — fixed-learned peak 179 vs MGE 183 km/s, tracking each other
  across R (a learned bump of ~+8 % near R=5 kpc). v_c is set by the shared m=0 mass.
- **Bar strength differs.** The learned in-plane Σ(R,φ) *is* the deprojected image, so its
  A2(R) is the image's bar: peak **A2≈0.33 at R≈3.4 kpc**. The MGE's smooth 2-Gaussian bar
  gives **A2≈0.17 at R≈2.3 kpc** — about half. This reflects the MGE under-representing a
  strong bar with few Gaussians (it is *not* the learned vertical step — both share the same
  in-plane image otherwise); our image-anchored in-plane retains the full bar m=2 amplitude,
  which matters for bar dynamics (torques, pattern speed). Learned A2 at R<1 and R>8 kpc is
  low-S/N image noise, not bar.

## Status / next

NGC 4371 validation complete: vertical structure matches the MGE to ~15–20 %, and after the
anchor fix the rotation curve matches too (175 vs 179 km/s). The anchor fix is universal
(≈no-op for the already-conserved NGC 4321). Minor remaining items (low priority, none
affect v_c or the inner disk): the learned q_m still has weak large-R (R>12) vertical
structure on this strong bar (tapered in the figure), and a slight z-asymmetry — the head
predicts on full-z knots, but a face-on deprojection has no up/down information, so
enforcing z-symmetry on q_m at the source would be a clean pipeline tidy-up (currently
symmetrised only for the figure).
