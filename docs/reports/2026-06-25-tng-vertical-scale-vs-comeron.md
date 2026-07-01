# TNG50 stellar vertical scale vs Comerón+2018 — the OOD-axis decision (Step 1)

**Date:** 2026-06-25
**Question (handoff Step 1):** is TNG50's stellar vertical scale *inflated* (softening/
resolution) or does it *span real galaxies*? This decides what option 2 (retrain the
`q_m` vertical head) must fix before any retrain.
**Branch:** `codex/milestone2-tng-ingestion-design` (not committed).
**Scripts:** `scripts/measure_tng_vertical_scale.py`, `scripts/compare_tng_comeron_vertical.py`.
**Outputs:** `outputs/tng50_vertical_scale/` (`tng_vertical_scale.{csv,png}`,
`tng_vs_comeron_vertical.png`, `*_summary.json`, `*_profiles.npz`).

## TL;DR verdict

**For the end-to-end (total stellar light) deprojection of galaxies in the TNG mass
selection (M\* > 10^9.5), TNG's vertical thickness is largely NOT the OOD.** Two
refinements to the first-pass "TNG is softening-inflated" reading (added 2026-06-25 after
discussion):

1. **Sample-weighted, the integrated thickness matches real S4G.** RMS|z| (the metric
   that drives the potential); per-galaxy TNG/Comerón ratios (v_c≤300, 8 extrapolation
   galaxies dropped):

   | selection | N | thick h_z ratio | **RMS\|z\| ratio** | % within ±40% (RMS) | thin h_z |
   |---|---|---|---|---|---|
   | all (>10⁹·⁵) | 177 | 1.19 | **1.07** | 81% | 2.0 |
   | >10¹⁰ | 155 | 1.12 | 1.04 | 91% | 2.0 |
   | >10¹⁰·⁵ | 114 | 1.01 | 0.98 | 96% | 1.8 |

   Median RMS|z| is within 7% of the relation over the whole selection (inside its own
   scatter, ρ≈0.55). The clear excess is confined to log M\* 9.5–10 (12% of the sample)
   and to the **thin disc** (≈2×) — which is resolution-floored *and* low-impact (its mass
   sits near the midplane, so its h_z barely changes v_c or RMS|z|).

2. **The stellar halo is the CORRECT target, not contamination.** A low-inclination image
   integrates *all* stellar density along z (disc+bulge+halo), so the reconstruction
   target IS the total stellar density; TNG's total-density truth is right, and the high
   `f_thick` / flaring are largely the legitimate halo+CMC in total light. Comerón's
   *disk-only* decomposition under-states the real total-light thickness, so the true
   over-thickening is even smaller than 1.07. (Re-confirms NGC 4321: learned RMS|z|≈1.54
   vs Comerón disk-only ≈1.4–1.7 at its mass; adding the galaxy's own halo makes the real
   total thicker → learned q_m is realistic-to-slightly-thin.)

**What's actually left as OOD:** (a) the unresolved thin disc (minor for the potential);
(b) whether TNG's *own* halo is realistic (softening puffs the inner regions; TNG
halo-mass/extent tensions) — unverifiable from face-on data, second order for v_c;
(c) **morphology** — TNG is massive-barred-only and lacks grand-design spirals like
NGC 4321. Thickness/h_z is mostly fine. See "Is the N-body retrain warranted?" below.

## What I measured (TNG truth)

From the milestone-2d fine-z truth grids `/mnt/e/dgdp-milestone2d/density_residual_table.npz`
(`truth_density` (1665,32,48,32) = (sample, R, φ, z); 185 unique galaxies × 9 projections,
truth identical across projections; z grid ±5 kpc at 0.3125 kpc; R log-spaced to 30 kpc).
All 185 are **barred** (log M* 9.5–12.2, median 10.67; bar length 2.0–7.7 kpc).

Per galaxy, in the **disk region** R ∈ [bar_length, R(95% beyond-bar mass)] (bar+bulge
excluded, per the caveat that the truth is *total* stellar density):
- `RMS|z|(R)` (model-free; reuses `rms_z_profile`), and `RMS|z|_disk` (mass-weighted).
- single- and thin+thick **sech²** fits in log-space → `h_thin, h_thick, f_thick`.
- Convention: ρ(z) ∝ sech²(z/2h), so `h` is the **exponential** scale height
  (ρ→e^{−|z|/h}); `RMS|z| = 1.814 h` per component.

Sanity checks: grids are centred (mean z ≈ 0, peak at the central cell); only ~40% of
disk-region stellar mass lies within |z|<0.5 kpc and ~45–50% beyond 1 kpc — genuinely
thick, not a centring artefact.

TNG medians by mass bin:

| log M* | N | RMS\|z\| | h_single | h_thin | h_thick | f_thick | flare |
|---|---|---|---|---|---|---|---|
| 9.5–10.0 | 22 | 1.57 | 1.02 | 0.50 | 1.38 | 0.49 | 1.77 |
| 10.0–10.5 | 41 | 1.60 | 1.05 | 0.49 | 1.63 | 0.42 | 1.97 |
| 10.5–11.0 | 81 | 1.73 | 1.16 | 0.57 | 1.71 | 0.50 | 1.57 |
| 11.0–11.5 | 36 | 1.93 | 1.35 | 0.50 | 1.92 | 0.65 | 1.44 |
| 11.5–12.5 | 5 | 2.43 | 2.12 | 0.59 | 2.50 | 0.81 | 1.29 |

## The real reference (Comerón, Salo & Knapen 2018, A&A 610, A5)

Read from the user-supplied PDF (`F:/OneDrive/Desktop/aa31415-17.pdf`); values are the
paper's, not fabricated. Key points (line refs in `compare_tng_comeron_vertical.py`):

- **Convention:** density ρ(z) ∝ sech²(z/z₀); the *reported* `zt, zT` are the profile
  **tail e-folding lengths = z₀/2 = the exponential scale height** — the **same** as my
  TNG `h`, so the comparison is direct (no factor-of-2). Verified against their own MW
  check: Eq. 18 gives thin 26+1.23·218 = 294 pc (~300) and thick(gas-rich)
  −217+7.42·218 = 1401 pc (~1450) at v_c=218.
- **Eq. 18 / Table 4:** ⟨z_i⟩ = A + B·v_c (pc, km/s) — thin all A=26 B=1.23;
  thick all A=−262 B=8.64; gas-rich thin A=−12 B=1.42, thick A=−217 B=7.42. Scatter is
  large (ρ≈0.55 all; 0.7–0.9 gas-rich).
- **v_c** is the **maximum circular velocity** (total-mass proxy); sample v_c=50–300,
  mostly 100–200; MW-sized (v_c>200) rare (9/141).
- **Mass ratio:** thick discs dominate low-mass galaxies, are sub-dominant in massive
  ones — MT/Mt 0.2–1 for v_c>120, up to ~3 below → `f_thick` falls from ~0.75 (low) to
  0.17–0.5 (massive).

**One external input (stated):** to put TNG (known M*) and Comerón (known v_c) on a
common axis I map M*→v_c with an MW-anchored stellar Tully-Fisher,
v_c = 218·(M*/10^10.78)^(1/n), n=4 (bracketed 3.5–4.5) — the *same* MW point
(v_c=218, Licquia & Newman M*) Comerón validate against. **It does not enter any h_z
value**; only the x-alignment. Low-mass excess is robust across the bracket (1.6–2.0×).

## Comparison (figure `tng_vs_comeron_vertical.png`, four panels)

| log M* | v_c est | TNG zT | Com zT | zT ratio | TNG zt | Com zt | TNG f_thick | Com f_thick | TNG RMS | Com RMS |
|---|---|---|---|---|---|---|---|---|---|---|
| 9.5–10.0 | 119 | 1.38 | 0.76 | **1.8×** | 0.50 | 0.17 | 0.49 | →0.6+ | 1.57 | 0.84 |
| 10.0–10.5 | 167 | 1.63 | 1.18 | 1.4× | 0.49 | 0.23 | 0.42 | 0.17–0.5 | 1.60 | 1.28 |
| 10.5–11.0 | 212 | 1.71 | 1.57 | **1.1×** | 0.57 | 0.29 | 0.50 | 0.17–0.5 | 1.73 | 1.69 |
| 11.0–11.5 | 264 | 1.92 | 2.02 | 0.95× | 0.50 | 0.35 | 0.65 | 0.17–0.5 | 1.93 | 2.18 |
| 11.5–12.5 | 397* | 2.50 | 3.16* | 0.79×* | 0.59 | 0.51 | 0.81 | 0.17–0.5 | 2.43 | 3.40* |

\* v_c>300 is **beyond Comerón's range** — Comerón columns there are an extrapolation of
a steep relation (B=8.64 pc/(km/s)) and are not trustworthy; the apparent "TNG thinner
than real" at the top mass bin is that artefact, not physics.

## Bonus: this corrects the NGC 4321 "2.3× too thick" alarm

The 2026-06-24 quick-check called the learned `q_m` prediction (RMS|z|≈1.54 kpc)
"~2.3× thicker than Comerón". That used a representative prior `h_thin/h_thick =
0.4/1.2 kpc` **as z₀** in sech²(z/z₀) (→ exp scale heights 0.2/0.6, RMS 0.67 kpc) — far
thinner than Comerón's *actual* relation. At NGC 4321's mass (log M*≈10.8, v_c≈210)
Comerón Eq. 18 gives exp `zt`≈0.30, `zT`≈1.6 kpc; with a realistic f_thick≈0.3 the real
thin+thick **RMS|z| ≈ 1.6–1.7 kpc**. So the learned 1.54 kpc is **realistic, not 2.3×
too thick** — the alarm was a too-thin / convention-confused reference, not an OOD
thickness. With the total-light framing the flaring and high `f_thick` are largely
legitimate (the halo's relative weight grows outward), so the genuine NGC 4321 residual
is **morphology** (a grand-design spiral vs barred TNG training) and the unresolved thin
disc — not gross over-thickness.

## Is the N-body retrain (option 2) still warranted?

**Largely no — and a naive pure-disc N-body retrain would be a step backward.**

- **No halo.** The deferred N-body library is pure-disc (collisionless), with no stellar
  halo. Having established that the deprojection target *should* include the halo,
  training q_m on halo-free truth would make it predict too-thin total-light density — an
  OOD in the opposite direction. N-body-only retraining is wrong unless a halo is
  synthetically added to every model.
- **Wrong morphology for the NGC 4321 use case.** Isolated collisionless N-body discs
  produce **bars + boxy/peanut bulges + transient/flocculent arms**, not the gas-mediated
  two-armed **grand-design** pattern of NGC 4321. So N-body does not fill the grand-design
  gap — it spans the *same* barred/B-P class as TNG, with more B-P diversity.
- **Thickness no longer needs fixing** (refinements 1–2), which was the library's original
  motivation.

**Where N-body *does* belong:** a controlled **validation / stress-test** set for what TNG
can't exercise — chiefly B-P/X-bulge recovery (milestone-2d: a leave-one-out MDN can't
recover the single strong X, subhalo 392276, with no X left in training). That is the
test-set role already used for the Shen2010 model, not a training replacement.

**If option 2 proceeds for morphology**, keep the halo: **broaden the hydro (TNG)
selection** — drop the barred-only cut; add unbarred / lower-mass / spiral-dominated
galaxies (which retain their halos and total-light vertical structure) — rather than swap
in pure-disc N-body. Re-scope option 2 from "span realistic h_z" to "broaden morphology
while keeping the halo-inclusive total-light target." Given thickness is mostly fine,
validating the current q_m on more real images is the cheaper first step.

## Caveats

- `f_thick` (and to a lesser extent `zT`) from a 2-component fit to *total* stellar
  density absorbs the accreted halo + CMC that Comerón's disk-only decomposition excludes.
  For the end-to-end total-light deprojection this is the **correct** target (the image
  contains halo light), so the high `f_thick` is a feature, not contamination — Comerón's
  disk-only MT/Mt is simply the wrong reference for it. It would only be a problem if
  TNG's *own* halo were unrealistic (see verdict (b)).
- `h_thin`≈0.5 kpc is at the **resolution floor** (0.3125 kpc grid, ~0.2–0.3 kpc
  softening); treat it as "≳ real, unresolved", not a measured value.
- M*→v_c via STFR carries scatter; v_c>300 (top ~5 galaxies) is extrapolation.
- TNG stellar masses are subhalo totals (up to log M*=12.2); only log M*≲11.3 overlaps
  Comerón. The comparison is strongest at log M* 10–11 (v_c 150–260), where the bulk of
  the TNG sample and NGC 4321 sit.
