# Idea 2 diagnostic: would a mass-dependent baseline help the MDN? — No.

**Date:** 2026-07-01
**Verdict:** **SKIP the retrain.** The TNG mass→thickness relation is weak, and the current
model already captures it (thickness error is flat in M*). An explicit mass-dependent sech²
baseline would not improve the in-distribution learned vertical structure.
**Branch:** `codex/milestone2-tng-ingestion-design` (not committed at time of writing).

## Background (premise correction)

Idea 2 proposed a mass-dependent baseline scale height `hz(M*)` to "make the training residual
clearer per galaxy so the MDN learns the essential vertical structure." Reading the code showed
the premise does not hold: the training target is the TNG truth's *normalized* vertical profile
(`vertical_mixture.weights_target`), PCA'd with **mean subtraction**; the geometric baseline
enters only as a z-integrated total mass feature and the z-integrated inference anchor — hz is
inert. The model already learns deviations, but from the **PCA population-mean profile**. The
real, implementable version of idea 2 (target = truth − sech²(hz(M*)), + a cluster retrain) was
gated behind this cheap read-only diagnostic. See
`docs/superpowers/specs/2026-07-01-mass-baseline-diagnostic-design.md`.

## Diagnostic

`scripts/diagnose_mass_baseline.py`, run on the cluster (login node, memmap'd big arrays) against
`outputs/tng50_milestone2d_rich/density_residual_table.npz` (185 galaxies × 198 projections =
36630 rows; train 21978 / val 7326) + the committed R=64 bundle. Pure numpy, no training.

## Results

**1. Mass→thickness relation in TNG truth (per galaxy).** eff RMS|z| (mass-weighted, R<12 kpc)
= 0.64–1.97 kpc over M* = 3.1×10⁹–1.1×10¹². Log-log fit **slope 0.108, R²=0.32, Pearson r=0.56,
scatter 0.076 dex**. Real but weak — mass explains ~⅓ of the thickness-scale variance;
`hz(M*)` ≈ 0.96 / 1.08 / 1.23 kpc at M* = 10¹⁰ / 10¹⁰·⁵ / 10¹¹.

**2. Does the model already learn it? (the decider — val split).** Predicted eff RMS|z|
**1.13 vs truth 1.11 kpc** (mean); MAE 0.156 kpc. Error vs log₁₀ M*: **slope +0.004 kpc/dex,
Pearson r=+0.01** — flat. Low-mass-half error +0.018 vs high-mass-half +0.034 kpc (indistinguishable).
The model has **no unlearned mass-dependent thickness bias** for a mass baseline to fix. (The
prediction error is instead *projection*-dependent — each galaxy's 198 projections form vertical
streaks in `model_error_vs_mass.png` — a separate robustness question, not addressed by hz(M*).)

**3. Variance ceiling.** Total thickness-scale spread only 0.091 dex; a perfect `hz(M*)` removes
the R²=0.32 of it, leaving 0.076 dex (radius/bar/environment-driven, not mass-settable), before
profile *shape* variance is even considered.

## Decision

Skip the mass-dependent-baseline retrain. Reasoning: (a) weak relation (R²=0.32), (b) model
already unbiased in mass, (c) both real test galaxies (NGC 4321 6×10¹⁰, NGC 4371 3.5×10¹⁰) sit
*inside* the TNG mass range 3×10⁹–1×10¹², so even the OOD-prior rationale does not apply.

Idea 2 is closed. ~~If vertical-structure accuracy is revisited later, the diagnostic points at
**projection robustness** (thickness error varies by viewing angle, not mass) as the larger lever,
not a mass baseline.~~ **Correction (same day):** the follow-up
`2026-07-01-projection-uncertainty.md` shows the error is 99 % **per-galaxy bias**, ~invariant to
viewing angle — the "streaks" were galaxy offsets, not projection scatter. The lever is a
galaxy-level uncertainty head, not projection robustness.

## Files

`scripts/diagnose_mass_baseline.py`; plots in `outputs/diag_mass_baseline/` (`mass_thickness.png`,
`model_error_vs_mass.png`, `variance_split.png`); design spec
`docs/superpowers/specs/2026-07-01-mass-baseline-diagnostic-design.md`.
