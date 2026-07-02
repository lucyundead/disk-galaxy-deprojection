# Design: mass-dependent-baseline diagnostic (idea 2, phase 0)

**Date:** 2026-07-01
**Status:** approved design; diagnostic only (no retrain, no architecture change).

## Problem

Idea 2 (handoff) proposed a mass-dependent geometric-baseline scale height `hz(M*)` (Gadotti
relation) instead of the fixed `hz=0.3` kpc, motivated by "making the training residual clearer
per galaxy so the MDN learns the essential vertical structure."

**Reading the code invalidates the premise.** The current pipeline does not learn a residual
against the geometric baseline:

- Training target = the TNG truth's *normalized* vertical profile `q_m(z;R)=a_rk/∫a_rk dz`
  (`vertical_mixture.weights_target`), fit over a fixed 7-height sech² dictionary, then PCA
  with **mean subtraction** (`deproject_fourier_rz_compare.fit_pca`).
- Inference = `pca_mean + MDN·deviation`, anchored by Σ_m(R) (`model.predict_weights`).
- The geometric baseline enters only as `baseline_grid_mass_msun` (z-integrated total mass →
  hz-invariant) in the features, and as the z-integrated anchor at inference (hz cancels).

So `hz` is inert in both training and inference. The model already learns deviations — but from
the **PCA population-mean profile** (one fixed average for all galaxies, anchored at TNG-sample
thickness), not from a physical mass-dependent baseline.

The real, implementable version of idea 2 is: replace/augment that implicit population-mean
baseline with an explicit **mass-dependent physical baseline**, so the MDN's target is a per-galaxy
residual (shape at mass-appropriate thickness). Likely payoff is OOD low-mass real galaxies
(NGC 4321 is ~1 dex below the TNG sample). This needs a target/reconstruction change **and a
cluster retrain**, with uncertain gain over the existing PCA-mean centering.

## Decision to make (cheaply, before any retrain)

Does an explicit mass-dependent baseline actually help, or does the model already learn the
mass→thickness relation? Answer with a read-only cluster diagnostic (no training).

## Diagnostic

`scripts/diagnose_mass_baseline.py` — pure numpy, runs on the cluster against the existing
`outputs/tng50_milestone2d_rich/density_residual_table.npz` + the committed R=64 bundle
`src/dgdp/models/dgdp_fixed_dict.npz`. Shipped/run via `cluster_dgdp.py sync` + `run`; small
report + PNGs fetched back. Per-galaxy M* = `baseline_grid_mass_msun`; galaxies deduped from
projections by a subhalo id if present, else by exact M* value.

Three measurements:

1. **Mass→thickness relation in TNG truth.** Per-galaxy effective RMS|z| (mass-weighted, R<12
   kpc) vs M* (all galaxies). Pearson/Spearman in log space, power-law fit `hz(M*)`, scatter (dex).
   *Weak relation ⇒ idea dies here.*

2. **Model's residual mass bias (the decider).** On the **val** split, per row: predicted RMS|z|(R)
   from the bundle's m=0 prediction (RMS|z| depends only on m=0, so no full 3D reconstruct) vs
   truth RMS|z|(R). Test whether the error `pred_eff − truth_eff` correlates with M* (slope +
   correlation). Flat ⇒ model already learns the relation, baseline won't help. Sloped ⇒ unlearned
   mass-dependent thickness bias a baseline would fix.

3. **Variance ceiling (bonus).** Fraction of truth vertical-profile variance that is overall scale
   (mass-settable) vs shape at fixed scale (not) — upper bound on what any mass-baseline could
   remove. Also: target-variance reduction from subtracting `sech²(hz(M*))` vs the PCA mean.

**Outputs:** `outputs/diag_mass_baseline/` — `report.txt` (the numbers + a verdict line) and
`mass_thickness.png`, `model_error_vs_mass.png`, `variance_split.png`.

**Decision rule:** strong relation (1) AND residual mass-bias in the model (2) → mass-baseline is
worth a cluster retrain (idea 2 proper: target = truth − sech²(hz(M*,R)), reconstruct = baseline +
deviation). Otherwise → skip the retrain; at most use `hz(M*)` as an OOD prior for low-mass real
galaxies.

## Scope / non-goals

No training, no architecture change, no bundle change in this phase. The retrain (if the diagnostic
says go) is a separate spec + cluster job.
