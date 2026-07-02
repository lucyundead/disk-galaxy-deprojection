# Projection-spread uncertainty proxy: tested and rejected — error is galaxy-driven

**Date:** 2026-07-01
**Verdict:** the cheap retrain-free uncertainty proxies both **fail**, for an interesting reason:
the model's thickness prediction is nearly **viewing-angle-invariant** (a robustness win), and its
error is ~entirely **per-galaxy bias** the image features don't capture. The honest deployable
error bar is a population-calibrated **constant**: eff RMS|z| **± 0.17 kpc (68 %) ≈ ±19 %**
(± 0.31 kpc at 90 %). Doing better than a constant requires a galaxy-level uncertainty head
(MDN predictive variance — retrain), not more projections.
**Branch:** `codex/milestone2-tng-ingestion-design`.

## Motivation

Idea 3 identified per-galaxy uncertainty as our one real gap vs Ding's AICp; the idea-2 diagnostic
plot appeared to show projection-driven error ("vertical streaks"). The cheap proxy candidate:
use the across-projection spread of the prediction as an empirical error bar, and/or condition the
TNG-calibrated error on inclination (which we always know at inference).

## Diagnostic

`scripts/diagnose_projection_uncertainty.py` (read-only, cluster, memmap; reuses the
idea-2 helpers). Val split = 37 galaxies × 198 projections (inclinations 10°–60° in 5° steps).
Per row: predicted vs truth eff RMS|z| (mass-weighted, R<12 kpc) from the R=64 bundle.

## Results

**A. Variance decomposition.** Within-galaxy (viewing-angle-driven) share of error variance:
**1 %**. Between-galaxy (per-galaxy bias): **99 %**. Median across-projection spread of the
prediction is **0.015 kpc**, vs median per-galaxy |bias| **0.125 kpc**. The idea-2 "streaks" were
tight per-galaxy clusters *offset* by galaxy bias, not projection scatter — **the idea-2
side-finding ("projection-dependent") is corrected to "galaxy-dependent"**. Flip side: the
prediction is impressively projection-invariant — a single image at any angle gives essentially
the model's answer for that galaxy.

**B. Spread informativeness.** Spearman(spread, MAE) across galaxies = **+0.00**; treating the
spread as 1σ covers only **10 %** of errors (target ~68 %). An ensemble-over-viewing-angles
uncertainty would be ~10× overconfident. Rejected.

**C. Inclination-conditioned calibration.** |error| P68 is flat from i=10° (0.161 kpc) to i=60°
(0.170 kpc); Spearman(|err|, incl) = +0.00, (|err|, bar angle) = −0.02. No usable viewing-angle
signal — the deployable error bar reduces to the population constant:

| quantile | absolute [kpc] | fractional |
|---|---|---|
| P50 | 0.126 | ~14 % |
| **P68** | **0.166** | **~19 %** |
| P90 | 0.314 | ~35 % |

(bias +0.026 kpc, negligible.) Consistent with the NGC 4371 external check, where the learned
thickness matched the independent MGE to ~15–20 %.

## Implications

1. **Report the constant band.** A one-line, TNG-val-calibrated accuracy statement for `dgdp`
   outputs: *eff RMS|z| accurate to ±19 % (68 %), ±35 % (90 %), viewing-angle-independent* —
   trivially wireable into `deproject()` docs/result if wanted (no retrain).
2. **The next real lever is a galaxy-level uncertainty head.** Since error is per-galaxy bias,
   only a predicted variance (e.g. restoring the MDN's variance output instead of collapsing to
   the mean head — retrain + bundle format change) can give per-galaxy error bars. That is now the
   *only* identified path to beat the constant band; viewing-angle ensembles cannot.
3. **Projection-invariance is a positive result** worth stating alongside Ding comparisons: their
   per-galaxy exploration depends on the given viewing angle; our learned prior returns the same
   density from any angle of the same galaxy (spread ~1 % of the estimate).

## Files

`scripts/diagnose_projection_uncertainty.py`; plots `outputs/diag_projection_uncertainty/`
(`spread_vs_error.png`, `error_vs_inclination.png`). Corrections noted in
`2026-07-01-mass-baseline-diagnostic.md` and `2026-07-01-ding-aicp-comparison.md`.
