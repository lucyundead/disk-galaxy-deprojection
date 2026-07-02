# Idea 3: learning from Ding et al. (non-parametric + AICp) vs our learned TNG prior

**Date:** 2026-07-01
**Source:** poster "Deprojection of Bar Galaxies", Huangshengzhi Ding (hding@mpe.mpg.de),
Jens Thomas, Roberto Saglia (MPE), Astro Lipari 2026.
**Branch:** `codex/milestone2-tng-ingestion-design`.

## What Ding et al. do

Non-parametric deprojection + AICp, validated on an N-body barred galaxy (known 3D truth),
fixed viewing angle (θ,φ,ψ)=(75°,45°,90°):

- **Non-parametric**: 3D density on a flexible grid, not a parametric model (MGE/GALFIT). The
  deprojection is degenerate (many 3D densities project to the same image), so they *explore the
  whole space of densities consistent with the observation* rather than committing to one form.
- **AICp regularization**: degeneracy + noise needs a smoothing strength α. The χ²(α) L-curve +
  AICp criterion pick α where χ² ≈ N_datapoints (fit signal, not noise; chosen log₁₀α≈−2.18). At
  that α the recovered bar pattern speed Ω_p lands on the N-body truth.
- **Accuracy**: RMS log₁₀-density error **0.23 (bar) / 0.15 (outer)**. Downstream, the density
  feeds Schwarzschild modelling (Tikhonenko et al., MNRAS submitted 2026), recovering Ω_p.

## Ours vs theirs — philosophically opposite

| axis | dgdp (ours) | Ding |
|---|---|---|
| resolves degeneracy by | learned TNG population prior (q_m) | exploring consistent-density space + AICp |
| prior | strong (TNG barred galaxies) | none (data + smoothness) |
| uncertainty | point estimate (MDN collapsed to mean) | family of consistent densities → uncertainty-aware |
| speed | ms (torch-free numpy) | per-galaxy optimisation (slow) |
| in-plane bar | = the image, exact (full m=2) | fit (radial scatter on the poster) |
| OOD risk | prior can bias | none, but noise-limited where data are thin |
| validation | TNG truth (in-dist) + real galaxies vs MGE | N-body truth |
| downstream | density cube + optional AGAMA potential | → Schwarzschild → Ω_p ✓ |

## Quantitative benchmark: our density accuracy in Ding's metric

`scripts/benchmark_density_accuracy.py` rebuilds our model density for every TNG val projection
(7326 rows) exactly as the pipeline does — image features → predicted q_m weights, in-plane anchor
Σ_m from the geometric baseline, `reconstruct_density` — and compares to the TNG truth in log₁₀
density per radial region (each row renormalised to the truth total mass first, to compare the
distribution not the normalisation).

| region | ours, mass-weighted | ours, unweighted (>1e-4·max) | Ding |
|---|---|---|---|
| bar (r<8 kpc) | **0.215** | 0.431 | 0.23 |
| outer (8–22 kpc) | **0.088** | 0.117 | 0.15 |

On the physically-meaningful **mass-weighted** metric (RMS log-density where the mass actually is),
our learned-prior deprojection is **comparable-to-better than the non-parametric+AICp method** —
at ms cost vs per-galaxy optimisation, with an exact in-plane bar. The unweighted-above-floor
number is worse in the bar (0.43), showing our faint/low-density and vertical-tail cells are less
accurate (consistent with the R>12 kpc edge weakness seen in the idea-1 NGC audit). Sample radial
profile: `outputs/diag_mass_baseline/density_accuracy_profile.png` (model tracks truth to ~0-22 kpc,
slight over-estimate beyond ~17 kpc).

**Caveats (indicative, not a strict head-to-head):** different simulation (TNG vs their N-body),
different galaxy and viewing angle, and per-cell (ours) vs radial-profile (likely theirs) metric.

## Takeaways

1. **The learned prior "buys" competitive density accuracy for ~free.** We match their density
   recovery where the mass is, at inference speed orders of magnitude lower, with the bar exact.
   The cost is what they have and we don't: (a) per-galaxy **uncertainty** and (b) freedom from a
   population prior (OOD robustness). For in-distribution barred galaxies the prior is a clear win.
2. **Uncertainty is our real gap.** Their AICp explores the consistent-density family; we emit one
   number. The dgdp-native analogs: the MDN's predictive variance (currently collapsed to the mean
   head — needs a retrain to expose) or, cheaply, the **projection spread** (idea-2 side-finding:
   our q_m prediction varies with viewing angle) as an empirical uncertainty proxy. Deferred.
3. **Shared downstream.** Our density cube + optional AGAMA CylSpline potential can feed the same
   Schwarzschild → pattern-speed pipeline (Tikhonenko et al.) they use — a natural benchmark /
   contact point with the MPE group (who also overlap with the NGC 4371 MGE work).

## Conclusion

Idea 3 done: the learned TNG prior is quantitatively competitive with Ding's non-parametric+AICp
on density recovery, trading their per-galaxy uncertainty for speed and an exact bar. The most
valuable next step (if pursued) is **uncertainty quantification** on our point estimate — either
the projection-spread proxy or exposing the MDN predictive variance — which is also the answer to
the idea-2 projection-robustness side-finding.

## Files

`scripts/benchmark_density_accuracy.py`; `outputs/diag_mass_baseline/density_accuracy_profile.png`.
