# Scope: replace the free-knot q_m with a positive sech² mixture (training parametrization)

**Date:** 2026-06-25
**Why:** the 2026-06-25 pre-check (`scripts/test_mixture_vs_freeknot.py`) showed a small
midplane-centered positive sech² mixture represents the milestone-2d vertical profiles
*better* than the current free-knot q_m (RMS|z| frac-err median 5% for 3 components vs 18%
free-knot; rel-L2 0.015 vs 0.087) **and** gives z-symmetry + positivity + ∫q dz=1 *by
construction* — retiring the post-hoc reconstruction patches (anchor renorm, z-symmetrize,
taper) we needed on NGC 4371. This scopes wiring it into training. **Steps 1–2 (module +
de-risk) DONE 2026-06-25 — the per-galaxy-scaled K=4 design is validated (see "De-risk
result" below); steps 3–7 (training swap + retrain) not done.** Confirmed decisions:
(1) per-galaxy-scaled heights; (2) positive weights + signed fallback for B/P (m=2);
(3) keep PCA + clip/renorm; (4) new mixture config (frozen allocation untouched).

## De-risk result (steps 1–2)

`src/dgdp/vertical_mixture.py` (kernels + per-galaxy scale + NNLS/lstsq weights +
reconstruct, with a `__main__` self-check) and `scripts/test_mixture_pergalaxy.py`. Per
galaxy s = mass-weighted mean disk RMS|z| / 1.814; heights = s·{0.3,0.7,1.5,3.0} (K=4):
- **m=0:** per-galaxy K=4 rel-L2 **0.013** / RMS|z| err **4.6%** (p90 15%) vs free-knot
  0.087 / 18% (p90 30%) — K=4 matches the dense 10-height dictionary (0.014) with only 4
  components, so the predicted weight vector is small. K=3 is 0.028 / 10%.
- **m=2:** positive-only K=4 median 0.050 but p90 0.188 (> free-knot 0.154); the **signed
  fallback** recovers p90 to 0.139 → confirms positive default + signed for B/P bars.
  (RMS|z| is undefined for a signed m=2 modulation — only rel-L2 applies there.)

## Design

Factorize ρ(R,φ,z) = Σ_image(R,φ) · g(z; R,φ); the network predicts only the vertical PDF g.

- **Fixed height dictionary** H = {h_1..h_K} kpc (K≈6–8), spanning thin→halo (~0.2–3.5).
  Normalised kernels φ_k(z) = sech²(z/2h_k) / ∫sech²(z/2h_k)dz, so ∫φ_k=1.
- **g per (R, mode) = Σ_k w_{m,k}(R) φ_k(z)**, weights w≥0, Σ_k w=1 → ∫g=1, symmetric,
  positive, unimodal-by-superposition — all structural.
- Heights are a **fixed hyperparameter**, not predicted (the pre-check's "sech2 full" =
  fixed dictionary + per-profile NNLS weights already hit rel-L2 0.014 / RMS 0.022). Flaring
  comes from w_k(R) shifting weight to larger h_k outward — the thin+thick+halo picture.

What the network predicts changes from a free z-knot vector to the **weights w_{m,k}(R)**;
everything else (features, PCA, MDN) is unchanged.

## Exact touch points

| file | block | change |
|---|---|---|
| **new** `src/dgdp/vertical_mixture.py` | — | kernels + basis, NNLS weight-target builder, mixture→density reconstruct; `__main__` self-check (assert ∫=1, symmetry, residual) |
| `scripts/deproject_fourier_rz_conserving.py` | target build (l.163–176) | replace `sep_resample`→`q=a/sig` with `vertical_mixture.weights_target(a_true[m], …)` |
| (same) | `reconstruct_smooth` (l.184–216) | replace knot→grid `q` with `vertical_mixture.reconstruct(weights, anchor, …)` |
| `scripts/deproject_real_image_ngc4321_learned.py` | target build (l.~225–236) + `reconstruct_smooth` (l.70–104) + `predict_learned` | same swaps via the shared module; **delete** the post-hoc anchor-renorm / z-symmetrise / taper (now by construction) |
| `fit_pca`/`train_mdn` (`deproject_fourier_rz_compare.py`) | — | unchanged (still PCA→MDN on the new weight target) |

The weights target is non-negative & smooth; keep **PCA→MDN as-is** and at reconstruction
**clip≥0 + renormalise per (R,mode)** → exact valid simplex (a principled projection, not a
drift-fix). z-symmetry/positivity hold regardless because the kernels are centred & positive.

## Decisions to confirm before coding

1. **K and the heights** — recommend K≈6 fixed (geomspace 0.2–3.5 kpc). Alternative: per-galaxy
   scale the heights by the galaxy's disk RMS|z| (more adaptive, slightly more complex). Start fixed.
2. **m>0 weights** — assume the bar/spiral vertical shapes are positive (simplex, like m=0;
   the pre-check's |a₂| fit was good). Keep a fallback to signed coeffs if the m=2 residual
   with positive-only weights is poor.
3. **Keep PCA** (clip+renorm at reconstruct) vs **direct softmax-simplex MDN output** — start
   with PCA (minimal change); switch to a simplex output only if clip+renorm proves lossy.
4. **Allocation** — the free-knot `outputs/nbody_shen2010/fourier_rz_allocation.json` is
   FROZEN; the mixture needs its own sizing (modes × R-knots × K) → write a **new** config,
   do not touch the frozen file.

## Step plan (each independently checkable)

1. DONE — `src/dgdp/vertical_mixture.py` + self-check.
2. DONE — per-galaxy K-de-risk (`scripts/test_mixture_pergalaxy.py`): K=4 validated (see above).
3. Swap target-build + `reconstruct_smooth` in `deproject_fourier_rz_conserving.py`; new config.
4. Re-run conserving training; compare recovery metrics (cellMAE, relL2, massAcc) to the
   free-knot baseline — expect match or improvement.
5. Mirror in `deproject_real_image_ngc4321_learned.py`; delete the post-hoc fixes.
6. Re-run NGC 4321 + NGC 4371 → confirm v_c & RMS|z| still match the MGE/baseline with **no**
   post-hoc patches and no artifacts.
7. (Separate, larger) richer training data — more inclinations (incl. >60°) and bar angles —
   is the data-side companion; it needs regenerating the milestone-2d truth table and is out
   of scope for this parametrization change.

## Risks & mitigations
- **Height coverage** — if some galaxies need h outside H, residual grows → per-galaxy height
  scaling (decision 1). De-risked by step 2.
- **m>0 sign** — positive-only may under-fit a B/P bar → signed fallback (decision 2).
- **PCA-then-clip lossy** — if the simplex projection distorts → direct softmax output (decision 3).
- **Retrain cost** — one conserving-training run; modest.

## Do NOT touch
`outputs/nbody_shen2010/fourier_rz_allocation.json` (frozen). Commit only when asked.
