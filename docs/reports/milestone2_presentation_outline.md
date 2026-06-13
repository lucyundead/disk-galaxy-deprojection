# Presentation Outline: 3D Mass Distributions of Barred Galaxies from Single Images

Audience: general astronomers (explain MDN / PCA / flow matching from scratch).
All figures are committed under
`outputs/tng50_milestone2c_clean3d/presentation_figures/`. Numbers are from the
verified Milestone 2c / 2d artifacts (see `2026-06-10-project-handoff.md`).
"TO MAKE" = a diagram best drawn in slide software.

## Method one-liners (say these out loud)

- **PCA**: "describe every 3D error field as a mix of ~32 standard shape
  patterns" (like eigen-spectra). Compresses ~49,000 grid cells to 32 numbers.
- **MDN** (Mixture Density Network): "a neural net whose output is a probability
  distribution, not a single number" — gives mean + spread per coefficient.
- **Coverage**: "we check error bars honestly: 68% intervals must contain the
  truth 68% of the time on galaxies the model never saw."
- **Flow matching**: "the tech behind image generators, repurposed for
  posteriors — powerful, but our diagnostics show we don't need it yet."

## Slides

1. **Title / pitch** — "One image at i<=60 deg in, calibrated 3D stellar mass
   distribution out." Fig: real barred galaxy + 3D render (stock).
2. **Why** — 3D mass -> gravitational potential -> bar torques -> gas inflow.
   End goal: S4G galaxies -> potentials -> hydro sims. Fig: pipeline arrows (TO MAKE).
3. **Why hard** — one projection, many 3D solutions; thin-disk + constant M/L
   fails for thick, boxy/peanut bars. Fig: edge-on peanut vs face-on (stock/TNG).
4. **Idea** — simulations give image + 3D truth -> supervised inverse. TNG50:
   185 barred galaxies, 9 projections, galaxy-level split (111/37/37). Fig:
   `outputs/tng50_milestone2/.../montage.png` (face-on gallery).
5. **Architecture** — physics-first: geometric baseline + ML predicts only the
   *residual*. Data-efficient by design. Fig: `example_reconstruction_554189.png`.
6. **PCA** — 32 shape patterns capture 84% of residual variance; log-R grid is
   bar-centric (23/32 radial bins inside 5 kpc). Fig: `pca_component_gallery.png`.
7. **MDN** — returns a posterior, not a point; 1 Gaussian beats 3/5 mixtures at
   this sample size. Fig: schematic (TO MAKE).
8. **Honest uncertainties** — correction heads + calibration; 4/6 physical
   summaries calibrated on held-out galaxies. Fig: per-summary coverage (TO MAKE
   from diagnostics, or a table).
9. **Headline result** — cell-mass MAE 66% better than baseline; total mass to
   ~1%; radial bias removed. Fig: `recovery_grid_554189_i40_bar0.png` (the 3x3:
   inclined image | truth face-on/edge-on | baseline | MDN).
10. **Bar region is best-recovered** — cell rel MAE 26% (bar) < 32% < 40% (outer);
    m=2 amplitude to ~1.5%, phase to 3.8 deg. Figs: `bar_region_mae.png`,
    `peanut_ratio_scatter.png`.
11. **Vertical structure** — thin-disk baseline is 0.42x the true bar thickness;
    MDN restores it to 1.04x and tracks per-galaxy anisotropy (r=0.89). Fig:
    `peanut_diagnostic_554189.png`.
12. **Stress tests** — PCA-64 rejected (data-limited, not representation-limited);
    inclination: random scatter benign, systematic +offset damaging (asymmetric).
    Fig: `inclination_sensitivity.png`.
13. **Boxy vs X (resolution)** — production 0.625 kpc grid smears the peanut X;
    re-binning to 0.31 kpc resolves it. Fig: `zrebin_peanut_392276.png`.
14. **Can we recover the X? (the honest test)** — held-out 392276: grid resolves
    the X, PCA basis can represent it (dip 0.595), but the MDN does not recover it
    (dip 1.000) because it is the only strong X in 185 galaxies. Recovery is
    gated by training data, not architecture. Fig: `mdn_x_recovery_392276.png`.
15. **Flow matching?** — diagnosed: posterior spread already calibrated; failures
    are biases, not distribution shape. Defer until geometry-marginalization makes
    the posterior genuinely non-Gaussian. (Text slide.)
16. **Roadmap** — bar-share + vertical heteroscedastic heads; inclination-
    uncertainty propagation; **N-body buckled bars** (to learn + validate
    X-recovery, since TNG50 is peanut-poor); then S4G. Fig: timeline (TO MAKE).
17. **Take-home** — single image -> calibrated 3D mass posterior, 66% over
    classical; bar (the dynamically interesting part) recovered best; method is
    X-*capable*, X-recovery awaits richer training data. Physics-first + minimal
    ML beats ML-first at astronomical sample sizes.

## Figures still TO MAKE (diagram-type, in slide software)

- pipeline arrows (slide 2), edge-on peanut stock image (slide 3),
  MDN schematic (slide 7), per-summary coverage graphic (slide 8),
  roadmap timeline (slide 16).

## Key numbers cheat-sheet

- sample 185 galaxies / 1665 rows, split 111/37/37 galaxies.
- cell-mass MAE 5.82e5 Msun, 66.2% better than baseline.
- bar region: cell rel MAE 0.26; m=2 amp err 0.004 on 0.267; m=2 phase 3.8 deg.
- bar thickness: baseline 0.42x truth, MDN 1.04x; anisotropy r=0.89.
- PCA 32 comps: 83% EVR at 0.625 kpc, 84% at 0.31 kpc; PCA-64 rejected.
- X galaxy 392276: truth dip 0.795, basis 0.595, MDN 1.000 (held out).
