# Disk Galaxy Deprojection Project Handoff

Date: 2026-06-11

Workspace: `/home/lucyundead/projects/disk-galaxy-deprojection`

Branch: `codex/milestone2-tng-ingestion-design`

This handoff is for another AI/tool to review and continue the project without
needing the previous chat context.

## Scientific Background And Motivation

The project is building a probabilistic deprojection workflow for barred disk
galaxies. The long-term target is:

```text
p(rho\\\_star\\\_3d | S4G-like image, inclination, PA, bar angle, metadata)
```

The first science target is stellar mass structure, not dark matter or total
dynamical mass. The observational analogue is S4G-like near-infrared imaging,
where old stellar light is used as a proxy for stellar mass after appropriate
preprocessing.

The core scientific difficulty is that a 2D image does not uniquely determine a
3D stellar density field. The project therefore treats deprojection as posterior
inference, not as a single deterministic inversion. TNG50 is used as the
simulation testbed because the true 3D stellar particle distribution is known.

The current strategy is intentionally conservative:

1. Build a classical geometric deprojection baseline.
2. Learn residual corrections for low-dimensional physical summaries.
3. Only then move to coarse 3D residuals around the baseline.
4. Do not jump to a large full-3D generator until summary-level calibration and
3D residual diagnostics are credible.

## Project Plan

The project follows a residual ladder.

### Milestone 1: Synthetic Summary Residuals

Goal:

```text
p(delta\\\_summaries | image\\\_mock, baseline\\\_summaries, geometry, metadata)
```

This phase used toy barred-galaxy mocks to validate the pipeline shape:

* generate mock images;
* compute true physical summaries;
* compute geometric baseline summaries;
* train a small MDN residual model;
* evaluate posterior mean and 68 percent coverage.

Primary files:

* `configs/milestone1.synthetic.toml`
* `scripts/build\\\_synthetic\\\_benchmark.py`
* `scripts/train\\\_summary\\\_residual\\\_mdn.py`
* `scripts/evaluate\\\_summary\\\_residual.py`
* `docs/reports/milestone1\\\_synthetic\\\_summary.md`

### Milestone 2: TNG50 Ingestion And Summary Benchmark

Goal: replace synthetic inputs with TNG50 barred-galaxy products while keeping
the first target low-dimensional.

The current TNG50 sample is:

* TNG50-1 snapshot `99`;
* central galaxies only;
* stellar mass `>= 10^9.5 Msun`;
* barred according to the TNG morphology/bar catalogue;
* primary bar strength `>= 0.2`;
* primary bar size `>= 2.0 kpc`;
* top 64 by stellar mass;
* train/validation/test split by galaxy: `38/13/13`.

The projection grid has 576 rows:

* 64 galaxies;
* 9 projections per galaxy;
* inclinations `20`, `40`, `60` degrees;
* face-on bar viewing angles `0`, `45`, `90` degrees;
* disk PA fixed at `0` degrees for the first controlled benchmark.

Primary files:

* `docs/reports/milestone2\\\_tng50\\\_ingestion.md`
* `outputs/tng50\\\_milestone2/manifest.csv`
* `outputs/tng50\\\_milestone2/residual\\\_table.npz`
* `outputs/tng50\\\_milestone2/metrics.json`
* `outputs/tng50\\\_milestone2/diagnostics/`

Milestone 2 summary benchmark status:

* projection-grid corrected summary MAE improved strongly over the geometric
baseline;
* diagnostic rerun with 256 posterior samples gave 68 percent coverage of about
`0.757`;
* weak spots remain vertical scale height, inner enclosed mass, central mass
fraction, and outer low-density annuli.

### Milestone 2b: Coarse 3D Baseline-Plus-Residual

Goal:

```text
p(delta\\\_rho\\\_3d | image\\\_mock, rho\\\_baseline\\\_3d, geometry, metadata)
```

The target is not yet an unconstrained full 3D generator. It is a coarse
cylindrical residual around a geometric baseline:

```text
delta\\\_rho\\\_3d = rho\\\_true\\\_cylindrical - rho\\\_baseline\\\_cylindrical
```

Current grid definition:

* frame: disk plane aligned to `z = 0`, bar major axis aligned to intrinsic `x`;
* shape `(n\\\_R, n\\\_phi, n\\\_z) = (32, 48, 32)`;
* radial edges: explicit `R = 0`, then logarithmic `0.05` to `30.0 kpc`;
* azimuth edges: `-pi` to `pi`;
* vertical edges: `-10.0` to `10.0 kpc`;
* density unit: `Msun/kpc^3`;
* mass unit: `Msun`.

Milestone 2b uses clean all-particle images:

* config: `configs/milestone2b.clean3d.toml`;
* image size: `192 x 192`;
* pixel scale: `0.35 kpc/pixel`;
* field of view: `67.2 kpc`;
* no PSF;
* no noise;
* source particles: all formed stellar particles from TNG chunks.

Primary artifacts:

* `outputs/tng50\\\_milestone2b/residual\\\_table.npz`
* `outputs/tng50\\\_milestone2b/density\\\_residual\\\_table.npz`
* `outputs/tng50\\\_milestone2b/density\\\_residual\\\_diagnostics/`
* `outputs/tng50\\\_milestone2b/baseline\\\_density\\\_grids\\\_logr\\\_cyl/`
* `outputs/tng50\\\_milestone2b/density\\\_grids\\\_logr\\\_cyl/`

## Current Implementation Structure

Core package code lives under `src/dgdp/`.

Important modules:

* `src/dgdp/tng50\\\_catalog.py`: TNG50 group-catalog reader.
* `src/dgdp/bar\\\_catalog.py`: bar-catalog inspection and join helpers.
* `src/dgdp/tng50.py`: offset-based stellar particle extraction.
* `src/dgdp/density3d.py`: cylindrical density grids, baseline helpers, and
projection diagnostics.
* `src/dgdp/summaries.py`: physical summary definitions.
* `src/dgdp/mdn.py`: mixture-density network utilities.

Important scripts:

* `scripts/cluster\\\_dgdp.py`: remote cluster wrapper entrypoint.
* `scripts/build\\\_tng50\\\_manifest.py`: remote TNG50 sample manifest builder.
* `scripts/extract\\\_tng50\\\_particles.py`: compact particle extraction.
* `scripts/build\\\_tng50\\\_benchmark.py`: TNG50 summary residual table.
* `scripts/build\\\_tng50\\\_all\\\_particle\\\_images.py`: clean all-particle images.
* `scripts/build\\\_tng50\\\_density\\\_grid.py`: true cylindrical 3D density grids.
* `scripts/build\\\_tng50\\\_baseline\\\_density\\\_grid.py`: geometric baseline 3D grids.
* `scripts/build\\\_tng50\\\_density\\\_residual\\\_table.py`: 3D residual table.
* `scripts/analyze\\\_tng50\\\_density\\\_residuals.py`: residual diagnostics and PCA.
* `scripts/train\\\_density\\\_residual\\\_pca.py`: deterministic PCA residual model.
* `scripts/train\\\_density\\\_residual\\\_pca\\\_mdn.py`: probabilistic PCA residual MDN.
* `scripts/sweep\\\_density\\\_residual\\\_pca\\\_mdn.py`: MDN seed/mixture sweep.
* `scripts/report\\\_milestone2b\\\_physical\\\_summaries.py`: physical-summary report
from 3D density predictions.
* `scripts/calibrate\\\_milestone2b\\\_physical\\\_summaries.py`: post-hoc physical
summary calibration from selected MDN posterior samples.
* `scripts/diagnose\\\_milestone2b\\\_summary\\\_coverage.py`: galaxy-bootstrap coverage
confidence intervals, bias/spread decomposition, and total-mass constraint
diagnostics for the physical summaries.
* `scripts/train\\\_total\\\_mass\\\_correction.py`: scalar total-mass correction head
that predicts `log(M\\\_true / M\\\_baseline)` per row with uncertainty.

## Cluster And Data Rules

Do not download full TNG50 data locally.

The TNG50 snapshots and heavy raw products stay on the remote cluster. Use the
existing cluster wrapper only when remote work is actually needed:

```bash
python scripts/cluster\\\_dgdp.py sync
python scripts/cluster\\\_dgdp.py check-env
python scripts/cluster\\\_dgdp.py run-tng50
python scripts/cluster\\\_dgdp.py run-tng50-density
python scripts/cluster\\\_dgdp.py fetch-tng50
python scripts/cluster\\\_dgdp.py run-tng50-milestone2b-clean
python scripts/cluster\\\_dgdp.py fetch-tng50-milestone2b
```

For current Milestone 2b modeling/calibration, remote work should usually not be
needed because the fetched local artifacts under `outputs/tng50\\\_milestone2b/`
are sufficient.

## Completed Recent Work

### Smooth Baseline Repair

The earlier 3D baseline deposited one pseudo-particle per image pixel into the
cylindrical grid. This produced artificial empty inner `R-phi` cells in some
held-out diagnostic panels.

The baseline was repaired to:

1. deproject the image into disk-plane surface density `Sigma(R, phi)`;
2. integrate surface density over each cylindrical `R-phi` cell area;
3. distribute mass vertically with normalized finite-bin `sech^2(z / h)`
weights.

Primary file:

* `scripts/build\\\_tng50\\\_baseline\\\_density\\\_grid.py`

The refreshed deterministic report is:

* `docs/reports/milestone2b\\\_density\\\_residual\\\_pca.md`

Key held-out results after repair:

* geometric baseline cell-mass MAE: `3.986e6 Msun`;
* deterministic PCA corrected cell-mass MAE: `1.150e6 Msun`;
* improvement over baseline: `71.14%`;
* total-mass fractional MAE stays `0.0453` by construction because the
residual correction preserves baseline grid mass.

### Probabilistic PCA Residual MDN

Report:

* `docs/reports/milestone2b\\\_density\\\_residual\\\_pca\\\_mdn.md`

Key held-out results:

* MDN posterior-mean cell-mass MAE: `1.099e6 Msun`;
* improvement over baseline: `72.44%`;
* PCA coefficient 68 percent coverage: `0.752`;
* total-mass fractional MAE still `0.0453` by construction.

Interpretation:

* the MDN is a useful uncertainty baseline and a slightly better point predictor
than the deterministic PCA MLP;
* uncertainty was initially somewhat conservative at coefficient level;
* larger models were deferred until calibration was checked.

### MDN Seed/Mixture Calibration Sweep

Report:

* `docs/reports/milestone2b\\\_density\\\_residual\\\_pca\\\_mdn\\\_sweep.md`

The sweep crossed:

* seeds `20260608`, `20260609`, `20260610`;
* mixture counts `1`, `3`, `5`;
* 200 epochs per run;
* local CPU execution from fetched Milestone 2b artifacts.

Key results across 9 runs:

* posterior-mean cell-mass MAE range: `1.042e6` to `1.187e6 Msun`;
* raw PCA coefficient coverage range: `0.703` to `0.807`;
* validation-fitted temperature scale range: `0.75` to `0.90`;
* temperature-scaled test coverage range: `0.658` to `0.694`.

Best default run:

* `outputs/tng50\\\_milestone2b/density\\\_residual\\\_pca\\\_mdn\\\_sweep/components\\\_1\\\_seed\\\_20260610/`
* posterior-mean cell-mass MAE: `1.042e6 Msun`;
* raw coefficient coverage: `0.768`;
* validation temperature scale: `0.75`;
* temperature-scaled test coefficient coverage: `0.658`.

Geometry-dependent uncertainty check:

* global temperature leaves `60 deg` projections undercovered;
* inclination-aware scaling improves the `60 deg` bin;
* for the 1-component family, inclination-aware coverage is balanced:
`20 deg: 0.669`, `40 deg: 0.677`, `60 deg: 0.687`.

Interpretation:

* MDN uncertainty is geometry-dependent, as expected;
* temperature calibration is enough for aggregate coefficient-level coverage;
* it is not enough to claim fully calibrated physical summaries.

### Physical Summary Calibration

Current report artifact:

* `outputs/tng50\\\_milestone2b/milestone2b\\\_physical\\\_summary\\\_calibration/milestone2b\\\_physical\\\_summary\\\_calibration.md`

Current metrics artifact:

* `outputs/tng50\\\_milestone2b/milestone2b\\\_physical\\\_summary\\\_calibration/milestone2b\\\_physical\\\_summary\\\_calibration\\\_metrics.json`

Script:

* `scripts/calibrate\\\_milestone2b\\\_physical\\\_summaries.py`

This script reuses the selected MDN posterior samples, keeps the posterior mean
fixed, and fits post-hoc temperature scales on validation physical summaries
before evaluating held-out test coverage.

Calibration modes tested:

1. coefficient temperature;
2. physical global temperature;
3. per-summary temperature;
4. per-summary plus inclination temperature.

Selected run:

* `components\\\_1\\\_seed\\\_20260610`;
* validation rows: `117`;
* test rows: `117`;
* target coverage: `0.68`;
* global physical temperature scale: `1.45`.

Test coverage by summary:

|Summary|Coeff. temp|Physical global|Per-summary|Per-summary + inclination|
|-|-:|-:|-:|-:|
|radial mass profile|0.482|0.658|0.611|0.605|
|vertical mass profile|0.593|0.749|0.800|0.802|
|vertical RMS height|0.684|0.855|0.872|0.863|
|central mass fraction|0.556|0.650|0.632|0.667|
|bar-axis mass fraction|0.684|0.778|0.735|0.829|
|bar-frame m=2 profile|0.528|0.729|0.687|0.685|

Interpretation:

* physical global temperature improves undercovered radial, central, and m=2
summaries but overcovers vertical and bar-axis summaries;
* per-summary and per-summary plus inclination calibration help selected
summaries, especially central mass fraction and m=2;
* radial profile remains undercovered;
* vertical structure becomes overcovered;
* post-hoc temperature calibration alone is not enough for a fully calibrated
physical-summary posterior.

### Summary Coverage Diagnostics

Script:

* `scripts/diagnose\\\_milestone2b\\\_summary\\\_coverage.py`

Artifacts:

* `outputs/tng50\\\_milestone2b/milestone2b\\\_summary\\\_coverage\\\_diagnostics/milestone2b\\\_summary\\\_coverage\\\_diagnostics.md`
* `outputs/tng50\\\_milestone2b/milestone2b\\\_summary\\\_coverage\\\_diagnostics/milestone2b\\\_summary\\\_coverage\\\_diagnostics\\\_metrics.json`

This diagnostic quantifies how much of the reported physical-summary
miscoverage is statistically meaningful given that held-out coverage is
estimated from only 13 test galaxies (117 correlated projection rows). It adds
galaxy-level cluster bootstrap confidence intervals on coverage, a per-summary
bias/spread decomposition (z-scores and PIT of the uncalibrated samples), and a
direct measurement of the total-mass constraint contribution.

Key results (per-summary temperature mode, 2000 bootstrap draws, seed
`20260610`):

* coverage 95 percent CIs contain the 0.68 target for 5 of 6 summaries,
including the radial profile (`0.611`, CI `[0.544, 0.687]`); the apparent
radial undercoverage is not statistically significant at the 13-galaxy sample
size;
* only the vertical mass profile is significantly miscovered (`0.800`, CI
`[0.682, 0.899]`), and its decomposition is bias-dominated (`mean z = -0.635`,
bias ratio `0.53`);
* several summaries carry statistically significant posterior-mean bias even
where coverage looks acceptable: radial profile (`mean z = -0.444`), vertical
profile (`-0.635`), central mass fraction (`+0.928`), bar-frame m=2 (`-0.374`);
wide intervals are hiding real bias;
* the baseline systematically overestimates total grid mass (mean fractional
offset `-0.030` to `-0.045` per split), and because the residual correction
preserves baseline total mass, this is an uncorrectable floor: it explains
about 77 percent of the observed radial-profile bias magnitude (correlation
`0.60` between constraint-implied and observed bias).

### Total-Mass Correction

Script:

* `scripts/train\\\_total\\\_mass\\\_correction.py`

Artifacts:

* `outputs/tng50\\\_milestone2b/total\\\_mass\\\_correction/` (model, normalization,
predictions, metrics)
* `outputs/tng50\\\_milestone2b/milestone2b\\\_summary\\\_coverage\\\_diagnostics\\\_mass\\\_corrected/`
* `outputs/tng50\\\_milestone2b/milestone2b\\\_physical\\\_summary\\\_calibration\\\_mass\\\_corrected/`

This is the structural fix recommended by the coverage diagnostics. A small
1-component MDN head (same features as the PCA residual MDN: pooled log image,
geometry metadata, log image/baseline masses) predicts
`log(M\\\_true / M\\\_baseline)` per row with uncertainty. The evaluation stack
(`calibrate`/`diagnose` scripts) accepts `--total-mass-predictions` and then
rescales every posterior sample grid to its own sampled target total mass, and
the posterior-mean grid to the mean predicted total mass, instead of pinning
all of them to the baseline total.

Key held-out results (seed `20260610`, 128 samples, matched to the selected
MDN run):

* test total-mass fractional MAE: `0.0453` (baseline-pinned) to `0.0218`
(corrected); log-ratio 68 percent coverage `0.632`;
* radial-profile posterior-mean MAE improves `25.0%` (`5.69e8` to `4.27e8`
Msun); vertical-profile posterior-mean MAE improves `27.5%` (`7.47e8` to
`5.42e8` Msun); fraction/shape summaries unchanged, as expected for a global
rescaling;
* vertical-profile mean bias halves (`mean z -0.635` to `-0.325`, 95 percent
CI now contains 0) and its decision moves from bias-dominated to
spread-miscalibrated;
* radial-profile bias flips sign and shrinks (`-0.444` to `+0.346`); the
remaining radial bias is no longer aligned with the total-mass offset
(correlation drops from `0.60` to `0.21`);
* coverage decisions: 5 of 6 summaries consistent with calibrated; the
vertical profile is now classified spread-miscalibrated (`0.819`, CI
`[0.687, 0.930]`);
* the largest remaining significant bias is the central mass fraction
(`mean z +0.928`, CI `[0.327, 1.583]`), a shape bias that total-mass
rescaling cannot and did not change.

### Inclination-Aware Central Features (And A PCA-64 Negative Result)

Code:

* `make\\\_central\\\_image\\\_features` in `scripts/train\\\_density\\\_residual\\\_pca.py`:
elliptical-aperture flux fractions (semi-major `1, 2, 4, 8 kpc`, axis ratio
`cos i`, apertures oriented with the apparent minor axis along image axis 0
per the `dgdp.projection` convention) plus a flux-weighted minor/major
second-moment ratio inside the `8 kpc` ellipse.
* `--central-pixel-scale-kpc` flag on
`scripts/train\\\_density\\\_residual\\\_pca\\\_mdn.py` and
`scripts/sweep\\\_density\\\_residual\\\_pca\\\_mdn.py` (use `0.35` for the
Milestone 2b images; off by default).

Artifacts:

* `outputs/tng50\\\_milestone2b/density\\\_residual\\\_pca\\\_mdn\\\_sweep\\\_central/`
(32-component basis + central features; selected run
`components\\\_1\\\_seed\\\_20260609`)
* `outputs/tng50\\\_milestone2b/milestone2b\\\_final\\\_evaluation\\\_central/`
* `outputs/tng50\\\_milestone2b/milestone2b\\\_summary\\\_coverage\\\_diagnostics\\\_central\\\_mass\\\_corrected/`
* `outputs/tng50\\\_milestone2b/density\\\_residual\\\_diagnostics\\\_pca64/` and
`...\\\_sweep\\\_central\\\_pca64/`, `...\\\_final\\\_evaluation\\\_central\\\_pca64/`,
`...\\\_diagnostics\\\_central\\\_pca64\\\_mass\\\_corrected/` (tested and rejected, see
below)

Motivation: attribution checks showed the central-mass-fraction bias was
concentrated at `i = 60 deg` (MDN bias `-0.0157` there vs `-0.0027` at
`20 deg`), the PCA basis could represent most of the central structure
(oracle projection bias only `-0.0029`), and regression-to-mean was minor.

Held-out central-mass-fraction results (signed bias, total-mass correction
applied; truth mean fraction `0.318`):

|Configuration|test bias|i=20|i=40|i=60|mean z|
|-|-:|-:|-:|-:|-:|
|geometric baseline|-0.0268|-0.0196|-0.0236|-0.0373|n/a|
|MDN, no central features|-0.0076|-0.0027|-0.0045|-0.0157|+0.93|
|MDN + central features (32c)|-0.0055|-0.0012|-0.0023|-0.0129|+0.63|
|MDN + central features (64c)|-0.0068|-0.0008|-0.0030|-0.0167|+0.79|
|oracle ceiling (32c basis)|-0.0029|-0.0011|-0.0023|-0.0054|n/a|

Conclusions:

* central features help everywhere; at `i = 20/40 deg` the MDN now sits at
the 32-component representation ceiling;
* a 64-component basis raises the ceiling (EVR `0.925` to `0.960`; oracle
`60 deg` bias `-0.0054` to `-0.0033`) but the end-to-end MDN gets worse
(`60 deg` bias `-0.0167`, global cell-mass MAE `1.051e6` vs `1.029e6 Msun`):
with 342 training rows, doubling the coefficient count costs more in
estimation error than the richer basis gains. PCA-64 is rejected;
* the adopted configuration is the 32-component basis with central features:
selected run `components\\\_1\\\_seed\\\_20260609` in
`density\\\_residual\\\_pca\\\_mdn\\\_sweep\\\_central`, best cell-mass MAE `1.029e6
Msun`, central-fraction mean z `+0.63` (from `+0.93`), all six summary
coverage decisions at "consistent with calibrated" except vertical RMS
height (spread-miscalibrated, scale fix);
* a gap to the ceiling remains only at `i = 60 deg` (`-0.0129` vs
`-0.0054`).

### Central-Fraction Correction Head

Script:

* `scripts/train\\\_central\\\_fraction\\\_correction.py`

Plumbing:

* `--central-fraction-predictions` on the calibrate and diagnose scripts. The
correction is applied at evaluation time as a mass-conserving logit-space
shift of each posterior sample's (and the posterior mean's) `R < 2 kpc` mass
fraction: inner cells rescale to the shifted fraction, outer cells absorb the
complement (`\\\_apply\\\_central\\\_fraction\\\_logit\\\_shift` in
`scripts/report\\\_milestone2b\\\_physical\\\_summaries.py`).

Artifacts:

* `outputs/tng50\\\_milestone2b/central\\\_fraction\\\_correction/`
* `outputs/tng50\\\_milestone2b/milestone2b\\\_summary\\\_coverage\\\_diagnostics\\\_central\\\_fraction\\\_corrected/`

Design: a 1-component MDN head predicts
`logit(f\\\_true) - logit(f\\\_posterior\\\_mean)` with uncertainty, conditioned on
the standard features plus central elliptical features plus the logit baseline
and posterior fractions. It is trained against the adopted MDN run
(`components\\\_1\\\_seed\\\_20260609`), so it must be retrained if that run
changes. Motivation: the central mass fraction feeds downstream gas dynamical
modeling, which justifies a dedicated head for this one summary.

Key held-out results (test split, on top of central features + total-mass
correction; truth mean fraction `0.318`):

* posterior-mean fraction bias: `-0.0055` to `-0.0011` overall; by
inclination `20/40/60 deg`: `-0.0012/-0.0023/-0.0129` to
`+0.0018/+0.0019/-0.0069`; the `60 deg` MAE improves `0.0147` to `0.0122`;
* the remaining `60 deg` bias (`-0.0069`) is now close to the 32-component
basis ceiling (`-0.0054`);
* bias decomposition: central mean z `+0.634` to `+0.163` with 95 percent CI
`[-0.229, +0.559]` containing zero; mean PIT `0.549`;
* coverage with the head's sampled uncertainty and NO extra temperature:
`0.761`, CI `[0.624, 0.889]`, containing the `0.68` target (slightly
conservative, `std z 0.80`);
* WARNING: do not apply the val-fitted per-summary temperature to the central
fraction after the head. The head is more accurate on validation (logit-delta
MAE `0.032`) than test (`0.048`), so the fitted shrink (`scale 0.65`)
overfits validation and undercovers test (`0.496`). Use scale `1.0` for this
summary; the diagnostics decision row for the central fraction reflects the
shrunk mode and should be read with this caveat;
* other summaries are essentially unchanged (radial mean z `+0.290` to
`+0.202`, the rest within noise).

### m=2 Amplitude Correction Head

Script:

* `scripts/train\\\_m2\\\_amplitude\\\_correction.py`

Plumbing:

* `--m2-predictions` on the calibrate and diagnose scripts. The correction
scales only the `k = +-2` azimuthal Fourier harmonics of each posterior
sample grid (and the posterior mean) per radial band
(`\\\_scale\\\_m2\\\_harmonic\\\_bands` in
`scripts/report\\\_milestone2b\\\_physical\\\_summaries.py`), which rescales the
m=2 profile exactly while preserving total mass, radial profiles, vertical
profiles, and the central fraction by construction (positivity clip aside).

Artifacts:

* `outputs/tng50\\\_milestone2b/m2\\\_amplitude\\\_correction/`
* `outputs/tng50\\\_milestone2b/milestone2b\\\_summary\\\_coverage\\\_diagnostics\\\_all\\\_corrections/`
(total-mass + central-fraction + m=2 corrections together)

Attribution that drove the design (note the sign: negative mean z means the
posterior OVERPREDICTS m=2 amplitude): the bias has geometry- and
radius-dependent sign (bar angle `0/45/90 deg`: `-0.012/+0.003/+0.025`;
`i = 60 deg`: `+0.017`; outer `R > 6 kpc` bins uniformly `+0.019` of
spurious m=2), the 32-component basis is nearly unbiased (oracle bias
`+0.0005`, MAE `0.0205` vs MDN `0.0372`), and the MDN genuinely lacks m=2
discrimination (test correlation `0.71`). A single per-row scalar cannot fix
sign flips with radius, so the head is band-resolved: a 1-component MDN
predicting 4 per-band log amplitude ratios, conditioned on the standard plus
central features plus per-band relative m=2 amplitudes of the baseline and
posterior grids.

Two design decisions verified empirically:

* the target must be the log ratio of band-summed NORMALIZED m=2 profile
values; a mass-amplitude-weighted target was tried first and overcorrects
(test bias flipped to `-0.0107`);
* hyperparameters were selected on validation coverage, not accuracy:
weight decay `1e-2` gives val head coverage `0.682` (vs `0.596` at `1e-4`)
at a small accuracy cost, following the central-fraction overfit lesson.

Held-out results (test, applied together with the other corrections):

* posterior-mean m=2 profile bias by stratum: `i = 60 deg` `+0.0165` to
`-0.0001` (MAE `0.0597` to `0.0485`); bar `90 deg` `+0.0249` to `+0.0111`;
bar `0 deg` `-0.0117` to `-0.0062`; outer band `+0.0188` to `+0.0033`;
overall MAE `0.0372` to `0.0348`;
* summary-level mean z `-0.384` to `-0.263` (CI `[-0.43, -0.08]`, a modest
residual bias is still detectable); `std z 0.963`; raw coverage `0.530` to
`0.631` with CI `[0.567, 0.685]` containing the target;
* trade-offs: the bar band (`1.2-5.2 kpc`) bin bias moves `+0.0027` to
`-0.0110` (mild overcorrection where the model was already unbiased), and
the bar-axis mass fraction mean z moves `+0.216` to `+0.334` (its raw
coverage improves `0.650` to `0.735` and it stays consistent with
calibrated);
* all other summaries are unchanged by construction;
* with all three corrections, all six summary decisions are "consistent
with calibrated" except vertical RMS height (spread-only fix).

Limitation: the head is capped by genuine m=2 unpredictability from these
features (head log-delta MAE `0.26` train vs `0.31` test); a `-0.26 sigma`
residual m=2 bias remains.

### Milestone 2c: Sample Scale-Up To 185 Galaxies

Full report: `docs/reports/milestone2c_sample_scaleup.md`. Config:
`configs/milestone2c.cluster.toml`. Artifacts:
`outputs/tng50_milestone2c_clean3d/`.

The sample lifts the "top 64 by stellar mass" cap to 185 galaxies (same
selection otherwise), giving 1665 rows split 999/333/333 by galaxy
(111/37/37 galaxies, split seed 20260604). Cluster products (manifest,
images, density tables) were fetched by a previous session; everything below
ran locally from the fetched artifacts, with the 2b scripts unmodified (CLI
path overrides only).

Key results (all held-out test, 37 galaxies / 333 rows):

* adopted configuration: 32-component PCA basis with central features,
sweep run `components_1_seed_20260608`
(`density_residual_pca_mdn_sweep_central`), 1-mixture family confirmed;
* posterior-mean cell-mass MAE `5.815e5 Msun`, 66.2 percent better than the
geometric baseline (`1.720e6`); raw coefficient coverage `0.702`;
* PCA-64 re-test: rejected again at 999 training rows; all nine 64-component
runs are worse end-to-end than the adopted 32-component run (best `5.926e5`
vs `5.815e5 Msun`) despite the better basis (EVR `0.893` vs `0.831`);
* total-mass head: fractional MAE `0.0194` to `0.0107`; the 2c baseline is
already much better behaved than 2b's (`0.0194` vs `0.0453` uncorrected);
* central-fraction head: bias `-0.0068` to `-0.0018`, coverage `0.703` at
scale 1.0; the 2b validation-overfit warning does NOT recur at this sample
size;
* m=2 head: weight decay `1e-2` re-selected on validation coverage; profile
MAE `0.0438` to `0.0369`; NOTE: unlike 2b, the uncorrected 2c posterior has
no significant m=2 bias (mean z `-0.002`), so the head is kept for accuracy
and protocol continuity only;
* with all corrections, bias decomposition: radial bias eliminated
(`-0.355` to `+0.035`), central fraction halved but still significant
(`+0.942` to `+0.385`, CI `[0.196, 0.574]`) - the 37-galaxy CIs now resolve
what 2b could not;
* coverage decisions: 4 of 6 summaries consistent with calibrated; both
vertical summaries are UNDERcovered (`0.575`, `0.556`) with `std z`
1.75-1.89 - the per-summary temperature fitted on validation does not
transfer to test, indicating galaxy-level heterogeneity in vertical spread.

Inclination sensitivity scan (2026-06-11): `scripts/scan_inclination_sensitivity.py`
re-evaluates the adopted run end-to-end with perturbed inclination everywhere an
observer would use it (rebuilt baseline, metadata, central features), heads
excluded, paired sampling noise. Artifacts:
`outputs/tng50_milestone2c_clean3d/milestone2c_inclination_sensitivity/`.
Key findings (test split):

* the response is strongly ASYMMETRIC: overestimating inclination by +3/+5 deg
inflates central-fraction MAE by 40/75 percent, bar-axis MAE by 35/65 percent,
m=2 MAE by 19/35 percent, and collapses central-fraction coverage (0.44 to
0.34/0.29); underestimating by the same amount IMPROVES all bar-frame
summaries and moves their biases toward zero;
* the improvement under reduced inclination indicates the thin-disk baseline
overstretches real (thick) disks at the true inclination - the classic
finite-thickness deprojection effect; an effective-inclination (or
thickness-aware q0) correction in the baseline is a promising cheap fix;
* zero-mean Gaussian inclination scatter (sigma 3 deg) is second-order:
5-15 percent MAE inflation, coverage shifts <= 0.04;
* vertical-structure summaries are essentially insensitive to inclination
errors of this size (the sech^2 prior plus learned residual dominates them);
* implication for geometry-uncertainty propagation: the systematic component
of the observer's inclination estimate matters far more than its variance;
prioritize modeling/removing the thin-disk inversion bias before adding
stochastic marginalization.

Bar-region recovery analysis (2026-06-12): `scripts/analyze_bar_region_recovery.py`
evaluates the adopted 2c run in bar-scaled regions (bar R < L_bar from the
manifest bar_length, transition [L, 1.5L), outer >= 1.5L), on the test split,
raw posterior (no heads, no temperature). Artifacts:
`outputs/tng50_milestone2c_clean3d/milestone2c_bar_region_recovery/`.
Methodology and conventions were adversarially audited (frame alignment,
m=2 phase wrapping, units, aggregation - all confirmed; truth bar-region m=2
phase clusters at 1.8 deg median, confirming bar alignment). Key findings
(median [p16, p84] over 37 galaxies):

* the bar region is the BEST-recovered region: cell relative MAE 0.261 (bar)
< 0.316 (transition) < 0.397 (outer); baseline 0.488/0.983/1.261;
* bar quadrupole: m=2 amplitude error -0.004 on true 0.267, absolute phase
error 3.76 deg median (1 of 37 galaxies above 10 deg, worst 14.2);
baseline: -0.050 and 5.89 deg;
* bar-region mass share (self-normalized fractions): posterior -0.009
[-0.018, +0.004] vs baseline -0.018 - same sign as the known central
concentration underprediction, halved by the MDN;
* vertical: the thin-disk baseline is 0.42x the true bar RMS z; the
posterior fixes it to 1.042 [0.902, 1.195]; per-galaxy vertical anisotropy
(along-bar vs perpendicular RMS z in [0.3, 0.8] L_bar) tracks truth with
Pearson r 0.888 (all 37 galaxies have ratio < 1, so the ratio measures
anisotropy, not a literal peanut detection; sign agreement is degenerate);
* bar-region coverage (raw): mass fraction 0.387 - same raw-posterior
narrowness as the central fraction (0.438 raw in the inclination scan
reference), NOT comparable to the 0.685 all-corrections headline; m=2
amplitude 0.661 (fine);
* TODO (upstream, found by audit): the PCA targets were normalized by the
TRUTH grid mass (`analyze_tng50_density_residuals.py`) but all
reconstructions multiply by the BASELINE grid mass
(`reconstruct_delta_mass_from_coefficients` callers) - a ~2 percent per-row
amplitude inconsistency, mostly absorbed by the preserve-total rescale.
Harmless for truth-referenced evaluation but should be unified at the next
training cycle; note the stored true_coefficients do not invert exactly
under the deployed transform.

Milestone 2d - finer-z X-recovery validation (2026-06-12): tested whether the
method recovers a boxy/peanut X-shaped bulge. The production grid (|z|<10 kpc,
0.625 kpc) cannot resolve the X bifurcation, so the full sample was re-binned at
|z|<5 kpc / 0.3125 kpc (option 1: 2x finer vertical resolution at zero grid
cost; truncates ~5-8% diffuse halo, raising baseline total-mass error to ~7.6%,
which a total-mass head would absorb and which does not affect vertical shape).
Cluster orchestration: `scripts/_milestone2d_cluster.py`; data under remote
`outputs/tng50_milestone2c_z5/` and local `/mnt/e/dgdp-milestone2d/`. Peanuts
are rare in this mass-selected sample: of the top 13 off-plane-mass candidates,
only subhalo 392276 has a genuine X (bar-end vertical profile dips to 0.795 of
its peak with maxima at |z|=0.78 kpc; invisible at 0.625 kpc where it reads
0.986). 392276 was forced into the held-out test split and the adopted MDN
config retrained at 0.3125 kpc. Findings (`scripts/analyze_mdn_x_recovery.py`,
`scripts/_check_basis_represents_x.py`):

* the 0.3125 kpc grid DOES resolve the X (truth dip 0.795);
* the PCA basis - even fit on an X-free training set - CAN represent the X:
reconstructing 392276 from its own basis coefficients gives dip 0.595;
* the MDN does NOT recover the X for the held-out galaxy (posterior dip 1.000,
single-peaked). It recovers the population-typical boxy/THICKENING (posterior
is vertically much thicker than the thin baseline) but not the off-plane
bifurcation;
* diagnosis: this is a PREDICTION failure, not a representation or resolution
failure - 392276 is the only strong X in 185 galaxies, so holding it out
leaves ~zero X training examples and the image->X-coefficient mapping is never
learned.

Implication: validating (or achieving) X-recovery is impossible with TNG50's
mass-selected sample because buckled bars are too rare (~1/185). This is a
quantitative motivation for the planned N-body step: generate many buckled-bar
models so the X is common enough to learn AND to validate on held-out X
galaxies. The grid + basis are already X-capable; only the training
distribution is lacking. NOTE the scoped 2d retrain trained only the adopted
MDN config (no sweep, no correction heads, no calibration) - it answers the
X-recovery question only; a full milestone 2d would add those.

Disk-space note (2026-06-11): the Windows host C: drive filled up during this
work (the WSL VHDX hit a 19.9 GB high-water mark; training jobs died with
SIGBUS and the guest FS briefly wedged). Non-adopted sweep prediction npz
files from 2b and 2c (43 files, 15.7 GB) were MOVED to `E:\dgdp-archive\`
(Windows path), mirroring project-relative paths; metrics, models, and sweep
summaries remain in place, and the adopted runs (2b
`components_1_seed_20260609`, 2c `components_1_seed_20260608`) kept their
predictions locally. Restoration is a plain copy back. All archived arrays
are also exactly regenerable by rerunning the sweep with the same seeds.
VHDX compaction is still pending (needs an elevated shell). Do not run two
heavy python jobs concurrently in this WSL (7.6 GiB RAM; OOM kills corrupt
npz/json writes).

### N-body Cross-Check: Shen2010 MW Bar (Out-Of-Distribution, 2026-06-13)

Full report: `docs/reports/2026-06-13-nbody-shen2010-deprojection.md`. Scripts:
`scripts/eval_nbody_shen2010_deprojection.py` (end-to-end),
`scripts/_nbody_shen2010_characterize.py`,
`scripts/_nbody_shen2010_check_xshape.py`. Artifacts:
`outputs/nbody_shen2010/`.

First out-of-distribution test of the adopted 2c MDN (no retraining, core MDN,
no heads) on a real N-body bar - the Shen et al. 2010 pure-disk MW model
(`/home/zli/Shen2010MW/t800info.dat`, 982,889 particles, already in physical
kpc/km-s; total mass normalized to 4.5e10 Msun). Aligned with the standard
pipeline helper, projected at inclination {20,40,60} x bar angle {20,40,60}.

Key findings:

* in-plane structure recovered well: 3D cell-mass MAE 47% better than the
  geometric baseline at every geometry (median 2.75e5 vs 5.09e5 Msun; flat
  43-50% across i and bar angle, vs 66% on in-distribution TNG - the OOD
  penalty); central mass fraction tracked (truth 0.490, MDN 0.46-0.49 vs
  baseline down to 0.444 at i=60); bar m=2 restored toward truth; total mass to
  0.4%;
* VERTICAL OVER-THICKENING is the failure mode: thin-disk baseline RMS z 0.41
  vs truth 0.77 kpc; the MDN thickens (correct direction) but overshoots to
  1.25 kpc (~1.6x truth) - it imposes the thicker TNG vertical prior on this
  intrinsically thin disk. Reinforces the existing top-priority vertical
  spread issue; motivates a thickness-aware vertical head or N-body disks of
  varied thickness in training;
* CORRECTION (do not repeat the earlier error): t800 IS a strong boxy/peanut X.
  The "boxy not X" claim used the wrong criterion - boxy/peanut/X is a contour
  SHAPE feature, not an off-plane density maximum; in a projected slab the thin
  disk fills the midplane so rho(z) stays single-peaked even with a clear X.
  Measured by iso-density contour shape it is a strong peanut comparable to
  observed B/P bulges. Use the contour/squareness diagnostics, not a dip.

`ruff check .` clean; `pytest -q` 102 passed (additive scripts only).

### Representation direction: grid-free SPH-KDE + superellipsoid (2026-06-14)

Full detail in `docs/reports/2026-06-13-nbody-shen2010-deprojection.md`. Driven by
gal3d (superellipsoid iso-density shapes) + Tahmasebzadeh, Zhu, Shen, Gerhard &
Qin 2021 (MGE deprojection of barred galaxies). Key results:

* flow matching ties the 1-Gaussian MDN in accuracy on the filter+global-PCA-32
  target (residual recon rel-L2 0.31 vs 0.33), marginally better calibrated; the
  bottleneck is image->coefficient predictability, not the generator (the
  slide-15 flow-matching deferral holds on TNG);
* mesh-free SPH-KDE + a nested superellipsoid reconstruction, 3D rel-L2 vs an
  SPH-KDE truth, on FULL particles (earlier TNG numbers used 80k-capped subsamples
  that starved the grid and inflated the margin - corrected 2026-06-22 by
  re-extracting uncapped, 392276 1.50M / 554189 0.56M stars): superellipsoid
  (144 params) vs 0.3125 kpc grid (49152 cells) = Shen2010 0.381 vs 0.465 (win,
  clean N-body); TNG 392276 0.437 vs 0.444 (~tie); TNG 554189 0.619 vs 0.600
  (slight loss, extended disk - a monolithic superellipsoid over-thickens a thin
  disk). So on a fair comparison the superellipsoid is COMPETITIVE with, not
  better than, the fine grid on real TNG galaxies; the rel-L2 "win" was largely a
  particle-noise artifact. The N-independent advantages are the point (~150 params
  vs 49152 cells, smooth, X not resolution-limited, potential-ready);
* even-m Fourier x smooth (R,z) FIXES the disk and is the ADOPTED backbone
  (2026-06-22): one non-stratified component (no image / disk-bulge decomposition),
  rho = sum_{m=0,2,4,6,8,10} a_m(R,z) cos(m phi) + b_m(R,z) sin(m phi), a_m a free
  2D map (`fourier_rz_fit`/`fourier_rz_reconstruct` in
  `reconstruct_superellipsoid_3d.py`). Full-particle 3D rel-L2 vs SPH-KDE truth
  (2002 coeffs): Shen2010 0.385, TNG 554189 0.307, TNG 392276 0.370 - BEATS the
  grid (0.465 / 0.600 / 0.444) on all three and FIXES the superellipsoid disk
  over-thickening (554189 0.619 -> 0.307). Per-shell orientation (tilt only) and an
  m=4 azimuthal term (in-plane bar only) do NOT fix the disk - it is an R-z
  composite problem. It is the AGAMA CylSpline form (potential-ready). The
  superellipsoid is kept only as a compact bulge / B-P squareness descriptor. (MGE
  cannot do B/P; for dynamics the right metric is potential/orbits - the paper:
  <10% potential, 85% orbit match even without the peanut.)

Next steps: (1) DONE and (2) DONE (2026-06-23, see
`docs/reports/2026-06-23-fourier-rz-target-deprojection.md`). (1) Power-weighted
(R,z) compression: 2002 -> 496 coeffs (capture 0.95) at preserved 3D rel-L2 vs the
SPH-KDE truth, still beating the grid on all three galaxies; high harmonics carry
~0.5-1% (more shot-noise than the recalled ~0.1%), so m=6,8,10 drop and m=0,2,4
keep full vertical resolution. (2) Wired the Fourier x (R,z) coefficients in as the
deprojection target (built from the fine-z residual grids, no particles needed)
and retrained MDN + a conditional flow: on the noisy TNG grid the Fourier target
does NOT beat PCA-on-grid for 3D recovery (grid ~20% better on cell-mass MAE; they
TIE on bar m=2 where the flow is best) because the even-m + (R,z) projection
discards ~25% of the grid residual the grid PCA can still partly predict - the
Fourier rep's value is smooth-truth fidelity (step 1) + potential-readiness, not
prediction; confirms the image->coefficient predictability bottleneck. Now the
even-m Fourier x (R,z) representation lives in `src/dgdp/fourier_rz.py` (tested).
(3) DONE (2026-06-23, `docs/reports/2026-06-23-potential-force-validation.md`):
potential/force validation on all three galaxies via a dependency-free isolated FFT
Poisson/force solver (`src/dgdp/poisson_fft.py`; AGAMA CylSpline blocked - no C++
compiler in the sandbox) . RESULT (after a softening/method robustness check,
`scripts/check_potential_softening.py`): under a FAIR self-consistent comparison —
every representation AND the particle reference run through the SAME FFT Poisson
solver (no direct sum, no softening choice) — the Fourier x (R,z) rep is COMPETITIVE
with the cylindrical grid on forces (full Fourier ~6-12% vs grid ~6-23%), and is
BETTER on the extended lower-N disk 554189. An earlier draft used only a direct-sum
reference at one softening (eps=0.3) and overstated the grid's advantage as ~3-4x;
that gap was an artifact — the direct sum penalizes smooth reps for sub-resolution /
N-body-discreteness force fluctuations they legitimately smooth (and the error swings
~10x over eps=0.05-1.0). So "potential-ready" is SUPPORTED: forces comparable to the
grid (better at low N), smooth/analytic/cheap CylSpline form. The rotation curve
v_c(R) still favors the grid (Fourier 10-20 vs grid 3-7 km/s) but that's a separate
coarse-radial-knot effect (more R-knots would close it), not the force-method issue. Still open: (4) the peanut census still needs
a proper B/P pipeline (the b4 metric is disk/bulge-confounded on real galaxies);
(5) the user has more N-body models for strong-X training/validation (deferred). MASS-CONSERVING DEPROJECTION (2026-06-23,
`scripts/deproject_fourier_rz_conserving.py`, report
`docs/reports/2026-06-23-fourier-rz-target-deprojection.md`): re-parameterized
a_m(R,z)=Sigma_m(R)*q_m(z;R), int q_m dz=1, with Sigma_m(R) measured from the
deprojected image (baseline) and the normalized vertical profiles q_m predicted.
Total mass = image mass EXACTLY by construction (only m=0 carries net mass); held-out
recovery ~ the grid pipeline (cell-mass MAE 6.26e5 vs 5.93e5, rel-L2 0.438 vs 0.428)
while the grid pipeline DRIFTS 6.8% off the image mass. Residual mass-vs-truth (7.5%)
= the baseline deprojection floor (fix w/ a 1-scalar Sigma_0 correction). Diagnostic:
true vertical profiles on the image anchor are no better than predicted => the
bottleneck is the radial deprojection Sigma_m(R), not the vertical model. This is the
recommended deprojection backbone; next: smooth knots + flow + Sigma_0 correction +
m>0 projection-consistency. Artifacts: full uncapped TNG particles in
`/mnt/e/dgdp-fullparticles/`, Shen2010 data/cache in `outputs/nbody_shen2010/`,
figures in `outputs/nbody_shen2010/figures/`.

SESSION UPDATE 2026-06-24 (committed): full session in the two 2026-06-23 reports plus
the radial-anchor (step a) and AGAMA cross-check results. Net status of the
even-m Fourier x (R,z) deprojection backbone:
- Representation: compresses 2002 -> ~500 coeff at preserved 3D rel-L2 vs SPH-KDE truth
  (compress_fourier_rz_target.py, pinned 14x13).
- Deprojection (image->3D, milestone2d): mass-conserving image-anchored a_m=Sigma_m(R)q_m(z;R)
  + radial-anchor Sigma_m(R) correction heads (deproject_fourier_rz_conserving.py) BEATS the
  grid PCA pipeline on held-out cell-mass MAE (5.68e5 vs 5.93e5) and rel-L2 (0.375 vs 0.428)
  with total mass a controlled scalar; remaining gap to the oracle anchor is image->coeff
  predictability-limited.
- Potential/forces: validated with an in-house FFT Poisson solver (src/dgdp/poisson_fft.py)
  AND the real AGAMA CylSpline (agama_potential_crosscheck.py). AGAMA is usable from .venv via
  PYTHONPATH=/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313 (user prebuilt agama
  1.0.159; setUnits(mass=1,length=1,velocity=1)).
- STEP (a) DONE 2026-06-24: bumped fit_fourier_rz DEFAULT to 25x25 (was 14x13). fourier_full
  now MATCHES the gold/grid on forces (AGAMA force-err 0.019/0.028/0.029 vs grid
  0.014/0.025/0.037; v_c-rms 2.3-5.7 km/s, <= grid) - representation-only, NO retrain. The
  earlier "Fourier worse on forces" was the coarse default. The deprojection PREDICTED target
  stays compact ~500 (prediction PCA-32-limited). compress_ + reconstruct_superellipsoid_ pin
  14x13 to keep their studies + the frozen alloc JSON.
NEXT (route B, deferred to next session): test whether predicting AGAMA's NATIVE CylSpline /
DensityAzimuthalHarmonic coefficients (finer quintic spline, native AGAMA object out) buys
anything for the deprojection. Feasible (agama.Density(type='DensityAzimuthalHarmonic',
density=callable,...).export()/import round-trips coeffs on a fixed grid+mmax), but likely
prediction-limited (no recovery gain); the cheaper alternative is a to_agama_density() wrapper
on our predicted density (no retrain). Also still open: flow + m>0 projection-consistency on
the conserving target; AGAMA potential/orbit demo end-to-end from one image; the deferred
N-body library / B-P census.

SESSION UPDATE 2026-06-24 (route B, NOT yet committed - report
`docs/reports/2026-06-24-route-b-agama-native-density.md`): both route-B items done.
- (1) AGAMA-NATIVE COEFF ORACLE - REFUTED (`scripts/agama_density_oracle_prototype.py`). Built the
  density as an AGAMA DensityAzimuthalHarmonic on a fixed grid+mmax+symmetry; export()/re-import
  round-trips the coefficients EXACTLY (3-5e-14). On the 3 reference galaxies' smooth SPH-KDE truth,
  AGAMA AZH (mmax=6, 2500 coeff) TIES the Fourier x (R,z) full 14x13 (2002 coeff) - 0.387/0.337/0.372
  vs 0.385/0.307/0.370 - and is WORSE than Fourier full 25x25 (0.351/0.293/0.345); mmax=10 is worse
  still (fits m>=6 shot noise). Forces (AGAMA CylSpline vs particle gold) erratic: AZH best on
  Shen/392276 (0.011/0.016) but worst on extended-low-N 554189 (0.078) vs Fourier 0.019/0.028/0.029.
  AGAMA AZH and our Fourier rep are the SAME representation class (even-m azimuthal harmonics x smooth
  (R,z) maps); the finer quintic spline can't recover the odd-m / fine-cell structure that caps the
  deprojection oracle (Fourier oracle flat above ~600 coeff). So the AGAMA-coeff oracle is NOT clearly
  better -> the gate FAILS -> NO full MDN/flow retrain. Confirms+sharpens the expectation
  (prediction is PCA-32 / image->coeff limited, not representation-limited).
- (2) to_agama_density() HELPER + DEMO DONE (`src/dgdp/agama_density.py`,
  `tests/test_agama_density.py`, `scripts/agama_predicted_potential_demo.py`). Wraps a predicted
  Fourier x (R,z)/conserving density as a native agama.Density (+ agama.Potential), NO retrain, with
  the predicted total mass preserved EXACTLY (harmonic fit is linear -> one rescale; residual ~1e-15).
  End-to-end demo on held-out TNG 554189 (image -> adopted grid-MDN predicted density loaded from
  mdn_z5 predictions -> agama.Potential -> v_c + orbits): mass exact, predicted v_c matches the
  truth-density potential to rms 3.4 km/s (R<15), orbits well behaved. ruff clean; pytest 111 passed
  /1 skipped (115 with AGAMA on PYTHONPATH). Frozen fourier_rz_allocation.json NOT regenerated.
  Still open (unchanged): conditional flow + m>0 projection-consistency on the conserving target; the
  deferred N-body library / B-P census.

REAL-IMAGE TEST 2026-06-24 (NOT committed - report `docs/reports/2026-06-24-ngc4321-real-image-deprojection.md`):
first end-to-end deprojection of a REAL S4G image (NGC 4321 / M100, `NGC4321_m_c_r_f.fits` in repo root;
3.6um, MJy/sr, 0.75"/pixel, stars subtracted). Confirmed the FITS is the ORIGINAL observed mosaic, NOT
pre-deprojected (isotropic CD 0.7500"/0.7500", SIP, ICRS at the galaxy; outer-disk eps~0.10-0.14 = inclined).
S4G geometry i=34.6 deg, disk PA=158.2 deg, D=15.2 Mpc (pixel 0.0553 kpc), M* normalized 6e10. Pipeline =
image-anchored mass-conserving even-m Fourier x (R,z) (scripts/deproject_real_image_ngc4321{,_learned}.py)
-> agama.Potential. FINDINGS: (1) IN-PLANE recovery is solid + observationally anchored - isophote ellipse
fit (scripts/ellipse_fit_ngc4321.py, photutils) recovers NGC4321's DOUBLE bar (nuclear a~0.6 kpc eps0.6;
main bar deprojected a=4.75 kpc b/a=0.43, matches the user's students' bar), and the stellar v_c ~160 km/s
(M*=6e10; full ~210 needs DM+gas) with finer-R structure. (2) VERTICAL structure is prior-dominated (a
face-on image gives NO vertical constraint): geometric sech^2 h=0.3 kpc is too thin; the TNG-learned q_m is
much thicker + FLARING (RMS|z| 0.5->2 kpc) - direction reasonable (old disk/bulge/bar are thick) but the
ABSOLUTE thickness is OOD/unconstrained; do NOT call it "over-thickening" of the truth. (Whether TNG's q_m is
"too thick" is a HYPOTHESIS, NOT measured this session - the "softening-inflated" claim was inferred, not
verified. OPEN PREREQUISITE: measure the TNG sample's stellar h_z(R) from the milestone2d truth grids
(RMS|z|(R) per galaxy, 185 gal) and compare to a real edge-on h_z sample (e.g. Comeron+2018) to learn whether
thickness is even the OOD axis or whether the OOD is morphology, TNG barred mocks vs real grand-design spiral.) (3) b4/boxiness test (scripts/peanut_strength_ngc4321.py): learned edge-on is
NOT a peanut (median b4 -0.016 neutral vs Shen +0.042); it rounds the disk (q 0.16->0.51), doesn't pinch; the
earlier "central double-peak" was a y=0 SLICE artifact (projection is single-peaked). (4) FINER R w/o RETRAIN:
the conserving anchor Sigma_m(R) is image-measured so --n-r-out 128 (log) re-measures it finely + interpolates
the fixed-knot q_m; resolves the inner v_c structure. v_c by direct-sum (rep caps radial at ~25 knots). Controlled
test (scripts/thick_disk_vc_test.py) confirms thick CENTER -> ~30-50% lower central force (user's physics), but
the learned flares so v_c ~unchanged. NEW DEPS installed in .venv: astropy, photutils (binary wheels; FITS/WCS
+ isophote fitting). New module src/dgdp/agama_density.py (+ tests/test_agama_density.py). ruff clean.

NEXT SESSION (user's choice, do NOT start until asked): make the deprojection NOT-OOD by RETRAINING the q_m
vertical head on training truth whose vertical scale spans REAL galaxies. The user explicitly REJECTED the
lazy alternative (swapping in external h_z scaling relations) - using other people's results makes the learned
method meaningless; the learned model IS the point. Plan:
  STEP 1 (prerequisite, cheap, data already on disk): measure the TNG50 milestone sample stellar h_z(R) -
    RMS|z|(R) per galaxy from the milestone2d truth grids (/mnt/e/dgdp-milestone2d/density_residual_table.npz,
    truth_density, 185 gal) - and compare the distribution to a real edge-on h_z sample (Comeron+2018 S4G
    edge-ons give thin/thick h_z for ~140 galaxies). This decides the OOD axis: (a) if TNG h_z overlaps real,
    thickness is NOT the OOD - the OOD is morphology (TNG barred mocks vs a real grand-design spiral) and option
    2 should broaden the morphology/training distribution; (b) if TNG is systematically thicker, the
    softening/resolution-inflation hypothesis holds and the retrain truth must span realistic h_z.
  STEP 2: retrain q_m on the chosen truth - the deferred N-body library (the user has more N-body models; can be
    built with controlled/varied, realistic vertical structure) and/or a TNG vertical-scale correction. Then
    re-apply to NGC4321 and re-check (b4, edge-on, v_c). Frozen fourier_rz_allocation.json stays as-is unless
    intentionally re-deriving the target.
  QUICK CHECK done 2026-06-24 (scripts/ngc4321_comeron_vs_learned.py, fig outputs/real_images/
    ngc4321_comeron_vs_learned.png): swapping the learned q_m for a representative Comeron+2018 thin+thick
    sech^2 prior (h=0.4/1.2 kpc, f_thick=0.3 at M*~6e10) gives RMS|z|(R<12)=0.67 kpc vs the TNG-learned
    1.54 kpc -> the learned PREDICTION is ~2.3x thicker than the published thin+thick decomposition AND flares
    (Comeron disks ~non-flaring). This is the learned OUTPUT vs published (a yardstick / motivation); STEP 1
    above still measures raw TNG TRUTH h_z vs Comeron. Caveat: Comeron h_z is disk-only - the bulge/bar is a
    separate thicker component, so the prior is a disk-level lower bound in the bar region; defaults are
    CLI-overridable (NGC4321 is face-on, no direct edge-on h_z).

STEP 1 DONE 2026-06-25 (report `docs/reports/2026-06-25-tng-vertical-scale-vs-comeron.md`; NOT committed):
measured the 185-galaxy TNG milestone-2d truth h_z and compared to Comeron+2018 (A&A 610 A5, the user's PDF).
Scripts `scripts/measure_tng_vertical_scale.py` + `scripts/compare_tng_comeron_vertical.py`; outputs in
`outputs/tng50_vertical_scale/`. VERDICT = BOTH axes, separable: (1) thick-disc h_z OVERLAPS real at the
masses where the TNG sample lives (zT ratio 1.8x@vc120 -> 1.1x@vc210 -> crossover ~vc235/logM10.95; sample
median logM10.67~vc205), so gross thick over-thickening is NOT confirmed; (2) softening/resolution inflation
IS real but localized to the THIN disc (pinned at ~0.5 kpc grid+softening floor, real zt 0.17-0.35) and the
LOW-mass end (1.8x@vc120); BUT sample-weighted this is MILD -- per-galaxy RMS|z| (the metric driving the
potential) median TNG/Comeron ratio = 1.07 over the whole >1e9.5 selection (thick-h_z 1.01 / RMS 0.98 for
logM>10.5; 81% within +-40%), so for the END-TO-END purpose thickness is LARGELY NOT the OOD. (3) REFRAMED
(user 2026-06-25): the stellar HALO is the CORRECT target, NOT contamination -- a low-incl image integrates
ALL stellar density along z (disc+bulge+halo), so the deprojection target IS total stellar density; TNG's
total-density truth is right, and the high f_thick (RISES 0.42->0.81 vs real MT/Mt FALLING) + flaring are
largely the LEGITIMATE halo+CMC in total light. Comeron's disk-only fit is the WRONG reference for f_thick and
UNDER-states real total-light thickness (true over-thickening < 1.07). Left as OOD: unresolved thin disc
(minor for v_c), whether TNG's OWN halo is realistic (softening/halo-mass tensions, unverifiable face-on), and
MORPHOLOGY (TNG massive-BARRED-only, no grand-design spirals). CONVENTION FIXED: Comeron's
zt,zT are EXPONENTIAL scale heights (= our sech^2(z/2h) h, direct, no factor 2) - verified vs their MW check.
BIG CORRECTION: the 2026-06-24 "learned q_m 2.3x thicker than Comeron" alarm was an ARTIFACT of a too-thin /
convention-confused representative prior (0.4/1.2 treated as z0 = exp 0.2/0.6, RMS 0.67); Comeron's ACTUAL
Eq.18 at NGC4321 (logM10.8, vc~210) gives exp zt~0.30/zT~1.6, real thin+thick RMS|z|~1.6-1.7 kpc -> the
learned 1.54 kpc is REALISTIC, not 2.3x too thick. N-BODY RETRAIN VERDICT (user asked 2026-06-25): largely
NOT needed / would be a step BACKWARD -- the deferred N-body library is PURE-DISC (no stellar halo), so
retraining q_m on it predicts too-THIN total-light density (reintroduces the OOD inverted); and isolated
collisionless discs make bars+B/P+flocculent arms, NOT the gas-driven grand-design of NGC4321, so it does NOT
fill the morphology gap either (same barred/B-P class as TNG, just more B/P diversity). Keep N-body as a
controlled B-P/X-recovery VALIDATION set (the Shen2010 role), not training. If option 2 proceeds it is for
MORPHOLOGY: broaden the HYDRO (TNG) selection (drop barred-only; add unbarred/lower-mass/spiral-dominated,
which KEEP their halos) -- NOT swap in pure-disc N-body. Thickness mostly fine -> cheapest next step is
validating the current q_m on more real images. (One external input, stated: M*->vc via an MW-anchored
stellar Tully-Fisher for axis alignment only, not in any h_z.)
Frozen `outputs/nbody_shen2010/fourier_rz_allocation.json` untouched. ruff clean; pytest 111 passed/1 skipped.

NGC 4371 MGE-VALIDATION 2026-06-25 (report docs/reports/2026-06-25-ngc4371-mge-validation.md; NOT committed):
tried the learned q_m on NGC 4371, a real SB0 from B. Tahmasebzadeh (S4G image+PSF+mask+GALFIT+MGE staged in
NGC4371/), and compared to his INDEPENDENT MGE deprojection. The learned NGC4321 pipeline now runs on WCS-less
S4G cutouts via new flags (--galaxy-name/--pix-arcsec/--pa-pix-deg/--center-x/-y/--mask) on
deproject_real_image_ngc4321_learned.py; geometry from GALFIT i=58(=arccos 0.536)/PA_pix~1.8(=90+GALFIT -88.2)/
0.75"/pix/centre(254.6,152.8)/D=16.194, M*=3.53e10 (M/L=1, matches MGE). i=58 is IN the TNG range (mocks 20/40/60).
MGE benchmark scripts/ngc4371_mge_benchmark.py: viewing angle (59,-11,89), disk intrinsic q~0.28, thin bar,
RMS|z| flares 0.3->2.3, v_c peak 178. RESULT (scripts/compare_ngc4371_learned_vs_mge.py, fig
outputs/real_images/ngc4371_learned_vs_mge.png): our learned RMS|z|(R) MATCHES the MGE to ~15-20% over
R=0.5-8 kpc (both flare 0.5->~1.6-2; thin sech2 baseline 0.27 is 3-6x too thin) -> the learned thick+flaring q_m
is INDEPENDENTLY VALIDATED on a real galaxy (physical, not a TNG artifact; strengthens the Step-1 'thickness is
not the OOD' verdict). v_c: our BASELINE peak 182 matches MGE 179 (geometry+total mass correct). BUG FOUND+FIXED:
the conserving reconstruction didn't preserve Sigma(R) on this strong-barred/peaked SB0 (max rel 198%, central
mass frac R<3 0.52->0.24, half-mass R 5.0 vs MGE/image 2.6-2.9 kpc) -> depressed learned v_c to 131 (v_c^2~M(<R)/R;
RMS|z| is a per-R moment so it stayed validated -- the gap was purely RADIAL, not vertical). Root cause:
reconstruct_smooth normalises int q dz=1 on the KNOT grid but resampling to the fine z grid drifts Sigma(R) for an
OOD q. FIX (in predict_learned): re-impose the image anchor per column -- keep the vertical shape, set
int(rho dz)=Sigma_image(R,phi). After fix: Sigma(R) preserved to 0.2%, learned v_c peak 175 ~= MGE 179, RMS|z|
unchanged; fix is universal (~no-op for the already-conserved NGC4321). DENSITY COMPARISON:
scripts/plot_ngc4371_density_comparison.py (fig outputs/real_images/ngc4371_density_faceon_edgeon.png) -- face-on
(bar along x) + edge-on (thick, flaring) both match the MGE; learned render z-symmetrised + tapered(R>11) +
percentile-scaled + grid-offset to remove the hot-pixel/x=0-seam/large-R/z-asymmetry rendering artifacts (q_m
itself still slightly z-asymmetric -> enforcing z-symmetry at source is an optional pipeline tidy-up, no v_c/inner
impact). NGC4371 validation COMPLETE: vertical AND v_c match the MGE. ROTATION CURVE + BAR
(scripts/compare_ngc4371_vc_and_bar.py, fig outputs/real_images/ngc4371_vc_and_bar.png): azimuthally-averaged
AGAMA v_c match (learned peak 179 vs MGE 183); bar A2(R) profile -- learned(=deprojected image) peak 0.33@R3.4
vs MGE 0.17@R2.3, i.e. the MGE's few-Gaussian bar is ~2x WEAKER in m=2 while our image-anchored in-plane keeps
the full bar strength (relevant for bar dynamics; the v_c is m=0-dominated so it agrees regardless).

PARAMETRIZATION PRE-CHECK 2026-06-25 (scripts/test_mixture_vs_freeknot.py, fig
outputs/tng50_vertical_scale/mixture_vs_freeknot.png; NOT committed): before committing to an option-2 retrain,
tested whether a CONSTRAINED positive symmetric vertical MIXTURE can replace the free-knot q_m. On the 185-gal
milestone-2d truth vertical profiles (m=0; m=2 similar), fit by ridge-stabilised NNLS over a sech^2/Gaussian
height dictionary: a 2-3 component sech^2 mixture is AT LEAST as good as the free-knot AND much better on the
physical RMS|z| moment -- rel-L2 med free-knot 0.087 vs sech2 best-2 0.027 / best-3 0.015; RMS|z| frac err med
free-knot 0.183(!) vs best-2 0.116 / best-3 0.052; median 3 components used. WHY the mixture wins: the free-knot's
sparse outer |z|-knots (0.97/1.95/4.0, clipped at z_max=4) FLATTEN the wings -> 18% RMS|z| error; the mixture's
dedicated thick/halo component captures them. sech^2 preferred over Gaussian for FEW components (exp tails; a
full 10-Gaussian dict also fits but that is overkill). => GREEN LIGHT: bake a ~3-component positive,
midplane-centered sech^2 mixture into the TRAINING parametrization (gives z-symmetry + positivity + int q dz=1
BY CONSTRUCTION, at NEGATIVE accuracy cost) instead of post-hoc reconstruction patches; ~3 comps ~ thin+thick+halo,
consistent with the total-light target. Caveat: the free-knot baseline here is a linear-interp reproduction of
the knot rep, but the wing-knot sparsity is inherent so the win holds qualitatively.

MIXTURE RETRAIN SCOPE + STEPS 1-2 DONE 2026-06-25 (scope doc docs/reports/2026-06-25-mixture-qm-retrain-scope.md;
NOT committed): user confirmed 4 design decisions -- (1) PER-GALAXY-SCALED heights, (2) positive weights +
SIGNED fallback for B/P (m=2), (3) keep PCA + clip/renorm, (4) NEW mixture config (frozen allocation untouched).
Built src/dgdp/vertical_mixture.py (sech^2 kernels phi_k=sech^2(z/2 s a_k)/(4 s a_k); per-galaxy scale
s=mean disk RMS|z|/1.814; NNLS/lstsq weights; reconstruct; __main__ self-check) + de-risk
scripts/test_mixture_pergalaxy.py. DE-RISK RESULT on milestone-2d truth (heights = s*{0.3,0.7,1.5,3.0}, K=4):
m=0 per-galaxy K=4 rel-L2 0.013 / RMS|z| frac-err 4.6% (p90 15%) vs free-knot 0.087 / 18% (p90 30%) -- K=4 (only
4 weights) matches the dense 10-height dict; K=3 = 0.028/10%. m=2: positive-only K=4 median 0.050 but p90 0.188
(>free-knot 0.154); SIGNED fallback recovers p90 to 0.139 (RMS|z| undefined for a signed m=2 modulation -- rel-L2
only). => design VALIDATED. NEXT = step 3: wire the mixture target + reconstruct into deproject_fourier_rz_conserving.py
(+ new config), retrain, compare recovery metrics; then mirror in deproject_real_image_ngc4321_learned.py (delete
the post-hoc anchor/symmetrise/taper) and re-run NGC4321/4371. ruff clean.

RICHER-GRID REBUILD RUNNING ON CLUSTER 2026-06-29 (orchestration scripts/_milestone2d_richgrid_cluster.py; NOT
committed): rebuilding the milestone-2d sample at a richer grid + R=64 for the mixture retrain. Grid = inclination
10-60 step5 (11) x bar -80..90 step10 (18) = 198 proj/galaxy x 185 = 36,630 rows; R=64 log (n_phi48, n_z32, z_max5).
Decisions (user): NO negative inclinations (degenerate with +i & flipped bar for a midplane-symmetric disk, already
spanned; q_m is z-symmetric anyway); -90 dropped (=+90 for an m=2 bar). CLUSTER IS TORQUE/PBS (Maui), NOT SLURM:
qsub/qstat; queue 'normal' (36 nodes/48h). The image builder re-projects all particles per view (~15s/proj,
~50min/galaxy at 198 proj -> ~6 days serial), so it's a PBS job ARRAY: build_tng50_all_particle_images.py got a new
--galaxy-index (subset to Nth galaxy, backward-compat); each task -> shard outputs/tng50_milestone2d_rich/shards/<i>/.
VALIDATED on a 2-task array (correct per-galaxy shards). JOBS (submitted 2026-06-29): 404003[] = image array (185
tasks, -t 0-184%20, 16gb/2h each, ~8h wall, 20 running/165 held at launch); 404008 = truth R=64 (single job, ~1.5-6h,
PARALLEL/independent of images). Monitor: `.venv/bin/python scripts/_milestone2d_richgrid_cluster.py qstat`
(also modes shards/check). RESUME in order once each finishes: (1) array done -> `... merge` (login concat shards ->
.../residual_table.npz + manifest.csv); (2) merge + 404008 done -> `... table` (PBS baseline + density_residual_table
-> density_residual_table.npz, ~36GB = truth+baseline+images per-row at R=64 x36,630 -- STAYS ON CLUSTER, too big to
fetch); (3) wire src/dgdp/vertical_mixture.py (per-galaxy-scaled K=4 sech^2, validated) into
deproject_fourier_rz_conserving.py target+reconstruct (NOT yet done), RETRAIN ON THE CLUSTER (paicos-conda has torch),
fetch only model/metrics/figures. base_manifest.csv (185 gal; split 111/37/37) already written. ruff clean; nothing
committed; frozen fourier_rz_allocation.json untouched.

## Current Git State To Expect

The last clean checkpoint before the physical-summary calibration work was:

```text
a42fa1a Add Milestone 2b PCA residual evaluation
```

Recent checkpoints:

```text
e28e117 Add Milestone 2b physical summary calibration
2c661e3 Add Milestone 2b summary coverage diagnostics
1a8fc16 Add scalar total-mass correction head
```

```text
b358c76 Add inclination-aware central image features
f1b7e28 Add central-fraction correction head
```

```text
6062beb Add band-resolved m=2 amplitude correction head
```

The Milestone 2c artifacts (config, report, and this handoff update) are the
next commit after those checkpoints. No code changes were needed for 2c.

Always run:

```bash
git status --short
```

before editing.

## Verification Status

Before this handoff doc was created, the current code state passed:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q
```

with:

* `ruff`: all checks passed;
* `pytest`: `100 passed` (including the summary coverage diagnostics, the
three correction heads, the central-feature tests, and the m=2 harmonic
scaling property test).

After editing documentation, rerun at least:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q
```

if code files are still part of the uncommitted diff.

## Recommended Next Modeling Step

Do not start a larger full-3D generator yet.

The earlier recommendation to train a summary-specific uncertainty model is now
superseded by the summary coverage diagnostics. With 13 held-out galaxies the
coverage standard error is roughly 0.08-0.13, and the bootstrap CIs show that
per-summary temperature calibration is already statistically consistent with
the 0.68 target for 5 of 6 summaries. Fitting a more expressive uncertainty
model on 13 validation galaxies would chase sampling noise.

The total-mass correction recommended by the diagnostics is now implemented and
verified (see "Total-Mass Correction" above): radial and vertical profile
posterior-mean MAE improved by 25.0 and 27.5 percent, the vertical profile left
the bias-dominated regime, and 5 of 6 summaries are statistically consistent
with calibrated coverage at the 13-galaxy resolution limit.

Status update: the central-feature work (see above) addressed the
central-mass-fraction priority. Note the bias sign: positive `mean z` means
the model UNDERpredicts central concentration (truth is more concentrated
than the posterior). The adopted configuration is now the 32-component basis
with central features (`density\\\_residual\\\_pca\\\_mdn\\\_sweep\\\_central`, run
`components\\\_1\\\_seed\\\_20260609`), with the total-mass correction applied at
evaluation time. More PCA components were tested and rejected (estimation
error beats basis richness at 342 training rows).

Remaining issues, in priority order (updated for Milestone 2c, which
supersedes the 2b-era list - the 37-galaxy test set resolves biases the
13-galaxy set could not):

1. Vertical spread undercoverage (NEW top priority): both vertical summaries
are undercovered on 2c (`0.575`, `0.556` vs target `0.68`) with `std z`
1.75-1.89, and the validation-fitted per-summary temperature does not
transfer to test. A single scale cannot capture the galaxy-level
heterogeneity. Candidates: a vertical-specific uncertainty head conditioned
on inclination and image thickness proxies, or cross-fitted
(leave-galaxy-out) temperature estimation.
2. Central-mass-fraction residual bias: after the head, `mean z +0.385`
(CI `[0.196, 0.574]`) - clearly significant at 2c resolution even though the
2b equivalent looked resolved. The head still halves the bias and fixes the
point estimate (`-0.0018`); the remainder likely needs better central
features or a finer central grid. Keep using scale `1.0` for this summary.
3. Bar-axis mass fraction coverage sits at the low edge of its CI
(`0.628`, CI `[0.568, 0.685]`); watch it if the m=2 head is retrained.
4. The m=2 head is optional on 2c accuracy grounds (no bias to fix in the
uncorrected posterior); kept for the MAE gain and protocol continuity.
5. PCA-64 is rejected at both 342 and 999 training rows; do not revisit
the basis size without an order-of-magnitude more galaxies or a different
representation (the deficit is estimation error, not representation).

Only if these shape biases resist residual-model improvements should a larger
3D model be considered.

Keep posterior-mean comparisons against the existing MDN and deterministic PCA
baselines, and keep reporting MAE, 68 percent coverage with galaxy-bootstrap
CIs, and interval width by inclination, bar viewing angle, and summary family.
Run the corrected evaluation by passing
`--total-mass-predictions outputs/tng50\\\_milestone2b/total\\\_mass\\\_correction/total\\\_mass\\\_correction\\\_predictions.npz`
to the calibrate and diagnose scripts.

Do not interpret point-estimate coverage differences smaller than the bootstrap
CI width as real; 13 held-out galaxies (2b) cannot resolve them, and even the
37 held-out galaxies of 2c leave per-summary coverage CIs roughly 0.08-0.25
wide. For new 2c evaluation runs, use the 2c artifact paths under
`outputs/tng50_milestone2c_clean3d/` (see the Milestone 2c section above).

