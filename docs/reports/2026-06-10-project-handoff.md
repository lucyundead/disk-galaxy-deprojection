# Disk Galaxy Deprojection Project Handoff

Date: 2026-06-10

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

## Current Git State To Expect

The last clean checkpoint before the physical-summary calibration work was:

```text
a42fa1a Add Milestone 2b PCA residual evaluation
```

The physical-summary calibration script, its tests, this handoff document, and
the README pointer were committed as:

```text
e28e117 Add Milestone 2b physical summary calibration
```

The summary coverage diagnostics script, its test, and the handoff updates in
this section are the next commit after that checkpoint.

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
* `pytest`: `92 passed in 32.06s` (including the summary coverage diagnostics
test).

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

The diagnostics instead point at posterior-mean bias, with one dominant and
correctable source: the baseline overestimates total grid mass by 3.0-4.5
percent, the residual correction preserves baseline total mass by construction,
and this constraint explains about 77 percent of the radial-profile bias.

Recommended next task, in order:

1. Relax the total-mass constraint with a scalar total-mass correction: predict
`log(M\\\_true / M\\\_baseline)` per row (with uncertainty) from the existing
geometry/image features, either as one extra MDN output dimension or as a
separate small head, and rescale corrected grids to the predicted total mass
instead of the baseline total mass.
2. Re-run the physical summary evaluation, calibration, and coverage
diagnostics to confirm: radial-profile and vertical-profile mean bias should
shrink substantially, and the vertical profile should leave the bias-dominated
regime.
3. Only if significant miscoverage remains after de-biasing, revisit a
lightweight per-summary variance model, fitted leave-one-galaxy-out on
validation.

Keep posterior-mean comparisons against the existing MDN and deterministic PCA
baselines, and keep reporting MAE, 68 percent coverage with galaxy-bootstrap
CIs, and interval width by inclination, bar viewing angle, and summary family.

Do not interpret point-estimate coverage differences smaller than the bootstrap
CI width as real; 13 held-out galaxies cannot resolve them.

