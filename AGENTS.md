# AGENTS.md

## Project Goal

Develop a probabilistic deprojection workflow for local barred galaxies. The first
scientific target is the stellar mass distribution:

```text
p(rho_star_3d | S4G-like image, inclination, PA, bar angle, metadata)
```

This is not a unique image-to-truth inversion. Treat the output as a posterior
distribution of plausible 3D stellar mass structures, calibrated on TNG50
galaxies whose true 3D stellar particle distribution is known.

## Working Assumptions

- Primary target: stellar mass, not total mass including dark matter.
- Primary observational analogue: S4G 3.6/4.5 micron imaging, preferably old
  stellar light or stellar mass maps after nonstellar contamination correction.
- Initial inclination range: face-on to moderately inclined systems, roughly
  `i < 60 deg`.
- TNG50 is the simulation testbed. Real S4G application comes only after a
  held-out TNG50 benchmark works.
- Inclination, disk PA, and bar PA should be conditioning variables, not hidden
  constants. If they are uncertain, represent that uncertainty explicitly.

## Non-Goals for the First Pass

- Do not try to infer dark matter or total dynamical mass from imaging alone.
- Do not infer fine vertical structure below the reliable resolution of TNG50.
- Do not train directly on real S4G images before validating on simulations.
- Do not build a large end-to-end black box before a simple baseline exists.
- Do not add speculative features, frameworks, or abstractions.

## Success Criteria

A useful first version should demonstrate all of the following on TNG50:

1. A reproducible barred-galaxy sample with documented selection cuts.
2. A forward model that turns true 3D stellar mass into S4G-like 2D images.
3. A classical baseline deprojection for comparison.
4. A probabilistic model that returns posterior samples or posterior summaries.
5. Held-out tests split by galaxy, not merely by projection angle.
6. Reprojection checks showing that inferred 3D samples reproduce the input
   image within the assumed noise model.
7. Calibration checks showing wider uncertainty for ambiguous geometries.
8. Quantitative improvement or clear complementary value relative to the
   baseline.

## Workflow 0: Define the Target

Goal: make the inference problem precise before coding models.

Tasks:

- Choose the 3D target representation:
  - coarse Cartesian density grid,
  - cylindrical `rho_star(R, phi, z)` grid,
  - radial and vertical profile parameters,
  - gravitational potential or force-field summaries.
- Start with the smallest useful target. A good first target is radial stellar
  mass profile plus vertical scale-height or bar-thickness summaries.
- Define coordinate conventions: galaxy center, disk plane, bar major axis,
  radial units, vertical units, and mass normalization.
- Define what metadata the model receives: inclination, PA, distance, pixel
  scale, PSF, bar angle, total stellar mass, and optional morphology labels.

Verify:

- A written target specification exists.
- A trivial synthetic density can be projected and its coordinates recovered.
- Every model output has a physical unit and a validation metric.

## Workflow 1: Select TNG50 Barred Galaxies

Goal: build a clean simulation sample before machine learning.

Tasks:

- Use a published TNG50 barred-galaxy catalogue if available, or implement a
  transparent bar selection using stellar mass, diskiness, and Fourier bar
  strength.
- Focus first on `z = 0` or the closest available local snapshot.
- Apply conservative cuts for stellar mass, number of stellar particles, disk
  morphology, bar size, bar strength, and isolation/merger contamination.
- Store a manifest with subhalo ID, snapshot, stellar mass, disk orientation,
  bar angle, bar length, particle count, and train/validation/test assignment.
- Split by galaxy first. Generate multiple projections only after the split.

Verify:

- The manifest can be regenerated from code and fixed config.
- Example face-on, edge-on, and bar-aligned images pass visual inspection.
- Train, validation, and test sets contain distinct galaxies.

## Workflow 2: Build the 3D Ground Truth

Goal: convert TNG50 stellar particles into reproducible target fields.

Tasks:

- Center each galaxy consistently.
- Align coordinates to the stellar disk plane and bar major axis.
- Convert stellar particles into mass density grids or profile summaries.
- Choose grid resolution based on TNG50 resolution, not visual ambition.
- Track total mass conservation between particles and gridded targets.
- Save derived products with enough metadata to audit orientation and units.

Verify:

- Gridded stellar mass matches particle stellar mass within a stated tolerance.
- Recomputed profiles are stable against reasonable bin-size changes.
- Coordinate alignment produces expected bar and disk geometry.

## Workflow 3: Make S4G-Like Mock Images

Goal: train and test on images that resemble the real observing problem.

Tasks:

- Project each TNG50 galaxy through sampled inclinations, disk PAs, and bar
  viewing angles.
- Convert stellar mass to mock 3.6 micron light or directly to stellar surface
  density using a documented mass-to-light assumption.
- Convolve with the S4G/IRAC-like PSF.
- Resample to realistic pixel scales and distances.
- Add realistic sky noise, masks, and optional contamination only when needed.
- Keep a noiseless image version for debugging.

Verify:

- Projection of the true 3D grid matches direct particle projection.
- The same 3D object at different inclinations produces expected image changes.
- Reprojection of the ground truth matches the stored mock image.

## Workflow 4: Implement Baselines

Goal: know what simple physics already solves before adding ML.

Tasks:

- Implement a geometric deprojection baseline using inclination and PA.
- Add a simple vertical-profile assumption, such as exponential or sech-squared.
- Optionally add a component baseline: disk plus bulge plus bar, or MGE-style
  components if the project needs dynamical modeling.
- Compare baselines against TNG50 ground truth using the same metrics as ML.

Verify:

- Baseline code runs on the same train/test manifests.
- Baseline failures are documented by geometry and morphology.
- ML claims are always compared against this baseline.

## Workflow 5: Probabilistic Model v1

Goal: learn a calibrated posterior for low-dimensional physical summaries.

Tasks:

- Begin with conditional density estimation for profile parameters, not a full
  high-dimensional 3D cube.
- Condition on the image and known geometry metadata.
- Candidate methods: conditional normalizing flow, mixture density network,
  conditional VAE, or another small probabilistic model.
- Train with held-out galaxies and multiple projections per galaxy.
- Report posterior mean, credible intervals, and posterior samples.

Verify:

- Posterior coverage is tested: nominal 68 percent intervals should contain
  truth about 68 percent of the time on held-out galaxies.
- Uncertainty increases for higher inclination, weak bars, noisy images, or
  unfavorable bar viewing angles.
- The model beats or complements the baseline on at least one meaningful metric.

## Workflow 6: Probabilistic Model v2

Goal: move from physical summaries to richer 3D structure only after v1 works.

Tasks:

- Use a coarse 3D density grid or residual around the baseline deprojection.
- Consider conditional diffusion or flow matching for posterior sampling.
- Add physics-aware losses only when they are measurable: mass conservation,
  reprojection consistency, positivity, and smoothness at resolved scales.
- Keep metadata conditioning explicit.
- Compare direct 3D prediction with baseline-plus-residual prediction.

Verify:

- Posterior samples reproject to the input image.
- Mass is conserved within tolerance.
- Held-out TNG50 galaxies are not memorized.
- The method improves scientifically relevant summaries, not only pixel loss.

## Workflow 7: Evaluation

Goal: test what matters scientifically.

Metrics to consider:

- total stellar mass error,
- enclosed mass profile error,
- vertical mass profile error,
- bar length, bar strength, and bar thickness error,
- disk scale length and scale height error,
- central mass concentration error,
- gravitational potential or force-field error,
- reprojection chi-square,
- posterior calibration and sharpness.

Always stratify results by:

- inclination,
- bar viewing angle,
- stellar mass,
- bar strength,
- disk thickness,
- image noise level,
- galaxy morphology.

## Workflow 8: Transfer to S4G

Goal: apply only after simulation validation is credible.

Tasks:

- Choose S4G products consistently: raw IRAC images, ICA old-stellar maps, or
  published stellar mass maps.
- Match preprocessing between mocks and real data: PSF, masks, background,
  pixel scale, orientation, and distance.
- Use catalogued inclination and PA where reliable; otherwise infer them with
  uncertainties and pass those uncertainties to the model.
- Run reprojection and posterior predictive checks on every real galaxy.
- Treat real-galaxy outputs as simulation-informed posterior estimates, not
  direct measurements.

Verify:

- Real S4G images look statistically similar to the mock-image training domain.
- Posterior predictive images match observed images within noise and masks.
- Results are robust to plausible changes in M/L, inclination, and PA.

## Recommended First Milestone

Build a "toy but honest" baseline-plus-residual benchmark:

1. Select a small but clean z~0 TNG50 barred-galaxy sample.
2. Generate S4G-like projections with `i = 0-60 deg`.
3. Run a classical deprojection baseline using known inclination, PA, and bar
   frame metadata.
4. Extract true summaries and baseline summaries: radial stellar mass profile,
   vertical scale-height, bar thickness, bar strength, and central
   concentration.
5. Train a probabilistic residual model:

```text
p(delta_summaries | image_mock, baseline_summaries, geometry, metadata)
```

6. Correct the baseline with posterior residual samples.
7. Produce a short report with metrics, calibration plots, reprojection checks,
   and failure cases.

This milestone is successful if the probabilistic model gives calibrated
uncertainties and either improves over the baseline or clearly identifies where
the 2D image is intrinsically ambiguous.

The next milestone extends the same residual idea to a coarse 3D density target:

```text
p(delta_rho_3d | image_mock, rho_baseline_3d, geometry, metadata)
```

## Agent Working Rules

- State assumptions before implementing.
- Ask when the target, data source, or physical definition is ambiguous.
- Make the smallest change that advances the current workflow.
- Do not refactor unrelated files or invent broad infrastructure.
- Prefer reproducible scripts and configs over notebook-only state.
- Use train/test splits by galaxy to avoid leakage.
- Record units, coordinate systems, and random seeds.
- When using external facts about TNG50, S4G, or methods, cite the source and
  access date in notes or documentation.
- Verify every workflow with at least one concrete check before moving on.

## Open Decisions

Resolve these before serious implementation:

- Exact first target: profile summaries or full 3D density grid.
- Exact TNG50 barred-galaxy source: published catalogue or custom selection.
- Mock image realism level for v1: clean images only, or S4G-like PSF/noise.
- Whether S4G inputs will use ICA old-stellar maps or a custom M/L conversion.
- Compute budget for training probabilistic models.
