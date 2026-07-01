# Session handoff — 2026-07-01

**State:** the research pipeline was packaged into an installable, torch-free **`dgdp`** package
(image -> 3D density + rotation curve + optional potential) with a **bundled R=64 model**,
pushed to GitHub, and made **PyPI-release-ready** (built + validated; trusted-publishing CI added).
Next: three refinement ideas (figures/MGE comparison, mass-dependent baseline scale height,
learn from the Ding AICp poster). This doc is the self-contained entry point.

## Environment / rules (unchanged)
- WSL `openSUSE-Tumbleweed`; project `/home/lucyundead/projects/disk-galaxy-deprojection`;
  run with `.venv/bin/python`, `PYTHONPATH=src`; branch `codex/milestone2-tng-ingestion-design`
  (remote `origin` = github.com/lucyundead/disk-galaxy-deprojection).
- Keep `ruff check .` and `PYTHONPATH=src pytest -q` clean (currently **124 passed, 1 skipped**).
- Cluster: Torque/PBS via `scripts/_milestone2d_richgrid_cluster.py` (has a `train-bundle` mode
  that trains the R=64 model on the cluster and writes `src/dgdp/models/dgdp_fixed_dict.npz`).
- Commit only when asked. AGAMA (optional, potential only) via
  `PYTHONPATH=/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313`.

## What shipped this session (all committed + pushed; see git log, don't re-derive)
The `dgdp` package — inference logic moved out of `scripts/` into `src/dgdp/`:
`util`, `harmonics`, `features` (golden-pinned), `image`, `model` (torch-free MLP mean head),
`rotation` (numpy v_c + optional AGAMA `potential`), `deproject` (`deproject()` +
`DeprojectionResult`), `cli`, `figures`, and the bundled `models/dgdp_fixed_dict.npz` (R=64,
430 KB). Trainer: `scripts/train_deprojection_model.py`. Full design + task detail:
`docs/superpowers/specs/2026-06-30-dgdp-package-design.md`,
`docs/superpowers/plans/2026-06-30-dgdp-package.md`. Method/model report:
`docs/reports/2026-06-30-mixture-qm-fixed-dictionary.md`. Memory `mixture-qm-fixed-dictionary`
loads automatically.

**Verified:** fresh-venv `pip install` runs end-to-end with only numpy/astropy/matplotlib
(torch/scipy/h5py never imported); NGC 4321 (v_c 163.7, RMS|z| 0.75->1.47) and NGC 4371
(v_c 178, RMS|z| 0.87), mass conserved.

## Git / PyPI state
- Tag **v0.2.0** pushed; release metadata (MIT license, classifiers, URLs) in `pyproject.toml`.
- `dist/` holds the validated `disk_galaxy_deprojection-0.2.0` wheel + sdist (`twine check` PASSED;
  gitignored). **0.2.0 still needs the manual `.venv/bin/twine upload dist/*`** (user's PyPI token).
- CI: `.github/workflows/release.yml` publishes to PyPI via **Trusted Publishing (OIDC)** on any
  `v*` tag once the user adds the trusted publisher on PyPI (owner `lucyundead`, repo
  `disk-galaxy-deprojection`, workflow `release.yml`, environment `pypi`). Future releases: bump
  `pyproject` version, tag `vX.Y.Z`, push -> auto-publish, no token.

## RESUME / NEXT — three refinement ideas

### 1. Figure audit of NGC 4321 + NGC 4371 with the R=64 bundle (+ NGC 4371 vs MGE, one-to-one)
The earlier NGC validation used the R=32 model; regenerate + inspect with the shipped **R=64**
bundle (via `dgdp.deproject`, or the research scripts). For BOTH galaxies produce/compare:
observed (original) image, **deprojected face-on AND edge-on** images, and the **rotation curve**.
For **NGC 4371**, a full **one-to-one comparison vs the MGE** (B. Tahmasebzadeh, data in `NGC4371/`):
deprojected face-on + edge-on images, rotation curve, **bar amplitude A2(R)**, **radial density
profile**, half-mass radius, etc. Prior numbers to beat/reproduce (R=32): v_c learned 179 vs MGE
183; A2 learned(=image) peak 0.33@R~3.4 vs MGE 0.17@R~2.3; thickness matched MGE ~15-20%.
Reuse `scripts/compare_ngc4371_learned_vs_mge.py`, `plot_ngc4371_density_comparison.py`,
`compare_ngc4371_vc_and_bar.py` (adapt to load the R=64 bundle / `dgdp.deproject`). See
`docs/reports/2026-06-25-ngc4371-mge-validation.md` and memory `ngc4371-mge-validation`
(note: the post-hoc anchor fix it describes is now structural — deleted).

### 2. Mass-dependent geometric-baseline scale height (Gadotti relation) instead of fixed hz=0.3 kpc
`dgdp.image.geometric_baseline` uses a fixed sech^2 `scale_height_kpc=0.3` (default in
`deproject`), which may be too thin. Since stellar mass (or M/L x L) is now an input, infer hz
from a Gadotti mass-scale-height relation and pass it through.
**Important subtlety to check first:** the learned output is currently **hz-independent** — the
image anchor is `Sigma_m = int(baseline_density) dz` (z-integrated, so hz cancels), and the
vertical structure comes entirely from the learned q_m. So a mass-dependent hz would improve the
**geometric-baseline comparison curve** (make the "thin-disk baseline" fairer/physical) and could
serve as an **independent cross-check** of the learned RMS|z|, but would NOT change the learned
density unless the architecture is changed (e.g., use hz as a prior/fallback for OOD galaxies).
Decide the intended use before implementing.

### 3. Learn from Ding et al. (AICp non-parametric deprojection)
Poster `F:\OneDrive\Desktop\Ding_astrolipari_poster_astro-9026-01_06_2026.pdf` — "Deprojection of
Bar Galaxies", Huangshengzhi Ding (hding@mpe.mpg.de), Jens Thomas, Roberto Saglia (MPE).
Their method: **non-parametric** deprojection (flexible enough to explore the whole range of
densities consistent with the observation) + **AICp** (an Akaike-information-criterion-based
method to optimally extract info from noisy data / regularize against noise); viewing angle
theta=75,phi=45,psi=90; bar+disk+bulge decomposition; density RMS bar 0.23 / outer 0.15. The
deprojection feeds **Schwarzschild modeling** (Tikhonenko et al., MNRAS submitted 2026) to recover
the bar **pattern speed** from an N-body test.
What to take from it: (a) **AICp / noise regularization** — directly relevant to adding
uncertainty/robustness to our point-estimate output; (b) their **non-parametric per-galaxy**
solution is the philosophical opposite of our **population-learned TNG prior** — a head-to-head on
the same galaxy would quantify what the learned prior buys (and whether we sit inside their
consistent-density space); (c) **downstream**: our density cube + optional AGAMA potential could
feed the same Schwarzschild -> pattern-speed pipeline. Possible contact/benchmark with the MPE
group (they overlap with our science goals and with the MGE work).

## Suggested skills for the next session
- `superpowers:brainstorming` before implementing ideas 1-3 (each has a real design fork — esp.
  idea 2's "what does hz actually change" and idea 3's "regularization vs learned prior").
- `superpowers:writing-plans` + `superpowers:executing-plans` once an idea is scoped.
- `claude-api` if any LLM/agent work; `anthropic-skills:pdf` for the poster (PyMuPDF is now in
  `.venv`).

## Gotchas
- The `/handoff` skill writes to the OS temp dir (ephemeral); this committed doc is the durable
  entry point (that's why it's in `docs/reports/`).
- Cluster inline python via `wsl bash -lc '...'`: no single quotes / no unescaped `$VAR`
  expansion in nested quoting — use the orchestration script modes or a heredoc.
- The 31 GB R=64 table stays on the cluster; only the 430 KB bundle is fetched/committed.
- Bundled model = R=64 fixed-dict sech^2 mixture; retrain via `_milestone2d_richgrid_cluster.py
  train-bundle` then base64-fetch `src/dgdp/models/dgdp_fixed_dict.npz`.
