# Disk Galaxy Deprojection

This repository builds a baseline-plus-residual probabilistic benchmark for
deprojecting barred-galaxy stellar mass structure from S4G-like images.

For project background, current Milestone 2b status, and continuation guidance,
start with `docs/reports/2026-06-10-project-handoff.md`.

## Deprojecting a galaxy image — the `dgdp` package

`dgdp` turns a galaxy image into its deprojected 3-D stellar-mass distribution, rotation curve,
and (optionally) gravitational potential, using a **bundled pretrained model** — no training
data or GPU required.

### Install

```bash
pip install git+https://github.com/lucyundead/disk-galaxy-deprojection
```

Core install is light (`numpy`, `astropy`, `matplotlib`) and ships the ~1 MB model. Retraining
the model additionally needs the TNG table and the `[train]` extra (`pip install '.[train]'`).

### Quickstart

Python:

```python
from dgdp import deproject
r = deproject("galaxy.fits", distance_mpc=15.2, inclination_deg=30, pa_onsky_deg=153,
              ml=1.0, stellar_mass=6e10)
r.density_3d              # (nR, nphi, nz) cylindrical stellar-mass cube [Msun]
r.v_circ([1, 2, 5, 10])   # rotation curve at those radii [km/s]
r.rms_z([1, 5, 10])       # vertical thickness RMS|z|(R) [kpc]
r.edge_on, r.face_on      # 2-D renderings
r.save("out/")            # density.npz + rotation_curve.csv + deprojection.png
```

CLI:

```bash
dgdp-deproject galaxy.fits --distance-mpc 15.2 --inclination-deg 30 --pa-onsky-deg 153 \
    --ml 1.0 --stellar-mass 6e10 -o out/
```

### Geometry & M/L

- Geometry: give `pix_arcsec` + `pa_pix_deg` (+ `center`) to bypass WCS (e.g. an S4G cutout),
  or let a FITS WCS supply the pixel scale and convert an on-sky `pa_onsky_deg`.
- The deprojected **shape** is M/L-independent; M/L only sets the v_c amplitude. Fix the absolute
  scale with either (a) `stellar_mass` (total M\*, M/L already folded in), (b) `luminosity` × `ml`,
  or (c) `zeropoint` + `band_solar_mag` + distance to calibrate the image to a luminosity. With
  none of these the outputs use a relative scale.

### Potential (optional)

`r.potential(R, z)` and `dgdp-deproject --potential` return the AGAMA CylSpline potential Φ(R,z).
AGAMA is not pip-installable — build it from source
(https://github.com/GalacticDynamics-Oxford/Agama) and put it on `PYTHONPATH`; everything else
works without it.

### Method

The in-plane surface density is anchored geometrically from the image; a learned fixed-dictionary
sech² vertical profile q_m(z;R) — trained on 185 TNG50 barred galaxies × 198 projections — supplies
the thickness, mass-conserving by construction. See
`docs/reports/2026-06-30-mixture-qm-fixed-dictionary.md`.

Milestone 1 predicts posterior residuals for physical summaries:

- radial stellar mass profile
- radial surface-density profile
- vertical scale-height summary
- bar thickness summary
- bar amplitude summary
- central concentration summary

Start with:

```bash
python -m pytest -q
python scripts/build_synthetic_benchmark.py --config configs/milestone1.synthetic.toml
python scripts/train_summary_residual_mdn.py --data outputs/milestone1_synthetic/residual_table.npz
python scripts/evaluate_summary_residual.py --run-dir outputs/milestone1_synthetic
```

## Synthetic Milestone 1 Benchmark

Run the synthetic benchmark:

```bash
python scripts/build_synthetic_benchmark.py --config configs/milestone1.synthetic.toml
python scripts/train_summary_residual_mdn.py --data outputs/milestone1_synthetic/residual_table.npz --output-dir outputs/milestone1_synthetic
python scripts/evaluate_summary_residual.py --run-dir outputs/milestone1_synthetic
```

The benchmark writes:

- `outputs/milestone1_synthetic/manifest.csv`
- `outputs/milestone1_synthetic/residual_table.npz`
- `outputs/milestone1_synthetic/summary_residual_mdn.pt`
- `outputs/milestone1_synthetic/normalization.npz`
- `outputs/milestone1_synthetic/metrics.json`

## Milestone 2 Cluster TNG50 Ingestion

Milestone 2 runs TNG-facing work on the remote cluster through the existing HPC
wrapper at /home/lucyundead/codex/hpc-agent/hpc.

    python scripts/cluster_dgdp.py sync
    python scripts/cluster_dgdp.py check-env
    python scripts/cluster_dgdp.py reproduce-milestone1
    python scripts/cluster_dgdp.py run-tng50
    python scripts/cluster_dgdp.py run-tng50-density
    python scripts/cluster_dgdp.py fetch-tng50

The cluster workflow keeps full TNG50 snapshots remote and fetches only compact
artifacts under outputs/tng50_milestone2/.
If local `rsync` is unavailable, the project CLI falls back to archive transfer
through the HPC wrapper and does not delete remote files.
