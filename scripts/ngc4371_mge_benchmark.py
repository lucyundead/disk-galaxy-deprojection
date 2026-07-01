"""Independent NGC 4371 benchmark: Behzad Tahmasebzadeh's MGE deprojection -> 3D density.

Behzad fit the S4G 3.6um image with a Multi-Gaussian Expansion (disk + bar fit
separately), searched the triaxial+oblate viewing-angle space, and picked the solution
(theta,phi,psi)=(59,-11,89) deg (i~58 deg, consistent with the GALFIT disk axis ratio
0.536). The MGE Gaussians + their intrinsic axis ratios at that angle are COPIED VERBATIM
from his notebook NGC4371/Bar_deprojection_parameter_space.ipynb (not refit here).

We rebuild the 3D stellar density from those Gaussians with explicit physical units
(kpc, Msun) and report the vertical structure RMS|z|(R) and the circular velocity v_c(R)
-- the two quantities we will compare against our TNG-learned q_m deprojection of the
same image, to test whether the learned vertical profile is reasonable for this SB0.

Each Gaussian (deprojected, in the intrinsic frame: disk axisymmetric along z, bar major
axis along x) is a 3D Gaussian with major-axis sigma Sx and axis ratios p=Sy/Sx, q=Sz/Sx;
mass is conserved from the observed projected luminosity M = ML * 2 pi Sigma0 sigma_pc^2 qObs.

Run:
    PYTHONPATH=/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313 \\
        .venv/bin/python scripts/ngc4371_mge_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import agama
except ImportError:
    sys.path.insert(0, "/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313")
    import agama

# ---- MGE fit outputs (copied from Behzad's notebook, cells 4 & 10) ----
# bar (5 Gaussians): peak surface brightness [Lsun/pc^2], sigma [arcsec], projected q
lob_pc = np.array([3704.801, 8404.87, 2675.332, 439.306, 43.923])
siob = np.array([0.294, 1.069, 2.119, 18.296, 45.0])
qob = np.array([0.99, 0.99, 0.981, 0.58, 0.614])
# disk (8 Gaussians)
lod_pc = np.array([3149.241, 3343.404, 2054.372, 147.203, 504.083, 177.944, 124.066, 25.73])
siod = np.array([2.343, 5.112, 8.667, 11.512, 14.211, 31.804, 60.354, 106.066])
qod = np.array([0.575, 0.58, 0.57, 0.965, 0.57, 0.57, 0.57, 0.57])
# intrinsic axis ratios at chosen viewing angle (59,-11,89): p=Sy/Sx, q=Sz/Sx, u=deproj
pb = np.array([0.96, 0.96, 0.93, 0.35, 0.37])
qb = np.array([0.95, 0.95, 0.93, 0.22, 0.12])
ub = np.array([0.98, 0.98, 0.98, 0.76, 0.75])
pd = np.array([1.0] * 8)
qd = np.array([0.3, 0.31, 0.28, 0.95, 0.28, 0.28, 0.28, 0.28])
ud = np.array([1.0] * 8)

DISTANCE_MPC = 16.194
ARCSEC2KPC = DISTANCE_MPC * 1e3 * np.pi / 648000.0  # 0.0785 kpc/arcsec


def build_density(ml=1.0):
    """AGAMA Density (sum of 3D Gaussians) in kpc, Msun. ml = stellar M/L (3.6um)."""
    Sigma0 = np.concatenate((lob_pc, lod_pc))   # Lsun/pc^2
    sig = np.concatenate((siob, siod))          # arcsec (projected major-axis sigma)
    qobs = np.concatenate((qob, qod))
    p = np.concatenate((pb, pd))
    q = np.concatenate((qb, qd))
    u = np.concatenate((ub, ud))
    comps = []
    for i in range(len(sig)):
        sig_pc = sig[i] * ARCSEC2KPC * 1e3                     # projected sigma in pc
        mass = ml * 2 * np.pi * Sigma0[i] * sig_pc**2 * qobs[i]  # Msun (conserved)
        Sx = (sig[i] / u[i]) * ARCSEC2KPC                     # intrinsic major sigma, kpc
        Sy, Sz = Sx * p[i], Sx * q[i]
        comps.append(agama.Density(
            type="Spheroid", axisRatioY=p[i], axisRatioZ=q[i],
            scaleRadius=1.0, gamma=0, beta=0, alpha=1,
            outerCutoffRadius=np.sqrt(2) * Sx, cutoffStrength=2,
            densityNorm=mass / ((2 * np.pi) ** 1.5 * Sx * Sy * Sz)))
    return agama.Density(*comps)


def rmsz_profile(dens, r_kpc, n_phi=48, z_max=8.0, n_z=161):
    """RMS|z|(R) = sqrt(<int rho z^2 dz>_phi / <int rho dz>_phi), same convention as TNG."""
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    z = np.linspace(-z_max, z_max, n_z)
    out = np.empty_like(r_kpc)
    for j, R in enumerate(r_kpc):
        x = (R * np.cos(phi))[:, None] * np.ones_like(z)[None, :]
        y = (R * np.sin(phi))[:, None] * np.ones_like(z)[None, :]
        zz = np.ones_like(phi)[:, None] * z[None, :]
        rho = dens.density(np.column_stack([x.ravel(), y.ravel(), zz.ravel()]))
        rho = rho.reshape(n_phi, n_z)
        num = (rho * zz**2).sum()
        den = max(rho.sum(), 1e-30)
        out[j] = np.sqrt(num / den)
    return out


def main():
    agama.setUnits(mass=1, length=1, velocity=1)  # Msun, kpc, km/s
    out = Path("outputs/real_images")
    out.mkdir(parents=True, exist_ok=True)

    for ml in (1.0, 0.6):
        dens = build_density(ml=ml)
        print(f"M/L={ml}: total stellar mass = {dens.totalMass():.3e} Msun")

    dens = build_density(ml=1.0)  # shape is M/L-independent; v_c reported for M/L=1 and 0.6
    r = np.geomspace(0.2, 25.0, 40)
    rmsz = rmsz_profile(dens, r)

    pot1 = agama.Potential(type="CylSpline", density=dens, mmax=6,
                           rmin=0.1, zmin=0.05, rmax=30.0, zmax=15.0)
    xyz = np.column_stack([r, np.zeros_like(r), np.zeros_like(r)])
    aR = pot1.force(xyz)[:, 0]              # radial acceleration at z=0 (inward<0)
    vc1 = np.sqrt(np.clip(-aR * r, 0, None))   # M/L=1
    vc06 = vc1 * np.sqrt(0.6)                   # v_c scales as sqrt(M/L)

    print(f"\n{'R[kpc]':>7}{'RMS|z|[kpc]':>12}{'vc(ML=1)':>10}{'vc(ML=.6)':>11}")
    for i in range(0, len(r), 3):
        print(f"{r[i]:7.2f}{rmsz[i]:12.3f}{vc1[i]:10.1f}{vc06[i]:11.1f}")

    np.savez(out / "ngc4371_mge_benchmark.npz", r_kpc=r, rmsz_kpc=rmsz,
             vc_ml1=vc1, vc_ml06=vc06)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(r, rmsz, "C0-o", ms=3)
    ax[0].set(xlabel="R [kpc]", ylabel="RMS|z| [kpc]",
              title="NGC4371 MGE benchmark (Behzad)", xlim=(0, 20))
    ax[1].plot(r, vc1, "C1-", label="M/L=1.0")
    ax[1].plot(r, vc06, "C1--", label="M/L=0.6")
    ax[1].set(xlabel="R [kpc]", ylabel="v_c [km/s]", title="stellar v_c", xlim=(0, 20))
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "ngc4371_mge_benchmark.png", dpi=130)
    print(f"\nwrote {out}/ngc4371_mge_benchmark.{{npz,png}}")


if __name__ == "__main__":
    main()
