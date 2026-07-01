"""De-risk the CHOSEN parametrization: per-galaxy-scaled positive sech^2 mixture (K=3,4),
positive-only weights with a signed fallback for the m=2 (bar) harmonic.

Per galaxy: s = mean RMS|z|(R) over the disk (mass-weighted) -> exp scale height; the K
kernel heights are s*ratios. Per (R, mode) we fit weights over that scaled basis (NNLS, or
lstsq for the signed test) and compare the residual to the current free-knot representation.

If per-galaxy K=3-4 matches/beats free-knot, the constrained parametrization is safe with a
small predicted weight vector. Read-only; uses src/dgdp/vertical_mixture.py.

Run: .venv/bin/python scripts/test_mixture_pergalaxy.py
"""

from __future__ import annotations

import numpy as np

from dgdp import vertical_mixture as vm

R3 = np.array([0.35, 1.0, 3.0])


def freeknot_recon(p, zabs):
    knots = np.concatenate([[0.0], np.geomspace(0.12, 4.0, 6)])
    return np.interp(zabs, knots, np.interp(knots, zabs, p))


def metrics(fit, p, zabs):
    rel = np.linalg.norm(fit - p) / max(np.linalg.norm(p), 1e-30)
    rms_t = np.sqrt((p * zabs**2).sum() / max(p.sum(), 1e-30))
    rms_f = np.sqrt((fit * zabs**2).sum() / max(fit.sum(), 1e-30))
    return rel, abs(rms_f - rms_t) / max(rms_t, 1e-30)


def main():
    z = np.load("/mnt/e/dgdp-milestone2d/density_residual_table.npz")
    td, gid = z["truth_density"], z["galaxy_id"]
    zc = 0.5 * (z["z_edges_kpc"][:-1] + z["z_edges_kpc"][1:])
    phi = 0.5 * (z["phi_edges_rad"][:-1] + z["phi_edges_rad"][1:])
    npos = zc.size // 2
    zabs = zc[npos:]
    _, first = np.unique(gid, return_index=True)
    gals = np.sort(first)[::4]

    m0 = ["free-knot", "pergal K=3", "pergal K=4"]
    res = {k: {"rel": [], "rms": []} for k in m0}
    res2 = {k: {"rel": [], "rms": []} for k in ("free-knot", "m2 K=4 positive", "m2 K=4 signed")}
    for gi in gals:
        g = td[gi].astype(np.float64)
        rho0 = g.mean(axis=1)
        a2 = np.abs((g * np.exp(-2j * phi)[None, :, None]).mean(axis=1))
        col = rho0.sum(axis=1)
        disk = col > 0.01 * col.max()
        rmsz_R = np.sqrt((rho0 * zc[None, :] ** 2).sum(axis=1) / np.maximum(col, 1e-30))
        s = vm.galaxy_scale(rmsz_R[disk], weights=col[disk])
        for ri in np.where(disk)[0]:
            p = rho0[ri, npos:] + rho0[ri, npos - 1::-1]
            p = p / max(p.max(), 1e-30)
            fits = {"free-knot": freeknot_recon(p, zabs),
                    "pergal K=3": vm.reconstruct_profile(zabs, s, vm.fit_weights(p, zabs, s, R3), R3),
                    "pergal K=4": vm.reconstruct_profile(zabs, s, vm.fit_weights(p, zabs, s))}
            for k in m0:
                rl, rm = metrics(fits[k], p, zabs)
                res[k]["rel"].append(rl)
                res[k]["rms"].append(rm)
            p2 = a2[ri, npos:] + a2[ri, npos - 1::-1]
            if p2.max() > 0:
                p2 = p2 / p2.max()
                f2 = {"free-knot": freeknot_recon(p2, zabs),
                      "m2 K=4 positive": vm.reconstruct_profile(zabs, s, vm.fit_weights(p2, zabs, s)),
                      "m2 K=4 signed": vm.reconstruct_profile(zabs, s, vm.fit_weights(p2, zabs, s, signed=True))}
                for k in res2:
                    rl, rm = metrics(f2[k], p2, zabs)
                    res2[k]["rel"].append(rl)
                    res2[k]["rms"].append(rm)

    print(f"per-galaxy-scaled sech^2 mixture vs free-knot   (n m=0 = {len(res['free-knot']['rel'])})\n")
    print(f"{'model':<17}{'relL2 med':>10}{'relL2 p90':>10}{'RMSz med':>10}{'RMSz p90':>10}")
    for k in m0:
        rel, rms = np.array(res[k]["rel"]), np.array(res[k]["rms"])
        print(f"{k:<17}{np.median(rel):>10.3f}{np.percentile(rel, 90):>10.3f}"
              f"{np.median(rms):>10.3f}{np.percentile(rms, 90):>10.3f}")
    print(f"\nm=2 bar harmonic -- positive-only vs signed fallback (n = {len(res2['free-knot']['rel'])}):")
    for k in ("free-knot", "m2 K=4 positive", "m2 K=4 signed"):
        rel, rms = np.array(res2[k]["rel"]), np.array(res2[k]["rms"])
        print(f"{k:<17}{np.median(rel):>10.3f}{np.percentile(rel, 90):>10.3f}"
              f"{np.median(rms):>10.3f}{np.percentile(rms, 90):>10.3f}")


if __name__ == "__main__":
    main()
