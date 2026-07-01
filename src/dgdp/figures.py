"""Summary figure for a DeprojectionResult (edge-on | face-on column | RMS|z| | rotation curve)."""
from __future__ import annotations

import os

import numpy as np


def save_summary_figure(result, out_dir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r, z = result.grid["r"], result.grid["z"]
    radii = np.linspace(r[0], min(r[-1], 18.0), 80)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)

    ax[0].imshow(result.edge_on.T, origin="lower", aspect="auto", cmap="magma",
                 extent=[r[0], r[-1], z[0], z[-1]])
    ax[0].set_title("edge-on (R,z) mass")
    ax[0].set_xlabel("R [kpc]")
    ax[0].set_ylabel("z [kpc]")

    ax[1].imshow(result.face_on.T, origin="lower", aspect="auto", cmap="magma",
                 extent=[r[0], r[-1], -np.pi, np.pi])
    ax[1].set_title("face-on column (R,phi)")
    ax[1].set_xlabel("R [kpc]")
    ax[1].set_ylabel("phi [rad]")

    ax[2].plot(radii, result.rms_z(radii), color="#c44e52", lw=2)
    ax[2].set_title("vertical thickness")
    ax[2].set_xlabel("R [kpc]")
    ax[2].set_ylabel("RMS |z| [kpc]")

    ax[3].plot(radii, result.v_circ(radii), color="#4c72b0", lw=2)
    ax[3].set_title("rotation curve" + ("" if not result.relative else " (relative)"))
    ax[3].set_xlabel("R [kpc]")
    ax[3].set_ylabel("v_c [km/s]")
    ax[3].set_ylim(0, None)

    path = os.path.join(out_dir, "deprojection.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
