#!/usr/bin/env python3
"""Is there a closed recirculation in the cavity's z-midplane?

The one unambiguous test: on the vertical centreline x = L/2, a lid-driven
cavity must show REVERSED streamwise velocity (u_x < 0) somewhere below the
lid -- that is the return branch of the primary vortex. No reversal means no
closed streamline, hence nothing for an immersed body to orbit.

Reported per snapshot:
  u_x profile on the centreline, the most negative value and where,
  the Gamma-1 vortex-core strength, and the plane-mean u_x (which a closed
  box at steady state requires to be ~0).

Usage: check_recirculation.py <dir_or_vtk> [stride]
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_cavity3d_body import read_vtk           # noqa: E402
from find_vortex import gamma1                    # noqa: E402


def report(fn, box=1e-6):
    d = read_vtk(fn)
    p, v = d["pts"], d["velocity"]
    # 2-D data has a single z plane; use it rather than L/2
    twod = (p[:, 2].max() - p[:, 2].min()) < 1e-12
    zc = p[:, 2].mean() if twod else 0.5 * box
    xs = np.unique(np.round(p[:, 0], 12))
    xc = xs[np.argmin(np.abs(xs - 0.5 * box))]

    # centreline through the midplane
    m = (np.abs(p[:, 0] - xc) < 1e-12) & (np.abs(p[:, 2] - zc) < 1e-12)
    o = np.argsort(p[m, 1])
    yy, ux = p[m, 1][o], v[m, 0][o]
    interior = (yy > 1e-12) & (yy < box - 1e-12)
    umin = ux[interior].min() if interior.any() else np.nan
    ymin = yy[interior][np.argmin(ux[interior])] if interior.any() else np.nan

    # plane-mean u_x at x = L/2 (closed box, steady -> ~0)
    mp = np.abs(p[:, 0] - xc) < 1e-12
    planemean = v[mp, 0].mean()

    # Gamma-1 core strength in the midplane
    from scipy.interpolate import griddata
    from scipy.ndimage import binary_erosion
    ztol = 1.0 if twod else 0.5 * box / 20
    ms = (np.abs(p[:, 2] - zc) < ztol) & (d["boundary"] < 0.5)
    g = np.linspace(0, box, 121)
    X, Y = np.meshgrid(g, g)
    UX = griddata((p[ms, 0], p[ms, 1]), v[ms, 0], (X, Y), method="linear")
    UY = griddata((p[ms, 0], p[ms, 1]), v[ms, 1], (X, Y), method="linear")
    ok = np.isfinite(UX) & np.isfinite(UY)
    G = gamma1(X, Y, np.nan_to_num(UX), np.nan_to_num(UY), 6)
    G = np.where(binary_erosion(ok, np.ones((13, 13))), G, 0.0)
    g1 = np.abs(G).max()

    rev = umin < -1e-3
    print("t=%.3e  min(u_x)=%+8.4f at y=%.3f um   |Gamma1|=%.3f   "
          "plane-mean u_x=%+7.4f   RECIRCULATION: %s"
          % (d["t"], umin, ymin * 1e6, g1, planemean,
             "YES" if rev else "no"))
    return d["t"], umin, g1, planemean


if __name__ == "__main__":
    tgt = sys.argv[1]
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    files = ([tgt] if tgt.endswith(".vtk")
             else sorted(glob.glob(os.path.join(tgt, "output_*.vtk")))[::stride])
    for fn in files:
        try:
            report(fn)
        except Exception as e:                     # partial/short file
            print("%s: %s" % (os.path.basename(fn), e))
