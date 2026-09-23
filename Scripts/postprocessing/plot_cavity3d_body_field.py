#!/usr/bin/env python3
"""Body path traced on the 3-D driven-cavity flow field (z-midplane).

The 3-D counterpart of plot_cavity_body_field.py: the VTK gives the frozen
velocity field in the z = L/2 plane, bodies.csv gives the whole centre track
up to that instant. Seeing the two together is the only way to tell "the body
is being carried around the vortex" from "the body happens to be moving in a
curve" -- the streamlines say which.

Usage: plot_cavity3d_body_field.py <run_dir> <out.png> [cube_side_m]
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import griddata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_cavity3d_body import read_vtk                      # noqa: E402
from find_vortex import core_from_grid                       # noqa: E402

C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
NM, BOX = 1e9, 1e-6

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})


def midplane(fn, n=90):
    """ux, uy on a uniform grid in the z = L/2 plane."""
    d = read_vtk(fn)
    p, v = d["pts"], d["velocity"]
    zc = 0.5 * BOX
    m = (np.abs(p[:, 2] - zc) < 0.6 * BOX / 30) & (d["boundary"] < 0.5)
    ok = np.isfinite(v[:, 0]) & np.isfinite(v[:, 1]) & np.isfinite(p[:, 0])
    m &= ok
    g = np.linspace(0, BOX, n)
    X, Y = np.meshgrid(g, g)
    UX = griddata((p[m, 0], p[m, 1]), v[m, 0], (X, Y), method="linear")
    UY = griddata((p[m, 0], p[m, 1]), v[m, 1], (X, Y), method="linear")
    return d["t"], X, Y, np.nan_to_num(UX), np.nan_to_num(UY)


def main():
    run, dst = sys.argv[1], sys.argv[2]
    L = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5e-7

    b = np.genfromtxt(os.path.join(run, "bodies.csv"), delimiter=",",
                      names=True)
    g = lambda k: np.atleast_1d(b[k])
    tb, cx, cy = g("t"), g("cx"), g("cy")
    fin = np.isfinite(cx) & np.isfinite(cy)
    if not fin.all():
        print("run went non-finite at t=%.3e; tracing the finite part only"
              % tb[~fin][0])
    tb, cx, cy = tb[fin], cx[fin], cy[fin]

    # latest snapshot whose field is still finite
    field = None
    for fn in sorted(glob.glob(os.path.join(run, "output_*.vtk")))[::-1]:
        t, X, Y, UX, UY = midplane(fn)
        if np.isfinite(UX).all() and np.abs(UX).max() > 0 and t <= tb[-1] * 1.01:
            field = (t, X, Y, UX, UY, fn)
            break
    if field is None:
        print("no usable field snapshot"); return
    t, X, Y, UX, UY, fn = field
    print("field from %s (t=%.3e)" % (os.path.basename(fn), t))

    vx, vy, G, vort = core_from_grid(X, Y, UX, UY)
    print("vortex core (%.1f, %.1f) nm  |Gamma1|=%.3f" % (vx*NM, vy*NM, abs(G)))

    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    spd = np.hypot(UX, UY)
    pc = ax.pcolormesh(X * NM, Y * NM, spd, cmap="BuPu", shading="auto",
                       alpha=0.85)
    cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("in-plane speed  (m/s)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.streamplot(X * NM, Y * NM, UX, UY, color=INK2, density=1.25,
                  linewidth=0.6, arrowsize=0.8)

    # body path, time-coloured
    pts = np.array([cx * NM, cy * NM]).T.reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap="autumn", lw=3.2, zorder=5)
    lc.set_array(tb[:-1] * 1e9)
    ax.add_collection(lc)
    cb2 = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.10)
    cb2.set_label("body track  t (ns)", fontsize=8)
    cb2.ax.tick_params(labelsize=7)

    hd = 0.5 * np.sqrt(3) * L * NM
    for k, col, lab in ((0, C_BLUE, "start"), (-1, C_ORANGE, "end")):
        ax.add_patch(plt.Circle((cx[k]*NM, cy[k]*NM), hd, fill=False,
                                ec=col, lw=1.6, ls="--", zorder=6))
        ax.plot(cx[k]*NM, cy[k]*NM, "o", color=col, ms=8, zorder=7, label=lab)
    ax.plot(vx*NM, vy*NM, "*", color=C_YELLOW, ms=18, mec=INK, mew=0.7,
            zorder=8, label="vortex core")

    ax.set_xlim(0, BOX*NM); ax.set_ylim(0, BOX*NM)
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    ax.annotate("lid  →", xy=(0.5*BOX*NM, 1.015*BOX*NM), ha="center",
                fontsize=9, color=MUTED)
    ax.set_title("3-D driven cavity, z-midplane at t = %.1f ns\n"
                 "free rigid body traced on the flow that carries it"
                 % (t*1e9), fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    print("saved", dst)


if __name__ == "__main__":
    main()
