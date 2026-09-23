#!/usr/bin/env python3
"""Plot driven-cavity fields from the solver's ASCII VTK output.

Usage: python3 tools/plot_cavity.py <output_dir> [step]
Plots the latest snapshot (or the given step): density, temperature,
speed with streamlines, and the u_x profile along the vertical centreline.
"""
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_vtk(fn):
    with open(fn) as f:
        lines = f.readlines()
    t = float(lines[1].split("t=")[1])
    npts = int(lines[4].split()[1])
    pts = np.array([list(map(float, lines[5 + i].split())) for i in range(npts)])
    data = {}
    i = 5 + npts
    while i < len(lines):
        if lines[i].startswith("SCALARS"):
            name = lines[i].split()[1]
            i += 2  # skip LOOKUP_TABLE
            data[name] = np.array([float(lines[i + k]) for k in range(npts)])
            i += npts
        elif lines[i].startswith("VECTORS"):
            name = lines[i].split()[1]
            i += 1
            data[name] = np.array(
                [list(map(float, lines[i + k].split())) for k in range(npts)])
            i += npts
        else:
            i += 1
    return t, pts, data


def save_streamplot(vtk_file, out="stream.eps", grid=None):
    """Save ONLY the streamline plot of a snapshot, publication-style.

    The meshfree cloud is binned onto a uniform grid (so it works even
    when particles have moved off the initial lattice or a body has
    removed some), and the streamlines are coloured by speed.  The output
    format follows the file extension -- .eps stays fully vector since
    the figure is lines only (no raster background).
    """
    t, pts, data = read_vtk(vtk_file)
    X, Y = pts[:, 0], pts[:, 1]
    U, V = data["velocity"][:, 0], data["velocity"][:, 1]

    # Bin count must match the particle lattice: a coarser grid averages
    # the lid row with the (much slower) row below it, so the colour scale
    # would top out well under the true wall speed (e.g. 0.7 instead of
    # 1.0 m/s for a 100x100 cavity binned at 60x60).
    G = grid or int(round(np.sqrt(len(X))))
    x0, x1, y0, y1 = X.min(), X.max(), Y.min(), Y.max()
    xe = np.linspace(x0, x1, G + 1)
    ye = np.linspace(y0, y1, G + 1)
    cnt, _, _ = np.histogram2d(X, Y, bins=[xe, ye])
    su, _, _ = np.histogram2d(X, Y, bins=[xe, ye], weights=U)
    sv, _, _ = np.histogram2d(X, Y, bins=[xe, ye], weights=V)
    with np.errstate(invalid="ignore"):
        gu = np.where(cnt > 0, su / np.maximum(cnt, 1), 0.0)
        gv = np.where(cnt > 0, sv / np.maximum(cnt, 1), 0.0)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    speed = np.hypot(gu, gv)

    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    strm = ax.streamplot(xc, yc, gu.T, gv.T, color=speed.T, cmap="viridis",
                         density=1.4, linewidth=1.0, arrowsize=1.0)
    fig.colorbar(strm.lines, ax=ax, shrink=0.85, label=r"$|u|$ (m/s)")
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               ec="k", lw=1.2))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
    fig.tight_layout()
    root, ext = os.path.splitext(out)
    outs = [out] if ext not in (".eps", ".png") else \
        [root + ".eps", root + ".png"]
    for o in outs:
        fig.savefig(o, dpi=200)
        print(f"saved {o}  (t={t:.3e} s)")
    plt.close(fig)
    return outs[0]


def main():
    # stream mode: python3 plot_cavity.py stream <vtk-or-dir> [out.eps]
    if len(sys.argv) > 1 and sys.argv[1] == "stream":
        target = sys.argv[2]
        if os.path.isdir(target):
            files = sorted(glob.glob(os.path.join(target, "output_*.vtk")))
            if not files:
                sys.exit(f"no output_*.vtk in {target}")
            target = files[-1]
        out = sys.argv[3] if len(sys.argv) > 3 else \
            os.path.splitext(target)[0] + "_stream.eps"
        save_streamplot(target, out)
        return

    outdir = sys.argv[1]
    files = sorted(glob.glob(os.path.join(outdir, "output_*.vtk")))
    if not files:
        sys.exit(f"no output_*.vtk in {outdir}")
    if len(sys.argv) > 2:
        fn = os.path.join(outdir, f"output_{int(sys.argv[2]):06d}.vtk")
    else:
        fn = files[-1]

    t, pts, data = read_vtk(fn)
    n = int(round(np.sqrt(len(pts))))
    x = pts[:, 0].reshape(n, n)
    y = pts[:, 1].reshape(n, n)
    rho = data["rho"].reshape(n, n)
    T = data["T"].reshape(n, n)
    u = data["velocity"][:, 0].reshape(n, n)
    v = data["velocity"][:, 1].reshape(n, n)
    sp = np.sqrt(u**2 + v**2)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))

    im = axes[0].pcolormesh(x, y, rho, cmap="RdBu_r", shading="auto")
    axes[0].set_title(f"rho  [{rho.min():.4f}, {rho.max():.4f}]")
    plt.colorbar(im, ax=axes[0])

    im = axes[1].pcolormesh(x, y, T, cmap="inferno", shading="auto")
    axes[1].set_title(f"T  [{T.min():.2f}, {T.max():.2f}] K")
    plt.colorbar(im, ax=axes[1])

    im = axes[2].pcolormesh(x, y, sp, cmap="viridis", shading="auto")
    axes[2].streamplot(np.linspace(x.min(), x.max(), x.shape[1]),
                       np.linspace(y.min(), y.max(), y.shape[0]), u, v, color="w",
                       density=1.2, linewidth=0.7, arrowsize=0.8)
    axes[2].set_title(f"|u|  max={sp.max():.3f} m/s")
    plt.colorbar(im, ax=axes[2])

    mid = n // 2
    axes[3].plot(u[:, mid], y[:, 0], "o-", ms=3)
    axes[3].axvline(0, color="gray", lw=0.5)
    axes[3].set_xlabel("u_x (m/s)")
    axes[3].set_ylabel("y (m)")
    axes[3].set_title("u_x on vertical centreline")
    axes[3].grid(alpha=0.3)

    for ax in axes[:3]:
        ax.set_aspect("equal")
    fig.suptitle(f"Driven cavity  t={t:.3e} s  ({os.path.basename(fn)})")
    plt.tight_layout()
    out = os.path.join(outdir, "cavity_plot.png")
    plt.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
