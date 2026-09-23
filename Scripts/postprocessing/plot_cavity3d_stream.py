#!/usr/bin/env python3
"""Streamlines-only mid-plane slice of the 3D driven cavity, EPS-ready.

The 3D cavity drives the z = L wall in +x, so the primary vortex lives in
the x-z plane: the default slice is the y = 0.5 L mid-plane (the classic
cavity view, as in the paper's xz-plane figure). --plane z gives the
z = const cut instead (secondary flow).

Particles within half a lattice spacing of the plane are binned onto a
G x G in-plane grid; streamlines are coloured by in-plane speed. The
output format follows the extension (.eps stays fully vector).

Usage:
  python3 postprocessing/plot_cavity3d_stream.py <vtk-or-dir> [out.eps]
          [--plane y|z] [--pos FRAC] [--grid G]
"""
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_vtk(path):
    with open(path) as f:
        lines = f.read().split("\n")
    t = None
    if len(lines) > 1 and "t=" in lines[1]:
        t = float(lines[1].split("t=")[-1].strip())
    pts = vel = None
    i = 0
    while i < len(lines):
        tok = lines[i].split()
        if tok and tok[0] == "POINTS":
            n = int(tok[1])
            vals = []
            i += 1
            while len(vals) < 3 * n:
                vals.extend(float(v) for v in lines[i].split())
                i += 1
            pts = np.array(vals).reshape(-1, 3)
            continue
        if tok and tok[0] == "VECTORS":
            n = pts.shape[0]
            vals = []
            i += 1
            while len(vals) < 3 * n:
                vals.extend(float(v) for v in lines[i].split())
                i += 1
            vel = np.array(vals).reshape(-1, 3)
            continue
        i += 1
    return t, pts, vel


def save_slice_stream(vtk_file, out, plane="y", pos=0.5, grid=None):
    t, pts, vel = read_vtk(vtk_file)
    # in-plane axes: slice normal 'y' -> (x, z); normal 'z' -> (x, y)
    axn = {"y": 1, "z": 2}[plane]
    axa, axb = (0, 2) if plane == "y" else (0, 1)
    lab = {"y": ("x (m)", "z (m)"), "z": ("x (m)", "y (m)")}[plane]

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    G = grid or max(16, int(round(len(pts) ** (1.0 / 3.0))))
    dxl = (hi[axn] - lo[axn]) / max(G - 1, 1)
    cpos = lo[axn] + pos * (hi[axn] - lo[axn])
    slab = np.abs(pts[:, axn] - cpos) < 0.55 * dxl

    A, B = pts[slab, axa], pts[slab, axb]
    UA, UB = vel[slab, axa], vel[slab, axb]
    ae = np.linspace(lo[axa], hi[axa], G + 1)
    be = np.linspace(lo[axb], hi[axb], G + 1)
    cnt, _, _ = np.histogram2d(A, B, bins=[ae, be])
    sa, _, _ = np.histogram2d(A, B, bins=[ae, be], weights=UA)
    sb, _, _ = np.histogram2d(A, B, bins=[ae, be], weights=UB)
    with np.errstate(invalid="ignore"):
        ga = np.where(cnt > 0, sa / np.maximum(cnt, 1), 0.0)
        gb = np.where(cnt > 0, sb / np.maximum(cnt, 1), 0.0)
    ac = 0.5 * (ae[:-1] + ae[1:])
    bc = 0.5 * (be[:-1] + be[1:])
    speed = np.hypot(ga, gb)

    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    strm = ax.streamplot(ac, bc, ga.T, gb.T, color=speed.T, cmap="viridis",
                         density=1.4, linewidth=1.0, arrowsize=1.0)
    fig.colorbar(strm.lines, ax=ax, shrink=0.85, label=r"$|u|$ (m/s)")
    ax.add_patch(plt.Rectangle((lo[axa], lo[axb]), hi[axa] - lo[axa],
                               hi[axb] - lo[axb], fill=False, ec="k",
                               lw=1.2))
    ax.set_xlim(lo[axa], hi[axa])
    ax.set_ylim(lo[axb], hi[axb])
    ax.set_aspect("equal")
    ax.set_xlabel(lab[0])
    ax.set_ylabel(lab[1])
    ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
    fig.tight_layout()
    root, ext = os.path.splitext(out)
    outs = [out] if ext not in (".eps", ".png") else \
        [root + ".eps", root + ".png"]
    for o in outs:
        fig.savefig(o, dpi=200)
        print(f"saved {o}  (t={t:.3e} s, plane {plane}={pos:g}L, "
              f"{int(slab.sum())} particles in slab)")
    plt.close(fig)
    return outs[0]


def main():
    args = sys.argv[1:]
    target = args.pop(0)
    out = None
    plane, pos, grid = "y", 0.5, None
    while args:
        a = args.pop(0)
        if a == "--plane":
            plane = args.pop(0)
        elif a == "--pos":
            pos = float(args.pop(0))
        elif a == "--grid":
            grid = int(args.pop(0))
        else:
            out = a
    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "output_*.vtk")))
        if not files:
            sys.exit(f"no output_*.vtk in {target}")
        target = files[-1]
    if out is None:
        out = os.path.splitext(target)[0] + f"_stream_{plane}mid.eps"
    save_slice_stream(target, out, plane, pos, grid)


if __name__ == "__main__":
    main()
