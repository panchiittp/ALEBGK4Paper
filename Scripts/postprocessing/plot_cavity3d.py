#!/usr/bin/env python3
"""Plot the xy-plane slice (z = L/2) of a 3D driven-cavity VTK snapshot.
This is the velocity-resolved plane of the Chu-reduced formulation and the
plane of the lid vortex (the paper's Fig. 5.3 "xz at y = L/2" in its own
axis naming). Lid on the y = L wall moving in +x.

Usage: python3 tools/plot_cavity3d.py <output_dir> [step]
Assumes an Nx x Nx x Nx lattice written in i-fastest (x), then j (y),
then k (z) order.
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
            i += 2
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


def main():
    outdir = sys.argv[1]
    files = sorted(glob.glob(os.path.join(outdir, "output_*.vtk")))
    if not files:
        sys.exit(f"no output_*.vtk in {outdir}")
    fn = (os.path.join(outdir, f"output_{int(sys.argv[2]):06d}.vtk")
          if len(sys.argv) > 2 else files[-1])

    t, pts, data = read_vtk(fn)
    n = int(round(len(pts) ** (1.0 / 3.0)))
    assert n**3 == len(pts), f"not a cubic lattice: {len(pts)} points"

    # index = i + n*(j + n*k)  →  reshape to [k, j, i]
    x = pts[:, 0].reshape(n, n, n)
    y = pts[:, 1].reshape(n, n, n)
    u = data["velocity"][:, 0].reshape(n, n, n)
    v = data["velocity"][:, 1].reshape(n, n, n)
    rho = data["rho"].reshape(n, n, n)
    T = data["T"].reshape(n, n, n)

    k = n // 2  # z = L/2 slice (xy is the velocity-resolved plane)
    xs, ys = x[k, :, :], y[k, :, :]
    us, vs = u[k, :, :], v[k, :, :]
    sp = np.sqrt(us**2 + vs**2)
    rslice, Tslice = rho[k, :, :], T[k, :, :]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    im = axes[0].pcolormesh(xs, ys, rslice, cmap="RdBu_r",
                            shading="auto")
    axes[0].set_title(f"rho  [{rho.min():.4f}, {rho.max():.4f}]")
    plt.colorbar(im, ax=axes[0])

    im = axes[1].pcolormesh(xs, ys, Tslice, cmap="inferno",
                            shading="auto")
    axes[1].set_title(f"T  [{T.min():.2f}, {T.max():.2f}] K")
    plt.colorbar(im, ax=axes[1])

    im = axes[2].pcolormesh(xs, ys, sp, cmap="viridis", shading="auto")
    axes[2].streamplot(xs[0, :], ys[:, 0], us, vs, color="w",
                       density=1.2, linewidth=0.7, arrowsize=0.8)
    axes[2].set_title(f"|u| in xy at z=L/2, max={sp.max():.3f} m/s")
    plt.colorbar(im, ax=axes[2])

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.suptitle(f"3D driven cavity, xy slice at z=L/2, t={t:.3e} s")
    plt.tight_layout()
    out = os.path.join(outdir, "cavity3d_plot.png")
    plt.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
