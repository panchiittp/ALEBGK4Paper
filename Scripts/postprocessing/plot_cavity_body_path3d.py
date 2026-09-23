#!/usr/bin/env python3
"""3D moving-body panels: the sphere's centre path inside the cubic cavity,
with the body drawn at each panel's time and the flow shown as a streamline
slice on the mid-height plane (the 3D analogue of the 2D figures).

Per panel: cavity wireframe, green centre trajectory truncated at the
panel time (read from the VTK header, or TIME= override), faded ghost
spheres at earlier poses, the sphere at its panel-time position, and the
velocity field of the y-normal mid plane rendered as a carpet of arrows on
that plane.

Usage:
  python3 postprocessing/plot_cavity_body_path3d.py out.png \
      "VTK=snapshotA.vtk,BODIES=bodies.csv,LABEL=t = 5e-8" \
      "VTK=snapshotB.vtk,BODIES=bodies.csv,LABEL=t = 1e-7" ...

Keys: SIZE (body diameter in nm, default 150 = 75 nm-radius sphere),
GHOSTS (default 3), TIME (override panel time), DOMAIN=x0:x1:y0:y1:z0:z1.
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SPEED_CMAP = LinearSegmentedColormap.from_list(
    "rest2fast", ["#101010", "#1f3fd0", "#2b6bff"])


def read_vtk(path):
    """Return (t, pts[N,3], vel[N,3])."""
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


def parse_spec(spec):
    out = {"SIZE": 150.0, "LABEL": "", "GHOSTS": 3, "TIME": "", "DOMAIN": "",
           "ZOOM": "0"}
    import re
    tokens = re.split(
        r",(?=(?:VTK|BODIES|LABEL|SIZE|GHOSTS|TIME|DOMAIN|ZOOM)=)", spec)
    for tkn in tokens:
        if "=" in tkn:
            k, v = tkn.split("=", 1)
            out[k.strip()] = v
    out["SIZE"] = float(out["SIZE"])
    out["GHOSTS"] = int(out["GHOSTS"])
    out["ZOOM"] = int(out["ZOOM"])
    return out


def sphere_mesh(xc, yc, zc, r, nu=14, nv=10):
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    X = xc + r * np.outer(np.cos(u), np.sin(v))
    Y = yc + r * np.outer(np.sin(u), np.sin(v))
    Z = zc + r * np.outer(np.ones_like(u), np.cos(v))
    return X, Y, Z


def draw_panel(ax, spec):
    t_panel, pts, vel = read_vtk(spec["VTK"])
    b = np.genfromtxt(spec["BODIES"], delimiter=",", names=True)

    if spec.get("DOMAIN"):
        x0, x1, y0, y1, z0, z1 = (float(v) for v in spec["DOMAIN"].split(":"))
    else:
        (x0, y0, z0), (x1, y1, z1) = pts.min(axis=0), pts.max(axis=0)

    # mid-height (y-normal) slice: bin the in-plane velocity to a G x G
    # grid and draw a sparse arrow carpet on that plane (skipped in ZOOM
    # panels, which focus on the trajectory)
    zoom = spec["ZOOM"]
    G = 16
    ymid = 0.5 * (y0 + y1)
    slab = np.abs(pts[:, 1] - ymid) < (y1 - y0) / 10
    px, pz = pts[slab, 0], pts[slab, 2]
    vx, vz = vel[slab, 0], vel[slab, 2]
    xe = np.linspace(x0, x1, G + 1)
    ze = np.linspace(z0, z1, G + 1)
    cnt, _, _ = np.histogram2d(px, pz, bins=[xe, ze])
    sx, _, _ = np.histogram2d(px, pz, bins=[xe, ze], weights=vx)
    sz, _, _ = np.histogram2d(px, pz, bins=[xe, ze], weights=vz)
    with np.errstate(invalid="ignore"):
        gx = np.where(cnt > 0, sx / np.maximum(cnt, 1), 0.0)
        gz = np.where(cnt > 0, sz / np.maximum(cnt, 1), 0.0)
    xc = 0.5 * (xe[:-1] + xe[1:])
    zc = 0.5 * (ze[:-1] + ze[1:])
    XC, ZC = np.meshgrid(xc, zc, indexing="ij")
    sp = np.hypot(gx, gz)
    smax = sp.max() if sp.max() > 0 else 1.0
    sc = 0.8 * (x1 - x0) / G / smax
    cmap = plt.get_cmap("rest2fast") if False else SPEED_CMAP
    m = cnt > 0
    if not zoom:
        for xi, zi, ui, wi, si in zip(XC[m], ZC[m], gx[m], gz[m], sp[m]):
            ax.plot([xi, xi + ui * sc], [ymid, ymid], [zi, zi + wi * sc],
                    color=cmap(si / smax), lw=0.9, alpha=0.85, zorder=1)

    # body history truncated at the panel time
    tb = np.atleast_1d(b["t"])
    cx = np.atleast_1d(b["cx"])
    cy = np.atleast_1d(b["cy"])
    cz = (np.atleast_1d(b["cz"]) if "cz" in b.dtype.names
          else np.full_like(cx, 0.5 * (z0 + z1)))
    if spec.get("TIME"):
        t_panel = float(spec["TIME"])
    if t_panel is None:
        t_panel = tb[-1]
    n_up = max(1, int(np.searchsorted(tb, t_panel * (1 + 1e-9), "right")))
    cx, cy, cz = cx[:n_up], cy[:n_up], cz[:n_up]

    r = 0.5 * spec["SIZE"] * 1e-9

    # trajectory + start marker
    ax.plot(cx, cy, cz, "-", color="#0b8f2a", lw=1.8, zorder=4)
    ax.scatter([cx[0]], [cy[0]], [cz[0]], marker="o", s=40,
               facecolors="none", edgecolors="#0b8f2a", lw=1.2)

    ng = spec["GHOSTS"]
    if zoom:
        # path-scale view: the body diameter dwarfs a short arc, so the
        # sphere becomes a marker and the trajectory carries the panel
        if ng > 0 and n_up > 2:
            for frac in np.linspace(0.0, 1.0, ng + 2)[1:-1]:
                k = int(round(frac * (n_up - 1)))
                ax.scatter([cx[k]], [cy[k]], [cz[k]], s=45,
                           facecolors="none", edgecolors="#e08080", lw=1.2)
        ax.scatter([cx[-1]], [cy[-1]], [cz[-1]], s=90, color="#ff5a5a",
                   edgecolors="#c00000", lw=1.2, zorder=5)
    else:
        # ghost spheres at earlier poses (wireframe)
        if ng > 0 and n_up > 2:
            for frac in np.linspace(0.0, 1.0, ng + 2)[1:-1]:
                k = int(round(frac * (n_up - 1)))
                X, Y, Z = sphere_mesh(cx[k], cy[k], cz[k], r)
                ax.plot_wireframe(X, Y, Z, color="#e08080", lw=0.4,
                                  rstride=2, cstride=2, alpha=0.6)

        # body at panel time
        X, Y, Z = sphere_mesh(cx[-1], cy[-1], cz[-1], r)
        ax.plot_surface(X, Y, Z, color="#ff5a5a", shade=True, alpha=0.95)

    # cavity wireframe
    for s0 in (x0, x1):
        for s1 in (y0, y1):
            ax.plot([s0, s0], [s1, s1], [z0, z1], "k-", lw=0.5, alpha=0.5)
    for s0 in (x0, x1):
        for s1 in (z0, z1):
            ax.plot([s0, s0], [y0, y1], [s1, s1], "k-", lw=0.5, alpha=0.5)
    for s0 in (y0, y1):
        for s1 in (z0, z1):
            ax.plot([x0, x1], [s0, s0], [s1, s1], "k-", lw=0.5, alpha=0.5)

    if zoom:
        # cubic box centred on and scaled to the trajectory itself
        ctr = np.array([0.5 * (cx.min() + cx.max()),
                        0.5 * (cy.min() + cy.max()),
                        0.5 * (cz.min() + cz.max())])
        span = max(cx.max() - cx.min(), cy.max() - cy.min(),
                   cz.max() - cz.min())
        half = max(span * 0.75, 1e-9)
        ax.set_xlim(ctr[0] - half, ctr[0] + half)
        ax.set_ylim(ctr[1] - half, ctr[1] + half)
        ax.set_zlim(ctr[2] - half, ctr[2] + half)
    else:
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(spec["LABEL"], fontsize=12)


def main():
    dst = sys.argv[1]
    specs = [parse_spec(s) for s in sys.argv[2:]]
    n = len(specs)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(7.2 * ncols, 6.6 * nrows))
    for k, spec in enumerate(specs):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection="3d")
        draw_panel(ax, spec)
    fig.tight_layout()
    fig.savefig(dst, dpi=140)
    print(f"saved {dst}  ({n} panel(s))")


if __name__ == "__main__":
    main()
