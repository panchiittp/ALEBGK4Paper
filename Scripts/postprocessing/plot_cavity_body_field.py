#!/usr/bin/env python3
"""Reference-style moving-body panels (Tiwari-Klar / meshfree4bgk figures):
per panel a lid-driven-cavity velocity quiver (arrow length & colour = speed,
black at rest -> blue at speed) overlaid with the rigid-body centre
trajectory (green) and the body drawn at its final pose (filled red).

Each panel is one (VTK snapshot, bodies.csv, label) triple.  The VTK gives
the frozen velocity field at the panel time; bodies.csv gives the whole
centre path up to that time plus the final orientation.

Usage:
  python3 tools/plot_cavity_body_field.py out.png \
      "VTK=field_a.vtk,BODIES=bodies_a.csv,LABEL=Kn=0.01, t=1e-6" \
      "VTK=field_c.vtk,BODIES=bodies_c.csv,LABEL=Kn=0.01, t=5e-7" \
      ...
The panels tile into the squarest grid that fits (1, 2, 3->1x3, 4->2x2, ...).
Body half-size L (nm) and shape default to the 75 nm square; override per
panel with SIZE= and SHAPE=square|circle.
"""
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# black (at rest) -> royal blue (fast): mirrors the reference quiver look
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "rest2fast", ["#101010", "#1f3fd0", "#2b6bff"])


def read_vtk(path):
    """Return (t, x, y, ux, uy) from an ASCII VTK unstructured grid.

    t is parsed from the title line ("ALEBGK t=<time>"), so each panel
    knows which instant its flow field represents."""
    with open(path) as f:
        lines = f.read().split("\n")
    t = None
    if len(lines) > 1 and "t=" in lines[1]:
        t = float(lines[1].split("t=")[-1].strip())
    i = 0
    pts = None
    vel = None
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
        if tok and tok[0] == "VECTORS" and "velocity" in lines[i]:
            n = pts.shape[0]
            vals = []
            i += 1
            while len(vals) < 3 * n:
                vals.extend(float(v) for v in lines[i].split())
                i += 1
            vel = np.array(vals).reshape(-1, 3)
            continue
        i += 1
    return t, pts[:, 0], pts[:, 1], vel[:, 0], vel[:, 1]


def load_bodies(csv):
    return np.genfromtxt(csv, delimiter=",", names=True)


def parse_spec(spec):
    d = {}
    for part in spec.split(",VTK=")[0:1] + []:  # placeholder, real parse below
        pass
    # robust key=value parse allowing commas inside LABEL: split on known keys
    out = {"SIZE": 75.0, "SHAPE": "square", "LABEL": "", "GHOSTS": 3,
           "TIME": "", "GRID": 30, "DOMAIN": "", "STYLE": "stream"}
    # split by ",KEY=" boundaries
    import re
    tokens = re.split(
        r",(?=(?:VTK|BODIES|LABEL|SIZE|SHAPE|GHOSTS|TIME|GRID|DOMAIN|STYLE)=)",
        spec)
    for tkn in tokens:
        if "=" in tkn:
            k, v = tkn.split("=", 1)
            out[k.strip()] = v
    out["SIZE"] = float(out["SIZE"])
    out["GHOSTS"] = int(out["GHOSTS"])
    out["GRID"] = int(out["GRID"])
    return out


def draw_panel(ax, spec):
    t_panel, x, y, ux, uy = read_vtk(spec["VTK"])
    b = load_bodies(spec["BODIES"])

    # nondimensionalise positions to microns for axis labels (SI metres in)
    X, Y = x, y

    # Cavity box: explicit DOMAIN=x0:x1:y0:y1 override, else particle extent.
    if spec.get("DOMAIN"):
        x0, x1, y0, y1 = (float(v) for v in spec["DOMAIN"].split(":"))
        keep = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        X, Y, ux, uy = X[keep], Y[keep], ux[keep], uy[keep]
    else:
        x0, x1, y0, y1 = X.min(), X.max(), Y.min(), Y.max()

    # Sanity check: a healthy run keeps the cloud near its initial extent.
    # A diverged run (body/cloud blown out of the cavity) produces coordinates
    # orders of magnitude larger -- warn loudly instead of silently
    # autoscaling the axes into nonsense.
    b0 = np.atleast_1d(b["cx"])[0], np.atleast_1d(b["cy"])[0]
    ref = max(abs(b0[0]), abs(b0[1]), 1e-30)
    diverged = max(abs(x0), abs(x1), abs(y0), abs(y1)) > 100 * ref
    if diverged:
        print(f"WARNING [{spec['VTK']}]: particle extent "
              f"[{x0:.2e},{x1:.2e}]x[{y0:.2e},{y1:.2e}] is >100x the body "
              f"start scale {ref:.2e} -- this run has DIVERGED (stale data "
              f"from a pre-fix binary or an unstable configuration). "
              f"The panel is not physically meaningful.", file=sys.stderr)

    # Bin the scattered cloud onto a uniform G x G grid and average the
    # velocity per cell.  This gives a clean, regular arrow lattice
    # independent of particle ordering (the reference figures' look), while
    # arrow length still tracks local speed (long near the lid, short/dots
    # where the gas is at rest).
    G = int(spec.get("GRID", 30))
    xe = np.linspace(x0, x1, G + 1)
    ye = np.linspace(y0, y1, G + 1)
    cnt, _, _ = np.histogram2d(X, Y, bins=[xe, ye])
    sux, _, _ = np.histogram2d(X, Y, bins=[xe, ye], weights=ux)
    suy, _, _ = np.histogram2d(X, Y, bins=[xe, ye], weights=uy)
    with np.errstate(invalid="ignore"):
        gux = np.where(cnt > 0, sux / cnt, 0.0)
        guy = np.where(cnt > 0, suy / cnt, 0.0)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    GX, GY = np.meshgrid(xc, yc, indexing="ij")
    m = cnt > 0
    gspeed = np.hypot(gux, guy)
    smax = gspeed.max() if gspeed.max() > 0 else 1.0

    if spec["STYLE"] == "stream":
        # streamlines coloured by speed; empty cells (body interior) are
        # NaN so lines break there instead of crossing the body
        sux2 = np.where(m, gux, np.nan).T   # streamplot wants (ny, nx)
        suy2 = np.where(m, guy, np.nan).T
        ax.streamplot(xc, yc, sux2, suy2, color=np.hypot(sux2, suy2),
                      cmap=SPEED_CMAP, norm=plt.Normalize(0, smax),
                      density=1.3, linewidth=0.9,
                      arrowsize=0.8, zorder=2)
    else:
        # quiver: length prop to speed, coloured by speed (black -> blue)
        ax.quiver(GX[m], GY[m], gux[m], guy[m], gspeed[m],
                  cmap=SPEED_CMAP, clim=(0, smax), angles="xy",
                  scale_units="xy", scale=smax / (0.9 * (x1 - x0) / G),
                  width=0.004, headwidth=3.5, headlength=4, zorder=2,
                  alpha=0.95, pivot="mid")

    # ── Body history truncated at the panel time (JCP style) ───────────────
    # Each panel shows the path travelled SO FAR and the body AT this
    # instant, so a sequence of panels shows the body moving around the
    # vortex.  (Drawing the full history with the body at its last logged
    # position would make every panel identical.)
    tb = np.atleast_1d(b["t"])
    cxall = np.atleast_1d(b["cx"])
    cyall = np.atleast_1d(b["cy"])
    wall = (np.atleast_1d(b["angVel"]) if "angVel" in b.dtype.names
            else np.zeros_like(tb))
    if spec.get("TIME"):
        t_panel = float(spec["TIME"])
    if t_panel is None:
        t_panel = tb[-1]
    n_up = max(1, int(np.searchsorted(tb, t_panel * (1 + 1e-9), "right")))
    tt, cx, cy, w = tb[:n_up], cxall[:n_up], cyall[:n_up], wall[:n_up]

    # Divergence check #2: the body centre must stay inside the cavity box.
    pad = (x1 - x0) * 0.05
    if ((cx < x0 - pad) | (cx > x1 + pad) |
            (cy < y0 - pad) | (cy > y1 + pad)).any():
        if not diverged:
            print(f"WARNING [{spec['BODIES']}]: body trajectory leaves the "
                  f"cavity box -- this run has DIVERGED; the panel is not "
                  f"physically meaningful.", file=sys.stderr)
        diverged = True

    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))

    def pose(k):
        """Centre and integrated rotation angle up to history sample k."""
        a = _trap(w[:k + 1], tt[:k + 1]) if k > 0 else 0.0
        return cx[k], cy[k], a

    h = 0.5 * spec["SIZE"] * 1e-9  # full side/diameter in nm -> half in m

    def square_corners(xc, yc, a):
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, -s], [s, c]])
        return (np.array([[-h, -h], [h, -h], [h, h], [-h, h]]) @ R.T) + [xc, yc]

    # trajectory travelled so far (green) + start marker
    ax.plot(cx, cy, "-", color="#0b8f2a", lw=1.6, zorder=4)
    ax.scatter([cx[0]], [cy[0]], marker="o", s=40, facecolors="none",
               edgecolors="#0b8f2a", lw=1.2, zorder=4)

    # ghost outlines at intermediate times: the body's earlier poses
    ng = spec["GHOSTS"]
    if ng > 0 and n_up > 2:
        for frac in np.linspace(0.0, 1.0, ng + 2)[1:-1]:
            k = int(round(frac * (n_up - 1)))
            xg, yg, ag = pose(k)
            if spec["SHAPE"] == "circle":
                ax.add_patch(plt.Circle((xg, yg), h, facecolor="none",
                                        edgecolor="#e08080", lw=1.0,
                                        ls="--", zorder=4))
            else:
                ax.add_patch(plt.Polygon(square_corners(xg, yg, ag),
                                         closed=True, facecolor="none",
                                         edgecolor="#e08080", lw=1.0,
                                         ls="--", zorder=4))

    # body drawn at the panel-time pose (filled red)
    xf, yf, ang = pose(n_up - 1)
    if spec["SHAPE"] == "circle":
        ax.add_patch(plt.Circle((xf, yf), h, facecolor="#ff5a5a",
                                 edgecolor="#c00000", lw=1.2, alpha=0.85,
                                 zorder=5))
    else:
        cr = square_corners(xf, yf, ang)
        ax.add_patch(plt.Polygon(cr, closed=True, facecolor="#ff5a5a",
                                 edgecolor="#c00000", lw=1.2, alpha=0.85,
                                 zorder=5))
        # orientation diagonal (red line), like the reference marker
        ax.plot([cr[0, 0], cr[2, 0]], [cr[0, 1], cr[2, 1]],
                color="#c00000", lw=1.4, zorder=6)

    # cavity walls
    x0, x1 = X.min(), X.max()
    y0, y1 = Y.min(), Y.max()
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               ec="k", lw=1.4, zorder=3))
    ax.set_xlim(x0 - 0.02 * (x1 - x0), x1 + 0.02 * (x1 - x0))
    ax.set_ylim(y0 - 0.02 * (y1 - y0), y1 + 0.08 * (y1 - y0))
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
    ax.set_title(spec["LABEL"], fontsize=12)
    if diverged:
        ax.text(0.5, 0.5, "DIVERGED RUN\n(data not physical)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=16, color="#c00000", fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.8, ec="#c00000"),
                zorder=10)


def main():
    dst = sys.argv[1]
    specs = [parse_spec(s) for s in sys.argv[2:]]
    n = len(specs)
    ncols = 2 if n in (2, 4) else (1 if n == 1 else 3 if n == 3 else int(np.ceil(np.sqrt(n))))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 6.2 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    for k, spec in enumerate(specs):
        draw_panel(axes[k], spec)
    for k in range(n, len(axes)):
        axes[k].axis("off")
    plt.tight_layout()
    import os
    root, ext = os.path.splitext(dst)
    outs = [dst] if ext not in (".eps", ".png") else [root + ".eps",
                                                      root + ".png"]
    for o in outs:
        plt.savefig(o, dpi=140)
        print(f"saved {o}")
    print(f"{n} panel(s)")


if __name__ == "__main__":
    main()
