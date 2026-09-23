#!/usr/bin/env python3
"""Figures for the 3-D driven cavity with an immersed rigid body
(problem 11: static sphere, problem 12: freely moving/rotating cube).

Geometry of both problems (solver/main.cpp cases 11/12):
  cavity [0, 1] um^3, lid on the y = 1 um wall driven in +x with the
  parabolic profile u = 10 (2s)^2 (2 - 2s)^2 m/s, s = x/L.  The primary
  vortex therefore lives in the x-y plane; the z = 0.5 um mid-plane is
  the slice through the body centre.

Produces
  <case>_slices.png     z-midplane speed field + streamlines, one panel
                        per snapshot, body cross-section overlaid
  <case>_3d.png         cavity wireframe + body surface particles +
                        mid-plane arrow carpet
  cavity3d_body_profiles.png   centreline profiles, sphere vs cube
  cavity3d_body_history.png    diagnostics + rigid-body dynamics

Usage:
  python plot_cavity3d_body.py <sphere_dir> <cube_dir> <figdir>
"""
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull

# ── palette ──────────────────────────────────────────────────────────────
# categorical slots (fixed order, never cycled)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# sequential: one hue, light -> dark (magnitude)
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eaf2fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                 "#256abf", "#184f95", "#0d366b"])
# diverging: two hues + neutral gray midpoint (polarity)
DIV = LinearSegmentedColormap.from_list(
    "div_bo", ["#184f95", "#6da7ec", "#f0efec", "#f0a077", "#b8410f"])

L = 1e-6              # cavity side (m)
NM = 1e9              # m -> nm for axis labels
BODY_C = (6e-7, 7e-7, 5e-7)

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.grid": False,
})


# ── VTK reader ───────────────────────────────────────────────────────────
def _f(v):
    """Tolerant float parse. MSVC writes non-finite values as '-nan(ind)' /
    '1.#INF', which float() rejects outright -- so one diverged snapshot would
    otherwise abort a whole figure instead of showing up as NaN and being
    filtered by the finite-checks downstream."""
    try:
        return float(v)
    except ValueError:
        t = v.lower()
        if "nan" in t or "ind" in t:
            return float("nan")
        return float("-inf") if t.startswith("-") else float("inf")


def read_vtk(path):
    """Return dict with t, pts[N,3], vel[N,3] and the scalar fields."""
    with open(path) as fh:
        lines = fh.read().split("\n")
    t = float(lines[1].split("t=")[-1])
    out, i, npts = {"t": t}, 0, None
    while i < len(lines):
        tok = lines[i].split()
        if tok and tok[0] == "POINTS":
            npts = int(tok[1])
            vals = []
            i += 1
            while len(vals) < 3 * npts:
                vals.extend(_f(v) for v in lines[i].split())
                i += 1
            out["pts"] = np.array(vals).reshape(-1, 3)
            continue
        if tok and tok[0] == "SCALARS":
            name = tok[1]
            i += 2                      # skip LOOKUP_TABLE
            vals = []
            while len(vals) < npts:
                vals.extend(_f(v) for v in lines[i].split())
                i += 1
            out[name] = np.array(vals)
            continue
        if tok and tok[0] == "VECTORS":
            name = tok[1]
            vals = []
            i += 1
            while len(vals) < 3 * npts:
                vals.extend(_f(v) for v in lines[i].split())
                i += 1
            out[name] = np.array(vals).reshape(-1, 3)
            continue
        i += 1
    return out


def body_mask(d, tol=1e-9):
    """Surface particles of the immersed body: flagged boundary, but not
    lying on any of the six cavity faces."""
    p = d["pts"]
    on_wall = np.zeros(len(p), bool)
    for k in range(3):
        on_wall |= (np.abs(p[:, k]) < tol) | (np.abs(p[:, k] - L) < tol)
    return (d["boundary"] > 0.5) & ~on_wall


def snapshots(outdir):
    return sorted(glob.glob(os.path.join(outdir, "output_*.vtk")))


# ── z mid-plane slice, interpolated to a regular grid ────────────────────
def midplane(d, zc=0.5 * L, half=1.2e-8, G=90):
    p, v = d["pts"], d["velocity"]
    m = np.abs(p[:, 2] - zc) < half
    xi = np.linspace(0, L, G)
    yi = np.linspace(0, L, G)
    X, Y = np.meshgrid(xi, yi)
    pts2 = p[m][:, :2]
    def grid(q):
        return griddata(pts2, q, (X, Y), method="linear")
    U, V = grid(v[m, 0]), grid(v[m, 1])
    T = grid(d["T"][m])
    return X, Y, U, V, T


def draw_body_xy(ax, kind, d=None):
    """Body cross-section on the z-midplane."""
    cx, cy = BODY_C[0] * NM, BODY_C[1] * NM
    if kind == "sphere":
        ax.add_patch(plt.Circle((cx, cy), 75.0, facecolor=C_ORANGE,
                                edgecolor=INK, lw=1.0, zorder=6))
    else:
        # true pose: project the surface particles near the mid-plane
        m = body_mask(d)
        bp = d["pts"][m]
        near = np.abs(bp[:, 2] - BODY_C[2]) < 1.0e-7
        ax.scatter(bp[near, 0] * NM, bp[near, 1] * NM, s=9, color=C_ORANGE,
                   zorder=6, linewidths=.3, edgecolors=INK)


# ── figure 1: mid-plane slice panels ─────────────────────────────────────
def fig_slices(outdir, kind, title, dst):
    files = snapshots(outdir)
    n = len(files)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.5 * nrow))
    axes = np.atleast_1d(axes).ravel()

    # common speed scale across panels
    fields = [midplane(read_vtk(f)) for f in files]
    smax = max(np.nanmax(np.hypot(U, V)) for _, _, U, V, _ in fields)

    for k, (fn, (X, Y, U, V, T)) in enumerate(zip(files, fields)):
        d = read_vtk(fn)
        ax = axes[k]
        S = np.hypot(U, V)
        # gamma < 1: the lid runs at 10 m/s while the recirculating core is
        # ~0.1 m/s, so a linear ramp would render the whole interior white
        pc = ax.pcolormesh(X * NM, Y * NM, S, cmap=SEQ,
                           norm=PowerNorm(0.45, vmin=0, vmax=smax),
                           shading="gouraud", rasterized=True)
        with np.errstate(invalid="ignore"):
            ax.streamplot(X * NM, Y * NM, U, V, color=INK, linewidth=0.6,
                          density=1.1, arrowsize=0.7)
        draw_body_xy(ax, kind, d)
        ax.set_xlim(0, L * NM); ax.set_ylim(0, L * NM)
        ax.set_aspect("equal")
        ax.set_title(f"t = {d['t'] * 1e9:.2f} ns", fontsize=9, pad=4)
        if k % ncol == 0:
            ax.set_ylabel("y (nm)")
        if k // ncol == nrow - 1:
            ax.set_xlabel("x (nm)")
    for ax in axes[n:]:
        ax.axis("off")

    ticks = [t for t in (0, 0.25, 0.5, 1, 2, 4, 6, 8, 10) if t <= smax]
    cb = fig.colorbar(pc, ax=axes.tolist(), fraction=0.025, pad=0.02,
                      ticks=ticks)
    cb.set_label("speed |u| (m/s)", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(title, fontsize=12, color=INK, y=0.985)
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", dst)


# ── figure 2: 3-D view ───────────────────────────────────────────────────
def fig_3d(outdir, kind, title, dst):
    """Cavity wireframe + the z-midplane speed field rendered as a coloured
    surface + the body reconstructed as the convex hull of its own surface
    particles (so the cube shows its true, rotated pose)."""
    d = read_vtk(snapshots(outdir)[-1])
    fig = plt.figure(figsize=(7.6, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("persp")
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_visible(False)
        pane._axinfo["grid"]["color"] = GRID
        pane._axinfo["grid"]["linewidth"] = 0.4

    # cavity wireframe
    for a in (0, L * NM):
        for b in (0, L * NM):
            ax.plot([a] * 2, [b] * 2, [0, L * NM], color=MUTED, lw=.7)
            ax.plot([a] * 2, [0, L * NM], [b] * 2, color=MUTED, lw=.7)
            ax.plot([0, L * NM], [a] * 2, [b] * 2, color=MUTED, lw=.7)

    # z = L/2 plane as a coloured surface (speed), semi-transparent
    X, Y, U, V, _ = midplane(d, G=70)
    S = np.hypot(U, V)
    smax = np.nanmax(S)
    norm = PowerNorm(0.45, vmin=0, vmax=smax)
    Sfill = np.nan_to_num(S)
    surf = ax.plot_surface(X * NM, Y * NM, np.full_like(X, 0.5 * L * NM),
                           facecolors=SEQ(norm(Sfill)), rstride=1, cstride=1,
                           shade=False, alpha=1.0, linewidth=0,
                           antialiased=False)
    # matplotlib's 3-D painter sorts by centroid depth; the plane and the
    # body share z = 500 nm, so pin the draw order explicitly.
    surf.set_sort_zpos(-1e3)

    # in-plane quiver on the same plane
    Xq, Yq, Uq, Vq, _ = midplane(d, G=15)
    Sq = np.hypot(Uq, Vq)
    ok = np.isfinite(Sq) & (Sq > 0)
    ln = 0.9 * (L * NM / 15)
    qv = ax.quiver(Xq[ok] * NM, Yq[ok] * NM,
                   np.full(ok.sum(), 0.5 * L * NM + 4),
                   Uq[ok] / Sq[ok], Vq[ok] / Sq[ok], np.zeros(ok.sum()),
                   length=ln, color=INK, alpha=0.55, linewidth=0.7,
                   arrow_length_ratio=0.35)
    qv.set_sort_zpos(-9e2)

    # lid: parabolic profile drawn on the y = L wall
    xs = np.linspace(0.02, 0.98, 60) * L * NM
    ul = 10.0 * (2 * xs / (L * NM)) ** 2 * (2 - 2 * xs / (L * NM)) ** 2
    for zc in (0.02, 0.5, 0.98):
        ax.plot(xs, np.full_like(xs, L * NM), np.full_like(xs, zc * L * NM),
                color=MUTED, lw=.7, alpha=.5)
    ax.plot(xs, np.full_like(xs, L * NM), 0.5 * L * NM + ul * 18,
            color=C_ORANGE, lw=2.0, zorder=6)
    for k in range(4, 60, 11):
        ax.quiver(xs[k], L * NM, 0.5 * L * NM + ul[k] * 18, 1, 0, 0,
                  length=70, color=C_ORANGE, linewidth=1.4,
                  arrow_length_ratio=0.4, zorder=6)

    # body: convex hull of its surface particles (true pose)
    bp = d["pts"][body_mask(d)] * NM
    hull = ConvexHull(bp)
    body = ax.plot_trisurf(bp[:, 0], bp[:, 1], bp[:, 2],
                           triangles=hull.simplices, color=C_ORANGE,
                           edgecolor=INK, linewidth=0.3, shade=True)
    body.set_sort_zpos(1e4)

    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)"); ax.set_zlabel("z (nm)")
    ax.set_xlim(0, L * NM); ax.set_ylim(0, L * NM); ax.set_zlim(0, L * NM)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=26, azim=-62)
    ax.set_title(f"{title}   —   t = {d['t']*1e9:.2f} ns\n"
                 f"z = 500 nm plane coloured by |u| (max {smax:.1f} m/s)\n"
                 f"orange: lid profile $u_x(x)$ as height over the "
                 f"y = 1000 nm wall", fontsize=10, pad=0)
    fig.tight_layout()
    fig.savefig(dst, dpi=150)
    plt.close(fig)
    print("saved", dst)


# ── figure 3: centreline profiles, sphere vs cube ────────────────────────
def line_profile(d, axis, fixed, comp, half=1.2e-8, n=60):
    """Interpolate a velocity component along a cavity centreline on the
    z-midplane. axis 0 -> vary x at y = fixed; axis 1 -> vary y at x."""
    p, v = d["pts"], d["velocity"]
    m = np.abs(p[:, 2] - 0.5 * L) < half
    pts2, q = p[m][:, :2], v[m, comp]
    s = np.linspace(0, L, n)
    if axis == 0:
        tgt = np.column_stack([s, np.full_like(s, fixed)])
    else:
        tgt = np.column_stack([np.full_like(s, fixed), s])
    return s, griddata(pts2, q, tgt, method="linear")


def fig_profiles(dirs, dst):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    styles = [("sphere (static, prob. 11)", C_BLUE, "-"),
              ("cube (free, prob. 12)", C_ORANGE, "--")]
    for (lbl, col, ls), outdir in zip(styles, dirs):
        d = read_vtk(snapshots(outdir)[-1])
        y, ux = line_profile(d, 1, 0.5 * L, 0)
        x, uy = line_profile(d, 0, 0.5 * L, 1)
        axes[0].plot(ux, y * NM, ls, color=col, lw=2.0, label=lbl)
        axes[1].plot(x * NM, uy, ls, color=col, lw=2.0, label=lbl)
    axes[0].set_xlabel(r"$u_x$ (m/s)"); axes[0].set_ylabel("y (nm)")
    axes[0].set_title("vertical centreline  x = 500 nm,  z = 500 nm")
    axes[1].set_xlabel("x (nm)"); axes[1].set_ylabel(r"$u_y$ (m/s)")
    axes[1].set_title("horizontal centreline  y = 500 nm,  z = 500 nm")
    for ax in axes:
        ax.axhline(0, color=GRID, lw=.8) if ax is axes[1] else \
            ax.axvline(0, color=GRID, lw=.8)
        ax.grid(True, color=GRID, lw=.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("3-D cavity centreline profiles at t = 7.5 ns "
                 "(z = 500 nm mid-plane)", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(dst, dpi=150)
    plt.close(fig)
    print("saved", dst)


# ── figure 4: diagnostics + rigid-body dynamics ──────────────────────────
def fig_history(sph_dir, cub_dir, dst):
    ts_s = np.genfromtxt(os.path.join(sph_dir, "timeseries.csv"),
                         delimiter=",", names=True)
    ts_c = np.genfromtxt(os.path.join(cub_dir, "timeseries.csv"),
                         delimiter=",", names=True)
    b = np.genfromtxt(os.path.join(cub_dir, "bodies.csv"),
                      delimiter=",", names=True)

    fig, ax = plt.subplots(2, 2, figsize=(9.8, 7.0))

    a = ax[0, 0]
    a.plot(ts_s["t"] * 1e9, ts_s["KE"], "-", color=C_BLUE, lw=2.0,
           label="sphere (static)")
    a.plot(ts_c["t"] * 1e9, ts_c["KE"], "--", color=C_ORANGE, lw=2.0,
           label="cube (free)")
    a.set_ylabel("mean kinetic energy  (J/m³)")
    a.set_title("cavity spin-up to steady state")

    a = ax[0, 1]
    a.plot(ts_s["t"] * 1e9, ts_s["T_avg"] - 270.0, "-", color=C_BLUE, lw=2.0,
           label="sphere (static)")
    a.plot(ts_c["t"] * 1e9, ts_c["T_avg"] - 270.0, "--", color=C_ORANGE,
           lw=2.0, label="cube (free)")
    a.set_ylabel(r"$\langle T\rangle - T_w$  (K)")
    a.set_title("thermal drift from the 270 K wall value")

    a = ax[1, 0]
    t = b["t"] * 1e9
    a.plot(t, (b["cx"] - BODY_C[0]) * NM, "-", color=C_BLUE, lw=2.0,
           label=r"$\Delta x$")
    a.plot(t, (b["cy"] - BODY_C[1]) * NM, "--", color=C_ORANGE, lw=2.0,
           label=r"$\Delta y$")
    a.plot(t, (b["cz"] - BODY_C[2]) * NM, ":", color=C_AQUA, lw=2.0,
           label=r"$\Delta z$")
    a.set_ylabel("cube centre displacement (nm)")
    a.set_title("cube translation under the lid-driven vortex")

    a = ax[1, 1]
    a.plot(t, b["omx"] * 1e-6, "-", color=C_BLUE, lw=2.0, label=r"$\omega_x$")
    a.plot(t, b["omy"] * 1e-6, "--", color=C_ORANGE, lw=2.0,
           label=r"$\omega_y$")
    a.plot(t, b["angVel"] * 1e-6, ":", color=C_AQUA, lw=2.0,
           label=r"$\omega_z$")
    a.set_ylabel(r"angular velocity  ($10^6$ rad/s)")
    a.set_title("cube spin-up (Newton-Euler, Rodrigues update)")

    for a in ax.ravel():
        a.set_xlabel("t (ns)")
        a.grid(True, color=GRID, lw=.6)
        a.set_axisbelow(True)
        a.legend(frameon=False, fontsize=8)
    fig.suptitle("3-D driven cavity with immersed rigid body — histories",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(dst, dpi=150)
    plt.close(fig)
    print("saved", dst)


def main():
    sph, cub, figdir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(figdir, exist_ok=True)
    j = lambda f: os.path.join(figdir, f)
    fig_slices(sph, "sphere",
               "3-D driven cavity + static sphere (R = 75 nm) — "
               "z = 500 nm mid-plane", j("cavity3d_sphere_slices.png"))
    fig_slices(cub, "cube",
               "3-D driven cavity + free cube (L = 150 nm) — "
               "z = 500 nm mid-plane", j("cavity3d_cube_slices.png"))
    fig_3d(sph, "sphere", "3-D driven cavity — static sphere",
           j("cavity3d_sphere_3d.png"))
    fig_3d(cub, "cube", "3-D driven cavity — freely moving/rotating cube",
           j("cavity3d_cube_3d.png"))
    fig_profiles([sph, cub], j("cavity3d_body_profiles.png"))
    fig_history(sph, cub, j("cavity3d_body_history.png"))


if __name__ == "__main__":
    main()
