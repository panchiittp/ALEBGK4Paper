#!/usr/bin/env python3
"""Pose-resolved figures for the 3-D driven cavity with a free rigid body.

Renders the body as a REAL solid (a true cube, a high-resolution chequered
sphere) at the exact pose recovered from its own surface particles, traces
the centre path, and tracks body-fixed surface markers so the rotation is
readable in a still frame.

Usage:
  python plot_cavity3d_pose.py <sphere_dir> <cube_dir> <figdir> [ref2d_dir]
"""
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_cavity3d_body import read_vtk, midplane, SEQ, C_BLUE, C_ORANGE, \
    C_AQUA, C_YELLOW, INK, INK2, MUTED, GRID, L, NM
from body_pose import body_points, kabsch, rot_angle, cube_faces, \
    cube_section, sphere_mesh, sphere_facecolors, visible

CUBE_L = 1.5e-7          # cube side  (main.cpp case 12, 3-D)
SPH_R = 7.5e-8           # sphere radius (main.cpp case 11/12, 3-D)
ELEV, AZIM = 24, -60

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})


# ── pose track ───────────────────────────────────────────────────────────
def pose_track(outdir):
    """Per-snapshot (t, centre, rotation matrix) from the surface particles.

    The reference shape A is taken from whichever point-count is the
    MAJORITY across snapshots, not simply the first file: a directory can
    carry one stale/mismatched snapshot (e.g. a leftover from an earlier
    run that wrote the same filename with a different body), and using it
    as the Kabsch reference would silently drop every real frame instead
    of just the bad one."""
    files = sorted(glob.glob(os.path.join(outdir, "output_*.vtk")))
    loaded = [(fn, read_vtk(fn)) for fn in files]
    bodies = [(fn, d, body_points(d)[0]) for fn, d in loaded]
    counts = {}
    for _, _, B in bodies:
        counts[B.shape[0]] = counts.get(B.shape[0], 0) + 1
    majority = max(counts, key=counts.get)
    ref_fn, ref_d, A = next(b for b in bodies if b[2].shape[0] == majority)

    out = []
    worst = 0.0
    skipped = []
    for fn, d, B in bodies:
        if B.shape != A.shape:
            skipped.append((fn, B.shape[0]))
            continue
        R, cA, cB, rmsd = kabsch(A, B)
        worst = max(worst, rmsd)
        out.append({"t": d["t"], "c": cB, "R": R, "file": fn, "vtk": d})
    if skipped:
        print(f"  pose_track({outdir}): skipped {len(skipped)} mismatched "
              f"snapshot(s) (expected {majority} body points): "
              + ", ".join(f"{os.path.basename(f)}={n}" for f, n in skipped))
    out.sort(key=lambda p: p["t"])
    return out, A - A.mean(0), worst


# ── body renderers ───────────────────────────────────────────────────────
def draw_cube(ax, c, R, alpha=1.0, face_hi=0, edge=INK, lw=0.8, base=C_ORANGE):
    faces, verts = cube_faces(c * NM, R, CUBE_L * NM)
    cols = [base] * 6
    cols[face_hi] = C_YELLOW              # one painted face -> spin is visible
    pc = Poly3DCollection(faces, facecolors=cols, edgecolors=edge,
                          linewidths=lw, alpha=alpha)
    ax.add_collection3d(pc)
    pc.set_sort_zpos(1e4)
    return verts


def draw_sphere(ax, c, R, alpha=1.0, nu=96, nv=48):
    w, n, TH, PH = sphere_mesh(c * NM, R, SPH_R * NM, nu, nv)
    fc = sphere_facecolors(TH, PH,
                           matplotlib.colors.to_rgba(C_ORANGE),
                           matplotlib.colors.to_rgba("#f7d9c4"))
    s = ax.plot_surface(w[..., 0], w[..., 1], w[..., 2], facecolors=fc,
                        rstride=1, cstride=1, linewidth=0, antialiased=False,
                        shade=False, alpha=alpha)
    s.set_sort_zpos(1e4)
    return s


# body-frame marker directions: the three body axes
MARKERS = [(np.array([1.0, 0, 0]), C_BLUE, "+x"),
           (np.array([0, 1.0, 0]), C_AQUA, "+y"),
           (np.array([0, 0, 1.0]), "#4a3aa7", "+z")]


def draw_markers(ax, c, R, r, size=70, cull=True):
    for b, col, lab in MARKERS:
        w = c * NM + (R @ b) * r * NM
        n = (R @ b)[None, :]
        if cull and not visible(n, ELEV, AZIM)[0]:
            continue
        sc = ax.scatter([w[0]], [w[1]], [w[2]], s=size, color=col,
                        edgecolors=INK, linewidths=0.6, depthshade=False)
        sc.set_sort_zpos(2e4)


def cavity_box(ax):
    for a in (0, L * NM):
        for b in (0, L * NM):
            ax.plot([a] * 2, [b] * 2, [0, L * NM], color=MUTED, lw=.6)
            ax.plot([a] * 2, [0, L * NM], [b] * 2, color=MUTED, lw=.6)
            ax.plot([0, L * NM], [a] * 2, [b] * 2, color=MUTED, lw=.6)


def style3d(ax):
    ax.set_proj_type("persp")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis._axinfo["grid"]["color"] = GRID
        axis._axinfo["grid"]["linewidth"] = 0.4
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=ELEV, azim=AZIM)


# ── figure: pose panels with the real body + path so far ────────────────
def fig_pose_panels(track, shape, title, dst, npanel=6):
    idx = np.unique(np.linspace(0, len(track) - 1, npanel).round().astype(int))
    ncol = 3
    nrow = int(np.ceil(len(idx) / ncol))
    fig = plt.figure(figsize=(4.6 * ncol, 4.5 * nrow))
    path = np.array([p["c"] for p in track]) * NM

    for k, i in enumerate(idx):
        ax = fig.add_subplot(nrow, ncol, k + 1, projection="3d")
        style3d(ax)
        cavity_box(ax)
        p = track[i]

        # mid-plane speed carpet, faint, for context
        X, Y, U, V, _ = midplane(p["vtk"], G=60)
        S = np.nan_to_num(np.hypot(U, V))
        norm = PowerNorm(0.45, vmin=0, vmax=max(S.max(), 1e-12))
        surf = ax.plot_surface(X * NM, Y * NM, np.full_like(X, 0.5 * L * NM),
                               facecolors=SEQ(norm(S)), rstride=2, cstride=2,
                               shade=False, linewidth=0, antialiased=False,
                               alpha=0.9)
        surf.set_sort_zpos(-1e3)

        # centre path up to this panel
        ax.plot(path[:i + 1, 0], path[:i + 1, 1], path[:i + 1, 2],
               color="#0b8f2a", lw=2.0)

        if shape == "cube":
            draw_cube(ax, p["c"], p["R"])
        else:
            draw_sphere(ax, p["c"], p["R"])
        draw_markers(ax, p["c"], p["R"],
                     CUBE_L * 0.87 if shape == "cube" else SPH_R)

        ax.set_xlim(0, L * NM); ax.set_ylim(0, L * NM); ax.set_zlim(0, L * NM)
        ax.set_xlabel("x (nm)", labelpad=-4)
        ax.set_ylabel("y (nm)", labelpad=-4)
        ax.set_zlabel("z (nm)", labelpad=-4)
        ax.tick_params(labelsize=7, pad=-2)
        ax.set_title(f"t = {p['t'] * 1e9:.1f} ns", fontsize=10, pad=-6)

    fig.suptitle(title, fontsize=13, color=INK, y=0.995)
    fig.tight_layout()
    fig.savefig(dst, dpi=140)
    plt.close(fig)
    print("saved", dst)


# ── figure: marker tracks (how the body-fixed points move) ──────────────
def fig_markers(track, shape, title, dst):
    r = (CUBE_L * 0.87 if shape == "cube" else SPH_R)
    t = np.array([p["t"] for p in track]) * 1e9
    c = np.array([p["c"] for p in track])
    ang = np.array([rot_angle(p["R"]) for p in track])

    # cumulative rotation angle: unwrap the per-step increment, which stays
    # small, instead of the raw arccos which folds back at pi
    dth = [0.0]
    for k in range(1, len(track)):
        dR = track[k]["R"] @ track[k - 1]["R"].T
        dth.append(dth[-1] + rot_angle(dR))
    dth = np.array(dth)

    fig = plt.figure(figsize=(13.5, 4.6))

    # (a) marker paths in the body-translating frame -> pure rotation
    ax = fig.add_subplot(131, projection="3d")
    style3d(ax)
    if shape == "sphere":
        draw_sphere(ax, np.zeros(3), track[-1]["R"], alpha=1.0, nu=72, nv=36)
    else:
        draw_cube(ax, np.zeros(3), track[-1]["R"])
    for b, col, lab in MARKERS:
        w = np.array([p["R"] @ b for p in track]) * r * NM
        ax.plot(w[:, 0], w[:, 1], w[:, 2], color=col, lw=1.6, label=lab)
        sc = ax.scatter([w[-1, 0]], [w[-1, 1]], [w[-1, 2]], s=55, color=col,
                        edgecolors=INK, linewidths=.6, depthshade=False)
        sc.set_sort_zpos(4e4)
    lim = r * NM * 1.45
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)"); ax.set_zlabel("z (nm)")
    ax.set_title("body-fixed markers, centre frame\n(pure rotation)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # (b) marker paths in the LAB frame -> rotation + translation
    ax = fig.add_subplot(132, projection="3d")
    style3d(ax)
    for b, col, lab in MARKERS:
        w = (c + np.array([p["R"] @ b for p in track]) * r) * NM
        ax.plot(w[:, 0], w[:, 1], w[:, 2], color=col, lw=1.3, label=lab)
    ax.plot(c[:, 0] * NM, c[:, 1] * NM, c[:, 2] * NM, color="#0b8f2a", lw=2.2,
            label="centre")
    ax.scatter([c[0, 0] * NM], [c[0, 1] * NM], [c[0, 2] * NM], s=45,
               facecolors="none", edgecolors="#0b8f2a", lw=1.3)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)"); ax.set_zlabel("z (nm)")
    ax.set_title("same markers, lab frame\n(rotation + translation)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # (c) cumulative rotation angle and marker coordinates
    ax = fig.add_subplot(133)
    ax.plot(t, np.degrees(dth), color=INK, lw=2.2,
            label="cumulative rotation")
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("total rotation angle (deg)")
    ax.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)
    ax2 = ax.twiny()
    ax2.set_visible(False)
    b0 = MARKERS[2][0]
    zc = np.array([(p["R"] @ b0)[2] for p in track])
    axr = ax.inset_axes([0.55, 0.12, 0.42, 0.35])
    axr.plot(t, zc, color=MARKERS[2][1], lw=1.6)
    axr.set_title(r"marker $+z$: $\hat{n}_z$", fontsize=7)
    axr.tick_params(labelsize=6)
    axr.grid(True, color=GRID, lw=.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("rotation history", fontsize=10)

    fig.suptitle(title, fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(dst, dpi=145)
    plt.close(fig)
    print("saved", dst)


# ── figure: centre paths of both bodies, zoomed ─────────────────────────
def fig_paths(tracks, labels, dst):
    fig = plt.figure(figsize=(12.5, 5.0))
    for k, (tr, lab) in enumerate(zip(tracks, labels)):
        ax = fig.add_subplot(1, len(tracks), k + 1, projection="3d")
        style3d(ax)
        c = np.array([p["c"] for p in tr]) * NM
        t = np.array([p["t"] for p in tr]) * 1e9
        for i in range(len(c) - 1):
            ax.plot(c[i:i + 2, 0], c[i:i + 2, 1], c[i:i + 2, 2],
                    color=SEQ(0.25 + 0.75 * i / max(len(c) - 2, 1)), lw=2.4)
        ax.scatter([c[0, 0]], [c[0, 1]], [c[0, 2]], s=55, facecolors="none",
                   edgecolors=INK, lw=1.2)
        ax.scatter([c[-1, 0]], [c[-1, 1]], [c[-1, 2]], s=70, color=C_ORANGE,
                   edgecolors=INK, lw=.8, depthshade=False)
        ctr, span = c.mean(0), max(c.max(0) - c.min(0))
        h = max(span * 0.75, 0.4)
        ax.set_xlim(ctr[0] - h, ctr[0] + h)
        ax.set_ylim(ctr[1] - h, ctr[1] + h)
        ax.set_zlim(ctr[2] - h, ctr[2] + h)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)"); ax.set_zlabel("z (nm)")
        ax.set_title(f"{lab}\ntotal excursion "
                     f"{np.linalg.norm(c[-1] - c[0]):.2f} nm  "
                     f"(open circle = start, {t[-1]:.0f} ns)", fontsize=10)
    fig.suptitle("Rigid-body centre paths, coloured by time "
                 "(light = early, dark = late)", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(dst, dpi=145)
    plt.close(fig)
    print("saved", dst)


# ── vortex core ─────────────────────────────────────────────────────────
def vortex_core(d, plane="z", G=140, margin=0.12):
    """Core of the primary vortex on the mid-plane: the interior minimum of
    the in-plane speed (the stagnation point the closed streamlines wrap)."""
    if plane == "z":
        X, Y, U, V, _ = midplane(d, G=G)
    else:                                  # 2-D snapshot: use the cloud itself
        p, v = d["pts"], d["velocity"]
        xi = np.linspace(0, L, G)
        X, Y = np.meshgrid(xi, xi)
        U = griddata(p[:, :2], v[:, 0], (X, Y), method="linear")
        V = griddata(p[:, :2], v[:, 1], (X, Y), method="linear")
    S = np.hypot(U, V)
    m = ((X > margin * L) & (X < (1 - margin) * L) &
         (Y > margin * L) & (Y < (1 - margin) * L) & np.isfinite(S))
    Sm = np.where(m, S, np.inf)
    j, i = np.unravel_index(np.argmin(Sm), Sm.shape)
    return X[j, i], Y[j, i]


def fig_vortex_core(series, dst):
    """series: list of (label, colour, dash, list-of-(t, x, y))"""
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for lab, col, ls, pts in series:
        t = np.array([p[0] for p in pts]) * 1e9
        y = np.array([p[2] for p in pts]) * NM
        x = np.array([p[1] for p in pts]) * NM
        ax[0].plot(t, y, ls, color=col, lw=2.0, label=lab)
        ax[1].plot(x, y, ls, color=col, lw=1.6, label=lab)
        ax[1].scatter([x[-1]], [y[-1]], s=45, color=col, edgecolors=INK,
                      lw=.6, zorder=5)
    body = plt.Circle((600, 700), 75, facecolor="none", edgecolor=MUTED,
                      lw=1.2, ls=(0, (3, 2)))
    ax[1].add_patch(body)
    ax[1].annotate("body sits here\n(600, 700) nm", (600, 700),
                   xytext=(760, 780), fontsize=7.5, color=INK2,
                   ha="left",
                   arrowprops=dict(arrowstyle="-", color=MUTED, lw=.8))
    for a in ax:
        a.axhline(500, color=C_ORANGE, lw=1.2, ls=(0, (4, 3)))
        a.grid(True, color=GRID, lw=.6); a.set_axisbelow(True)
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    ax[1].legend(frameon=False, fontsize=8, loc="lower left")
    ax[0].set_xlabel("t (ns)"); ax[0].set_ylabel("vortex core height y (nm)")
    ax[0].set_title("core height vs time\n"
                    "(orange dashed = cavity mid-height, y = 500 nm)",
                    fontsize=10)
    ax[1].set_xlabel("core x (nm)"); ax[1].set_ylabel("core y (nm)")
    ax[1].set_xlim(0, 1000); ax[1].set_ylim(0, 1000)
    ax[1].set_aspect("equal")
    ax[1].set_title("core position in the cavity\n(dashed circle = body "
                    "footprint)", fontsize=10)
    fig.suptitle("Why the 3-D vortex core sits below the 2-D reference's "
                 "position", fontsize=12.5, color=INK)
    note = ("2-D reference (Kn=0.013): core near (568, 669) nm — close to "
            "the natural Stokes-flow position, mildly pulled down by the "
            "body.\n"
            "3-D case (Kn=0.127, the solver's only verified-stable 3-D "
            "envelope): two effects push the core further down —\n"
            "  (1) front/back end-wall drag, absent in 2-D, already pulls "
            "the plain 3-D cavity's core down to y ≈ 580 nm at matched Kn;\n"
            "  (2) the body sits almost exactly where the natural vortex "
            "core would form, physically displacing it.\n"
            "Matching the 2-D Kn in 3-D needs roughly a 200³ grid "
            "(memory-infeasible here); reducing Kn at the current grid "
            "(tested at Nx=21 and Nx=31) destabilised the solver both times.")
    fig.text(0.02, -0.02, note, fontsize=7.8, color=INK2, va="top",
             linespacing=1.5)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(dst, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print("saved", dst)


# ── slices with the exact body cross-section ────────────────────────────
def fig_slices_pose(track, shape, title, dst, npanel=6):
    idx = np.unique(np.linspace(0, len(track) - 1, npanel).round().astype(int))
    ncol = 3
    nrow = int(np.ceil(len(idx) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    flds = [midplane(track[i]["vtk"]) for i in idx]
    smax = max(np.nanmax(np.hypot(U, V)) for _, _, U, V, _ in flds)

    for k, (i, (X, Y, U, V, _)) in enumerate(zip(idx, flds)):
        ax = axes[k]
        p = track[i]
        S = np.hypot(U, V)
        pc = ax.pcolormesh(X * NM, Y * NM, S, cmap=SEQ,
                           norm=PowerNorm(0.45, vmin=0, vmax=smax),
                           shading="gouraud", rasterized=True)
        with np.errstate(invalid="ignore"):
            ax.streamplot(X * NM, Y * NM, U, V, color=INK, linewidth=0.6,
                          density=1.1, arrowsize=0.7)
        # exact cross-section of the body at the plane through its centre
        z0 = p["c"][2]
        if shape == "cube":
            poly = cube_section(p["c"], p["R"], CUBE_L, z0)
            if poly is not None:
                ax.add_patch(plt.Polygon(poly * NM, closed=True,
                                         facecolor=C_ORANGE, edgecolor=INK,
                                         lw=1.0, zorder=6))
        else:
            ax.add_patch(plt.Circle((p["c"][0] * NM, p["c"][1] * NM),
                                    SPH_R * NM, facecolor=C_ORANGE,
                                    edgecolor=INK, lw=1.0, zorder=6))
            for b, col, _lab in MARKERS:              # in-plane markers
                w = p["c"] + (p["R"] @ b) * SPH_R
                if abs(w[2] - z0) < SPH_R * 0.55:
                    ax.plot([w[0] * NM], [w[1] * NM], "o", ms=4.5, color=col,
                            mec=INK, mew=.5, zorder=7)
        ax.set_xlim(0, L * NM); ax.set_ylim(0, L * NM); ax.set_aspect("equal")
        ax.set_title(f"t = {p['t'] * 1e9:.1f} ns", fontsize=9, pad=4)
        if k % ncol == 0:
            ax.set_ylabel("y (nm)")
        if k // ncol == nrow - 1:
            ax.set_xlabel("x (nm)")
    for a in axes[len(idx):]:
        a.axis("off")
    ticks = [v for v in (0, 0.25, 0.5, 1, 2, 4, 6, 8, 10) if v <= smax]
    cb = fig.colorbar(pc, ax=axes.tolist(), fraction=0.025, pad=0.02,
                      ticks=ticks)
    cb.set_label("speed |u| (m/s)", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(title, fontsize=12, color=INK, y=0.985)
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", dst)


def main():
    sph_dir, cub_dir, figdir = sys.argv[1], sys.argv[2], sys.argv[3]
    ref2d = sys.argv[4] if len(sys.argv) > 4 else None
    os.makedirs(figdir, exist_ok=True)
    j = lambda f: os.path.join(figdir, f)

    tr_s, _, err_s = pose_track(sph_dir)
    tr_c, _, err_c = pose_track(cub_dir)
    print(f"pose recovery residual: sphere {err_s:.2e} m, cube {err_c:.2e} m")

    fig_slices_pose(tr_s, "sphere",
                    "3-D cavity + free sphere (R = 75 nm), z-mid-plane",
                    j("kn_sphere_slices.png"))
    fig_slices_pose(tr_c, "cube",
                    "3-D cavity + free cube (L = 150 nm), z-mid-plane",
                    j("kn_cube_slices.png"))
    fig_pose_panels(tr_s, "sphere",
                    "Free sphere: true pose, body-fixed markers, centre path",
                    j("pose_sphere.png"))
    fig_pose_panels(tr_c, "cube",
                    "Free cube: true pose (one face painted), centre path",
                    j("pose_cube.png"))
    fig_markers(tr_s, "sphere", "Sphere — surface-marker tracking",
                j("markers_sphere.png"))
    fig_markers(tr_c, "cube", "Cube — surface-marker tracking",
                j("markers_cube.png"))
    fig_paths([tr_s, tr_c], ["free sphere (R = 75 nm)",
                             "free cube (L = 150 nm)"], j("paths.png"))

    series = [("3-D free sphere, Kn = 0.127", C_BLUE, "-",
               [(p["t"],) + vortex_core(p["vtk"]) for p in tr_s]),
              ("3-D free cube, Kn = 0.127", C_ORANGE, "--",
               [(p["t"],) + vortex_core(p["vtk"]) for p in tr_c])]
    if ref2d:
        f2 = sorted(glob.glob(os.path.join(ref2d, "output_*.vtk")))
        pts = []
        for fn in f2:
            d = read_vtk(fn)
            if d["t"] > 0:
                pts.append((d["t"],) + vortex_core(d, plane="2d"))
        if pts:
            series.append(("2-D reference, Kn = 0.013", C_AQUA, ":", pts))
    fig_vortex_core(series, j("vortex_core.png"))


if __name__ == "__main__":
    main()
