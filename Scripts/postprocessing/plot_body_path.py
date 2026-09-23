#!/usr/bin/env python3
"""Trajectory + rotation history of the free rigid body in the 3-D cavity.

Answers the two things a long run is for: where did the body GO, and how far
did it TURN. The path comes from bodies.csv (cx, cy, cz per saveEvery); the
turn is recovered independently by Kabsch from the body's own surface
particles in the VTK snapshots, so a bug in one is not hidden by the other.

Usage: plot_body_path.py <run_dir> <out.png> [cube_side_m]
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_cavity3d_body import read_vtk                       # noqa: E402
from body_pose import body_points, kabsch, rot_angle          # noqa: E402

C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
NM, BOX = 1e9, 1e-6

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})


def timecolored(a, x, y, t, lw=2.0):
    """Path drawn as a time-coloured line so direction reads without arrows."""
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap="viridis", lw=lw)
    lc.set_array(t[:-1])
    a.add_collection(lc)
    return lc


def surface_excursion(run, box=1e-6):
    """Per snapshot: (t, R_unused, max distance any surface particle lies
    OUTSIDE the domain box). Positive means the body is really sticking out."""
    out = []
    for fn in sorted(glob.glob(os.path.join(run, "output_*.vtk"))):
        d = read_vtk(fn)
        B, _ = body_points(d)
        if len(B) == 0:
            continue
        out.append((d["t"], None, max(-B.min(), B.max() - box)))
    return out


def turn_history(run):
    """Cumulative turn angle (deg) vs t, from the surface particles."""
    out, ref = [], None
    for fn in sorted(glob.glob(os.path.join(run, "output_*.vtk"))):
        d = read_vtk(fn)
        B, _ = body_points(d)
        if ref is None:
            ref = B
        if B.shape != ref.shape:
            continue
        R, _, _, _ = kabsch(ref, B)
        out.append((d["t"], R))
    out.sort(key=lambda p: p[0])
    t = np.array([p[0] for p in out])
    inc = [0.0]
    for k in range(1, len(out)):
        inc.append(inc[-1] + rot_angle(out[k][1] @ out[k - 1][1].T))
    return t, np.degrees(inc)


def main():
    run, dst = sys.argv[1], sys.argv[2]
    L = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5e-7

    b = np.genfromtxt(os.path.join(run, "bodies.csv"), delimiter=",",
                      names=True)
    g = lambda k: np.atleast_1d(b[k])
    t, cx, cy, cz = g("t"), g("cx") * NM, g("cy") * NM, g("cz") * NM
    wz, wx, wy = g("angVel"), g("omx"), g("omy")
    fin = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(wz)
    if not fin.all():
        print("NOTE: run went non-finite at t = %.3e s (row %d of %d) -- "
              "plotting the finite part only" % (t[~fin][0], fin.argmin(),
                                                 len(t)))
    t, cx, cy, cz = t[fin], cx[fin], cy[fin], cz[fin]
    wz, wx, wy = wz[fin], wx[fin], wy[fin]
    tn = t * 1e9

    tt, ang = turn_history(run)
    pose_excursion = surface_excursion(run)

    fig = plt.figure(figsize=(12.6, 7.4))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.28)

    # (a) x-y path inside the cavity, at cavity scale.
    #     The solver has NO body-wall contact model, so the only thing that
    #     keeps the body inside the box is the flow. Draw the region the
    #     centre must stay in for the body to be fully inside (the box inset
    #     by the half-diagonal) so a wall breach is visible, not inferred.
    a = fig.add_subplot(gs[:, 0])
    a.add_patch(plt.Rectangle((0, 0), BOX * NM, BOX * NM, fill=False,
                              ec=MUTED, lw=1.2))
    hd = 0.5 * np.sqrt(3.0) * L * NM            # cube half-diagonal
    a.add_patch(plt.Rectangle((hd, hd), BOX * NM - 2 * hd, BOX * NM - 2 * hd,
                              fill=False, ec=C_YELLOW, lw=1.1, ls="--"))
    a.annotate("centre must stay inside\nfor the body to fit",
               xy=(hd + 8, BOX * NM - hd - 60), fontsize=7,
               color="#b07a00")
    a.annotate("lid  →", xy=(0.5 * BOX * NM, 1.02 * BOX * NM),
               ha="center", fontsize=8, color=MUTED)
    # body footprint at the end pose
    a.add_patch(plt.Circle((cx[-1], cy[-1]), hd, fill=False, ec=C_ORANGE,
                           lw=1.0, ls=":"))
    lc = timecolored(a, cx, cy, tn, lw=2.4)
    # first wall breach
    # True containment test, from the body's OWN surface particles. Testing
    # the centre against the bounding-sphere half-diagonal (as this did) is
    # conservative to the point of being wrong: a cube presenting a FACE has
    # extent L/2, not L*sqrt(3)/2, so a legitimately-contained body trips it.
    excursion = np.array([e for _, _, e in pose_excursion]) * NM
    et = np.array([t_ for t_, _, e in pose_excursion])
    br = np.where(excursion > 0.0)[0]
    clr = -excursion
    if br.size:
        k = br[0]
        tk = et[k]
        kk = int(np.argmin(np.abs(t - tk)))
        a.plot(cx[kk], cy[kk], "x", color="#c0392b", ms=9, mew=2.0,
               label="wall breach  t = %.0f ns" % (tk * 1e9))
        print("WALL BREACH: a surface particle is outside the box at "
              "t = %.3e s (by %.1f nm); max over run %.1f nm"
              % (tk, excursion[k], excursion.max()))
    else:
        print("CONTAINED: no surface particle ever leaves the box; "
              "closest approach %.1f nm inside the wall (clearance floor "
              "0.4*dx = %.1f nm)"
              % (-excursion.max(), 0.4 * (1e-6 / 14) * NM))
    a.plot(cx[0], cy[0], "o", color=C_BLUE, ms=7, label="start")
    a.plot(cx[-1], cy[-1], "s", color=C_ORANGE, ms=7, label="end")
    a.set_xlim(-40, BOX * NM + 40)
    a.set_ylim(-40, BOX * NM + 60)
    a.set_aspect("equal")
    a.set_xlabel("x (nm)")
    a.set_ylabel("y (nm)")
    a.set_title("centre path in the cavity")
    a.legend(frameon=False, fontsize=8, loc="lower left",
             bbox_to_anchor=(-0.02, -0.02))
    cb = fig.colorbar(lc, ax=a, fraction=0.046, pad=0.03)
    cb.set_label("t (ns)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # (b) same path, zoomed to its own extent
    a = fig.add_subplot(gs[0, 1])
    timecolored(a, cx, cy, tn, lw=2.0)
    a.plot(cx[0], cy[0], "o", color=C_BLUE, ms=6)
    a.plot(cx[-1], cy[-1], "s", color=C_ORANGE, ms=6)
    a.set_aspect("equal")
    a.margins(0.12)
    a.set_xlabel("x (nm)")
    a.set_ylabel("y (nm)")
    a.set_title("path, zoomed to its own extent")

    # (c) displacement components
    a = fig.add_subplot(gs[0, 2])
    a.plot(tn, cx - cx[0], "-", color=C_BLUE, lw=1.8, label=r"$\Delta x$")
    a.plot(tn, cy - cy[0], "--", color=C_ORANGE, lw=1.8, label=r"$\Delta y$")
    a.plot(tn, cz - cz[0], ":", color=C_AQUA, lw=1.8, label=r"$\Delta z$")
    a.set_xlabel("t (ns)")
    a.set_ylabel("displacement (nm)")
    a.set_title("centre displacement")
    a.legend(frameon=False, fontsize=8)

    # (d) angular velocity
    a = fig.add_subplot(gs[1, 1])
    a.plot(tn, wz * 1e-6, "-", color=C_AQUA, lw=2.0, label=r"$\omega_z$")
    a.plot(tn, wx * 1e-6, "-", color=C_BLUE, lw=1.2, label=r"$\omega_x$")
    a.plot(tn, wy * 1e-6, "--", color=C_ORANGE, lw=1.2, label=r"$\omega_y$")
    a.axhline(0, color=GRID, lw=0.8)
    a.set_xlabel("t (ns)")
    a.set_ylabel(r"$\omega$  ($10^6$ rad/s)")
    a.set_title("angular velocity")
    if br.size:
        a.axvline(et[br[0]] * 1e9, color="#c0392b", lw=1.0,
                  ls="--")
    a.legend(frameon=False, fontsize=8)

    # (e) cumulative turn, with full-revolution gridlines
    a = fig.add_subplot(gs[1, 2])
    a.plot(tt * 1e9, ang, "-", color=C_ORANGE, lw=2.2)
    for k in range(1, int(ang[-1] // 360) + 2):
        if k * 360 <= ang[-1] * 1.15:
            a.axhline(k * 360, color=MUTED, lw=0.8, ls=":")
            a.annotate("%d turn%s" % (k, "" if k == 1 else "s"),
                       xy=(0.02, k * 360), xycoords=("axes fraction", "data"),
                       fontsize=7, color=MUTED, va="bottom")
    a.set_xlabel("t (ns)")
    a.set_ylabel("cumulative turn (deg)")
    a.set_title("rotation (Kabsch pose, from surface particles)")
    if br.size:
        a.axvline(et[br[0]] * 1e9, color="#c0392b", lw=1.0, ls="--")
        a.annotate("wall breach", xy=(t[br[0]] * 1e9, 0.05), rotation=90,
                   xycoords=("data", "axes fraction"), fontsize=7,
                   color="#c0392b", ha="right")

    for a in fig.axes[:-1]:
        a.grid(True, color=GRID, lw=0.6)
        a.set_axisbelow(True)

    fig.suptitle("Free rigid body in the 3-D driven cavity — path and "
                 "rotation   (%s)" % os.path.basename(run),
                 fontsize=12, color=INK)
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    print("saved", dst)
    print("t_end = %.4e s   turn = %.2f deg (%.3f revolutions)"
          % (t[-1], ang[-1], ang[-1] / 360.0))
    print("centre moved %.2f nm; final omega_z = %.4e rad/s"
          % (np.hypot(cx[-1] - cx[0], cy[-1] - cy[0]), wz[-1]))


if __name__ == "__main__":
    main()
