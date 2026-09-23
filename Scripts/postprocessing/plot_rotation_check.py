#!/usr/bin/env python3
"""Rotation-diagnostic figure for the 3-D driven cavity with a free rigid body.

Two INDEPENDENT views of the same rotation, so the log and the field cannot
quietly disagree again:
  * omega(t) as the solver logs it in bodies.csv;
  * the cumulative turn angle recovered by Kabsch from the body's OWN surface
    particles in the VTK snapshots -- which knows nothing about the log.
Plus the cube drawn at its first and last recovered pose.

Usage: plot_rotation_check.py <run_dir> <out.png> [cube_side_m]
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_cavity3d_body import read_vtk                      # noqa: E402
from body_pose import body_points, kabsch, rot_angle, cube_faces  # noqa: E402

C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
NM = 1e9

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})

def pose_track(outdir):
    """(t, centre, R) per snapshot, from the body's own surface particles."""
    out, ref = [], None
    for fn in sorted(glob.glob(os.path.join(outdir, "output_*.vtk"))):
        d = read_vtk(fn)
        B, _ = body_points(d)
        if ref is None:
            ref = B
        if B.shape != ref.shape:
            print("  skip %s (%d body pts, expected %d)"
                  % (os.path.basename(fn), B.shape[0], ref.shape[0]))
            continue
        R, _, cB, rmsd = kabsch(ref, B)
        out.append({"t": d["t"], "c": cB, "R": R, "rmsd": rmsd})
    out.sort(key=lambda p: p["t"])
    return out

def draw_cube(ax, c, R, L, base, hi_face, label):
    faces, v = cube_faces(c * NM, R, L * NM)
    cols = [base] * 6
    cols[hi_face] = C_YELLOW          # one painted face -> the spin is visible
    pc = Poly3DCollection(faces, facecolors=cols, edgecolors=INK,
                          linewidths=0.9, alpha=0.95)
    ax.add_collection3d(pc)
    ax.set_title(label, fontsize=9)
    return v

def main():
    run, dst = sys.argv[1], sys.argv[2]
    L = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5e-7

    b = np.genfromtxt(os.path.join(run, "bodies.csv"),
                      delimiter=",", names=True)
    track = pose_track(run)
    print("pose snapshots: %d, worst Kabsch rmsd = %.2e m"
          % (len(track), max(p["rmsd"] for p in track)))

    tp = np.array([p["t"] for p in track])
    # cumulative turn: sum the per-snapshot INCREMENT, so it keeps climbing
    # past 180 deg instead of folding back into arccos's [0, pi] branch
    inc = [0.0]
    for k in range(1, len(track)):
        dR = track[k]["R"] @ track[k - 1]["R"].T
        inc.append(inc[-1] + rot_angle(dR))
    ang_pose = np.degrees(inc)

    tb = np.atleast_1d(b["t"])
    wx, wy, wz = (np.atleast_1d(b["omx"]), np.atleast_1d(b["omy"]),
                  np.atleast_1d(b["angVel"]))
    wmag = np.sqrt(wx ** 2 + wy ** 2 + wz ** 2)
    # trapezoid of |omega| -> the same turn angle the log predicts
    ang_log = np.degrees(np.concatenate(
        [[0.0], np.cumsum(0.5 * (wmag[1:] + wmag[:-1]) * np.diff(tb))]))

    fig = plt.figure(figsize=(11.0, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.32,
                          wspace=0.30)

    ax = fig.add_subplot(gs[0, :2])
    ax.plot(tb * 1e9, wz * 1e-6, "-", color=C_AQUA, lw=2.0,
            label=r"$\omega_z$  (lid-driven vortex axis)")
    ax.plot(tb * 1e9, wx * 1e-6, "-", color=C_BLUE, lw=1.6,
            label=r"$\omega_x$")
    ax.plot(tb * 1e9, wy * 1e-6, "--", color=C_ORANGE, lw=1.6,
            label=r"$\omega_y$")
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_xlabel("t  (ns)")
    ax.set_ylabel(r"angular velocity  ($10^6$ rad/s)")
    ax.set_title("body spin-up, as logged in bodies.csv "
                 "(all three components now written)")

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(tp * 1e9, ang_pose, "-", color=C_ORANGE, lw=2.2,
            label="Kabsch pose (VTK)")
    ax.plot(tb * 1e9, ang_log, "--", color=C_BLUE, lw=1.6,
            label=r"$\int|\omega|\,dt$ (log)")
    ax.set_xlabel("t  (ns)")
    ax.set_ylabel("cumulative turn  (deg)")
    ax.set_title("two independent measures agree")

    for a in fig.axes:
        a.grid(True, color=GRID, lw=0.6)
        a.set_axisbelow(True)
        a.legend(frameon=False, fontsize=8)

    # bottom row: the pose itself. A few degrees of turn is invisible in a
    # plain 3-D render, so the third panel overlays the exact z-midplane
    # cross-sections of the first and last pose at body scale, where the
    # corner displacement is unmistakable.
    from body_pose import cube_section
    for k, (idx, base) in enumerate([(0, C_BLUE), (-1, C_ORANGE)]):
        a = fig.add_subplot(gs[1, k], projection="3d")
        p = track[idx]
        draw_cube(a, p["c"], p["R"], L, base, 0,
                  r"t = %.2f ns    turn = %.2f$^\circ$"
                  % (p["t"] * 1e9, ang_pose[idx]))
        ctr, h = track[0]["c"] * NM, L * NM * 1.05
        a.set_xlim(ctr[0] - h, ctr[0] + h)
        a.set_ylim(ctr[1] - h, ctr[1] + h)
        a.set_zlim(ctr[2] - h, ctr[2] + h)
        a.set_box_aspect((1, 1, 1))
        a.view_init(elev=22, azim=-58)
        a.set_xlabel("x (nm)", fontsize=7, labelpad=-4)
        a.set_ylabel("y (nm)", fontsize=7, labelpad=-4)
        a.set_zlabel("z (nm)", fontsize=7, labelpad=-4)
        a.tick_params(labelsize=6)
        a.grid(False)

    a = fig.add_subplot(gs[1, 2])

    def section(idx):
        q = track[idx]
        # section through the body's OWN centre, so the panel shows rotation
        # and not the translation the body also undergoes
        sec = cube_section(q["c"] * NM, q["R"], L * NM, q["c"][2] * NM)
        if sec is None:
            return None
        sec = np.vstack([sec, sec[:1]])
        return sec - np.array([q["c"][0], q["c"][1]]) * NM

    for idx, col, sty, lab in [(0, C_BLUE, "--", "t = %.2f ns" % (track[0]["t"] * 1e9)),
                               (-1, C_ORANGE, "-", "t = %.2f ns" % (track[-1]["t"] * 1e9))]:
        sec = section(idx)
        if sec is not None:
            a.plot(sec[:, 0], sec[:, 1], sty, color=col, lw=2.0, label=lab)
    # corner-zoom inset: a few degrees of turn is a few nm of corner travel,
    # which the full outline cannot resolve
    s0, s1 = section(0), section(-1)
    if s0 is not None and s1 is not None:
        cnr = s1[np.argmax(s1[:-1, 0] + s1[:-1, 1])]
        ins = a.inset_axes([0.05, 0.03, 0.44, 0.29])
        for sec, col, sty in [(s0, C_BLUE, "--"), (s1, C_ORANGE, "-")]:
            ins.plot(sec[:, 0], sec[:, 1], sty, color=col, lw=1.8)
        w = 0.20 * L * NM
        ins.set_xlim(cnr[0] - w, cnr[0] + w)
        ins.set_ylim(cnr[1] - w, cnr[1] + w)
        ins.set_aspect("equal")
        ins.set_xticks([]); ins.set_yticks([])
        ins.set_title("corner, zoomed", fontsize=7, color=MUTED, pad=2)
        for sp_ in ins.spines.values():
            sp_.set_color(MUTED)
        a.indicate_inset_zoom(ins, edgecolor=MUTED)
    a.set_aspect("equal")
    # explicit limits: the square keeps the upper band, the empty lower band
    # is reserved for the corner-zoom inset so the two never overlap
    lim = 0.78 * L * NM
    a.set_xlim(-lim, lim)
    a.set_ylim(-2.0 * lim, lim)
    a.set_xlabel("x - x$_c$  (nm)")
    a.set_ylabel("y - y$_c$  (nm)")
    a.set_title("z-midplane section: start vs end\n"

                r"(net turn %.2f$^\circ$ about the vortex axis)" % ang_pose[-1],
                fontsize=9)
    a.grid(True, color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("3-D driven cavity, free rigid cube — rotation check   "
                 "(%s)" % os.path.basename(run), fontsize=12, color=INK)
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    print("saved", dst)
    print("final: |omega| = %.3e rad/s, cumulative turn = %.3f deg"
          % (wmag[-1], ang_pose[-1]))

if __name__ == "__main__":
    main()
