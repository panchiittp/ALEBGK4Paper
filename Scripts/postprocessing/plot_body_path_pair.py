#!/usr/bin/env python3
"""Two-panel trajectory figure: the body's path at cavity scale, and the same
path zoomed to its own extent.

The full plot_body_path.py carries six panels (path, displacement components,
angular velocity, Kabsch turn). This is the trajectory alone, for when the
question is simply "where did the body go" -- at cavity scale the orbit is a
small feature, and zoomed it is the whole story, so both are needed together.

Containment is tested from the body's OWN surface particles rather than from
the centre against a bounding sphere: a cube presenting a FACE has extent L/2,
not L*sqrt(3)/2, so the sphere test flags a legitimately contained body.

Usage: plot_body_path_pair.py <run_dir> <out.png> [cube_side_m]
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
from body_pose import body_points                             # noqa: E402

C_BLUE, C_ORANGE, C_YELLOW = "#2a78d6", "#eb6834", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
NM, BOX = 1e9, 1e-6

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})


def timecolored(a, x, y, t, lw=2.4):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap="viridis", lw=lw)
    lc.set_array(t[:-1])
    a.add_collection(lc)
    return lc


run, dst = sys.argv[1], sys.argv[2]
L = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5e-7

b = np.genfromtxt(os.path.join(run, "bodies.csv"), delimiter=",", names=True)
g = lambda k: np.atleast_1d(b[k])
t, cx, cy = g("t"), g("cx") * NM, g("cy") * NM
fin = np.isfinite(cx) & np.isfinite(cy)
if not fin.all():
    print("NOTE: non-finite from t = %.3e s -- plotting the finite part only"
          % t[~fin][0])
t, cx, cy = t[fin], cx[fin], cy[fin]
tn = t * 1e9

excursion = []
for fn in sorted(glob.glob(os.path.join(run, "output_*.vtk"))):
    d = read_vtk(fn)
    B, _ = body_points(d)
    if len(B):
        excursion.append(max(-B.min(), B.max() - BOX))
excursion = np.array(excursion) * NM

fig, ax = plt.subplots(1, 2, figsize=(11.2, 5.2))
hd = 0.5 * np.sqrt(3.0) * L * NM

# --- (a) cavity scale ------------------------------------------------------
a = ax[0]
a.add_patch(plt.Rectangle((0, 0), BOX * NM, BOX * NM, fill=False, ec=MUTED,
                          lw=1.2))
a.add_patch(plt.Rectangle((hd, hd), BOX * NM - 2 * hd, BOX * NM - 2 * hd,
                          fill=False, ec=C_YELLOW, lw=1.1, ls="--"))
a.annotate("centre must stay inside\nfor the body to fit",
           xy=(hd + 8, BOX * NM - hd - 60), fontsize=7, color="#b07a00")
a.annotate("lid  →", xy=(0.5 * BOX * NM, 1.02 * BOX * NM), ha="center",
           fontsize=8, color=MUTED)
a.add_patch(plt.Circle((cx[-1], cy[-1]), hd, fill=False, ec=C_ORANGE, lw=1.0,
                       ls=":"))
lc = timecolored(a, cx, cy, tn)
a.plot(cx[0], cy[0], "o", color=C_BLUE, ms=7, label="start")
a.plot(cx[-1], cy[-1], "s", color=C_ORANGE, ms=7, label="end")
a.set_xlim(-40, BOX * NM + 40)
a.set_ylim(-40, BOX * NM + 60)
a.set_aspect("equal")
a.set_xlabel("x (nm)"); a.set_ylabel("y (nm)")
a.set_title("centre path in the cavity")
a.legend(frameon=False, fontsize=8, loc="lower left")
cb = fig.colorbar(lc, ax=a, fraction=0.046, pad=0.03)
cb.set_label("t (ns)", fontsize=8)
cb.ax.tick_params(labelsize=7)

# --- (b) zoomed to the path's own extent ----------------------------------
a = ax[1]
lc2 = timecolored(a, cx, cy, tn, lw=2.8)
a.plot(cx[0], cy[0], "o", color=C_BLUE, ms=7)
a.plot(cx[-1], cy[-1], "s", color=C_ORANGE, ms=7)
a.annotate("start", xy=(cx[0], cy[0]), xytext=(6, 6),
           textcoords="offset points", fontsize=8, color=C_BLUE)
a.annotate("end", xy=(cx[-1], cy[-1]), xytext=(6, -12),
           textcoords="offset points", fontsize=8, color=C_ORANGE)
pad = 0.08 * max(cx.ptp(), cy.ptp()) + 5
mx, my = 0.5 * (cx.max() + cx.min()), 0.5 * (cy.max() + cy.min())
half = 0.5 * max(cx.ptp(), cy.ptp()) + pad
a.set_xlim(mx - half, mx + half)
a.set_ylim(my - half, my + half)
a.set_aspect("equal")
a.set_xlabel("x (nm)"); a.set_ylabel("y (nm)")
a.set_title("path, zoomed to its own extent")
a.grid(alpha=0.3, color=GRID)

if excursion.size and excursion.max() > 0:
    print("WALL BREACH: max %.1f nm outside the box" % excursion.max())
else:
    print("CONTAINED: closest approach %.1f nm inside the wall"
          % -excursion.max())
fig.suptitle("Free rigid body in the 3-D driven cavity — trajectory   (%s)"
             % os.path.basename(run.rstrip("/\\")), fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
fig.savefig(dst, dpi=160)
fig.savefig(os.path.splitext(dst)[0] + ".eps")
print("saved %s (+ .eps)   t_end = %.4e s" % (dst, t[-1]))
