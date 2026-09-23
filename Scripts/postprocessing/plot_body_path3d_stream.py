#!/usr/bin/env python3
"""3-D companion to plot_body_path_pair.py: the body's trajectory in the
cavity volume, with 3-D streamlines of the flow that carries it.

Left  -- the cavity box at full scale, the time-coloured centre path, the body
         drawn from its OWN surface particles at the final pose, and the path
         projected onto three walls so the 3-D shape of the orbit is readable
         without rotating the view.
Right -- the same path with 3-D streamlines integrated by RK4 through the
         velocity field.

The cloud is a regular lattice, so the scattered particle data is reshaped
back onto that grid and trilinearly interpolated; no external interpolation
dependency. Body surface particles are excluded from the grid fill -- they
carry the WALL velocity, not the gas velocity, and streamlines seeded from
them would draw the body's own motion as if it were flow. Voxels the body has
emptied come back NaN and are zeroed, so an integrator entering the body stops
rather than propagating NaN down the streamline.

Usage: plot_body_path3d_stream.py <run_dir> <out.png> [cube_side_m]
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_cavity3d_body import read_vtk                       # noqa: E402
from body_pose import body_points                             # noqa: E402

C_BLUE, C_ORANGE, C_YELLOW = "#2a78d6", "#eb6834", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
NM, BOX = 1e9, 1e-6

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})

run, dst = sys.argv[1], sys.argv[2]
L = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5e-7

# ---- trajectory ----------------------------------------------------------
b = np.genfromtxt(os.path.join(run, "bodies.csv"), delimiter=",", names=True)
gg = lambda k: np.atleast_1d(b[k])
t, cx, cy, cz = gg("t"), gg("cx"), gg("cy"), gg("cz")
fin = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(cz)
t, cx, cy, cz = t[fin], cx[fin] * NM, cy[fin] * NM, cz[fin] * NM
tn = t * 1e9

# ---- final field ---------------------------------------------------------
snaps = sorted(glob.glob(os.path.join(run, "output_*.vtk")))
d = read_vtk(snaps[-1])
pts = d["pts"]
vel = d["velocity"] if "velocity" in d else d["vel"]
Bp, bidx = body_points(d)

mask = np.ones(len(pts), bool)
mask[bidx] = False
xs = np.unique(np.round(pts[mask, 0], 12))
ys = np.unique(np.round(pts[mask, 1], 12))
zs = np.unique(np.round(pts[mask, 2], 12))
nx, ny, nz = len(xs), len(ys), len(zs)
U = np.full((nx, ny, nz, 3), np.nan)
ix = np.searchsorted(xs, np.round(pts[mask, 0], 12))
iy = np.searchsorted(ys, np.round(pts[mask, 1], 12))
iz = np.searchsorted(zs, np.round(pts[mask, 2], 12))
U[ix, iy, iz] = vel[mask]
U = np.nan_to_num(U)


def sample(p):
    """Trilinear sample of U at physical point p (metres)."""
    fx = np.interp(p[0], xs, np.arange(nx))
    fy = np.interp(p[1], ys, np.arange(ny))
    fz = np.interp(p[2], zs, np.arange(nz))
    i0, j0, k0 = int(fx), int(fy), int(fz)
    i1 = min(i0 + 1, nx - 1)
    j1 = min(j0 + 1, ny - 1)
    k1 = min(k0 + 1, nz - 1)
    a, bb, c = fx - i0, fy - j0, fz - k0
    out = np.zeros(3)
    for di, wa in ((0, 1.0 - a), (1, a)):
        for dj, wb in ((0, 1.0 - bb), (1, bb)):
            for dk, wc in ((0, 1.0 - c), (1, c)):
                out += wa * wb * wc * U[(i0 if di == 0 else i1),
                                        (j0 if dj == 0 else j1),
                                        (k0 if dk == 0 else k1)]
    return out


def streamline(p0, h, n):
    """RK4 streamline in arclength, stopping at the box or where flow dies."""
    P = [np.array(p0, float)]
    for _ in range(n):
        p = P[-1]
        k1 = sample(p)
        if np.linalg.norm(k1) < 1e-6:
            break
        u1 = k1 / np.linalg.norm(k1)
        k2 = sample(p + 0.5 * h * u1)
        u2 = k2 / max(np.linalg.norm(k2), 1e-30)
        k3 = sample(p + 0.5 * h * u2)
        u3 = k3 / max(np.linalg.norm(k3), 1e-30)
        k4 = sample(p + h * u3)
        u4 = k4 / max(np.linalg.norm(k4), 1e-30)
        step = (u1 + 2.0 * u2 + 2.0 * u3 + u4) / 6.0
        step /= max(np.linalg.norm(step), 1e-30)
        q = p + h * step
        if np.any(q < 0.0) or np.any(q > BOX):
            break
        P.append(q)
    return np.array(P) * NM


def box_wire(a):
    r = [0.0, BOX * NM]
    edges = [((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 1, 0)),
             ((0, 0, 0), (0, 0, 1)), ((1, 1, 1), (0, 1, 1)),
             ((1, 1, 1), (1, 0, 1)), ((1, 1, 1), (1, 1, 0)),
             ((1, 0, 0), (1, 1, 0)), ((1, 0, 0), (1, 0, 1)),
             ((0, 1, 0), (1, 1, 0)), ((0, 1, 0), (0, 1, 1)),
             ((0, 0, 1), (1, 0, 1)), ((0, 0, 1), (0, 1, 1))]
    for s, e in edges:
        a.plot([r[s[0]], r[e[0]]], [r[s[1]], r[e[1]]], [r[s[2]], r[e[2]]],
               color=MUTED, lw=0.8, alpha=0.6)


def path3d(a, lw=2.4):
    P = np.array([cx, cy, cz]).T.reshape(-1, 1, 3)
    seg = np.concatenate([P[:-1], P[1:]], axis=1)
    lc = Line3DCollection(seg, cmap="viridis", lw=lw)
    lc.set_array(tn[:-1])
    a.add_collection3d(lc)
    return lc


fig = plt.figure(figsize=(12.4, 5.8))

a = fig.add_subplot(1, 2, 1, projection="3d")
box_wire(a)
lc = path3d(a)
a.plot(cx, cy, np.zeros_like(cz), color=MUTED, lw=0.9, alpha=0.55)
a.plot(cx, np.full_like(cy, BOX * NM), cz, color=MUTED, lw=0.9, alpha=0.55)
a.plot(np.zeros_like(cx), cy, cz, color=MUTED, lw=0.9, alpha=0.55)
a.scatter(Bp[:, 0] * NM, Bp[:, 1] * NM, Bp[:, 2] * NM, s=5,
          color=C_ORANGE, alpha=0.85, depthshade=False)
a.scatter([cx[0]], [cy[0]], [cz[0]], color=C_BLUE, s=45)
a.scatter([cx[-1]], [cy[-1]], [cz[-1]], color=C_ORANGE, s=45, marker="s")
a.set_title("trajectory in the cavity volume\n"
            "(grey = projections on the walls)", fontsize=9.5)
cb = fig.colorbar(lc, ax=a, fraction=0.03, pad=0.10)
cb.set_label("t (ns)", fontsize=8)
cb.ax.tick_params(labelsize=7)

a2 = fig.add_subplot(1, 2, 2, projection="3d")
box_wire(a2)
rng = np.random.default_rng(0)
h = BOX / 90.0
for _ in range(26):
    s0 = [rng.uniform(0.18, 0.82) * BOX for _ in range(3)]
    for sgn in (1.0, -1.0):
        S = streamline(s0, sgn * h, 260)
        if len(S) > 12:
            a2.plot(S[:, 0], S[:, 1], S[:, 2], color=C_BLUE, lw=0.65,
                    alpha=0.45)
path3d(a2, lw=3.0)
a2.scatter(Bp[:, 0] * NM, Bp[:, 1] * NM, Bp[:, 2] * NM, s=5,
           color=C_ORANGE, alpha=0.9, depthshade=False)
a2.set_title("body path on 3-D streamlines of the carrying flow",
             fontsize=9.5)

for ax_ in (a, a2):
    ax_.set_xlim(0, BOX * NM)
    ax_.set_ylim(0, BOX * NM)
    ax_.set_zlim(0, BOX * NM)
    ax_.set_xlabel("x (nm)", fontsize=8)
    ax_.set_ylabel("y (nm)", fontsize=8)
    ax_.set_zlabel("z (nm)", fontsize=8)
    ax_.tick_params(labelsize=7)
    ax_.view_init(elev=22, azim=-58)
    try:
        ax_.set_box_aspect((1, 1, 1))
    except Exception:
        pass

fig.suptitle("Free rigid body in the 3-D driven cavity - 3-D trajectory and "
             "flow   (%s, t = %.0f ns)"
             % (os.path.basename(run.rstrip("/\\")), t[-1] * 1e9),
             fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
fig.savefig(dst, dpi=160)
print("saved %s   grid %dx%dx%d   %d body surface pts   t_end=%.4e s"
      % (dst, nx, ny, nz, len(Bp), t[-1]))
