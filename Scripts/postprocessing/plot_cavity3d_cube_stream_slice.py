#!/usr/bin/env python3
"""Paper figure: free cube in the 3-D driven cavity.

Left  -- 3-D streamlines of the carrying flow, integrated by RK4 through the
         velocity field, with the cavity wireframe and the cube drawn from its
         own surface particles at the final pose.
Right -- the y = L/2 slice on a white ground: in-plane streamlines, the
         body-centre path, and the cube's cross-section outlined at several
         instants so the rotation is visible.

The MOVING LID is drawn as the top face. In the 3-D body cases the solver
drives the y = L wall (the body-free cavity drives z = L), so the figure plots
the solver's y as the vertical axis and labels it z, and the solver's spanwise
z horizontally as y. That is a relabelling of the frame, not a change to the
data -- a cube with one moving wall has no intrinsic axis naming -- and it
makes the body case agree with the body-free one. Coordinates quoted beside
this figure must use the same convention: the body starts at (0.6, 0.5, 0.7)L
here, which is (0.6, 0.7, 0.5)L in the solver's frame.

Panel (b) carries no colour bars and no shaded speed field. The time
information a bar would carry is instead given by open markers placed at equal
time intervals along the path, so the sequence is readable without a scale.

The slice is taken at mid-span because the motion is very nearly planar: over
the integrated interval the centre's z stays within a few thousandths of L of
mid-span while it moves ~0.15L in plane, so the slice carries essentially the
whole trajectory rather than a projection of it.

The cloud is a regular lattice, so scattered particle data is reshaped back
onto that grid; no external interpolation dependency. Body surface particles
are excluded from the grid fill -- they carry the WALL velocity, not the gas
velocity, and streamlines seeded from them would draw the body's own motion as
if it were flow.

Usage: plot_cavity3d_cube_stream_slice.py <run_dir> <out.png> [n_poses]
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


def blend(color, a, bg="white"):
    """Flatten a colour against the background instead of using alpha.

    PostScript has no alpha channel, so every translucent artist is rendered
    OPAQUE in the .eps -- which would turn the faded pose sequence into a
    solid blob and bury the path. Pre-blending the colours here means the
    figure uses no transparency at all, and the .eps, .pdf and .png come out
    identical. Overlaps then resolve by z-order rather than compositing,
    which for a light-to-dark sequence looks the same.
    """
    import matplotlib.colors as mcolors
    c = np.array(mcolors.to_rgb(color))
    b = np.array(mcolors.to_rgb(bg))
    return tuple(a * c + (1.0 - a) * b)


NM, BOX = 1.0, 1e-6   # plot in metres; axes show the 1e-6 multiplier

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

run = sys.argv[1]
dst = sys.argv[2]
NPOSE = int(sys.argv[3]) if len(sys.argv) > 3 else 5

# ---- trajectory ----------------------------------------------------------
b = np.genfromtxt(os.path.join(run, "bodies.csv"), delimiter=",", names=True)
gg = lambda k: np.atleast_1d(b[k])
t, cx, cy, cz = gg("t"), gg("cx"), gg("cy"), gg("cz")
fin = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(cz)
t, cx, cy, cz = t[fin], cx[fin], cy[fin], cz[fin]
tn = t * 1e9

# ---- final field ---------------------------------------------------------
snaps = sorted(glob.glob(os.path.join(run, "output_*.vtk")))
if not snaps:
    sys.exit("no output_*.vtk in %s" % run)
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
filled = np.isfinite(U[..., 0])        # False inside the body and in its wake


def fill_holes(A, V, iters=10):
    """Nearest-neighbour dilation fill of lattice sites with no particle.

    The body occupies some sites, but a moving body also leaves a thin wake
    of sites that refill has not restored -- here 1839 of 216000, spanning
    rather more than the body's own cross-section. Left empty they show as a
    ragged void far larger than the cube, and a zero-velocity hole also makes
    the streamline integrator see a spurious stagnation region. They are
    therefore filled from their filled neighbours FOR DISPLAY, and the body's
    true cross-section is drawn on top so the reader sees the body rather
    than the void.
    """
    A, V = np.nan_to_num(A).copy(), V.copy()
    nd = V.ndim
    for _ in range(iters):
        if V.all():
            break
        acc = np.zeros_like(A)
        cnt = np.zeros(V.shape)
        for ax in range(nd):
            for sh in (1, -1):
                acc += np.roll(np.where(V[..., None], A, 0.0), sh, axis=ax)
                cnt += np.roll(V, sh, axis=ax)
        new = (~V) & (cnt > 0)
        A[new] = acc[new] / np.maximum(cnt[new], 1)[..., None]
        V = V | new
    return A


Uf = fill_holes(U, filled)


def sample(p):
    """Trilinear sample of the velocity at physical point p (metres)."""
    fx = np.interp(p[0], xs, np.arange(nx))
    fy = np.interp(p[1], ys, np.arange(ny))
    fz = np.interp(p[2], zs, np.arange(nz))
    i0, j0, k0 = int(fx), int(fy), int(fz)
    i1, j1, k1 = min(i0 + 1, nx - 1), min(j0 + 1, ny - 1), min(k0 + 1, nz - 1)
    a, bb, c = fx - i0, fy - j0, fz - k0
    out = np.zeros(3)
    for di, wa in ((0, 1.0 - a), (1, a)):
        for dj, wb in ((0, 1.0 - bb), (1, bb)):
            for dk, wc in ((0, 1.0 - c), (1, c)):
                out += wa * wb * wc * Uf[(i0 if di == 0 else i1),
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


def hull2d(P):
    """Convex hull (monotone chain) of an (n,2) array; no scipy needed."""
    pt = sorted(map(tuple, P))
    if len(pt) < 3:
        return np.array(pt)
    def half(seq):
        h = []
        for q in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (q[1] - h[-2][1]) -
                                   (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(q)
        return h
    return np.array(half(pt)[:-1] + half(pt[::-1])[:-1])


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
               color=blend(MUTED, 0.6), lw=0.8)


# ==========================================================================
fig = plt.figure(figsize=(12.2, 5.6))

# ---- (a) 3-D streamlines -------------------------------------------------
a1 = fig.add_subplot(1, 2, 1, projection="3d")
# mpl tints the 3-D panes by default, which reads as a wash beside the white
# 2-D panel; make them white so the pair matches.
a1.set_facecolor("white")        # the axes patch, not just the panes
for _pane in (a1.xaxis.pane, a1.yaxis.pane, a1.zaxis.pane):
    _pane.set_facecolor("white")
    _pane.set_edgecolor("#cfcfcf")
    _pane.set_alpha(1.0)
a1.grid(color="#e6e6e6", lw=0.5)
box_wire(a1)
# Seed on a structured lattice rather than at random: the vortex axis is
# along z, so random seeds all wind onto the same few helices. A short
# integration (about one turn) keeps the sense of rotation readable.
h = BOX / 90.0
seeds = [(fx * BOX, fy * BOX, fz * BOX)
         for fx in (0.22, 0.5, 0.78)
         for fy in (0.25, 0.55, 0.85)
         for fz in (0.25, 0.5, 0.75)]
# The MOVING LID is the axis drawn vertically, so that the driven wall is at
# the top of the panel where a reader expects it.
#
# In the 3-D body cases the solver drives the y = L face (the body-free
# cavity drives z = L instead). The figure therefore plots the solver's y as
# the vertical axis and labels it z, and the solver's spanwise z as the
# horizontal axis labelled y. This is a relabelling of the frame, not a
# change to the data: the cavity is a cube with one moving wall, so which
# axis carries that wall is a naming convention. The figure's convention is
# lid-normal = z, matching the body-free case; anything quoting coordinates
# alongside this figure must use the same one.
for s0 in seeds:
    for sgn in (1.0, -1.0):
        S = streamline(s0, sgn * h, 120)
        if len(S) > 12:
            a1.plot(S[:, 0], S[:, 2], S[:, 1], color=blend(C_BLUE, 0.5), lw=0.7)
a1.scatter(Bp[:, 0] * NM, Bp[:, 2] * NM, Bp[:, 1] * NM, s=26,
           color=C_ORANGE, depthshade=False, edgecolors="none")
a1.set_xlim(0, BOX * NM); a1.set_ylim(0, BOX * NM); a1.set_zlim(0, BOX * NM)
a1.set_xlabel("x (m)", fontsize=8)
a1.set_ylabel("y (m)", fontsize=8)          # solver's spanwise z
a1.set_zlabel("z (m)", fontsize=8)          # solver's y: the lid normal
a1.tick_params(labelsize=7)
# Looking down on the box, lid-normal vertical: the driven wall is the top
# face and the vortex rings, which lie in planes containing it, stand upright.
a1.view_init(elev=26, azim=-60)
try:
    a1.set_box_aspect((1, 1, 1))
except Exception:
    pass
# No panel caption: the manuscript supplies (a) and (b) separately.

# ---- (b) mid-span slice with the path ------------------------------------
a2 = fig.add_subplot(1, 2, 2)
k = int(np.argmin(np.abs(zs - 0.5 * BOX)))
u = Uf[:, :, k, 0].T
v = Uf[:, :, k, 1].T

# streamplot requires exactly uniform axes. The lattice is uniform to
# round-off only (spread ~1e-12 m against a 1.7e-8 m spacing, from rounding
# the coordinates to 12 decimals), so regularise the axes rather than
# resampling the field.
X = np.linspace(xs[0], xs[-1], nx) * NM
Y = np.linspace(ys[0], ys[-1], ny) * NM

a2.set_facecolor("white")
a2.streamplot(X, Y, u, v, color=INK2, linewidth=0.55, density=1.25,
              arrowsize=0.7)

# Cube cross-sections, taken as a genuine slab about the slice plane rather
# than a projection of the whole body: a tolerance of order the lattice
# spacing gives the cross-section, whereas a wide one returns the silhouette
# and the poses then overlap into an unreadable blob.
dz_slab = 1.5 * abs(zs[1] - zs[0])
sel = np.linspace(0, len(snaps) - 1, max(NPOSE, 2)).astype(int)
poses = []
for si in sel:
    dd = d if si == len(snaps) - 1 else read_vtk(snaps[si])
    Bq, _ = body_points(dd)
    near = np.abs(Bq[:, 2] - zs[k]) < dz_slab
    Q = (Bq[near] if near.sum() >= 3 else Bq)[:, :2] * NM
    H = hull2d(Q)
    if len(H) >= 3:
        poses.append((dd["t"], H))

# On the full slice only the final pose is drawn; the whole sequence goes in
# the inset, where it is legible.
if poses:
    H = poses[-1][1]
    a2.fill(H[:, 0], H[:, 1], facecolor=blend(C_ORANGE, 0.55), zorder=6,
            edgecolor=C_ORANGE, lw=1.5)

P = np.array([cx * NM, cy * NM]).T.reshape(-1, 1, 2)
seg = np.concatenate([P[:-1], P[1:]], axis=1)
lc = LineCollection(seg, cmap="autumn", lw=2.6, zorder=8)
lc.set_array(tn[:-1])
a2.add_collection(lc)

# Start and end markers kept, but unlabelled: no legend text on the panel.
a2.scatter([cx[0] * NM], [cy[0] * NM], color=C_BLUE, s=45, zorder=10)
a2.scatter([cx[-1] * NM], [cy[-1] * NM], color=C_ORANGE, s=48, marker="s",
           zorder=10)
a2.set_xlim(0, BOX * NM); a2.set_ylim(0, BOX * NM)
a2.set_aspect("equal")
a2.set_xlabel("x (m)", fontsize=8)
a2.set_ylabel("z (m)", fontsize=8)      # lid normal: the lid is the top edge
a2.tick_params(labelsize=7)
# Slice at the solver's z = L/2, which is y = L/2 in the figure's frame.
# No panel caption here either.

# ---- inset: the path at its own scale ------------------------------------
# The centre moves ~0.15L while the cube is 0.15L across, so at full-cavity
# scale the path is hidden under the body. The inset zooms on the travelled
# region so the translation and the rotation are both visible.
pad = 0.09 * BOX * NM
x0, x1 = cx.min() * NM - pad, cx.max() * NM + pad
y0, y1 = cy.min() * NM - pad, cy.max() * NM + pad
side = max(x1 - x0, y1 - y0)
xc, yc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
x0, x1 = xc - 0.5 * side, xc + 0.5 * side
y0, y1 = yc - 0.5 * side, yc + 0.5 * side

ins = a2.inset_axes([0.59, 0.04, 0.39, 0.39])
ins.set_facecolor("white")
for n, (tt, H) in enumerate(poses):
    last = (n == len(poses) - 1)
    ins.fill(H[:, 0], H[:, 1],
             facecolor=blend(C_ORANGE, 0.50 if last else 0.12),
             edgecolor=C_ORANGE if last else MUTED,
             lw=1.2 if last else 0.7, zorder=4)
lc2 = LineCollection(seg, cmap="autumn", lw=2.4, zorder=6)
lc2.set_array(tn[:-1])
ins.add_collection(lc2)

# With no colour bar, equal-time markers carry the progression instead.
nmk = 6
idx = np.linspace(0, len(cx) - 1, nmk).astype(int)
ins.scatter(cx[idx] * NM, cy[idx] * NM, facecolor="white", edgecolor=INK,
            s=16, lw=0.9, zorder=11)
ins.scatter([cx[0] * NM], [cy[0] * NM], color=C_BLUE, s=26, zorder=12)
ins.scatter([cx[-1] * NM], [cy[-1] * NM], color=C_ORANGE, s=28, marker="s",
            zorder=12)
ins.set_xlim(x0, x1); ins.set_ylim(y0, y1)
ins.set_aspect("equal")
ins.tick_params(labelsize=6, length=2)
for s in ins.spines.values():
    s.set_edgecolor(INK2)
# No inset caption either; it also collided with the axis multiplier.
a2.indicate_inset_zoom(ins, edgecolor=blend(INK2, 0.7), alpha=1.0)

fig.tight_layout()
os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
fig.savefig(dst, dpi=200)
base, _ = os.path.splitext(dst)
# PDF as well as EPS: PostScript has no alpha channel, so in the .eps the
# faded earlier poses and the translucent streamlines are flattened to
# opaque. Prefer the .pdf in the manuscript if the build allows it.
fig.savefig(base + ".pdf")
fig.savefig(base + ".eps", format="eps")
print("saved %s (+ .pdf, .eps)" % dst)
print("  grid %dx%dx%d   slice k=%d at z=%.4g m   %d body pts   t_end=%.4g s"
      % (nx, ny, nz, k, zs[k] * NM, len(Bp), t[-1]))
print("  path: start (%.3f, %.3f, %.3f)L  end (%.3f, %.3f, %.3f)L"
      % (cx[0] / BOX, cy[0] / BOX, cz[0] / BOX,
         cx[-1] / BOX, cy[-1] / BOX, cz[-1] / BOX))
print("  |dz| from mid-span: max %.4f L" % (np.max(np.abs(cz - 0.5 * BOX)) / BOX))
print("  view: elev=26 azim=-60, lid-normal vertical   poses %d   markers %d"
      % (len(poses), nmk))
print("  frame: figure (x,y,z) = solver (x,z,y); lid is the figure's z=L face")
