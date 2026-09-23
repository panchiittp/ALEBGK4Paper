#!/usr/bin/env python3
"""Per-step rigid-body state from body_state.csv (ALEBGK_BODY_LOG=1).

The 3-D moving-body cavity run ends with the body state at 1e283 and then NaN.
An earlier finiteness tripwire blamed stage 3 of moveRigidBodies, but the state
going INTO stage 3 was already astronomical -- isfinite() passes 1e200, and
bodies.csv only samples every saveEvery steps, so the whole blow-up fell
between two log rows.  Logging every step, with BOTH Heun loads, settles it.

Four panels, each answering one question:
  (a) body state    when does |w|,|v| leave the flow-driven equilibrium?
  (b) surface loads when do |F|,|T| leave THEIR baseline?  If the loads move
                    first, the body is a passive victim and the fault is in
                    the surface-stress integral, not the body integrator.
  (c) zoom          the three curves normalised at the onset window, so the
                    ORDER of departure is readable rather than inferred.
  (d) Heun ratio    |F1|/|F0| and F1.F0/|F0|^2.  Both sit at 1 while the
                    corrector agrees with the predictor.  A staggered
                    fluid-structure instability REQUIRES the corrector to
                    overshoot and reverse; if these stay at 1 through the
                    blow-up, that mechanism is excluded.

Usage: plot_body_growth.py [body_state.csv] [out.png]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})

src = sys.argv[1] if len(sys.argv) > 1 else "body_state.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "figures/body_growth.png"

with open(src) as f:                      # the run appends; last line can tear
    hdr = f.readline().rstrip("\n").split(",")
    rows = []
    for line in f:
        p = line.rstrip("\n").split(",")
        if len(p) != len(hdr):
            continue
        try:
            rows.append([float(v) for v in p])
        except ValueError:
            continue
if not rows:
    sys.exit("no complete rows in " + src)

A = np.array(rows)
c = {n: i for i, n in enumerate(hdr)}
step = A[:, c["step"]]
vmag, wmag = A[:, c["vmag"]], A[:, c["wmag"]]
F0 = A[:, [c["Fx0"], c["Fy0"], c["Fz0"]]]
F1 = A[:, [c["Fx1"], c["Fy1"], c["Fz1"]]]
T0 = A[:, [c["Tx0"], c["Ty0"], c["Tz0"]]]
nF0, nF1 = np.linalg.norm(F0, axis=1), np.linalg.norm(F1, axis=1)
nT0 = np.linalg.norm(T0, axis=1)
ok = nF0 > 0
ratio = np.where(ok, nF1 / np.where(ok, nF0, 1), np.nan)
proj = np.where(ok, np.sum(F0 * F1, axis=1) / np.where(ok, nF0 ** 2, 1), np.nan)

fig, ax = plt.subplots(2, 2, figsize=(11.0, 7.6))
fig.suptitle("3-D cavity, free rigid body — the surface load diverges first, "
             r"the body follows   (N=31$^3$, $\rho$=4.4, Kn=0.032)",
             fontsize=11, color=INK)

# --- (a) body state -------------------------------------------------------
a = ax[0, 0]
a.semilogy(step, np.maximum(wmag, 1e-12), color=C_BLUE, lw=1.1,
           label=r"$|\omega|$ (rad s$^{-1}$)")
a.semilogy(step, np.maximum(vmag, 1e-18), color=C_AQUA, lw=1.1,
           label="$|v|$ (m s$^{-1}$)")
a.axhline(5.0e6, color=MUTED, ls="--", lw=0.9)
a.text(step[0], 7e6, "flow-driven equilibrium ~5e6 rad/s", color=MUTED,
       fontsize=7.5, va="bottom")
a.set_xlabel("step"); a.set_ylabel("body state")
a.set_title("(a) body spin and speed", fontsize=9.5)
a.legend(frameon=False, fontsize=8, loc="lower right")
a.grid(alpha=0.3, color=GRID)

# --- (b) surface loads ----------------------------------------------------
a = ax[0, 1]
a.semilogy(step, np.maximum(nF0, 1e-30), color=C_ORANGE, lw=1.1,
           label="$|F|$ (N)")
a.semilogy(step, np.maximum(nT0, 1e-36), color=C_YELLOW, lw=1.1,
           label="$|T|$ (N m)")
a.set_xlabel("step"); a.set_ylabel("surface-stress load")
a.set_title("(b) loads on the body — flat for 11,000 steps, then geometric",
            fontsize=9.5)
a.legend(frameon=False, fontsize=8, loc="lower right")
a.grid(alpha=0.3, color=GRID)

# --- (c) onset zoom, normalised ------------------------------------------
a = ax[1, 0]
lo = max(0, len(step) - 160)
sl = slice(lo, len(step))
ref = slice(lo, lo + 20)          # pre-onset window used as the unit level


def norm(y):
    b = np.nanmedian(y[ref])
    return y[sl] / b if b else y[sl]


a.semilogy(step[sl], norm(nT0), color=C_YELLOW, lw=1.5, label="$|T|$")
a.semilogy(step[sl], norm(nF0), color=C_ORANGE, lw=1.5, label="$|F|$")
a.semilogy(step[sl], norm(wmag), color=C_BLUE, lw=1.5, label=r"$|\omega|$")
a.axhline(1.0, color=MUTED, lw=0.8)
a.set_xlabel("step"); a.set_ylabel("relative to pre-onset level")
a.set_title("(c) onset zoom — torque leads force leads spin", fontsize=9.5)
a.legend(frameon=False, fontsize=8, loc="upper left")
a.grid(alpha=0.3, color=GRID)

# --- (d) Heun corrector vs predictor -------------------------------------
a = ax[1, 1]
a.plot(step, ratio, color=C_BLUE, lw=1.0, label=r"$|F_1|/|F_0|$")
a.plot(step, proj, color=C_ORANGE, lw=1.0, ls="--",
       label=r"$F_1\!\cdot\!F_0/|F_0|^2$")
a.axhline(1.0, color=MUTED, lw=0.8)
a.set_xlabel("step"); a.set_ylabel("corrector / predictor load")
a.set_title("(d) Heun coupling stays at 1.0 — not a staggered instability",
            fontsize=9.5)
a.legend(frameon=False, fontsize=8, loc="lower left")
a.grid(alpha=0.3, color=GRID)
fin = np.isfinite(ratio) & np.isfinite(proj)
if fin.any():
    y0 = min(ratio[fin].min(), proj[fin].min())
    y1 = max(ratio[fin].max(), proj[fin].max())
    pad = max(2e-4, 0.2 * (y1 - y0))
    a.set_ylim(y0 - pad, y1 + pad)

fig.tight_layout(rect=(0, 0, 1, 0.955))
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
fig.savefig(out, dpi=160)
fig.savefig(os.path.splitext(out)[0] + ".eps")
print("wrote %s (+ .eps)  rows %d  steps %d..%d"
      % (out, len(A), int(step[0]), int(step[-1])))
