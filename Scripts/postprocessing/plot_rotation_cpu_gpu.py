#!/usr/bin/env python3
"""Overlay two runs of the free-body rotation over the same time window.

NOTE ON WHAT THIS DOES AND DOES NOT SHOW. If the two directories were
produced at the SAME grid and Nv, this is a backend equivalence test: the
GPU path steps the fluid on the device but hands the body coupling back to
the same host moveRigidBodies every step (the v2 hybrid), so the spin
histories must coincide, and any gap means the download/upload round-trip
is dropping body state. If the grids DIFFER, it is only a resolution
comparison -- the curves are expected to separate, and nothing here tests
the backends against each other. The panel titles are labelled with each
run's grid so the two cases cannot be confused.

Usage: plot_rotation_cpu_gpu.py <run_a_dir> <run_b_dir> <out.png>
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})


def load(d):
    b = np.genfromtxt(os.path.join(d, "bodies.csv"), delimiter=",",
                      names=True)
    g = lambda k: np.atleast_1d(b[k])
    return (g("t"), g("omx"), g("omy"), g("angVel"),
            g("cx"), g("cy"), g("cz"))


def main():
    cpu_dir, gpu_dir, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    tc, cx_, cy_, cz_, ccx, ccy, ccz = load(cpu_dir)
    tg, gx_, gy_, gz_, gcx, gcy, gcz = load(gpu_dir)

    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.9))

    a = ax[0]
    a.plot(tc * 1e9, cz_ * 1e-6, "-", color=C_BLUE, lw=2.4,
           label="CPU  15$^3$ / Nv=10")
    a.plot(tg * 1e9, gz_ * 1e-6, "--", color=C_ORANGE, lw=2.0,
           label="GPU  21$^3$ / Nv=20")
    a.axhline(-3.93, color=MUTED, lw=1.0, ls=":")
    a.annotate(r"vorticity/2 = $-3.93$", xy=(0.42, 0.12),
               xycoords="axes fraction", fontsize=8, color=MUTED)
    a.set_ylabel(r"$\omega_z$  ($10^6$ rad/s)")
    a.set_title("spin-up toward the equilibrium set by the flow")

    a = ax[1]
    for t, wx, wy, col, sty, lab in [
            (tc, cx_, cy_, C_BLUE, "-", "CPU"),
            (tg, gx_, gy_, C_ORANGE, "--", "GPU")]:
        a.plot(t * 1e9, wx * 1e-3, sty, color=col, lw=1.8,
               label=r"%s  $\omega_x$" % lab)
        a.plot(t * 1e9, wy * 1e-3, sty, color=col, lw=1.0, alpha=0.55,
               label=r"%s  $\omega_y$" % lab)
    a.set_ylabel(r"$\omega_{x},\ \omega_{y}$  ($10^3$ rad/s)")
    a.set_title(r"off-axis spin (3 orders below $\omega_z$)")

    a = ax[2]
    a.plot((ccx - ccx[0]) * 1e9, (ccy - ccy[0]) * 1e9, "-", color=C_BLUE,
           lw=2.4, label="CPU")
    a.plot((gcx - gcx[0]) * 1e9, (gcy - gcy[0]) * 1e9, "--", color=C_ORANGE,
           lw=2.0, label="GPU")
    a.set_xlabel(r"$\Delta x$  (nm)")
    a.set_ylabel(r"$\Delta y$  (nm)")
    a.set_title("centre track over the same window")

    for a in ax[:2]:
        a.set_xlabel("t  (ns)")
    for a in ax:
        a.grid(True, color=GRID, lw=0.6)
        a.set_axisbelow(True)
        a.legend(frameon=False, fontsize=8)

    same = (len(tc) == len(tg) and np.allclose(tc, tg))
    fig.suptitle("Free rigid body in the 3-D driven cavity — same 15 ns "
                 "window, CPU 15$^3$/Nv=10 vs GPU 21$^3$/Nv=20\n"
                 "(different grids: a RESOLUTION comparison, not a backend "
                 "equivalence test)" if not same else
                 "Free rigid body in the 3-D driven cavity — CPU vs GPU "
                 "backend at identical resolution",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    print("saved", dst)
    print("CPU  omega_z(end) = %.4e rad/s" % cz_[-1])
    print("GPU  omega_z(end) = %.4e rad/s" % gz_[-1])
    print("relative difference = %.2f %%"
          % (100 * abs(cz_[-1] - gz_[-1]) / abs(gz_[-1])))


if __name__ == "__main__":
    main()
