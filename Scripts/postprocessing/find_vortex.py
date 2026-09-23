#!/usr/bin/env python3
"""Locate the primary vortex core in a z-midplane slice of the cavity.

Uses the Graftieaux Gamma-1 criterion, the standard detector for scattered
/ interpolated velocity data:

    Gamma1(P) = (1/N) * sum_M  sin(angle between PM and u(M))

which tends to +-1 at a vortex centre and to 0 in shear or uniform flow.
Unlike "smallest |u|" it cannot be fooled by a quiescent corner, and unlike
a streamfunction extremum it does not care about the integration path or the
lid boundary layer -- both of which gave wrong answers on this cavity.

`selftest` validates the detector against a synthetic Lamb-Oseen vortex at a
known location before any real field is trusted.

Usage:
  find_vortex.py selftest
  find_vortex.py <file.vtk> [exclude_x exclude_y exclude_r]
"""
import sys

import numpy as np


def gamma1(X, Y, UX, UY, radius_pts=6):
    """Gamma-1 field on a uniform grid. radius_pts = neighbourhood radius."""
    ny, nx = X.shape
    r = radius_pts
    off = [(i, j) for i in range(-r, r + 1) for j in range(-r, r + 1)
           if 0 < i * i + j * j <= r * r]
    G = np.zeros_like(X)
    cnt = np.zeros_like(X)
    dx = X[0, 1] - X[0, 0]
    for i, j in off:
        # PM vector is constant for a given offset on a uniform grid
        pmx, pmy = j * dx, i * dx
        pm = np.hypot(pmx, pmy)
        um = np.roll(np.roll(UX, -i, axis=0), -j, axis=1)
        vm = np.roll(np.roll(UY, -i, axis=0), -j, axis=1)
        sp = np.hypot(um, vm)
        # sin of the angle between PM and u(M) = cross(PM, u) / (|PM||u|)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = (pmx * vm - pmy * um) / (pm * sp)
        good = np.isfinite(s)
        G[good] += s[good]
        cnt[good] += 1
    valid = cnt > 0
    G[valid] /= cnt[valid]
    G[~valid] = 0.0
    # the roll-based neighbourhood wraps at the edges; discard that band
    G[:r, :] = G[-r:, :] = G[:, :r] = G[:, -r:] = 0.0
    return G


def core_from_grid(X, Y, UX, UY, radius_pts=6, valid=None):
    """valid: boolean mask of grid points that carry REAL interpolated data.
    Without it the detector happily reports a 'vortex' in the constant
    fill_value region outside the data hull -- which is what the corner hits
    with vorticity exactly 0.0 were."""
    G = gamma1(X, Y, UX, UY, radius_pts)
    if valid is not None:
        from scipy.ndimage import binary_erosion
        core_ok = binary_erosion(valid, np.ones((2 * radius_pts + 1,) * 2))
        G = np.where(core_ok, G, 0.0)
    k = np.unravel_index(np.argmax(np.abs(G)), G.shape)
    dx = X[0, 1] - X[0, 0]
    vort = (np.gradient(UY, dx, axis=1) - np.gradient(UX, dx, axis=0))[k]
    return X[k], Y[k], G[k], vort


def selftest():
    """Lamb-Oseen vortex at a deliberately off-centre, non-grid location."""
    cx0, cy0, Gam, rc = 0.37e-6, 0.62e-6, -2.0e-6, 1.2e-7
    g = np.linspace(0, 1e-6, 121)
    X, Y = np.meshgrid(g, g)
    dx_, dy_ = X - cx0, Y - cy0
    r2 = dx_ ** 2 + dy_ ** 2 + 1e-30
    ut = Gam / (2 * np.pi) * (1 - np.exp(-r2 / rc ** 2)) / np.sqrt(r2)
    UX, UY = -ut * dy_ / np.sqrt(r2), ut * dx_ / np.sqrt(r2)
    # add a lid-like shear and a slow corner, the two things that fooled the
    # other criteria
    UX += 6.0 * np.exp(-((Y - 1e-6) / 8e-8) ** 2)
    cx, cy, G, w = core_from_grid(X, Y, UX, UY)
    err = np.hypot(cx - cx0, cy - cy0)
    print("selftest: true (%.4f, %.4f) um -> found (%.4f, %.4f) um"
          % (cx0 * 1e6, cy0 * 1e6, cx * 1e6, cy * 1e6))
    print("          error %.4f um (grid spacing %.4f um), |Gamma1| = %.3f"
          % (err * 1e6, (g[1] - g[0]) * 1e6, abs(G)))
    ok = err <= 2.5 * (g[1] - g[0])
    print("          %s" % ("PASS" if ok else "FAIL"))
    return ok


def from_vtk(fn, exclude=None):
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_cavity3d_body import read_vtk
    from scipy.interpolate import griddata
    d = read_vtk(fn)
    p, v = d["pts"], d["velocity"]
    m = (np.abs(p[:, 2] - 0.5e-6) < 0.7e-7) & (d["boundary"] < 0.5)
    if exclude is not None:
        m &= np.hypot(p[:, 0] - exclude[0], p[:, 1] - exclude[1]) > exclude[2]
    g = np.linspace(0, 1e-6, 121)
    X, Y = np.meshgrid(g, g)
    UX = griddata((p[m, 0], p[m, 1]), v[m, 0], (X, Y), method="linear")
    UY = griddata((p[m, 0], p[m, 1]), v[m, 1], (X, Y), method="linear")
    valid = np.isfinite(UX) & np.isfinite(UY)
    UX, UY = np.nan_to_num(UX), np.nan_to_num(UY)
    cx, cy, G, w = core_from_grid(X, Y, UX, UY, valid=valid)
    print("%-46s t=%.3e  core=(%.4f, %.4f) um  |G1|=%.3f  vort=%.3e  "
          "omega_eq=%.3e"
          % (os.path.basename(os.path.dirname(fn)), d["t"], cx * 1e6,
             cy * 1e6, abs(G), w, w / 2))
    return cx, cy, w


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        sys.exit(0 if selftest() else 1)
    ex = None
    if len(sys.argv) > 4:
        ex = (float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
    from_vtk(sys.argv[1], ex)
