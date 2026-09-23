#!/usr/bin/env python3
"""Exact rigid-body pose recovery from the solver's own surface particles,
plus solid-body renderers (real cube, high-resolution sphere).

Why this works exactly.  moveRigidBodies (solver/Kernels.cpp) advances every
surface particle of a 3-D body by the same Rodrigues rotation about w-hat
followed by the centre translation, and manageRemove/manageRefill only ever
touch *fluid* particles.  So a body's surface particles keep their identity
and their index order in every VTK snapshot, and the map from snapshot 0 to
snapshot k is a pure rigid motion with the correspondence already known.
Kabsch with identity correspondence therefore returns the pose to machine
precision (measured residual ~5e-14 m against a 150 nm body) -- there is no
need to integrate omega, and no registration ambiguity.
"""
import numpy as np

__all__ = ["body_points", "kabsch", "cube_faces", "cube_section",
           "sphere_mesh", "sphere_facecolors", "visible"]


def body_points(d, box=1e-6, tol=1e-9):
    """Surface particles of the immersed body: flagged boundary but not
    lying on any of the six cavity faces.  Returns (points, index array)."""
    p = d["pts"]
    on_wall = np.zeros(len(p), bool)
    # 2-D clouds carry a single z plane (usually z=0), which coincides with a
    # box face -- testing that axis would mark EVERY particle as "on a wall"
    # and return an empty body. Only test axes the cloud actually spans.
    axes = [k for k in range(3) if p[:, k].max() - p[:, k].min() > tol]
    for k in axes:
        on_wall |= (np.abs(p[:, k]) < tol) | (np.abs(p[:, k] - box) < tol)
    m = (d["boundary"] > 0.5) & ~on_wall
    return p[m], np.where(m)[0]


def kabsch(A, B):
    """Rigid motion taking point set A onto B (same ordering).

    Returns (R, cA, cB, rmsd) with B ~= (A - cA) @ R.T + cB."""
    cA, cB = A.mean(0), B.mean(0)
    H = (A - cA).T @ (B - cB)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    rmsd = float(np.sqrt((((A - cA) @ R.T + cB - B) ** 2).sum(1).mean()))
    return R, cA, cB, rmsd


def rot_angle(R):
    """Rotation angle of R in radians (0 .. pi)."""
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


# ── cube ────────────────────────────────────────────────────────────────
_CUBE_V = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                    for sz in (-1, 1)], float) * 0.5
_CUBE_F = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
           (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
_CUBE_E = [(a, b) for a in range(8) for b in range(a + 1, 8)
           if np.sum(np.abs(_CUBE_V[a] - _CUBE_V[b]) > 1e-12) == 1]


def cube_faces(c, R, L):
    """Six quadrilateral faces of the real cube at pose (c, R)."""
    v = (_CUBE_V * L) @ R.T + c
    return [v[list(f)] for f in _CUBE_F], v


def cube_section(c, R, L, z0):
    """Exact cross-section polygon of the rotated cube at the plane z = z0,
    as the ordered set of cube-edge / plane intersections."""
    v = (_CUBE_V * L) @ R.T + c
    pts = []
    for a, b in _CUBE_E:
        za, zb = v[a, 2], v[b, 2]
        if (za - z0) * (zb - z0) <= 0 and abs(zb - za) > 1e-18:
            s = (z0 - za) / (zb - za)
            if -1e-9 <= s <= 1 + 1e-9:
                pts.append(v[a] + s * (v[b] - v[a]))
    if len(pts) < 3:
        return None
    P = np.array(pts)[:, :2]
    P = np.unique(np.round(P, 15), axis=0)
    ctr = P.mean(0)
    return P[np.argsort(np.arctan2(P[:, 1] - ctr[1], P[:, 0] - ctr[0]))]


# ── sphere ──────────────────────────────────────────────────────────────
def sphere_mesh(c, R, r, nu=96, nv=48):
    """High-resolution sphere surface at pose (c, R).

    The mesh is built in the BODY frame and then rotated, so the returned
    (theta, phi) parameterisation is painted onto the body: rotating the
    body carries the pattern with it, which is what makes the spin visible
    on an otherwise featureless ball."""
    th = np.linspace(0, 2 * np.pi, nu)
    ph = np.linspace(0, np.pi, nv)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    b = np.stack([r * np.sin(PH) * np.cos(TH),
                  r * np.sin(PH) * np.sin(TH),
                  r * np.cos(PH)], axis=-1)
    w = b @ R.T + c
    n = b / r                                   # body-frame outward normal
    return w, n @ R.T, TH, PH


def sphere_facecolors(TH, PH, c_a, c_b, nlon=8, nlat=6):
    """Body-frame chequerboard: the only way a rotating sphere reads as
    rotating in a still frame."""
    k = ((np.floor(TH / (2 * np.pi) * nlon) +
          np.floor(PH / np.pi * nlat)) % 2)
    out = np.empty(TH.shape + (4,))
    out[k == 0] = c_a
    out[k == 1] = c_b
    return out[:-1, :-1]


def visible(normals, elev, azim):
    """Front-facing test for markers drawn on a solid body: matplotlib's
    3-D painter has no z-buffer, so a marker on the far side would
    otherwise show through the surface."""
    e, a = np.radians(elev), np.radians(azim)
    v = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return normals @ v > 0.10
