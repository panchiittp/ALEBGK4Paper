#pragma once

#include "Types.hpp"
#include <cmath>

/* ================================================================== */
/*  Inline math helpers for the BGK kinetic solver (serial CPU)         */
/* ================================================================== */

/* ------------------------------------------------------------------
   0. Fully periodic problems (KH, Taylor-Green, RT, KH-3Layer)
   ------------------------------------------------------------------ */
ALEBGK_HD inline bool isPeriodicProblem(ProblemID pid)
{
    return pid == ProblemID::KelvinHelmholtz ||
           pid == ProblemID::TaylorGreen     ||
           pid == ProblemID::RayleighTaylor  ||
           pid == ProblemID::KHLayered       ||
           pid == ProblemID::ShearLayer;
}

/* Per-axis periodicity. KH, Taylor-Green, KH-3Layer: fully periodic.
   Rayleigh-Taylor: periodic across (x, and z in 3D), diffuse walls in y
   (the gravity axis). */
inline void problemPeriodicity(ProblemID pid, int ndim,
                               bool *px, bool *py, bool *pz)
{
    if (pid == ProblemID::RayleighTaylor)
    {
        *px = true;
        *py = false;
        *pz = (ndim == 3);
    }
    else if (isPeriodicProblem(pid))
    {
        *px = *py = true;
        *pz = (ndim == 3);
    }
    else
    {
        *px = *py = *pz = false;
    }
}

/* Minimum-image wrap of a displacement component on a periodic axis */
ALEBGK_HD inline double minImage(double d, double L)
{
    return d - L * std::round(d / L);
}

/* ------------------------------------------------------------------
   1. Adaptive velocity-grid sizing
   ------------------------------------------------------------------ */
inline void computeAdaptiveGrid(double ux, double uy, double uz,
                                double T, double R, double cutoff,
                                int NvMin, int NvMax,
                                double* vlo, double* vhi,
                                int* Nv_out, double* dv_out)
{
    (void)uz;   /* uz reserved for 3-D adaptive grids */
    double sigma    = std::sqrt(R * T);
    double halfSpan = cutoff * sigma;

    /* Centre on the dominant macroscopic velocity component */
    double centre = (std::fabs(ux) > std::fabs(uy)) ? ux : uy;

    *vlo = centre - halfSpan;
    *vhi = centre + halfSpan;

    /* Target resolution: ~4 points per sigma */
    int Nv_target = static_cast<int>(std::ceil(2.0 * halfSpan / (sigma / 4.0)));

    /* Clamp to [NvMin, NvMax] */
    if (Nv_target < NvMin) Nv_target = NvMin;
    if (Nv_target > NvMax) Nv_target = NvMax;

    /* Make even for symmetric quadrature */
    if (Nv_target % 2 != 0) Nv_target += 1;

    *Nv_out = Nv_target;
    *dv_out = (*vhi - *vlo) / static_cast<double>(Nv_target - 1);
}

/* ------------------------------------------------------------------
   2. Maxwellian reduced distributions (2-D projection)
   ------------------------------------------------------------------ */
ALEBGK_HD inline double maxwellianG1(double vx, double vy,
                           double ux, double uy,
                           double T, double rho, double R)
{
    double twoRT = 2.0 * R * T;
    double exponent = -((vx - ux) * (vx - ux) + (vy - uy) * (vy - uy)) / twoRT;
    return rho / (M_PI * twoRT) * std::exp(exponent);
}

/* ------------------------------------------------------------------
   3. G2 = 0.5 * R * T * G1  (energy moment)
   ------------------------------------------------------------------ */
ALEBGK_HD inline double maxwellianG2(double vx, double vy,
                           double ux, double uy,
                           double T, double rho, double R)
{
    return 0.5 * R * T * maxwellianG1(vx, vy, ux, uy, T, rho, R);
}

/* ------------------------------------------------------------------
   4. Full 3-D Maxwellian on a resolved (vx,vy,vz) velocity grid.
      3-D physical space uses no Chu reduction: a single distribution f
      carries every moment. Normalised so ∫M dv³ = rho with Θ = 2 R T:
          M = rho / (π Θ)^{3/2} · exp(-|v-u|² / Θ)
   ------------------------------------------------------------------ */
ALEBGK_HD inline double maxwellian3D(double vx, double vy, double vz,
                           double ux, double uy, double uz,
                           double T, double rho, double R)
{
    double twoRT = 2.0 * R * T;
    double exponent = -((vx - ux) * (vx - ux)
                      + (vy - uy) * (vy - uy)
                      + (vz - uz) * (vz - uz)) / twoRT;
    double norm = std::pow(M_PI * twoRT, 1.5);
    return rho / norm * std::exp(exponent);
}

/* ------------------------------------------------------------------
   5. Compactly-supported weight for MLS interpolation
   ------------------------------------------------------------------ */
ALEBGK_HD inline double mlsWeight(double r, double h)
{
    if (r >= h) return 0.0;
    double q = r / h;
    double w = 1.0 - q * q;       // (1 - q^2)
    return w * w;                  // (1 - q^2)^2
}

// Same weight from the SQUARED distance. The callers all computed
// r = sqrt(dx^2+dy^2+dz^2) purely to hand it to mlsWeight, which squared it
// straight back -- and nothing else in the neighbour loop needs r, since
// fillBasis takes the components. On Turing, FP64 sqrt runs at 1/32 rate and
// this fires once per neighbour per velocity node, so it was a large slice of
// the hottest kernel in the solver bought for nothing. Skipping the round
// trip is also very slightly MORE accurate.
ALEBGK_HD inline double mlsWeightSq(double r2, double h)
{
    double q2 = r2 / (h * h);
    if (q2 >= 1.0) return 0.0;
    double w = 1.0 - q2;
    return w * w;
}

/* ------------------------------------------------------------------
   6. Fill MLS basis vector (coordinates normalised by the support h)
      order 1 (linear):    2-D -> [1, sx, sy]            (3 terms)
                           3-D -> [1, sx, sy, sz]        (4 terms)
      order 2 (quadratic): 2-D -> [1, sx, sy, sx2, sxsy, sy2] (6 terms)
                           (3-D quadratic not implemented -- falls back
                            to linear)
   ------------------------------------------------------------------ */
ALEBGK_HD inline int fillBasis(double dx, double dy, double dz,
                     int ndim, double h, double basis[], int order = 1)
{
    double sx = dx / h;
    double sy = dy / h;
    double sz = dz / h;

    basis[0] = 1.0;
    basis[1] = sx;
    basis[2] = sy;

    if (ndim == 2)
    {
        if (order >= 2)
        {
            basis[3] = sx * sx;
            basis[4] = sx * sy;
            basis[5] = sy * sy;
            return 6;
        }
        return 3;
    }
    else /* ndim == 3 */
    {
        basis[3] = sz;
        if (order >= 2)
        {
            basis[4] = sx * sx;
            basis[5] = sy * sy;
            basis[6] = sz * sz;
            basis[7] = sx * sy;
            basis[8] = sx * sz;
            basis[9] = sy * sz;
            return 10;
        }
        return 4;
    }
}

/* Number of MLS basis terms for a given dimension and order */
ALEBGK_HD inline int basisSize(int ndim, int order)
{
    if (ndim == 2) return (order >= 2) ? 6 : 3;
    return (order >= 2) ? 10 : 4;
}

/* ------------------------------------------------------------------
   7. Sanity / PAD check on a particle
      Returns false (i.e. "failed") when any field is unphysical.
   ------------------------------------------------------------------ */
inline bool padCheck(Particle& p)
{
    if (p.rho < 1.0e-10) return false;
    if (p.T   < 1.0e-6)  return false;

    // Positivity of the mass-carrying distribution: g1 (first Nv²) in 2-D,
    // the whole single distribution (Nv³) in 3-D.
    int nmass = (p.dim == 3) ? p.Nv_local * p.Nv_local * p.Nv_local
                             : p.Nv_local * p.Nv_local;
    for (int k = 0; k < nmass; ++k)
    {
        if (p.g[k] < -1.0e-14) return false;
    }
    return true;
}

/* ------------------------------------------------------------------
   8. Detect which domain wall a boundary particle lies on and return
      the outward-pointing normal, wall velocity, and wall temperature.
   ------------------------------------------------------------------ */
inline void getWallNormal(Particle& p,
                          DomainBoundary& dom,
                          BoundaryConditions& BC,
                          double* nx, double* ny, double* nz,
                          double* Uwx, double* Uwy, double* Uwz,
                          double* Tw)
{
    constexpr double eps = 1.0e-13;

    *nx  = 0.0;  *ny  = 0.0;  *nz  = 0.0;
    *Uwx = 0.0;  *Uwy = 0.0;  *Uwz = 0.0;
    double Tw_sum = 0.0;
    int nHit = 0;

    // A particle on an EDGE or CORNER of the box touches two or three faces
    // at once (e.g. x==xleft AND z==zfront on a vertical edge). 3-D has
    // genuine edges -- O(Nx) particles, not the O(1) point-corners 2-D has
    // -- and the lid meets FOUR of them (vs. 2-D's two corner points), so
    // this is exercised far more heavily in 3-D. Accumulate every matching
    // face instead of returning on the first, then normalise: each face
    // contributes its own outward normal component and wall state.
    auto hit = [&](double n_add_x, double n_add_y, double n_add_z,
                  const UBC &u, double tw_override) {
        *nx += n_add_x; *ny += n_add_y; *nz += n_add_z;
        *Uwx += u.Ux; *Uwy += u.Uy; *Uwz += u.Uz;
        Tw_sum += (tw_override > 0.0) ? tw_override : BC.Tw;
        ++nHit;
    };

    /* Left wall (x == xleft) */
    if (std::fabs(p.x - dom.xleft) < eps)
        hit(1.0, 0.0, 0.0, BC.Left, -1.0);
    /* Right wall (x == xright) */
    if (std::fabs(p.x - dom.xright) < eps)
        hit(-1.0, 0.0, 0.0, BC.Right, -1.0);
    /* Top wall (y == ytop) */
    if (std::fabs(p.y - dom.ytop) < eps)
        hit(0.0, -1.0, 0.0, BC.Top, BC.TwTop);
    /* Bottom wall (y == ybottom) */
    if (std::fabs(p.y - dom.ybottom) < eps)
        hit(0.0, 1.0, 0.0, BC.Bottom, -1.0);
    /* Front wall (z == zfront) */
    if (std::fabs(p.z - dom.zfront) < eps)
        hit(0.0, 0.0, -1.0, BC.Front, -1.0);
    /* Back wall (z == zback) */
    if (std::fabs(p.z - dom.zback) < eps)
        hit(0.0, 0.0, 1.0, BC.Back, -1.0);

    if (nHit == 0) { *Tw = BC.Tw; return; }
    if (nHit == 1) { *Tw = Tw_sum; return; }

    // Edge/corner: average the accumulated wall velocities and
    // temperatures over the faces touched, then re-normalise the summed
    // normal to a unit vector (the faces are axis-aligned and mutually
    // orthogonal, so the sum is already the correct bisector direction).
    double nlen = std::sqrt((*nx) * (*nx) + (*ny) * (*ny) + (*nz) * (*nz));
    if (nlen > eps) { *nx /= nlen; *ny /= nlen; *nz /= nlen; }
    *Uwx /= nHit; *Uwy /= nHit; *Uwz /= nHit;
    *Tw = Tw_sum / nHit;
}
