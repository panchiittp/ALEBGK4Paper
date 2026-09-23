// ===========================================================================
// MLSTransport.cpp - Moving-least-squares semi-Lagrangian transport (CPU).
//
// Split into a compute pass (writes p.gt, reads neighbours' p.g) and a
// commit pass (gt -> g), so every particle's update reads the previous
// time level regardless of loop order or threading.
// ===========================================================================
#include "Config.hpp"
#include "Types.hpp"
#include "MLSBasis.hpp"
#include "Geometry.hpp"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>
#include <algorithm>
#include <limits>

static void solveLinearN(double *A, double *b, double *sol, int N)
{
    double aug[MLS_MAX_N][MLS_MAX_N + 1];
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j) aug[i][j] = A[i * N + j];
        aug[i][N] = b[i];
    }
    // Forward elimination with partial pivoting
    for (int col = 0; col < N; ++col)
    {
        int piv = col;
        double mx = std::fabs(aug[col][col]);
        for (int row = col + 1; row < N; ++row)
        {
            if (std::fabs(aug[row][col]) > mx)
            {
                mx = std::fabs(aug[row][col]);
                piv = row;
            }
        }
        if (piv != col)
            for (int j = 0; j <= N; ++j)
                std::swap(aug[col][j], aug[piv][j]);

        double d = aug[col][col];
        if (std::fabs(d) < 1.0e-30) continue;

        for (int row = col + 1; row < N; ++row)
        {
            double f = aug[row][col] / d;
            for (int j = col; j <= N; ++j)
                aug[row][j] -= f * aug[col][j];
        }
    }
    // Back substitution
    for (int i = N - 1; i >= 0; --i)
    {
        double s = aug[i][N];
        for (int j = i + 1; j < N; ++j) s -= aug[i][j] * sol[j];
        sol[i] = (std::fabs(aug[i][i]) > 1.0e-30) ? s / aug[i][i] : 0.0;
    }
}

// =====================================================================
// initWallData — store each boundary particle's wall state once.
//
// Box-wall particles get their normal/velocity/temperature from
// getWallNormal; problem-specific setup (velocity profiles, rigid-body
// surfaces) may override the stored fields afterwards. The BC kernels
// and the transport gating read only the stored state, so box walls,
// profiled walls, and body surfaces share one code path.
// =====================================================================
static inline double wallCN(const Particle &p, double vx, double vy)
{
    double cn = (vx - p.wUx) * p.wnx + (vy - p.wUy) * p.wny;
    if (p.dim == 3) cn += (p.uz - p.wUz) * p.wnz;
    return cn;
}

// =====================================================================
// 3a. mlsTransportCompute — MLS semi-Lagrangian characteristic transport
//     Reads neighbours' g (time level n), writes this particle's gt.
//
// On fully periodic domains (cp.periodic) the neighbour displacement to
// the foot-point uses the minimum-image convention, so characteristics
// crossing a seam interpolate from the opposite side of the domain.
// =====================================================================
void mlsTransportCompute(std::vector<Particle> &P,
                         double dt, const CalcParameters &cp,
                         DomainBoundary dom,
                         BoundaryConditions BC,
                         int mlsOrder, bool onlyFlagged)
{
    int N = static_cast<int>(P.size());
    double h = cp.radius;
    int cLo, cHi;
    alebgk::chunk(N, &cLo, &cHi);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int idx = cLo; idx < cHi; ++idx)
    {
        Particle &p = P[idx];
        if (!p.validg) continue;
        if (onlyFlagged && !p.moodFlag) continue;

        int Nv     = p.Nv_local;
        double dv  = p.dv_local;
        double vlo = p.vlo;
        int ndim   = p.dim;
        // Wall particles have one-sided stencils; a one-sided quadratic
        // fit oscillates and blows up (seen on RT's y-walls). Use the
        // linear basis there regardless of the requested order.
        int order_p = (p.boundary || onlyFlagged) ? 1 : mlsOrder;
        int poly   = basisSize(ndim, order_p);
        int Nv2    = Nv * Nv;
        int nn     = static_cast<int>(p.neighindex.size());

        // Default: copy g -> gt (outgoing BC nodes stay unchanged)
        int ntot = static_cast<int>(p.g.size());
        for (int k = 0; k < ntot; ++k) p.gt[k] = p.g[k];

        double A[MLS_MAX_N * MLS_MAX_N];
        double b1[MLS_MAX_N], b2[MLS_MAX_N];
        double sol1[MLS_MAX_N], sol2[MLS_MAX_N];
        double basis[MLS_MAX_N];

        if (ndim == 3)
        {
            // Full 3-D velocity grid, single distribution. Foot-points use
            // the grid vz (the true z-characteristic), not a bulk-Uz proxy.
            for (int kv = 0; kv < Nv; ++kv)
            {
                double vz = vlo + kv * dv;
                for (int jv = 0; jv < Nv; ++jv)
                {
                    double vy = vlo + jv * dv;
                    for (int iv = 0; iv < Nv; ++iv)
                    {
                        double vx = vlo + iv * dv;
                        int lin = iv + Nv * (jv + Nv * kv);

                        if (p.boundary)
                        {
                            double cn = (vx - p.wUx) * p.wnx
                                      + (vy - p.wUy) * p.wny
                                      + (vz - p.wUz) * p.wnz;
                            if (cn >= 0.0) continue;
                        }

                        double xf = p.x - (vx - p.meshVx) * dt;
                        double yf = p.y - (vy - p.meshVy) * dt;
                        double zf = p.z - (vz - p.meshVz) * dt;

                        for (int q = 0; q < poly * poly; ++q) A[q] = 0.0;
                        for (int q = 0; q < poly; ++q) b1[q] = 0.0;
                        bool hasNeigh = false;

                        for (int ni = 0; ni < nn; ++ni)
                        {
                            const Particle &pn = P[p.neighindex[ni]];
                            if (!pn.validg) continue;
                            double dx_n = pn.x - xf;
                            double dy_n = pn.y - yf;
                            double dz_n = pn.z - zf;
                            if (cp.periodicX) dx_n = minImage(dx_n, cp.Lx);
                            if (cp.periodicY) dy_n = minImage(dy_n, cp.Ly);
                            if (cp.periodicZ) dz_n = minImage(dz_n, cp.Lz);
                            double r2 = dx_n*dx_n + dy_n*dy_n + dz_n*dz_n;
                            double w = mlsWeightSq(r2, h);
                            if (w < 1.0e-30) continue;
                            fillBasis(dx_n, dy_n, dz_n, 3, h, basis, order_p);
                            for (int p1 = 0; p1 < poly; ++p1)
                                for (int p2 = 0; p2 < poly; ++p2)
                                    A[p1*poly + p2] += w * basis[p1] * basis[p2];
                            double gn = pn.g[lin];   // Fixed uniform grid
                            for (int q = 0; q < poly; ++q)
                                b1[q] += w * basis[q] * gn;
                            hasNeigh = true;
                        }
                        if (!hasNeigh) continue;
                        solveLinearN(A, b1, sol1, poly);
                        p.gt[lin] = sol1[0];
                    }
                }
            }

            double mPos = 0.0, mNeg = 0.0;
            for (int k = 0; k < ntot; ++k)
            {
                double v = p.gt[k];
                if (v >= 0.0) mPos += v; else mNeg += v;
            }
            if (mNeg < -1.0e-30 && mPos > 1.0e-30 && (mPos + mNeg) > 0.0)
            {
                double s = (mPos + mNeg) / mPos;
                for (int k = 0; k < ntot; ++k)
                    p.gt[k] = (p.gt[k] < 0.0) ? 0.0 : p.gt[k] * s;
            }
            continue;
        }

        // ── 2-D Chu reduction (g1, g2) ──────────────────────────────────
        for (int jv = 0; jv < Nv; ++jv)
        {
            double vy = vlo + jv * dv;
            for (int iv = 0; iv < Nv; ++iv)
            {
                double vx = vlo + iv * dv;
                int lin = iv + Nv * jv;

                // Boundary particles: only update incoming (cn < 0) nodes
                if (p.boundary)
                {
                    double cn = wallCN(p, vx, vy);
                    if (cn >= 0.0) continue;
                }

                // Foot-point of characteristic
                double xf = p.x - (vx - p.meshVx) * dt;
                double yf = p.y - (vy - p.meshVy) * dt;

                // Zero the normal equations
                for (int q = 0; q < poly * poly; ++q) A[q] = 0.0;
                for (int q = 0; q < poly; ++q) b1[q] = b2[q] = 0.0;
                bool hasNeigh = false;

                for (int ni = 0; ni < nn; ++ni)
                {
                    int nidx = p.neighindex[ni];
                    const Particle &pn = P[nidx];

                    if (!pn.validg) continue;

                    double dx_n = pn.x - xf;
                    double dy_n = pn.y - yf;
                    if (cp.periodicX) dx_n = minImage(dx_n, cp.Lx);
                    if (cp.periodicY) dy_n = minImage(dy_n, cp.Ly);
                    double r2 = dx_n * dx_n + dy_n * dy_n;
                    double w = mlsWeightSq(r2, h);
                    if (w < 1.0e-30) continue;

                    fillBasis(dx_n, dy_n, 0.0, 2, h, basis, order_p);

                    int nb = poly;
                    for (int p1 = 0; p1 < nb; ++p1)
                        for (int p2 = 0; p2 < nb; ++p2)
                            A[p1 * nb + p2] += w * basis[p1] * basis[p2];

                    // Map velocity node to neighbour's grid
                    int Nv_n     = pn.Nv_local;
                    double dv_n  = pn.dv_local;
                    double vlo_n = pn.vlo;
                    int iv_n = std::max(0, std::min(Nv_n - 1,
                               static_cast<int>(std::round((vx - vlo_n) / dv_n))));
                    int jv_n = std::max(0, std::min(Nv_n - 1,
                               static_cast<int>(std::round((vy - vlo_n) / dv_n))));
                    int lin_n = iv_n + Nv_n * jv_n;
                    int Nv2_n = Nv_n * Nv_n;

                    double gn1 = pn.g[lin_n];
                    double gn2 = pn.g[lin_n + Nv2_n];

                    for (int q = 0; q < nb; ++q)
                    {
                        b1[q] += w * basis[q] * gn1;
                        b2[q] += w * basis[q] * gn2;
                    }
                    hasNeigh = true;
                }

                if (!hasNeigh) continue;  // keep old g

                solveLinearN(A, b1, sol1, poly);
                solveLinearN(A, b2, sol2, poly);

                p.gt[lin]       = sol1[0];
                p.gt[lin + Nv2] = sol2[0];
            }
        }

        // Mass-conserving positivity fix for the g1 layer
        {
            double mPos = 0.0, mNeg = 0.0;
            for (int k = 0; k < Nv2; ++k)
            {
                double v = p.gt[k];
                if (v >= 0.0) mPos += v;
                else          mNeg += v;
            }
            if (mNeg < -1.0e-30 && mPos > 1.0e-30 && (mPos + mNeg) > 0.0)
            {
                double s = (mPos + mNeg) / mPos;
                for (int k = 0; k < Nv2; ++k)
                {
                    if (p.gt[k] < 0.0)
                    {
                        p.gt[k]       = 0.0;
                        p.gt[k + Nv2] = 0.0;
                    }
                    else
                    {
                        p.gt[k]       *= s;
                        p.gt[k + Nv2] *= s;
                    }
                }
            }
        }
    }
}

// checkTransportPAD — inspect the freshly transported gt (before commit,
// so the pre-transport g is still available to neighbours) and flag
// particles whose transported state is inadmissible: either PAD (Physical
// Admissibility -- non-finite or non-positive mass) or NAD (Numerical
// Admissibility -- the reconstructed density is a new extremum outside
// the range bracketed by this particle's own neighbours, the classic MOOD
// discrete-maximum-principle check). PAD alone only catches gross failure
// and never trips on a smooth, positive, slowly-growing overshoot -- which
// is exactly the failure mode a missing NAD check lets through silently
// for many steps until it has compounded into a genuine blow-up.
// Flagged particles are re-transported with the linear basis by the
// onlyFlagged pass of mlsTransportCompute.
int checkTransportPAD(std::vector<Particle> &P)
{
    int nfail = 0;
    int N = static_cast<int>(P.size());
    // Chunked like every other per-step kernel: only this rank's owned
    // particles carry a transported gt (decomposed MPI frees ghost gt),
    // and only their flags feed the chunked linear-fallback re-transport.
    int cLo, cHi;
    alebgk::chunk(N, &cLo, &cHi);
    for (int idx = cLo; idx < cHi; ++idx)
    {
        Particle &p = P[idx];
        p.moodFlag = false;
        if (!p.validg || p.boundary) continue;
        int Nv = p.Nv_local;
        int ndim = p.dim;
        int nmass = (ndim == 3) ? Nv * Nv * Nv : Nv * Nv;
        double dvN = (ndim == 3) ? p.dv_local * p.dv_local * p.dv_local
                                 : p.dv_local * p.dv_local;
        double m = 0.0;
        bool bad = false;
        for (int k2 = 0; k2 < nmass; ++k2)
        {
            double v = p.gt[k2];
            if (!std::isfinite(v)) { bad = true; break; }
            m += v;
        }
        if (bad || m <= 0.0) { p.moodFlag = true; nfail++; continue; }

        // NAD: compare the properly-weighted density this transport step
        // produced against the [min, max] of the neighbourhood's CURRENT
        // (pre-transport) densities. A small relative slack keeps this
        // from over-firing on a genuine, physically sharp local extremum.
        double rho_new = m * dvN;
        double rhoMin = std::numeric_limits<double>::infinity();
        double rhoMax = -std::numeric_limits<double>::infinity();
        for (int ni : p.neighindex)
        {
            const Particle &pn = P[ni];
            if (!pn.validg) continue;
            rhoMin = std::min(rhoMin, pn.rho);
            rhoMax = std::max(rhoMax, pn.rho);
        }
        if (std::isfinite(rhoMin) && std::isfinite(rhoMax))
        {
            double slack = 0.1 * std::max(rhoMax - rhoMin, 1.0e-12) + 1.0e-9;
            if (rho_new < rhoMin - slack || rho_new > rhoMax + slack)
            {
                p.moodFlag = true;
                nfail++;
            }
        }
    }
    return nfail;
}

// =====================================================================
// 3b. mlsTransportCommit — copy gt -> g after ALL particles are computed
// =====================================================================
void mlsTransportCommit(std::vector<Particle> &P)
{
    int N = static_cast<int>(P.size());
    int cLo, cHi;
    alebgk::chunk(N, &cLo, &cHi);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int idx = cLo; idx < cHi; ++idx)
    {
        Particle &p = P[idx];
        if (!p.validg) continue;
        int ntot = static_cast<int>(p.g.size());
        for (int k = 0; k < ntot; ++k) p.g[k] = p.gt[k];
    }
}

// =====================================================================
// 3c. applyGravity — constant body acceleration a = (0, gY) as an exact
//     velocity-space advection: f(t+dt, vy) = f(t, vy - gY*dt), applied
//     to every layer by linear interpolation along the vy axis. The
//     per-step shift |gY|*dt is far smaller than dv, so linear
//     interpolation is accurate; samples beyond the grid clamp to the
//     edge (Maxwellian tails ~ 0 there).
// =====================================================================
