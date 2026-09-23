// ===========================================================================
// DiffuseWall.cpp - Diffuse-reflection wall boundary condition.
//
// The reflected Maxwellian is scaled by rho_w = -intM/intP so the discrete
// net mass flux through each wall particle is exactly zero.
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

void applyBoundaryBC(std::vector<Particle> &P,
                     double R, DomainBoundary dom, BoundaryConditions BC)
{
    int N = static_cast<int>(P.size());
    int cLo, cHi;
    alebgk::chunk(N, &cLo, &cHi);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 16)
#endif
    for (int idx = cLo; idx < cHi; ++idx)
    {
        Particle &p = P[idx];
        if (!p.boundary) continue;

        int Nv     = p.Nv_local;
        double dv  = p.dv_local;
        double vlo = p.vlo;
        int Nv2    = Nv * Nv;

        double nx = p.wnx, ny = p.wny, nz = p.wnz;
        double Uwx = p.wUx, Uwy = p.wUy, Uwz = p.wUz;
        double Tw = p.wT;
        double RT2w = 2.0 * R * Tw;

        if (p.dim == 3)
        {
            double dv3 = dv * dv * dv;
            double norm = std::pow(M_PI * RT2w, 1.5);
            double intM = 0.0, intP = 0.0;
            for (int kv = 0; kv < Nv; ++kv)
            {
                double vz = vlo + kv * dv;
                for (int jv = 0; jv < Nv; ++jv)
                {
                    double vy = vlo + jv * dv;
                    for (int iv = 0; iv < Nv; ++iv)
                    {
                        double vx = vlo + iv * dv;
                        double cn = (vx-Uwx)*nx + (vy-Uwy)*ny + (vz-Uwz)*nz;
                        int lin = iv + Nv * (jv + Nv * kv);
                        if (cn < 0.0)
                            intM += cn * p.g[lin] * dv3;
                        else if (cn > 0.0)
                        {
                            double Mw = std::exp(-((vx-Uwx)*(vx-Uwx)
                                                 + (vy-Uwy)*(vy-Uwy)
                                                 + (vz-Uwz)*(vz-Uwz)) / RT2w)
                                      / norm;
                            intP += cn * Mw * dv3;
                        }
                    }
                }
            }
            double rho_w = (intP > 1.0e-30) ? (-intM / intP) : 1.0;
            for (int kv = 0; kv < Nv; ++kv)
            {
                double vz = vlo + kv * dv;
                for (int jv = 0; jv < Nv; ++jv)
                {
                    double vy = vlo + jv * dv;
                    for (int iv = 0; iv < Nv; ++iv)
                    {
                        double vx = vlo + iv * dv;
                        double cn = (vx-Uwx)*nx + (vy-Uwy)*ny + (vz-Uwz)*nz;
                        if (cn > 0.0)
                        {
                            int lin = iv + Nv * (jv + Nv * kv);
                            p.g[lin] = rho_w
                                     * std::exp(-((vx-Uwx)*(vx-Uwx)
                                                + (vy-Uwy)*(vy-Uwy)
                                                + (vz-Uwz)*(vz-Uwz)) / RT2w)
                                     / norm;
                        }
                    }
                }
            }
            p.rho = rho_w; p.ux = Uwx; p.uy = Uwy; p.uz = Uwz;
            p.T = Tw; p.p = rho_w * R * Tw;
            continue;
        }

        // ── 2-D Chu-reduced diffuse reflection ──────────────────────────
        double dv2 = dv * dv;
        double intM = 0.0, intP = 0.0;
        for (int jv = 0; jv < Nv; ++jv)
        {
            double vy = vlo + jv * dv;
            for (int iv = 0; iv < Nv; ++iv)
            {
                double vx = vlo + iv * dv;
                double cn = (vx - Uwx) * nx + (vy - Uwy) * ny;
                int lin = iv + Nv * jv;

                if (cn < 0.0)
                    intM += cn * p.g[lin] * dv2;
                if (cn > 0.0)
                {
                    double Mw = std::exp(-((vx - Uwx) * (vx - Uwx)
                                         + (vy - Uwy) * (vy - Uwy)) / RT2w)
                              / (M_PI * RT2w);
                    intP += cn * Mw * dv2;
                }
            }
        }

        double rho_w = (intP > 1.0e-30) ? (-intM / intP) : 1.0;

        for (int jv = 0; jv < Nv; ++jv)
        {
            double vy = vlo + jv * dv;
            for (int iv = 0; iv < Nv; ++iv)
            {
                double vx = vlo + iv * dv;
                double cn = (vx - Uwx) * nx + (vy - Uwy) * ny;
                int lin = iv + Nv * jv;

                if (cn > 0.0)
                {
                    double Mw = rho_w
                              * std::exp(-((vx - Uwx) * (vx - Uwx)
                                         + (vy - Uwy) * (vy - Uwy)) / RT2w)
                              / (M_PI * RT2w);
                    p.g[lin]       = Mw;
                    p.g[lin + Nv2] = 0.5 * R * Tw * Mw;
                }
            }
        }

        p.rho = rho_w;
        p.ux  = Uwx;
        p.uy  = Uwy;
        p.T = Tw;
        p.p = rho_w * R * Tw;
    }
}

// =====================================================================
// (Periodic closure is handled inside the neighbour search and the MLS
//  transport via the minimum-image convention — periodic domains have no
//  boundary particles and need no per-step BC pass.)
// =====================================================================

// =====================================================================
// 7. moodCheck — Check PAD admissibility for non-boundary particles
// =====================================================================
