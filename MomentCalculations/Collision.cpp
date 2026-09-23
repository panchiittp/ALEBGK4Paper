// ===========================================================================
// Collision.cpp - BGK collision operator and body-force source term.
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

void applyGravity(std::vector<Particle> &P, double gY, double dt)
{
    if (gY == 0.0) return;
    int N = static_cast<int>(P.size());
    int cLo, cHi;
    alebgk::chunk(N, &cLo, &cHi);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 16)
#endif
    for (int idx = cLo; idx < cHi; ++idx)
    {
        Particle &p = P[idx];
        if (!p.validg || p.boundary) continue;

        int Nv     = p.Nv_local;
        double dv  = p.dv_local;
        int Nv2    = Nv * Nv;

        double s = -gY * dt / dv;
        int    s0 = static_cast<int>(std::floor(s));
        double fr = s - s0;

        auto shiftColumn = [&](int base, int iv)
        {
            double col[MAX_NV];
            for (int jv = 0; jv < Nv; ++jv)
                col[jv] = p.g[base + iv + Nv * jv];
            for (int jv = 0; jv < Nv; ++jv)
            {
                int j0 = std::max(0, std::min(Nv - 1, jv + s0));
                int j1 = std::max(0, std::min(Nv - 1, jv + s0 + 1));
                p.g[base + iv + Nv * jv] = (1.0 - fr) * col[j0] + fr * col[j1];
            }
        };

        if (p.dim == 3)
        {
            for (int kv = 0; kv < Nv; ++kv)
                for (int iv = 0; iv < Nv; ++iv)
                    shiftColumn(kv * Nv2, iv);
        }
        else
        {
            for (int layer = 0; layer < 2; ++layer)
                for (int iv = 0; iv < Nv; ++iv)
                    shiftColumn(layer * Nv2, iv);
        }
    }
}

// =====================================================================
// 4. bgkCollision — Implicit BGK relaxation toward Maxwellian
// =====================================================================
void bgkCollision(std::vector<Particle> &P, double dt, double tao, double R,
                  bool tauVariable, double dHS)
{
    double lam   = dt / tao;
    double coeff = 1.0 / (1.0 + lam);
    int N = static_cast<int>(P.size());
    int cLo, cHi;
    alebgk::chunk(N, &cLo, &cHi);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 16)
#endif
    for (int idx = cLo; idx < cHi; ++idx)
    {
        Particle &p = P[idx];
        if (p.boundary) continue;

        if (tauVariable)
        {
            // Local hard-sphere tau(rho, T), as in meshfree4bgk2d
            constexpr double kB = 1.380649e-23;
            double l    = kB / (std::sqrt(2.0) * M_PI * p.rho * R * dHS * dHS);
            double cbar = std::sqrt(8.0 * R * p.T / M_PI);
            double tp   = 4.0 * l / (M_PI * cbar);
            lam   = dt / tp;
            coeff = 1.0 / (1.0 + lam);
        }

        int Nv     = p.Nv_local;
        double dv  = p.dv_local;
        double vlo = p.vlo;
        int Nv2    = Nv * Nv;

        if (p.dim == 3)
        {
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
                        double M = maxwellian3D(vx, vy, vz, p.ux, p.uy, p.uz,
                                                p.T, p.rho, R);
                        p.g[lin] = coeff * (p.g[lin] + lam * M);
                    }
                }
            }
        }
        else
        {
            for (int jv = 0; jv < Nv; ++jv)
            {
                double vy = vlo + jv * dv;
                for (int iv = 0; iv < Nv; ++iv)
                {
                    double vx = vlo + iv * dv;
                    int lin = iv + Nv * jv;
                    double Mg1 = maxwellianG1(vx, vy, p.ux, p.uy, p.T, p.rho, R);
                    double Mg2 = maxwellianG2(vx, vy, p.ux, p.uy, p.T, p.rho, R);
                    p.g[lin]       = coeff * (p.g[lin]       + lam * Mg1);
                    p.g[lin + Nv2] = coeff * (p.g[lin + Nv2] + lam * Mg2);
                }
            }
        }
    }
}

// =====================================================================
// 5. applyBoundaryBC — Diffuse-reflection BC for wall particles
// =====================================================================
