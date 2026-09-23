// ===========================================================================
// Moments.cpp - Macroscopic moments: density, velocity and temperature from the
// discrete distribution function.
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

void computeMoments(std::vector<Particle> &P, double R)
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
        // Boundary particles carry the imposed wall state (velocity = wall
        // velocity from applyBoundaryBC): constant lid on the driven wall,
        // zero on the rest. Recomputing from the mixed half-Maxwellian wall
        // distribution would report the slip moment, wrong for a Dirichlet
        // wall — leave boundary macro fields untouched.
        if (!p.validg || p.boundary) continue;

        int Nv     = p.Nv_local;
        double dv  = p.dv_local;
        double vlo = p.vlo;
        int Nv2    = Nv * Nv;

        double rho_s = 0.0, rhoUx = 0.0, rhoUy = 0.0, rhoUz = 0.0, E = 0.0;

        if (p.dim == 3)
        {
            double dv3 = dv * dv * dv;
            for (int kv = 0; kv < Nv; ++kv)
            {
                double vz = vlo + kv * dv;
                for (int jv = 0; jv < Nv; ++jv)
                {
                    double vy = vlo + jv * dv;
                    for (int iv = 0; iv < Nv; ++iv)
                    {
                        double vx = vlo + iv * dv;
                        double f = p.g[iv + Nv * (jv + Nv * kv)];
                        rho_s += f * dv3;
                        rhoUx += vx * f * dv3;
                        rhoUy += vy * f * dv3;
                        rhoUz += vz * f * dv3;
                        E     += (vx*vx + vy*vy + vz*vz) * f * dv3;
                    }
                }
            }
        }
        else
        {
            double dv2 = dv * dv;
            for (int jv = 0; jv < Nv; ++jv)
            {
                double vy = vlo + jv * dv;
                for (int iv = 0; iv < Nv; ++iv)
                {
                    double vx = vlo + iv * dv;
                    int lin = iv + Nv * jv;
                    double g1 = p.g[lin];
                    double g2 = p.g[lin + Nv2];
                    rho_s += g1 * dv2;
                    rhoUx += vx * g1 * dv2;
                    rhoUy += vy * g1 * dv2;
                    E     += (vx * vx + vy * vy) * g1 * dv2 + 2.0 * g2 * dv2;
                }
            }
        }

        if (rho_s > 1.0e-30)
        {
            p.rho = rho_s;
            p.ux  = rhoUx / rho_s;
            p.uy  = rhoUy / rho_s;
            if (p.dim == 3) p.uz = rhoUz / rho_s;

            double u2 = p.ux * p.ux + p.uy * p.uy + p.uz * p.uz;
            p.T = std::max((E / rho_s - u2) / (3.0 * R), 1.0e-6);
            p.p = rho_s * R * p.T;
        }
    }
}

// =====================================================================
// Local Gaussian elimination with partial pivoting (NxN)
// =====================================================================
