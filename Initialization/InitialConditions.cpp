// ===========================================================================
// InitialConditions.cpp - The equilibrium state each case starts from.
//
// All six cavity cases start from the same state: gas at rest, uniform
// density and uniform temperature equal to the wall temperature. The density
// is passed in rather than inferred from a problem id, because it differs
// between the cases (1.0 for the plain cavity, 11.0 for the 2-D body cases,
// 4.4 for the 3-D ones) and getting it from a shared default silently starts
// a body case in the wrong gas.
// ===========================================================================
#include "Config.hpp"
#include "Types.hpp"
#include "MLSBasis.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

void applyIC(std::vector<Particle> &P, double rho0, double R,
             double Lx, double Ly, double Lz)
{
    (void)Lx; (void)Ly; (void)Lz;   // uniform state: no spatial dependence
    const int N = static_cast<int>(P.size());

    for (int idx = 0; idx < N; ++idx)
    {
        Particle &p = P[idx];
        const double rho = rho0;
        const double ux = 0.0, uy = 0.0, uz = 0.0;
        const double T = 270.0;

        // Store macro fields
        p.rho = rho;
        p.ux  = ux;
        p.uy  = uy;
        p.uz  = uz;
        p.T   = T;
        p.p   = rho * R * T;
        p.meshVx = p.meshVy = p.meshVz = 0.0;

        // Fill g from the Maxwellian
        int Nv     = p.Nv_local;
        double dv  = p.dv_local;
        double vlo = p.vlo;
        int Nv2    = Nv * Nv;

        if (p.dim == 3)
        {
            for (int kv = 0; kv < Nv; ++kv)
            {
                double vz_v = vlo + kv * dv;
                for (int jv = 0; jv < Nv; ++jv)
                {
                    double vy_v = vlo + jv * dv;
                    for (int iv = 0; iv < Nv; ++iv)
                    {
                        double vx_v = vlo + iv * dv;
                        int lin = iv + Nv * (jv + Nv * kv);
                        p.g[lin] = maxwellian3D(vx_v, vy_v, vz_v,
                                                ux, uy, uz, T, rho, R);
                    }
                }
            }
        }
        else
        {
            for (int jv = 0; jv < Nv; ++jv)
            {
                double vy_v = vlo + jv * dv;
                for (int iv = 0; iv < Nv; ++iv)
                {
                    double vx_v = vlo + iv * dv;
                    int lin = iv + Nv * jv;
                    double Mg1 = maxwellianG1(vx_v, vy_v, ux, uy, T, rho, R);
                    p.g[lin]       = Mg1;
                    p.g[lin + Nv2] = 0.5 * R * T * Mg1;
                }
            }
        }
        p.validg = true;
    }
}

// =====================================================================
// 2. computeMoments — Recover macroscopic fields from g
// =====================================================================

