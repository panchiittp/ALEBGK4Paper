// ===========================================================================
// Conservation.cpp - Conservation bookkeeping: admissibility check, total mass, global rescale.
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

bool moodCheck(std::vector<Particle> &P)
{
    bool anyFail = false;
    int N = static_cast<int>(P.size());
    int cLo, cHi;
    alebgk::chunk(N, &cLo, &cHi);

    for (int idx = cLo; idx < cHi; ++idx)
    {
        Particle &p = P[idx];
        if (p.boundary) continue;

        bool ok = padCheck(p);
        p.moodFlag = !ok;
        if (!ok) anyFail = true;
    }
#ifdef ALEBGK_WITH_MPI
    int f = anyFail ? 1 : 0, g = 0;
    MPI_Allreduce(&f, &g, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    anyFail = (g != 0);
#endif
    return anyFail;
}

// =====================================================================
// 8. sumMass — Sum g1 * dv^2 over valid particles
//
// Kept serial on purpose: a parallel reduction changes the summation
// order and hence the global mass-correction factor by ~1 ulp, making
// results depend on thread count. Serial summation keeps the solver
// bitwise deterministic; the cost is negligible next to transport.
// =====================================================================
double sumMass(std::vector<Particle> &P)
{
    double total = 0.0;
    int N = static_cast<int>(P.size());
    int cLo, cHi;
    alebgk::chunk(N, &cLo, &cHi);

    for (int idx = cLo; idx < cHi; ++idx)
    {
        const Particle &p = P[idx];
        if (!p.validg) continue;

        int Nv = p.Nv_local;
        double m = 0.0;
        if (p.dim == 3)
        {
            double dv3 = p.dv_local * p.dv_local * p.dv_local;
            int ntot = Nv * Nv * Nv;
            for (int k = 0; k < ntot; ++k) m += p.g[k] * dv3;
        }
        else
        {
            double dv2 = p.dv_local * p.dv_local;
            int Nv2 = Nv * Nv;
            for (int k = 0; k < Nv2; ++k) m += p.g[k] * dv2;
        }
        total += m;
    }
#ifdef ALEBGK_WITH_MPI
    // Deterministic reduction: gather rank-ordered partial sums and add
    // them in rank order so results are independent of the rank count.
    int n = alebgk::mpiSize();
    std::vector<double> parts(n);
    MPI_Allgather(&total, 1, MPI_DOUBLE, parts.data(), 1, MPI_DOUBLE,
                  MPI_COMM_WORLD);
    total = 0.0;
    for (double v : parts) total += v;
#endif
    return total;
}

// =====================================================================
// 9. scaleG — Multiply all g values by scale for valid particles
// =====================================================================
void scaleG(std::vector<Particle> &P, double scale)
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
        for (int k = 0; k < ntot; ++k) p.g[k] *= scale;
    }
}

// =====================================================================
// 10. Rigid-body coupling (moving bodies)
// =====================================================================
#include "Geometry.hpp"

// computeBodyLoads — surface stress integral relative to the wall
// velocity currently stored on each body-surface particle. g is
// read-only, so the loads can be re-evaluated against trial wall states
// within one step. 2-D: Chu-reduced g1 over Nv², torque = z-component
// (Tz). 3-D: single f over Nv³, full 3x3 stress, torque vector = r x t.
