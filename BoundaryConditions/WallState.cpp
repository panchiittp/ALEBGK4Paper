// ===========================================================================
// WallState.cpp - Wall state initialisation for cavity boundary particles.
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

void initWallData(std::vector<Particle> &P, DomainBoundary dom,
                  BoundaryConditions BC)
{
    int N = static_cast<int>(P.size());
    for (int idx = 0; idx < N; ++idx)
    {
        Particle &p = P[idx];
        if (!p.boundary || p.bodyId >= 0) continue;
        getWallNormal(p, dom, BC, &p.wnx, &p.wny, &p.wnz,
                      &p.wUx, &p.wUy, &p.wUz, &p.wT);
    }
}

// =====================================================================
// wallCN helper — c_n = (v - Uw) . n from the stored wall state
// =====================================================================
