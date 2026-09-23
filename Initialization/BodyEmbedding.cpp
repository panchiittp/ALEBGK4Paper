// ===========================================================================
// BodyEmbedding.cpp - Embeds a rigid body in the cloud: removes overrun lattice points and
// appends the body's surface particles carrying its wall state.
// ===========================================================================
#include "Config.hpp"
#include "Types.hpp"
#include "MLSBasis.hpp"
#include "Geometry.hpp"
#include "Initialization.hpp"
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>
#include <algorithm>

void embedRigidBodies(std::vector<Particle> &P,
                      const std::vector<std::unique_ptr<RigidBody>> &bodies,
                      const SimParameters &sp, CalcParameters &cp)
{
    if (bodies.empty()) return;
    int ndim = cp.dim;

    // Surface particle positions per body (also used for the hug test).
    // 2-D bodies fill surf2; 3-D bodies fill surf3.
    std::vector<std::vector<std::pair<double, double>>> surf2(bodies.size());
    std::vector<std::vector<std::array<double, 3>>>     surf3(bodies.size());
    for (std::size_t b = 0; b < bodies.size(); ++b)
    {
        if (ndim == 3) bodies[b]->generateBoundaryParticles3(surf3[b]);
        else           bodies[b]->generateBoundaryParticles(surf2[b]);
    }

    double hug2 = std::pow(0.4 * std::fabs(cp.dx), 2);
    std::vector<Particle> keep;
    keep.reserve(P.size());
    for (auto &p : P)
    {
        bool drop = false;
        for (std::size_t b = 0; b < bodies.size() && !drop; ++b)
        {
            if (ndim == 3)
            {
                if (bodies[b]->isInObject3(p.x, p.y, p.z)) { drop = true; break; }
                for (auto &q : surf3[b])
                {
                    double dx = p.x - q[0], dy = p.y - q[1], dz = p.z - q[2];
                    if (dx * dx + dy * dy + dz * dz < hug2) { drop = true; break; }
                }
            }
            else
            {
                if (bodies[b]->isInObject(p.x, p.y)) { drop = true; break; }
                for (auto &q : surf2[b])
                {
                    double dx = p.x - q.first, dy = p.y - q.second;
                    if (dx * dx + dy * dy < hug2) { drop = true; break; }
                }
            }
        }
        if (!drop) keep.push_back(std::move(p));
    }
    int removed = static_cast<int>(P.size() - keep.size());
    P.swap(keep);

    std::size_t nsurf = 0;
    for (std::size_t b = 0; b < bodies.size(); ++b)
    {
        std::size_t nq = (ndim == 3) ? surf3[b].size() : surf2[b].size();
        nsurf += nq;
        for (std::size_t k = 0; k < nq; ++k)
        {
            Particle p(sp.Nv, ndim);
            if (ndim == 3)
            {
                p.x = surf3[b][k][0]; p.y = surf3[b][k][1]; p.z = surf3[b][k][2];
                bodies[b]->getNormal3(p.x, p.y, p.z, &p.wnx, &p.wny, &p.wnz);
            }
            else
            {
                p.x = surf2[b][k].first; p.y = surf2[b][k].second; p.z = 0.0;
                bodies[b]->getNormal(p.x, p.y, &p.wnx, &p.wny);
                p.wnz = 0.0;
            }
            p.dim = ndim;
            p.boundary = true;
            p.bodyId = static_cast<int>(b);
            p.Nv_local = sp.Nv;
            p.vlo = sp.VMin; p.vhi = sp.VMax;
            p.dv_local = (sp.VMax - sp.VMin) / (sp.Nv - 1);
            p.wUx = bodies[b]->velx; p.wUy = bodies[b]->vely;
            p.wUz = bodies[b]->velz;
            p.wT  = bodies[b]->T;
            p.T = bodies[b]->T; p.rho = 1.0;
            p.validg = false;
            P.push_back(std::move(p));
        }
    }
    cp.N = static_cast<int>(P.size());
    std::cout << "Rigid bodies: removed " << removed
              << " lattice particles, added "
              << nsurf << " surface particles\n";
}

// ── main ─────────────────────────────────────────────────────────────────────
// ── Host memory availability ─────────────────────────────────────────────────
// Available (not merely free) physical RAM in GB: on Linux this is
// /proc/meminfo MemAvailable, which counts reclaimable page cache -- the
// honest headroom for a large allocation; on Windows, ullAvailPhys.
