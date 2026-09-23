// ===========================================================================
// CloudManagement.cpp - Point-cloud maintenance around a moving body: remove overrun particles,
// merge too-close pairs, refill vacated voxels.
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

int manageRemove(std::vector<Particle> &P,
                 std::vector<std::unique_ptr<RigidBody>> &bodies,
                 const CalcParameters &cp)
{
    if (bodies.empty()) return 0;
    double hug2 = std::pow(0.4 * std::fabs(cp.dx), 2);
    double min2 = std::pow(cp.minDist, 2);
    bool d3 = (cp.dim == 3);

    int removed = 0;

    // The hug test below asks "is this fluid particle within 0.4*dx of ANY
    // body-surface particle". Scanning all of P for each candidate made it
    // O(N^2): 29,666 x 29,762 = 883 million iterations per step to find the
    // 96 surface particles, a 0.3% hit rate. Measured, that was 2864 ms --
    // 83% of the host block and 66% of the whole step, dwarfing every GPU
    // kernel. Hoisting the surface coordinates once cuts the inner loop to
    // 29,666 x 96. Only x/y/z are needed, and they are copied out before the
    // move-out loop below touches P, so this is a pure reordering: the same
    // particles are tested against the same coordinates in the same order.
    struct SurfPt { double x, y, z; };
    std::vector<SurfPt> surf;
    surf.reserve(256);
    for (const auto &q : P)
        if (q.boundary && q.bodyId >= 0)
            surf.push_back({q.x, q.y, q.z});

    std::vector<Particle> keep;
    keep.reserve(P.size());
    for (auto &p : P)
    {
        bool drop = false;
        if (!p.boundary)
        {
            for (auto &b : bodies)
            {
                bool in = d3 ? b->isInObject3(p.x, p.y, p.z)
                             : b->isInObject(p.x, p.y);
                if (in) { drop = true; break; }
            }
            if (!drop)
                for (const auto &q : surf)
                {
                    double dx = p.x - q.x, dy = p.y - q.y;
                    double dz = d3 ? p.z - q.z : 0.0;
                    if (dx * dx + dy * dy + dz * dz < hug2)
                    { drop = true; break; }
                }
        }
        if (drop) { removed++; continue; }
        keep.push_back(std::move(p));
    }
    P.swap(keep);
    (void)min2;
    return removed;
}

// mergeCloseParticles — meshfree4bgk2d's removeParticle_ Case 3: two
// interior particles closer than minDist merge into one at the mean
// position with the averaged distribution (g is intensive/collocational,
// so averaging is the consistent merge; the per-step global mass
// correction absorbs the point-count change). Requires up-to-date
// neighbour lists; caller rebuilds afterwards if the count changed.
int mergeCloseParticles(std::vector<Particle> &P, const CalcParameters &cp)
{
    double min2 = cp.minDist * cp.minDist;
    if (min2 <= 0.0) return 0;
    int N = static_cast<int>(P.size());
    std::vector<char> dead(N, 0);
    int merged = 0;
    for (int i = 0; i < N; ++i)
    {
        if (dead[i] || P[i].boundary || !P[i].validg) continue;
        for (int j : P[i].neighindex)
        {
            if (j <= i || j >= N || dead[j]) continue;
            Particle &q = P[j];
            if (q.boundary || !q.validg) continue;
            double dx = P[i].x - q.x, dy = P[i].y - q.y;
            double dz = (cp.dim == 3) ? P[i].z - q.z : 0.0;
            if (dx*dx + dy*dy + dz*dz < min2)
            {
                P[i].x = 0.5 * (P[i].x + q.x);
                P[i].y = 0.5 * (P[i].y + q.y);
                if (cp.dim == 3) P[i].z = 0.5 * (P[i].z + q.z);
                for (std::size_t k2 = 0; k2 < P[i].g.size(); ++k2)
                    P[i].g[k2] = 0.5 * (P[i].g[k2] + q.g[k2]);
                P[i].gt = P[i].g;
                dead[j] = 1;
                merged++;
                break;
            }
        }
    }
    if (merged)
    {
        std::vector<Particle> keep;
        keep.reserve(N - merged);
        for (int i = 0; i < N; ++i)
            if (!dead[i]) keep.push_back(std::move(P[i]));
        P.swap(keep);
    }
    return merged;
}

int manageRefill(std::vector<Particle> &P,
                 std::vector<std::unique_ptr<RigidBody>> &bodies,
                 const SimParameters &sp, const CalcParameters &cp,
                 const DomainBoundary &dom, double R)
{
    if (bodies.empty()) return 0;

    bool d3 = (cp.dim == 3);
    double dxl = std::fabs(cp.dx);

    // The ALE motion only vacates space around the moving bodies, and it
    // does so at sub-voxel scale: a neighbour-search voxel is ~3 dx wide,
    // so the 1-2 dx wake behind a moving body never empties one and a
    // voxel-occupancy criterion never refills (verified on the 2D cavity:
    // the wake grows past 1.5 dx while the healthy cloud's largest
    // nearest-particle gap is ~0.7 dx, and the starved MLS stencils blow
    // up once the body has moved ~2.5 dx).  Refill therefore scans the
    // INITIAL dx-lattice in a margin box around the body surfaces and
    // fills any site whose nearest particle is farther than 0.8 dx.
    double hole2 = std::pow(0.8 * dxl, 2);

    double bx0 = 1e300, bx1 = -1e300, by0 = 1e300, by1 = -1e300;
    double bz0 = 1e300, bz1 = -1e300;
    bool anySurf = false;
    for (auto &q : P)
    {
        if (!q.boundary || q.bodyId < 0) continue;
        bx0 = std::min(bx0, q.x); bx1 = std::max(bx1, q.x);
        by0 = std::min(by0, q.y); by1 = std::max(by1, q.y);
        if (d3) { bz0 = std::min(bz0, q.z); bz1 = std::max(bz1, q.z); }
        anySurf = true;
    }
    if (!anySurf) return 0;
    double margin = 3.5 * dxl;
    bx0 -= margin; bx1 += margin; by0 -= margin; by1 += margin;
    if (d3) { bz0 -= margin; bz1 += margin; }

    int nxs = static_cast<int>(std::llround((dom.xright - dom.xleft) / dxl));
    int nys = static_cast<int>(std::llround((dom.ytop - dom.ybottom) / dxl));
    int nzs = d3 ? static_cast<int>(std::llround((dom.zback - dom.zfront) / dxl)) : 0;
    int i0 = std::max(1, static_cast<int>(std::ceil((bx0 - dom.xleft) / dxl)));
    int i1 = std::min(nxs - 1, static_cast<int>(std::floor((bx1 - dom.xleft) / dxl)));
    int j0 = std::max(1, static_cast<int>(std::ceil((by0 - dom.ybottom) / dxl)));
    int j1 = std::min(nys - 1, static_cast<int>(std::floor((by1 - dom.ybottom) / dxl)));
    int k0 = d3 ? std::max(1, static_cast<int>(std::ceil((bz0 - dom.zfront) / dxl))) : 0;
    int k1 = d3 ? std::min(nzs - 1, static_cast<int>(std::floor((bz1 - dom.zfront) / dxl))) : 0;

    int added = 0;
    for (int vk = k0; vk <= k1; ++vk)
    for (int vj = j0; vj <= j1; ++vj)
        for (int vi = i0; vi <= i1; ++vi)
        {
            double xm = dom.xleft   + vi * dxl;
            double ym = dom.ybottom + vj * dxl;
            double zm = d3 ? dom.zfront + vk * dxl : 0.0;

            // Hole test: no particle within 0.8 dx of the lattice site.
            bool hole = true;
            for (auto &q : P)
            {
                double dxq = q.x - xm, dyq = q.y - ym;
                double dzq = d3 ? q.z - zm : 0.0;
                if (dxq * dxq + dyq * dyq + dzq * dzq < hole2)
                { hole = false; break; }
            }
            if (!hole) continue;

            bool inBody = false;
            for (auto &b : bodies)
            {
                bool in = d3 ? b->isInObject3(xm, ym, zm)
                             : b->isInObject(xm, ym);
                if (in) { inBody = true; break; }
            }
            if (inBody) continue;

            // Keep clearance from body-surface particles — a refill on
            // top of one flaps against the hug-removal every step and
            // degenerates the MLS stencil.
            double clr2 = std::pow(0.6 * dxl, 2);
            bool hugs = false;
            for (auto &q : P)
            {
                if (!q.boundary || q.bodyId < 0) continue;
                double dx = q.x - xm, dy = q.y - ym;
                double dz = d3 ? q.z - zm : 0.0;
                if (dx * dx + dy * dy + dz * dz < clr2) { hugs = true; break; }
            }
            if (hugs) continue;

            // Neighbour-averaged moments within the search radius
            double wsum = 0, rho = 0, ux = 0, uy = 0, uz = 0, T = 0;
            double r2 = cp.radius * cp.radius;
            for (auto &q : P)
            {
                double dx = q.x - xm, dy = q.y - ym;
                double dz = d3 ? q.z - zm : 0.0;
                double d2 = dx * dx + dy * dy + dz * dz;
                if (d2 >= r2 || !q.validg) continue;
                double w = mlsWeightSq(d2, cp.radius);
                wsum += w; rho += w * q.rho; ux += w * q.ux;
                uy += w * q.uy; uz += w * q.uz; T += w * q.T;
            }
            if (wsum < 1e-30) continue;   // no gas nearby — skip
            rho /= wsum; ux /= wsum; uy /= wsum; uz /= wsum; T /= wsum;

            Particle p(sp.Nv, cp.dim);
            p.x = xm; p.y = ym; p.z = zm; p.dim = cp.dim;
            p.boundary = false;
            p.Nv_local = sp.Nv;
            p.vlo = sp.VMin; p.vhi = sp.VMax;
            p.dv_local = (sp.VMax - sp.VMin) / (sp.Nv - 1);
            p.rho = rho; p.ux = ux; p.uy = uy; p.T = T;
            if (d3) p.uz = uz;
            p.p = rho * R * T;
            int Nv = p.Nv_local, Nv2 = Nv * Nv;
            if (d3)
            {
                for (int kv = 0; kv < Nv; ++kv)
                {
                    double vz = p.vlo + kv * p.dv_local;
                    for (int jv = 0; jv < Nv; ++jv)
                    {
                        double vy = p.vlo + jv * p.dv_local;
                        for (int iv = 0; iv < Nv; ++iv)
                        {
                            double vx = p.vlo + iv * p.dv_local;
                            p.g[iv + Nv * (jv + Nv * kv)] =
                                maxwellian3D(vx, vy, vz, ux, uy, uz, T, rho, R);
                        }
                    }
                }
            }
            else
            {
                for (int jv = 0; jv < Nv; ++jv)
                {
                    double vy = p.vlo + jv * p.dv_local;
                    for (int iv = 0; iv < Nv; ++iv)
                    {
                        double vx = p.vlo + iv * p.dv_local;
                        double Mg1 = maxwellianG1(vx, vy, ux, uy, T, rho, R);
                        p.g[iv + Nv * jv]       = Mg1;
                        p.g[iv + Nv * jv + Nv2] = 0.5 * R * T * Mg1;
                    }
                }
            }
            p.gt = p.g;
            p.validg = true;
            P.push_back(std::move(p));
            added++;
        }
    return added;
}
