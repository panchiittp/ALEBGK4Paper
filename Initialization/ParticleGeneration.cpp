// ===========================================================================
// ParticleGeneration.cpp - Lattice generation: places the point cloud and its velocity grids.
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

void generateParticles(std::vector<Particle> &P,
                       const SimParameters   &sp,
                       const DomainBoundary  &dom,
                       CalcParameters        &cp)
{
    int Nx = sp.Nx, Ny = sp.Ny, Nz = sp.Nz;
    int ndim = (sp.dim == Dimension::D3) ? 3 : 2;
    if (ndim == 2) Nz = 1;

    bool px, py, pz;
    problemPeriodicity(sp.problem, ndim, &px, &py, &pz);
    double Lx = dom.xright - dom.xleft;
    double Ly = dom.ytop   - dom.ybottom;
    double Lz = (ndim == 3) ? (dom.zback - dom.zfront) : 1.0;

    // Periodic axis: seam-free lattice — N points with spacing L/N, so the
    // node at L is the image of the node at 0 and is not duplicated.
    double dx = px ? Lx / Nx : Lx / (Nx - 1);
    double dy = py ? Ly / Ny : Ly / (Ny - 1);
    double dz = (ndim == 3 && Nz > 1)
              ? (pz ? Lz / Nz : Lz / (Nz - 1)) : 1.0;

    cp.dx = dx; cp.dy = dy; cp.dz = dz; cp.dim = ndim;
    cp.periodicX = px; cp.periodicY = py; cp.periodicZ = pz;
    cp.Lx = Lx; cp.Ly = Ly; cp.Lz = Lz;
    cp.radius  = 2.5 * std::max({std::fabs(dx), std::fabs(dy),
                                 (ndim == 3 ? std::fabs(dz) : std::fabs(dx))});
    cp.minDist = 0.5 * std::min({std::fabs(dx), std::fabs(dy),
                                 (ndim == 3 ? std::fabs(dz) : std::fabs(dy))});

#ifdef ALEBGK_WITH_MPI
    // Slab decomposition plan (along the outermost generation axis).
    // Static-cloud problems on a Fixed grid only; body problems (11/12)
    // and the Adaptive grid keep the replicated path.
    {
        bool axisPeriodic = (ndim == 3) ? pz : py;
        double axisSp     = (ndim == 3) ? dz : dy;
        long long stride  = (ndim == 3)
                          ? static_cast<long long>(Nx) * Ny
                          : static_cast<long long>(Nx);
        int rows          = (ndim == 3) ? Nz : Ny;
        bool elig = (sp.velMode == VelGridMode::Fixed)
                 && (static_cast<int>(sp.problem) < 11);
        alebgk_decompPlan(rows, stride, axisPeriodic, axisSp,
                          cp.radius, ndim, elig);
    }
#endif

    int Nv = sp.Nv;
    double dv = (sp.VMax - sp.VMin) / (Nv - 1);
    cp.dv = dv; cp.Nv_actual = Nv;

    // Per-axis voxel boxes. Periodic axes must tile the domain exactly so
    // the stencil can wrap modulo the box count; box size ≥ 3*spacing ≥
    // search radius on that axis.
    double hb, vb, wb;
    if (px) {
        cp.nbhBox = std::max(1, static_cast<int>(std::floor(Lx / (3.0 * dx))));
        hb = Lx / cp.nbhBox;
    } else {
        hb = 3.0 * dx;
        cp.nbhBox = static_cast<int>(std::ceil(Lx / hb)) + 1;
    }
    if (py) {
        cp.nbvBox = std::max(1, static_cast<int>(std::floor(Ly / (3.0 * dy))));
        vb = Ly / cp.nbvBox;
    } else {
        vb = 3.0 * dy;
        cp.nbvBox = static_cast<int>(std::ceil(Ly / vb)) + 1;
    }
    if (ndim == 3) {
        if (pz) {
            cp.nbwBox = std::max(1, static_cast<int>(
                            std::floor(std::fabs(Lz) / (3.0 * std::fabs(dz)))));
            wb = Lz / cp.nbwBox;
        } else {
            wb = 3.0 * dz;
            cp.nbwBox = static_cast<int>(std::ceil(Lz / wb)) + 1;
        }
    } else {
        wb = 1.0;
        cp.nbwBox = 1;
    }
    cp.hBox = hb; cp.vBox = vb; cp.wBox = wb;
    cp.nvox = cp.nbhBox * cp.nbvBox * cp.nbwBox;

    P.clear();
    P.reserve(Nx * Ny * Nz);

    for (int k = 0; k < Nz; k++) {
        double z = (ndim == 3) ? dom.zfront + k * dz : 0.0;
        for (int j = 0; j < Ny; j++) {
            double y = dom.ybottom + j * dy;
            for (int i = 0; i < Nx; i++) {
                double x = dom.xleft + i * dx;

                Particle p(Nv, ndim);
                p.x = x; p.y = y; p.z = z;
                p.ux = 0; p.uy = 0; p.uz = 0;
                p.meshVx = 0; p.meshVy = 0; p.meshVz = 0;
                p.T = 270; p.rho = 1; p.p = 0;
                p.dim = ndim;

                // Wall particles exist only on non-periodic axes.
                bool bx = !px && (i == 0 || i == Nx - 1);
                bool by = !py && (j == 0 || j == Ny - 1);
                bool bz = (ndim == 3) && !pz && (k == 0 || k == Nz - 1);
                p.boundary = (bx || by || bz);
                p.validg = false;

                // Adaptive velocity grid is 2-D only (Chu reduction). 3-D
                // uses a single f on a Fixed uniform Nv³ grid — the
                // constructor already sized g/gt to Nv³.
                if (sp.velMode == VelGridMode::Adaptive && ndim == 2) {
                    double vlo, vhi, dv_l;
                    int Nv_l;
                    computeAdaptiveGrid(0, 0, 0, 270.0, 208.0, sp.velCutoff,
                                        sp.NvMin, sp.NvMax,
                                        &vlo, &vhi, &Nv_l, &dv_l);
                    p.Nv_local = Nv_l;
                    p.vlo = vlo; p.vhi = vhi; p.dv_local = dv_l;
                    p.g.assign(Nv_l * Nv_l * 2, 0.0);
                    p.gt.assign(Nv_l * Nv_l * 2, 0.0);
                } else {
                    p.Nv_local = Nv;
                    p.vlo = sp.VMin; p.vhi = sp.VMax; p.dv_local = dv;
                }

                int vi = std::max(0, std::min(cp.nbhBox - 1,
                         static_cast<int>(std::floor((x - dom.xleft) / hb))));
                int vj = std::max(0, std::min(cp.nbvBox - 1,
                         static_cast<int>(std::floor((y - dom.ybottom) / vb))));
                int vk = (ndim == 3)
                       ? std::max(0, std::min(cp.nbwBox - 1,
                         static_cast<int>(std::floor((z - dom.zfront) / wb))))
                       : 0;
                p.voxel = vi + cp.nbhBox * (vj + cp.nbvBox * vk);

#ifdef ALEBGK_WITH_MPI
                // Decomposed mode: rows outside this rank's slab (owned +
                // halo) never allocate distributions — avoids the transient
                // full-cloud footprint during generation.
                {
                    long long gidx = (static_cast<long long>(k) * Ny + j)
                                   * Nx + i;
                    if (!alebgk_decompKeep(gidx))
                    {
                        p.g.clear();  p.g.shrink_to_fit();
                        p.gt.clear(); p.gt.shrink_to_fit();
                    }
                }
#endif
                P.push_back(std::move(p));
            }
        }
    }

#ifdef ALEBGK_WITH_MPI
    // Decomposed mode: keep only this rank's slab (owned + halo rows) and
    // activate the owned-range chunk(). Must run before the neighbour
    // build so neighbour lists carry local indices.
    alebgk_decompTrim(P);
#endif
    cp.N = static_cast<int>(P.size());
    if (ALEBGK_ROOT) {
        std::cout << "Generated " << cp.N << " particles ("
                  << Nx << "x" << Ny;
        if (ndim == 3) std::cout << "x" << Nz;
        std::cout << ") dim=" << ndim << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// updateVoxelsAndNeighbours — Build voxel grid, find neighbours within radius
// ─────────────────────────────────────────────────────────────────────────────
