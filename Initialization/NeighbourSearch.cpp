// ===========================================================================
// NeighbourSearch.cpp - Voxel binning and neighbour lists for the MLS stencils.
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

void updateVoxelsAndNeighbours(std::vector<Particle> &P,
                               const CalcParameters  &cp,
                               const DomainBoundary  &dom)
{
    int N    = static_cast<int>(P.size());
    int ndim = cp.dim;

    // Bin particles into voxels
    std::vector<std::vector<int>> voxParts(cp.nvox);
    for (int i = 0; i < N; i++) {
        int vi = std::max(0, std::min(cp.nbhBox - 1,
                 static_cast<int>(std::floor((P[i].x - dom.xleft) / cp.hBox))));
        int vj = std::max(0, std::min(cp.nbvBox - 1,
                 static_cast<int>(std::floor((P[i].y - dom.ybottom) / cp.vBox))));
        int vk = (ndim == 3)
               ? std::max(0, std::min(cp.nbwBox - 1,
                 static_cast<int>(std::floor((P[i].z - dom.zfront) / cp.wBox))))
               : 0;
        P[i].voxel = vi + cp.nbhBox * (vj + cp.nbvBox * vk);
        if (P[i].voxel < cp.nvox)
            voxParts[P[i].voxel].push_back(i);
    }

    // Neighbour search using voxel stencil
    double r2max  = cp.radius * cp.radius;
    int    dk_max = (ndim == 3) ? 1 : 0;

    // Each iteration writes only P[i].neighindex and reads P[j] positions and
    // voxParts read-only, so this is embarrassingly parallel. It was the one
    // hot host loop in this file left serial while the rest of the solver's
    // CPU kernels are threaded; measured at 149 ms per call in the
    // moving-body path, where it runs every step.
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < N; i++) {
        P[i].neighindex.clear();

        int vi  = P[i].voxel % cp.nbhBox;
        int tmp = P[i].voxel / cp.nbhBox;
        int vj  = tmp % cp.nbvBox;
        int vk  = tmp / cp.nbvBox;

        bool done = false;
        for (int dk = -dk_max; dk <= dk_max && !done; dk++)
            for (int dj = -1; dj <= 1 && !done; dj++)
                for (int di = -1; di <= 1 && !done; di++) {
                    int ni = vi + di, nj = vj + dj, nk = vk + dk;
                    // Wrap the stencil across seams on periodic axes,
                    // skip out-of-range boxes on wall axes.
                    if (cp.periodicX) ni = (ni + cp.nbhBox) % cp.nbhBox;
                    else if (ni < 0 || ni >= cp.nbhBox) continue;
                    if (cp.periodicY) nj = (nj + cp.nbvBox) % cp.nbvBox;
                    else if (nj < 0 || nj >= cp.nbvBox) continue;
                    if (ndim == 3 && cp.periodicZ)
                        nk = (nk + cp.nbwBox) % cp.nbwBox;
                    else if (nk < 0 || nk >= cp.nbwBox) continue;
                    int vox = ni + cp.nbhBox * (nj + cp.nbvBox * nk);
                    bool anyPeriodic = cp.periodicX || cp.periodicY ||
                                       cp.periodicZ;
                    for (int j : voxParts[vox]) {
                        if (j == i) continue;
                        double ddx = P[j].x - P[i].x;
                        double ddy = P[j].y - P[i].y;
                        double ddz = (ndim == 3) ? P[j].z - P[i].z : 0.0;
                        if (cp.periodicX) ddx = minImage(ddx, cp.Lx);
                        if (cp.periodicY) ddy = minImage(ddy, cp.Ly);
                        if (ndim == 3 && cp.periodicZ)
                            ddz = minImage(ddz, cp.Lz);
                        if (ddx*ddx + ddy*ddy + ddz*ddz < r2max) {
                            // Wrapped stencils can revisit a box when an
                            // axis has < 3 boxes — don't add a neighbour twice
                            if (anyPeriodic &&
                                std::find(P[i].neighindex.begin(),
                                          P[i].neighindex.end(), j)
                                    != P[i].neighindex.end()) continue;
                            P[i].neighindex.push_back(j);
                            if (static_cast<int>(P[i].neighindex.size()) >= MAX_NEIGH) {
                                done = true; break;
                            }
                        }
                    }
                }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// embedRigidBodies — remove lattice particles inside (or hugging) a body,
// then append its surface particles carrying the body's wall state.
// Must run before the neighbour search.
// ─────────────────────────────────────────────────────────────────────────────
