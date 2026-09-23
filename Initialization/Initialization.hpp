// ===========================================================================
// Initialization.hpp - Cloud construction: lattice generation, initial state, neighbour
// lists and rigid-body embedding.
// ===========================================================================
#pragma once

#include "Types.hpp"
#include "Geometry.hpp"
#include <memory>
#include <vector>

// Places the point cloud on a lattice and sizes each particle's velocity grid.
void generateParticles(std::vector<Particle> &P,
                       const SimParameters   &sp,
                       const DomainBoundary  &dom,
                       CalcParameters        &cp);

// Sets every particle's distribution to the case's initial equilibrium:
// gas at rest at density rho0 and the wall temperature.
void applyIC(std::vector<Particle> &P, double rho0, double R,
             double Lx, double Ly, double Lz);

// Bins particles into voxels and rebuilds the MLS neighbour stencils.
// A pure function of the positions, so it is recomputed rather than stored.
void updateVoxelsAndNeighbours(std::vector<Particle> &P,
                               const CalcParameters &cp,
                               const DomainBoundary &dom);

// Removes lattice points the body overruns and appends its surface
// particles, which carry the body's wall state.
void embedRigidBodies(std::vector<Particle> &P,
                      const std::vector<std::unique_ptr<RigidBody>> &bodies,
                      const SimParameters &sp, CalcParameters &cp);
