// ===========================================================================
// BoundaryConditions.hpp - Cavity wall boundary conditions.
// ===========================================================================
#pragma once

#include "Types.hpp"
#include <vector>

// Tags wall particles with their outward normal and wall state. A particle
// on an edge or corner touches two or three faces and accumulates all of
// them, rather than taking whichever the search tested first.
void initWallData(std::vector<Particle> &P, DomainBoundary dom,
                  BoundaryConditions BC);

// Diffuse reflection. The outgoing Maxwellian is scaled by
// rho_w = -intM/intP so the discrete net mass flux at each wall particle is
// exactly zero.
void applyBoundaryBC(std::vector<Particle> &P,
                     double R, DomainBoundary dom,
                     BoundaryConditions BC);
