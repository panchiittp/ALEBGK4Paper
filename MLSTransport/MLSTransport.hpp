// ===========================================================================
// MLSTransport.hpp - Moving-least-squares semi-Lagrangian transport.
// ===========================================================================
#pragma once

#include "Types.hpp"
#include "MLSBasis.hpp"
#include <vector>

// Compute pass: reconstructs each particle's distribution at its departure
// point and writes p.gt. Reads neighbours' p.g only, so the result is
// independent of loop order and safe to thread.
void mlsTransportCompute(std::vector<Particle> &P,
                         double dt, const CalcParameters &cp,
                         DomainBoundary dom,
                         BoundaryConditions BC,
                         int mlsOrder, bool onlyFlagged = false);

// Positivity/admissibility screen over the freshly transported state.
// Returns the number of particles repaired.
int checkTransportPAD(std::vector<Particle> &P);

// Commit pass: gt -> g, completing the Jacobi update.
void mlsTransportCommit(std::vector<Particle> &P);
