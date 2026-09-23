// ===========================================================================
// RigidBody.hpp - Rigid-body dynamics and the point-cloud maintenance a moving body needs.
// ===========================================================================
#pragma once

#include "Types.hpp"
#include "Geometry.hpp"
#include <memory>
#include <vector>

// Integrates the body: surface-stress loads, then a Heun-coupled
// Newton-Euler update of position, velocity and orientation.
void moveRigidBodies(std::vector<Particle> &P,
                     std::vector<std::unique_ptr<RigidBody>> &bodies,
                     double dt, const DomainBoundary &dom,
                     const CalcParameters &cp);

// Cloud maintenance as the body sweeps the lattice. Each returns the number
// of particles affected.
int manageRemove(std::vector<Particle> &P,
                 std::vector<std::unique_ptr<RigidBody>> &bodies,
                 const CalcParameters &cp);
int mergeCloseParticles(std::vector<Particle> &P, const CalcParameters &cp);
int manageRefill(std::vector<Particle> &P,
                 std::vector<std::unique_ptr<RigidBody>> &bodies,
                 const SimParameters &sp, const CalcParameters &cp,
                 const DomainBoundary &dom, double R);
