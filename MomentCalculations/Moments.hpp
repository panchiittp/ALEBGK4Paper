// ===========================================================================
// Moments.hpp - Macroscopic moments, the BGK collision operator, and the conservation
// bookkeeping that keeps total mass fixed.
// ===========================================================================
#pragma once

#include "Types.hpp"
#include <vector>

// rho, u and T from the discrete distribution.
void computeMoments(std::vector<Particle> &P, double R);

// Implicit BGK relaxation toward the local Maxwellian.
void bgkCollision(std::vector<Particle> &P,
                  double dt, double tao, double R,
                  bool tauVariable, double dHS);

// Uniform body-force source term (unused by the cavity cases).
void applyGravity(std::vector<Particle> &P, double gY, double dt);

// Numerical-admissibility screen. True if every particle is admissible.
bool moodCheck(std::vector<Particle> &P);

// Total gas mass on the cloud, and a uniform rescale of g.
//
// A uniform rescale is safe to apply at any point in the step: u and T are
// ratios of moments and so are invariant under g -> s*g. Only the density
// moves, which is what a mass restore is for.
double sumMass(std::vector<Particle> &P);
void   scaleG(std::vector<Particle> &P, double scale);
