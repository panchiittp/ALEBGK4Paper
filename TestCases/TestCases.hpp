// =============================================================================
// TestCases.hpp - The six lid-driven cavity cases this solver reproduces.
//
// Each case is a named, self-contained configuration: domain, gas state,
// grids, time step and (where present) the immersed body. Nothing is selected
// by environment variable, so a case is reproduced by naming it and no other
// state is needed to know what was run.
// =============================================================================
#pragma once

#include "Types.hpp"
#include "Geometry.hpp"
#include <memory>
#include <string>
#include <vector>

enum class TestCase : int
{
    Cavity2D        = 1,   // 2-D lid-driven cavity
    Cavity3D        = 2,   // 3-D lid-driven cavity
    Cavity2D_Square = 3,   // 2-D cavity, square body
    Cavity2D_Circle = 4,   // 2-D cavity, circular body
    Cavity3D_Sphere = 5,   // 3-D cavity, spherical body
    Cavity3D_Cube   = 6    // 3-D cavity, cubic body
};

constexpr int kNumTestCases = 6;

// How the body is coupled to the flow. A case with no body ignores this.
// How the lid drives the cavity. The two plain cavity cases use a uniform
// lid; the body cases use the reference's parabolic profile, which vanishes
// at both lid corners and so avoids the corner singularity a uniform lid has.
enum class LidProfile : int
{
    UniformTop   = 0,   // uniform Ux on the y = L wall        (2-D cavity)
    UniformFront = 1,   // uniform Ux on the z = L wall        (3-D cavity)
    ParabolicTop = 2    // parabolic Ux on the y = L wall      (body cases)
};

enum class BodyMotion : int
{
    Fixed = 0,   // body held at its initial pose; flow solved around it
    Free  = 1    // body translates and rotates under the hydrodynamic load
};

struct CaseInfo
{
    TestCase     id;
    const char  *name;         // short name used on the command line
    const char  *description;  // one line for the run banner
    int          dim;          // 2 or 3
    bool         hasBody;
    LidProfile   lid;
    double       lidSpeed;     // uniform value, or the parabolic peak
};

// Static description of every case, indexed 0 .. kNumTestCases-1.
const CaseInfo &caseInfo(TestCase tc);
const CaseInfo *allCases();

// Resolves a command-line token - either the case number or its name - to a
// case id. Returns false if the token matches nothing.
bool parseCaseName(const std::string &token, TestCase *out);

// Fills the solver configuration for a case. This is the single place a
// case's physics is defined.
void configureCase(TestCase tc, BodyMotion motion,
                   SimParameters &sp, DomainBoundary &dom,
                   BoundaryConditions &BC, GasConstants &gc);

// Writes the lid velocity onto the cavity's lid particles. Called after
// initWallData, which sets the generic wall state; body surface particles
// carry their own and are left alone.
void applyLidProfile(TestCase tc, std::vector<Particle> &P,
                     const DomainBoundary &dom);

// Constructs the case's immersed body, or an empty vector for the two plain
// cavity cases. bodyScale rescales the body's linear size (1.0 = reference).
std::vector<std::unique_ptr<RigidBody>>
makeBodies(TestCase tc, BodyMotion motion, double gasDensity,
           double bodyScale);
