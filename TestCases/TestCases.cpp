// =============================================================================
// TestCases.cpp - Configuration of the six cavity cases.
//
// Every number here is stated with its reason. Where a value was established
// by a numerical experiment rather than taken from a reference, the finding is
// recorded next to it, because several of them look arbitrary otherwise and
// have been "tidied" into failures before.
//
// Common to all six:
//   domain    1 um cube (1 um square in 2-D)
//   gas       argon, hard-sphere diameter 0.368 nm, walls at 270 K
//   lid       tangential, on one face: uniform 1 m/s for the plain cavity,
//             parabolic peaking at 10 m/s for the body cases
//   walls     diffuse reflection, fully accommodating
// =============================================================================
#include "TestCases.hpp"

#include <cstring>
#include <cstdlib>
#include <cmath>

namespace {

// Lid speeds, m/s. The plain cavity is driven at 1 m/s, a low Reynolds
// number that keeps the primary vortex near the lid. The body cases use the
// reference's parabolic profile peaking at 10 m/s.
constexpr double kUniformLid   = 1.0;
constexpr double kParabolicPeak = 10.0;

// Wall temperature, K, and the hard-sphere diameter of argon, m.
constexpr double kWallTemp = 270.0;
constexpr double kHSDiam   = 0.368e-9;

// Body density as a multiple of the gas density. The reference uses a body
// ten times the gas; the rotational spin-up time scales with that ratio, so
// it must follow the gas density rather than be hard-wired.
constexpr double kBodyDensityRatio = 10.0;

// Reference body sizes, m. The 3-D pair is chosen so the sphere's diameter
// equals the cube's side; the 2-D pair follows the same convention, so the
// circle's diameter equals the square's side. Comparing shapes then varies
// only the shape, not the blockage.
constexpr double kSquareSide   = 7.5e-8;
constexpr double kCircleRadius = kSquareSide / 2.0;
constexpr double kCubeSide     = 1.5e-7;
constexpr double kSphereRadius = kCubeSide / 2.0;

// Body surface sampling: points distributed over the body's surface, which
// both carry its wall state and provide the quadrature for the surface-stress
// integral that produces the force and torque.
constexpr int kSquarePts = 16;
constexpr int kCirclePts = 32;
constexpr int kSpherePts = 48;   // Fibonacci sampling
constexpr int kCubePts   = 96;   // 4x4 per face

const CaseInfo kCases[kNumTestCases] = {
  { TestCase::Cavity2D,        "cavity2d",
    "2-D lid-driven cavity",                       2, false,
    LidProfile::UniformTop,   kUniformLid },
  { TestCase::Cavity3D,        "cavity3d",
    "3-D lid-driven cavity",                       3, false,
    LidProfile::UniformFront, kUniformLid },
  { TestCase::Cavity2D_Square, "cavity2d-square",
    "2-D lid-driven cavity with a square body",    2, true,
    LidProfile::ParabolicTop, kParabolicPeak },
  { TestCase::Cavity2D_Circle, "cavity2d-circle",
    "2-D lid-driven cavity with a circular body",  2, true,
    LidProfile::ParabolicTop, kParabolicPeak },
  { TestCase::Cavity3D_Sphere, "cavity3d-sphere",
    "3-D lid-driven cavity with a spherical body", 3, true,
    LidProfile::ParabolicTop, kParabolicPeak },
  { TestCase::Cavity3D_Cube,   "cavity3d-cube",
    "3-D lid-driven cavity with a cubic body",     3, true,
    LidProfile::ParabolicTop, kParabolicPeak },
};

void zeroWalls(BoundaryConditions &BC)
{
    BC.Left = BC.Right = BC.Top = BC.Bottom = BC.Front = BC.Back = {0, 0, 0};
}

}  // namespace

const CaseInfo &caseInfo(TestCase tc)
{
    return kCases[static_cast<int>(tc) - 1];
}

const CaseInfo *allCases() { return kCases; }

bool parseCaseName(const std::string &token, TestCase *out)
{
    if (!token.empty() && token.find_first_not_of("0123456789") ==
        std::string::npos)
    {
        int n = std::atoi(token.c_str());
        if (n >= 1 && n <= kNumTestCases)
        {
            *out = static_cast<TestCase>(n);
            return true;
        }
        return false;
    }
    for (int i = 0; i < kNumTestCases; ++i)
        if (token == kCases[i].name)
        {
            *out = kCases[i].id;
            return true;
        }
    return false;
}

void configureCase(TestCase tc, BodyMotion motion,
                   SimParameters &sp, DomainBoundary &dom,
                   BoundaryConditions &BC, GasConstants &gc)
{
    (void)gc;
    const CaseInfo &info = caseInfo(tc);

    BC.Tw = kWallTemp;
    zeroWalls(BC);
    // Must be set explicitly: SimParameters defaults to D3, and a 2-D case
    // left at that default is asked for a 3-D velocity grid AND 3-D body
    // surface points, which the 2-D shapes do not provide - the body then
    // silently embeds with no surface at all.
    sp.dim      = (info.dim == 3) ? Dimension::D3 : Dimension::D2;
    sp.d        = kHSDiam;
    sp.velMode  = VelGridMode::Fixed;
    sp.r        = 5.0e-8;    // MLS support radius
    sp.rb       = 8.0e-8;    // neighbour-search radius
    sp.movingBodies = info.hasBody && (motion == BodyMotion::Free);

    // Domain: 1 um square in 2-D, 1 um cube in 3-D.
    //
    // The lid differs between the two families. The plain cavity drives the
    // y = L wall in 2-D and the z = L wall in 3-D - in 3-D the velocity grid
    // resolves vz, so a z-normal lid does transfer momentum. The body cases
    // drive the y = L wall in BOTH dimensions, and set it through the
    // parabolic profile rather than the uniform wall state, so their
    // BoundaryConditions lid entry stays zero here.
    dom = (info.dim == 2) ? DomainBoundary{0, 1e-6, 1e-6, 0, 0, 0}
                          : DomainBoundary{0, 1e-6, 1e-6, 0, 1e-6, 0};
    if (info.lid == LidProfile::UniformTop)   BC.Top   = {info.lidSpeed, 0, 0};
    if (info.lid == LidProfile::UniformFront) BC.Front = {info.lidSpeed, 0, 0};

    switch (tc)
    {
    // -- Plain cavity, 2-D ---------------------------------------------
    // Kn ~ 0.11. The velocity grid spans +-5.5 sigma with sigma = sqrt(R*T)
    // = 237 m/s; Nv=44 gives dv/sigma = 0.11, comfortably resolved.
    case TestCase::Cavity2D:
        BC.rho   = 1.0;
        sp.tao   = 3.7142e-10;      // hard-sphere tau at rho=1, T=270
        sp.Nx    = 100; sp.Ny = 100; sp.Nz = 1;
        sp.Nv    = 44;
        sp.VMax  = 1303; sp.VMin = -1303;
        sp.dt    = 1e-13;
        sp.tfinal = 5e-9;
        sp.saveEvery = 500;
        break;

    // -- Plain cavity, 3-D ---------------------------------------------
    // Nv is capped at 20 because the memory cost is Nv^3 per particle.
    case TestCase::Cavity3D:
        BC.rho   = 1.0;
        sp.tao   = 3.7142e-10;
        sp.Nx    = 31; sp.Ny = 31; sp.Nz = 31;
        sp.Nv    = 20;
        sp.VMax  = 1303; sp.VMin = -1303;
        sp.dt    = 1e-11;
        sp.tfinal = 5e-9;
        sp.saveEvery = 500;
        break;

    // -- Cavity with a body, 2-D ---------------------------------------
    // Denser gas (rho=11, Kn ~ 0.01) so the body sits in a near-continuum
    // flow, as in the reference configuration.
    //
    // dt = 1.5e-11 gives a foot-point CFL of about 1.3. That is deliberately
    // NOT the reference's 5e-12: this solver's MLS semi-Lagrangian transport
    // is unstable at small CFL for this Knudsen number - a cold dense spot
    // grows from about 1000 steps and destroys the field. Verified with a
    // FIXED body, so it is a property of the base scheme and not of the
    // body coupling.
    case TestCase::Cavity2D_Square:
    case TestCase::Cavity2D_Circle:
        BC.rho   = 11.0;
        sp.tao   = 3.37e-11;        // hard-sphere tau ~ 1/rho
        sp.Nx    = 60; sp.Ny = 60; sp.Nz = 1;
        sp.Nv    = 30;
        sp.VMax  = 1500; sp.VMin = -1500;
        sp.dt    = 1.5e-11;
        sp.tfinal = 1e-6;           // one orbit of the driving vortex
        sp.saveEvery = 200;
        break;

    // -- Cavity with a body, 3-D ---------------------------------------
    // rho = 4.4 (Kn = 0.032) rather than the 2-D gas. This is the value the
    // 3-D body cases were established at, and the reason is resolution of
    // the Knudsen layer: lambda/dx = 0.96 here, whereas the denser 2-D gas
    // leaves lambda ~ 0.25 dx unresolved and the lid-edge wall fluxes then
    // heat the gas until the velocity grid truncates. Verified with a FIXED
    // body, so again it is not a body-coupling effect.
    //
    // Nv = 10 gives dv/sigma = 1.22, which is coarse by the usual standard.
    // It is deliberate: refining to Nv = 20 (dv/sigma = 0.58) made the
    // transport LESS stable, not more. Measured against Nv = 20 at the same
    // grid, Nv = 10 changes the body's accumulated turn by 0.14%, so the
    // coarse velocity grid costs nothing that matters here.
    //
    // Nx = 40 is the default. A four-grid study (31/40/50/60) puts the
    // converged turn at about 65.3 degrees, with Nx = 40 some 11% below it
    // and Nx = 50 within 2.7%; raise Nx for production runs.
    case TestCase::Cavity3D_Sphere:
    case TestCase::Cavity3D_Cube:
        BC.rho   = 4.4;
        sp.tao   = 8.425e-11;       // lambda/dx = 0.96 at Nx = 31
        sp.Nx    = 40; sp.Ny = 40; sp.Nz = 40;
        sp.Nv    = 10;
        sp.VMax  = 1303; sp.VMin = -1303;
        sp.dt    = 1.5e-11;         // foot-point CFL 0.76 at Nx = 40
        sp.tfinal = 1.8e-7;
        sp.saveEvery = 500;
        break;
    }

    sp.outDir = std::string("output/") + info.name;
}

void applyLidProfile(TestCase tc, std::vector<Particle> &P,
                     const DomainBoundary &dom)
{
    const CaseInfo &info = caseInfo(tc);
    if (info.lid != LidProfile::ParabolicTop) return;   // uniform: already set

    // Parabolic lid, zero at both corners and peaking at mid-span:
    //     Ux(s) = peak * (2s)^2 * (2 - 2s)^2,   s = x / L
    // Only box-wall particles on y = ytop are touched; body surface
    // particles (bodyId >= 0) carry their own wall state.
    const double L = dom.xright - dom.xleft;
    for (auto &p : P)
    {
        if (!p.boundary || p.bodyId >= 0) continue;
        if (std::fabs(p.y - dom.ytop) > 1e-13) continue;
        const double s = p.x / L;
        p.wUx = info.lidSpeed * (2.0 * s) * (2.0 * s)
                              * (2.0 - 2.0 * s) * (2.0 - 2.0 * s);
    }
}

std::vector<std::unique_ptr<RigidBody>>
makeBodies(TestCase tc, BodyMotion motion, double gasDensity,
           double bodyScale)
{
    std::vector<std::unique_ptr<RigidBody>> bodies;
    if (!caseInfo(tc).hasBody) return bodies;

    const double rhoBody = gasDensity * kBodyDensityRatio;

    // Body centre. Offset from the cavity centre so the body sits inside the
    // primary vortex rather than on its axis, which is what sets it moving.
    constexpr double cx = 6e-7, cy = 7e-7, cz = 5e-7;

    switch (tc)
    {
    case TestCase::Cavity2D_Square:
        bodies.push_back(std::make_unique<Square>(
            cx, cy, rhoBody, kWallTemp, kSquareSide * bodyScale, kSquarePts));
        break;
    case TestCase::Cavity2D_Circle:
        bodies.push_back(std::make_unique<Circle>(
            cx, cy, rhoBody, kWallTemp, kCircleRadius * bodyScale,
            kCirclePts));
        break;
    case TestCase::Cavity3D_Sphere:
        bodies.push_back(std::make_unique<Sphere>(
            cx, cy, cz, rhoBody, kWallTemp, kSphereRadius * bodyScale,
            kSpherePts));
        break;
    case TestCase::Cavity3D_Cube:
        bodies.push_back(std::make_unique<Cube>(
            cx, cy, cz, rhoBody, kWallTemp, kCubeSide * bodyScale,
            kCubePts));
        break;
    default:
        break;
    }

    // A fixed body is held in place by configureCase leaving
    // sp.movingBodies false, so moveRigidBodies is never called. Its loads
    // are still integrated and reported; it simply does not act on them.
    (void)motion;
    return bodies;
}
