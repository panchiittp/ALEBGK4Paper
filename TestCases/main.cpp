// =============================================================================
// main.cpp - Entry point.
//
//   alebgk <case> [--fixed] [--Nx N] [--Nv N] [--dt T] [--tfinal T]
//                 [--body-scale S] [--tag NAME]
//
// <case> is a number 1-6 or its name; run with no arguments to list them.
// Every case is fully specified by TestCases.cpp, so naming one is enough to
// reproduce it.
// =============================================================================
#include "Config.hpp"
#include "Types.hpp"
#include "Output.hpp"
#include "Geometry.hpp"
#include "Initialization.hpp"
#include "BoundaryConditions.hpp"
#include "TestCases.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fstream>
#include <unistd.h>
#endif

// Time-stepping loop (Solver/Solver.cpp).
void runSolver(std::vector<Particle> &P, const SimParameters &sp,
               const GasConstants &gc, const DomainBoundary &dom,
               const BoundaryConditions &BC, CalcParameters &cp,
               std::vector<std::unique_ptr<RigidBody>> &bodies);

namespace {

double availableMemoryGB()
{
#ifdef _WIN32
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st))
        return double(st.ullAvailPhys) / (1024.0 * 1024.0 * 1024.0);
    return -1.0;
#else
    std::ifstream f("/proc/meminfo");
    std::string key;
    long kb = 0;
    while (f >> key)
    {
        if (key == "MemAvailable:") { f >> kb; break; }
        std::getline(f, key);
    }
    return kb > 0 ? double(kb) / (1024.0 * 1024.0) : -1.0;
#endif
}

// Dominant cost is the distribution function: Nv^3 doubles per particle in
// 3-D (2*Nv^2 in 2-D), held twice (g and the transport scratch gt).
double requiredMemoryGB(long N, int Nv, int ndim)
{
    const double perParticle =
        (ndim == 3) ? double(Nv) * Nv * Nv : 2.0 * double(Nv) * Nv;
    return 2.0 * double(N) * perParticle * sizeof(double) /
           (1024.0 * 1024.0 * 1024.0);
}

void listCases()
{
    std::printf("\nAvailable cases:\n\n");
    const CaseInfo *c = allCases();
    for (int i = 0; i < kNumTestCases; ++i)
        std::printf("  %d  %-16s  %s\n", int(c[i].id), c[i].name,
                    c[i].description);
    std::printf("\nOptions:\n"
        "  --fixed          hold the body at its initial pose (body cases)\n"
        "  --Nx N           particles per spatial direction\n"
        "  --Nv N           velocity-grid points per direction\n"
        "  --dt T           time step, s\n"
        "  --tfinal T       final time, s\n"
        "  --body-scale S   scale the body's linear size (1.0 = reference)\n"
        "  --tag NAME       suffix for the output directory\n\n"
        "Example:  alebgk cavity3d-cube --Nx 50 --tfinal 1.8e-7\n\n");
}

// Returns false if the value is missing or malformed.
bool nextValue(int argc, char **argv, int &i, double *out)
{
    if (i + 1 >= argc) return false;
    char *end = nullptr;
    double v = std::strtod(argv[++i], &end);
    if (end == argv[i]) return false;
    *out = v;
    return true;
}

}  // namespace

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::printf("Usage: alebgk <case> [options]\n");
        listCases();
        return 1;
    }

    TestCase tc;
    if (!parseCaseName(argv[1], &tc))
    {
        std::printf("Unknown case '%s'.\n", argv[1]);
        listCases();
        return 1;
    }

    BodyMotion motion = BodyMotion::Free;
    double argNx = -1, argNv = -1, argDt = -1, argTf = -1, bodyScale = 1.0;
    std::string tag;

    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        bool ok = true;
        if      (a == "--fixed")      motion = BodyMotion::Fixed;
        else if (a == "--Nx")         ok = nextValue(argc, argv, i, &argNx);
        else if (a == "--Nv")         ok = nextValue(argc, argv, i, &argNv);
        else if (a == "--dt")         ok = nextValue(argc, argv, i, &argDt);
        else if (a == "--tfinal")     ok = nextValue(argc, argv, i, &argTf);
        else if (a == "--body-scale") ok = nextValue(argc, argv, i, &bodyScale);
        else if (a == "--tag")
        {
            if (i + 1 >= argc) { ok = false; }
            else tag = argv[++i];
        }
        else
        {
            std::printf("Unknown option '%s'.\n", a.c_str());
            listCases();
            return 1;
        }
        if (!ok)
        {
            std::printf("Option '%s' needs a value.\n", a.c_str());
            return 1;
        }
    }

    const CaseInfo &info = caseInfo(tc);
    SimParameters      sp;
    DomainBoundary     dom;
    BoundaryConditions BC;
    GasConstants       gc;
    CalcParameters     cp{};

    configureCase(tc, motion, sp, dom, BC, gc);
    sp.problem = ProblemID::DrivenCavity;   // shared initial state for all six

    // Command-line overrides, applied after the case defaults.
    if (argNx > 0)
    {
        sp.Nx = sp.Ny = int(argNx);
        if (info.dim == 3) sp.Nz = int(argNx);
    }
    if (argNv > 0) sp.Nv = sp.NvMin = sp.NvMax = int(argNv);
    if (argDt > 0) sp.dt = argDt;
    if (argTf > 0) sp.tfinal = argTf;

    sp.outDir += "_Nx" + std::to_string(sp.Nx) + "_Nv" + std::to_string(sp.Nv);
    if (motion == BodyMotion::Fixed && info.hasBody) sp.outDir += "_fixed";
    if (!tag.empty()) sp.outDir += "_" + tag;

    const long N = long(sp.Nx) * sp.Ny * (info.dim == 3 ? sp.Nz : 1);
    const double needGB = requiredMemoryGB(N, sp.Nv, info.dim);
    const double haveGB = availableMemoryGB();

    std::printf("\n==============================================\n");
    std::printf("  ALE-BGK meshfree kinetic solver\n");
    std::printf("  Case %d: %s\n", int(tc), info.description);
    if (info.hasBody)
        std::printf("  Body:   %s\n",
                    motion == BodyMotion::Free ? "free to translate and rotate"
                                               : "fixed");
    std::printf("  Grid:   %d^%d particles, Nv = %d\n",
                sp.Nx, info.dim, sp.Nv);
    std::printf("  Time:   dt = %.3e s, tfinal = %.3e s (%ld steps)\n",
                sp.dt, sp.tfinal, long(sp.tfinal / sp.dt + 0.5));
    std::printf("  Memory: ~%.2f GB needed", needGB);
    if (haveGB > 0) std::printf(", %.2f GB available", haveGB);
    std::printf("\n  Output: %s\n", sp.outDir.c_str());
    std::printf("==============================================\n\n");
    std::fflush(stdout);

    if (haveGB > 0 && needGB > haveGB)
    {
        std::printf("Not enough memory for this configuration. "
                    "Reduce --Nx or --Nv.\n");
        return 1;
    }

    const auto t0 = std::chrono::high_resolution_clock::now();

    auto bodies = makeBodies(tc, motion, BC.rho, bodyScale);

    std::vector<Particle> P;
    generateParticles(P, sp, dom, cp);
    embedRigidBodies(P, bodies, sp, cp);
    updateVoxelsAndNeighbours(P, cp, dom);

    // Generic wall state first, then the case's lid. Body surface particles
    // already carry their own wall state and are left untouched by both.
    initWallData(P, dom, BC);
    applyLidProfile(tc, P, dom);

    // The initial distribution is set by runSolver, which calls applyIC
    // once the cloud and its wall state are complete.

    double avgNeigh = 0.0;
    for (const auto &q : P) avgNeigh += double(q.neighindex.size());
    avgNeigh /= double(P.size());
    std::printf("Particles: %d   average neighbours: %.1f\n\n",
                cp.N, avgNeigh);
    std::fflush(stdout);

    runSolver(P, sp, gc, dom, BC, cp, bodies);

    const double total = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();
    std::printf("Total wall time: %.2f s\n", total);
    return 0;
}
