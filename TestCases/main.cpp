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

#ifdef ALEBGK_WITH_MPI
// Pre-flight queries for the startup memory check (Solver/MpiExchange.cpp).
int  alebgk_ranksPerNode();
bool alebgk_decompWillEngage(int rowsTotal, bool problemEligible);
#endif

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
        "  --estimate N     project the whole run from step 2; 0 = stop there\n"
        "  --slab           require the slab decomposition under MPI and abort\n"
        "                   rather than fall back to a replicated cloud\n"
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
#ifdef ALEBGK_WITH_MPI
    MPI_Init(&argc, &argv);
#endif
    if (argc < 2)
    {
        std::printf("Usage: alebgk <case> [options]\n");
        listCases();
#ifdef ALEBGK_WITH_MPI
        MPI_Finalize();
#endif
        return 1;
    }

    TestCase tc;
    if (!parseCaseName(argv[1], &tc))
    {
        std::printf("Unknown case '%s'.\n", argv[1]);
        listCases();
#ifdef ALEBGK_WITH_MPI
        MPI_Finalize();
#endif
        return 1;
    }

    BodyMotion motion = BodyMotion::Free;
    double argNx = -1, argNv = -1, argDt = -1, argTf = -1, bodyScale = 1.0;
    // --estimate: time step 2 (step 1 carries warm-up) and project the whole
    // run from it. 0 = print the projection and stop there; 1 = print it and
    // run to tfinal. Left at -1 the instrumentation is off.
    double argEstimate = -1;
    std::string tag;

    bool requireSlab = false;

    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        bool ok = true;
        if      (a == "--slab")       requireSlab = true;
        else if (a == "--fixed")      motion = BodyMotion::Fixed;
        else if (a == "--Nx")         ok = nextValue(argc, argv, i, &argNx);
        else if (a == "--Nv")         ok = nextValue(argc, argv, i, &argNv);
        else if (a == "--dt")         ok = nextValue(argc, argv, i, &argDt);
        else if (a == "--tfinal")     ok = nextValue(argc, argv, i, &argTf);
        else if (a == "--body-scale") ok = nextValue(argc, argv, i, &bodyScale);
        else if (a == "--estimate")   ok = nextValue(argc, argv, i, &argEstimate);
        else if (a == "--tag")
        {
            if (i + 1 >= argc) { ok = false; }
            else tag = argv[++i];
        }
        else
        {
            std::printf("Unknown option '%s'.\n", a.c_str());
            listCases();
#ifdef ALEBGK_WITH_MPI
            MPI_Finalize();
#endif
            return 1;
        }
        if (!ok)
        {
            std::printf("Option '%s' needs a value.\n", a.c_str());
#ifdef ALEBGK_WITH_MPI
            MPI_Finalize();
#endif
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
    if (argEstimate >= 0) sp.estimateFlag = int(argEstimate);
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
    const double cloudGB = requiredMemoryGB(N, sp.Nv, info.dim);
    const double haveGB  = availableMemoryGB();

    // What this node is actually asked for. Under MPI the figure above is one
    // rank's view of the cloud; in replicated mode every rank sharing the node
    // allocates it again, so the node needs ranksPerNode times as much.
    // Comparing one rank's share against the whole node lets a collectively
    // oversized run start, and the OOM killer then takes every rank with
    // signal 9 and no diagnostic of its own.
    const int  rows        = (info.dim == 3) ? sp.Nz : sp.Ny;
    int        ranksPerNode = 1;
    bool       decomposed   = false;
    double     needGB       = cloudGB;
#ifdef ALEBGK_WITH_MPI
    // --slab, or ALEBGK_DECOMP=2 for job scripts that cannot edit the command
    // line. Read before generateParticles so alebgk_decompPlan sees it.
    {
        const char *env = std::getenv("ALEBGK_DECOMP");
        alebgk::requireDecomp = requireSlab || (env && env[0] == '2');
    }
    ranksPerNode = alebgk_ranksPerNode();
    decomposed   = alebgk_decompWillEngage(
                       rows, sp.velMode == VelGridMode::Fixed && !sp.movingBodies);
    if (decomposed)
        // Owned rows plus a four-row halo per side. The halo is duplicated, so
        // this is above cloudGB / P.
        needGB = cloudGB * (1.0 / alebgk::mpiSize() + 8.0 / double(rows))
                 * ranksPerNode;
    else
        needGB = cloudGB * ranksPerNode;
#endif

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
#ifdef ALEBGK_WITH_MPI
    std::printf("  Ranks:  %d total, %d on this node\n",
                alebgk::mpiSize(), ranksPerNode);
    if (alebgk::mpiSize() > 1)
        std::printf("  Decomp: %s%s\n",
                    decomposed ? "SLAB (owned rows + 4 ghost rows per side)"
                               : "REPLICATED (every rank holds the whole cloud)",
                    alebgk::requireDecomp ? "   [--slab: required]" : "");
    std::printf("  Memory: ~%.2f GB per cloud, ~%.2f GB needed on this node",
                cloudGB, needGB);
#else
    std::printf("  Memory: ~%.2f GB needed", needGB);
#endif
    if (haveGB > 0) std::printf(", %.2f GB available", haveGB);
    std::printf("\n  Output: %s\n", sp.outDir.c_str());
    std::printf("==============================================\n\n");
    std::fflush(stdout);

#ifdef ALEBGK_WITH_MPI
    // Refuse before the memory check, so --slab reports the decomposition as
    // the reason rather than the replicated footprint it would have caused.
    // alebgk_decompPlan repeats this with the exact row counts, as the
    // authoritative test; this is the early, clearer exit.
    if (alebgk::requireDecomp && alebgk::mpiSize() > 1 && !decomposed)
    {
        if (ALEBGK_ROOT)
        {
            std::printf("--slab was given but the slab decomposition is not "
                        "available for this configuration.\n");
            std::printf("It needs at least 8 owned rows per rank; %d rows "
                        "across %d ranks gives %d. Use -n %d or fewer, or drop "
                        "--slab to run replicated.\n",
                        rows, alebgk::mpiSize(), rows / alebgk::mpiSize(),
                        rows / 8);
        }
        MPI_Finalize();
        return 2;
    }
#endif

    if (haveGB > 0 && needGB > haveGB)
    {
        if (ALEBGK_ROOT)
        {
            std::printf("Not enough memory for this configuration: this node "
                        "needs ~%.2f GB but has %.2f GB.\n", needGB, haveGB);
#ifdef ALEBGK_WITH_MPI
            if (!decomposed && ranksPerNode > 1)
            {
                // The usual cause: too many ranks for the slab decomposition,
                // so each one keeps a full copy. Fewer ranks is both the fix
                // and a smaller footprint, which is counter-intuitive enough
                // to be worth spelling out.
                std::printf("The slab decomposition is not active, so each of "
                            "the %d ranks on this node holds the whole %.2f GB "
                            "cloud.\n", ranksPerNode, cloudGB);
                const int pMax = rows / 8;          // largest eligible count
                if (pMax >= 2)
                    std::printf("It needs at least 8 owned rows per rank, so "
                                "with %d rows use -n %d or fewer; each rank "
                                "then holds about %.2f GB.\n",
                                rows, pMax,
                                cloudGB * (1.0 / pMax + 8.0 / double(rows)));
                else
                    std::printf("It needs at least 8 owned rows per rank, and "
                                "%d rows cannot be split that way, so this "
                                "case is single-rank only.\n", rows);
            }
            else
#endif
                std::printf("Reduce --Nx or --Nv.\n");
        }
#ifdef ALEBGK_WITH_MPI
        MPI_Finalize();
#endif
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
#ifdef ALEBGK_WITH_MPI
    MPI_Finalize();
#endif
    return 0;
}
