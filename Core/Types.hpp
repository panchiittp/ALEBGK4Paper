#pragma once
// =============================================================================
// Types.hpp — Core data structures for the serial CPU ALE-BGK kinetic solver
// =============================================================================

#include "Config.hpp"
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

/* ------------------------------------------------------------------ */
/*  Compile-time limits                                                */
/* ------------------------------------------------------------------ */
constexpr int MAX_NV     = 48;
// 3-D neighbour search (radius = 2.5*dx) finds ~(4/3)*pi*2.5^3 ~= 65
// neighbours for an interior particle -- above the old 64 cap, which
// silently truncated the MLS stencil asymmetrically (whichever voxels the
// search loop reached first) for every 3-D case. 2-D only ever needs
// ~pi*2.5^2 ~= 20, so this was invisible there. 128 gives full headroom.
constexpr int MAX_NEIGH  = 128;
constexpr int MAX_G_SIZE = MAX_NV * MAX_NV * 3;
constexpr int MLS_MAX_N  = 10;

/* ------------------------------------------------------------------ */
/*  Enumerations                                                       */
/* ------------------------------------------------------------------ */
enum class ProblemID : int
{
    DrivenCavity     = 0,
    KelvinHelmholtz  = 1,
    SodShockTube     = 2,
    LaxShockTube     = 3,
    RiemannConfig12  = 4,
    TaylorGreen      = 5,
    RayleighTaylor   = 6,
    CouetteFlow      = 7,
    ThermalConductor = 8,
    KHLayered        = 9,  // 3-layer Kelvin-Helmholtz (HOFKS §5.3)
    ShearLayer       = 10  // double shear layer (meshfree4bgk2d, sim1)
};

enum class Dimension : int
{
    D2 = 2,
    D3 = 3
};

enum class VelGridMode : int
{
    Fixed    = 0,
    Adaptive = 1
};

enum class LimiterMode : int
{
    None  = 0,
    MUSCL = 1,
    MOOD  = 2
};

enum class TimeScheme : int
{
    Euler     = 0,
    SSP_RK2   = 1,
    IMEX_SSP3 = 2
};

/* ------------------------------------------------------------------ */
/*  Boundary / initial-condition descriptors                           */
/* ------------------------------------------------------------------ */
struct DomainBoundary
{
    double xleft   = 0.0;
    double xright  = 1.0;
    double ytop    = 1.0;
    double ybottom = 0.0;
    double zfront  = 1.0;
    double zback   = 0.0;
};

struct UBC
{
    double Ux = 0.0;
    double Uy = 0.0;
    double Uz = 0.0;
};

struct BoundaryConditions
{
    double Tw  = 270.0;
    double rho = 1.0;

    /* Optional top-wall temperature override (used when > 0); lets
       stratified problems (RT) match the hydrostatic T at each wall. */
    double TwTop = -1.0;

    UBC Left;
    UBC Right;
    UBC Top;
    UBC Bottom;
    UBC Front;
    UBC Back;
};

/* ------------------------------------------------------------------ */
/*  Gas model                                                          */
/* ------------------------------------------------------------------ */
struct GasConstants
{
    double R     = 208.0;
    double alpha = 6.0;
    double gamma = 5.0 / 3.0;
};

/* ------------------------------------------------------------------ */
/*  Simulation parameters                                              */
/* ------------------------------------------------------------------ */
struct SimParameters
{
    double VMax   = 0.0;
    double VMin   = 0.0;
    double d      = 0.0;
    double tao    = 0.0;
    double dt     = 0.0;
    double tfinal = 0.0;
    double r      = 0.0;
    double rb     = 0.0;

    int Nx = 0;
    int Ny = 0;
    int Nz = 0;
    int Nv = 0;

    Dimension   dim       = Dimension::D3;
    VelGridMode velMode   = VelGridMode::Adaptive;
    int         NvMin     = 8;
    int         NvMax     = 48;
    double      velCutoff = 5.5;

    LimiterMode limiter    = LimiterMode::MOOD;
    TimeScheme  timeScheme = TimeScheme::IMEX_SSP3;

    /* MLS transport reconstruction order: 1 = linear (3 terms in 2D),
       2 = quadratic (6 terms in 2D — far less numerical diffusion at
       small CFL; 3D currently stays linear). */
    int mlsOrder = 1;

    /* Constant body acceleration in y (e.g. gravity for Rayleigh-Taylor).
       Applied as a per-step velocity-space shift of the distribution. */
    double gravityY = 0.0;

    /* Integrate rigid-body motion (forces, Newton-Euler, cloud
       management) each step. Static-body problems leave this off. */
    bool movingBodies = false;
    /* Variable relaxation time tau(rho,T) from the hard-sphere mean free
       path (meshfree4bgk2d's tauFunc). Default OFF. */
    bool tauVariable = false;

    double meshVx = 0.0;
    double meshVy = 0.0;
    double meshVz = 0.0;

    int         saveEvery = 50;
    // CLI [estimate] flag: -1 off (default), 0 estimate-only (stop at
    // step 2 or 50 per the projected cost), 1 print estimates, run fully.
    int         estimateFlag = -1;
    std::string outDir    = "output";
    ProblemID   problem   = ProblemID::DrivenCavity;
};

/* ------------------------------------------------------------------ */
/*  Derived / calculated mesh parameters                               */
/* ------------------------------------------------------------------ */
struct CalcParameters
{
    double dx  = 0.0;
    double dy  = 0.0;
    double dz  = 0.0;
    double dv  = 0.0;
    double dv2 = 0.0;

    int N = 0;

    double hBox = 0.0;
    double vBox = 0.0;
    double wBox = 0.0;

    int nbhBox = 0;
    int nbvBox = 0;
    int nbwBox = 0;
    int nvox   = 0;

    double radius  = 0.0;
    double minDist = 0.0;

    int Nv_actual = 0;
    int dim       = 3;

    /* Per-axis periodicity: seam-free lattice, wrapping neighbour
       search, and minimum-image displacements in the MLS stencil on
       each periodic axis; diffuse walls on the others. */
    bool   periodicX = false;
    bool   periodicY = false;
    bool   periodicZ = false;
    double Lx = 0.0, Ly = 0.0, Lz = 0.0;
};

/* ------------------------------------------------------------------ */
/*  Particle (one spatial node)                                        */
/* ------------------------------------------------------------------ */
struct Particle
{
    double x  = 0.0, y  = 0.0, z  = 0.0;
    double ux = 0.0, uy = 0.0, uz = 0.0;
    double meshVx = 0.0, meshVy = 0.0, meshVz = 0.0;
    double T   = 270.0;
    double rho = 1.0;
    double p   = 0.0;

    bool boundary = false;
    bool validg   = false;

    int voxel    = 0;
    int Nv_local = 0;
    int dim      = 3;

    double vlo      = 0.0;
    double vhi      = 0.0;
    double dv_local = 0.0;

    bool moodFlag = false;

    /* Stored wall state for boundary particles: outward (gas-facing)
       normal, wall velocity, wall temperature. Initialised once at setup
       from getWallNormal (box walls, possibly with a profile) or from a
       rigid body. bodyId = -1 for box walls, >= 0 for body surfaces. */
    int    bodyId = -1;
    double wnx = 0.0, wny = 0.0, wnz = 0.0;
    double wUx = 0.0, wUy = 0.0, wUz = 0.0;
    double wT  = 0.0;

    std::vector<int>    neighindex;
    std::vector<double> g;
    std::vector<double> gt;

    /* nvel = velocity-grid points per dimension, ndim = spatial dimension.
       2-D: Chu reduction → two reduced distributions on an nvel² grid
            (nvel²·2 values: g1 then g2).
       3-D: single distribution on a full nvel³ velocity grid. */
    Particle(int nvel, int ndim = 3)
        : T(270.0), rho(1.0), Nv_local(nvel), dim(ndim)
    {
        int gSize = (ndim == 3) ? nvel * nvel * nvel : nvel * nvel * 2;
        g .assign(gSize, 0.0);
        gt.assign(gSize, 0.0);
    }

    Particle() = default;
};
