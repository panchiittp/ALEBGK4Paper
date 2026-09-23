#pragma once
// =============================================================================
// Config.hpp — build-configuration and portability layer for ALE-BGK.
//
// One source tree, three execution backends selected by macros (set by
// CMake options, never by editing code):
//
//   (default)           serial CPU, optional OpenMP threading
//   ALEBGK_WITH_MPI     rank-parallel CPU: every kernel loops over its
//                       rank's chunk [chunkLo, chunkHi) of the particle
//                       array and the chunks are Allgather'd at the three
//                       sync points per step (after transport commit,
//                       after the wall BC, and after collision+moments)
//   ALEBGK_WITH_CUDA    GPU offload of the per-particle kernels
//                       (transport, collision, moments); host keeps
//                       problem setup, bodies, and I/O
//
// Platform notes (Linux/GCC-Clang and Windows/MSVC):
//   * MSVC does not define M_PI unless _USE_MATH_DEFINES precedes
//     <cmath> — this header must therefore be included FIRST everywhere.
//   * OpenMP pragmas are limited to OpenMP 2.0 constructs (MSVC's cap).
//   * No VLAs, no GNU extensions anywhere in the tree.
// =============================================================================

#if defined(_MSC_VER) && !defined(_USE_MATH_DEFINES)
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Host/device qualifiers ───────────────────────────────────────────────────
// Core per-node math (Maxwellians, MLS basis, linear solves, min-image)
// is plain functions marked ALEBGK_HD so the CUDA backend compiles the
// exact same expressions the CPU backends run.
#if defined(ALEBGK_WITH_CUDA) && defined(__CUDACC__)
#define ALEBGK_HD __host__ __device__
#else
#define ALEBGK_HD
#endif

// ── MPI rank chunking ────────────────────────────────────────────────────────
#ifdef ALEBGK_WITH_MPI
#include <mpi.h>
namespace alebgk
{
inline int mpiRank()
{
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}
inline int mpiSize()
{
    int n = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    return n;
}
/* Domain decomposition state (defined in MpiExchange.cpp).  When the
   decomposed mode is active, every rank stores only its slab of the
   lattice (owned rows plus halo/ghost rows) and chunk() returns the
   owned index range of that LOCAL array — so all kernels, which already
   bound their loops with chunk(), compute exactly the owned particles
   and read ghosts only through their neighbour lists. */
extern bool decompOn;
extern int  decompLo, decompHi;

/* This rank's chunk [lo, hi) of an N-element particle array. */
inline void chunk(int N, int *lo, int *hi)
{
    if (decompOn) { *lo = decompLo; *hi = decompHi; return; }
    int r = mpiRank(), n = mpiSize();
    *lo = static_cast<int>(static_cast<long long>(r) * N / n);
    *hi = static_cast<int>(static_cast<long long>(r + 1) * N / n);
}
} // namespace alebgk
#else
namespace alebgk
{
inline int  mpiRank() { return 0; }
inline int  mpiSize() { return 1; }
inline void chunk(int N, int *lo, int *hi) { *lo = 0; *hi = N; }
} // namespace alebgk
#endif

// Root-only console output (all backends)
#define ALEBGK_ROOT (alebgk::mpiRank() == 0)
