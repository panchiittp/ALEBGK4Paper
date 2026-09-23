// =============================================================================
// CudaKernels.cu — GPU backend (compiled only when ALEBGK_WITH_CUDA).
//
// The device kernels mirror the validated CPU kernels in Kernels.cpp
// one-to-one and call the same ALEBGK_HD core math from Functions.hpp
// (Maxwellians, MLS basis, minimum-image), so all three backends share a
// single source of numerical truth. Data lives on the device as flat SoA
// arrays for the whole run; the host downloads moments/g only at output
// steps.
//
// Scope of the GPU path (documented in the README):
//   * Fixed velocity grid (uniform Nv per particle),
//   * static point clouds and static rigid bodies (the moving-body cloud
//     management is host-side and stays on the CPU backends),
//   * sm_60+ uses native double atomicAdd; sm<60 uses a CAS fallback (see below).
// =============================================================================
#include "Config.hpp"

#ifdef ALEBGK_WITH_CUDA

#include "Types.hpp"
#include "MLSBasis.hpp"
#include "Geometry.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <climits>
#include <cstring>
#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>

// Portable double atomicAdd.
// sm_60+ (Pascal, Volta, Turing, Ampere, Hopper) has it natively.
// Older architectures (Maxwell sm_52, Kepler) need a CAS loop.
// The __CUDA_ARCH__ guard is evaluated per-PTX compilation pass:
//   host pass: guard is false → no definition, no conflict with host headers
//   device pass sm<60: injects the CAS implementation
//   device pass sm>=60: guard is false → native instruction used
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ inline double atomicAdd(double* addr, double val)
{
    unsigned long long int* a =
        reinterpret_cast<unsigned long long int*>(addr);
    unsigned long long int old = *a, assumed;
    do {
        assumed = old;
        old = atomicCAS(a, assumed,
                        __double_as_longlong(val +
                            __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

#define CUCHECK(x)                                                          \
    do {                                                                    \
        cudaError_t e = (x);                                                \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                    cudaGetErrorString(e), __FILE__, __LINE__);             \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// ── Device-side particle cloud (SoA) ─────────────────────────────────────────
struct DeviceCloud
{
    int    N = 0, Nv = 0, layers = 2, maxNeigh = MAX_NEIGH, dim = 2;
    // Per-particle distribution length: 2*Nv^2 (2-D Chu g1/g2) or Nv^3
    // (3-D single f). Set once in cudaUploadCloud; used by every kernel so
    // the dimension-agnostic ones (commit, scale) need no branch.
    size_t gStride = 0;
    double dv = 0, vlo = 0;
    double *x = nullptr, *y = nullptr, *z = nullptr;
    double *g = nullptr, *gt = nullptr;               // N * gStride
    double *rho = nullptr, *ux = nullptr, *uy = nullptr, *uz = nullptr,
           *T = nullptr;
    unsigned char *boundary = nullptr;
    double *wnx = nullptr, *wny = nullptr, *wnz = nullptr;
    double *wUx = nullptr, *wUy = nullptr, *wUz = nullptr, *wT = nullptr;
    double *mvx = nullptr, *mvy = nullptr, *mvz = nullptr;  // ALE mesh vel
    int    *neigh = nullptr, *nneigh = nullptr;       // N*maxNeigh, N
    double *scratch = nullptr;                        // reductions
    // Per-particle limiter activity for THIS step, as a bitmask:
    //   1 = kPositivity rescaled this particle (negative mass repaired)
    //   2 = kNAD3D clamped it (density outside the neighbour range)
    // Zeroed at the top of every cudaStep. Without this the GPU path had no
    // way to report limiter activity at all -- mood_count came from the host
    // Particle::moodFlag, which nothing on the device ever wrote, so it read
    // 0 forever regardless of what the limiters actually did.
    double *limFlag = nullptr;
    // Separate integer flag word for the MLS diagnostics. limFlag is a DOUBLE
    // accumulated with +=, which works for positivity and NAD because each
    // fires at most once per particle (kNAD3D is if/else-if). The MLS checks
    // fire PER VELOCITY NODE, so adding a bit value there overflows into the
    // neighbouring bit positions and corrupts every count in the word. An
    // int with atomicOr is idempotent and cannot alias.
    int *mlsFlag = nullptr;
    // Per-particle L1 sum of the extrapolation overshoot, in density units
    // (excess * dv^3). The COUNT of extrapolating particles turned out to be
    // the wrong observable -- it sits flat at ~230 (0.78%) for 11,500 steps
    // and does not move before the onset at 11,587. That is what a linear
    // instability seeded by a FIXED set of sites looks like: the amplitude
    // grows, the number of sites does not. So track the amplitude.
    double *mlsOvr = nullptr;
    // Diagnostic: per-particle x-momentum change caused by kWallBC3D this
    // step. A stationary diffuse wall facing gas at its own temperature must
    // be exactly momentum-neutral; this records where it is not.
    double *bcdPx = nullptr;
};

static DeviceCloud dc;

// Limiter firing counts from the most recent download (see cudaLimiterCounts).
static int g_limPos = 0, g_limNad = 0;
static int g_limMlsDeg = 0, g_limMlsExt = 0;
static double g_massAfterCorr = 0.0, g_massAfterColl = 0.0;
static double g_mlsOvrSum = 0.0, g_mlsOvrMax = 0.0;

// ── Kernels ──────────────────────────────────────────────────────────────────
__global__ void kMoments(DeviceCloud c, double R)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Boundary particles keep the imposed wall state (set in kWallBC);
    // don't overwrite it with the mixed-distribution slip moment.
    if (i >= c.N || c.boundary[i]) return;
    int Nv = c.Nv, Nv2 = Nv * Nv;
    double dv2 = c.dv * c.dv;
    const double *g = c.g + (size_t)i * Nv2 * c.layers;

    double rho = 0, rux = 0, ruy = 0, E = 0;
    for (int jv = 0; jv < Nv; ++jv)
    {
        double vy = c.vlo + jv * c.dv;
        for (int iv = 0; iv < Nv; ++iv)
        {
            double vx = c.vlo + iv * c.dv;
            double g1 = g[iv + Nv * jv];
            double g2 = g[iv + Nv * jv + Nv2];
            rho += g1 * dv2;
            rux += vx * g1 * dv2;
            ruy += vy * g1 * dv2;
            E   += (vx * vx + vy * vy) * g1 * dv2 + 2.0 * g2 * dv2;
        }
    }
    if (rho > 1e-30)
    {
        c.rho[i] = rho;
        c.ux[i]  = rux / rho;
        c.uy[i]  = ruy / rho;
        double u2 = c.ux[i] * c.ux[i] + c.uy[i] * c.uy[i];
        double T  = (E / rho - u2) / (3.0 * R);
        c.T[i] = T > 1e-6 ? T : 1e-6;
    }
}

__global__ void kCollision(DeviceCloud c, double dt, double tao, double R)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.N || c.boundary[i]) return;
    int Nv = c.Nv, Nv2 = Nv * Nv;
    double lam = dt / tao, coeff = 1.0 / (1.0 + lam);
    double *g = c.g + (size_t)i * Nv2 * c.layers;
    double rho = c.rho[i], ux = c.ux[i], uy = c.uy[i], T = c.T[i];

    for (int jv = 0; jv < Nv; ++jv)
    {
        double vy = c.vlo + jv * c.dv;
        for (int iv = 0; iv < Nv; ++iv)
        {
            double vx = c.vlo + iv * c.dv;
            int lin = iv + Nv * jv;
            double Mg1 = maxwellianG1(vx, vy, ux, uy, T, rho, R);
            g[lin]       = coeff * (g[lin]       + lam * Mg1);
            g[lin + Nv2] = coeff * (g[lin + Nv2] + lam * 0.5 * R * T * Mg1);
        }
    }
}

// Node-parallel MLS semi-Lagrangian transport: ONE THREAD PER
// (particle, velocity node). The previous one-thread-per-particle design
// serialised all Nv^2 nodes inside a single register-heavy thread and
// starved the GPU of occupancy (measured: an A100 ran no faster than a
// 64-core CPU). Splitting the node loop across threads exposes N*Nv^2-way
// parallelism; each thread performs one neighbour gather and one small
// Gaussian elimination. gt is pre-filled by kCopyGt (coalesced), so
// outgoing boundary nodes keep their value by simply returning early.
__global__ void kCopyGt(DeviceCloud c)
{
    size_t total = (size_t)c.N * c.gStride;
    for (size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         k < total; k += (size_t)gridDim.x * blockDim.x)
        c.gt[k] = c.g[k];
}

__global__ void kTransportNode(DeviceCloud c, double dt, double h,
                               int mlsOrder,
                               int periodicX, int periodicY,
                               double Lx, double Ly)
{
    int Nv = c.Nv, Nv2 = Nv * Nv;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)c.N * Nv2;
    if (tid >= total) return;
    int i   = (int)(tid / Nv2);
    int lin = (int)(tid % Nv2);
    int iv = lin % Nv, jv = lin / Nv;
    double vx = c.vlo + iv * c.dv;
    double vy = c.vlo + jv * c.dv;

    if (c.boundary[i])
    {
        double cn = (vx - c.wUx[i]) * c.wnx[i]
                  + (vy - c.wUy[i]) * c.wny[i];
        if (cn >= 0.0) return;      // outgoing: keep kCopyGt value
    }
    int poly = (mlsOrder >= 2 && !c.boundary[i]) ? 6 : 3;

    double xf = c.x[i] - (vx - c.mvx[i]) * dt;
    double yf = c.y[i] - (vy - c.mvy[i]) * dt;

    double A[36], b1[6], b2[6], s1[6], s2[6], basis[10];
    for (int q = 0; q < poly * poly; ++q) A[q] = 0;
    for (int q = 0; q < poly; ++q) { b1[q] = 0; b2[q] = 0; }
    bool has = false;

    int nn = c.nneigh[i];
    const int *nb = c.neigh + (size_t)i * c.maxNeigh;
    for (int m = 0; m < nn; ++m)
    {
        int j = nb[m];
        double dxn = c.x[j] - xf, dyn = c.y[j] - yf;
        if (periodicX) dxn = minImage(dxn, Lx);
        if (periodicY) dyn = minImage(dyn, Ly);
        double r = sqrt(dxn * dxn + dyn * dyn);
        double w = mlsWeight(r, h);
        if (w < 1e-30) continue;
        fillBasis(dxn, dyn, 0.0, 2, h, basis, poly >= 6 ? 2 : 1);
        for (int p1 = 0; p1 < poly; ++p1)
            for (int p2 = 0; p2 < poly; ++p2)
                A[p1 * poly + p2] += w * basis[p1] * basis[p2];
        const double *gj = c.g + (size_t)j * c.gStride;
        double gn1 = gj[lin], gn2 = gj[lin + Nv2];
        for (int q = 0; q < poly; ++q)
        {
            b1[q] += w * basis[q] * gn1;
            b2[q] += w * basis[q] * gn2;
        }
        has = true;
    }
    if (!has) return;

    for (int col = 0; col < poly; ++col)
    {
        int piv = col;
        double mx = fabs(A[col * poly + col]);
        for (int row = col + 1; row < poly; ++row)
            if (fabs(A[row * poly + col]) > mx)
            { mx = fabs(A[row * poly + col]); piv = row; }
        if (piv != col)
        {
            for (int q = 0; q < poly; ++q)
            {
                double tmp = A[col * poly + q];
                A[col * poly + q] = A[piv * poly + q];
                A[piv * poly + q] = tmp;
            }
            double tb = b1[col]; b1[col] = b1[piv]; b1[piv] = tb;
            tb = b2[col]; b2[col] = b2[piv]; b2[piv] = tb;
        }
        double d = A[col * poly + col];
        if (fabs(d) < 1e-30) continue;
        for (int row = col + 1; row < poly; ++row)
        {
            double f = A[row * poly + col] / d;
            for (int q = col; q < poly; ++q)
                A[row * poly + q] -= f * A[col * poly + q];
            b1[row] -= f * b1[col];
            b2[row] -= f * b2[col];
        }
    }
    for (int row = poly - 1; row >= 0; --row)
    {
        double v1 = b1[row], v2 = b2[row];
        for (int q = row + 1; q < poly; ++q)
        {
            v1 -= A[row * poly + q] * s1[q];
            v2 -= A[row * poly + q] * s2[q];
        }
        double d = A[row * poly + row];
        s1[row] = fabs(d) > 1e-30 ? v1 / d : 0.0;
        s2[row] = fabs(d) > 1e-30 ? v2 / d : 0.0;
    }
    double *gt = c.gt + (size_t)i * c.gStride;
    gt[lin]       = s1[0];
    gt[lin + Nv2] = s2[0];
}

// Mass-conserving positivity fix on gt (g1 layer): one thread per particle
// (light: no gathers, no solves).
__global__ void kPositivity(DeviceCloud c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.N) return;
    int Nv2 = c.Nv * c.Nv;
    double *gt = c.gt + (size_t)i * c.gStride;
    double mPos = 0, mNeg = 0;
    for (int k = 0; k < Nv2; ++k)
    {
        double v = gt[k];
        if (v >= 0) mPos += v; else mNeg += v;
    }
    if (mNeg < -1e-30 && mPos > 1e-30 && (mPos + mNeg) > 0)
    {
        c.limFlag[i] += 1.0;               // positivity repair fired here
        double sc = (mPos + mNeg) / mPos;
        for (int k = 0; k < Nv2; ++k)
        {
            if (gt[k] < 0) { gt[k] = 0; gt[k + Nv2] = 0; }
            else           { gt[k] *= sc; gt[k + Nv2] *= sc; }
        }
    }
}

__global__ void kCommit(DeviceCloud c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.N) return;
    size_t n = c.gStride;
    double *g = c.g + (size_t)i * n;
    const double *gt = c.gt + (size_t)i * n;
    for (size_t k = 0; k < n; ++k) g[k] = gt[k];
}

__global__ void kWallBC(DeviceCloud c, double R)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.N || !c.boundary[i]) return;
    int Nv = c.Nv, Nv2 = Nv * Nv;
    double dv2 = c.dv * c.dv;
    double *g = c.g + (size_t)i * Nv2 * c.layers;
    double nx = c.wnx[i], ny = c.wny[i];
    double Uwx = c.wUx[i], Uwy = c.wUy[i], Tw = c.wT[i];
    double RT2w = 2.0 * R * Tw;

    double intM = 0, intP = 0;
    for (int jv = 0; jv < Nv; ++jv)
    {
        double vy = c.vlo + jv * c.dv;
        for (int iv = 0; iv < Nv; ++iv)
        {
            double vx = c.vlo + iv * c.dv;
            double cn = (vx - Uwx) * nx + (vy - Uwy) * ny;
            int lin = iv + Nv * jv;
            if (cn < 0) intM += cn * g[lin] * dv2;
            if (cn > 0)
            {
                double Mw = exp(-((vx - Uwx) * (vx - Uwx)
                                + (vy - Uwy) * (vy - Uwy)) / RT2w)
                          / (M_PI * RT2w);
                intP += cn * Mw * dv2;
            }
        }
    }
    double rho_w = (intP > 1e-30) ? (-intM / intP) : 1.0;
    for (int jv = 0; jv < Nv; ++jv)
    {
        double vy = c.vlo + jv * c.dv;
        for (int iv = 0; iv < Nv; ++iv)
        {
            double vx = c.vlo + iv * c.dv;
            double cn = (vx - Uwx) * nx + (vy - Uwy) * ny;
            int lin = iv + Nv * jv;
            if (cn > 0)
            {
                double Mw = rho_w
                          * exp(-((vx - Uwx) * (vx - Uwx)
                                + (vy - Uwy) * (vy - Uwy)) / RT2w)
                          / (M_PI * RT2w);
                g[lin]       = Mw;
                g[lin + Nv2] = 0.5 * R * Tw * Mw;
            }
        }
    }
    c.rho[i] = rho_w; c.ux[i] = Uwx; c.uy[i] = Uwy; c.T[i] = Tw;
}

__global__ void kSumMass(DeviceCloud c, double *total)
{
    // Hierarchical reduction: per-thread partial -> shared-memory block
    // tree -> ONE atomic per block. The naive version (one atomicAdd per
    // particle to a single address) fully serialised and cost 2.3 s/call
    // on an A100 at N=160k -- 97% of the whole step.
    __shared__ double sh[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double m = 0;
    if (i < c.N)
    {
        int Nv2 = c.Nv * c.Nv;
        double dv2 = c.dv * c.dv;
        const double *g = c.g + (size_t)i * c.gStride;
        for (int k = 0; k < Nv2; ++k) m += g[k] * dv2;
    }
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int sft = blockDim.x / 2; sft > 0; sft >>= 1)
    {
        if (threadIdx.x < sft) sh[threadIdx.x] += sh[threadIdx.x + sft];
        __syncthreads();
    }
    if (threadIdx.x == 0) total[blockIdx.x] = sh[0];
}

__global__ void kSumMass3D(DeviceCloud c, double *total)
{
    __shared__ double sh[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double m = 0;
    if (i < c.N)
    {
        double dv3 = c.dv * c.dv * c.dv;
        const double *g = c.g + (size_t)i * c.gStride;
        for (size_t k = 0; k < c.gStride; ++k) m += g[k] * dv3;
    }
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int sft = blockDim.x / 2; sft > 0; sft >>= 1)
    {
        if (threadIdx.x < sft) sh[threadIdx.x] += sh[threadIdx.x + sft];
        __syncthreads();
    }
    if (threadIdx.x == 0) total[blockIdx.x] = sh[0];
}

// Diagnostic only (energy-budget instrumentation): total kinetic+thermal
// energy Sum (vx^2+vy^2+vz^2)*g*dv^3 over every particle, mirroring
// kSumMass3D's reduction exactly so it brackets the same population mass
// does. Used to find which pipeline stage injects the energy that grows
// unboundedly while mass stays exactly conserved by scaleG.
__global__ void kSumEnergy3D(DeviceCloud c, double *total, int useGt)
{
    __shared__ double sh[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double e = 0;
    if (i < c.N)
    {
        double dv3 = c.dv * c.dv * c.dv;
        const double *g = (useGt ? c.gt : c.g) + (size_t)i * c.gStride;
        int Nv = c.Nv;
        for (int kv = 0; kv < Nv; ++kv)
        {
            double vz = c.vlo + kv * c.dv;
            for (int jv = 0; jv < Nv; ++jv)
            {
                double vy = c.vlo + jv * c.dv;
                for (int iv = 0; iv < Nv; ++iv)
                {
                    double vx = c.vlo + iv * c.dv;
                    e += (vx*vx + vy*vy + vz*vz)
                       * g[iv + Nv * (jv + Nv * kv)] * dv3;
                }
            }
        }
    }
    sh[threadIdx.x] = e;
    __syncthreads();
    for (int sft = blockDim.x / 2; sft > 0; sft >>= 1)
    {
        if (threadIdx.x < sft) sh[threadIdx.x] += sh[threadIdx.x + sft];
        __syncthreads();
    }
    if (threadIdx.x == 0) total[blockIdx.x] = sh[0];
}

__global__ void kScaleG(DeviceCloud c, double s)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.N) return;
    size_t n = c.gStride;
    double *g = c.g + (size_t)i * n;
    for (size_t k = 0; k < n; ++k) g[k] *= s;
}

// =============================================================================
//  3-D single-distribution kernels (full Nv^3 velocity grid, no Chu reduction)
//
//  Genuine 3-D resolves all three velocity components, so a single f carries
//  every moment (there is no g2/g3). These mirror the CPU 3-D branches in
//  Kernels.cpp one-to-one and share the same ALEBGK_HD math from Functions.hpp
//  (maxwellian3D, fillBasis order-2 10-term, minImage). Storage is Nv^3 per
//  particle (c.gStride); z-transport uses the true grid vz.
// =============================================================================

// Gaussian elimination with partial pivoting (in place on A,b), matching the
// CPU solveLinearN so serial/GPU agree to roundoff. n <= MLS_MAX_N (=10).
__device__ inline void dSolveLinearN(double *A, double *b, double *sol, int n)
{
    for (int col = 0; col < n; ++col)
    {
        int piv = col;
        double mx = fabs(A[col * n + col]);
        for (int row = col + 1; row < n; ++row)
            if (fabs(A[row * n + col]) > mx)
            { mx = fabs(A[row * n + col]); piv = row; }
        if (piv != col)
        {
            for (int q = 0; q < n; ++q)
            { double t = A[col*n+q]; A[col*n+q] = A[piv*n+q]; A[piv*n+q] = t; }
            double tb = b[col]; b[col] = b[piv]; b[piv] = tb;
        }
        double d = A[col * n + col];
        if (fabs(d) < 1e-30) continue;
        for (int row = col + 1; row < n; ++row)
        {
            double f = A[row * n + col] / d;
            for (int q = col; q < n; ++q) A[row*n+q] -= f * A[col*n+q];
            b[row] -= f * b[col];
        }
    }
    for (int row = n - 1; row >= 0; --row)
    {
        double s = b[row];
        for (int q = row + 1; q < n; ++q) s -= A[row*n+q] * sol[q];
        double d = A[row * n + row];
        sol[row] = (fabs(d) > 1e-30) ? s / d : 0.0;
    }
}

__global__ void kMoments3D(DeviceCloud c, double R)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.N || c.boundary[i]) return;
    int Nv = c.Nv;
    double dv3 = c.dv * c.dv * c.dv;
    const double *g = c.g + (size_t)i * c.gStride;

    double rho = 0, rux = 0, ruy = 0, ruz = 0, E = 0;
    for (int kv = 0; kv < Nv; ++kv)
    {
        double vz = c.vlo + kv * c.dv;
        for (int jv = 0; jv < Nv; ++jv)
        {
            double vy = c.vlo + jv * c.dv;
            for (int iv = 0; iv < Nv; ++iv)
            {
                double vx = c.vlo + iv * c.dv;
                double f = g[iv + Nv * (jv + Nv * kv)];
                rho += f * dv3;
                rux += vx * f * dv3;
                ruy += vy * f * dv3;
                ruz += vz * f * dv3;
                E   += (vx*vx + vy*vy + vz*vz) * f * dv3;
            }
        }
    }
    if (rho > 1e-30)
    {
        c.rho[i] = rho;
        c.ux[i]  = rux / rho;
        c.uy[i]  = ruy / rho;
        c.uz[i]  = ruz / rho;
        double u2 = c.ux[i]*c.ux[i] + c.uy[i]*c.uy[i] + c.uz[i]*c.uz[i];
        double T  = (E / rho - u2) / (3.0 * R);
        c.T[i] = T > 1e-6 ? T : 1e-6;
    }
}

__global__ void kCollision3D(DeviceCloud c, double dt, double tao, double R)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.N || c.boundary[i]) return;
    int Nv = c.Nv;
    double lam = dt / tao, coeff = 1.0 / (1.0 + lam);
    double *g = c.g + (size_t)i * c.gStride;
    double rho = c.rho[i], ux = c.ux[i], uy = c.uy[i], uz = c.uz[i], T = c.T[i];

    for (int kv = 0; kv < Nv; ++kv)
    {
        double vz = c.vlo + kv * c.dv;
        for (int jv = 0; jv < Nv; ++jv)
        {
            double vy = c.vlo + jv * c.dv;
            for (int iv = 0; iv < Nv; ++iv)
            {
                double vx = c.vlo + iv * c.dv;
                int lin = iv + Nv * (jv + Nv * kv);
                double M = maxwellian3D(vx, vy, vz, ux, uy, uz, T, rho, R);
                g[lin] = coeff * (g[lin] + lam * M);
            }
        }
    }
}

// Node-parallel 3-D transport: one thread per (particle, velocity node)
// on the single-f Nv^3 grid (same occupancy rationale as kTransportNode).
__global__ void kTransportNode3D(DeviceCloud c, double dt, double h,
                                 int mlsOrder,
                                 int periodicX, int periodicY, int periodicZ,
                                 double Lx, double Ly, double Lz)
{
    int Nv = c.Nv;
    size_t Nv3 = (size_t)Nv * Nv * Nv;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)c.N * Nv3;
    if (tid >= total) return;
    int i   = (int)(tid / Nv3);
    int lin = (int)(tid % Nv3);
    int iv = lin % Nv, jv = (lin / Nv) % Nv, kv = lin / (Nv * Nv);
    double vx = c.vlo + iv * c.dv;
    double vy = c.vlo + jv * c.dv;
    double vz = c.vlo + kv * c.dv;

    if (c.boundary[i])
    {
        double cn = (vx - c.wUx[i]) * c.wnx[i]
                  + (vy - c.wUy[i]) * c.wny[i]
                  + (vz - c.wUz[i]) * c.wnz[i];
        if (cn >= 0.0) return;
    }
    int order_p = c.boundary[i] ? 1 : mlsOrder;
    int poly = basisSize(3, order_p);

    double xf = c.x[i] - (vx - c.mvx[i]) * dt;
    double yf = c.y[i] - (vy - c.mvy[i]) * dt;
    double zf = c.z[i] - (vz - c.mvz[i]) * dt;

    double A[MLS_MAX_N * MLS_MAX_N], b1[MLS_MAX_N], s1[MLS_MAX_N];
    double basis[MLS_MAX_N];
    for (int q = 0; q < poly * poly; ++q) A[q] = 0.0;
    for (int q = 0; q < poly; ++q) b1[q] = 0.0;
    bool has = false;

    int nn = c.nneigh[i];
    const int *nb = c.neigh + (size_t)i * c.maxNeigh;
    for (int m = 0; m < nn; ++m)
    {
        int j = nb[m];
        double dxn = c.x[j] - xf;
        double dyn = c.y[j] - yf;
        double dzn = c.z[j] - zf;
        if (periodicX) dxn = minImage(dxn, Lx);
        if (periodicY) dyn = minImage(dyn, Ly);
        if (periodicZ) dzn = minImage(dzn, Lz);
        double r2 = dxn*dxn + dyn*dyn + dzn*dzn;
        double w = mlsWeightSq(r2, h);
        if (w < 1e-30) continue;
        fillBasis(dxn, dyn, dzn, 3, h, basis, order_p);
        for (int p1 = 0; p1 < poly; ++p1)
            for (int p2 = 0; p2 < poly; ++p2)
                A[p1*poly + p2] += w * basis[p1] * basis[p2];
        double gn = c.g[(size_t)j * c.gStride + lin];
        for (int q = 0; q < poly; ++q)
            b1[q] += w * basis[q] * gn;
        has = true;
    }
    if (!has) return;
    dSolveLinearN(A, b1, s1, poly);
    c.gt[(size_t)i * c.gStride + lin] = s1[0];
}

// Templated node-parallel 3-D transport: POLY known at compile time so
// the per-thread arrays live in registers instead of spilled local
// memory (the runtime-sized version spent ~100x its arithmetic cost on
// spill traffic: 36.8 s/step vs ~0.4 s expected at 40^3/Nv=20). The MLS
// normal matrix A = B^T W B is symmetric positive-definite, so a
// pivot-free Cholesky replaces the pivoted Gaussian elimination:
// branch-free (no warp divergence) and ~3x fewer operations. Interior
// (POLY=10, quadratic) and boundary (POLY=4, linear) particles are
// dispatched as two launches so each instantiation is branch-light.
// MLS pivot-collapse threshold, relative to the largest assembled diagonal
// (env ALEBGK_MLS_TOL, default 1e-12). Tunable because the right value is an
// empirical question: 1e-12 catches TOTAL rank loss, which is what produced
// the NaN in the translation-only case, but it demonstrably does NOT catch
// whatever drives the rotating case -- that run is bit-for-bit unchanged with
// the fallback in place. Raising it tests whether MARGINAL ill-conditioning
// is the amplifier there, or whether that is a separate mechanism.
__device__ double dMlsTol = 1e-12;

// Clamp the MLS reconstruction to the range of the neighbour values that
// produced it (env ALEBGK_MLS_CLAMP=1). Rationale: for a body-surface
// particle the foot-point is displaced by the wall velocity,
// x_f = x - (v - mv)*dt, and can land INSIDE the body, which is empty of
// particles. The stencil is then entirely one-sided and the reconstruction
// EXTRAPOLATES. Extrapolation amplifies with gain > 1 while leaving the
// moment matrix perfectly well conditioned -- which is why raising the
// pivot threshold from 1e-12 to 1e-8 left the rotating run bit-identical.
// A discrete maximum principle catches exactly this and nothing else.
__device__ int dMlsClamp = 0;
// Self-test (ALEBGK_MLS_FORCEFLAG=1): raise the extrapolation flag on every
// reconstruction. If the reported count is then ~N, the flag path works and a
// zero reading is a real measurement; if it still reads zero, the
// instrumentation is broken. Cheap insurance -- an earlier version of this
// counter was silently wrong.
__device__ int dMlsForce = 0;

template <int POLY, bool BOUNDARY>
__global__ void kTransportNode3DT(DeviceCloud c, double dt, double h,
                                  int periodicX, int periodicY, int periodicZ,
                                  double Lx, double Ly, double Lz)
{
    int Nv = c.Nv;
    size_t Nv3 = (size_t)Nv * Nv * Nv;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)c.N * Nv3) return;
    int i   = (int)(tid / Nv3);
    if ((c.boundary[i] != 0) != BOUNDARY) return;
    int lin = (int)(tid % Nv3);
    int iv = lin % Nv, jv = (lin / Nv) % Nv, kv = lin / (Nv * Nv);
    double vx = c.vlo + iv * c.dv;
    double vy = c.vlo + jv * c.dv;
    double vz = c.vlo + kv * c.dv;

    if (BOUNDARY)
    {
        double cn = (vx - c.wUx[i]) * c.wnx[i]
                  + (vy - c.wUy[i]) * c.wny[i]
                  + (vz - c.wUz[i]) * c.wnz[i];
        if (cn >= 0.0) return;
    }

    double xf = c.x[i] - (vx - c.mvx[i]) * dt;
    double yf = c.y[i] - (vy - c.mvy[i]) * dt;
    double zf = c.z[i] - (vz - c.mvz[i]) * dt;

    double A[POLY * POLY];
    double b1[POLY], basis[POLY];
    #pragma unroll
    for (int q = 0; q < POLY * POLY; ++q) A[q] = 0.0;
    #pragma unroll
    for (int q = 0; q < POLY; ++q) b1[q] = 0.0;
    bool has = false;
    // Shepard (weighted-average) fallback, accumulated alongside the MLS
    // normal equations at negligible cost. Used when the moment matrix turns
    // out to be rank-deficient -- see the pivot test below.
    double sumW = 0.0, sumWG = 0.0;
    // Range of the contributing neighbour values, for the maximum-principle
    // test after the solve.
    double gmin = 1e300, gmax = -1e300;

    int nn = c.nneigh[i];
    const int *nb = c.neigh + (size_t)i * c.maxNeigh;
    for (int m = 0; m < nn; ++m)
    {
        int j = nb[m];
        double dxn = c.x[j] - xf;
        double dyn = c.y[j] - yf;
        double dzn = c.z[j] - zf;
        if (periodicX) dxn = minImage(dxn, Lx);
        if (periodicY) dyn = minImage(dyn, Ly);
        if (periodicZ) dzn = minImage(dzn, Lz);
        double r2 = dxn*dxn + dyn*dyn + dzn*dzn;
        double w = mlsWeightSq(r2, h);
        if (w < 1e-30) continue;
        fillBasis(dxn, dyn, dzn, 3, h, basis, POLY >= 10 ? 2 : 1);
        #pragma unroll
        for (int p1 = 0; p1 < POLY; ++p1)
        {
            double wb = w * basis[p1];
            #pragma unroll
            for (int p2 = p1; p2 < POLY; ++p2)      // symmetric: upper only
                A[p1*POLY + p2] += wb * basis[p2];
            b1[p1] += wb * c.g[(size_t)j * c.gStride + lin];
        }
        double gj = c.g[(size_t)j * c.gStride + lin];
        sumW  += w;
        sumWG += w * gj;
        gmin = fmin(gmin, gj);
        gmax = fmax(gmax, gj);
        has = true;
    }
    if (!has) return;

    // Largest assembled diagonal: the scale a pivot must be judged against.
    // An absolute test is meaningless here -- the entries carry the weight
    // and basis scaling, so "small" only has meaning relative to the matrix.
    double dmax = 0.0;
    #pragma unroll
    for (int k = 0; k < POLY; ++k) dmax = fmax(dmax, A[k*POLY + k]);
    bool degenerate = (dmax <= 0.0);

    // Cholesky A = L L^T on the upper triangle (mirrored), then solve.
    #pragma unroll
    for (int k = 0; k < POLY; ++k)
    {
        double d = A[k*POLY + k];
        #pragma unroll
        for (int j = 0; j < POLY; ++j)
            if (j < k) { double l = A[j*POLY + k]; d -= l * l; }
        // A collapsing pivot means the neighbour set does not span the basis.
        // For a body-surface particle whose stencil has been thinned by
        // manageRemove and flattened onto one face that is the normal case,
        // not an exotic one -- measured at the failure: 18 neighbours against
        // a cloud mean of 72, on the body surface.
        //
        // The old guard substituted 1e-150 and carried on. That is not a safe
        // fallback: the substitutions below divide by this pivot twice more,
        // so 1e-150 becomes ~1e+150 then ~1e+300, and one more step overflows
        // to Inf -- Inf - Inf is the NaN that reached computeBodyLoads.
        // Marginal (rather than total) collapse is the same failure in slow
        // motion: it amplifies smoothly, which is the geometric ramp seen in
        // the rotating case. Flag it and fall back instead of inventing a
        // pivot.
        if (!(d > dMlsTol * dmax)) degenerate = true;
        d = degenerate ? 1.0 : sqrt(d);
        A[k*POLY + k] = d;
        #pragma unroll
        for (int m2 = k + 1; m2 < POLY; ++m2)
        {
            double v = A[k*POLY + m2];
            #pragma unroll
            for (int j = 0; j < POLY; ++j)
                if (j < k) v -= A[j*POLY + k] * A[j*POLY + m2];
            A[k*POLY + m2] = v / d;
        }
    }
    // forward: L y = b (L^T stored in upper triangle: L[m][k] = A[k*P+m])
    #pragma unroll
    for (int k = 0; k < POLY; ++k)
    {
        double v = b1[k];
        #pragma unroll
        for (int j = 0; j < POLY; ++j)
            if (j < k) v -= A[j*POLY + k] * b1[j];
        b1[k] = v / A[k*POLY + k];
    }
    // backward: L^T x = y
    #pragma unroll
    for (int k = POLY - 1; k >= 0; --k)
    {
        double v = b1[k];
        #pragma unroll
        for (int j = 0; j < POLY; ++j)
            if (j > k) v -= A[k*POLY + j] * b1[j];
        b1[k] = v / A[k*POLY + k];
    }
    // Shepard fallback: the weight-normalised average of the neighbour
    // values. Only first-order accurate, but unconditionally well posed with
    // a single neighbour, and its result is bounded by the values it averages
    // -- so it can neither overflow nor manufacture a new extremum, which is
    // precisely what the degenerate MLS solve was doing.
    double out = b1[0];
    if (degenerate || !isfinite(out))
        out = (sumW > 0.0) ? (sumWG / sumW)
                           : c.g[(size_t)i * c.gStride + lin];
    if (c.mlsFlag && (degenerate || !isfinite(b1[0])))
        atomicOr(&c.mlsFlag[i], 1);      // MLS-degenerate

    // Discrete maximum principle. A reconstruction that lands outside the
    // range of the values it was built from is an extrapolation overshoot,
    // not an interpolation -- the one signature that distinguishes
    // "stencil is one-sided" from "stencil is ill-conditioned".
    // Excess is computed from the ACTUAL value, never from dMlsForce, so the
    // magnitude stays honest even when the self-test forces the flag on.
    double excess = fmax(fmax(out - gmax, gmin - out), 0.0);
    bool overshoot = (out < gmin) || (out > gmax) || dMlsForce;
    if (overshoot && c.mlsFlag)
        atomicOr(&c.mlsFlag[i], 2);      // MLS-extrapolation
    if (excess > 0.0 && c.mlsOvr)
    {
        double dv3 = c.dv * c.dv * c.dv;
        atomicAdd(&c.mlsOvr[i], excess * dv3);   // density units
    }
    if (overshoot && dMlsClamp)
        out = fmin(fmax(out, gmin), gmax);
    c.gt[(size_t)i * c.gStride + lin] = out;
}

// 3-D positivity fix over the whole single distribution
// Fall back a bad (negative) node to its own last-known-good value (the
// pre-transport c.g, still intact here -- kCommit hasn't run yet) instead
// of zero. Zeroing discards whatever momentum/energy that node carried
// with no correction anywhere else in the step; on an isothermal wall
// with no matching global energy correction (only mass gets one, via
// scaleG), that uncorrected perturbation is a plausible steady energy
// source that compounds every step into the slow thermal runaway seen
// without this fix. Falling back to g[lin] keeps a real, physically
// consistent value instead of a hole, so the mass-restoring rescale below
// disturbs momentum/energy far less than rescaling around a zero.
__global__ void kPositivity3D(DeviceCloud c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.N) return;
    double *gt = c.gt + (size_t)i * c.gStride;
    const double *g0 = c.g + (size_t)i * c.gStride;
    double mPos = 0, mNeg = 0, mFallback = 0;
    for (size_t k = 0; k < c.gStride; ++k)
    {
        double v = gt[k];
        if (v >= 0) mPos += v;
        else { mNeg += v; mFallback += g0[k]; }
    }
    if (mNeg < -1e-30 && (mPos + mNeg) > 0)
    {
        c.limFlag[i] += 1.0;                  // positivity repair fired here
        double target = mPos + mNeg;          // mass the reconstruction implied
        double replaced = mPos + mFallback;    // mass after the gentle fallback
        double sc = (replaced > 1e-30) ? target / replaced : 0.0;
        for (size_t k = 0; k < c.gStride; ++k)
        {
            double v = (gt[k] < 0.0) ? g0[k] : gt[k];
            gt[k] = v * sc;
        }
    }
}

// NAD (Numerical Admissibility Detection) clamp -- the GPU path's
// transport step previously had NO discrete-maximum-principle check at
// all (kPositivity3D only rescales away negative mass; it never catches a
// smooth, positive overshoot). That let a slow, compounding growth mode
// pass every step unflagged, needing tens of ns to become visible as a
// blow-up. Compare the freshly-transported density against the range
// bracketed by this particle's neighbours' CURRENT densities (c.rho,
// still holding last step's post-collision value at this point in the
// pipeline) and rescale gt back inside that range (with slack) if it
// overshot -- the same rescale kPositivity3D already applies for the
// undershoot side, extended to the overshoot side.
__global__ void kNAD3D(DeviceCloud c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.N || c.boundary[i]) return;
    double *gt = c.gt + (size_t)i * c.gStride;
    double dv3 = c.dv * c.dv * c.dv;
    double rho_new = 0.0;
    for (size_t k = 0; k < c.gStride; ++k) rho_new += gt[k];
    rho_new *= dv3;

    int nn = c.nneigh[i];
    const int *nb = c.neigh + (size_t)i * c.maxNeigh;
    double rmin = 1e300, rmax = -1e300;
    for (int k = 0; k < nn; ++k)
    {
        double rn = c.rho[nb[k]];
        rmin = min(rmin, rn);
        rmax = max(rmax, rn);
    }
    if (rmin > rmax) return;   // no valid neighbours
    double slack = 0.1 * max(rmax - rmin, 1e-12) + 1e-9;
    double lo = rmin - slack, hi = rmax + slack;
    if (rho_new > hi && rho_new > 1e-30)
    {
        c.limFlag[i] += 2.0;                  // NAD clamp fired (overshoot)
        double sc = hi / rho_new;
        for (size_t k = 0; k < c.gStride; ++k) gt[k] *= sc;
    }
    else if (rho_new < lo && rho_new > 1e-30)
    {
        c.limFlag[i] += 2.0;                  // NAD clamp fired (undershoot)
        double sc = lo / rho_new;
        for (size_t k = 0; k < c.gStride; ++k) gt[k] *= sc;
    }
}

__global__ void kWallBC3D(DeviceCloud c, double R)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.N || !c.boundary[i]) return;
    int Nv = c.Nv;
    double dv3 = c.dv * c.dv * c.dv;
    double *g = c.g + (size_t)i * c.gStride;
    double nx = c.wnx[i], ny = c.wny[i], nz = c.wnz[i];
    double Uwx = c.wUx[i], Uwy = c.wUy[i], Uwz = c.wUz[i], Tw = c.wT[i];
    double RT2w = 2.0 * R * Tw;
    double norm = pow(M_PI * RT2w, 1.5);

    double pxBefore = 0.0;
    for (int kv = 0; kv < Nv; ++kv)
        for (int jv = 0; jv < Nv; ++jv)
            for (int iv = 0; iv < Nv; ++iv)
                pxBefore += (c.vlo + iv * c.dv)
                          * g[iv + Nv * (jv + Nv * kv)] * dv3;

    double intM = 0, intP = 0;
    for (int kv = 0; kv < Nv; ++kv)
    {
        double vz = c.vlo + kv * c.dv;
        for (int jv = 0; jv < Nv; ++jv)
        {
            double vy = c.vlo + jv * c.dv;
            for (int iv = 0; iv < Nv; ++iv)
            {
                double vx = c.vlo + iv * c.dv;
                double cn = (vx-Uwx)*nx + (vy-Uwy)*ny + (vz-Uwz)*nz;
                int lin = iv + Nv * (jv + Nv * kv);
                if (cn < 0.0) intM += cn * g[lin] * dv3;
                else if (cn > 0.0)
                {
                    double Mw = exp(-((vx-Uwx)*(vx-Uwx) + (vy-Uwy)*(vy-Uwy)
                                    + (vz-Uwz)*(vz-Uwz)) / RT2w) / norm;
                    intP += cn * Mw * dv3;
                }
            }
        }
    }
    double rho_w = (intP > 1e-30) ? (-intM / intP) : 1.0;
    for (int kv = 0; kv < Nv; ++kv)
    {
        double vz = c.vlo + kv * c.dv;
        for (int jv = 0; jv < Nv; ++jv)
        {
            double vy = c.vlo + jv * c.dv;
            for (int iv = 0; iv < Nv; ++iv)
            {
                double vx = c.vlo + iv * c.dv;
                double cn = (vx-Uwx)*nx + (vy-Uwy)*ny + (vz-Uwz)*nz;
                if (cn > 0.0)
                {
                    int lin = iv + Nv * (jv + Nv * kv);
                    g[lin] = rho_w
                           * exp(-((vx-Uwx)*(vx-Uwx) + (vy-Uwy)*(vy-Uwy)
                                 + (vz-Uwz)*(vz-Uwz)) / RT2w) / norm;
                }
            }
        }
    }

    if (c.bcdPx)
    {
        double pxAfter = 0.0;
        for (int kv = 0; kv < Nv; ++kv)
            for (int jv = 0; jv < Nv; ++jv)
                for (int iv = 0; iv < Nv; ++iv)
                    pxAfter += (c.vlo + iv * c.dv)
                             * g[iv + Nv * (jv + Nv * kv)] * dv3;
        c.bcdPx[i] = pxAfter - pxBefore;
    }
    c.rho[i] = rho_w; c.ux[i] = Uwx; c.uy[i] = Uwy; c.uz[i] = Uwz; c.T[i] = Tw;
}




// ── GPU info + memory budget table (printed before the time loop) ───────────
constexpr int LEFT_W  = 46;
constexpr int RIGHT_W = 18;

// Plain-ASCII table borders: UTF-8 box-drawing characters are mangled by
// the default Windows console codepage (CP437/CP850), so keep to 7-bit.
template<typename T>
static void printRow(const std::string& label, const T& value)
{
    std::cout << "| "
              << std::left  << std::setw(LEFT_W) << label
              << "| "
              << std::left  << std::setw(RIGHT_W) << value
              << "|\n";
}

static void printGPUMemoryTable(int N, int Nv, int dim, size_t gStride)
{
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);

    // CUDA cores: SM count x cores-per-SM
    int coresPerSM = 0;
    int major = prop.major, minor = prop.minor;
    if      (major==8 && minor==0) coresPerSM=64;    // A100 (Ampere sm_80)
    else if (major==8)             coresPerSM=128;   // other Ampere / Ada 8.9
    else if (major==9)             coresPerSM=128;   // Hopper
    else if (major==7 && minor==5) coresPerSM=64;    // Turing
    else if (major==7)             coresPerSM=64;    // Volta
    else if (major==6)             coresPerSM=128;   // Pascal
    else                           coresPerSM=128;   // fallback
    int cudaCores = prop.multiProcessorCount * coresPerSM;
    int memClockKHz = 0;
    int deviceId = 0;

#if CUDART_VERSION >= 13000
	cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, deviceId);
#else
    memClockKHz = prop.memoryClockRate;
#endif
    double bwGB = (double)memClockKHz * 2.0
                * (prop.memoryBusWidth / 8.0) / 1e6;

    size_t freeMem=0, totalMem=0;
    cudaMemGetInfo(&freeMem, &totalMem);

    // ── Per-allocation sizes (DeviceCloud SoA layout) ────────────────────
    size_t szG      = (size_t)N * gStride * sizeof(double);       // g
    size_t szGt     = szG;                                        // gt
    size_t szFields = (size_t)N * (13 * sizeof(double) + 1);      // x..wT, flags
    size_t szNeigh  = (size_t)N * (MAX_NEIGH * sizeof(int) + sizeof(int));
    size_t szUsed   = szG + szGt + szFields + szNeigh + sizeof(double);
    size_t szFreeAfter = freeMem > szUsed ? freeMem - szUsed : 0;

    auto GB = [](size_t b){ return (double)b/1024/1024/1024; };

    std::cout << "\n";
    std::cout << "+=================================================================+\n";
    std::cout << "|                 GPU Architecture & Memory                       |\n";
    std::cout << "+=================================================================+\n";
    int coreClockKHz = 0;
#if CUDART_VERSION >= 13000
    cudaDeviceGetAttribute(&coreClockKHz, cudaDevAttrClockRate, deviceId);
#else
    coreClockKHz = prop.clockRate;
#endif
    printRow("Device", prop.name);
    { std::ostringstream ss; ss << major << "." << minor;
      printRow("Compute capability", ss.str()); }
    printRow("Streaming multiprocessors (SMs)", prop.multiProcessorCount);
    { std::ostringstream ss; ss << cudaCores << " (" << coresPerSM << "/SM)";
      printRow("CUDA cores", ss.str()); }
    { std::ostringstream ss; ss << std::fixed << std::setprecision(0)
         << coreClockKHz / 1e3 << " MHz";
      printRow("GPU clock rate", ss.str()); }
    { std::ostringstream ss; ss << std::fixed << std::setprecision(0)
         << memClockKHz / 1e3 << " MHz";
      printRow("Memory clock rate", ss.str()); }
    { std::ostringstream ss; ss << prop.memoryBusWidth << " bit";
      printRow("Memory bus width", ss.str()); }
    { std::ostringstream ss; ss << std::fixed << std::setprecision(1)
         << bwGB << " GB/s";
      printRow("Peak memory bandwidth", ss.str()); }
    { std::ostringstream ss; ss << std::fixed << std::setprecision(1)
         << GB(totalMem) << " GB";
      printRow("Total GPU memory", ss.str()); }
    { std::ostringstream ss; ss << std::fixed << std::setprecision(1)
         << GB(freeMem) << " GB";
      printRow("Free GPU memory (before allocations)", ss.str()); }

    std::cout << "+=================================================================+\n";
    std::cout << "|                  Memory Budget (DeviceCloud)                    |\n";
    std::cout << "+=================================================================+\n";

    printRow("N particles", N);
    { std::ostringstream ss;
      ss << Nv << (dim==3 ? "^3 (single f)" : "^2 x 2 (Chu g1/g2)");
      printRow("Velocity nodes / particle", ss.str()); }
    { std::ostringstream ss;
      ss << gStride * sizeof(double) << " bytes";
      printRow("Distribution bytes / particle", ss.str()); }
    { std::ostringstream ss; ss << std::fixed << std::setprecision(2)
         << GB(szG) << " GB";
      printRow("g  (distributions)", ss.str()); }
    { std::ostringstream ss; ss << std::fixed << std::setprecision(2)
         << GB(szGt) << " GB";
      printRow("gt (transport scratch)", ss.str()); }
    { std::ostringstream ss; ss << std::fixed << std::setprecision(2)
         << GB(szFields + szNeigh) << " GB";
      printRow("fields + neighbour lists", ss.str()); }
    { std::ostringstream ss; ss << std::fixed << std::setprecision(2)
         << GB(szUsed) << " GB";
      printRow("Total to allocate", ss.str()); }
    { std::ostringstream ss; ss << std::fixed << std::setprecision(2)
         << GB(szFreeAfter) << " GB";
      printRow("Free after allocations", ss.str()); }
    { std::ostringstream ss; ss << std::fixed << std::setprecision(1)
         << 100.0 * (double)szUsed / (double)totalMem << " %";
      printRow("Fraction of GPU used", ss.str()); }

    std::cout << "+=================================================================+\n\n";
}


// Node-parallel collision: one thread per (particle, velocity node) —
// the per-particle loop over all nodes serialised ~2000 exp() calls per
// thread.
__global__ void kCollisionNode(DeviceCloud c, double dt, double tao, double R)
{
    int Nv = c.Nv, Nv2 = Nv * Nv;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)c.N * Nv2) return;
    int i = (int)(tid / Nv2), lin = (int)(tid % Nv2);
    if (c.boundary[i]) return;
    int iv = lin % Nv, jv = lin / Nv;
    double vx = c.vlo + iv * c.dv, vy = c.vlo + jv * c.dv;
    double lam = dt / tao, coeff = 1.0 / (1.0 + lam);
    double *g = c.g + (size_t)i * c.gStride;
    double Mg1 = maxwellianG1(vx, vy, c.ux[i], c.uy[i], c.T[i], c.rho[i], R);
    g[lin]       = coeff * (g[lin]       + lam * Mg1);
    g[lin + Nv2] = coeff * (g[lin + Nv2] + lam * 0.5 * R * c.T[i] * Mg1);
}

__global__ void kCollisionNode3D(DeviceCloud c, double dt, double tao, double R)
{
    int Nv = c.Nv;
    size_t Nv3 = (size_t)Nv * Nv * Nv;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (size_t)c.N * Nv3) return;
    int i = (int)(tid / Nv3), lin = (int)(tid % Nv3);
    if (c.boundary[i]) return;
    int iv = lin % Nv, jv = (lin / Nv) % Nv, kv = lin / (Nv * Nv);
    double vx = c.vlo + iv * c.dv, vy = c.vlo + jv * c.dv,
           vz = c.vlo + kv * c.dv;
    double lam = dt / tao, coeff = 1.0 / (1.0 + lam);
    double *g = c.g + (size_t)i * c.gStride;
    double M = maxwellian3D(vx, vy, vz, c.ux[i], c.uy[i], c.uz[i],
                            c.T[i], c.rho[i], R);
    g[lin] = coeff * (g[lin] + lam * M);
}

// ── Host-side driver API (called from Solver.cpp under ALEBGK_WITH_CUDA) ────
// Free every device buffer and null the pointers so a subsequent
// cudaUploadCloud reallocates cleanly (used when the moving-body ALE
// remove/refill changes the particle count).
void cudaFreeCloud()
{
    double *dptrs[] = {dc.x, dc.y, dc.z, dc.g, dc.gt, dc.rho, dc.ux, dc.uy,
                       dc.uz, dc.T, dc.wnx, dc.wny, dc.wnz, dc.wUx, dc.wUy,
                       dc.wUz, dc.wT, dc.mvx, dc.mvy, dc.mvz, dc.scratch,
                       dc.limFlag, dc.bcdPx};
    for (double *p : dptrs) if (p) cudaFree(p);
    if (dc.boundary) cudaFree(dc.boundary);
    if (dc.neigh)    cudaFree(dc.neigh);
    if (dc.nneigh)   cudaFree(dc.nneigh);
    if (dc.mlsFlag)  cudaFree(dc.mlsFlag);
    if (dc.mlsOvr)   cudaFree(dc.mlsOvr);
    dc.x = dc.y = dc.z = dc.g = dc.gt = dc.rho = dc.ux = dc.uy = dc.uz =
        dc.T = dc.wnx = dc.wny = dc.wnz = dc.wUx = dc.wUy = dc.wUz = dc.wT =
        dc.mvx = dc.mvy = dc.mvz = dc.scratch = dc.limFlag = dc.bcdPx = nullptr;
    dc.boundary = nullptr; dc.neigh = nullptr; dc.nneigh = nullptr;
    dc.mlsFlag = nullptr; dc.mlsOvr = nullptr;
    dc.N = 0; dc.gStride = 0;
}

void cudaUploadCloud(const std::vector<Particle> &P, const SimParameters &sp)
{
    static bool tolSet = false;
    if (!tolSet)
    {
        tolSet = true;
        if (const char *e = std::getenv("ALEBGK_MLS_FORCEFLAG"))
            if (e[0] && std::atoi(e))
            {
                int one = 1;
                CUCHECK(cudaMemcpyToSymbol(dMlsForce, &one, sizeof(int)));
                printf("[MLS] FORCEFLAG self-test enabled\n");
            }
        if (const char *e = std::getenv("ALEBGK_MLS_CLAMP"))
            if (e[0] && std::atoi(e))
            {
                int one = 1;
                CUCHECK(cudaMemcpyToSymbol(dMlsClamp, &one, sizeof(int)));
                printf("[MLS] extrapolation clamp ENABLED\n");
            }
        if (const char *e = std::getenv("ALEBGK_MLS_TOL"))
            if (e[0])
            {
                double v = std::atof(e);
                if (v > 0.0)
                {
                    CUCHECK(cudaMemcpyToSymbol(dMlsTol, &v, sizeof(double)));
                    printf("[MLS] degeneracy threshold = %.3e (relative)\n", v);
                }
            }
    }
    int N = (int)P.size();
    int    newDim    = N > 0 ? P[0].dim : 2;
    size_t newStride = (newDim == 3)
                     ? (size_t)sp.Nv * sp.Nv * sp.Nv
                     : (size_t)sp.Nv * sp.Nv * 2;
    // Re-upload path (moving-body ALE): if the layout is unchanged we keep
    // the existing device buffers and just re-copy the host state — no
    // per-step malloc/free churn.  Reallocate only when N or gStride change.
    bool reuse = (dc.g != nullptr && dc.N == N && dc.gStride == newStride
                  && dc.dim == newDim);

    dc.N = N; dc.Nv = sp.Nv; dc.layers = 2;
    dc.dim = newDim;
    // Per-particle distribution length: Nv^3 (3-D single f) or 2*Nv^2 (2-D Chu)
    dc.gStride = newStride;
    dc.dv = (sp.VMax - sp.VMin) / (sp.Nv - 1);
    dc.vlo = sp.VMin;
    size_t gsz = (size_t)N * dc.gStride;

    if (!reuse)
    {
        if (dc.g != nullptr) cudaFreeCloud();
        dc.N = N; dc.Nv = sp.Nv; dc.dim = newDim; dc.gStride = newStride;
        // GPU info + memory budget: full table once at start-up; later
        // reallocations (moving-body remove/refill changing N) get one line.
        static bool tablePrinted = false;
        if (!tablePrinted)
        {
            printGPUMemoryTable(N, dc.Nv, dc.dim, dc.gStride);
            tablePrinted = true;
        }
        else
            printf("[GPU] cloud realloc: N=%d\n", N);
        auto dalloc = [](auto **p, size_t n) {
            CUCHECK(cudaMalloc((void **)p, n));
        };
        dalloc(&dc.x, N * 8); dalloc(&dc.y, N * 8); dalloc(&dc.z, N * 8);
        dalloc(&dc.g, gsz * 8); dalloc(&dc.gt, gsz * 8);
        dalloc(&dc.rho, N * 8); dalloc(&dc.ux, N * 8); dalloc(&dc.uy, N * 8);
        dalloc(&dc.uz, N * 8); dalloc(&dc.T, N * 8);
        dalloc(&dc.boundary, N);
        dalloc(&dc.wnx, N * 8); dalloc(&dc.wny, N * 8); dalloc(&dc.wnz, N * 8);
        dalloc(&dc.wUx, N * 8); dalloc(&dc.wUy, N * 8); dalloc(&dc.wUz, N * 8);
        dalloc(&dc.wT, N * 8);
        dalloc(&dc.mvx, N * 8); dalloc(&dc.mvy, N * 8); dalloc(&dc.mvz, N * 8);
        dalloc(&dc.neigh, (size_t)N * MAX_NEIGH * 4); dalloc(&dc.nneigh, N * 4);
        dalloc(&dc.scratch, ((size_t)(N + 255) / 128 + 2) * 8);
        dalloc(&dc.limFlag, N * 8);
        dalloc(&dc.mlsFlag, N * 4);
        dalloc(&dc.mlsOvr, N * 8);
        dalloc(&dc.bcdPx, N * 8);
    }

    std::vector<double> h(N);
    std::vector<unsigned char> hb(N);
    std::vector<int> hn((size_t)N * MAX_NEIGH, 0), hnn(N);
    auto up = [&](double *dst, auto get) {
        for (int i = 0; i < N; ++i) h[i] = get(P[i]);
        CUCHECK(cudaMemcpy(dst, h.data(), N * 8, cudaMemcpyHostToDevice));
    };
    up(dc.x, [](const Particle &p) { return p.x; });
    up(dc.y, [](const Particle &p) { return p.y; });
    up(dc.z, [](const Particle &p) { return p.z; });
    // dc.rho is otherwise only ever WRITTEN by kMoments3D and DOWNLOADED
    // later -- never uploaded from the host. Before kMoments3D has run even
    // once (the very first step, or right after any moving-body re-upload
    // that reallocated the buffer), it holds raw cudaMalloc garbage. Any
    // kernel reading a neighbour's dc.rho before the first moments pass
    // (kNAD3D does, for its discrete-maximum-principle reference) needs a
    // real seed value here, not leftover device memory.
    up(dc.rho, [](const Particle &p) { return p.rho; });
    // ux/uy/uz/T have exactly the same problem rho does, and were missed when
    // rho was fixed: they are written only by kMoments3D (interior) and
    // kWallBC3D (boundary), never uploaded. After a moving-body reallocation
    // the buffers are freshly cudaMalloc'd, so any read before the next
    // kMoments3D pass returns garbage -- in practice zeros, which is how a
    // save step landing right after a realloc reported T=0 and |u|=0 for the
    // whole cloud while rho looked perfectly healthy.
    up(dc.ux, [](const Particle &p) { return p.ux; });
    up(dc.uy, [](const Particle &p) { return p.uy; });
    up(dc.uz, [](const Particle &p) { return p.uz; });
    up(dc.T,  [](const Particle &p) { return p.T; });
    up(dc.wnx, [](const Particle &p) { return p.wnx; });
    up(dc.wny, [](const Particle &p) { return p.wny; });
    up(dc.wnz, [](const Particle &p) { return p.wnz; });
    up(dc.wUx, [](const Particle &p) { return p.wUx; });
    up(dc.wUy, [](const Particle &p) { return p.wUy; });
    up(dc.wUz, [](const Particle &p) { return p.wUz; });
    up(dc.wT, [](const Particle &p) { return p.wT; });
    up(dc.mvx, [](const Particle &p) { return p.meshVx; });
    up(dc.mvy, [](const Particle &p) { return p.meshVy; });
    up(dc.mvz, [](const Particle &p) { return p.meshVz; });
    for (int i = 0; i < N; ++i) hb[i] = P[i].boundary ? 1 : 0;
    CUCHECK(cudaMemcpy(dc.boundary, hb.data(), N, cudaMemcpyHostToDevice));
    for (int i = 0; i < N; ++i)
    {
        hnn[i] = (int)P[i].neighindex.size();
        for (int m = 0; m < hnn[i]; ++m)
            hn[(size_t)i * MAX_NEIGH + m] = P[i].neighindex[m];
    }
    CUCHECK(cudaMemcpy(dc.neigh, hn.data(), hn.size() * 4,
                       cudaMemcpyHostToDevice));
    CUCHECK(cudaMemcpy(dc.nneigh, hnn.data(), N * 4,
                       cudaMemcpyHostToDevice));

    // Gather the per-particle vectors into one contiguous staging buffer.
    // Same story as the scatter in cudaDownloadFull: 29.7 million doubles one
    // element at a time on one thread dominated the H2D copy itself. Note the
    // per-particle length is P[i].g.size(), which can be SHORTER than gStride
    // for a freshly refilled particle, so the tail must stay zeroed -- the
    // vector<double> value-initialises it and memcpy only writes what exists.
    // Static for the same reason as the download buffer. Zero-fill the region
    // in use rather than relying on a fresh allocation: a particle whose g is
    // SHORTER than gStride (freshly refilled) must leave a zeroed tail, and a
    // reused buffer still holds the previous step's values there.
    static std::vector<double> hg;
    if (hg.size() < gsz) hg.resize(gsz);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
    {
        double *dst = hg.data() + (size_t)i * dc.gStride;
        size_t have = P[i].g.size();
        std::memcpy(dst, P[i].g.data(), have * sizeof(double));
        if (have < dc.gStride)
            std::memset(dst + have, 0, (dc.gStride - have) * sizeof(double));
    }
    CUCHECK(cudaMemcpy(dc.g, hg.data(), gsz * 8, cudaMemcpyHostToDevice));
}

// Atomic-free mass sum: per-block partials + tiny host readback.
static double sumMassHost(int nb, int bs, bool d3)
{
    if (d3) kSumMass3D<<<nb, bs>>>(dc, dc.scratch);
    else    kSumMass  <<<nb, bs>>>(dc, dc.scratch);
    static std::vector<double> h;
    h.resize(nb);
    CUCHECK(cudaMemcpy(h.data(), dc.scratch, (size_t)nb * 8,
                       cudaMemcpyDeviceToHost));
    double m = 0;
    for (int k = 0; k < nb; ++k) m += h[k];
    return m;
}

// kSumMomX3D -- total x-momentum, sum over EVERY particle and velocity node
// of vx * f * dv^3. In a closed box driven only by tangential wall motion,
// the walls can inject x-momentum (that is what the lid is for) but no
// interior stage may: transport, the limiters, the commit and the global
// mass rescale must all be momentum-neutral. Bracketing the stages with this
// says which one is not, which the plane-flux diagnostic could only infer.
// ---------------------------------------------------------------------------
// Non-finite scan of g (ALEBGK_NAN_SCAN=1).
//
// The 3-D moving-body case dies with computeBodyLoads returning NaN. That
// routine has no division -- it is a pure weighted sum of p.g -- so the NaN
// must already be in g on a body-surface particle. Every macroscopic
// diagnostic stays healthy through it (rho, T, KE fine; positivity and NAD
// both report ZERO limiter activations), because one bad velocity node in one
// particle out of ~29,730 moves no average. So the only way to find it is to
// look at g directly, after each stage, and name the first entry that goes
// non-finite.
//
// Reports the LOWEST offending particle index (atomicMin) plus that entry's
// velocity node, so the answer is deterministic across runs rather than
// whichever warp happened to win.
__global__ void kScanNonFinite(DeviceCloud c, int *out, int useGt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.N) return;
    const double *g = (useGt ? c.gt : c.g) + (size_t)i * c.gStride;
    for (int k = 0; k < (int)c.gStride; ++k)
        if (!isfinite(g[k]))
        {
            int old = atomicMin(&out[0], i);
            if (i <= old) { out[1] = k; }
            return;
        }
}

// Returns the offending particle index, or -1 if g is clean. Sets *node to
// the flat velocity index within that particle.
static int scanNonFinite(int nb, int bs, int useGt, int *node)
{
    static int *dOut = nullptr;
    if (!dOut) CUCHECK(cudaMalloc((void **)&dOut, 2 * sizeof(int)));
    int init[2] = {INT_MAX, -1};
    CUCHECK(cudaMemcpy(dOut, init, 2 * sizeof(int), cudaMemcpyHostToDevice));
    kScanNonFinite<<<nb, bs>>>(dc, dOut, useGt);
    CUCHECK(cudaGetLastError());
    int h[2];
    CUCHECK(cudaMemcpy(h, dOut, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    *node = h[1];
    return (h[0] == INT_MAX) ? -1 : h[0];
}

__global__ void kSumMomX3D(DeviceCloud c, double *total, int useGt,
                           int interiorOnly)
{
    __shared__ double sh[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double m = 0;
    if (i < c.N && !(interiorOnly && c.boundary[i]))
    {
        double dv3 = c.dv * c.dv * c.dv;
        const double *g = (useGt ? c.gt : c.g) + (size_t)i * c.gStride;
        int Nv = c.Nv;
        for (int kv = 0; kv < Nv; ++kv)
            for (int jv = 0; jv < Nv; ++jv)
                for (int iv = 0; iv < Nv; ++iv)
                {
                    double vx = c.vlo + iv * c.dv;
                    m += vx * g[iv + Nv * (jv + Nv * kv)] * dv3;
                }
    }
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int sft = blockDim.x / 2; sft > 0; sft >>= 1)
    {
        if (threadIdx.x < sft) sh[threadIdx.x] += sh[threadIdx.x + sft];
        __syncthreads();
    }
    if (threadIdx.x == 0) total[blockIdx.x] = sh[0];
}

// Diagnostic only -- see kSumEnergy3D.
static double sumEnergyHost(int nb, int bs, bool useGt = false)
{
    kSumEnergy3D<<<nb, bs>>>(dc, dc.scratch, useGt ? 1 : 0);
    static std::vector<double> h;
    h.resize(nb);
    CUCHECK(cudaMemcpy(h.data(), dc.scratch, (size_t)nb * 8,
                       cudaMemcpyDeviceToHost));
    double e = 0;
    for (int k = 0; k < nb; ++k) e += h[k];
    return e;
}

// Diagnostic only -- see kSumMomX3D.
static double sumMomXHost(int nb, int bs, bool useGt = false,
                          bool interiorOnly = false)
{
    kSumMomX3D<<<nb, bs>>>(dc, dc.scratch, useGt ? 1 : 0,
                           interiorOnly ? 1 : 0);
    static std::vector<double> h;
    h.resize(nb);
    CUCHECK(cudaMemcpy(h.data(), dc.scratch, (size_t)nb * 8,
                       cudaMemcpyDeviceToHost));
    double m = 0;
    for (int k = 0; k < nb; ++k) m += h[k];
    return m;
}

void cudaStep(const SimParameters &sp, const CalcParameters &cp, double dt,
              double R)
{
    int bs = 128, nb = (dc.N + bs - 1) / bs;
    double massOld, massNew;
    bool d3 = (dc.dim == 3);

    // Limiter activity is per-step, so clear it before the transport stage
    // that writes it.
    if (dc.limFlag) CUCHECK(cudaMemset(dc.limFlag, 0, (size_t)dc.N * 8));
    if (dc.mlsFlag) CUCHECK(cudaMemset(dc.mlsFlag, 0, (size_t)dc.N * 4));
    if (dc.mlsOvr)  CUCHECK(cudaMemset(dc.mlsOvr, 0, (size_t)dc.N * 8));

    // One-shot per-kernel timing of the FIRST step, always printed.
    // The real launches are bracketed with events (kernels in the default
    // stream serialise, so group timings are exact) — no kernel is re-run,
    // so the physics is untouched and validation runs stay bit-identical.
    // ALEBGK_PROFILE_FROM=<step>: profile a WINDOW of steady-state steps and
    // report the mean, instead of only the first call.
    //
    // The one-shot first-call timing below is unusable for optimisation work:
    // it is dominated by JIT compilation and cold caches, so two runs of the
    // same kernel differ by 2x for no physical reason. Measuring a sqrt
    // removal against it showed "2x faster" when the true steady-state gain
    // was 2%. Anything that changes the kernel must be judged on a window
    // taken after the ramp, averaged over enough steps to be stable.
    static int pfFrom = -1, pfLen = 0;
    if (pfFrom < 0) {
        const char *e = std::getenv("ALEBGK_PROFILE_FROM");
        pfFrom = (e && e[0]) ? std::atoi(e) : -1;
        const char *l = std::getenv("ALEBGK_PROFILE_LEN");
        pfLen = (l && l[0]) ? std::atoi(l) : 20;
    }
    static bool profiled = false;
    static long pfStep = 0;
    static double pfAcc[24] = {0};
    static int pfN = 0, pfSlot = 0;
    ++pfStep;
    bool window = (pfFrom > 0 && pfStep >= pfFrom && pfStep < pfFrom + pfLen);
    bool prof = (!profiled && pfFrom <= 0) || window;
    profiled = true;
    if (window) { pfSlot = 0; ++pfN; }
    cudaEvent_t pe0 = nullptr, pe1 = nullptr;
    if (prof) { cudaEventCreate(&pe0); cudaEventCreate(&pe1); }
    auto pBeg = [&] { if (prof) cudaEventRecord(pe0); };
    // ALEBGK_NAN_SCAN=1: check g (and the transport scratch gt) for
    // non-finite entries after EVERY stage, every step. computeBodyLoads
    // dies with NaN loads while every macroscopic diagnostic stays clean and
    // both limiters report zero activations, so the corruption is one
    // velocity node in one particle -- invisible to any average. Riding on
    // pEnd covers all sixteen stage boundaries from one place.
    static int nsOn = -1;
    if (nsOn < 0) {
        const char *e = std::getenv("ALEBGK_NAN_SCAN");
        nsOn = (e && e[0] && std::atoi(e)) ? 1 : 0;
    }
    static long nsStep = 0;
    static bool nsHit = false;
    static const char *pfName[24] = {0};
    auto pEnd = [&](const char *nm) {
        if (nsOn && !nsHit)
        {
            int node = -1;
            int bad = scanNonFinite(nb, bs, 0, &node);
            const char *buf = "g";
            if (bad < 0) { bad = scanNonFinite(nb, bs, 1, &node); buf = "gt"; }
            if (bad >= 0)
            {
                nsHit = true;
                // Pull that particle's identity back so the report says WHERE
                // on the body it sits, not just an index.
                double hx = 0, hy = 0, hz = 0;
                unsigned char hb = 0;
                int hnn = 0;
                cudaMemcpy(&hx, dc.x + bad, 8, cudaMemcpyDeviceToHost);
                cudaMemcpy(&hy, dc.y + bad, 8, cudaMemcpyDeviceToHost);
                cudaMemcpy(&hz, dc.z + bad, 8, cudaMemcpyDeviceToHost);
                cudaMemcpy(&hb, dc.boundary + bad, 1, cudaMemcpyDeviceToHost);
                cudaMemcpy(&hnn, dc.nneigh + bad, 4, cudaMemcpyDeviceToHost);
                int Nv = dc.Nv;
                int iv = node % Nv, jv = (node / Nv) % Nv, kv = node / (Nv * Nv);
                printf("[NANSCAN] step %ld  FIRST non-finite in %s "
                       "after stage '%s'\n"
                       "          particle %d of %d  boundary=%d  neigh=%d\n"
                       "          pos (%.2f, %.2f, %.2f) nm\n"
                       "          velocity node %d = (iv,jv,kv)=(%d,%d,%d)  "
                       "c=(%.1f, %.1f, %.1f) m/s\n",
                       nsStep, buf, nm, bad, dc.N, (int)hb, hnn,
                       hx * 1e9, hy * 1e9, hz * 1e9, node, iv, jv, kv,
                       dc.vlo + iv * dc.dv, dc.vlo + jv * dc.dv,
                       dc.vlo + kv * dc.dv);
                fflush(stdout);
            }
        }
        if (!prof) return;
        cudaEventRecord(pe1);
        cudaEventSynchronize(pe1);
        float ms = 0;
        cudaEventElapsedTime(&ms, pe0, pe1);
        if (window) {
            if (pfSlot < 24) { pfAcc[pfSlot] += ms; pfName[pfSlot] = nm; ++pfSlot; }
            // Last stage of the last step in the window: emit the mean.
            if (pfN >= pfLen && pfSlot > 0 && std::strcmp(nm, "moments(post)") == 0) {
                double tot = 0.0;
                for (int q = 0; q < pfSlot; ++q) tot += pfAcc[q] / pfN;
                printf("[GPU-STEADY] mean over %d steps (steps %d-%d)\n",
                       pfN, pfFrom, pfFrom + pfLen - 1);
                for (int q = 0; q < pfSlot; ++q)
                    printf("[GPU-STEADY]   %-16s %10.3f ms  %5.1f%%\n",
                           pfName[q], pfAcc[q] / pfN, 100.0 * (pfAcc[q]/pfN) / tot);
                printf("[GPU-STEADY]   %-16s %10.3f ms\n", "TOTAL", tot);
                fflush(stdout);
                pfN = 0;
                for (int q = 0; q < 24; ++q) pfAcc[q] = 0.0;
            }
            return;
        }
        printf("[GPU-PROFILE] %-16s %10.2f ms\n", nm, ms);
    };
    if (nsOn) ++nsStep;

    pBeg();
    massOld = sumMassHost(nb, bs, d3);
    pEnd("sumMass");

    static int ebSteps0 = -1;
    if (ebSteps0 < 0)
    {
        const char *e = std::getenv("ALEBGK_ENERGY_BUDGET");
        ebSteps0 = (e && e[0]) ? std::atoi(e) : 0;
    }
    static int ebCount0 = 0;
    bool eb0 = d3 && ebSteps0 > 0 && ebCount0 < ebSteps0;
    double Estart = 0, EpostTransport = 0;
    if (eb0) Estart = sumEnergyHost(nb, bs, false);

    // ALEBGK_MOMENTUM_BUDGET=N: bracket total x-momentum around each stage
    // for the first N steps. Same idea as the energy budget above.
    static int mbSteps = -1;
    if (mbSteps < 0)
    {
        const char *e = std::getenv("ALEBGK_MOMENTUM_BUDGET");
        mbSteps = (e && e[0]) ? std::atoi(e) : 0;
    }
    static int mbCount = 0;
    bool mb = d3 && mbSteps > 0 && mbCount < mbSteps;
    double M0 = 0, M1 = 0, M2 = 0, M3 = 0, M4 = 0, M5 = 0, M6 = 0;
    double I0 = 0, I1 = 0, I2 = 0, I3 = 0, I4 = 0, I5 = 0, I6 = 0;
    if (mb) M0 = sumMomXHost(nb, bs, false); I0 = sumMomXHost(nb, bs, false, true);

    {
        // gt pre-fill (coalesced), then one thread per (particle, node)
        size_t totalNodes = (size_t)dc.N * dc.gStride;
        int nbCopy = (int)std::min<size_t>((totalNodes + bs - 1) / bs, 65535);
        pBeg();
        kCopyGt<<<nbCopy, bs>>>(dc);
        pEnd("copyGt");
        if (d3)
        {
            size_t tn = (size_t)dc.N * dc.Nv * dc.Nv * dc.Nv;
            int nbT = (int)((tn + bs - 1) / bs);
            pBeg();
            if (sp.mlsOrder >= 2)
                kTransportNode3DT<10,false><<<nbT, bs>>>(dc, dt, cp.radius,
                                 cp.periodicX ? 1 : 0, cp.periodicY ? 1 : 0,
                                 cp.periodicZ ? 1 : 0, cp.Lx, cp.Ly, cp.Lz);
            else
                kTransportNode3DT<4,false><<<nbT, bs>>>(dc, dt, cp.radius,
                                 cp.periodicX ? 1 : 0, cp.periodicY ? 1 : 0,
                                 cp.periodicZ ? 1 : 0, cp.Lx, cp.Ly, cp.Lz);
            kTransportNode3DT<4,true><<<nbT, bs>>>(dc, dt, cp.radius,
                                 cp.periodicX ? 1 : 0, cp.periodicY ? 1 : 0,
                                 cp.periodicZ ? 1 : 0, cp.Lx, cp.Ly, cp.Lz);
            pEnd("transport3D");
            pBeg();
            kPositivity3D<<<nb, bs>>>(dc);
            pEnd("positivity");
            pBeg();
            if (mb) M1 = sumMomXHost(nb, bs, true); I1 = sumMomXHost(nb, bs, true, true);   // gt, post-transport
            kNAD3D<<<nb, bs>>>(dc);
            pEnd("nad");
            if (mb) M2 = sumMomXHost(nb, bs, true); I2 = sumMomXHost(nb, bs, true, true);   // gt, post-limiters
        }
        else
        {
            size_t tn = (size_t)dc.N * dc.Nv * dc.Nv;
            int nbT = (int)((tn + bs - 1) / bs);
            pBeg();
            kTransportNode<<<nbT, bs>>>(dc, dt, cp.radius, sp.mlsOrder,
                               cp.periodicX ? 1 : 0, cp.periodicY ? 1 : 0,
                               cp.Lx, cp.Ly);
            pEnd("transport2D");
            pBeg();
            kPositivity<<<nb, bs>>>(dc);
            pEnd("positivity");
        }
    }
    pBeg();
    kCommit<<<nb, bs>>>(dc);
    pEnd("commit");
    if (mb) M3 = sumMomXHost(nb, bs, false); I3 = sumMomXHost(nb, bs, false, true);          // g, post-commit

    pBeg();
    massNew = sumMassHost(nb, bs, d3);
    if (massNew > 1e-30)
        kScaleG<<<nb, bs>>>(dc, massOld / massNew);
    pEnd("massCorrection");

    // ALEBGK_MASS_BUDGET=1: attribute the mass drift to a stage.
    //
    // The run loses 10.7% of its mass over 88,000 steps, linearly. kScaleG
    // above restores massOld exactly -- but it runs BEFORE wallBC and
    // collision, so anything those stages do to the total is never corrected,
    // and the host-side remove/refill happens outside this function
    // altogether. Measuring the total after each of them says which one
    // actually leaks instead of leaving it to inference.
    static int mbud = -1;
    if (mbud < 0) {
        const char *e = std::getenv("ALEBGK_MASS_BUDGET");
        mbud = (e && e[0] && std::atoi(e)) ? 1 : 0;
    }
    double mAfterCorr = 0.0;
    if (mbud) mAfterCorr = sumMassHost(nb, bs, d3);
    if (mb) M4 = sumMomXHost(nb, bs, false); I4 = sumMomXHost(nb, bs, false, true);          // g, post-mass-rescale

    size_t tnAll = (size_t)dc.N * (d3 ? (size_t)dc.Nv*dc.Nv*dc.Nv
                                      : (size_t)dc.Nv*dc.Nv);
    int nbN = (int)((tnAll + bs - 1) / bs);
    // Diagnostic only: ALEBGK_ENERGY_BUDGET=1 brackets total system energy
    // around each pipeline stage for the first N steps, to find which
    // stage injects the energy that grows unboundedly while scaleG holds
    // mass exactly fixed. Off by default (extra reduction + sync cost).
    static int ebSteps = -1;
    if (ebSteps < 0)
    {
        const char *e = std::getenv("ALEBGK_ENERGY_BUDGET");
        ebSteps = (e && e[0]) ? std::atoi(e) : 0;
    }
    static int ebCount = 0;
    bool eb = d3 && ebSteps > 0 && ebCount < ebSteps;
    if (d3)
    {
        double E0 = 0, E1 = 0, E2 = 0, E3 = 0;
        if (eb) E0 = sumEnergyHost(nb, bs);
        pBeg();
        kWallBC3D  <<<nb, bs>>>(dc, R);
        pEnd("wallBC");
        if (mb) M5 = sumMomXHost(nb, bs, false); I5 = sumMomXHost(nb, bs, false, true);      // g, post-wall BC
        if (eb) E1 = sumEnergyHost(nb, bs);
        pBeg();
        kMoments3D <<<nb, bs>>>(dc, R);
        pEnd("moments");
        pBeg();
        kCollisionNode3D<<<nbN, bs>>>(dc, dt, sp.tao, R);
        pEnd("collision");
        // Second mass restore, at the END of the step.
        //
        // kScaleG above runs after commit but BEFORE wallBC and collision, so
        // whatever those two do to the total was never corrected. Measured:
        // -1.17e-6 per step, which compounds to -9.8% over 88,000 steps
        // against the -10.7% actually observed -- essentially the whole leak.
        // (The host ALE contributes exactly zero, so remove/refill is not
        // involved.)
        //
        // A uniform rescale of g is safe to apply here: u and T are ratios of
        // moments, so both are invariant under g -> s*g. Only the density is
        // touched, which is precisely what needs restoring.
        // ALEBGK_NO_MASS_FIX=1 disables it for A/B comparison.
        static int nofix = -1;
        if (nofix < 0) {
            const char *e = std::getenv("ALEBGK_NO_MASS_FIX");
            nofix = (e && e[0] && std::atoi(e)) ? 1 : 0;
        }
        double mColl = (mbud || !nofix) ? sumMassHost(nb, bs, d3) : 0.0;
        if (!nofix && mColl > 1e-30)
            kScaleG<<<nb, bs>>>(dc, massOld / mColl);
        if (mbud)
        {
            g_massAfterCorr = mAfterCorr;
            g_massAfterColl = mColl;
        }
        if (eb) E2 = sumEnergyHost(nb, bs);
        pBeg();
        kMoments3D <<<nb, bs>>>(dc, R);
        pEnd("moments(post)");
        if (mb)
        {
            M6 = sumMomXHost(nb, bs, false); I6 = sumMomXHost(nb, bs, false, true);          // g, post-collision
            printf("[MOMX] step %3d  Px(all)=%+.6e  Px(interior)=%+.6e\n"
                   "   all      transport %+.3e | limiters %+.3e | "
                   "commit %+.3e | massRescale %+.3e\n"
                   "            wallBC    %+.3e | collision %+.3e | "
                   "net %+.3e\n"
                   "   interior transport %+.3e | limiters %+.3e | "
                   "commit %+.3e | massRescale %+.3e\n"
                   "            wallBC    %+.3e | collision %+.3e | "
                   "net %+.3e\n",
                   mbCount, M0, I0,
                   M1 - M0, M2 - M1, M3 - M2, M4 - M3,
                   M5 - M4, M6 - M5, M6 - M0,
                   I1 - I0, I2 - I1, I3 - I2, I4 - I3,
                   I5 - I4, I6 - I5, I6 - I0);
            fflush(stdout);
            ++mbCount;
        }
        if (eb)
        {
            E3 = sumEnergyHost(nb, bs);
            printf("[ENERGY] step %3d  preWall=%.10e  postWall=%.10e "
                   "(dWall=%+.3e)  postColl=%.10e (dColl=%+.3e)  "
                   "postMomPost=%.10e (dMomPost=%+.3e)\n",
                   ebCount, E0, E1, E1 - E0, E2, E2 - E1, E3, E3 - E2);
            fflush(stdout);
            ++ebCount;
        }
    }
    else
    {
        pBeg();
        kWallBC  <<<nb, bs>>>(dc, R);
        pEnd("wallBC");
        pBeg();
        kMoments <<<nb, bs>>>(dc, R);
        pEnd("moments");
        pBeg();
        kCollisionNode<<<nbN, bs>>>(dc, dt, sp.tao, R);
        pEnd("collision");
        // Second mass restore, at the END of the step.
        //
        // kScaleG above runs after commit but BEFORE wallBC and collision, so
        // whatever those two do to the total was never corrected. Measured:
        // -1.17e-6 per step, which compounds to -9.8% over 88,000 steps
        // against the -10.7% actually observed -- essentially the whole leak.
        // (The host ALE contributes exactly zero, so remove/refill is not
        // involved.)
        //
        // A uniform rescale of g is safe to apply here: u and T are ratios of
        // moments, so both are invariant under g -> s*g. Only the density is
        // touched, which is precisely what needs restoring.
        // ALEBGK_NO_MASS_FIX=1 disables it for A/B comparison.
        static int nofix = -1;
        if (nofix < 0) {
            const char *e = std::getenv("ALEBGK_NO_MASS_FIX");
            nofix = (e && e[0] && std::atoi(e)) ? 1 : 0;
        }
        double mColl = (mbud || !nofix) ? sumMassHost(nb, bs, d3) : 0.0;
        if (!nofix && mColl > 1e-30)
            kScaleG<<<nb, bs>>>(dc, massOld / mColl);
        if (mbud)
        {
            g_massAfterCorr = mAfterCorr;
            g_massAfterColl = mColl;
        }
        pBeg();
        kMoments <<<nb, bs>>>(dc, R);
        pEnd("moments(post)");
    }
    CUCHECK(cudaGetLastError());

    if (prof)
    {
        fflush(stdout);
        cudaEventDestroy(pe0);
        cudaEventDestroy(pe1);
    }
}

// ── Hybrid moving-body coupling (v1) ────────────────────────────────────────
// Fluid stays resident on the GPU; each step the body-surface particles'
// distributions are downloaded (a few MB), the validated host Newton-Euler
// (moveRigidBodies: loads, Heun, rigid motion, wall state) runs on the host
// mirror P, and the updated positions / normals / wall+mesh velocities are
// uploaded back. v1 scope: NO cloud remove/refill on the device — valid
// while the body displacement stays well under a lattice spacing.
void moveRigidBodies(std::vector<Particle> &P,
                     std::vector<std::unique_ptr<RigidBody>> &bodies,
                     double dt, const DomainBoundary &dom,
                     const CalcParameters &cp);

void cudaSyncBodies(std::vector<Particle> &P,
                    std::vector<std::unique_ptr<RigidBody>> &bodies,
                    double dt, const DomainBoundary &dom,
                    const CalcParameters &cp)
{
    int N = dc.N;
    // download surface-particle g rows into the host mirror
    for (int i = 0; i < N; ++i)
        if (P[i].boundary && P[i].bodyId >= 0)
            CUCHECK(cudaMemcpy(P[i].g.data(), dc.g + (size_t)i * dc.gStride,
                               dc.gStride * 8, cudaMemcpyDeviceToHost));
    moveRigidBodies(P, bodies, dt, dom, cp);
    // upload refreshed state (full small fields: simple and a few ms)
    std::vector<double> h(N);
    auto up2 = [&](double *dst, auto get) {
        for (int i = 0; i < N; ++i) h[i] = get(P[i]);
        CUCHECK(cudaMemcpy(dst, h.data(), (size_t)N * 8,
                           cudaMemcpyHostToDevice));
    };
    up2(dc.x,   [](const Particle &p) { return p.x; });
    up2(dc.y,   [](const Particle &p) { return p.y; });
    up2(dc.z,   [](const Particle &p) { return p.z; });
    up2(dc.wnx, [](const Particle &p) { return p.wnx; });
    up2(dc.wny, [](const Particle &p) { return p.wny; });
    up2(dc.wnz, [](const Particle &p) { return p.wnz; });
    up2(dc.wUx, [](const Particle &p) { return p.wUx; });
    up2(dc.wUy, [](const Particle &p) { return p.wUy; });
    up2(dc.wUz, [](const Particle &p) { return p.wUz; });
    up2(dc.mvx, [](const Particle &p) { return p.meshVx; });
    up2(dc.mvy, [](const Particle &p) { return p.meshVy; });
    up2(dc.mvz, [](const Particle &p) { return p.meshVz; });
}

void cudaDownloadMoments(std::vector<Particle> &P)
{
    int N = dc.N;
    std::vector<double> h(N);
    auto dn = [&](double *src, auto set) {
        CUCHECK(cudaMemcpy(h.data(), src, N * 8, cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; ++i) set(P[i], h[i]);
    };
    dn(dc.rho, [](Particle &p, double v) { p.rho = v; });
    dn(dc.ux,  [](Particle &p, double v) { p.ux = v; });
    dn(dc.uy,  [](Particle &p, double v) { p.uy = v; });
    if (dc.dim == 3)
        dn(dc.uz, [](Particle &p, double v) { p.uz = v; });
    dn(dc.T,   [](Particle &p, double v) { p.T = v; });
    // Limiter activity for the step just taken. moodFlag drives both the
    // mood_count column of timeseries.csv and the moodFlag field in the VTK,
    // so this makes the GPU limiters visible in exactly the places the CPU
    // path already reports them -- plus a per-kind breakdown for the log.
    g_limPos = g_limNad = 0;
    g_limMlsDeg = g_limMlsExt = 0;
    g_mlsOvrSum = g_mlsOvrMax = 0.0;
    if (dc.limFlag)
    {
        CUCHECK(cudaMemcpy(h.data(), dc.limFlag, N * 8, cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; ++i)
        {
            int f = (int)(h[i] + 0.5);
            P[i].moodFlag = (f != 0);
            if (f & 1) ++g_limPos;
            if (f & 2) ++g_limNad;
        }
    }
    if (dc.mlsFlag)
    {
        std::vector<int> hm(N);
        CUCHECK(cudaMemcpy(hm.data(), dc.mlsFlag, N * 4,
                           cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; ++i)
        {
            if (hm[i] & 1) ++g_limMlsDeg;
            if (hm[i] & 2) ++g_limMlsExt;
        }
    }
    if (dc.mlsOvr)
    {
        std::vector<double> ho(N);
        CUCHECK(cudaMemcpy(ho.data(), dc.mlsOvr, N * 8,
                           cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; ++i)
        {
            g_mlsOvrSum += ho[i];
            if (ho[i] > g_mlsOvrMax) g_mlsOvrMax = ho[i];
        }
    }
    for (int i = 0; i < N; ++i) P[i].p = P[i].rho * P[i].T;
}

// Wall-BC x-momentum attribution: copies the per-particle change kWallBC3D
// made this step, so the host can bin it by which face/edge each wall
// particle sits on.
void cudaWallMomentum(std::vector<double> &out)
{
    out.assign(dc.N, 0.0);
    if (dc.bcdPx)
        CUCHECK(cudaMemcpy(out.data(), dc.bcdPx, (size_t)dc.N * 8,
                           cudaMemcpyDeviceToHost));
}

// Per-kind limiter counts from the most recent cudaDownloadMoments.
void cudaLimiterCounts(int *nPositivity, int *nNAD)
{
    if (nPositivity) *nPositivity = g_limPos;
    if (nNAD)        *nNAD        = g_limNad;
}

// MLS reconstruction health for the most recent step: how many particles hit
// a collapsed pivot, and how many produced a value outside their own
// neighbours' range (extrapolation rather than interpolation).
void cudaMlsCounts(int *nDegenerate, int *nExtrapolate)
{
    if (nDegenerate)  *nDegenerate  = g_limMlsDeg;
    if (nExtrapolate) *nExtrapolate = g_limMlsExt;
}

// Extrapolation overshoot AMPLITUDE for the most recent step, in density
// units: the L1 total across the cloud, and the worst single particle.
void cudaMlsOvershoot(double *sum, double *worst)
{
    if (sum)   *sum   = g_mlsOvrSum;
    if (worst) *worst = g_mlsOvrMax;
}

// Same quantities, reduced ON DEMAND rather than as a side effect of
// cudaDownloadMoments. The report cadence (saveEvery) is far too coarse to
// answer the question that matters: does the extrapolation error start
// growing BEFORE the surface load does, or only alongside it? Ordering is
// what separates cause from correlation, and it needs per-step sampling
// through the onset. The buffer is N doubles (~240 KB), so calling this
// every step for a few hundred steps costs nothing.
// Total gas mass on the device right now (host-callable), plus the two
// snapshots the mass budget takes inside the step.
double cudaTotalMass()
{
    int bs = 128;
    int nb = (int)(((size_t)dc.N + bs - 1) / bs);
    if (nb < 1) nb = 1;
    return sumMassHost(nb, bs, dc.dim == 3);
}

void cudaMassBudget(double *afterCorrection, double *afterCollision)
{
    if (afterCorrection) *afterCorrection = g_massAfterCorr;
    if (afterCollision)  *afterCollision  = g_massAfterColl;
}

void cudaMlsOvershootNow(double *sum, double *worst)
{
    double s = 0.0, w = 0.0;
    if (dc.mlsOvr && dc.N > 0)
    {
        static std::vector<double> ho;
        if ((int)ho.size() < dc.N) ho.resize(dc.N);
        CUCHECK(cudaMemcpy(ho.data(), dc.mlsOvr, (size_t)dc.N * 8,
                           cudaMemcpyDeviceToHost));
        for (int i = 0; i < dc.N; ++i)
        {
            s += ho[i];
            if (ho[i] > w) w = ho[i];
        }
    }
    if (sum)   *sum   = s;
    if (worst) *worst = w;
}

// Download the full device state (per-particle distribution g + moments)
// into the host mirror.  Used by the moving-body ALE path so the host-side
// remove/refill/merge sees the current post-collision field.  Positions and
// connectivity are host-authoritative (the device never moves particles),
// so they are not read back here.
void cudaDownloadFull(std::vector<Particle> &P)
{
    int N = dc.N;
    // Static staging buffer. A fresh vector here meant allocating AND
    // value-initialising 238 MB every step -- the zero-fill alone writes the
    // whole buffer before the D2H copy overwrites it, and the fresh mapping
    // takes a page fault per 4 KB. Reusing one buffer keeps the pages warm;
    // resize() only ever grows it.
    static std::vector<double> hg;
    size_t need = (size_t)N * dc.gStride;
    if (hg.size() < need) hg.resize(need);
    CUCHECK(cudaMemcpy(hg.data(), dc.g, need * 8, cudaMemcpyDeviceToHost));
    // Scatter into the per-particle vectors. The bulk D2H copy above is ~24 ms
    // for 238 MB; this loop was the other ~170 ms, because it moved 29.7
    // MILLION doubles one element at a time on one thread. memcpy per particle
    // plus threading over particles costs the same bytes at a fraction of the
    // instructions -- each i touches only its own P[i].g, so there is no
    // sharing to guard.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
    {
        if (P[i].g.size() != dc.gStride) P[i].g.resize(dc.gStride);
        std::memcpy(P[i].g.data(), hg.data() + (size_t)i * dc.gStride,
                    dc.gStride * sizeof(double));
    }
    cudaDownloadMoments(P);
}

#endif // ALEBGK_WITH_CUDA
