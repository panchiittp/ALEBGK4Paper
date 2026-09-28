// =============================================================================
// MpiExchange.cpp — rank synchronisation for the MPI backend.
//
// Compiled only when ALEBGK_WITH_MPI is defined. Each kernel writes its
// rank's chunk [chunkLo, chunkHi); one exchange per time step (after the
// MOOD stage) makes every rank's distribution and macro fields globally
// consistent, which is all the next step's MLS transport (the only
// neighbour-reading kernel) and the rigid-body coupling need.
// =============================================================================
#include "Config.hpp"

#ifdef ALEBGK_WITH_MPI

#include "Types.hpp"
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

void syncParticles(std::vector<Particle> &P)
{
    int N = static_cast<int>(P.size());
    int nprocs = alebgk::mpiSize();
    int rank   = alebgk::mpiRank();
    if (nprocs == 1) return;

    std::vector<int> lo(nprocs), hi(nprocs);
    for (int r = 0; r < nprocs; ++r)
    {
        lo[r] = static_cast<int>(static_cast<long long>(r) * N / nprocs);
        hi[r] = static_cast<int>(static_cast<long long>(r + 1) * N / nprocs);
    }

    // ── Phase 1: distribution data g ─────────────────────────────────────
    std::vector<int> gCounts(nprocs), gDispls(nprocs);
    int totalG = 0;
    for (int r = 0; r < nprocs; ++r)
    {
        int count = 0;
        for (int i = lo[r]; i < hi[r]; ++i)
            count += static_cast<int>(P[i].g.size());
        gCounts[r] = count;
        gDispls[r] = totalG;
        totalG += count;
    }

    std::vector<double> sendG(gCounts[rank]);
    {
        int off = 0;
        for (int i = lo[rank]; i < hi[rank]; ++i)
            for (double v : P[i].g) sendG[off++] = v;
    }
    std::vector<double> recvG(totalG);
    MPI_Allgatherv(sendG.data(), gCounts[rank], MPI_DOUBLE,
                   recvG.data(), gCounts.data(), gDispls.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);
    for (int r = 0; r < nprocs; ++r)
    {
        if (r == rank) continue;
        int off = gDispls[r];
        for (int i = lo[r]; i < hi[r]; ++i)
            for (double &v : P[i].g) v = recvG[off++];
    }

    // ── Phase 2: macro fields (8 doubles per particle) ───────────────────
    constexpr int STRIDE = 8;
    std::vector<int> mCounts(nprocs), mDispls(nprocs);
    int totalM = 0;
    for (int r = 0; r < nprocs; ++r)
    {
        mCounts[r] = (hi[r] - lo[r]) * STRIDE;
        mDispls[r] = totalM;
        totalM += mCounts[r];
    }
    std::vector<double> sendM(mCounts[rank]);
    {
        int off = 0;
        for (int i = lo[rank]; i < hi[rank]; ++i)
        {
            const Particle &p = P[i];
            sendM[off++] = p.rho;
            sendM[off++] = p.ux;
            sendM[off++] = p.uy;
            sendM[off++] = p.uz;
            sendM[off++] = p.T;
            sendM[off++] = p.p;
            sendM[off++] = p.moodFlag ? 1.0 : 0.0;
            sendM[off++] = p.validg ? 1.0 : 0.0;
        }
    }
    std::vector<double> recvM(totalM);
    MPI_Allgatherv(sendM.data(), mCounts[rank], MPI_DOUBLE,
                   recvM.data(), mCounts.data(), mDispls.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);
    for (int r = 0; r < nprocs; ++r)
    {
        if (r == rank) continue;
        int off = mDispls[r];
        for (int i = lo[r]; i < hi[r]; ++i)
        {
            Particle &p = P[i];
            p.rho      = recvM[off++];
            p.ux       = recvM[off++];
            p.uy       = recvM[off++];
            p.uz       = recvM[off++];
            p.T        = recvM[off++];
            p.p        = recvM[off++];
            p.moodFlag = recvM[off++] > 0.5;
            p.validg   = recvM[off++] > 0.5;
        }
    }
}

// =============================================================================
// Domain decomposition (slab + halo exchange).
//
// The replicated design above bounds the problem size by
// (per-node memory) / (ranks per node): every rank stores the full cloud.
// The decomposed mode below stores, per rank, only a slab of the lattice —
// the owned rows plus haloRows ghost rows on each side — so memory per rank
// is ~N/nranks instead of N.  It applies to static-cloud problems on a
// Fixed velocity grid (driven cavity, Kelvin–Helmholtz, 2D and 3D); body
// problems and the Adaptive grid keep the replicated path.
//
// Slab axis = the outermost generation axis (y in 2D, z in 3D), so a slab
// is a CONTIGUOUS index range of the generated lattice and, after trimming,
// the owned particles are a contiguous range of the local vector — which is
// exactly what chunk() hands to every kernel.  Ghost rows cover the MLS
// stencil radius; per step only the ghost rows' distributions are exchanged
// with the two slab neighbours (point-to-point), replacing the full
// Allgather.  On periodic axes the slab neighbours wrap and the ghost rows
// keep their natural coordinates: the voxel stencil wraps modulo the box
// count and displacements use the minimum image, so the existing neighbour
// search finds them unchanged.
//
// Mode selection: automatic when eligible; set ALEBGK_DECOMP=0 to force the
// replicated path (e.g. for the bitwise serial-vs-MPI comparison on
// periodic problems, where local neighbour-list ordering can differ by
// round-off).  On non-periodic problems the decomposed sums visit
// particles in the same order as the serial code, so results stay bitwise
// identical there too.
// =============================================================================

namespace alebgk { bool decompOn = false; int decompLo = 0, decompHi = 0; }

namespace
{
struct DecompState
{
    bool planned  = false;   // plan attempted
    bool eligible = false;   // slab mode selected
    bool periodic = false;   // slab axis periodic (wrap partners)
    int  rowsTotal = 0;      // lattice rows along the slab axis
    long long rowStride = 0; // particles per row (Nx in 2D, Nx*Ny in 3D)
    int  haloRows = 0;       // ghost rows per side
    int  ownRow0 = 0, ownRow1 = 0;   // this rank's owned rows [r0, r1)
    long long globalN = 0;
    int  dim = 2;
    int  upRank = MPI_PROC_NULL, downRank = MPI_PROC_NULL;
    std::vector<char> rowKept;       // rowsTotal flags
    std::vector<int>  rowLocalPos;   // kept-row -> position among kept rows
    // local layout (indices into the trimmed local vector)
    int ownLo = 0, ownHi = 0;
    long long ghostBelowStart = -1, ghostAboveStart = -1;
    int ghostRowsBelow = 0, ghostRowsAbove = 0;
    // output gather (rank 0)
    std::vector<int> outCounts, outDispls;
} S;
} // namespace

// Called from generateParticles once the lattice geometry is known and
// BEFORE the particle loop, so g allocation can be skipped for rows this
// rank does not keep.
void alebgk_decompPlan(int rowsTotal, long long rowStride, bool axisPeriodic,
                       double axisSpacing, double radius, int dim,
                       bool problemEligible)
{
    S = DecompState();
    S.planned = true;
    int n = alebgk::mpiSize(), r = alebgk::mpiRank();

    const char *env = std::getenv("ALEBGK_DECOMP");
    bool envOff = (env && env[0] == '0');

    S.rowsTotal = rowsTotal;
    S.rowStride = rowStride;
    S.periodic  = axisPeriodic;
    S.dim       = dim;
    S.haloRows  = static_cast<int>(
                      std::ceil(radius / std::fabs(axisSpacing))) + 1;

    int minRows = rowsTotal;   // smallest owned-slab height over all ranks
    for (int q = 0; q < n; ++q)
    {
        int lo = static_cast<int>(static_cast<long long>(q) * rowsTotal / n);
        int hi = static_cast<int>(static_cast<long long>(q + 1) * rowsTotal / n);
        minRows = std::min(minRows, hi - lo);
    }

    // Slabs must be at least two halos tall: a neighbour serves a full halo
    // from its own rows, and (two-rank periodic) a rank's below- and
    // above-ghost blocks must be distinct rows of the same partner.
    S.eligible = problemEligible && n > 1 && !envOff
                 && minRows >= 2 * S.haloRows;
    if (!S.eligible)
    {
        if (ALEBGK_ROOT && problemEligible && n > 1)
            printf("[decomp] replicated mode (%s)\n",
                   envOff ? "ALEBGK_DECOMP=0"
                          : "slab thinner than 2 halos — raise Nx or use "
                            "fewer ranks for slab mode");
        return;
    }

    S.ownRow0 = static_cast<int>(static_cast<long long>(r) * rowsTotal / n);
    S.ownRow1 = static_cast<int>(static_cast<long long>(r + 1) * rowsTotal / n);
    S.downRank = (r > 0)     ? r - 1 : (S.periodic ? n - 1 : MPI_PROC_NULL);
    S.upRank   = (r < n - 1) ? r + 1 : (S.periodic ? 0     : MPI_PROC_NULL);

    S.rowKept.assign(rowsTotal, 0);
    for (int row = S.ownRow0; row < S.ownRow1; ++row) S.rowKept[row] = 1;
    auto keepWrapped = [&](int row) {
        if (S.periodic) S.rowKept[((row % rowsTotal) + rowsTotal) % rowsTotal] = 1;
        else if (row >= 0 && row < rowsTotal) S.rowKept[row] = 1;
    };
    for (int k = 1; k <= S.haloRows; ++k)
    {
        if (S.downRank != MPI_PROC_NULL) keepWrapped(S.ownRow0 - k);
        if (S.upRank   != MPI_PROC_NULL) keepWrapped(S.ownRow1 - 1 + k);
    }
    S.ghostRowsBelow = (S.downRank != MPI_PROC_NULL) ? S.haloRows : 0;
    S.ghostRowsAbove = (S.upRank   != MPI_PROC_NULL) ? S.haloRows : 0;

    // Position of every kept row among the kept rows (ascending order —
    // the order the trimmed local vector inherits from the generation).
    S.rowLocalPos.assign(rowsTotal, -1);
    int pos = 0;
    for (int row = 0; row < rowsTotal; ++row)
        if (S.rowKept[row]) S.rowLocalPos[row] = pos++;
}

// Predicate used at the g-allocation site in generateParticles: rows this
// rank does not keep never allocate their distributions, so the transient
// full-cloud footprint of the replicated design is avoided.
bool alebgk_decompKeep(long long globalIdx)
{
    if (!S.eligible) return true;
    int row = static_cast<int>(globalIdx / S.rowStride);
    return S.rowKept[row] != 0;
}

// Called at the end of generateParticles: drop non-kept particles (order
// preserving), free ghost transport buffers, and activate the owned-range
// chunk() so every kernel computes exactly the owned particles.
void alebgk_decompTrim(std::vector<Particle> &P)
{
    if (!S.eligible) { alebgk::decompOn = false; return; }
    S.globalN = static_cast<long long>(P.size());

    std::size_t w = 0;
    for (std::size_t i = 0; i < P.size(); ++i)
    {
        int row = static_cast<int>(static_cast<long long>(i) / S.rowStride);
        if (!S.rowKept[row]) continue;
        if (w != i) P[w] = std::move(P[i]);
        ++w;
    }
    P.resize(w);
    P.shrink_to_fit();

    auto rowStart = [&](int row) -> long long {
        return static_cast<long long>(S.rowLocalPos[row]) * S.rowStride;
    };
    S.ownLo = static_cast<int>(rowStart(S.ownRow0));
    S.ownHi = S.ownLo
            + static_cast<int>((S.ownRow1 - S.ownRow0) * S.rowStride);

    if (S.ghostRowsBelow > 0)
    {
        int firstBelow = S.ownRow0 - S.haloRows;
        if (S.periodic)
            firstBelow = ((firstBelow % S.rowsTotal) + S.rowsTotal)
                         % S.rowsTotal;
        S.ghostBelowStart = rowStart(firstBelow);
    }
    if (S.ghostRowsAbove > 0)
    {
        int firstAbove = S.ownRow1 % S.rowsTotal;   // wraps only if periodic
        S.ghostAboveStart = rowStart(firstAbove);
    }

    // Ghosts are never transported: their gt is dead weight — free it.
    for (int i = 0; i < static_cast<int>(P.size()); ++i)
        if (i < S.ownLo || i >= S.ownHi)
        {
            P[i].gt.clear();
            P[i].gt.shrink_to_fit();
        }

    // Output-gather bookkeeping: counts/offsets of every rank's owned block
    // in GLOBAL index space (globalIdx of ownRow0 * fields per particle).
    int n = alebgk::mpiSize();
    S.outCounts.assign(n, 0);
    S.outDispls.assign(n, 0);
    int ownedN = S.ownHi - S.ownLo;
    MPI_Allgather(&ownedN, 1, MPI_INT, S.outCounts.data(), 1, MPI_INT,
                  MPI_COMM_WORLD);
    long long own0Global = static_cast<long long>(S.ownRow0) * S.rowStride;
    std::vector<long long> own0All(n);
    MPI_Allgather(&own0Global, 1, MPI_LONG_LONG, own0All.data(), 1,
                  MPI_LONG_LONG, MPI_COMM_WORLD);
    for (int q = 0; q < n; ++q)
        S.outDispls[q] = static_cast<int>(own0All[q]);

    alebgk::decompOn = true;
    alebgk::decompLo = S.ownLo;
    alebgk::decompHi = S.ownHi;

    printf("[decomp] rank %d: rows [%d,%d) of %d, owned %d + ghosts %d/%d "
           "rows, local N=%zu (of %lld)\n",
           alebgk::mpiRank(), S.ownRow0, S.ownRow1, S.rowsTotal,
           S.ownHi - S.ownLo, S.ghostRowsBelow, S.ghostRowsAbove,
           P.size(), S.globalN);
    fflush(stdout);
}

bool alebgk_decompActive() { return alebgk::decompOn; }
long long alebgk_decompGlobalN() { return S.globalN; }

// Per-step halo exchange: each rank refreshes its ghost rows' distributions
// from the neighbouring ranks' owned rows.  Two Sendrecv phases; on wall
// (non-periodic) edges the partner is MPI_PROC_NULL and the phase is a
// no-op.  Only g moves — ghost moments are never read by any kernel.
void alebgk_haloExchange(std::vector<Particle> &P)
{
    if (!alebgk::decompOn) return;
    const std::size_t gSize = P[S.ownLo].g.size();
    const long long   rowDoubles = S.rowStride * static_cast<long long>(gSize);
    const int haloCount = static_cast<int>(S.haloRows * rowDoubles);

    auto pack = [&](long long start, int rows, std::vector<double> &buf) {
        buf.resize(static_cast<std::size_t>(rows) * rowDoubles);
        std::size_t off = 0;
        long long np = rows * S.rowStride;
        for (long long i = start; i < start + np; ++i)
            for (double v : P[static_cast<std::size_t>(i)].g)
                buf[off++] = v;
    };
    auto unpack = [&](long long start, int rows,
                      const std::vector<double> &buf) {
        std::size_t off = 0;
        long long np = rows * S.rowStride;
        for (long long i = start; i < start + np; ++i)
            for (double &v : P[static_cast<std::size_t>(i)].g)
                v = buf[off++];
    };

    std::vector<double> sendBuf, recvBuf;

    // Phase A: my TOP owned rows -> up partner's below-ghosts;
    //          my below-ghosts   <- down partner's top owned rows.
    if (S.upRank != MPI_PROC_NULL || S.downRank != MPI_PROC_NULL)
    {
        int sendCnt = (S.upRank   != MPI_PROC_NULL) ? haloCount : 0;
        int recvCnt = (S.downRank != MPI_PROC_NULL) ? haloCount : 0;
        if (sendCnt)
            pack(S.ownHi - static_cast<long long>(S.haloRows) * S.rowStride,
                 S.haloRows, sendBuf);
        recvBuf.resize(recvCnt);
        MPI_Sendrecv(sendBuf.data(), sendCnt, MPI_DOUBLE, S.upRank,   101,
                     recvBuf.data(), recvCnt, MPI_DOUBLE, S.downRank, 101,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (recvCnt) unpack(S.ghostBelowStart, S.haloRows, recvBuf);
    }
    // Phase B: my BOTTOM owned rows -> down partner's above-ghosts;
    //          my above-ghosts     <- up partner's bottom owned rows.
    if (S.upRank != MPI_PROC_NULL || S.downRank != MPI_PROC_NULL)
    {
        int sendCnt = (S.downRank != MPI_PROC_NULL) ? haloCount : 0;
        int recvCnt = (S.upRank   != MPI_PROC_NULL) ? haloCount : 0;
        if (sendCnt) pack(S.ownLo, S.haloRows, sendBuf);
        recvBuf.resize(recvCnt);
        MPI_Sendrecv(sendBuf.data(), sendCnt, MPI_DOUBLE, S.downRank, 102,
                     recvBuf.data(), recvCnt, MPI_DOUBLE, S.upRank,   102,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (recvCnt) unpack(S.ghostAboveStart, S.haloRows, recvBuf);
    }
}

// Collective output gather: every rank contributes its owned particles'
// positions and moments (12 doubles each — no distributions), rank 0
// assembles the full cloud into Pout for diagnostics and VTK.  ~100 B per
// particle, so a 10^6-particle gather is ~12 MB.
void alebgk_gatherOutput(const std::vector<Particle> &P,
                         std::vector<Particle> &Pout)
{
    if (!alebgk::decompOn) return;
    constexpr int F = 12;
    int ownedN = S.ownHi - S.ownLo;

    std::vector<double> sendBuf(static_cast<std::size_t>(ownedN) * F);
    {
        std::size_t off = 0;
        for (int i = S.ownLo; i < S.ownHi; ++i)
        {
            const Particle &p = P[i];
            sendBuf[off++] = p.x;
            sendBuf[off++] = p.y;
            sendBuf[off++] = p.z;
            sendBuf[off++] = p.boundary ? 1.0 : 0.0;
            sendBuf[off++] = p.rho;
            sendBuf[off++] = p.ux;
            sendBuf[off++] = p.uy;
            sendBuf[off++] = p.uz;
            sendBuf[off++] = p.T;
            sendBuf[off++] = p.p;
            sendBuf[off++] = p.moodFlag ? 1.0 : 0.0;
            sendBuf[off++] = static_cast<double>(p.Nv_local);
        }
    }

    int n = alebgk::mpiSize();
    std::vector<int> counts(n), displs(n);
    for (int q = 0; q < n; ++q)
    {
        counts[q] = S.outCounts[q] * F;
        displs[q] = S.outDispls[q] * F;
    }
    std::vector<double> recvBuf;
    if (ALEBGK_ROOT)
        recvBuf.resize(static_cast<std::size_t>(S.globalN) * F);
    MPI_Gatherv(sendBuf.data(), ownedN * F, MPI_DOUBLE,
                recvBuf.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (!ALEBGK_ROOT) return;
    if (static_cast<long long>(Pout.size()) != S.globalN)
        Pout.assign(static_cast<std::size_t>(S.globalN), Particle(1, S.dim));
    std::size_t off = 0;
    for (long long i = 0; i < S.globalN; ++i)
    {
        Particle &p = Pout[static_cast<std::size_t>(i)];
        p.x        = recvBuf[off++];
        p.y        = recvBuf[off++];
        p.z        = recvBuf[off++];
        p.boundary = recvBuf[off++] > 0.5;
        p.rho      = recvBuf[off++];
        p.ux       = recvBuf[off++];
        p.uy       = recvBuf[off++];
        p.uz       = recvBuf[off++];
        p.T        = recvBuf[off++];
        p.p        = recvBuf[off++];
        p.moodFlag = recvBuf[off++] > 0.5;
        p.Nv_local = static_cast<int>(recvBuf[off++] + 0.5);
    }
}

#endif // ALEBGK_WITH_MPI
