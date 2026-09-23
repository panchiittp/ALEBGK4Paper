// =============================================================================
// Solver.cpp — BGK kinetic solver main time loop (serial CPU)
//
// Per step: MLS transport (Jacobi: compute all gt, then commit) → global mass
// correction → boundary conditions → moments → implicit BGK collision →
// moments → MOOD admissibility check → output.
// =============================================================================
#include "Config.hpp"
#include "Types.hpp"
#include "Checkpoint.hpp"
#include "MLSBasis.hpp"
#include "Output.hpp"
#include "Geometry.hpp"
#include <memory>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <fstream>

// ── External kernel functions (Kernels.cpp) ─────────────────────────────────
extern void applyIC(std::vector<Particle> &P, double rho0, double R,
                    double Lx, double Ly, double Lz);
extern void computeMoments(std::vector<Particle> &P, double R);
extern void mlsTransportCompute(std::vector<Particle> &P,
                                double dt, const CalcParameters &cp,
                                DomainBoundary dom,
                                BoundaryConditions BC,
                                int mlsOrder, bool onlyFlagged = false);
extern int checkTransportPAD(std::vector<Particle> &P);
extern void mlsTransportCommit(std::vector<Particle> &P);
extern void bgkCollision(std::vector<Particle> &P,
                         double dt, double tao, double R,
                         bool tauVariable, double dHS);
extern void applyBoundaryBC(std::vector<Particle> &P,
                            double R, DomainBoundary dom,
                            BoundaryConditions BC);
extern void applyGravity(std::vector<Particle> &P, double gY, double dt);
extern void moveRigidBodies(std::vector<Particle> &P,
                            std::vector<std::unique_ptr<RigidBody>> &bodies,
                            double dt, const DomainBoundary &dom,
                            const CalcParameters &cp);
extern int mergeCloseParticles(std::vector<Particle> &P,
                               const CalcParameters &cp);
extern int manageRemove(std::vector<Particle> &P,
                        std::vector<std::unique_ptr<RigidBody>> &bodies,
                        const CalcParameters &cp);
extern int manageRefill(std::vector<Particle> &P,
                        std::vector<std::unique_ptr<RigidBody>> &bodies,
                        const SimParameters &sp, const CalcParameters &cp,
                        const DomainBoundary &dom, double R);
extern void updateVoxelsAndNeighbours(std::vector<Particle> &P,
                                      const CalcParameters &cp,
                                      const DomainBoundary &dom);
#ifdef ALEBGK_WITH_MPI
void syncParticles(std::vector<Particle> &P);
// Domain decomposition (MpiExchange.cpp): slab mode replaces the full
// Allgather with a ghost-row exchange and gathers moments for output.
bool alebgk_decompActive();
long long alebgk_decompGlobalN();
void alebgk_haloExchange(std::vector<Particle> &P);
void alebgk_gatherOutput(const std::vector<Particle> &P,
                         std::vector<Particle> &Pout);
#endif
#ifdef ALEBGK_WITH_CUDA
void cudaUploadCloud(const std::vector<Particle> &P, const SimParameters &sp);
void cudaStep(const SimParameters &sp, const CalcParameters &cp, double dt,
              double R);
void cudaDownloadMoments(std::vector<Particle> &P);
void cudaDownloadFull(std::vector<Particle> &P);
void cudaSyncBodies(std::vector<Particle> &P,
                    std::vector<std::unique_ptr<RigidBody>> &bodies,
                    double dt, const DomainBoundary &dom,
                    const CalcParameters &cp);
// Limiter firings during the step most recently downloaded: how many
// particles the positivity repair and the NAD clamp each touched. Reported
// alongside MOOD so a silent limiter and a saturated one look different in
// the log -- they did not before, because nothing on the device ever wrote
// the flag that mood_count is computed from.
void cudaLimiterCounts(int *nPositivity, int *nNAD);
void cudaMlsCounts(int *nDegenerate, int *nExtrapolate);
void cudaMlsOvershoot(double *sum, double *worst);
void cudaMlsOvershootNow(double *sum, double *worst);
double cudaTotalMass();
void cudaMassBudget(double *afterCorrection, double *afterCollision);
void cudaWallMomentum(std::vector<double> &out);
#endif
// -- Body state log (shared by the CPU and CUDA time loops) -----------------
// One writer for both paths. They used to carry their own copy of this block
// and silently drifted: the CUDA copy grew the 3-D columns while the CPU copy
// kept the 2-D-only header and wrote b.angularVel, which is identically zero
// for a 3-D body (whose spin lives in omx/omy/omz). Every CPU 3-D run
// therefore logged "no rotation" no matter what the body actually did.
static void writeBodyLog(const std::string &outDir, double t,
                         const std::vector<std::unique_ptr<RigidBody>> &bodies)
{
    static bool hdr = false;
    std::ofstream f(outDir + "/bodies.csv",
                    hdr ? std::ios::app : std::ios::out);
    if (!hdr)
    {
        f << "t,body,cx,cy,velx,vely,angVel,Fx,Fy,torque,"
             "cz,velz,omx,omy,Fz,tqx,tqy\n";
        hdr = true;
    }
    for (std::size_t b = 0; b < bodies.size(); ++b)
    {
        const RigidBody &rb = *bodies[b];
        // angVel is the z-spin in either dimension: the 2-D scalar spin, or
        // omega_z of the 3-D vector. omx/omy carry the rest in 3-D.
        double wz = (rb.dim == 3) ? rb.omz : rb.angularVel;
        f << std::scientific << t << "," << b << ","
          << rb.cx << "," << rb.cy << ","
          << rb.velx << "," << rb.vely << "," << wz << ","
          << rb.forcex << "," << rb.forcey << "," << rb.tqz << ","
          << rb.cz << "," << rb.velz << "," << rb.omx << "," << rb.omy << ","
          << rb.forcez << "," << rb.tqx << "," << rb.tqy << "\n";
    }
}

extern bool moodCheck(std::vector<Particle> &P);
extern double sumMass(std::vector<Particle> &P);
extern void scaleG(std::vector<Particle> &P, double scale);

// =====================================================================
// runSolver — Main time-stepping loop (single process, no MPI)
// =====================================================================
void runSolver(std::vector<Particle>    &P,
               const SimParameters      &sp,
               const GasConstants       &gc,
               const DomainBoundary     &dom,
               const BoundaryConditions &BC,
               CalcParameters           &cp,
               std::vector<std::unique_ptr<RigidBody>> &bodies)
{
    if (ALEBGK_ROOT) fs::create_directories(sp.outDir);

    int N    = cp.N;
    int ndim = cp.dim;

    double Lx = dom.xright - dom.xleft;
    double Ly = dom.ytop   - dom.ybottom;
    double Lz = (ndim == 3) ? (dom.zback - dom.zfront) : 1.0;

    // ── Apply initial conditions ───────────────────────────────────────────
    applyIC(P, BC.rho, gc.R, Lx, Ly, Lz);

    if (ALEBGK_ROOT)
        printf("=== ALE-BGK Solver: N=%d  dim=%d  ranks=%d ===\n",
               N, ndim, alebgk::mpiSize());

    double t  = 0.0;
    double dt = sp.dt;

    // ── Foot-point guard ───────────────────────────────────────────────────
    // Semi-Lagrangian MLS transport has no advective CFL limit; what it
    // needs is the characteristic foot-point staying inside the MLS
    // stencil (support radius = 2.5 dx). Guard at 1.5 dx for headroom.
    {
        // Spacings by magnitude — the z lattice is stored descending
        // (dom.zfront = L, dom.zback = 0), so cp.dz is negative in 3D.
        double vmax  = std::max(std::fabs(sp.VMax), std::fabs(sp.VMin));
        double dxmin = std::fabs(cp.dx);
        if (std::fabs(cp.dy) < dxmin) dxmin = std::fabs(cp.dy);
        if (ndim == 3 && std::fabs(cp.dz) < dxmin) dxmin = std::fabs(cp.dz);
        double cfl    = vmax * dt / dxmin;
        // Clamp target: CFL 0.75, NOT the 1.5 stencil bound. A foot point
        // sitting exactly on the stencil edge has a one-sided, near-singular
        // reconstruction and diverges within a few steps (verified on the
        // 3D KH at Nx=40: dt clamped to CFL 1.5 blew up immediately, while
        // CFL <= 0.75 is inside every validated envelope).
        double dt_safe = 0.75 * dxmin / vmax;
        printf("[FOOT-POINT CHECK] dt= %.3g, v_max*dt/dx = %.3f (max allowed 1.5)\n",
               dt, cfl);
        if (cfl > 1.5)
        {
            if (ALEBGK_ROOT)
            {
                printf("[FOOT-POINT WARNING] dt= %.3g, v_max*dt/dx = %.3f > 1.5 "
                       "(outside MLS stencil).\n", dt, cfl);
                printf("              Clamping dt from %.3e to %.3e (CFL 0.75).\n",
                       dt, dt_safe);
                if (cfl > 15.0)
                    printf("              NOTE: the requested dt is %.0fx the stencil "
                           "limit. If you meant a FINAL TIME,\n"
                           "              the CLI is: <problem> <dim> [Nx] [Nv] "
                           "[dt] [tfinal] [estimate] -- dt comes BEFORE tfinal.\n",
                           cfl / 1.5);
            }
            dt = dt_safe;
        }
        else if (ALEBGK_ROOT)
        {
            printf("[FOOT-POINT OK] v_max*dt/dx = %.3f "
                   "(within MLS stencil)\n", cfl);
        }
    }

    int step = 0;
    auto tStart = std::chrono::high_resolution_clock::now();

    // ── Runtime-estimate instrumentation (CLI [estimate] flag) ────────────
    // With the flag given (0 or 1), the duration of step 2 (the steady
    // per-step cost — step 1 carries warm-up) projects the wall time of the
    // whole run, refined by the measured first-50-steps time. Flag 0 is
    // estimate-only and stops the run early: at step 2 if the first 50
    // steps are projected to exceed one hour, at step 50 otherwise. Flag 1
    // prints the same estimates but never breaks. Under MPI, rank 0 decides
    // and the decision is broadcast so all ranks leave the loop together.
    const long estTotalSteps = (sp.dt > 0.0)
        ? static_cast<long>(std::ceil(sp.tfinal / sp.dt)) : 0L;
    double estT1 = 0.0;
    // Estimate = step-2 timing only (step 1 carries warm-up). Flag 0
    // prints the projection and stops the run at step 2; flag 1 prints it
    // and continues to tfinal.
    auto estCheck = [&](int stepNow) -> bool {   // true => stop the run
        double el = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - tStart).count();
        if (stepNow == 1) { estT1 = el; return false; }
        if (stepNow == 2)
        {
            double perStep = el - estT1;
            printf("[ESTIMATE] step 2 took %.3f s -> projected full run for time = %.3g"
                   "(%ld steps): %.3g h\n",
                   perStep, sp.tfinal, estTotalSteps,
                   perStep * estTotalSteps / 3600.0);
            fflush(stdout);
            return sp.estimateFlag == 0;   // estimate-only: stop here
        }
        return false;
    };
    auto estStop = [&](int stepNow) -> bool {
        if (sp.estimateFlag < 0 || stepNow > 2)
            return false;
        int stopNow = 0;
#ifdef ALEBGK_WITH_MPI
        if (ALEBGK_ROOT) stopNow = estCheck(stepNow) ? 1 : 0;
        MPI_Bcast(&stopNow, 1, MPI_INT, 0, MPI_COMM_WORLD);
#else
        stopNow = estCheck(stepNow) ? 1 : 0;
#endif
        return stopNow != 0;
    };

#ifdef ALEBGK_WITH_CUDA
    // ── GPU main loop ──────────────────────────────────────────────────────
    // Scope: 2D (Chu g1/g2) or 3D (single f on Nv^3), fixed velocity grid,
    // no gravity. The fluid stays resident on the device; moving bodies use
    // the v2 hybrid (full read-back + host ALE + reuse-aware re-upload each
    // step), so arbitrary body displacement — a full orbit — is supported.
    // Static-body / body-free runs keep the fluid on-device throughout.
    if (sp.velMode != VelGridMode::Fixed || sp.gravityY != 0.0)
    {
        fprintf(stderr, "[CUDA] unsupported configuration for the GPU "
                        "path (needs a Fixed velocity grid, no gravity). "
                        "Rebuild without ALEBGK_WITH_CUDA or choose another "
                        "problem.\n");
        return;
    }
    // ALEBGK_RESTART=<file>: resume a run that was interrupted. Must happen
    // BEFORE the first upload so the device gets the restored state. The
    // neighbour lists are rebuilt rather than stored -- they are a pure
    // function of the positions, and storing them would triple the file.
    {
        const char *rf = std::getenv("ALEBGK_RESTART");
        if (rf && rf[0])
        {
            long rstep = 0; double rt = 0.0;
            if (ckpt::restore(rf, &rstep, &rt, P, bodies))
            {
                step = static_cast<int>(rstep);
                t = rt;
                updateVoxelsAndNeighbours(P, cp, dom);
            }
            else
            {
                printf("[ckpt] restore FAILED -- refusing to start from a "
                       "half-known state\n");
                return;
            }
        }
    }
    cudaUploadCloud(P, sp);
    // Initial configuration (step 0): paper figures need the t=0 state,
    // which the save-every-N logic below never emits.
    if (step == 0) writeVTK(P, sp.outDir, 0, 0.0);
    // ALEBGK_CHECKPOINT_EVERY=<steps>: how often to dump a resumable state.
    // The file is N*gStride*8 bytes -- 4.1 GB at Nx=40/Nv=20 -- so this wants
    // to be thousands of steps, not hundreds.
    int ckEvery = 0;
    if (const char *e = std::getenv("ALEBGK_CHECKPOINT_EVERY"))
        if (e[0]) ckEvery = std::atoi(e);
    std::string ckPath = sp.outDir + "/restart.ckpt";
    while (t < sp.tfinal)
    {
        dt = std::min(dt, sp.tfinal - t);
        cudaStep(sp, cp, dt, gc.R);

        // ALEBGK_OVR_LOG_FROM=<step>: from that step on, record the
        // extrapolation-error amplitude EVERY step to ovr_trace.csv.
        // body_state.csv already carries the surface loads per step, so the
        // two join on `step` and give the ordering directly: if ovrL1 turns
        // up before |F| does, extrapolation drives the instability; if they
        // move together it is a correlate, and the clamp is curing a symptom
        // by a route I have not explained. The saveEvery cadence only
        // brackets the onset (reports at 11250/11500/11750 around an onset
        // at 11587) and cannot resolve that ordering.
        {
            static int ovrFrom = -2;
            static long ovrStep = 0;
            static std::ofstream ovrFile;
            ++ovrStep;
            if (ovrFrom == -2)
            {
                const char *e = std::getenv("ALEBGK_OVR_LOG_FROM");
                ovrFrom = (e && e[0]) ? std::atoi(e) : -1;
                if (ovrFrom >= 0)
                {
                    ovrFile.open("ovr_trace.csv");
                    ovrFile << "step,t,ovrL1,ovrMax\n";
                    ovrFile.precision(10);
                }
            }
            if (ovrFrom >= 0 && ovrStep >= ovrFrom)
            {
                double os = 0.0, ow = 0.0;
                cudaMlsOvershootNow(&os, &ow);
                ovrFile << ovrStep << ',' << (t + dt) << ',' << os << ','
                        << ow << "\n";
                if ((ovrStep % 50) == 0) ovrFile.flush();
            }
        }
        // Hybrid moving-body coupling (v2): the fluid is stepped on the GPU,
        // then the full post-collision state is read back so the validated
        // host ALE runs exactly as in the CPU path — stress-integrated loads
        // + Heun Newton-Euler, then cloud remove/refill/merge as the body
        // sweeps the lattice.  The (reuse-aware) re-upload puts the remeshed
        // cloud back on the device.  Correct for arbitrary body displacement
        // (a full orbit), unlike v1 which held the cloud fixed.
        if (sp.movingBodies && !bodies.empty())
        {
            // ALEBGK_HOST_PROFILE=1: per-stage timing of this host block.
            // Measured: the moving-body case costs 4.34 s/step against
            // 0.838 s/step for the same grid with a STATIC body, so 81% of
            // the step is spent here, on the host, and no GPU upgrade can
            // touch it. That 3.5 s is far more than the work looks worth --
            // a voxel-binned neighbour search over 30k particles should be
            // tens of ms -- so the breakdown decides whether this is an
            // inherent cost or a fixable hotspot.
            static int hpOn = -1;
            if (hpOn < 0) {
                const char *e = std::getenv("ALEBGK_HOST_PROFILE");
                hpOn = (e && e[0] && std::atoi(e)) ? 1 : 0;
            }
            using hpClock = std::chrono::high_resolution_clock;
            static double hpT[7] = {0,0,0,0,0,0,0};
            static long   hpN = 0;
            auto hpNow = [&] { return hpClock::now(); };
            auto hpAdd = [&](int slot, hpClock::time_point t0) {
                if (hpOn) hpT[slot] += std::chrono::duration<double>(
                    hpClock::now() - t0).count();
            };

            auto h0 = hpNow();
            cudaDownloadFull(P);
            hpAdd(0, h0);

            h0 = hpNow();
            moveRigidBodies(P, bodies, dt, dom, cp);
            hpAdd(1, h0);

            h0 = hpNow();
            int removed = manageRemove(P, bodies, cp);
            hpAdd(2, h0);

            h0 = hpNow();
            updateVoxelsAndNeighbours(P, cp, dom);
            hpAdd(3, h0);

            h0 = hpNow();
            int mergedN = mergeCloseParticles(P, cp);
            hpAdd(4, h0);

            h0 = hpNow();
            if (mergedN > 0) updateVoxelsAndNeighbours(P, cp, dom);
            int added = manageRefill(P, bodies, sp, cp, dom, gc.R);
            if (added > 0) updateVoxelsAndNeighbours(P, cp, dom);
            hpAdd(5, h0);

            cp.N = static_cast<int>(P.size());

            h0 = hpNow();
            cudaUploadCloud(P, sp);
            hpAdd(6, h0);

            // ALEBGK_MASS_BUDGET=1: close the books on the step. The GPU
            // budget covers correction -> collision; this adds the host ALE
            // block, which removes and refills particles OUTSIDE the mass
            // rescale entirely. Reported as fractions of the step's starting
            // mass so a 1e-6 per-step leak is visible long before it becomes
            // the 10.7% observed over a full run.
            static int mb2 = -1;
            if (mb2 < 0) {
                const char *e = std::getenv("ALEBGK_MASS_BUDGET");
                mb2 = (e && e[0] && std::atoi(e)) ? 1 : 0;
            }
            if (mb2)
            {
                static long mbStep = 0;
                double mCorr = 0.0, mColl = 0.0;
                cudaMassBudget(&mCorr, &mColl);
                double mHost = cudaTotalMass();
                if (++mbStep % 250 == 0 && mCorr > 0.0)
                    printf("        mass: afterCorr=%.9e  wallBC+coll %+.3e  "
                           "hostALE %+.3e  (fractions of afterCorr)\n",
                           mCorr, (mColl - mCorr) / mCorr,
                           (mHost - mColl) / mCorr);
            }

            if (hpOn && ++hpN % 100 == 0) {
                static const char *nm[7] = {
                    "cudaDownloadFull", "moveRigidBodies", "manageRemove",
                    "updateVoxNeigh", "mergeClose", "refill(+rebuilds)",
                    "cudaUploadCloud"};
                double tot = 0; for (int q = 0; q < 7; ++q) tot += hpT[q];
                printf("[HOST-PROFILE] mean over %ld steps (total %.3f s/step)\n",
                       hpN, tot / hpN);
                for (int q = 0; q < 7; ++q)
                    printf("[HOST-PROFILE]   %-18s %8.1f ms  %5.1f%%\n",
                           nm[q], 1e3 * hpT[q] / hpN, 100.0 * hpT[q] / tot);
                fflush(stdout);
            }

            if (step % sp.saveEvery == 0)
                writeBodyLog(sp.outDir, t + dt, bodies);
        }
        t += dt;
        step++;
        if (estStop(step)) break;
        if (step % sp.saveEvery == 0)
        {
            cudaDownloadMoments(P);
            auto d = computeDiagnostics(P);
            printStep(step, t, dt, d);
            {
                int nPos = 0, nNad = 0, nDeg = 0, nExt = 0;
                cudaLimiterCounts(&nPos, &nNad);
                cudaMlsCounts(&nDeg, &nExt);
                // mlsDeg  = collapsed Cholesky pivot (stencil does not span
                //           the basis) -> Shepard fallback
                // mlsExt  = reconstruction outside its own neighbours' range
                //           -> extrapolation, not interpolation
                double ovrSum = 0.0, ovrMax = 0.0;
                cudaMlsOvershoot(&ovrSum, &ovrMax);
                // ovrL1/ovrMax are the AMPLITUDE of the extrapolation error,
                // in density units. The count (mlsExt) sits flat at ~0.78%
                // through the whole run including the onset, so it cannot
                // distinguish a benign steady population of one-sided
                // stencils from an amplifying one. The amplitude can.
                printf("        limiters: positivity=%d  NAD=%d  "
                       "mlsDeg=%d  mlsExt=%d  ovrL1=%.3e  ovrMax=%.3e  "
                       "(of N=%d)\n",
                       nPos, nNad, nDeg, nExt, ovrSum, ovrMax, cp.N);
            }
            // ALEBGK_EQ_TEST=1: uniform-state reproduction test. Started
            // from an exact equilibrium (gas at rest at wall temperature,
            // every wall stationary) the discrete solution must be a fixed
            // point: a least-squares reconstruction whose basis contains a
            // constant reproduces a constant exactly, so rho must not move
            // off its initial value at all. Printed at FULL precision --
            // the VTK writer leaves its stream in scientific/6, which
            // resolves only ~1e-6 near rho=1.1 and quantises this test into
            // uselessness.
            static int eqOn = -1;
            if (eqOn < 0) {
                const char *e = std::getenv("ALEBGK_EQ_TEST");
                eqOn = (e && e[0] && std::atoi(e)) ? 1 : 0;
            }
            if (eqOn && ALEBGK_ROOT)
            {
                double mi = 0, mb = 0, ui = 0, ub = 0;
                for (const auto &q : P)
                {
                    double dr = std::fabs(q.rho - BC.rho);
                    double uu = std::sqrt(q.ux*q.ux + q.uy*q.uy + q.uz*q.uz);
                    if (q.boundary) { mb = std::max(mb, dr); ub = std::max(ub, uu); }
                    else            { mi = std::max(mi, dr); ui = std::max(ui, uu); }
                }
                printf("        [EQ] max|drho| interior=%.17e boundary=%.17e\n"
                       "             max|u|    interior=%.17e boundary=%.17e\n",
                       mi, mb, ui, ub);
            }
            // ALEBGK_WALL_MOMENTUM=1: attribute the wall BC's x-momentum
            // change to the face/edge/corner each wall particle sits on. A
            // stationary diffuse wall must be momentum-neutral, so any
            // non-cancelling row here is the bias.
            static int wmOn = -1;
            if (wmOn < 0) {
                const char *e = std::getenv("ALEBGK_WALL_MOMENTUM");
                wmOn = (e && e[0] && std::atoi(e)) ? 1 : 0;
            }
            if (wmOn && ALEBGK_ROOT)
            {
                std::vector<double> d;
                cudaWallMomentum(d);
                // classify by how many box faces the particle touches
                const char *lbl[4] = {"interior", "face", "edge", "corner"};
                double sum[4] = {0,0,0,0};
                int    cnt[4] = {0,0,0,0};
                double perFace[6] = {0,0,0,0,0,0};
                int    perFaceN[6] = {0,0,0,0,0,0};
                const char *fl[6] = {"x=lo","x=hi","y=lo","y=hi",
                                     "z=lo","z=hi"};
                double eps = 1e-12;
                double xl=std::min(dom.xleft,dom.xright);
                double xh=std::max(dom.xleft,dom.xright);
                double yl=std::min(dom.ybottom,dom.ytop);
                double yh=std::max(dom.ybottom,dom.ytop);
                double zl=std::min(dom.zback,dom.zfront);
                double zh=std::max(dom.zback,dom.zfront);
                for (std::size_t k = 0; k < P.size(); ++k)
                {
                    if (!P[k].boundary || P[k].bodyId >= 0) continue;
                    bool f[6] = {
                        std::fabs(P[k].x-xl)<eps, std::fabs(P[k].x-xh)<eps,
                        std::fabs(P[k].y-yl)<eps, std::fabs(P[k].y-yh)<eps,
                        (cp.dim==3) && std::fabs(P[k].z-zl)<eps,
                        (cp.dim==3) && std::fabs(P[k].z-zh)<eps};
                    int nf = 0;
                    for (int q=0;q<6;++q) if (f[q]) ++nf;
                    int cls = (nf>3)?3:nf;
                    sum[cls] += d[k]; cnt[cls]++;
                    if (nf == 1)
                        for (int q=0;q<6;++q)
                            if (f[q]) { perFace[q]+=d[k]; perFaceN[q]++; }
                }
                printf("        [WALLMOM] by contact count:");
                for (int q=1;q<4;++q)
                    printf("  %s n=%d dPx=%+.3e", lbl[q], cnt[q], sum[q]);
                printf("\n        [WALLMOM] pure faces:");
                for (int q=0;q<6;++q)
                    printf("  %s n=%d dPx=%+.3e", fl[q], perFaceN[q],
                           perFace[q]);
                printf("\n");
            }
            fflush(stdout);
            writeVTK(P, sp.outDir, step, t);
            writeTimeseries(sp.outDir + "/timeseries.csv", t, d);
        }

        // Checkpoint. For a moving-body run the host already holds the full
        // state (the ALE block downloads it every step); otherwise pull it
        // down first, since only the device has g.
        if (ckEvery > 0 && (step % ckEvery) == 0)
        {
            if (!(sp.movingBodies && !bodies.empty())) cudaDownloadFull(P);
            ckpt::save(ckPath, step, t, P, bodies);
        }
    }
    {
        double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - tStart).count();
        cudaDownloadMoments(P);
        auto d = computeDiagnostics(P);
        printStep(step, t, dt, d);
        printf("\nDone. Steps=%d  t=%.4e  Wall=%.2f s\n", step, t, elapsed);
    }
    return;
#endif

    // ── Output cloud accessor ──────────────────────────────────────────────
    // Replicated/serial: every rank holds the full cloud, use P directly.
    // Decomposed: gather owned moments into Pgat on rank 0 (collective —
    // must be called by ALL ranks, hence outside the ALEBGK_ROOT guards).
#ifdef ALEBGK_WITH_MPI
    std::vector<Particle> Pgat;
#endif
    auto outCloud = [&]() -> std::vector<Particle> & {
#ifdef ALEBGK_WITH_MPI
        if (alebgk_decompActive())
        {
            alebgk_gatherOutput(P, Pgat);
            return Pgat;
        }
#endif
        return P;
    };

    // ── Main time loop ─────────────────────────────────────────────────────
    // Initial configuration (step 0): paper figures need the t=0 state,
    // which the save-every-N logic below never emits.
    {
        auto &PO = outCloud();
        if (ALEBGK_ROOT)
            writeVTK(PO, sp.outDir, 0, 0.0);
    }
    while (t < sp.tfinal)
    {
        dt = std::min(dt, sp.tfinal - t);

        // 1. MLS semi-Lagrangian transport (Jacobi: all gt from old g).
        //    On periodic domains the stencil wraps via minimum image.
        double massOld = sumMass(P);
        mlsTransportCompute(P, dt, cp, dom, BC, sp.mlsOrder);
        if (sp.limiter == LimiterMode::MOOD)
        {
            int nf = checkTransportPAD(P);
            if (nf > 0)
                mlsTransportCompute(P, dt, cp, dom, BC, 1, true);
        }
        mlsTransportCommit(P);
        double massNew = sumMass(P);

        // 2. Global mass correction
        if (massNew > 1.0e-30)
            scaleG(P, massOld / massNew);

        // 3. Body force (gravity): exact velocity-space advection split
        applyGravity(P, sp.gravityY, dt);

        // 4. Boundary conditions on wall particles (periodic closure lives
        //    in the transport stencil; fully periodic domains have none)
        applyBoundaryBC(P, gc.R, dom, BC);

        // 5. Compute moments from transported + BC-corrected distribution
        computeMoments(P, gc.R);

        // 6. Implicit BGK collision (L-stable for any dt/tao)
        bgkCollision(P, dt, sp.tao, gc.R, sp.tauVariable, sp.d);

        // 7. Update moments after collision
        computeMoments(P, gc.R);

        // 8. MOOD admissibility check
        if (sp.limiter == LimiterMode::MOOD)
        {
            if (moodCheck(P))
            {
                // Re-relax toward equilibrium and recompute moments
                bgkCollision(P, dt, sp.tao, gc.R, sp.tauVariable, sp.d);
                computeMoments(P, gc.R);
            }
        }

#ifdef ALEBGK_WITH_MPI
        // One exchange per step. Decomposed mode: refresh only the ghost
        // rows' distributions from the slab neighbours (point-to-point).
        // Replicated mode: full Allgather of distribution + macro fields.
        if (alebgk_decompActive())
            alebgk_haloExchange(P);
        else
            syncParticles(P);
#endif

        // 8b. Moving rigid bodies: stress-integrated forces, Newton-Euler
        //     update, rigid surface motion, then cloud management (remove
        //     overrun particles, rebuild neighbours, refill vacated voxels
        //     from neighbour moments).
        if (sp.movingBodies && !bodies.empty())
        {
            moveRigidBodies(P, bodies, dt, dom, cp);
            int removed = manageRemove(P, bodies, cp);
            updateVoxelsAndNeighbours(P, cp, dom);
            int mergedN = mergeCloseParticles(P, cp);
            if (mergedN > 0) updateVoxelsAndNeighbours(P, cp, dom);
            int added = manageRefill(P, bodies, sp, cp, dom, gc.R);
            if (added > 0)
                updateVoxelsAndNeighbours(P, cp, dom);
            cp.N = static_cast<int>(P.size());
            if ((removed || added) && step % sp.saveEvery == 0 && ALEBGK_ROOT)
                printf("  [bodies] removed %d, added %d, N=%d\n",
                       removed, added, cp.N);

            // Non-finite tripwire. The cloud-management pass (remove /
            // merge / refill) is the one place new particle STATE is
            // synthesised mid-run -- refilled particles get a Maxwellian
            // built from neighbour-averaged moments -- so it is where a bad
            // value can enter without any kernel misbehaving. Checking here,
            // every step, turns "the field went NaN somewhere in the last
            // saveEvery steps" into an exact step and count. Cheap: one pass
            // over the macro fields, no distribution touched.
            if (ALEBGK_ROOT)
            {
                static bool tripped = false;
                if (!tripped)
                {
                    int nbad = 0; int first = -1;
                    for (std::size_t q = 0; q < P.size(); ++q)
                    {
                        const Particle &pq = P[q];
                        if (!std::isfinite(pq.rho) || !std::isfinite(pq.T) ||
                            !std::isfinite(pq.ux) || !std::isfinite(pq.uy) ||
                            !std::isfinite(pq.uz))
                        { if (first < 0) first = (int)q; ++nbad; }
                    }
                    if (nbad > 0)
                    {
                        tripped = true;
                        printf("[NONFINITE] step %d t=%.6e : %d of %d "
                               "particles bad, first index %d (boundary=%d, "
                               "bodyId=%d)\n", step, t + dt, nbad,
                               (int)P.size(), first,
                               (int)P[first].boundary, P[first].bodyId);
                        fflush(stdout);
                    }
                }
            }

            // Body state log (per saveEvery): decay curves live here
            if (step % sp.saveEvery == 0 && ALEBGK_ROOT)
                writeBodyLog(sp.outDir, t + dt, bodies);
        }

        t += dt;
        step++;
        if (estStop(step)) break;

        // 9. Output every saveEvery steps (outCloud() is collective)
        if (step % sp.saveEvery == 0)
        {
            auto &PO = outCloud();
            if (ALEBGK_ROOT)
            {
                auto d = computeDiagnostics(PO);
                printStep(step, t, dt, d);
                fflush(stdout);
                writeVTK(PO, sp.outDir, step, t);
                writeTimeseries(sp.outDir + "/timeseries.csv", t, d);
            }
        }
    }

    // ── Final output ───────────────────────────────────────────────────────
    {
        auto &PO = outCloud();
        if (ALEBGK_ROOT)
        {
            double elapsed = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - tStart).count();

            auto d = computeDiagnostics(PO);
            printStep(step, t, dt, d);
            writeVTK(PO, sp.outDir, step, t);
            writeTimeseries(sp.outDir + "/timeseries.csv", t, d);

            printf("\nDone. Steps=%d  t=%.4e  Wall=%.2f s\n",
                   step, t, elapsed);
        }
    }
}
