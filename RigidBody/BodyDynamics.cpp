// ===========================================================================
// BodyDynamics.cpp - Rigid-body dynamics: surface-stress loads and the Heun-coupled
// Newton-Euler update of position, velocity and orientation.
// ===========================================================================
#include "Config.hpp"
#include "Types.hpp"
#include "MLSBasis.hpp"
#include "Geometry.hpp"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>
#include <algorithm>
#include <limits>

static void computeBodyLoads(const std::vector<Particle> &P,
                             const std::vector<std::unique_ptr<RigidBody>> &bodies,
                             std::vector<double> &Fx, std::vector<double> &Fy,
                             std::vector<double> &Fz,
                             std::vector<double> &Tx, std::vector<double> &Ty,
                             std::vector<double> &Tz)
{
    std::fill(Fx.begin(), Fx.end(), 0.0);
    std::fill(Fy.begin(), Fy.end(), 0.0);
    std::fill(Fz.begin(), Fz.end(), 0.0);
    std::fill(Tx.begin(), Tx.end(), 0.0);
    std::fill(Ty.begin(), Ty.end(), 0.0);
    std::fill(Tz.begin(), Tz.end(), 0.0);

    int N = static_cast<int>(P.size());
    for (int i = 0; i < N; ++i)
    {
        const Particle &p = P[i];
        if (!p.boundary || p.bodyId < 0) continue;
        const RigidBody &b = *bodies[p.bodyId];

        double cdx = p.x - b.cx, cdy = p.y - b.cy, cdz = p.z - b.cz;
        int Nv = p.Nv_local;
        double dv = p.dv_local, vlo = p.vlo;
        double area = b.getSurfaceArea();

        if (p.dim == 3)
        {
            double dv3 = dv * dv * dv;
            double pxx = 0, pyy = 0, pzz = 0, pxy = 0, pxz = 0, pyz = 0;
            for (int kv = 0; kv < Nv; ++kv)
            {
                double cz = (vlo + kv * dv) - p.wUz;
                for (int jv = 0; jv < Nv; ++jv)
                {
                    double cy = (vlo + jv * dv) - p.wUy;
                    for (int iv = 0; iv < Nv; ++iv)
                    {
                        double cx = (vlo + iv * dv) - p.wUx;
                        double f = p.g[iv + Nv * (jv + Nv * kv)] * dv3;
                        pxx += f * cx * cx; pyy += f * cy * cy;
                        pzz += f * cz * cz;
                        pxy += f * cx * cy; pxz += f * cx * cz;
                        pyz += f * cy * cz;
                    }
                }
            }
            double tx = -(p.wnx * pxx + p.wny * pxy + p.wnz * pxz);
            double ty = -(p.wnx * pxy + p.wny * pyy + p.wnz * pyz);
            double tz = -(p.wnx * pxz + p.wny * pyz + p.wnz * pzz);
            Fx[p.bodyId] += area * tx;
            Fy[p.bodyId] += area * ty;
            Fz[p.bodyId] += area * tz;
            Tx[p.bodyId] += area * (cdy * tz - cdz * ty);
            Ty[p.bodyId] += area * (cdz * tx - cdx * tz);
            Tz[p.bodyId] += area * (cdx * ty - cdy * tx);
        }
        else
        {
            double dv2 = dv * dv;
            double phiX = 0.0, phiY = 0.0, phiXY = 0.0;
            for (int jv = 0; jv < Nv; ++jv)
            {
                double cy = (vlo + jv * dv) - p.wUy;
                for (int iv = 0; iv < Nv; ++iv)
                {
                    double cx = (vlo + iv * dv) - p.wUx;
                    double g1 = p.g[iv + Nv * jv];
                    phiX  += g1 * dv2 * cx * cx;
                    phiY  += g1 * dv2 * cy * cy;
                    phiXY += g1 * dv2 * cx * cy;
                }
            }
            double valX = -p.wnx * phiX  - p.wny * phiXY;
            double valY = -p.wnx * phiXY - p.wny * phiY;
            Fx[p.bodyId] += area * valX;
            Fy[p.bodyId] += area * valY;
            Tz[p.bodyId] += area * (cdx * valY - cdy * valX);
        }
    }
}

// setSurfaceVelocity — write the wall velocity implied by a trial body
// state (V, omega per body) onto its surface particles; positions and
// normals unchanged. In 2-D omega is the z-component; in 3-D the full
// omega x r applies.
static void setSurfaceVelocity(std::vector<Particle> &P,
                               const std::vector<std::unique_ptr<RigidBody>> &bodies,
                               const std::vector<double> &vx,
                               const std::vector<double> &vy,
                               const std::vector<double> &vz,
                               const std::vector<double> &ox,
                               const std::vector<double> &oy,
                               const std::vector<double> &oz)
{
    int N = static_cast<int>(P.size());
    for (int i = 0; i < N; ++i)
    {
        Particle &p = P[i];
        if (!p.boundary || p.bodyId < 0) continue;
        const RigidBody &b = *bodies[p.bodyId];
        int bi = p.bodyId;
        double rx = p.x - b.cx, ry = p.y - b.cy, rz = p.z - b.cz;
        p.wUx = vx[bi] + oy[bi] * rz - oz[bi] * ry;
        p.wUy = vy[bi] + oz[bi] * rx - ox[bi] * rz;
        if (p.dim == 3)
            p.wUz = vz[bi] + ox[bi] * ry - oy[bi] * rx;
    }
}

// moveRigidBodies — surface stress integration + Newton-Euler update +
// rigid motion of the body's surface particles (meshfree4bgk2d port).
// Interior particles stay Eulerian; only the body surface is ALE.
//
// The body ODE advances with a Heun predictor-corrector on the interface
// loads: L0 at the current wall state; predict (V*, w*); re-evaluate L1
// with the predicted wall velocities on the same frozen g (capturing the
// diffuse wall's drag/torque Jacobian dF/dV — the feedback channel of the
// staggered coupling); advance with (L0+L1)/2. The previous scheme
// advanced the body with a force lagged by one full step, and that lag
// made the freely rotating body oscillate around its correct equilibrium
// (w -> vorticity/2) with growing amplitude until divergence — the
// classic staggered/partitioned interface-load instability. The reference
// (meshfree4bgk2d) sidesteps it by coupling the body twice per ARS222
// stage; Heun removes the lag entirely, needs no tuning parameter, and is
// exact at load equilibrium.
void moveRigidBodies(std::vector<Particle> &P,
                     std::vector<std::unique_ptr<RigidBody>> &bodies,
                     double dt, const DomainBoundary &dom,
                     const CalcParameters &cp)
{
    if (bodies.empty()) return;
    int nb = static_cast<int>(bodies.size());

    // ── Stage tripwire (ALEBGK_BODY_TRACE=1) ──────────────────────────────
    // The body state has been observed to go entirely non-finite in a single
    // step -- centre, velocity, omega, force and torque all at once -- with
    // the fluid completely healthy either side of it. A whole-step check
    // cannot say WHICH of the six stages did it, and the candidates behave
    // very differently: the surface-stress integral divides by a quadrature
    // sum, the Newton-Euler advance divides by mass and inertia, and the
    // contact projection takes min/max over the surface particles (so a
    // single bad particle poisons the whole body). This reports the first
    // stage at which anything stops being finite, and the state going in.
    static int btOn = -1;
    if (btOn < 0) {
        const char *e = std::getenv("ALEBGK_BODY_TRACE");
        btOn = (e && e[0] && std::atoi(e)) ? 1 : 0;
    }
    static long btStep = 0;
    ++btStep;
    static bool btTripped = false;
    auto btFinite = [&](const char *stage) {
        if (!btOn || btTripped) return;
        for (int q = 0; q < nb; ++q)
        {
            const RigidBody &rb = *bodies[q];
            bool ok = std::isfinite(rb.cx) && std::isfinite(rb.cy)
                   && std::isfinite(rb.cz) && std::isfinite(rb.velx)
                   && std::isfinite(rb.vely) && std::isfinite(rb.velz)
                   && std::isfinite(rb.omx) && std::isfinite(rb.omy)
                   && std::isfinite(rb.omz) && std::isfinite(rb.forcex)
                   && std::isfinite(rb.forcey) && std::isfinite(rb.forcez)
                   && std::isfinite(rb.tqx) && std::isfinite(rb.tqy)
                   && std::isfinite(rb.tqz);
            int nbadp = 0;
            for (const auto &pp : P)
                if (pp.boundary && pp.bodyId == q &&
                    !(std::isfinite(pp.x) && std::isfinite(pp.y) &&
                      std::isfinite(pp.z)))
                    ++nbadp;
            if (!ok || nbadp)
            {
                btTripped = true;
                printf("[BODYTRACE] step %ld body %d FIRST NON-FINITE "
                       "after stage '%s'  (bad surface particles %d)\n"
                       "            c=(%g,%g,%g) v=(%g,%g,%g) "
                       "w=(%g,%g,%g)\n"
                       "            F=(%g,%g,%g) T=(%g,%g,%g) "
                       "mass=%g I=%g\n",
                       btStep, q, stage, nbadp,
                       rb.cx, rb.cy, rb.cz, rb.velx, rb.vely, rb.velz,
                       rb.omx, rb.omy, rb.omz,
                       rb.forcex, rb.forcey, rb.forcez,
                       rb.tqx, rb.tqy, rb.tqz, rb.mass,
                       rb.getMomentOfInertia());
                fflush(stdout);
            }
        }
    };
    btFinite("entry");
    std::vector<double> Fx0(nb), Fy0(nb), Fz0(nb), Tx0(nb), Ty0(nb), Tz0(nb);
    std::vector<double> Fx1(nb), Fy1(nb), Fz1(nb), Tx1(nb), Ty1(nb), Tz1(nb);
    std::vector<double> vxs(nb), vys(nb), vzs(nb);
    std::vector<double> oxs(nb), oys(nb), ozs(nb);

    // ALEBGK_BODY_FREEZE=1: pin the body. Loads are still integrated and the
    // whole moving-body pipeline still runs every step -- download, remove,
    // merge, refill, neighbour rebuild, re-upload -- but the body never
    // actually moves, so its surface velocity stays zero and the foot-points
    // are never displaced by wall motion.
    //
    // This separates two things that have been confounded all along. The
    // STATIC-body case (problem 11, same grid and gas) ran 40,001 steps
    // clean; the moving-body case dies at 11,655. The difference could be the
    // body's MOTION (wall velocity in the BC, displaced foot-points) or
    // merely the moving-body CODE PATH (per-step remove/refill/re-upload,
    // which problem 11 never executes). Freezing the body runs the code path
    // without the motion:
    //   still fails  -> the pipeline is at fault, motion is irrelevant
    //   survives     -> the body's own motion is required
    static int frz = -1;
    if (frz < 0) {
        const char *e = std::getenv("ALEBGK_BODY_FREEZE");
        frz = (e && e[0] && std::atoi(e)) ? 1 : 0;
        if (frz) printf("[body] ALEBGK_BODY_FREEZE=1 -> body pinned; "
                        "moving-body pipeline still runs every step\n");
    }
    if (frz)
        for (auto &bp : bodies) {
            RigidBody &b = *bp;
            b.velx = b.vely = b.velz = 0.0;
            b.omx = b.omy = b.omz = 0.0;
            b.angularVel = 0.0;
        }

    // ALEBGK_BODY_FAKEV=<m/s> (with FREEZE): tell the wall BC and the
    // foot-point transport that the surface is moving at this speed, while
    // nothing actually moves. The frozen test showed the moving-body PIPELINE
    // is innocent -- 16,001 steps clean -- but it did no real work either
    // (zero cloud reallocations, against 230 in a moving run), so three
    // things that motion brings are still live:
    //   (1) non-zero wall velocity in the diffuse BC (Maxwellian shifted in
    //       velocity space relative to the fixed Nv grid)
    //   (2) foot-points displaced by the mesh velocity
    //   (3) real remove/refill events as the body sweeps the lattice
    // This activates (1) and (2) with (3) still absent, so a failure here
    // convicts the boundary treatment and a survival convicts remove/refill.
    // ALEBGK_BODY_PRESCRIBE=<m/s>: the body really translates at this fixed
    // speed along +x. Loads are still computed but do NOT drive it, so the
    // body sweeps the cloud for real -- stencils change, particles are
    // removed and refilled -- while its motion carries no feedback from the
    // fluid and its surface velocity stays uniform and constant.
    //
    // This is the last split. Everything with a PINNED body survives:
    // static body 40,001 steps, frozen body + full pipeline 16,001, frozen
    // body + fictitious wall velocity and mesh velocity 16,001. Only a body
    // that physically moves through the cloud fails, at 11,655. What remains
    // is whether that is the GEOMETRIC coupling (stencils changing,
    // remove/refill firing) or the two-way FEEDBACK between fluid loads and
    // body motion. Prescribed motion has the former without the latter.
    // ALEBGK_BODY_REPLAY=<body_state.csv>: drive the body along a RECORDED
    // trajectory instead of by the fluid loads. Loads are still integrated,
    // and still ignored.
    //
    // The straight-line prescribed test survived, but it logged only 3 cloud
    // reallocations against 230 in the free run -- translation without spin
    // barely disturbs the lattice -- so it could not separate "the two-way
    // feedback matters" from "the cloud churn matters". Replaying the free
    // body's own orbit reproduces the churn exactly while removing only the
    // feedback:
    //   survives -> the feedback is the driver
    //   fails    -> the geometric churn alone suffices, feedback irrelevant
    struct RepRow { double cx, cy, cz, vx, vy, vz, ox, oy, oz; };
    static std::vector<RepRow> rep;
    static int repOn = -1;
    if (repOn < 0)
    {
        repOn = 0;
        const char *e = std::getenv("ALEBGK_BODY_REPLAY");
        if (e && e[0])
        {
            std::ifstream rf(e);
            std::string ln;
            std::getline(rf, ln);                 // header
            while (std::getline(rf, ln))
            {
                std::vector<double> v;
                std::size_t a = 0;
                while (a <= ln.size())
                {
                    std::size_t c = ln.find(',', a);
                    if (c == std::string::npos) c = ln.size();
                    v.push_back(std::atof(ln.substr(a, c - a).c_str()));
                    a = c + 1;
                }
                if (v.size() < 12) continue;
                rep.push_back({v[3], v[4], v[5], v[6], v[7], v[8],
                               v[9], v[10], v[11]});
            }
            repOn = rep.empty() ? 0 : 1;
            printf("[body] ALEBGK_BODY_REPLAY -> %zu recorded steps loaded "
                   "from %s\n", rep.size(), e);
        }
    }
    if (repOn)
    {
        std::size_t k = (std::size_t)(btStep - 1);
        if (k >= rep.size()) k = rep.size() - 1;
        const RepRow &r = rep[k];
        for (auto &bp : bodies) {
            RigidBody &b = *bp;
            b.cx = r.cx; b.cy = r.cy; b.cz = r.cz;
            b.velx = r.vx; b.vely = r.vy; b.velz = r.vz;
            b.omx = r.ox; b.omy = r.oy; b.omz = r.oz;
            b.angularVel = r.oz;
        }
    }

    static double presV = -1.0;
    if (presV < 0.0) {
        const char *e = std::getenv("ALEBGK_BODY_PRESCRIBE");
        presV = (e && e[0]) ? std::fabs(std::atof(e)) : 0.0;
        if (presV > 0.0)
            printf("[body] ALEBGK_BODY_PRESCRIBE=%.4g m/s -> body translates "
                   "at fixed speed; loads computed but not fed back\n",
                   presV);
    }
    if (presV > 0.0)
        for (auto &bp : bodies) {
            RigidBody &b = *bp;
            b.velx = presV; b.vely = 0.0; b.velz = 0.0;
            b.omx = b.omy = b.omz = 0.0; b.angularVel = 0.0;
        }

    static double fakeV = -1.0;
    if (fakeV < 0.0) {
        const char *e = std::getenv("ALEBGK_BODY_FAKEV");
        fakeV = (e && e[0]) ? std::fabs(std::atof(e)) : 0.0;
        if (fakeV > 0.0)
            printf("[body] ALEBGK_BODY_FAKEV=%.4g m/s -> wall BC and "
                   "foot-points see a moving surface; geometry pinned\n",
                   fakeV);
    }

    // 1. Loads at the current state (p.wU* carry V_n, w_n)
    computeBodyLoads(P, bodies, Fx0, Fy0, Fz0, Tx0, Ty0, Tz0);
    if (btOn && !btTripped)
        for (int q = 0; q < nb; ++q)
            if (!(std::isfinite(Fx0[q]) && std::isfinite(Fy0[q]) &&
                  std::isfinite(Fz0[q]) && std::isfinite(Tx0[q]) &&
                  std::isfinite(Ty0[q]) && std::isfinite(Tz0[q])))
            {
                btTripped = true;
                printf("[BODYTRACE] step %ld body %d NON-FINITE LOADS L0: "
                       "F=(%g,%g,%g) T=(%g,%g,%g)\n", btStep, q,
                       Fx0[q], Fy0[q], Fz0[q], Tx0[q], Ty0[q], Tz0[q]);
                fflush(stdout);
            }

    // 2. Predictor body state. 2-D spin lives in the z-component.
    for (int bi = 0; bi < nb; ++bi)
    {
        RigidBody &b = *bodies[bi];
        double I = b.getMomentOfInertia();
        vxs[bi] = b.velx + Fx0[bi] * dt / b.mass;
        vys[bi] = b.vely + Fy0[bi] * dt / b.mass;
        vzs[bi] = (b.dim == 3) ? b.velz + Fz0[bi] * dt / b.mass : 0.0;
        if (b.rotates)
        {
            if (b.dim == 3)
            {
                oxs[bi] = b.omx + Tx0[bi] * dt / I;
                oys[bi] = b.omy + Ty0[bi] * dt / I;
                ozs[bi] = b.omz + Tz0[bi] * dt / I;
            }
            else
            {
                oxs[bi] = 0.0; oys[bi] = 0.0;
                ozs[bi] = b.angularVel + Tz0[bi] * dt / I;
            }
        }
        else
        {
            oxs[bi] = (b.dim == 3) ? b.omx : 0.0;
            oys[bi] = (b.dim == 3) ? b.omy : 0.0;
            ozs[bi] = (b.dim == 3) ? b.omz : b.angularVel;
        }
    }

    // 3. Corrector loads at the predicted wall state (same frozen g)
    setSurfaceVelocity(P, bodies, vxs, vys, vzs, oxs, oys, ozs);
    computeBodyLoads(P, bodies, Fx1, Fy1, Fz1, Tx1, Ty1, Tz1);
    if (btOn && !btTripped)
        for (int q = 0; q < nb; ++q)
            if (!(std::isfinite(Fx1[q]) && std::isfinite(Fy1[q]) &&
                  std::isfinite(Fz1[q]) && std::isfinite(Tx1[q]) &&
                  std::isfinite(Ty1[q]) && std::isfinite(Tz1[q])))
            {
                btTripped = true;
                printf("[BODYTRACE] step %ld body %d NON-FINITE LOADS L1 "
                       "(predictor v=(%g,%g,%g) w=(%g,%g,%g)): "
                       "F=(%g,%g,%g) T=(%g,%g,%g)\n", btStep, q,
                       vxs[q], vys[q], vzs[q], oxs[q], oys[q], ozs[q],
                       Fx1[q], Fy1[q], Fz1[q], Tx1[q], Ty1[q], Tz1[q]);
                fflush(stdout);
            }

    int N = static_cast<int>(P.size());

    // 4. Rigid motion of the surface particles with the OLD body state.
    //    2-D: planar rotation. 3-D: Rodrigues rotation about w-hat by
    //    |w| dt, then translation.
    for (int i = 0; i < N; ++i)
    {
        Particle &p = P[i];
        if (!p.boundary || p.bodyId < 0) continue;
        RigidBody &b = *bodies[p.bodyId];
        double rx = p.x - b.cx, ry = p.y - b.cy, rz = p.z - b.cz;
        if (p.dim == 3)
        {
            double wmag = std::sqrt(b.omx*b.omx + b.omy*b.omy + b.omz*b.omz);
            double nxr = rx, nyr = ry, nzr = rz;
            if (wmag > 1e-300)
            {
                double ax = b.omx / wmag, ay = b.omy / wmag, az = b.omz / wmag;
                double th = wmag * dt, c = std::cos(th), s = std::sin(th);
                double adotr = ax*rx + ay*ry + az*rz;
                double crx = ay*rz - az*ry;
                double cry = az*rx - ax*rz;
                double crz = ax*ry - ay*rx;
                nxr = rx*c + crx*s + ax*adotr*(1.0-c);
                nyr = ry*c + cry*s + ay*adotr*(1.0-c);
                nzr = rz*c + crz*s + az*adotr*(1.0-c);
            }
            p.x = b.cx + dt * b.velx + nxr;
            p.y = b.cy + dt * b.vely + nyr;
            p.z = b.cz + dt * b.velz + nzr;
        }
        else
        {
            double th = b.angularVel * dt;
            p.x = b.cx + dt * b.velx + std::cos(th) * rx - std::sin(th) * ry;
            p.y = b.cy + dt * b.vely + std::sin(th) * rx + std::cos(th) * ry;
        }
    }

    // 5. Newton-Euler advance with the averaged (Heun) loads
    for (int bi = 0; bi < nb; ++bi)
    {
        RigidBody &b = *bodies[bi];
        double I = b.getMomentOfInertia();
        if (frz) continue;            // pinned: loads measured, motion suppressed
        // Replay: the pose for this step was already snapped at the top of
        // the routine, and stage 4 moved the surface particles with those
        // velocities. Integrating here as well would advance the body a
        // second time, putting it one step ahead of the record.
        if (repOn) continue;
        if (presV > 0.0)              // prescribed: advance, then re-pin below
        {
            b.cx += dt * presV;
            b.velx = presV; b.vely = b.velz = 0.0;
            b.omx = b.omy = b.omz = 0.0; b.angularVel = 0.0;
            continue;
        }
        b.cx += dt * b.velx;
        b.cy += dt * b.vely;
        if (b.dim == 3) b.cz += dt * b.velz;

        if (b.dim == 3)
            // 3-D orientation advances with the OLD angular velocity,
            // matching the rigid surface-point rotation in step 4. The
            // (ox, oy) planar orientation vector is 2-D-only state and is
            // deliberately left alone here.
            b.rotate3(b.omx, b.omy, b.omz, dt);
        else
        {
            double th = b.angularVel * dt;   // rotate 2-D orientation
            double ox2 = b.ox, oy2 = b.oy;
            b.ox = std::cos(th) * ox2 - std::sin(th) * oy2;
            b.oy = std::sin(th) * ox2 + std::cos(th) * oy2;
        }

        b.forcex = 0.5 * (Fx0[bi] + Fx1[bi]);
        b.forcey = 0.5 * (Fy0[bi] + Fy1[bi]);
        b.forcez = 0.5 * (Fz0[bi] + Fz1[bi]);
        b.tqx    = 0.5 * (Tx0[bi] + Tx1[bi]);
        b.tqy    = 0.5 * (Ty0[bi] + Ty1[bi]);
        b.tqz    = 0.5 * (Tz0[bi] + Tz1[bi]);
        b.torque = b.tqz;                // 2-D scalar view

        b.velx += b.forcex * dt / b.mass;
        b.vely += b.forcey * dt / b.mass;
        if (b.dim == 3) b.velz += b.forcez * dt / b.mass;
        if (b.rotates)
        {
            if (b.dim == 3)
            {
                b.omx += b.tqx * dt / I;
                b.omy += b.tqy * dt / I;
                b.omz += b.tqz * dt / I;
            }
            else
            {
                b.angularVel += b.tqz * dt / I;
            }
        }
        // Mirror the 3-D z-spin onto the 2-D scalar view. angularVel is
        // otherwise dead state on a 3-D body -- permanently zero -- and any
        // consumer that reads it (logs, plots, diagnostics) then reports a
        // freely spinning body as not rotating at all.
        if (b.dim == 3) b.angularVel = b.omz;
    }

    btFinite("4+5 rigid-motion & Newton-Euler");

    // 5b. Box-wall contact. Nothing else in the solver stops a free body
    //     leaving the domain: manageRemove/manageRefill police body-vs-FLUID
    //     only, and the diffuse wall acts on the gas, not on the body. A body
    //     swept into a wall therefore passed straight through it (measured:
    //     87 nm of a 150 nm cube outside the cavity), and every step after
    //     that is meaningless.
    //
    //     Model: inelastic, frictionless normal contact, resolved by
    //     projection. The body's extent is taken from its OWN surface
    //     particles, so this is shape-agnostic (cube, sphere, square alike)
    //     and stays correct as the body rotates -- no per-shape support
    //     function to keep in sync with Geometry.hpp. The whole body (centre
    //     + every surface particle) is translated by one rigid shift, so
    //     rigidity is preserved exactly; then the wall-normal velocity
    //     component is zeroed if it still points into the wall.
    //
    //     Restitution is zero because a 150 nm body in argon at Re ~ 0.5 has
    //     no meaningful rebound: the approach is overdamped by the gas long
    //     before elastic contact matters.
    //
    //     The body is held one "hug" distance (0.4 dx -- the same constant
    //     embedRigidBodies and manageRemove use to decide a fluid particle is
    //     too close to the surface) clear of the wall rather than exactly
    //     touching. At closer range there are no fluid particles left between
    //     body and wall for the MLS stencil to work with, so contact at zero
    //     gap is not a resolvable state on this cloud. That gap is a
    //     discretisation floor, not a physical stand-off distance.
    {
        double gap = 0.4 * std::fabs(cp.dx);
        double xlo = std::min(dom.xleft, dom.xright);
        double xhi = std::max(dom.xleft, dom.xright);
        double ylo = std::min(dom.ybottom, dom.ytop);
        double yhi = std::max(dom.ybottom, dom.ytop);
        double zlo = std::min(dom.zback, dom.zfront);
        double zhi = std::max(dom.zback, dom.zfront);
        bool d3 = (cp.dim == 3);

        for (int bi = 0; bi < nb; ++bi)
        {
            RigidBody &b = *bodies[bi];
            // extent of this body, from its own surface particles
            double pxlo = 1e300, pxhi = -1e300, pylo = 1e300, pyhi = -1e300;
            double pzlo = 1e300, pzhi = -1e300;
            int nSurf = 0;
            for (int i = 0; i < N; ++i)
            {
                const Particle &p = P[i];
                if (!p.boundary || p.bodyId != bi) continue;
                pxlo = std::min(pxlo, p.x); pxhi = std::max(pxhi, p.x);
                pylo = std::min(pylo, p.y); pyhi = std::max(pyhi, p.y);
                pzlo = std::min(pzlo, p.z); pzhi = std::max(pzhi, p.z);
                ++nSurf;
            }
            if (nSurf == 0) continue;

            // per-axis shift that brings the body back inside, with the gap.
            // Both signs are computed and summed so a body wider than the
            // free span is centred rather than flung out of the far wall.
            double sx = std::max(0.0, (xlo + gap) - pxlo)
                      - std::max(0.0, pxhi - (xhi - gap));
            double sy = std::max(0.0, (ylo + gap) - pylo)
                      - std::max(0.0, pyhi - (yhi - gap));
            double sz = d3 ? std::max(0.0, (zlo + gap) - pzlo)
                           - std::max(0.0, pzhi - (zhi - gap))
                          : 0.0;
            if (sx == 0.0 && sy == 0.0 && sz == 0.0) continue;

            b.cx += sx; b.cy += sy; b.cz += sz;
            for (int i = 0; i < N; ++i)
            {
                Particle &p = P[i];
                if (!p.boundary || p.bodyId != bi) continue;
                p.x += sx; p.y += sy; p.z += sz;
            }
            // Inelastic normal contact: kill only the component still driving
            // the body into the wall. A body sliding ALONG the wall keeps its
            // tangential motion, and the spin is untouched (frictionless).
            if (sx > 0.0 && b.velx < 0.0) b.velx = 0.0;
            if (sx < 0.0 && b.velx > 0.0) b.velx = 0.0;
            if (sy > 0.0 && b.vely < 0.0) b.vely = 0.0;
            if (sy < 0.0 && b.vely > 0.0) b.vely = 0.0;
            if (d3)
            {
                if (sz > 0.0 && b.velz < 0.0) b.velz = 0.0;
                if (sz < 0.0 && b.velz > 0.0) b.velz = 0.0;
            }
        }
    }

    btFinite("5b-wall-contact");

    // 6. Refresh surface wall state from the new positions and body state
    for (int i = 0; i < N; ++i)
    {
        Particle &p = P[i];
        if (!p.boundary || p.bodyId < 0) continue;
        RigidBody &b = *bodies[p.bodyId];
        double rx = p.x - b.cx, ry = p.y - b.cy, rz = p.z - b.cz;
        if (p.dim == 3)
        {
            b.getNormal3(p.x, p.y, p.z, &p.wnx, &p.wny, &p.wnz);
            p.wUx = b.velx + b.omy * rz - b.omz * ry;
            p.wUy = b.vely + b.omz * rx - b.omx * rz;
            p.wUz = b.velz + b.omx * ry - b.omy * rx;
            if (fakeV > 0.0) { p.wUx = fakeV; p.wUy = 0.0; p.wUz = 0.0; }
            p.meshVz = p.wUz;
        }
        else
        {
            b.getNormal(p.x, p.y, &p.wnx, &p.wny);
            p.wUx = b.velx - b.angularVel * ry;
            p.wUy = b.vely + b.angularVel * rx;
        }
        p.meshVx = p.wUx;   // transport foot-points follow the surface
        p.meshVy = p.wUy;
    }

    // -- Per-step body-state log + magnitude tripwire (ALEBGK_BODY_LOG=1) ---
    // The finiteness tripwire above answers "which stage produced the NaN"
    // but that turned out to be the wrong question: when it fired, the state
    // going IN was already 1e275 in velocity and 1e283 in omega. The body had
    // been diverging exponentially for ~1660 steps and only tripped when it
    // overflowed to inf. isfinite() cannot see that -- 1e200 is finite -- and
    // bodies.csv samples every saveEvery steps, so the entire blow-up fell
    // between two log rows.
    //
    // This logs |v|, |w| and BOTH Heun loads every single step. L0 vs L1 is
    // the diagnostic that matters: in a staggered fluid-structure coupling
    // instability the corrector load L1 (evaluated at the predicted state)
    // grows away from the predictor load L0 with alternating sign, so the
    // ratio F1/F0 is the fingerprint. The magnitude trip reports the first
    // step the state leaves physical bounds rather than the step it overflows.
    static int blOn = -1;
    static std::ofstream blFile;
    if (blOn < 0)
    {
        const char *e = std::getenv("ALEBGK_BODY_LOG");
        blOn = (e && e[0] && std::atoi(e)) ? 1 : 0;
        if (blOn)
        {
            blFile.open("body_state.csv");
            blFile << "step,t,body,cx,cy,cz,velx,vely,velz,omx,omy,omz,"
                      "vmag,wmag,Fx0,Fy0,Fz0,Tx0,Ty0,Tz0,"
                      "Fx1,Fy1,Fz1,Tx1,Ty1,Tz1\n";
        }
    }
    if (blOn)
    {
        static double blT = 0.0;
        blT += dt;
        // Physical bounds for this case: the lid runs at 10 m/s and the
        // steady vortex spins the body at |w_z| ~ 5e6 rad/s. Anything two
        // decades past either is unambiguously numerical, not flow.
        static bool blTripped = false;
        blFile.precision(10);
        for (int q = 0; q < nb; ++q)
        {
            const RigidBody &rb = *bodies[q];
            double vm = std::sqrt(rb.velx * rb.velx + rb.vely * rb.vely
                                + rb.velz * rb.velz);
            double wm = std::sqrt(rb.omx * rb.omx + rb.omy * rb.omy
                                + rb.omz * rb.omz);
            blFile << btStep << ',' << blT << ',' << q << ','
                   << rb.cx << ',' << rb.cy << ',' << rb.cz << ','
                   << rb.velx << ',' << rb.vely << ',' << rb.velz << ','
                   << rb.omx << ',' << rb.omy << ',' << rb.omz << ','
                   << vm << ',' << wm << ','
                   << Fx0[q] << ',' << Fy0[q] << ',' << Fz0[q] << ','
                   << Tx0[q] << ',' << Ty0[q] << ',' << Tz0[q] << ','
                   << Fx1[q] << ',' << Fy1[q] << ',' << Fz1[q] << ','
                   << Tx1[q] << ',' << Ty1[q] << ',' << Tz1[q] << "\n";
            if (!blTripped && (vm > 1.0e3 || wm > 5.0e8))
            {
                blTripped = true;
                printf("[BODYMAG] step %ld body %d LEFT PHYSICAL BOUNDS "
                       "t=%.6e  |v|=%.6e (lid 10)  |w|=%.6e (flow 5e6)\n"
                       "          v=(%.6e,%.6e,%.6e) w=(%.6e,%.6e,%.6e)\n"
                       "          L0 F=(%.6e,%.6e,%.6e) T=(%.6e,%.6e,%.6e)\n"
                       "          L1 F=(%.6e,%.6e,%.6e) T=(%.6e,%.6e,%.6e)\n",
                       btStep, q, blT, vm, wm,
                       rb.velx, rb.vely, rb.velz, rb.omx, rb.omy, rb.omz,
                       Fx0[q], Fy0[q], Fz0[q], Tx0[q], Ty0[q], Tz0[q],
                       Fx1[q], Fy1[q], Fz1[q], Tx1[q], Ty1[q], Tz1[q]);
                fflush(stdout);
            }
        }
        if ((btStep % 100) == 0) blFile.flush();
    }
}

// manageParticles — Eulerian-cloud housekeeping around moving bodies:
// remove interior particles the body has overrun (inside or hugging),
// merge too-close interior pairs, refill voxels the body has vacated
// with particles whose g is a Maxwellian of neighbour-averaged moments.
// Returns true if the cloud changed (neighbour lists must be rebuilt by
// the caller BEFORE the refill pass reads neighbour moments — so this
// runs in two phases with a rebuild callback).
