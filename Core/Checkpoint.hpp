// -----------------------------------------------------------------------------
// Checkpoint.hpp — binary save/restore of the full solver state.
//
// The VTK snapshots carry moments only (rho, u, T, flags). For a kinetic
// solver the DISTRIBUTION FUNCTION is the state, so they cannot be resumed
// from: rebuilding g as a Maxwellian from its moments would discard the
// non-equilibrium part, which at Kn ~ 0.03 is precisely the physics being
// solved. A long run interrupted partway was therefore a total loss -- a
// 13-day full rotation at Nv=20 is one power cut from worthless.
//
// This writes everything needed to continue bit-identically: per-particle g
// plus the geometric and wall state, the rigid-body state including the 3-D
// orientation matrix, and the step/time counters.
//
// Format is raw little-endian POD with a magic+version header. It is a
// restart file, not an archive: it is only ever read back by the same build
// on the same machine, so no portable encoding is warranted. The header
// records N, gStride and dim, and restore() refuses a file that disagrees
// with the configured run rather than reading past the end of a buffer.
//
// Size is N * gStride * 8 bytes -- 4.1 GB at Nx=40/Nv=20 -- so the cadence
// wants to be thousands of steps, not hundreds.
// -----------------------------------------------------------------------------
#pragma once

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "Types.hpp"
#include "Geometry.hpp"

namespace ckpt {

constexpr uint64_t MAGIC   = 0x414c4542474b3031ULL;   // "ALEBGK01"
constexpr uint32_t VERSION = 1;

struct Header {
    uint64_t magic;
    uint32_t version;
    uint32_t dim;
    uint64_t N;
    uint64_t gStride;
    uint64_t step;
    double   t;
    uint32_t nBodies;
    uint32_t pad;
};

// Per-particle fields that cannot be recomputed from the cloud geometry.
// neighindex and gt are deliberately absent: the neighbour lists are rebuilt
// on load by the caller, and gt is transport scratch, valid only within a step.
struct PRec {
    double x, y, z;
    double ux, uy, uz, T, rho, p;
    double meshVx, meshVy, meshVz;
    double wnx, wny, wnz, wUx, wUy, wUz, wT;
    double vlo, vhi, dv_local;
    int32_t Nv_local, dim, bodyId, voxel;
    uint8_t boundary, validg, moodFlag, pad;
};

struct BRec {
    double cx, cy, cz;
    double velx, vely, velz;
    double angularVel, omx, omy, omz;
    double forcex, forcey, forcez;
    double torque, tqx, tqy, tqz;
    double ox, oy;
    double Rm[9];
    int32_t dim;
    uint8_t rotates, pad[3];
};

inline bool save(const std::string &path, long step, double t,
                 const std::vector<Particle> &P,
                 const std::vector<std::unique_ptr<RigidBody>> &bodies)
{
    if (P.empty()) return false;
    // Write to a temporary and rename: a checkpoint half-written when the
    // power fails must not overwrite the last good one.
    std::string tmp = path + ".tmp";
    FILE *f = std::fopen(tmp.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "[ckpt] cannot open %s\n", tmp.c_str()); return false; }

    Header h{};
    h.magic = MAGIC; h.version = VERSION;
    h.dim = static_cast<uint32_t>(P[0].dim);
    h.N = P.size();
    h.gStride = P[0].g.size();
    h.step = static_cast<uint64_t>(step);
    h.t = t;
    h.nBodies = static_cast<uint32_t>(bodies.size());
    std::fwrite(&h, sizeof h, 1, f);

    std::vector<PRec> pr(P.size());
    for (size_t i = 0; i < P.size(); ++i) {
        const Particle &p = P[i];
        PRec &r = pr[i];
        r.x=p.x; r.y=p.y; r.z=p.z;
        r.ux=p.ux; r.uy=p.uy; r.uz=p.uz; r.T=p.T; r.rho=p.rho; r.p=p.p;
        r.meshVx=p.meshVx; r.meshVy=p.meshVy; r.meshVz=p.meshVz;
        r.wnx=p.wnx; r.wny=p.wny; r.wnz=p.wnz;
        r.wUx=p.wUx; r.wUy=p.wUy; r.wUz=p.wUz; r.wT=p.wT;
        r.vlo=p.vlo; r.vhi=p.vhi; r.dv_local=p.dv_local;
        r.Nv_local=p.Nv_local; r.dim=p.dim; r.bodyId=p.bodyId; r.voxel=p.voxel;
        r.boundary=p.boundary?1:0; r.validg=p.validg?1:0;
        r.moodFlag=p.moodFlag?1:0; r.pad=0;
    }
    std::fwrite(pr.data(), sizeof(PRec), pr.size(), f);

    for (const auto &p : P) {
        if (p.g.size() != h.gStride) {          // ragged g would desync the read
            std::fclose(f); std::remove(tmp.c_str());
            std::fprintf(stderr, "[ckpt] ragged g (%zu vs %llu), not saving\n",
                         p.g.size(), (unsigned long long)h.gStride);
            return false;
        }
        std::fwrite(p.g.data(), sizeof(double), h.gStride, f);
    }

    std::vector<BRec> br(bodies.size());
    for (size_t b = 0; b < bodies.size(); ++b) {
        const RigidBody &rb = *bodies[b];
        BRec &r = br[b];
        r.cx=rb.cx; r.cy=rb.cy; r.cz=rb.cz;
        r.velx=rb.velx; r.vely=rb.vely; r.velz=rb.velz;
        r.angularVel=rb.angularVel; r.omx=rb.omx; r.omy=rb.omy; r.omz=rb.omz;
        r.forcex=rb.forcex; r.forcey=rb.forcey; r.forcez=rb.forcez;
        r.torque=rb.torque; r.tqx=rb.tqx; r.tqy=rb.tqy; r.tqz=rb.tqz;
        r.ox=rb.ox; r.oy=rb.oy;
        r.dim=rb.dim; r.rotates=rb.rotates?1:0;
        std::memset(r.pad, 0, sizeof r.pad);
        const double *Rm = rb.orientation();
        if (Rm) std::memcpy(r.Rm, Rm, sizeof r.Rm);
        else { std::memset(r.Rm, 0, sizeof r.Rm); r.Rm[0]=r.Rm[4]=r.Rm[8]=1.0; }
    }
    if (!br.empty()) std::fwrite(br.data(), sizeof(BRec), br.size(), f);

    std::fflush(f);
    std::fclose(f);
    std::remove(path.c_str());
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::fprintf(stderr, "[ckpt] rename failed for %s\n", path.c_str());
        return false;
    }
    std::printf("[ckpt] saved step %ld  t=%.6e  N=%llu  gStride=%llu  (%.2f GB)\n",
                step, t, (unsigned long long)h.N, (unsigned long long)h.gStride,
                (double)h.N * h.gStride * 8.0 / (1024.0*1024.0*1024.0));
    std::fflush(stdout);
    return true;
}

inline bool restore(const std::string &path, long *step, double *t,
                    std::vector<Particle> &P,
                    std::vector<std::unique_ptr<RigidBody>> &bodies)
{
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "[ckpt] cannot open %s\n", path.c_str()); return false; }

    Header h{};
    if (std::fread(&h, sizeof h, 1, f) != 1 || h.magic != MAGIC
        || h.version != VERSION) {
        std::fprintf(stderr, "[ckpt] %s is not a v%u checkpoint\n",
                     path.c_str(), VERSION);
        std::fclose(f); return false;
    }
    // Refuse a mismatched run rather than reading past the end of a buffer.
    if (P.size() != h.N || P.empty() || P[0].g.size() != h.gStride) {
        std::fprintf(stderr,
            "[ckpt] shape mismatch: file N=%llu gStride=%llu, run N=%zu "
            "gStride=%zu -- the checkpoint is from a different resolution\n",
            (unsigned long long)h.N, (unsigned long long)h.gStride,
            P.size(), P.empty() ? 0 : P[0].g.size());
        std::fclose(f); return false;
    }
    if (bodies.size() != h.nBodies) {
        std::fprintf(stderr, "[ckpt] body count mismatch (%u vs %zu)\n",
                     h.nBodies, bodies.size());
        std::fclose(f); return false;
    }

    std::vector<PRec> pr(h.N);
    if (std::fread(pr.data(), sizeof(PRec), h.N, f) != h.N) {
        std::fprintf(stderr, "[ckpt] truncated particle records\n");
        std::fclose(f); return false;
    }
    for (size_t i = 0; i < h.N; ++i) {
        Particle &p = P[i]; const PRec &r = pr[i];
        p.x=r.x; p.y=r.y; p.z=r.z;
        p.ux=r.ux; p.uy=r.uy; p.uz=r.uz; p.T=r.T; p.rho=r.rho; p.p=r.p;
        p.meshVx=r.meshVx; p.meshVy=r.meshVy; p.meshVz=r.meshVz;
        p.wnx=r.wnx; p.wny=r.wny; p.wnz=r.wnz;
        p.wUx=r.wUx; p.wUy=r.wUy; p.wUz=r.wUz; p.wT=r.wT;
        p.vlo=r.vlo; p.vhi=r.vhi; p.dv_local=r.dv_local;
        p.Nv_local=r.Nv_local; p.dim=r.dim; p.bodyId=r.bodyId; p.voxel=r.voxel;
        p.boundary=r.boundary!=0; p.validg=r.validg!=0; p.moodFlag=r.moodFlag!=0;
    }
    for (size_t i = 0; i < h.N; ++i) {
        if (P[i].g.size() != h.gStride) P[i].g.resize(h.gStride);
        if (std::fread(P[i].g.data(), sizeof(double), h.gStride, f) != h.gStride) {
            std::fprintf(stderr, "[ckpt] truncated g at particle %zu\n", i);
            std::fclose(f); return false;
        }
    }
    if (h.nBodies) {
        std::vector<BRec> br(h.nBodies);
        if (std::fread(br.data(), sizeof(BRec), h.nBodies, f) != h.nBodies) {
            std::fprintf(stderr, "[ckpt] truncated body records\n");
            std::fclose(f); return false;
        }
        for (size_t b = 0; b < h.nBodies; ++b) {
            RigidBody &rb = *bodies[b]; const BRec &r = br[b];
            rb.cx=r.cx; rb.cy=r.cy; rb.cz=r.cz;
            rb.velx=r.velx; rb.vely=r.vely; rb.velz=r.velz;
            rb.angularVel=r.angularVel; rb.omx=r.omx; rb.omy=r.omy; rb.omz=r.omz;
            rb.forcex=r.forcex; rb.forcey=r.forcey; rb.forcez=r.forcez;
            rb.torque=r.torque; rb.tqx=r.tqx; rb.tqy=r.tqy; rb.tqz=r.tqz;
            rb.ox=r.ox; rb.oy=r.oy; rb.rotates=r.rotates!=0;
            rb.setOrientation(r.Rm);
        }
    }
    std::fclose(f);
    *step = static_cast<long>(h.step);
    *t = h.t;
    std::printf("[ckpt] restored step %llu  t=%.6e  N=%llu  %u bodies\n",
                (unsigned long long)h.step, h.t, (unsigned long long)h.N,
                h.nBodies);
    std::fflush(stdout);
    return true;
}

}  // namespace ckpt
