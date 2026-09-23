#pragma once
// =============================================================================
// Output.hpp — VTK output + CSV diagnostics for the serial CPU BGK solver
// =============================================================================
#include "Types.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>

// -----------------------------------------------------------------------------
// writeVTK — ASCII VTK unstructured grid (one file per rank per step)
// -----------------------------------------------------------------------------
inline void writeVTK(const std::vector<Particle> &P,
                     const std::string &outDir, int step, double time)
{
    std::ostringstream fname;
    fname << outDir << "/output_"
          << std::setw(6) << std::setfill('0') << step << ".vtk";
    std::ofstream f(fname.str());
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname.str().c_str()); return; }

    int N = static_cast<int>(P.size());
    f << "# vtk DataFile Version 3.0\n";
    f << "ALEBGK t=" << std::scientific << time << "\n";
    f << "ASCII\nDATASET UNSTRUCTURED_GRID\n";
    f << "POINTS " << N << " double\n";
    for (const auto &p : P) f << p.x << " " << p.y << " " << p.z << "\n";
    f << "CELLS 0 0\nCELL_TYPES 0\nPOINT_DATA " << N << "\n";

    // Helper: write a named scalar field
    auto scalar = [&](const char *name, auto fn) {
        f << "SCALARS " << name << " double 1\nLOOKUP_TABLE default\n";
        for (const auto &p : P) f << fn(p) << "\n";
    };

    scalar("rho",      [](const Particle &p) { return p.rho; });
    scalar("T",        [](const Particle &p) { return p.T; });
    scalar("p",        [](const Particle &p) { return p.p; });
    scalar("Nv_local", [](const Particle &p) { return static_cast<double>(p.Nv_local); });
    scalar("moodFlag", [](const Particle &p) { return static_cast<double>(p.moodFlag); });
    scalar("boundary", [](const Particle &p) { return static_cast<double>(p.boundary); });

    f << "VECTORS velocity double\n";
    for (const auto &p : P) f << p.ux << " " << p.uy << " " << p.uz << "\n";

    f.close();
}

// -----------------------------------------------------------------------------
// Diagnostics — aggregate statistics over interior (non-boundary) particles
// -----------------------------------------------------------------------------
struct Diagnostics {
    double rho_avg  = 0.0;
    double KE       = 0.0;
    double T_avg    = 0.0;
    double p_avg    = 0.0;
    int    moodCount = 0;
    int    n         = 0;
};

inline Diagnostics computeDiagnostics(const std::vector<Particle> &P)
{
    Diagnostics d;
    for (const auto &p : P) {
        if (p.boundary) continue;
        d.rho_avg += p.rho;
        d.KE      += 0.5 * p.rho * (p.ux * p.ux + p.uy * p.uy + p.uz * p.uz);
        d.T_avg   += p.T;
        d.p_avg   += p.p;
        d.moodCount += static_cast<int>(p.moodFlag);
        d.n++;
    }
    if (d.n > 0) {
        double n = d.n;
        d.rho_avg /= n;
        d.KE      /= n;
        d.T_avg   /= n;
        d.p_avg   /= n;
    }
    return d;
}

// -----------------------------------------------------------------------------
// writeTimeseries — append one CSV row (write header on first call)
// -----------------------------------------------------------------------------
inline void writeTimeseries(const std::string &path, double t,
                            const Diagnostics &d)
{
    static bool hdr = false;
    std::ofstream f(path, hdr ? std::ios::app : std::ios::out);
    if (!hdr) {
        f << "t,rho_avg,KE,T_avg,p_avg,mood_count\n";
        hdr = true;
    }
    f << std::scientific
      << t          << ","
      << d.rho_avg  << ","
      << d.KE       << ","
      << d.T_avg    << ","
      << d.p_avg    << ","
      << d.moodCount << "\n";
}

// -----------------------------------------------------------------------------
// printStep — compact one-line summary to stdout
// -----------------------------------------------------------------------------
inline void printStep(int step, double t, double dt, const Diagnostics &d)
{
    printf("Step%6d|t=%9.3e|dt=%7.2e|rho=%7.4f|T=%7.4f|KE=%7.3e|MOOD=%3d\n",
           step, t, dt, d.rho_avg, d.T_avg, d.KE, d.moodCount);
}
