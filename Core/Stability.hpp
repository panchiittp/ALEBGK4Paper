#pragma once
// =============================================================================
// Stability.hpp - abort a run once the solution has left physical bounds.
//
// A diverged kinetic solve does not usually stop by itself. It freezes into a
// non-physical state and keeps stepping, which costs the whole remaining
// wall-clock for nothing: an 80,000-step run observed here blew up at step
// 4,500 and then produced bit-identical diagnostics for the remaining 74,500
// steps, wasting about twelve GPU-hours.
//
// The bounds are deliberately chosen against that observed failure rather than
// picked for neatness. Its sequence was
//
//     step 4000   rho/rho0 = 1.00   T/Tw = 1.001   KE/KE0 = 1
//     step 4500   rho/rho0 = 0.85   T/Tw = 1.449   KE/KE0 = 3.6e4     <- first
//     step 5000   rho/rho0 = 67.8   T/Tw = 1.314   KE/KE0 = 1.3e8
//
// so at the FIRST bad step the density was still well inside any sane band and
// only the temperature and the kinetic energy had moved. A density-only check,
// or a temperature band as loose as 1.5*Tw, would both have missed it and
// caught the run only 500 steps later. Hence a tight temperature band and a
// kinetic-energy ceiling, with density as the third, slower witness.
//
// Every bound is a multiple of a reference the case itself provides: the wall
// temperature, the initial density, and the kinetic energy at the first
// diagnostic. Overridable by environment variable, and switchable off with
// ALEBGK_NO_GUARD=1 for deliberate stress tests.
// =============================================================================
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "Output.hpp"      // Diagnostics

namespace stability
{

inline double envFac(const char *name, double fallback)
{
    const char *e = std::getenv(name);
    if (!e || !*e) return fallback;
    const double v = std::atof(e);
    return (v > 0.0) ? v : fallback;
}

class Guard
{
public:
    Guard(double wallTemp, double rhoInitial)
        : Tw_(wallTemp), rho0_(rhoInitial), keRef_(0.0), haveKe_(false),
          enabled_(std::getenv("ALEBGK_NO_GUARD") == nullptr),
          tLo_(envFac("ALEBGK_GUARD_TMIN",   0.60)),
          tHi_(envFac("ALEBGK_GUARD_TMAX",   1.30)),
          rLo_(envFac("ALEBGK_GUARD_RHOMIN", 0.20)),
          rHi_(envFac("ALEBGK_GUARD_RHOMAX", 5.00)),
          keHi_(envFac("ALEBGK_GUARD_KEMAX", 100.0))
    {}

    // Returns true when the run should stop. Reports the reason once.
    bool diverged(int step, double t, const Diagnostics &d)
    {
        if (!enabled_) return false;

        // Non-finite is unconditional: no tolerance can make it acceptable.
        if (!std::isfinite(d.rho_avg) || !std::isfinite(d.T_avg) ||
            !std::isfinite(d.KE))
            return report(step, t, d, "a diagnostic is NaN or infinite");

        if (d.T_avg < tLo_ * Tw_)
            return report(step, t, d, "mean temperature below %.4g K (%.2f x Tw)",
                          tLo_ * Tw_, tLo_);
        if (d.T_avg > tHi_ * Tw_)
            return report(step, t, d, "mean temperature above %.4g K (%.2f x Tw)",
                          tHi_ * Tw_, tHi_);

        if (rho0_ > 0.0)
        {
            if (d.rho_avg < rLo_ * rho0_)
                return report(step, t, d, "mean density below %.4g (%.2f x rho0)",
                              rLo_ * rho0_, rLo_);
            if (d.rho_avg > rHi_ * rho0_)
                return report(step, t, d, "mean density above %.4g (%.2f x rho0)",
                              rHi_ * rho0_, rHi_);
        }

        // The kinetic energy reference is taken from the first diagnostic that
        // carries a usable value, not from the initial state: the gas and the
        // body both start at rest, so an initial KE of zero would make any
        // multiplicative ceiling meaningless.
        if (!haveKe_ && d.KE > 0.0) { keRef_ = d.KE; haveKe_ = true; }
        if (haveKe_ && d.KE > keHi_ * keRef_)
            return report(step, t, d, "kinetic energy %.3g x its early value "
                          "(ceiling %.0f x)", d.KE / keRef_, keHi_);

        return false;
    }

private:
    template <typename... A>
    bool report(int step, double t, const Diagnostics &d,
                const char *why, A... args)
    {
        std::printf("\n*** STOPPING: the solution has left physical bounds.\n");
        std::printf("***   step %d, t = %.4e s\n", step, t);
        std::printf("***   ");
        std::printf(why, args...);
        std::printf("\n");
        std::printf("***   rho = %.6g (rho0 = %.6g), T = %.6g K (Tw = %.6g K), "
                    "KE = %.6g\n", d.rho_avg, rho0_, d.T_avg, Tw_, d.KE);
        std::printf("***\n");
        std::printf("*** A diverged solve does not recover; continuing would\n");
        std::printf("*** burn the remaining wall-clock on a dead state. Restart\n");
        std::printf("*** from the last good checkpoint, or relax the bounds with\n");
        std::printf("*** ALEBGK_GUARD_TMAX / _TMIN / _RHOMAX / _RHOMIN / _KEMAX,\n");
        std::printf("*** or disable the check entirely with ALEBGK_NO_GUARD=1.\n\n");
        std::fflush(stdout);
        return true;
    }

    double Tw_, rho0_, keRef_;
    bool   haveKe_, enabled_;
    double tLo_, tHi_, rLo_, rHi_, keHi_;
};

}  // namespace stability
