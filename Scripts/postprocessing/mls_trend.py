#!/usr/bin/env python3
"""MLS limiter activity over a run, from the solver's per-report log lines.

The 3-D moving-body cavity has two distinct failures. The first -- a collapsed
Cholesky pivot feeding 1e-150 into two divisions until it overflowed -- is
fixed. The second is an exponential growth of g at body-surface particles that
leaves the moment matrix perfectly well conditioned, so no pivot test sees it.
The hypothesis is MLS EXTRAPOLATION: a surface particle's foot-point is
displaced by the wall velocity into the body's empty interior, the stencil goes
one-sided, and a least-squares fit outside its own data amplifies.

mlsExt counts particles whose reconstruction landed outside the range of the
neighbour values it was built from -- a discrete maximum principle, which is
the one test that separates "one-sided stencil" from "ill-conditioned stencil".

The question this answers: does mlsExt GROW into the onset (supporting the
hypothesis), or sit flat while the blow-up happens anyway (refuting it)?

Note the counter had to be validated before its output meant anything: an
earlier version accumulated per-velocity-node bit values into a shared double,
which aliased, and a later one never read the device flag at all. Run with
ALEBGK_MLS_FORCEFLAG=1 to self-test -- mlsExt must then equal N exactly.

Usage: mls_trend.py <run.log> [more.log ...]
"""
import re
import sys

STEP = re.compile(r"Step\s+(\d+)\|t=([\d.e+-]+)\|dt=[\d.e+-]+\|"
                  r"rho=\s*([-\w.()+]+)\|T=\s*([-\w.()+]+)\|KE=([-\w.()+]+)")
LIM = re.compile(r"positivity=(\d+)\s+NAD=(\d+)\s+mlsDeg=(\d+)\s+"
                 r"mlsExt=(\d+)(?:\s+ovrL1=([-\w.+]+)\s+ovrMax=([-\w.+]+))?"
                 r"\s+\(of N=(\d+)\)")

if len(sys.argv) < 2:
    sys.exit(__doc__)

for path in sys.argv[1:]:
    cur = None
    rows = []
    trip = None
    try:
        fh = open(path, errors="replace")
    except OSError as e:
        print("%s: %s" % (path, e))
        continue
    for line in fh:
        m = STEP.match(line)
        if m:
            cur = m.groups()
            continue
        m = LIM.search(line)
        if m and cur:
            rows.append(cur + m.groups())
            continue
        if trip is None and ("BODYMAG" in line or "BODYTRACE" in line
                             or "NANSCAN" in line):
            trip = line.strip()

    print("=" * 78)
    print("%s   %d reports" % (path, len(rows)))
    if trip:
        print("first trip: %s" % trip[:110])
    if not rows:
        print("no limiter reports found")
        continue

    print()
    print("   step        t        rho        T          KE      "
          "pos   NAD  mlsDeg  mlsExt   %ofN     ovrL1      ovrMax   x/prev")
    # Show the first few, then thin the middle, then every report near the end
    # -- the onset is what matters and it is always at the end.
    n = len(rows)
    keep = set(range(min(3, n))) | set(range(max(0, n - 12), n))
    keep |= {i for i in range(n) if i % max(1, n // 18) == 0}
    def ov(r):
        try:
            return float(r[9]) if r[9] else 0.0
        except (TypeError, ValueError):
            return 0.0

    for i in sorted(keep):
        r = rows[i]
        s_, t, rho, T, ke, pos, nad, deg, ext = r[0], r[1], r[2], r[3], r[4],             r[5], r[6], r[7], r[8]
        N = r[11]
        pct = 100.0 * int(ext) / int(N)
        o = ov(r)
        # Per-report growth factor of the overshoot amplitude. A LINEAR
        # instability seeded by a fixed set of sites shows a flat count and a
        # geometrically growing amplitude -- that ratio is the whole point.
        prev = ov(rows[i - 1]) if i > 0 else 0.0
        fac = (o / prev) if prev > 0 else float("nan")
        print("%7s  %.3e  %8s %9s %10s %5s %5s %6s %7s  %6.2f%%  %9.3e %9.3e %7.3f"
              % (s_, float(t), rho, T, ke, pos, nad, deg, ext, pct,
                 o, float(r[10]) if r[10] else 0.0, fac))

    ext = [int(r[8]) for r in rows]
    ovr = [ov(r) for r in rows]
    if any(ovr):
        print()
        print("ovrL1: first %.3e  last %.3e  ratio %.1f over %d reports"
              % (ovr[0], ovr[-1],
                 (ovr[-1] / ovr[0]) if ovr[0] else float("nan"), len(ovr)))
    print()
    print("mlsExt: min %d  max %d  first nonzero at report %s  last %d"
          % (min(ext), max(ext),
             next((rows[i][0] for i, v in enumerate(ext) if v), "never"),
             ext[-1]))
    tail = ext[-10:]
    head = ext[:10]
    print("mean over first 10 reports %.1f ; over last 10 %.1f"
          % (sum(head) / len(head), sum(tail) / len(tail)))
