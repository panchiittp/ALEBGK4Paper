#!/usr/bin/env python3
"""body_growth.py -- locate the onset of the rigid-body coupling instability.

The 3-D moving-body cavity run ends with the body state at 1e283 and then NaN.
The finiteness tripwire in moveRigidBodies reported that as a stage-3 failure,
but the state going INTO stage 3 was already astronomical: the body had been
diverging exponentially for ~1660 steps.  isfinite() cannot see that, and
bodies.csv samples every saveEvery steps, so the whole blow-up fell between two
log rows.

ALEBGK_BODY_LOG=1 writes body_state.csv every step.  This script answers the
two questions that instrumentation was added for:

  1. WHEN does |w| stop tracking the flow and start growing geometrically?
     Reported as the first step whose local growth rate exceeds `--thresh`
     and stays above it -- not the first step above some magnitude, which
     only finds the blow-up long after it started.

  2. Do the two Heun loads diverge from each other?  In a staggered
     fluid-structure coupling instability the corrector load L1 (evaluated at
     the predicted state) grows away from the predictor load L0 with
     alternating sign, so |F1/F0| leaving 1 -- and especially going negative --
     is the fingerprint.  If L1/L0 stays at 1 while |w| explodes, the coupling
     is NOT the mechanism and the loads themselves are the suspect.

Usage:  python postprocessing/body_growth.py [body_state.csv] [--thresh 1.02]
"""
import sys
import math

path = "body_state.csv"
thresh = 1.02          # per-step growth factor counted as "growing"
run = 20               # consecutive steps above thresh to call it onset
start = 2000           # skip the physical spin-up transient (see below)
args = [a for a in sys.argv[1:]]
skip = False
for i, a in enumerate(args):
    if skip:
        skip = False           # this token is a flag's value, not the path
        continue
    if a == "--thresh":
        thresh = float(args[i + 1]); skip = True
    elif a == "--run":
        run = int(args[i + 1]); skip = True
    elif a == "--start":
        start = int(args[i + 1]); skip = True
    elif not a.startswith("--"):
        path = a

rows = []
with open(path) as f:
    hdr = f.readline().rstrip("\n").split(",")
    col = {n: i for i, n in enumerate(hdr)}
    for line in f:
        p = line.rstrip("\n").split(",")
        if len(p) != len(hdr):
            continue           # torn final line: the run is still writing
        try:
            rows.append([float(v) for v in p])
        except ValueError:
            continue

if not rows:
    sys.exit("no complete rows in " + path)

g = lambda r, n: r[col[n]]
print("rows %d  steps %d..%d  t %.4e..%.4e"
      % (len(rows), int(g(rows[0], "step")), int(g(rows[-1], "step")),
         g(rows[0], "t"), g(rows[-1], "t")))

# --- 1. growth-rate scan on |w| ------------------------------------------
# The body starts at rest and spins up to its equilibrium |w| ~ 5e6 rad/s
# over the first few hundred steps, with per-step ratios far above any
# instability threshold. That is the physical transient, so the scan starts
# after it -- otherwise the detector just reports step 5 every time.
onset = None
streak = 0
i0 = 1
while i0 < len(rows) and g(rows[i0], "step") < start:
    i0 += 1
for i in range(i0, len(rows)):
    w0, w1 = g(rows[i - 1], "wmag"), g(rows[i], "wmag")
    if w0 <= 0.0 or not math.isfinite(w1):
        streak = 0
        continue
    if w1 / w0 > thresh:
        if streak == 0:
            cand = i
        streak += 1
        if streak >= run and onset is None:
            onset = cand
    else:
        streak = 0

print()
if onset is None:
    print("no sustained growth above %.3f/step over %d consecutive steps"
          % (thresh, run))
else:
    r = rows[onset]
    print("ONSET  step %d  t=%.6e  |w|=%.4e  |v|=%.4e"
          % (int(g(r, "step")), g(r, "t"), g(r, "wmag"), g(r, "vmag")))
    print("       centre (%.1f, %.1f, %.1f) nm"
          % (g(r, "cx") * 1e9, g(r, "cy") * 1e9, g(r, "cz") * 1e9))

# --- 2. sampled trace, with the Heun load ratio ---------------------------
print()
print("  step        t         |v|         |w|      w1/w0    |F1|/|F0|   F1.F0/|F0|^2")
n = len(rows)
idx = sorted(set([0] + [n * k // 40 for k in range(1, 40)] + [n - 1]))
for i in idx:
    r = rows[i]
    if i > 0:
        w0 = g(rows[i - 1], "wmag")
        ratio = g(r, "wmag") / w0 if w0 > 0 else float("nan")
    else:
        ratio = float("nan")
    F0 = [g(r, "Fx0"), g(r, "Fy0"), g(r, "Fz0")]
    F1 = [g(r, "Fx1"), g(r, "Fy1"), g(r, "Fz1")]
    n0 = math.sqrt(sum(c * c for c in F0))
    n1 = math.sqrt(sum(c * c for c in F1))
    dot = sum(a * b for a, b in zip(F0, F1))
    # projection of L1 onto L0: 1 means the corrector agrees with the
    # predictor, negative means it overshot and reversed -- divergence.
    proj = dot / (n0 * n0) if n0 > 0 else float("nan")
    print("%7d  %.4e  %.3e  %.3e  %8.4f  %10.4f  %12.4f"
          % (int(g(r, "step")), g(r, "t"), g(r, "vmag"), g(r, "wmag"),
             ratio, (n1 / n0 if n0 > 0 else float("nan")), proj))
