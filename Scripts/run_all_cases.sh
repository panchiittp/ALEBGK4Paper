#!/usr/bin/env bash
# =============================================================================
# run_all_cases.sh - Run the six cavity cases.
#
#   ./Scripts/run_all_cases.sh            short runs, for checking the build
#   ./Scripts/run_all_cases.sh production full-length runs
#
# The short runs take a couple of minutes in total and exercise every code
# path. The production runs are the configurations the results were produced
# at; the 3-D body cases take hours to days, so they checkpoint.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/bin/alebgk"
[ -x "$BIN" ] || BIN="$ROOT/build/bin/Release/alebgk.exe"
if [ ! -x "$BIN" ]; then
    echo "Solver not built. Run Scripts/build.sh first." >&2
    exit 1
fi

MODE="${1:-smoke}"
CASES=(cavity2d cavity3d cavity2d-square cavity2d-circle
       cavity3d-sphere cavity3d-cube)

for c in "${CASES[@]}"; do
    echo
    echo "=== $c ==="
    if [ "$MODE" = "production" ]; then
        # Case defaults are the production configuration. The 3-D body cases
        # checkpoint every 2000 steps: a run interrupted partway resumes with
        # ALEBGK_RESTART=<outdir>/restart.ckpt rather than starting over.
        case "$c" in
            cavity3d-sphere|cavity3d-cube)
                ALEBGK_CHECKPOINT_EVERY=2000 "$BIN" "$c" ;;
            *)  "$BIN" "$c" ;;
        esac
    else
        # Coarse grid, a handful of steps: enough to confirm the case builds,
        # embeds its body and takes a step.
        case "$c" in
            cavity2d)        "$BIN" "$c" --Nx 40 --tfinal 1e-12  ;;
            cavity3d)        "$BIN" "$c" --Nx 15 --tfinal 1e-10  ;;
            cavity2d-*)      "$BIN" "$c" --Nx 30 --tfinal 1.5e-10 ;;
            cavity3d-*)      "$BIN" "$c" --Nx 15 --tfinal 1.5e-10 ;;
        esac
    fi
done

echo
echo "All six cases completed. Output is under output/."
