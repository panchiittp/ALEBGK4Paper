#!/usr/bin/env bash
# =============================================================================
# run_all_cases.sh - Run the six cavity cases, or collect cost estimates.
#
#   ./Scripts/run_all_cases.sh                 short runs, for checking the build
#   ./Scripts/run_all_cases.sh production      full-length runs
#   ./Scripts/run_all_cases.sh estimate        resolution sweep, serial
#   ./Scripts/run_all_cases.sh estimate 16     the same sweep on 16 MPI ranks
#
# The short runs take a couple of minutes in total and exercise every code
# path. The production runs are the configurations the results were produced
# at; the 3-D body cases take hours to days, so they checkpoint.
#
# estimate mode advances two time steps per configuration and projects the
# whole run from the second (the first carries warm-up), then stops. A sweep
# therefore costs minutes rather than months, and prints a summary table of
# projected wall times at the end.
#
#   Sweep:  cavity2d  Nx = 400, 600, 800, 1200, 1600
#           cavity3d  Nx = 20, 40, 60, 80
#
# Environment overrides:
#   NV2D=20   NV3D=20     velocity points per direction
#   NX2D="400 800"        override the 2-D resolution list
#   NX3D="20 40"          override the 3-D resolution list
#   SLAB=1                require the slab decomposition (passes --slab), so a
#                         configuration that would silently replicate fails
#                         instead; keeps every point of a sweep in one mode
#   PROFILE=1             per-stage step profile instead of a cost estimate:
#                         runs ten real time steps with ALEBGK_MPI_PROFILE=1
#                         and prints the breakdown. --estimate is not used,
#                         since it stops at step 2 where warm-up dominates
#   MPIRUN=/path/to/mpirun
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-smoke}"
NP="${2:-}"

# --- locate the solver ------------------------------------------------------
# MPI runs need the MPI build; everything else prefers the plain one.
find_bin() {
    local d
    for d in "$@"; do
        [ -x "$ROOT/$d/bin/alebgk" ] && { echo "$ROOT/$d/bin/alebgk"; return 0; }
        [ -x "$ROOT/$d/bin/Release/alebgk.exe" ] && { echo "$ROOT/$d/bin/Release/alebgk.exe"; return 0; }
    done
    return 1
}

if [ -n "$NP" ]; then
    BIN="$(find_bin build_mpi build)" || {
        echo "MPI solver not built. Run: ./Scripts/build.sh mpi" >&2; exit 1; }
else
    BIN="$(find_bin build build_mpi)" || {
        echo "Solver not built. Run: ./Scripts/build.sh" >&2; exit 1; }
fi

# =============================== ESTIMATE ==================================
if [ "$MODE" = "estimate" ]; then

    NV2D="${NV2D:-20}"
    NV3D="${NV3D:-20}"
    # ${VAR-default}, not ${VAR:-default}: an explicitly empty NX2D="" then
    # skips that dimension instead of falling back to the full list.
    read -r -a NX2D <<< "${NX2D-400 600 800 1200 1600}"
    read -r -a NX3D <<< "${NX3D-20 40 60 80}"

    # Either empty or exactly "--slab", so an unquoted expansion is safe.
    SLABFLAG=""
    [ -n "${SLAB:-}" ] && SLABFLAG="--slab"

    # PROFILE=1 swaps the cost estimate for a short real run with per-stage
    # timing. Ten steps is enough for the means to settle and still costs
    # seconds at the resolutions a sweep uses.
    if [ -n "${PROFILE:-}" ]; then
        export ALEBGK_MPI_PROFILE=1
        echo "Profiling mode: ten time steps per configuration, --estimate off"
    fi

    LAUNCH=()
    if [ -n "$NP" ]; then
        case "$NP" in
            ''|*[!0-9]*) echo "Rank count must be a positive integer, got '$NP'." >&2; exit 1 ;;
        esac
        MPIRUN="${MPIRUN:-}"
        if [ -z "$MPIRUN" ]; then
            MPIRUN="$(command -v mpirun 2>/dev/null || command -v mpiexec 2>/dev/null || true)"
        fi
        [ -n "$MPIRUN" ] || { echo "No mpirun/mpiexec found. Set MPIRUN=<path>." >&2; exit 1; }
        LAUNCH=("$MPIRUN" -n "$NP")
        echo "Launcher: $MPIRUN -n $NP"
    fi

    # Per-rank footprint of the replicated cloud, in GiB.  g and the transport
    # buffer, 2*Nv^2 doubles per particle in 2-D and Nv^3 in 3-D.
    mem_gib() {   # dim nx nv
        awk -v d="$1" -v nx="$2" -v nv="$3" 'BEGIN{
            if (d == 2) b = 32.0*nx*nx*nv*nv;
            else        b = 16.0*nx*nx*nx*nv*nv*nv;
            printf "%.2f", b/1073741824.0 }'
    }

    LOGDIR="$ROOT/output/estimates"
    mkdir -p "$LOGDIR"
    STAMP="$(date +%Y%m%d_%H%M%S)"
    SUMMARY="$LOGDIR/summary_${STAMP}.txt"
    : > "$SUMMARY"

    run_one() {   # case dim nx nv
        local case="$1" dim="$2" nx="$3" nv="$4"
        local tag mem log line
        tag="${case}_Nx${nx}_Nv${nv}"
        mem="$(mem_gib "$dim" "$nx" "$nv")"
        log="$LOGDIR/${tag}${NP:+_np$NP}_${STAMP}.log"

        echo
        echo "--- $case  Nx=$nx  Nv=$nv  (whole cloud ${mem} GiB) ---"
        if [ -n "$NP" ]; then
            # Slab mode needs two halo widths of owned rows per rank, so the
            # smallest slab must be at least 2*4 = 8 rows. Below that the code
            # falls back to replicating the whole cloud on every rank, and the
            # startup check in main.cpp sizes one rank against the whole node,
            # so such a run passes it and is then OOM-killed.
            awk -v m="$mem" -v p="$NP" -v nx="$nx" 'BEGIN{
                minrows = int(nx / p);
                if (minrows >= 8) {
                    per = m * (1.0 / p + 8.0 / nx);
                    printf "    slab mode: %d rows/rank, %.2f GiB per rank, %.1f GiB total\n",
                           minrows, per, per * p;
                } else {
                    printf "    SLAB INELIGIBLE: %d rows/rank, 8 required -> replicated fallback\n",
                           minrows;
                    printf "    %.1f GiB across %d ranks on one node; expect an OOM kill. Use -n %d or fewer.\n",
                           m * p, p, int(nx / 8);
                }
            }'
        fi

        # Ten steps at the case default dt: 1e-13 in 2-D, 1e-11 in 3-D.
        if [ -n "${PROFILE:-}" ]; then
            [ "$dim" = 3 ] && modeargs="--tfinal 1e-10" \
                           || modeargs="--tfinal 1e-12"
        else
            modeargs="--estimate 0"
        fi

        if "${LAUNCH[@]}" "$BIN" "$case" --Nx "$nx" --Nv "$nv" \
               $SLABFLAG $modeargs > "$log" 2>&1; then
            # Report the mode the solver actually chose, not the one predicted
            # above, so a sweep that silently changed mode is visible.
            mode="$(grep -m1 'Decomp:' "$log" || true)"
            [ -n "$mode" ] && echo "   ${mode#*Decomp: }"

            if [ -n "${PROFILE:-}" ]; then
                # Echo the profile table and record its headline rows.
                sed -n '/=== CPU step profile/,/imbalance indicator/p' "$log" \
                    | sed 's/^/   /'
                exch="$(awk '/Rank exchange/{
                           printf "exchange %.2f ms (%.2f%% of step)", $3, $5;
                           exit}' "$log")"
                printf '%-26s Nv=%-3s %s\n' "$tag" "$nv" \
                    "${exch:-no profile produced}" >> "$SUMMARY"
                return 0
            fi

            line="$(grep -m1 '\[ESTIMATE\]' "$log" || true)"
            if [ -n "$line" ]; then
                printf '%-26s Nv=%-3s %s\n' "$tag" "$nv" \
                    "${line#*-> }" >> "$SUMMARY"
                echo "    ${line#*\[ESTIMATE\] }"
            else
                printf '%-26s Nv=%-3s no estimate line (see %s)\n' \
                    "$tag" "$nv" "$(basename "$log")" >> "$SUMMARY"
                echo "    no [ESTIMATE] line produced; see $log"
            fi
        else
            # Non-zero exit: out of memory, ineligible decomposition, or a
            # launcher mismatch. Keep sweeping rather than aborting.
            printf '%-26s Nv=%-3s FAILED (see %s)\n' \
                "$tag" "$nv" "$(basename "$log")" >> "$SUMMARY"
            echo "    FAILED - see $log"
            tail -n 3 "$log" | sed 's/^/      | /' || true
        fi
    }

    echo "=============================================="
    echo "  Estimate sweep (two steps per configuration)"
    echo "  Solver: $BIN"
    [ -n "$NP" ] && echo "  Ranks:  $NP"
    echo "  Logs:   $LOGDIR"
    echo "=============================================="

    for nx in "${NX2D[@]}"; do run_one cavity2d 2 "$nx" "$NV2D"; done
    for nx in "${NX3D[@]}"; do run_one cavity3d 3 "$nx" "$NV3D"; done

    echo
    echo "================= SUMMARY ===================="
    cat "$SUMMARY"
    echo "=============================================="
    echo "Saved: $SUMMARY"
    exit 0
fi

# ========================= SMOKE / PRODUCTION ==============================
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
