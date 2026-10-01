#!/usr/bin/env bash
# =============================================================================
# build.sh - Configure and build on Linux.
#
#   ./Scripts/build.sh              CPU backend    -> build/
#   ./Scripts/build.sh cuda         GPU backend    -> build_cuda/
#   ./Scripts/build.sh mpi          MPI backend    -> build_mpi/
#   ./Scripts/build.sh mpi intel    MPI, Intel oneAPI
#   ./Scripts/build.sh mpi system   MPI, system OpenMPI/MPICH
#
# "mpi" with no second argument uses whichever MPI wrapper is already on PATH
# and falls back to Intel oneAPI if one is installed but not yet loaded.
#
# Each backend builds in its own directory.  CMake options are sticky once
# cached, so sharing one directory would silently keep a previous backend
# enabled.
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILDDIR="build"
BACKEND="serial CPU"
IS_MPI=0
OPTS="-DALEBGK_CUDA=OFF -DALEBGK_MPI=OFF -DALEBGK_INTEL_MPI=OFF"

# oneAPI environment scripts reference unset variables, so -u must be relaxed
# across the source.
load_oneapi() {
    local vars="$1"
    set +u
    # shellcheck disable=SC1090
    source "$vars" >/dev/null 2>&1 || true
    set -u
}

find_oneapi() {
    local d
    for d in /opt/intel/oneapi/mpi/latest \
             "${HOME}/intel/oneapi/mpi/latest"; do
        if [ -r "$d/env/vars.sh" ]; then
            echo "$d"
            return 0
        fi
    done
    return 1
}

case "${1:-}" in
    "")
        ;;
    cuda)
        BUILDDIR="build_cuda"
        BACKEND="CUDA"
        OPTS="-DALEBGK_CUDA=ON -DALEBGK_MPI=OFF -DALEBGK_INTEL_MPI=OFF"
        ;;
    mpi)
        BUILDDIR="build_mpi"
        IS_MPI=1
        OPTS="-DALEBGK_MPI=ON -DALEBGK_INTEL_MPI=OFF -DALEBGK_CUDA=OFF"
        MODE="${2:-auto}"

        if [ "$MODE" = "intel" ] || { [ "$MODE" = "auto" ] && ! command -v mpicxx >/dev/null 2>&1 && ! command -v mpic++ >/dev/null 2>&1; }; then
            if [ -z "${I_MPI_ROOT:-}" ]; then
                if ONEAPI="$(find_oneapi)"; then
                    echo "Loading Intel oneAPI from ${ONEAPI}"
                    load_oneapi "${ONEAPI}/env/vars.sh"
                elif [ "$MODE" = "intel" ]; then
                    echo "Intel MPI was requested but oneAPI was not found." >&2
                    echo "Source its vars.sh first, or use: ./Scripts/build.sh mpi system" >&2
                    exit 1
                fi
            fi
        fi

        # CMake locates MPI through the compiler wrapper on PATH; the direct
        # Intel link path in CMakeLists applies to Windows only, so
        # ALEBGK_INTEL_MPI stays off here for Intel too.
        if ! command -v mpicxx >/dev/null 2>&1 && ! command -v mpic++ >/dev/null 2>&1; then
            echo "No MPI compiler wrapper found on PATH." >&2
            echo "Install one, e.g.  sudo apt install libopenmpi-dev openmpi-bin" >&2
            echo "or source the Intel oneAPI vars.sh, then re-run." >&2
            exit 1
        fi

        MPICXX="$(command -v mpicxx 2>/dev/null || command -v mpic++)"
        BACKEND="MPI"
        echo "MPI wrapper: ${MPICXX}"
        if [ -n "${I_MPI_ROOT:-}" ]; then
            BACKEND="MPI (Intel MPI)"
        elif "$MPICXX" --showme:version >/dev/null 2>&1; then
            BACKEND="MPI (Open MPI)"
        fi
        ;;
    *)
        echo "Unknown backend '$1'.  Use: cuda | mpi | mpi intel | mpi system" >&2
        exit 1
        ;;
esac

echo
echo "=== ${BACKEND} -> ${BUILDDIR}/ ==="
cmake -B "$BUILDDIR" $OPTS
cmake --build "$BUILDDIR" -j"$(nproc 2>/dev/null || echo 4)"

echo
echo "Built: ${BUILDDIR}/bin/alebgk"
if [ "$IS_MPI" -eq 1 ]; then
    cat <<EOF

Launch with the mpirun that matches the wrapper above - Open MPI and Intel MPI
are not interchangeable, and the wrong launcher yields N independent 1-rank
runs rather than an error.

  mpirun -n 8 ./${BUILDDIR}/bin/alebgk cavity2d --Nx 1200 --Nv 20

Add --estimate 0 to project the full run from the first steps and exit.
EOF
else
    echo "Run with no arguments to list the cases."
fi
