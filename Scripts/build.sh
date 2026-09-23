#!/usr/bin/env bash
# =============================================================================
# build.sh - Configure and build.
#
#   ./Scripts/build.sh          CPU backend
#   ./Scripts/build.sh cuda     GPU backend
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OPTS=""
if [ "${1:-}" = "cuda" ]; then
    OPTS="-DALEBGK_CUDA=ON"
    echo "Building the CUDA backend."
else
    echo "Building the CPU backend. Pass 'cuda' for the GPU backend."
fi

cmake -B build $OPTS
cmake --build build --config Release -j
echo
echo "Built: build/bin/alebgk    (run with no arguments to list the cases)"
