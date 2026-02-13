#!/usr/bin/env bash
# =================================================================
# Astro-Tools Benchmark Laboratory — Build & Run Script
#
# Usage:
#   ./run.sh              # Build all + run all experiments (N=1000)
#   ./run.sh build        # Build only
#   ./run.sh run          # Run only (assumes built)
#   ./run.sh run 5000     # Run with custom N
# =================================================================

set -euo pipefail
cd "$(dirname "$0")"

LAB_ROOT="$(pwd)"

# ---- Colours (safe for non-TTY) ----
if [ -t 1 ]; then
    BOLD="\033[1m"
    GREEN="\033[32m"
    YELLOW="\033[33m"
    RESET="\033[0m"
else
    BOLD="" GREEN="" YELLOW="" RESET=""
fi

log()  { echo -e "${GREEN}▸${RESET} $*"; }
warn() { echo -e "${YELLOW}⚠${RESET} $*"; }

# ---- Submodules ----
init_submodules_if_needed() {
    if [ "${FORCE_SUBMODULE_SYNC:-0}" = "1" ]; then
        warn "FORCE_SUBMODULE_SYNC=1 set: syncing submodules to pinned commits."
        git submodule update --init --recursive
        return
    fi

    if git submodule status --recursive | grep -q '^-'; then
        log "Initializing missing git submodules..."
        git submodule update --init --recursive
    else
        warn "Submodules already initialized; skipping sync to preserve local checkouts."
        warn "Set FORCE_SUBMODULE_SYNC=1 to reset submodules to pinned commits."
    fi
}

# ---- Build ----
build_all() {
    init_submodules_if_needed

    log "Building ERFA adapter (C)..."
    make -C pipeline/adapters/erfa_adapter -j"$(nproc)" 2>&1 | tail -3

    log "Building libnova adapter (C)..."
    make -C pipeline/adapters/libnova_adapter -j"$(nproc)" 2>&1 | tail -3

    log "Building Siderust adapter (Rust, release)..."
    (cd pipeline/adapters/siderust_adapter && cargo build --release 2>&1 | tail -3)

    log "Setting up Python virtual environment..."
    if [ ! -d .venv ]; then
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    pip install -q pyerfa numpy

    log "Build complete ✓"
}

# ---- Run ----
run_all() {
    local N="${1:-1000}"
    source .venv/bin/activate

    log "Running all experiments (N=$N, seed=42)..."
    python3 pipeline/orchestrator.py --experiment all --n "$N" --seed 42

    log "Done ✓  Results in results/"
}

# ---- Main ----
case "${1:-all}" in
    build)
        build_all
        ;;
    run)
        run_all "${2:-1000}"
        ;;
    all|"")
        build_all
        run_all "${2:-1000}"
        ;;
    *)
        echo "Usage: $0 [build|run|all] [N]"
        exit 1
        ;;
esac
