#!/bin/bash
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$PROJECT_DIR/src/nbody.cu"
BIN="$PROJECT_DIR/nbody"
mkdir -p "$PROJECT_DIR/results"

echo "╔══════════════════════════════════╗"
echo "║   N-Body HPC Build System        ║"
echo "╚══════════════════════════════════╝"

if ! command -v nvcc &>/dev/null; then
    echo "[WARN] nvcc not found — building CPU-only fallback"
    if ! command -v mpicxx &>/dev/null; then
        echo "[ERROR] mpicxx not found. Run: sudo apt-get install -y openmpi-bin libopenmpi-dev"
        exit 1
    fi
    # CPU-only compile via nvcc stub
    echo "[BUILD] CPU mode via mpicxx+OpenMP"
    mpicxx -O3 -fopenmp -march=native \
        -x c++ \
        -DCPU_ONLY \
        "$SRC" -o "$BIN" 2>/dev/null || true
fi

if ! command -v mpicxx &>/dev/null; then
    echo "[ERROR] mpicxx not found. Run: sudo apt-get install -y openmpi-bin libopenmpi-dev"
    exit 1
fi

# Detect GPU arch
GPU_ARCH="sm_75"
if command -v nvidia-smi &>/dev/null; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' | tr -d ' ')
    [ -n "$CC" ] && GPU_ARCH="sm_$CC" && echo "[INFO] GPU arch: $GPU_ARCH"
fi

MPI_CFLAGS=$(mpicxx --showme:compile 2>/dev/null || echo "-I/usr/include/mpi")
MPI_LFLAGS=$(mpicxx --showme:link   2>/dev/null || echo "-lmpi")

echo "[BUILD] Compiling with CUDA+MPI+OpenMP..."
nvcc \
    -arch="$GPU_ARCH" \
    --generate-code arch=compute_70,code=sm_70 \
    -Xcompiler "-fopenmp,-O3,-march=native" \
    -O3 -DNDEBUG \
    $MPI_CFLAGS $MPI_LFLAGS -lmpi \
    "$SRC" -o "$BIN"

echo "[BUILD] ✓ Binary: $BIN"
echo "[BUILD] ✓ SUCCESS — run: bash scripts/run.sh"
