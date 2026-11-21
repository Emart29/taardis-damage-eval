#!/usr/bin/env bash
# Build Cody's TAARDIS llama.cpp fork (stock llama.cpp outputs garbage: weights are in a rotated basis).
# Usage: bash scripts/setup_llama.sh [CUDA_ARCH]   (75 = T4, the Kaggle GPU; pass "cpu" for CPU-only)
set -euo pipefail
ARCH="${1:-75}"
DEST="${LLAMA_DIR:-/tmp/llama.cpp}"
if [ ! -d "$DEST" ]; then
  git clone --depth 1 -b q1_0_g128-port https://github.com/CodeMasterCody3D/taardis-llama.cpp "$DEST"
fi
cd "$DEST"
git rev-parse HEAD > /tmp/llama_commit.txt
if [ "$ARCH" = "cpu" ]; then
  cmake -B build -DLLAMA_CURL=OFF
else
  cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DLLAMA_CURL=OFF
fi
cmake --build build -j"$(nproc)" --target llama-cli llama-server llama-perplexity
ls -la build/bin/
