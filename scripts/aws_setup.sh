#!/usr/bin/env bash
# Set up an AWS GPU instance for experiments/gpu.toml. Run from the repo root on the instance:
#
#   bash scripts/aws_setup.sh
#
# Used on g6.2xlarge (NVIDIA L4 24 GB, 8 vCPU, 32 GB RAM) with the "Deep Learning Base AMI with Single CUDA
# (Ubuntu 24.04)", which ships the NVIDIA driver and the CUDA toolkit. 8 vCPU rather than 4, because blocks T1/T2 run part of the model on the CPU.
#
# Builds llama.cpp at the same commit as the Mac runs (build 11669) with CUDA, installs uv and the Python deps,
# and downloads the model so the first run does not time out on it.
set -euo pipefail

LLAMA_COMMIT=620d2225c
LLAMA_DIR=$HOME/llama.cpp
MODEL=bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q8_0

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version | tail -1

sudo apt-get update -q
sudo apt-get install -yq build-essential cmake git curl libssl-dev

if [ ! -d "$LLAMA_DIR" ]; then
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
git -C "$LLAMA_DIR" fetch -q origin
git -C "$LLAMA_DIR" checkout -q "$LLAMA_COMMIT"
# native: compile CUDA kernels only for this machine's GPU (all architectures would take far longer)
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_DIR/build" -j "$(nproc)" --target llama-server
"$LLAMA_DIR/build/bin/llama-server" --version

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=$HOME/.local/bin:$PATH
fi
uv sync --group analysis

# download the model (same GGUF as on the Mac; experiment.json records its sha256 to confirm), then stop
"$LLAMA_DIR/build/bin/llama-server" -hf "$MODEL" --port 8080 -ngl 99 -c 4096 > /tmp/llama_download.log 2>&1 &
pid=$!
until curl -sf localhost:8080/health >/dev/null; do
    kill -0 $pid 2>/dev/null || { tail -20 /tmp/llama_download.log; exit 1; }
    sleep 5
done
kill -INT $pid
wait $pid || true

echo
echo "Ready. Inside tmux (so a dropped SSH session does not stop them):"
echo "  # 1. transfer vs recompute microbenchmark, ~1 h"
echo "  uv run --group analysis python -m scripts.bench_transfer --out logs/bench_transfer --placements gpu kv_cpu layers18 --llama-cmd $LLAMA_DIR/build/bin/llama-server"
echo "  # 2. the agent experiments, ~5 h"
echo "  uv run python -m scripts.run_experiment --matrix experiments/gpu.toml --llama-cmd $LLAMA_DIR/build/bin/llama-server --cooldown 10"
