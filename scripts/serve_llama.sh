#!/usr/bin/env bash
# Start official LiquidAI/LFM3-1.7B-GGUF server via llama.cpp on 127.0.0.1:8081

set -euo pipefail

LLAMA_BIN="${LLAMA_SERVER_BIN:-$(which llama-server || echo "/opt/homebrew/bin/llama-server")}"

if [ ! -x "$LLAMA_BIN" ]; then
    echo "ERROR: llama-server executable not found."
    echo "Install it with: brew install llama.cpp"
    exit 1
fi

HOST="127.0.0.1"
PORT="8081"
MODEL_REPO="LiquidAI/LFM3-1.7B-GGUF:Q4_K_M"
ALIAS="lfm3-rag"
CONTEXT="8192"
GPU_LAYERS="99"

echo "=========================================================="
echo " Starting local LiquidAI LFM3-1.7B server via llama.cpp"
echo " Host:        http://${HOST}:${PORT}"
echo " Model:       ${MODEL_REPO} (alias: ${ALIAS})"
echo " Context:     ${CONTEXT} tokens"
echo " GPU Offload: Metal (${GPU_LAYERS} layers)"
echo "=========================================================="

exec "$LLAMA_BIN" \
    -hf "$MODEL_REPO" \
    --alias "$ALIAS" \
    --host "$HOST" \
    --port "$PORT" \
    -c "$CONTEXT" \
    -ngl "$GPU_LAYERS" \
    -np 1 \
    --jinja \
    --reasoning off
