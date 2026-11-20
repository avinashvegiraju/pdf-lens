#!/usr/bin/env bash
# Verify health and basic inference of local llama-server

set -euo pipefail

BASE_URL="${LLAMA_BASE_URL:-http://127.0.0.1:8081}"
MODEL="${LLAMA_MODEL:-lfm3-rag}"

echo "1. Checking health endpoint: ${BASE_URL}/health"
if curl --fail -s "${BASE_URL}/health" > /dev/null; then
    echo "   [SUCCESS] llama-server is healthy and running!"
else
    echo "   [FAILURE] llama-server is not reachable at ${BASE_URL}/health"
    echo "   Make sure to start it with: ./scripts/serve_llama.sh"
    exit 1
fi

echo ""
echo "2. Sending test chat completion to ${BASE_URL}/v1/chat/completions"
RESPONSE=$(curl --fail -s "${BASE_URL}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Explain RAG in one sentence.\"}],
    \"temperature\": 0.1,
    \"max_tokens\": 60,
    \"stream\": false
  }")

echo "Response received:"
echo "$RESPONSE"
echo ""
echo "[SUCCESS] llama.cpp LFM3 is operating correctly!"
