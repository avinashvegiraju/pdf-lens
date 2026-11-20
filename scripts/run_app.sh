#!/usr/bin/env bash
# Start Streamlit PDF Lens application

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"

cd "$APP_DIR"

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

echo "Starting PDF Lens Streamlit application on http://127.0.0.1:8501 ..."
streamlit run app.py --server.port 8501 --server.address 127.0.0.1
