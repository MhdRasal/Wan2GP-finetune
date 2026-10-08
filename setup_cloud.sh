#!/usr/bin/env bash
# ==============================================================================
# Wan2GP 1-Click Cloud Setup Wizard
# ==============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="$HOME/.local/bin:$PATH"

# Choose best available Python interpreter
if [ -f "/home/rasal/Desktop/dirve Rasal/work/modal/.venv/bin/python" ]; then
    PY_BIN="/home/rasal/Desktop/dirve Rasal/work/modal/.venv/bin/python"
elif [ -f "$HOME/.wan2gp-cloud-venv/bin/python" ]; then
    PY_BIN="$HOME/.wan2gp-cloud-venv/bin/python"
elif [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PY_BIN="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PY_BIN="python3"
else
    PY_BIN="python"
fi

"$PY_BIN" cloud/modal/setup_modal.py "$@"
