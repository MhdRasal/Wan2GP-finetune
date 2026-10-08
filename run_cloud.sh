#!/usr/bin/env bash
# ==============================================================================
# Wan2GP Cloud Launcher for Linux / macOS
# Zero local GPU requirements - runs in cloud client mode
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="$HOME/.local/bin:$PATH"

echo "============================================================"
echo "          Starting Wan2GP in Cloud Mode (Modal GPU)         "
echo "============================================================"

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

# Check if .env is configured with Modal
if [ ! -f ".env" ] || ! grep -q "MODAL_ENDPOINT" ".env"; then
    echo ""
    echo "[!] Cloud Modal is not set up yet on your device."
    echo "[*] Launching the 1-click Modal setup wizard now..."
    echo ""
    "$PY_BIN" cloud/modal/setup_modal.py
    echo ""
fi

# Launch Wan2GP in Cloud Client Mode
echo "[*] Launching Wan2GP Web UI..."
"$PY_BIN" wgp.py --cloud --provider modal --open-browser "$@"
