#!/usr/bin/env bash
# ==============================================================================
# Wan2GP Cloud Launcher for Linux / macOS
# Zero local GPU requirements - runs in cloud client mode
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="$HOME/.local/bin:$PATH"

if [ -f ".env" ]; then
    set -a
    source ".env"
    set +a
fi

echo "============================================================"
echo "          Starting Wan2GP in Cloud Mode (Modal GPU)         "
echo "============================================================"

# Choose best available Python interpreter
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PY_BIN="$SCRIPT_DIR/.venv/bin/python"
elif [ -f "$HOME/.wan2gp-cloud-venv/bin/python" ]; then
    PY_BIN="$HOME/.wan2gp-cloud-venv/bin/python"
elif [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    PY_BIN="$VIRTUAL_ENV/bin/python"
elif [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
    PY_BIN="$CONDA_PREFIX/bin/python"
elif command -v python3 &>/dev/null; then
    PY_BIN="python3"
else
    PY_BIN="python"
fi

# Check if provider is RunPod or Modal
PROV_CHECK="${WAN2GP_CLOUD_PROVIDER:-${WAN2GP_DEFAULT_PROVIDER:-modal}}"
if [[ "$*" =~ "runpod" ]] || [ "$PROV_CHECK" = "runpod" ]; then
    echo "[*] Cloud Provider: RunPod Serverless (Beta Preview)"
    if [ ! -f ".env" ] || ! grep -q "RUNPOD_API_KEY" ".env"; then
        echo "[!] Note: RUNPOD_API_KEY is not configured in .env yet."
        echo "[*] See docs/RUNPOD_SETUP.md or configure .env to connect."
    fi
elif [ ! -f ".env" ] || ! grep -q "MODAL_ENDPOINT" ".env"; then
    echo ""
    echo "[!] Cloud Modal is not set up yet on your device."
    echo "[*] Launching the 1-click Modal setup wizard now..."
    echo ""
    "$PY_BIN" cloud/modal/setup_modal.py
    echo ""
fi

# Launch Wan2GP in Cloud Client Mode
echo "[*] Launching Wan2GP Web UI..."
BROWSER_FLAG="--open-browser"
if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
    BROWSER_FLAG="--listen"
fi

PROVIDER_FLAG=""
if [[ ! "$*" =~ "--provider" ]]; then
    PROVIDER_FLAG="--provider $PROV_CHECK"
fi

"$PY_BIN" wgp.py --cloud $PROVIDER_FLAG $BROWSER_FLAG "$@"
