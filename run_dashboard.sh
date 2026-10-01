#!/usr/bin/env bash
# ==============================================================================
# run_dashboard.sh: Web Control Panel for Autonomous GUI Operator
# Single-command launch workflow: starts server, launches browser, exits cleanly
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export DISPLAY="${DISPLAY:-:0.0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

PORT="8765"
HOST="127.0.0.1"
URL="http://${HOST}:${PORT}"

# Auto-detect Python interpreter
if [ -f "${SCRIPT_DIR}/venv/bin/python3" ]; then
    PY="${SCRIPT_DIR}/venv/bin/python3"
elif [ -f "${SCRIPT_DIR}/.venv/bin/python3" ]; then
    PY="${SCRIPT_DIR}/.venv/bin/python3"
else
    PY="$(command -v python3 || echo "python3")"
fi

echo -e "\033[1;36m╔═══════════════════════════════════════════════════════════════╗\033[0m"
echo -e "\033[1;36m║            Autonomous Linux GUI Operator Control Panel        ║\033[0m"
echo -e "\033[1;36m║          Starting Localhost Dashboard on ${URL}          ║\033[0m"
echo -e "\033[1;36m╚═══════════════════════════════════════════════════════════════╝\033[0m"

# Launch default browser asynchronously after startup delay
(
    sleep 1.2
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "${URL}" >/dev/null 2>&1 || true
    elif command -v firefox >/dev/null 2>&1; then
        firefox "${URL}" >/dev/null 2>&1 || true
    elif command -v chromium >/dev/null 2>&1; then
        chromium "${URL}" >/dev/null 2>&1 || true
    fi
) &
BROWSER_PID=$!

cleanup() {
    echo -e "\n\033[1;33m[Shutting down]:\033[0m Stopping dashboard server and child processes..."
    kill "${BROWSER_PID}" 2>/dev/null || true
    # Drop emergency stop sentinel
    for p in "/dev/shm/gui_agent_stop" "/tmp/gui_agent_stop"; do
        rm -f "$p" 2>/dev/null || true
    done
    exit 0
}

trap cleanup INT TERM

exec "${PY}" -m uvicorn gui_agent.web.app:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --log-level info
