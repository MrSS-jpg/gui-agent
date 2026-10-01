#!/usr/bin/env bash
# ==============================================================================
# run_agent.sh: Single-command runner for autonomous Linux GUI Operator
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export DISPLAY="${DISPLAY:-:0.0}"

# Auto-detect Python interpreter (venv preferred, then system python3)
if [ -f "${SCRIPT_DIR}/venv/bin/python3" ]; then
    PY="${SCRIPT_DIR}/venv/bin/python3"
elif [ -f "${SCRIPT_DIR}/.venv/bin/python3" ]; then
    PY="${SCRIPT_DIR}/.venv/bin/python3"
else
    PY="$(command -v python3 || echo "python3")"
fi

# Clean exit on Ctrl+C
trap 'echo -e "\n\033[1;33m[Stopped]:\033[0m Agent process cancelled by user."; exit 130' INT TERM

exec "${PY}" -m gui_agent.cli "$@"
