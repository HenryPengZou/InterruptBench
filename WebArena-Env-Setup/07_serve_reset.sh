#!/bin/bash

# stop if any error occur
set -e

source 00_vars.sh

# Pick a usable Python interpreter.
# Prefer python3.10 (original expectation), but fall back gracefully.
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  for c in python3 python; do
    if command -v "${c}" >/dev/null 2>&1; then
      PYTHON_BIN="${c}"
      break
    fi
  done
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: cannot find a Python interpreter (tried: python3.10, python3, python)." >&2
  echo "Hint: install Python 3.10+ or set PYTHON_BIN to a valid python path." >&2
  exit 1
fi

# venv location (override if desired)
VENV_RESET_DIR="${VENV_RESET_DIR:-venv_reset}"

# If a previous run created a root-owned venv (e.g. via sudo), creating/updating it
# as the current user will fail with EACCES. Detect and guide the user early.
if [ -e "${VENV_RESET_DIR}" ] && [ ! -w "${VENV_RESET_DIR}" ]; then
  echo "Error: '${VENV_RESET_DIR}' exists but is not writable by $(id -un)." >&2
  echo "Hint: it was likely created via sudo. Remove it then rerun without sudo:" >&2
  echo "  sudo rm -rf '${VENV_RESET_DIR}'" >&2
  exit 1
fi

# install flask in a venv
# sudo yum install python3-pip -y
# If your filesystem or policy disallows symlinks, set: VENV_ARGS="--copies"
VENV_ARGS="${VENV_ARGS:-}"
"${PYTHON_BIN}" -m venv ${VENV_ARGS} "${VENV_RESET_DIR}"
source "${VENV_RESET_DIR}/bin/activate"

cd reset_server/
python server.py --port ${RESET_PORT} 2>&1 | tee -a server.log

# visit http://$PUBLIC_HOSTNAME:7565/reset
