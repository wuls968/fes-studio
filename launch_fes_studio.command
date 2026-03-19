#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "Python was not found. Install Python 3.11+ and dependencies first."
  exit 1
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "$PYTHON_BIN" -m fes_studio.launcher launch "$@"
