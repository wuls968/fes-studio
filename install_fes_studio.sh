#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "Python 3.11+ was not found. Install Python first."
  exit 1
fi

"$PYTHON_BIN" -m venv .venv
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/pip" install -e .

echo
echo "FES Studio has been installed into $ROOT/.venv"
echo "Launch it with:"
echo "  ./launch_fes_studio.sh"

