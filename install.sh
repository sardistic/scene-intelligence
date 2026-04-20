#!/usr/bin/env bash
set -euo pipefail

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python was not found. Install Python 3.10 or 3.11 first."
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

./.venv/bin/python -m pip install --upgrade pip wheel
./.venv/bin/python -m pip install .

echo
echo "Scene Intelligence is installed."
echo "Run ./run-webcam.sh to start the default webcam."
