#!/usr/bin/env bash
set -euo pipefail

if [ ! -x ".venv/bin/scene-intelligence" ]; then
  bash ./install.sh
fi

SOURCE="${1:-0}"
./.venv/bin/scene-intelligence --source "$SOURCE"
