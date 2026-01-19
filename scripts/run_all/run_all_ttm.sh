#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <ttm_config_yaml>"
  exit 1
fi

TTM_CONFIG="$1"

# go to repo root (two levels up from scripts/run_all)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT"

python "models/ttm_model/src/model.py" "$TTM_CONFIG"
