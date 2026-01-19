#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <m4c_config_yaml> <ctgan_static_parquet>"
  echo "Example:"
  echo "  bash $0 models/mamba4cast_model/config/m4c_base.yaml results/ctgan/physionet2012/ctgan_static.parquet"
  exit 1
fi

CFG="$1"
CTGAN_STATIC="$2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python -u models/mamba4cast_model/src/m4c_ctgan_full_runner.py "$CFG" "$CTGAN_STATIC"
