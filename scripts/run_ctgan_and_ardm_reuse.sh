#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <ardm_config_yaml> <ardm_run_dir> <ctgan_static_parquet>"
  exit 1
fi

ARDM_CONFIG="$1"
ARDM_RUN_DIR="$2"
CTGAN_STATIC="$3"

# Optional env vars
OUT_DIR="${OUT_DIR:-results/ardm/full_ctgan_trajectories}"
DEVICE="${DEVICE:-cpu}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-_official_out/Checkpoints_24/checkpoint-1.pt}"
NON_STRICT_BASELINE="${NON_STRICT_BASELINE:-0}"

# Run from repo root (so relative paths work)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARGS=(
  "models/ardm_model/src/ardm_ctgan_reuse_runner.py"
  "$ARDM_CONFIG"
  "$ARDM_RUN_DIR"
  "$CTGAN_STATIC"
  "--out_dir" "$OUT_DIR"
  "--device" "$DEVICE"
  "--checkpoint_name" "$CHECKPOINT_NAME"
)

if [[ "$NON_STRICT_BASELINE" == "1" ]]; then
  ARGS+=("--non_strict_baseline")
fi

python "${ARGS[@]}"
