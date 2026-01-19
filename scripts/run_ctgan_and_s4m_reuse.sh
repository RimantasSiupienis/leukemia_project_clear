#!/usr/bin/env bash
set -e

# Run CTGAN to bootstrap to S4M reuse
#
# Arguments:
# 1 S4M config yaml (repo-side, mostly for consistency)
# 2 CTGAN static parquet
# 3 Trained S4M checkpoint (.pth)
# 4 Output directory

CFG="$1"
CTGAN_STATIC="$2"
CHECKPOINT="$3"
OUT_DIR="$4"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Matching environment setup used by other model runners
export PYTHONPATH="$REPO_ROOT/models/s4m_model/s4m_official/s4m:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true

python models/s4m_model/src/s4m_ctgan_full_runner.py \
  --config "$CFG" \
  --ctgan_static "$CTGAN_STATIC" \
  --checkpoint "$CHECKPOINT" \
  --out_dir "$OUT_DIR" \
  --device cpu
