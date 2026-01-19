#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S4M_RUN="$REPO_ROOT/models/s4m_model/s4m_official/s4m/run.py"

# === YOUR DATA ===
DATA_ROOT="$REPO_ROOT/data/processed"
DATA_PATH="physionet2012_long.parquet"
# ================

# === YOUR RUN DIR ===
RUN_DIR="$REPO_ROOT/results/s4m/physionet2012/s4m_official_run1"
# ====================

mkdir -p "$RUN_DIR/checkpoints"

# disable wandb prompts
export PYTHONPATH="$REPO_ROOT/models/s4m_model/s4m_official/s4m:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export WANDB_SILENT=true

python "$S4M_RUN" \
  --is_training 1 \
  --model S4M \
  --data parquet_long \
  --root_path "$DATA_ROOT" \
  --data_path "$DATA_PATH" \
  --checkpoints "$RUN_DIR/checkpoints" \
  --use_gpu 0 \
  --seq_len 24 \
  --label_len 0 \
  --pred_len 12 \
  --features M \
  --mask \
  --train_epochs 5 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --num_workers 0
