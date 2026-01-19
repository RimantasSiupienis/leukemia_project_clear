#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH_REL=${1:-models/ardm_model/configs/ardm_base.yaml}
CONFIG_PATH="${ROOT_DIR}/${CONFIG_PATH_REL}"

echo "[ARDM] Running with config: ${CONFIG_PATH_REL}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

python -u "${ROOT_DIR}/models/ardm_model/src/ardm_runner.py" \
  --config "${CONFIG_PATH}"

echo "[ARDM] Done."
