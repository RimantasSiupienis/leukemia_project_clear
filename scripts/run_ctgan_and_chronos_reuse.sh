#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <chronos_config_yaml> <chronos_existing_run_dir> <ctgan_static_parquet>"
  exit 1
fi

CHRONOS_CONFIG="$1"
CHRONOS_RUN_DIR="$2"
CTGAN_STATIC="$3"

DEVICE="${DEVICE:-cpu}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# default output folder
# results/chronos/full_ctgan_trajectories/<dataset>/<run_name>
DATASET_NAME="$(basename "$(python - <<'PY'
import yaml
from pathlib import Path
p = Path("'"$CHRONOS_CONFIG"'")
cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
out_dir = (cfg.get("output", {}) or {}).get("dir", "")
print(Path(str(out_dir)).name if out_dir else "unknown_dataset")
PY
)")"

RUN_NAME="$(basename "$CHRONOS_RUN_DIR")"
DEFAULT_OUT_DIR="${REPO_ROOT}/results/chronos/full_ctgan_trajectories/${DATASET_NAME}/${RUN_NAME}"

OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"
mkdir -p "$OUT_DIR"

python models/chronos_model/src/chronos_ctgan_full_runner.py \
  --chronos_config "$CHRONOS_CONFIG" \
  --chronos_run_dir "$CHRONOS_RUN_DIR" \
  --ctgan_static_parquet "$CTGAN_STATIC" \
  --out_dir "$OUT_DIR" \
  --device "$DEVICE"
