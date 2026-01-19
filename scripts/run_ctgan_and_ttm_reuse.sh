#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <ttm_config_yaml> <ttm_run_dir> <ctgan_static_parquet>"
  echo "Example:"
  echo "  bash scripts/run_ctgan_and_ttm_reuse.sh \\"
  echo "    models/ttm_model/configs/ttm_base.yaml \\"
  echo "    results/ttm/physionet2012/ttm_run \\"
  echo "    results/ctgan/physionet2012/ctgan_static.parquet"
  exit 1
fi

TTM_CONFIG="$1"
TTM_RUN_DIR="$2"
CTGAN_STATIC="$3"

DEVICE="${DEVICE:-cpu}"
OUT_DIR="${OUT_DIR:-}"                       # optional override
BASELINE_MAP_JSON="${BASELINE_MAP_JSON:-}"   # optional
NON_STRICT_BASELINE="${NON_STRICT_BASELINE:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CMD=(python models/ttm_model/src/ttm_ctgan_reuse_runner.py "$TTM_CONFIG" "$TTM_RUN_DIR" "$CTGAN_STATIC" --device "$DEVICE")

if [[ -n "${OUT_DIR}" ]]; then
  CMD+=(--out_dir "$OUT_DIR")
fi

if [[ -n "${BASELINE_MAP_JSON}" ]]; then
  CMD+=(--baseline_map_json "$BASELINE_MAP_JSON")
fi

if [[ "${NON_STRICT_BASELINE}" == "1" ]]; then
  CMD+=(--non_strict_baseline)
fi

"${CMD[@]}"

#  bash scripts/run_ctgan_and_ttm_reuse.sh \
#  models/ttm_model/configs/ttm_base.yaml \
#  results/ttm/physionet2012/ttm_run \
#  results/ctgan/physionet2012/ctgan_static.parquet
