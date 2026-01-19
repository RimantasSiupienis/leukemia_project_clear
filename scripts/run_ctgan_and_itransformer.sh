#!/usr/bin/env bash
set -e

###############################################################################
# CTGAN → bootstrap → iTransformer (full synthetic trajectory generation)
#
# Assumptions:
# - CTGAN already produced:
#     results/ctgan/physionet2012/ctgan_static.parquet
#     results/ctgan/physionet2012/real_static.parquet
# - iTransformer already trained on real data
###############################################################################

# -------- CONFIG --------
DEVICE=cpu

# iTransformer config + trained run
ITRANS_CONFIG="models/itransformer_model/configs/itransformer_base.yaml"
ITRANS_RUN_DIR="results/itransformer/physionet2012_itransformer"

# CTGAN static output
CTGAN_STATIC="results/ctgan/physionet2012/ctgan_static.parquet"

# Output directory (will be created)
OUT_DIR="results/itransformer/full_ctgan"

# -------- RUN --------
python models/itransformer_model/src/run_ctgan_and_itransformer.py \
  --itransformer_config "${ITRANS_CONFIG}" \
  --itransformer_run_dir "${ITRANS_RUN_DIR}" \
  --ctgan_static_parquet "${CTGAN_STATIC}" \
  --out_dir "${OUT_DIR}" \
  --device "${DEVICE}"

echo "CTGAN + iTransformer synthetic generation finished."
echo "Outputs in: ${OUT_DIR}"
