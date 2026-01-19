#!/usr/bin/env bash
set -euo pipefail

CFG="models/itransformer_model/configs/itransformer_base.yaml"

python models/itransformer_model/src/model.py "$CFG"
