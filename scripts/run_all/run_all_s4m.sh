#!/usr/bin/env bash
set -euo pipefail


CFG="models/s4m_model/configs/s4m_base.yaml"

python -u models/s4m_model/src/s4m_runner.py "$CFG"
echo "S4M finished."
