#!/usr/bin/env bash
set -euo pipefail

echo "[env] python: $(command -v python)"
python -V

# ---- local mode: do NOT install deps (ThinkPad disk) ----
export M4C_SKIP_INSTALL="${M4C_SKIP_INSTALL:-1}"
export M4C_SKIP_MODEL="${M4C_SKIP_MODEL:-1}"

# optional smoke-test caps
export M4C_DEBUG="${M4C_DEBUG:-1}"
export M4C_MAX_PATIENTS_TOTAL="${M4C_MAX_PATIENTS_TOTAL:-200}"
export M4C_MAX_TEST_PATIENTS="${M4C_MAX_TEST_PATIENTS:-20}"
export M4C_MAX_TARGETS="${M4C_MAX_TARGETS:-3}"

python -u models/mamba4cast_model/src/m4c_runner.py models/mamba4cast_model/config/m4c_base.yaml
