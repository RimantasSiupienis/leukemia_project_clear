#!/usr/bin/env bash
set -e

# run Chronos with the default config

CONFIG_PATH="models/chronos_model/configs/chronos_base.yaml"

echo "Running Chronos with config:"
echo "  ${CONFIG_PATH}"
echo

python models/chronos_model/src/model.py "${CONFIG_PATH}"
