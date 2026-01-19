#!/usr/bin/env bash
set -e

echo "CTGAN running generator"
python models/ctgan_model/run_ctgan.py

echo "CTGAN running evaluation"
python evaluation/eval_ctgan_static.py

echo "CTGAN finished."
