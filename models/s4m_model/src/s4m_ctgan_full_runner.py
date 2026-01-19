"""
Run S4M on CTGAN-generated patients.

What this script does:
- Takes already trained S4M from cloned repo
- Takes CTGAN static patients
- Builds a short fake history window
- Runs S4M inference to generate future values

This is intentionally very similar to:
- chronos_ctgan_full_runner.py
- ttm_ctgan_reuse_runner.py
- m4c_ctgan_full_runner.py
- S4M is not retrained here
- This is reuse / inference only
"""

import argparse
import json
import subprocess
from pathlib import Path

from scripts.bootstrap_ctgan_to_history import bootstrap_ctgan_to_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Repo-side S4M yaml (for consistency)")
    parser.add_argument("--ctgan_static", required=True, help="CTGAN static parquet file")
    parser.add_argument("--checkpoint", required=True, help="Path to trained S4M checkpoint.pth")
    parser.add_argument("--out_dir", required=True, help="Output run directory")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1 Bootstrap CTGAN static patients into a fake history window
    # same logic as all other longitudinal models
    bootstrap_dir = out_dir / "bootstrap"
    bootstrap_dir.mkdir(exist_ok=True)

    bootstrap_ctgan_to_history(
        ctgan_static_path=args.ctgan_static,
        out_dir=str(bootstrap_dir),
    )

    # 2 Runs OFFICIAL S4M inference using the existing checkpoint
    cmd = [
        "python",
        "models/s4m_model/s4m_official/s4m/run.py",
        "--is_training", "0",
        "--model", "S4M",
        "--data", "parquet_long",
        "--root_path", str(bootstrap_dir),
        "--data_path", "bootstrap_future_long.parquet",
        "--features", "M",
        "--target", "HR",
        "--freq", "h",
        "--seq_len", "8",
        "--label_len", "0",
        "--pred_len", "1",
        "--enc_in", "8",
        "--dec_in", "8",
        "--c_out", "8",
        "--d_var", "8",
        "--d_model", "8",
        "--d_ff", "16",
        "--e_layers", "1",
        "--n_heads", "1",
        "--memnet", "0",
        "--use_gpu", "False",
        "--checkpoints", str(Path(args.checkpoint).parent),
        "--des", "CTGAN_REUSE",
    ]

    subprocess.run(cmd, check=True)

    # 3 Saves minimal metadata so the run is traceable later
    meta = {
        "model": "S4M",
        "mode": "ctgan_reuse",
        "checkpoint": args.checkpoint,
        "ctgan_static": args.ctgan_static,
        "bootstrap_dir": str(bootstrap_dir),
    }

    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
