import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml

from models.s4m_model.src.utils import set_seed, ensure_dir, save_json  # type: ignore
from models.ardm_model.src.ardm_shared import reshape_preds, build_ctgan_future_long  # type: ignore

from ardm_adapter_dataset import prep_dataset
from official_api import train_and_predict_official, get_official_git_commit


def run_cmd(cmd: List[str]) -> None:
    p = subprocess.run(cmd, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (exit={p.returncode}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser(description="End-to-end: CTGAN static -> bootstrap -> ARDM -> CTGAN trajectories")
    ap.add_argument("--config", required=True, help="ARDM YAML config (same one used by ardm_runner.py)")

    # CTGAN part
    ap.add_argument("--ctgan_static_parquet", required=True, help="Path to CTGAN static parquet (ctgan patients)")
    ap.add_argument("--ctgan_id_col", default=None, help="ID column name in CTGAN static parquet (defaults to cfg.data.id_col)")
    ap.add_argument("--run_ctgan", action="store_true", help="If set, runs python run_ctgan.py before bootstrapping")
    ap.add_argument("--ctgan_script", default="run_ctgan.py", help="Path to run_ctgan.py (only used if --run_ctgan)")

    # Bootstrap part
    ap.add_argument("--bootstrap_script", default="scripts/bootstrap_ctgan_to_history.py")
    ap.add_argument("--baseline_map_json", default=None)
    ap.add_argument("--bootstrap_out_dir", default=None, help="Where to write bootstrap outputs (default: <out_dir>/ctgan_bootstrap)")
    ap.add_argument("--non_strict_baseline", action="store_true")

    # Output naming
    ap.add_argument("--output_name", default="ctgan_synth_future_long.parquet")

    args = ap.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["seed"]))

    repo = Path(cfg["official"]["repo_dir"])
    if not repo.exists():
        raise RuntimeError("ARDM repo not cloned")

    out_root = Path(cfg["output"]["dir"])
    run_name = str(cfg["output"]["run_name"])
    out_dir = out_root / run_name
    ensure_dir(str(out_dir))

    id_col = cfg["data"]["id_col"]
    time_col = cfg["data"]["time_col"]
    target_cols = list(cfg["data"]["target_cols"])
    history_len = int(cfg["data"]["history_len"])
    pred_len = int(cfg["data"]["pred_len"])

    if args.run_ctgan:
        run_cmd(["python", args.ctgan_script])

    ctgan_static_path = Path(args.ctgan_static_parquet)
    if not ctgan_static_path.exists():
        raise FileNotFoundError(f"CTGAN static parquet not found: {ctgan_static_path}")

    # Preparing ARDM dataset 
    # train/val from real longitudinal parquet
    t0 = time.time()
    train_p, val_p, _test_p_real, manifest = prep_dataset(cfg)
    t1 = time.time()

    # Bootstraping CTGAN into history windows using train templates
    split_tmp = out_dir / "_bootstrap_train_split_ids.json"
    with open(split_tmp, "w", encoding="utf-8") as f:
        json.dump({"train": manifest["split_ids"]["train"]}, f)

    bootstrap_out = Path(args.bootstrap_out_dir) if args.bootstrap_out_dir else (out_dir / "ctgan_bootstrap")
    ensure_dir(str(bootstrap_out))

    real_long_parquet = cfg["data"]["parquet_path"]
    ctgan_id_col = args.ctgan_id_col if args.ctgan_id_col else id_col

    cmd = [
        "python",
        args.bootstrap_script,
        "--ctgan_static_parquet",
        str(ctgan_static_path),
        "--real_long_parquet",
        str(real_long_parquet),
        "--out_dir",
        str(bootstrap_out),
        "--id_col",
        str(id_col),
        "--time_col",
        str(time_col),
        "--target_cols",
        ",".join(target_cols),
        "--history_len",
        str(history_len),
        "--pred_len",
        str(pred_len),
        "--seed",
        str(int(cfg["seed"])),
        "--train_split_ids_json",
        str(split_tmp),
    ]
    if args.baseline_map_json:
        cmd += ["--baseline_map_json", args.baseline_map_json]
    if args.non_strict_baseline:
        cmd += ["--non_strict_baseline"]

    run_cmd(cmd)

    bootstrap_windows = bootstrap_out / "bootstrap_windows_for_model.npy"
    if not bootstrap_windows.exists():
        raise FileNotFoundError(f"Expected bootstrap windows not found: {bootstrap_windows}")

    # Train + Predict ARDM using CTGAN bootstraped windows as test set
    official_out = out_dir / "_official_out_ctgan"
    ensure_dir(str(official_out))

    pred_npy = train_and_predict_official(
        cfg=cfg,
        repo_dir=repo,
        train_path=train_p,
        val_path=val_p,
        test_path=str(bootstrap_windows),
        out_dir=official_out,
    )

    preds = np.load(pred_npy)
    ctgan_df = pd.read_parquet(ctgan_static_path)
    if ctgan_id_col not in ctgan_df.columns:
        raise ValueError(f"CTGAN id col '{ctgan_id_col}' not in {ctgan_static_path}. Columns={list(ctgan_df.columns)[:30]}")

    ctgan_ids = ctgan_df[ctgan_id_col].astype(str).tolist()

    preds_3d = reshape_preds(
        preds,
        n_series=len(ctgan_ids),
        pred_len=pred_len,
        n_targets=len(target_cols),
    )

    # Preserves this scripts convention from before (1..pred_len)
    # using time_start = 1
    ctgan_synth_future_long = build_ctgan_future_long(
        ctgan_ids=ctgan_ids,
        id_col=id_col,
        time_col=time_col,
        target_cols=target_cols,
        preds_3d=preds_3d,
        time_start=1,
    )

    out_path = out_dir / args.output_name
    ctgan_synth_future_long.to_parquet(out_path, index=False)

    run_meta = {
        "mode": "ctgan_static_to_ardm_trajectories",
        "seed": int(cfg["seed"]),
        "timing_sec": {
            "prep_dataset": float(t1 - t0),
        },
        "ctgan": {
            "ctgan_static_parquet": str(ctgan_static_path),
            "ctgan_id_col": ctgan_id_col,
            "n_ctgan": int(len(ctgan_ids)),
        },
        "bootstrap": {
            "out_dir": str(bootstrap_out),
            "windows_npy": str(bootstrap_windows),
            "history_len": history_len,
            "pred_len": pred_len,
            "baseline_map_json": args.baseline_map_json,
            "non_strict_baseline": bool(args.non_strict_baseline),
        },
        "ardm": {
            "official_repo_dir": str(repo),
            "official_git_commit": get_official_git_commit(repo),
            "train_path": str(train_p),
            "val_path": str(val_p),
            "test_path": str(bootstrap_windows),
            "pred_npy": str(pred_npy),
        },
        "outputs": {
            "ctgan_synth_future_long": str(out_path),
        },
        "targets": target_cols,
        "schema": {
            "ctgan_synth_future_long": [id_col, time_col] + target_cols,
        },
    }
    save_json(run_meta, out_dir / "ctgan_run_meta.json")

    print(f"[ARDM+CTGAN] Wrote: {out_path}")
    print(f"[ARDM+CTGAN] Meta : {out_dir / 'ctgan_run_meta.json'}")


if __name__ == "__main__":
    main()
