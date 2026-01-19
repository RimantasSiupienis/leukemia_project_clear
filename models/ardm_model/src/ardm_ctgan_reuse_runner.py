import sys
from pathlib import Path

# Setting repo root is on PYTHONPATH
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Imports
import argparse
import json

import numpy as np
import pandas as pd
import yaml

from models.s4m_model.src.utils import ensure_dir, save_json, set_seed  # type: ignore
from models.ardm_model.src.official_api_predict_only import predict_only_official  # type: ignore
from models.ardm_model.src.ardm_shared import reshape_preds, build_ctgan_future_long  # type: ignore


def _load_train_ids_from_ardm_split(split_ids_json: Path) -> list[str]:
    obj = json.load(open(split_ids_json, "r", encoding="utf-8"))

    # Accepts multiple formats:
    # 1. "train": [], "val": [], "test": []
    if isinstance(obj, dict) and "train" in obj and isinstance(obj["train"], list):
        return [str(x) for x in obj["train"]]

    # 2. "split_ids": {"train": [], ...}
    if (
        isinstance(obj, dict)
        and "split_ids" in obj
        and isinstance(obj["split_ids"], dict)
        and "train" in obj["split_ids"]
        and isinstance(obj["split_ids"]["train"], list)
    ):
        return [str(x) for x in obj["split_ids"]["train"]]

    # 3. legacy: "train_ids": []
    if isinstance(obj, dict) and "train_ids" in obj and isinstance(obj["train_ids"], list):
        return [str(x) for x in obj["train_ids"]]

    raise KeyError(
        f"Could not find train ids in {split_ids_json}. "
        "Expected one of: "
        "{train:[...]}, {split_ids:{train:[...]}}, or {train_ids:[...]}."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ardm_config", help="Path to ARDM wrapper config yaml (your project config)")
    ap.add_argument("ardm_run_dir", help="Trained ARDM run dir (contains split_ids.json and _official_out)")
    ap.add_argument("ctgan_static_parquet", help="CTGAN static parquet")
    ap.add_argument("--out_dir", default="results/ardm/full_ctgan_trajectories", help="Output dir")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--checkpoint_name", default="_official_out/Checkpoints_24/checkpoint-1.pt")
    ap.add_argument("--non_strict_baseline", action="store_true")
    args = ap.parse_args()

    ardm_cfg_path = Path(args.ardm_config)
    ardm_run_dir = Path(args.ardm_run_dir)
    ctgan_static = Path(args.ctgan_static_parquet)
    out_dir = Path(args.out_dir)

    ensure_dir(str(out_dir))

    with open(ardm_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    batch_size = int(cfg.get("train", {}).get("batch_size", 128))

    id_col = cfg["data"]["id_col"]
    time_col = cfg["data"]["time_col"]
    target_cols = list(cfg["data"]["target_cols"])

    # Supports multiple config layouts for history_len/pred_len (debugging)
    if "task" in cfg:
        history_len = int(cfg["task"]["history_len"])
        pred_len = int(cfg["task"]["pred_len"])
    elif "data" in cfg and ("history_len" in cfg["data"] or "pred_len" in cfg["data"]):
        history_len = int(cfg["data"].get("history_len"))
        pred_len = int(cfg["data"].get("pred_len"))
    elif "train" in cfg and ("history_len" in cfg["train"] or "pred_len" in cfg["train"]):
        history_len = int(cfg["train"].get("history_len"))
        pred_len = int(cfg["train"].get("pred_len"))
    else:
        raise KeyError(
            "Could not find history_len/pred_len in config. Expected one of:\n"
            "  task.history_len + task.pred_len\n"
            "  data.history_len + data.pred_len\n"
            "  train.history_len + train.pred_len"
        )

    # Finding real long parquet path from config 
    # supports multiple key names (debugging)
    data_cfg = cfg.get("data", {})
    candidates = [
        "real_long_parquet",
        "real_long_path",
        "long_parquet",
        "long_path",
        "longitudinal_path",
        "longitudinal_parquet",
        "path_long",
    ]
    real_long_val = None
    for k in candidates:
        if k in data_cfg and data_cfg[k]:
            real_long_val = data_cfg[k]
            break

    if real_long_val is None:
        import os

        env_val = os.environ.get("REAL_LONG_PARQUET", "").strip()
        if env_val:
            real_long_val = env_val

    if real_long_val is None:
        raise KeyError(
            "Could not find real longitudinal parquet path in cfg['data'].\n"
            f"Tried keys: {candidates}\n"
            "Either add one of these keys under data: in  YAML, or set env var REAL_LONG_PARQUET."
        )

    real_long_parquet = Path(real_long_val)
    if not real_long_parquet.exists():
        raise FileNotFoundError(f"Real long parquet not found: {real_long_parquet}")

    split_ids_json = ardm_run_dir / "split_ids.json"
    if not split_ids_json.exists():
        raise FileNotFoundError(f"Expected split_ids.json in trained ARDM run dir: {split_ids_json}")

    train_ids = _load_train_ids_from_ardm_split(split_ids_json)
    train_only_json = out_dir / "train_ids_for_bootstrap.json"
    with open(train_only_json, "w", encoding="utf-8") as f:
        json.dump({"train_ids": train_ids}, f)

    bootstrap_dir = out_dir / "bootstrap"
    ensure_dir(str(bootstrap_dir))

    # Running bootstrap script
    bootstrap_py = REPO_ROOT / "scripts" / "bootstrap_ctgan_to_history.py"
    if not bootstrap_py.exists():
        raise FileNotFoundError(f"bootstrap script not found: {bootstrap_py}")

    cmd = [
        "python",
        str(bootstrap_py),
        "--ctgan_static_parquet",
        str(ctgan_static),
        "--real_long_parquet",
        str(real_long_parquet),
        "--out_dir",
        str(bootstrap_dir),
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
        str(seed),
        "--train_split_ids_json",
        str(train_only_json),
    ]
    if args.non_strict_baseline:
        cmd.append("--non_strict_baseline")

    import subprocess

    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"Bootstrap failed: {' '.join(cmd)}")

    windows_npy = bootstrap_dir / "bootstrap_windows_for_model.npy"
    if not windows_npy.exists():
        raise FileNotFoundError(f"Expected bootstrap windows not found: {windows_npy}")

    ckpt = ardm_run_dir / args.checkpoint_name
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"[reuse] Using ARDM checkpoint: {ckpt}")

    repo_dir = Path(cfg["official"]["repo_dir"])
    if not repo_dir.exists():
        raise FileNotFoundError(f"Official ARDM repo not found: {repo_dir}")

    ardm_pred_out = out_dir / "ardm_pred"
    ensure_dir(str(ardm_pred_out))

    pred_npy = predict_only_official(
        repo_dir=repo_dir,
        checkpoint_path=ckpt,
        test_npy_path=windows_npy,
        out_dir=ardm_pred_out,
        history_len=history_len,
        pred_len=pred_len,
        batch_size=batch_size,
        device=args.device,
    )

    preds = np.load(pred_npy)
    static = pd.read_parquet(ctgan_static)
    if "patient_id" not in static.columns:
        raise ValueError("CTGAN static parquet must contain 'patient_id' column")

    ids = static["patient_id"].astype(str).tolist()

    preds_3d = reshape_preds(
        preds,
        n_series=len(ids),
        pred_len=pred_len,
        n_targets=len(target_cols),
    )

    out_long = out_dir / "ctgan_synth_future_long.parquet"
    out_df = build_ctgan_future_long(
        ctgan_ids=ids,
        id_col=id_col,
        time_col=time_col,
        target_cols=target_cols,
        preds_3d=preds_3d,
        time_start=0,  # to match validated output (t=0..pred_len-1)
    )
    out_df.to_parquet(out_long, index=False)

    meta = {
        "seed": seed,
        "history_len": history_len,
        "pred_len": pred_len,
        "target_cols": target_cols,
        "checkpoint": str(ckpt),
        "pred_npy": str(pred_npy),
        "out_long_parquet": str(out_long),
    }
    save_json(meta, out_dir / "run_meta.json")
    print(f"[reuse] Wrote: {out_long}")


if __name__ == "__main__":
    main()
