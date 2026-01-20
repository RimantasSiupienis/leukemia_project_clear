from __future__ import annotations
def build_ctgan_future_long(
    ctgan_ids: List[str],
    id_col: str,
    time_col: str,
    target_cols: List[str],
    preds_3d: np.ndarray,
    time_start: int = 0,
) -> pd.DataFrame:
    """
    Builds long-format future table for CTGAN patients:
      columns: [id_col, time_col] + target_cols
      rows: N * pred_len
      time index = [time_start .. time_start + pred_len - 1]
    """
    n, pred_len, c = preds_3d.shape
    if len(ctgan_ids) != n:
        raise ValueError(f"ctgan_ids length {len(ctgan_ids)} != preds_3d first dim {n}")
    if c != len(target_cols):
        raise ValueError(f"preds_3d last dim {c} != len(target_cols) {len(target_cols)}")

    rows = []
    for i, pid in enumerate(ctgan_ids):
        for t in range(pred_len):
            row = {id_col: str(pid), time_col: int(time_start + t)}
            for j, col in enumerate(target_cols):
                row[col] = float(preds_3d[i, t, j])
            rows.append(row)
    return pd.DataFrame(rows)

def reshape_preds(preds: np.ndarray, n_series: int, pred_len: int, n_targets: int) -> np.ndarray:
    """
    Normalize predictions to shape [n_series, pred_len, n_targets].

    Accepts common layouts:
      - [n_series, pred_len, n_targets]
      - [n_series, n_targets, pred_len]  -> transposed
      - [n_series, pred_len * n_targets] -> reshaped
      - [n_series * pred_len, n_targets] -> reshaped
    """
    if preds.ndim == 3:
        if preds.shape == (n_series, pred_len, n_targets):
            return preds
        if preds.shape == (n_series, n_targets, pred_len):
            return np.transpose(preds, (0, 2, 1))
        raise ValueError(f"Unexpected 3D preds shape: {preds.shape}, expected (N,T,C) or (N,C,T)")

    if preds.ndim == 2:
        if preds.shape == (n_series, pred_len * n_targets):
            return preds.reshape(n_series, pred_len, n_targets)
        if preds.shape == (n_series * pred_len, n_targets):
            return preds.reshape(n_series, pred_len, n_targets)
        raise ValueError(
            f"Unexpected 2D preds shape: {preds.shape}; "
            f"expected (N, T*C) or (N*T, C) with N={n_series}, T={pred_len}, C={n_targets}"
        )

    raise ValueError(f"Unexpected preds ndim={preds.ndim}, shape={preds.shape}")


import argparse
import subprocess
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml

from models.s4m_model.src.utils import set_seed, ensure_dir, save_json
from models.ardm_model.src.official_api_predict_only import predict_only_official


def _run(cmd: List[str], cwd: Path | None = None) -> None:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (exit={p.returncode}): {' '.join(cmd)}")


def _resolve_path(p: str | Path, base: Path) -> Path:
    """
    Robust path resolver for Kaggle.
    Accepts:
      - absolute paths: /kaggle/input/..., /kaggle/working/...
      - kaggle-root relative: kaggle/input/..., kaggle/working/... (auto-add leading '/')
      - normal relatives: resolved relative to base
    """
    p = str(p)
    # Expand ~ just in case
    p = str(Path(p).expanduser())

    # If user wrote kaggle/input/... without leading slash, fix it.
    if p.startswith("kaggle/"):
        p = "/" + p

    pp = Path(p)
    if pp.is_absolute():
        return pp

    return (base / pp).resolve()

    rows = []
    for i, pid in enumerate(ctgan_ids):
        for t in range(pred_len):
            row = {id_col: str(pid), time_col: int(time_start + t)}
            for j, col in enumerate(target_cols):
                row[col] = float(preds_3d[i, t, j])
            rows.append(row)
    return pd.DataFrame(rows)


def json_load(p: Path) -> dict:
    import json
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main(cfg_path: str) -> None:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))

    # Base path for resolving relative paths in Kaggle (/kaggle/working/<project>)
    base = Path.cwd().resolve()

    # output
    out_root = _resolve_path(cfg["output"]["dir"], base)
    run_name = str(cfg["output"]["run_name"])
    out_dir = ensure_dir(out_root / run_name)
    ctgan_dir = ensure_dir(out_dir / "ctgan_pipeline")
    bootstrap_dir = ensure_dir(ctgan_dir / "bootstrap")
    predict_dir = ensure_dir(ctgan_dir / "_predict_only")

    # required inputs
    ctgan_static_path = _resolve_path(cfg["ctgan"]["static_parquet"], base)
    if not ctgan_static_path.exists():
        raise FileNotFoundError(f"ctgan.static_parquet not found: {ctgan_static_path}")

    checkpoint_path = _resolve_path(cfg["ctgan"]["checkpoint_path"], base)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"ctgan.checkpoint_path not found: {checkpoint_path}")

    repo_dir = _resolve_path(cfg["official"]["repo_dir"], base)
    if not repo_dir.exists():
        raise FileNotFoundError(
            f"official.repo_dir not found: {repo_dir}\n"
            f"Clone ARMD there in the notebook, or set official.repo_dir to the correct location."
        )

    real_long_parquet = _resolve_path(cfg["data"]["parquet_path"], base)
    if not real_long_parquet.exists():
        raise FileNotFoundError(f"data.parquet_path not found: {real_long_parquet}")

    # config_resolved.yaml: use one from training if present, otherwise write it now
    cfg_resolved = out_dir / "config_resolved.yaml"
    if not cfg_resolved.exists():
        with open(cfg_resolved, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    # split_ids.json from training run
    split_ids_path = out_dir / "split_ids.json"
    if not split_ids_path.exists():
        raise FileNotFoundError(
            f"split_ids.json not found at: {split_ids_path}\n"
            f"Run ardm_runner.py first (same out_dir/run_name) so it writes split_ids.json."
        )

    # Normalize split_ids format for bootstrap script -> {"train_ids":[...]}
    split_obj = json_load(split_ids_path)
    train_ids = split_obj.get("train_ids", None)
    if train_ids is None:
        if "split_ids" in split_obj and isinstance(split_obj["split_ids"], dict):
            train_ids = split_obj["split_ids"].get("train", None)
        elif "train" in split_obj:
            train_ids = split_obj.get("train")

    if not isinstance(train_ids, list):
        raise ValueError(
            f"Could not extract train ids list from {split_ids_path}. "
            f"Expected keys train_ids OR split_ids.train OR train."
        )

    train_ids_json = ctgan_dir / "train_ids_for_bootstrap.json"
    save_json({"train_ids": [str(x) for x in train_ids]}, train_ids_json)

    # bootstrap script (expects these args exactly)
    bootstrap_script = _resolve_path("scripts/bootstrap_ctgan_to_history.py", base)
    if not bootstrap_script.exists():
        raise FileNotFoundError(f"Missing bootstrap script: {bootstrap_script}")

    id_col = cfg["data"]["id_col"]
    time_col = cfg["data"]["time_col"]
    target_cols = list(cfg["data"]["target_cols"])
    history_len = int(cfg["data"]["history_len"])
    pred_len = int(cfg["data"]["pred_len"])

    cmd = [
        "python",
        str(bootstrap_script),
        "--ctgan_static_parquet",
        str(ctgan_static_path),
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
        str(int(cfg.get("seed", 42))),
        "--train_split_ids_json",
        str(train_ids_json),
        "--non_strict_baseline",
    ]

    print("[ctgan_reuse] bootstrap cmd:", " ".join(cmd))
    _run(cmd)

    windows_npy = bootstrap_dir / "bootstrap_windows_for_model.npy"
    if not windows_npy.exists():
        raise RuntimeError(f"Bootstrap did not write: {windows_npy}")

    # predict only (uses cfg_resolved if checkpoint lacks model config)
    pred_path = predict_only_official(
        repo_dir=repo_dir,
        checkpoint_path=checkpoint_path,
        test_npy_path=windows_npy,
        out_dir=predict_dir,
        history_len=history_len,
        pred_len=pred_len,
        batch_size=int(cfg["train"]["batch_size"]),
        device=str(cfg.get("device", "cuda")),
        config_yaml_path=cfg_resolved,
    )

    # build CTGAN ids order
    df_static = pd.read_parquet(ctgan_static_path)
    if id_col not in df_static.columns:
        raise ValueError(f"CTGAN static missing id_col='{id_col}'")
    ctgan_ids = df_static[id_col].astype(str).tolist()

    preds = np.load(pred_path)
    preds_3d = reshape_preds(preds, n_series=len(ctgan_ids), pred_len=pred_len, n_targets=len(target_cols))

    synth_future_long = build_ctgan_future_long(
        ctgan_ids=ctgan_ids,
        id_col=id_col,
        time_col=time_col,
        target_cols=target_cols,
        preds_3d=preds_3d,
        time_start=0,
    )
    out_parquet = ctgan_dir / "ctgan_synth_future_long.parquet"
    synth_future_long.to_parquet(out_parquet, index=False)

    meta = {
        "model": "ARDM_CTGAN_REUSE",
        "ctgan_static_parquet": str(ctgan_static_path),
        "official_repo_dir": str(repo_dir),
        "checkpoint_path": str(checkpoint_path),
        "bootstrap_windows_npy": str(windows_npy),
        "pred_npy": str(pred_path),
        "out_parquet": str(out_parquet),
        "n_ctgan_patients": int(len(ctgan_ids)),
        "history_len": int(history_len),
        "pred_len": int(pred_len),
        "targets": target_cols,
    }
    save_json(meta, ctgan_dir / "ctgan_run_meta.json")

    print(f"[ARDM+CTGAN] wrote: {ctgan_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
