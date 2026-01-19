from __future__ import annotations

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


def build_ctgan_future_long(
    ctgan_ids: List[str],
    id_col: str,
    time_col: str,
    target_cols: List[str],
    preds_3d: np.ndarray,
    time_start: int = 0,
) -> pd.DataFrame:
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
    # Want: [N, pred_len, C]
    if preds.ndim == 3:
        if preds.shape == (n_series, pred_len, n_targets):
            return preds
        if preds.shape == (n_series, n_targets, pred_len):
            return np.transpose(preds, (0, 2, 1))
        raise ValueError(f"Unexpected 3D preds shape: {preds.shape}")

    if preds.ndim == 2:
        if preds.shape[0] != n_series:
            raise ValueError(f"preds first dim {preds.shape[0]} != n_series {n_series}")
        if preds.shape[1] == pred_len * n_targets:
            return preds.reshape(n_series, pred_len, n_targets)
        raise ValueError(f"Unexpected 2D preds shape: {preds.shape}")

    raise ValueError(f"Unexpected preds ndim={preds.ndim}, shape={preds.shape}")


def main(cfg_path: str) -> None:
    cfg_path = str(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["seed"]))

    # output
    out_root = Path(cfg["output"]["dir"])
    run_name = str(cfg["output"]["run_name"])
    out_dir = ensure_dir(out_root / run_name)
    ensure_dir(out_dir / "_predict_only")

    # CTGAN static parquet
    ctgan_static_path = Path(cfg["ctgan"]["static_parquet"]).resolve()
    if not ctgan_static_path.exists():
        raise FileNotFoundError(f"ctgan.static_parquet not found: {ctgan_static_path}")

    # checkpoint
    checkpoint_path = Path(cfg["ctgan"]["checkpoint_path"]).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"ctgan.checkpoint_path not found: {checkpoint_path}")

    # official repo dir (must be cloned already by your notebook script OR present locally)
    repo_dir = Path(cfg["official"]["repo_dir"]).resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(
            f"official.repo_dir not found: {repo_dir}\n"
            f"Either clone ARMD into that folder in your notebook, or point repo_dir to where it already is."
        )

    id_col = cfg["data"]["id_col"]
    time_col = cfg["data"]["time_col"]
    target_cols = list(cfg["data"]["target_cols"])
    history_len = int(cfg["data"]["history_len"])
    pred_len = int(cfg["data"]["pred_len"])
    window_len = history_len + pred_len

    # Load CTGAN static (only for ids + maybe for bootstrap script)
    df_static = pd.read_parquet(ctgan_static_path)
    if id_col not in df_static.columns:
        raise ValueError(f"CTGAN static missing id_col='{id_col}'")
    ctgan_ids = df_static[id_col].astype(str).tolist()
    n_series = len(ctgan_ids)
    n_targets = len(target_cols)

    # Bootstrap windows using your existing script
    # REQUIRED: your script must output [N, window_len, C] into out_npy
    windows_npy = out_dir / "ctgan_bootstrap_windows.npy"
    bootstrap_script = Path("scripts/bootstrap_ctgan_to_history.py").resolve()
    if not bootstrap_script.exists():
        raise FileNotFoundError(f"Missing bootstrap script: {bootstrap_script}")

    _run([
        "python", str(bootstrap_script),
        "--static_parquet", str(ctgan_static_path),
        "--out_npy", str(windows_npy),
        "--id_col", str(id_col),
        "--target_cols", ",".join(target_cols),
        "--history_len", str(history_len),
        "--pred_len", str(pred_len),
    ])

    if not windows_npy.exists():
        raise RuntimeError("Bootstrap script did not create ctgan_bootstrap_windows.npy")

    # Predict only (GPU if cfg.device=cuda and available)
    pred_path = predict_only_official(
        repo_dir=repo_dir,
        checkpoint_path=checkpoint_path,
        test_npy_path=windows_npy,
        out_dir=out_dir / "_predict_only",
        history_len=history_len,
        pred_len=pred_len,
        batch_size=int(cfg["train"]["batch_size"]),
        device=str(cfg.get("device", "cuda")),
    )

    preds = np.load(pred_path)
    preds_3d = reshape_preds(preds, n_series=n_series, pred_len=pred_len, n_targets=n_targets)

    # Convert to long parquet trajectories
    synth_future_long = build_ctgan_future_long(
        ctgan_ids=ctgan_ids,
        id_col=id_col,
        time_col=time_col,
        target_cols=target_cols,
        preds_3d=preds_3d,
        time_start=0,
    )

    synth_future_long.to_parquet(out_dir / "ctgan_synth_future_long.parquet", index=False)

    meta = {
        "model": "ARDM_CTGAN_REUSE",
        "ctgan_static_parquet": str(ctgan_static_path),
        "official_repo_dir": str(repo_dir),
        "checkpoint_path": str(checkpoint_path),
        "bootstrap_windows_npy": str(windows_npy),
        "pred_npy": str(pred_path),
        "out_parquet": "ctgan_synth_future_long.parquet",
        "n_ctgan_patients": int(n_series),
        "history_len": int(history_len),
        "pred_len": int(pred_len),
        "targets": target_cols,
    }
    save_json(meta, out_dir / "ctgan_run_meta.json")

    print(f"[ARDM+CTGAN] wrote: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
