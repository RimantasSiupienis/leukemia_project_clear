
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    """Deterministic-ish runs (as much as PyTorch/NumPy allow)."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # optional
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # make cuDNN deterministic if present
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass


def ensure_dir(path: str | Path) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_parquet(path: str | Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    p = str(path)
    return pd.read_parquet(p, columns=list(columns) if columns else None)


def df_checks(df: pd.DataFrame, id_col: str, time_col: str) -> None:
    # minimal checks only (don’t “fix” data)
    if id_col not in df.columns:
        raise ValueError(f"Missing id_col='{id_col}' in dataframe columns={list(df.columns)}")
    if time_col not in df.columns:
        raise ValueError(f"Missing time_col='{time_col}' in dataframe columns={list(df.columns)}")
    if df.empty:
        raise ValueError("Dataframe is empty")
    # allow numeric or sortable times
    try:
        _ = df[[id_col, time_col]].head(1)
    except Exception as e:
        raise ValueError(f"Dataframe cannot access [{id_col}, {time_col}]") from e


def build_windows(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    cont_cols: List[str],
    cat_cols: Optional[List[str]] = None,
    static_cont_cols: Optional[List[str]] = None,
    static_cat_cols: Optional[List[str]] = None,
    lookback_len: int = 128,
    horizon_len: int = 24,
    step_stride: int = 1,
    build_missing_mask: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Window a long dataframe into sliding windows per patient.

    Returns:
      X_past:   (N, lookback_len, C)
      Y_future: (N, horizon_len, C)
      M_past:   (N, lookback_len, C)  1=observed, 0=missing (only for cont cols)
      M_future: (N, horizon_len, C)
      ids:      (N,) patient id per window (dtype=object)
    """
    if cat_cols or static_cont_cols or static_cat_cols:
        # Keep signature-compatible with your pipeline but do not implement categorical/static here.
        raise ValueError("build_windows in this project expects only cont_cols for now (cat/static not supported).")

    lookback_len = int(lookback_len)
    horizon_len = int(horizon_len)
    step_stride = int(step_stride)
    if lookback_len <= 0 or horizon_len <= 0:
        raise ValueError("lookback_len and horizon_len must be > 0")
    if step_stride <= 0:
        raise ValueError("step_stride must be > 0")

    need_cols = [id_col, time_col] + list(cont_cols)
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df[need_cols].copy()
    d[id_col] = d[id_col].astype(str)
    d = d.sort_values([id_col, time_col]).reset_index(drop=True)

    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    MX_list: List[np.ndarray] = []
    MY_list: List[np.ndarray] = []
    ID_list: List[str] = []

    window_len = lookback_len + horizon_len

    for pid, g in d.groupby(id_col, sort=False):
        g = g.sort_values(time_col)
        vals = g[cont_cols].to_numpy(dtype=np.float32)

        if build_missing_mask:
            mask = (~np.isnan(vals)).astype(np.float32)
            vals = np.nan_to_num(vals, nan=0.0).astype(np.float32)
        else:
            mask = np.ones_like(vals, dtype=np.float32)

        T = vals.shape[0]
        if T < window_len:
            continue

        for start in range(0, T - window_len + 1, step_stride):
            block = vals[start : start + window_len]
            mblock = mask[start : start + window_len]

            x = block[:lookback_len]
            y = block[lookback_len:]
            mx = mblock[:lookback_len]
            my = mblock[lookback_len:]

            X_list.append(x)
            Y_list.append(y)
            MX_list.append(mx)
            MY_list.append(my)
            ID_list.append(pid)

    if not X_list:
        C = len(cont_cols)
        X = np.zeros((0, lookback_len, C), dtype=np.float32)
        Y = np.zeros((0, horizon_len, C), dtype=np.float32)
        MX = np.zeros((0, lookback_len, C), dtype=np.float32)
        MY = np.zeros((0, horizon_len, C), dtype=np.float32)
        IDS = np.zeros((0,), dtype=object)
        return X, Y, MX, MY, IDS

    X = np.stack(X_list, axis=0).astype(np.float32)
    Y = np.stack(Y_list, axis=0).astype(np.float32)
    MX = np.stack(MX_list, axis=0).astype(np.float32)
    MY = np.stack(MY_list, axis=0).astype(np.float32)
    IDS = np.asarray(ID_list, dtype=object)

    return X, Y, MX, MY, IDS
