from __future__ import annotations

from typing import Any, List
import numpy as np
import pandas as pd


def reshape_preds(preds: np.ndarray, n_series: int, pred_len: int, n_targets: int) -> np.ndarray:
    """
    Returns [n_series, pred_len, n_targets]
    Accepts:
      - [n_series, pred_len, n_targets]
      - [n_series, n_targets, pred_len]
      - [n_series, pred_len * n_targets]
      - [n_series * pred_len, n_targets]
    """
    if preds.ndim == 3:
        if preds.shape == (n_series, pred_len, n_targets):
            return preds
        if preds.shape == (n_series, n_targets, pred_len):
            return np.transpose(preds, (0, 2, 1))
        raise ValueError(f"Unexpected 3D predictions shape: {preds.shape}")

    if preds.ndim == 2:
        if preds.shape == (n_series, pred_len * n_targets):
            return preds.reshape(n_series, pred_len, n_targets)
        if preds.shape == (n_series * pred_len, n_targets):
            return preds.reshape(n_series, pred_len, n_targets)
        raise ValueError(f"Unexpected 2D predictions shape: {preds.shape}")

    raise ValueError(f"Unexpected predictions ndim={preds.ndim}, shape={preds.shape}")


def build_ctgan_future_long(
    ctgan_ids: List[str],
    id_col: str,
    time_col: str,
    target_cols: List[str],
    preds_3d: np.ndarray,
    time_start: int = 0,
) -> pd.DataFrame:
    """
    Builds long format future table for CTGAN patients:
      columns: [id_col, time_col] + target_cols
      rows: N * pred_len
    time index = [time_start .. time_start+pred_len-1]
    """
    n, pred_len, c = preds_3d.shape
    if len(ctgan_ids) != n:
        raise ValueError(f"ctgan_ids length {len(ctgan_ids)} != preds_3d first dim {n}")
    if c != len(target_cols):
        raise ValueError(f"preds_3d last dim {c} != len(target_cols) {len(target_cols)}")

    rows: list[dict[str, Any]] = []
    for i, pid in enumerate(ctgan_ids):
        for t in range(pred_len):
            row = {id_col: str(pid), time_col: int(time_start + t)}
            for j, col in enumerate(target_cols):
                row[col] = float(preds_3d[i, t, j])
            rows.append(row)
    return pd.DataFrame(rows)
