from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from models.s4m_model.src.utils import load_parquet, df_checks


def stable_hash_to_u64(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def split_patient_ids(ids: List[str], split_cfg: dict) -> Dict[str, List[str]]:
    ids = [str(x) for x in ids]

    split_type = split_cfg["type"]
    train_frac = float(split_cfg["train"])
    val_frac = float(split_cfg["val"])
    test_frac = float(split_cfg["test"])

    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-9:
        raise ValueError("data split fractions have to sum to 1")

    n = len(ids)
    n_train = int(np.floor(n * train_frac))
    n_val = int(np.floor(n * val_frac))
    n_test = n - n_train - n_val

    if split_type == "patient_random":
        seed = int(split_cfg.get("seed", 0))
        rng = np.random.default_rng(seed)
        perm = np.array(ids, dtype=object)
        rng.shuffle(perm)
        perm = perm.tolist()
        return {
            "train": perm[:n_train],
            "val": perm[n_train : n_train + n_val],
            "test": perm[n_train + n_val : n_train + n_val + n_test],
        }

    if split_type == "patient_hash":
        salt = str(split_cfg.get("salt", ""))
        scored = [(pid, stable_hash_to_u64(f"{salt}::{pid}")) for pid in ids]
        scored.sort(key=lambda x: x[1])
        ordered = [pid for pid, _ in scored]
        return {
            "train": ordered[:n_train],
            "val": ordered[n_train : n_train + n_val],
            "test": ordered[n_train + n_val : n_train + n_val + n_test],
        }

    raise ValueError(f"Unknown data split type='{split_type}'")


def impute_groupwise_ffill_then_zero(df: pd.DataFrame, id_col: str, target_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    out[target_cols] = out.groupby(id_col, sort=False)[target_cols].ffill().fillna(0.0)
    return out


def build_real_future_long(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    target_cols: List[str],
    history_len: int,
    pred_len: int,
    test_patient_ids: List[str],
) -> pd.DataFrame:
    rows = []
    window_len = history_len + pred_len
    for pid in test_patient_ids:
        g = df[df[id_col] == str(pid)].sort_values(time_col)
        if len(g) < window_len:
            continue
        last_block = g.iloc[-window_len:]
        future = last_block.iloc[history_len:]
        rows.append(future[[id_col, time_col] + target_cols])
    if not rows:
        return pd.DataFrame(columns=[id_col, time_col] + target_cols)
    return pd.concat(rows, axis=0, ignore_index=True)


def build_packed_windows(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    target_cols: List[str],
    window_len: int,
    stride: int,
    patient_id_order: List[str],
) -> np.ndarray:
    """
    Returns X_all shape [N, window_len, C]
    Enumerates windows per patient in patient_id_order.
    """
    X: List[np.ndarray] = []
    for pid in patient_id_order:
        g = df[df[id_col] == str(pid)].sort_values(time_col)
        vals = g[target_cols].to_numpy(dtype=np.float32)
        t = vals.shape[0]
        if t < window_len:
            continue
        for start in range(0, t - window_len + 1, stride):
            X.append(vals[start : start + window_len])
    if not X:
        return np.zeros((0, window_len, len(target_cols)), dtype=np.float32)
    return np.stack(X, axis=0)


def compute_patient_window_ranges(
    df: pd.DataFrame,
    id_col: str,
    window_len: int,
    stride: int,
    patient_id_order: List[str],
) -> Dict[str, Tuple[int, int]]:
    """
    Returns dict pid -> (offset, n_windows) consistent with build_packed_windows enumeration.
    """
    ranges: Dict[str, Tuple[int, int]] = {}
    offset = 0
    for pid in patient_id_order:
        g = df[df[id_col] == str(pid)]
        t = len(g)
        if t < window_len:
            n_win = 0
        else:
            n_win = ((t - window_len) // stride) + 1
        ranges[str(pid)] = (offset, n_win)
        offset += n_win
    return ranges


def select_window_indices_for_patients(
    patient_ranges: Dict[str, Tuple[int, int]],
    patient_ids: List[str],
) -> np.ndarray:
    idxs: List[int] = []
    for pid in patient_ids:
        off, n_win = patient_ranges[str(pid)]
        if n_win > 0:
            idxs.extend(range(off, off + n_win))
    return np.asarray(idxs, dtype=np.int64)


def select_last_window_indices_per_patient(
    patient_ranges: Dict[str, Tuple[int, int]],
    patient_ids: List[str],
) -> np.ndarray:
    idxs: List[int] = []
    for pid in patient_ids:
        off, n_win = patient_ranges[str(pid)]
        if n_win > 0:
            idxs.append(off + n_win - 1)
    return np.asarray(idxs, dtype=np.int64)


def prep_dataset(cfg: dict) -> Tuple[str, str, str, dict]:
    data_cfg = cfg["data"]
    id_col = str(data_cfg["id_col"])
    time_col = str(data_cfg["time_col"])
    target_cols = list(data_cfg["target_cols"])

    history_len = int(data_cfg["history_len"])
    pred_len = int(data_cfg["pred_len"])
    stride = int(data_cfg["stride"])
    window_len = history_len + pred_len

    cache_root = Path("models/ardm_model/data_cache")
    cache_root.mkdir(parents=True, exist_ok=True)

    df = load_parquet(data_cfg["parquet_path"])
    df_checks(df, id_col, time_col)

    # enforce stable types + ordering
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)
    df[id_col] = df[id_col].astype(str)

    # impute targets
    df = impute_groupwise_ffill_then_zero(df, id_col, target_cols)

    all_patient_ids = df[id_col].drop_duplicates().tolist()
    split_ids = split_patient_ids(all_patient_ids, data_cfg["split"])

    patient_id_order = all_patient_ids
    patient_ranges = compute_patient_window_ranges(
        df=df, id_col=id_col, window_len=window_len, stride=stride, patient_id_order=patient_id_order
    )

    # Build all windows once
    X_all = build_packed_windows(
        df=df,
        id_col=id_col,
        time_col=time_col,
        target_cols=target_cols,
        window_len=window_len,
        stride=stride,
        patient_id_order=patient_id_order,
    )

    train_idxs = select_window_indices_for_patients(patient_ranges, split_ids["train"])
    val_idxs = select_window_indices_for_patients(patient_ranges, split_ids["val"])
    test_idxs = select_last_window_indices_per_patient(patient_ranges, split_ids["test"])

    # test masking: zero out future to avoid leakage when sampling forecast
    X_test = X_all[test_idxs].copy()
    X_test[:, history_len:, :] = 0.0

    cache_key_payload = {
        "parquet_path": str(data_cfg["parquet_path"]),
        "id_col": id_col,
        "time_col": time_col,
        "target_cols": target_cols,
        "history_len": history_len,
        "pred_len": pred_len,
        "stride": stride,
        "split": data_cfg["split"],
        "packing": "history+future for train/val ; history+zeros for test",
    }
    cache_key = hashlib.sha256(json.dumps(cache_key_payload, sort_keys=True).encode()).hexdigest()[:12]
    cache_dir = cache_root / f"ardm_{cache_key}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_path = cache_dir / "train.npy"
    val_path = cache_dir / "val.npy"
    test_path = cache_dir / "test.npy"
    real_future_path = cache_dir / "real_future_long.parquet"

    np.save(train_path, X_all[train_idxs])
    np.save(val_path, X_all[val_idxs])
    np.save(test_path, X_test)

    real_future_long = build_real_future_long(
        df=df,
        id_col=id_col,
        time_col=time_col,
        target_cols=target_cols,
        history_len=history_len,
        pred_len=pred_len,
        test_patient_ids=split_ids["test"],
    )
    real_future_long.to_parquet(real_future_path, index=False)

    manifest = {
        "cache_key": cache_key,
        "paths": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "real_future_long": str(real_future_path),
        },
        "split_ids": split_ids,
        "windowing": {
            "history_len": history_len,
            "pred_len": pred_len,
            "stride": stride,
            "window_len": window_len,
            "test_policy": "last_window_per_patient (future masked to zeros)",
            "train_val_policy": "all_windows_per_patient",
        },
        "n_windows": {
            "train": int(len(train_idxs)),
            "val": int(len(val_idxs)),
            "test": int(len(test_idxs)),
        },
    }

    with open(cache_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return str(train_path), str(val_path), str(test_path), manifest
