import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from models.s4m_model.src.utils import load_parquet, df_checks


# Deterministic splitting helpers
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


# Preprocessing
def impute_groupwise_ffill_then_zero(df: pd.DataFrame, id_col: str, target_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    out[target_cols] = out.groupby(id_col)[target_cols].ffill().fillna(0.0)
    return out


def apply_debug_patient_limit(df: pd.DataFrame, cfg: dict, id_col: str) -> pd.DataFrame:
    dbg = cfg.get("data", {}).get("debug", None)
    if not dbg:
        return df

    max_patients = int(dbg.get("max_patients", 0) or 0)
    if max_patients <= 0:
        return df

    patient_ids = df[id_col].astype(str).unique().tolist()
    order = str(dbg.get("patient_order", "hash"))

    if order == "sorted":
        patient_ids = sorted(patient_ids)
    elif order == "hash":
        salt = str(dbg.get("hash_salt", ""))
        patient_ids = sorted(patient_ids, key=lambda x: stable_hash_to_u64(f"{salt}::{x}"))
    else:
        raise ValueError("data.debug.patient_order must be 'sorted' or 'hash'")

    keep = set(patient_ids[:max_patients])
    return df[df[id_col].astype(str).isin(keep)].copy()


# Windowing / packing
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
        g = df[df[id_col].astype(str) == str(pid)].sort_values(time_col)
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
) -> Tuple[np.ndarray, List[str]]:
    """
    Build packed windows of shape [N, window_len, C] using real future values.
    Windows are enumerated per patient in patient id order.
    Returns:
      X_all: np.ndarray [N, window_len, C)]
      win_patient_ids: list[str] length N (a patient id for each window)
    """
    X: List[np.ndarray] = []
    win_pids: List[str] = []

    for pid in patient_id_order:
        g = df[df[id_col].astype(str) == str(pid)].sort_values(time_col)
        vals = g[target_cols].to_numpy(dtype=np.float32)
        t = vals.shape[0]
        if t < window_len:
            continue
        for start in range(0, t - window_len + 1, stride):
            X.append(vals[start : start + window_len])
            win_pids.append(str(pid))

    if not X:
        return np.zeros((0, window_len, len(target_cols)), dtype=np.float32), []
    return np.stack(X, axis=0), win_pids


def compute_patient_window_ranges_from_lengths(
    patient_lengths: Dict[str, int],
    window_len: int,
    stride: int,
    patient_id_order: List[str],
) -> Dict[str, Tuple[int, int, int]]:
    """
    Returns dict pid [offset, n_windows, length] consistent with build_packed_windows enumeration order.
    """
    ranges: Dict[str, Tuple[int, int, int]] = {}
    offset = 0
    for pid in patient_id_order:
        t = int(patient_lengths.get(str(pid), 0))
        if t < window_len:
            n_win = 0
        else:
            n_win = ((t - window_len) // stride) + 1
        ranges[str(pid)] = (offset, n_win, t)
        offset += n_win
    return ranges


def select_window_indices_for_patients(
    patient_ranges: Dict[str, Tuple[int, int, int]],
    patient_ids: List[str],
) -> np.ndarray:
    idxs: List[int] = []
    for pid in patient_ids:
        off, n_win, _ = patient_ranges[str(pid)]
        if n_win > 0:
            idxs.extend(range(off, off + n_win))
    return np.asarray(idxs, dtype=np.int64)


def select_last_window_indices_per_patient(
    patient_ranges: Dict[str, Tuple[int, int, int]],
    patient_ids: List[str],
) -> np.ndarray:
    idxs: List[int] = []
    for pid in patient_ids:
        off, n_win, _ = patient_ranges[str(pid)]
        if n_win > 0:
            idxs.append(off + n_win - 1)
    return np.asarray(idxs, dtype=np.int64)


# Main dataset prep
def prep_dataset(cfg: dict) -> Tuple[str, str, str, dict]:
    data_cfg = cfg["data"]
    id_col = data_cfg["id_col"]
    time_col = data_cfg["time_col"]
    target_cols = list(data_cfg["target_cols"])

    history_len = int(data_cfg["history_len"])
    pred_len = int(data_cfg["pred_len"])
    stride = int(data_cfg["stride"])
    window_len = history_len + pred_len

    cache_root = Path("models/ardm_model/data_cache")
    cache_root.mkdir(parents=True, exist_ok=True)

    df = load_parquet(data_cfg["parquet_path"])
    df_checks(df, id_col, time_col)

    df = df.sort_values([id_col, time_col]).reset_index(drop=True)
    df[id_col] = df[id_col].astype(str)

    # debug limit before windowing(optional)
    df = apply_debug_patient_limit(df, cfg, id_col)

    # imputing before window packing
    df = impute_groupwise_ffill_then_zero(df, id_col, target_cols)

    all_patient_ids = df[id_col].unique().tolist()
    split_ids = split_patient_ids(all_patient_ids, data_cfg["split"])

    # keeping enumeration order stable
    patient_id_order = all_patient_ids

    patient_lengths = df.groupby(id_col, sort=False).size().to_dict()
    patient_lengths = {str(k): int(v) for k, v in patient_lengths.items()}

    # Packing real windows for all patients
    X_all_real, _ = build_packed_windows(
        df=df,
        id_col=id_col,
        time_col=time_col,
        target_cols=target_cols,
        window_len=window_len,
        stride=stride,
        patient_id_order=patient_id_order,
    )

    def _zero_future(x: np.ndarray) -> np.ndarray:
        if x.shape[1] != window_len:
            raise ValueError("Unexpected window length while masking future")
        x2 = x.copy()
        x2[:, history_len:, :] = 0.0
        return x2

    patient_ranges = compute_patient_window_ranges_from_lengths(
        patient_lengths=patient_lengths,
        window_len=window_len,
        stride=stride,
        patient_id_order=patient_id_order,
    )

    train_idxs = select_window_indices_for_patients(patient_ranges, split_ids["train"])
    val_idxs = select_window_indices_for_patients(patient_ranges, split_ids["val"])
    test_idxs = select_last_window_indices_per_patient(patient_ranges, split_ids["test"])

    cache_key_payload = {
        "parquet_path": str(data_cfg["parquet_path"]),
        "id_col": id_col,
        "time_col": time_col,
        "target_cols": target_cols,
        "history_len": history_len,
        "pred_len": pred_len,
        "stride": stride,
        "split": data_cfg["split"],
        "debug": cfg.get("data", {}).get("debug", None),
        "packing": "real_future_for_train_val__zero_future_for_test",
    }
    cache_key = hashlib.sha256(json.dumps(cache_key_payload, sort_keys=True).encode()).hexdigest()[:12]
    cache_dir = cache_root / f"ardm_{cache_key}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_path = cache_dir / "train.npy"
    val_path = cache_dir / "val.npy"
    test_path = cache_dir / "test.npy"
    real_future_path = cache_dir / "real_future_long.parquet"

    np.save(train_path, X_all_real[train_idxs])
    np.save(val_path, X_all_real[val_idxs])
    np.save(test_path, _zero_future(X_all_real[test_idxs]))

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
            "test_policy": "last_window_per_patient",
            "train_val_policy": "all_windows_per_patient",
        },
        "packing": {
            "train_val": "history+real_future",
            "test": "history+zeros (future masked)",
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
