# scripts/bootstrap_utils.py

import json
import numpy as np
import pandas as pd
from pathlib import Path
import random


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_parquet(path):
    return pd.read_parquet(path)


def df_checks(df, id_col, time_col):
    assert id_col in df.columns
    assert time_col in df.columns


def build_windows(arr, history_len, pred_len):
    n, t, c = arr.shape
    out = np.zeros((n, history_len + pred_len, c), dtype=np.float32)
    out[:, :history_len, :] = arr[:, :history_len, :]
    return out
