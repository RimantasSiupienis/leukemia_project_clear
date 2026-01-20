import json
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_parquet(path):
    return pd.read_parquet(path)

def df_checks(df, id_col="patient_id", time_col="t"):
    if id_col not in df.columns or time_col not in df.columns:
        raise ValueError("Missing ID or time column")
    if df[time_col].isna().any():
        raise ValueError("Time column contains NaNs")
