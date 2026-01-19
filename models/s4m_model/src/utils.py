# Minimal shared utils used across runners (ARDM, etc.)
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_parquet(path: str, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns) if columns is not None else None)


def df_checks(df: pd.DataFrame, id_col: str, time_col: str) -> None:
    if id_col not in df.columns:
        raise ValueError(f"Missing id_col='{id_col}' in parquet columns: {list(df.columns)[:50]}")
    if time_col not in df.columns:
        raise ValueError(f"Missing time_col='{time_col}' in parquet columns: {list(df.columns)[:50]}")
    if df.empty:
        raise ValueError("DataFrame is empty")
