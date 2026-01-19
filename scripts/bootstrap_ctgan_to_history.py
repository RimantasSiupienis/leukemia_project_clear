import sys
from pathlib import Path
import re

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]  # leukemia_project/
sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.bootstrap_utils import (
    set_seed,
    load_parquet,
    df_checks,
    build_windows,
    ensure_dir,
    save_json,
)


# -------------------------
# Small helpers
# -------------------------
def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def _parse_target_cols(s: str) -> List[str]:
    cols = [c.strip() for c in s.split(",") if c.strip()]
    if not cols:
        raise ValueError("target_cols is empty. Provide comma-separated list.")
    return cols


def _load_train_ids(train_split_ids_json: str) -> List[str]:
    obj = _read_json(train_split_ids_json)

    for k in ["train_ids", "train", "train_patient_ids", "train_subject_ids"]:
        v = obj.get(k)
        if isinstance(v, list):
            return [str(x) for x in v]

    raise ValueError(
        f"Could not find train ids list in {train_split_ids_json}. "
        "Expected a key like 'train_ids'."
    )


def _load_baseline_map(baseline_map_json: Optional[str]) -> Optional[Dict[str, str]]:
    if not baseline_map_json:
        return None
    m = _read_json(baseline_map_json)
    if not isinstance(m, dict):
        raise ValueError("baseline_map_json must be a JSON object {target_col: baseline_col}.")
    return {str(k): str(v) for k, v in m.items()}


def _static_row_to_baseline_vector(
    row: pd.Series,
    target_cols: List[str],
    baseline_map: Optional[Dict[str, str]],
    strict: bool,
) -> np.ndarray:
    """
    CTGAN static row -> baseline vector b (C,).

    Rules:
      - if target col exists directly in CTGAN static: use it
      - else if mapping provided: use mapped column
      - else error (strict) or NaN (non-strict)
    """
    b = np.zeros(len(target_cols), dtype=float)

    for j, tgt in enumerate(target_cols):
        if tgt in row.index:
            val = row[tgt]
        elif baseline_map is not None and tgt in baseline_map and baseline_map[tgt] in row.index:
            val = row[baseline_map[tgt]]
        else:
            if strict:
                raise ValueError(
                    f"Missing baseline value for target '{tgt}'. "
                    f"Either include '{tgt}' in CTGAN static output columns, "
                    f"or provide --baseline_map_json mapping for it."
                )
            b[j] = np.nan
            continue

        if pd.isna(val):
            if strict:
                raise ValueError(f"Baseline value is NaN for target '{tgt}'.")
            b[j] = np.nan
        else:
            b[j] = float(val)

    return b


def _normalize_train_ids_to_match_real_ids(train_ids: List[str], real_ids: pd.Series) -> List[str]:
    """
    If train_ids are like "A_12345" but real ids are "12345", normalize.
    Chooses the normalization that yields the most matches.
    """
    real_set = set(real_ids.astype(str).tolist())

    def score(ids: List[str]) -> int:
        return sum(1 for x in ids if x in real_set)

    ids_raw = [str(x) for x in train_ids]
    ids_split = [x.split("_", 1)[1] if "_" in x else x for x in ids_raw]
    ids_regex = [re.sub(r"^[A-Za-z]+_", "", x) for x in ids_raw]

    scored = [
        ("raw", score(ids_raw), ids_raw),
        ("split_underscore", score(ids_split), ids_split),
        ("regex_strip_prefix", score(ids_regex), ids_regex),
    ]
    scored.sort(key=lambda t: t[1], reverse=True)

    best_name, best_score, best_ids = scored[0]
    raw_score = next(s for (name, s, _) in scored if name == "raw")

    print(f"[bootstrap] train_id match counts: raw={raw_score}, best={best_score} ({best_name})")
    return best_ids if best_score > raw_score else ids_raw


def _build_template_bank(
    real_long: pd.DataFrame,
    train_ids: List[str],
    id_col: str,
    time_col: str,
    target_cols: List[str],
    history_len: int,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Returns:
      templates: (N_templates, history_len, C)
      template_means: (N_templates, C)
      meta: dict (counts, skipped ids, etc.)

    Note: This function pads shorter series on the LEFT (repeat first row) to reach history_len,
    then imputes within-window by ffill/bfill and falls back to global train means.
    """
    df = real_long.copy()
    df[id_col] = df[id_col].astype(str)

    train_set = set(str(x) for x in train_ids)
    df = df[df[id_col].isin(train_set)].copy()
    df = df.sort_values([id_col, time_col])

    templates: list[np.ndarray] = []
    kept_ids: list[str] = []
    skipped_ids: list[str] = []

    global_means = df[target_cols].mean(numeric_only=True).fillna(0.0)

    for pid, g in df.groupby(id_col, sort=False):
        g = g.sort_values(time_col)
        w = g[target_cols].iloc[-history_len:].copy()

        if len(w) < history_len:
            if len(w) == 0:
                skipped_ids.append(str(pid))
                continue
            pad_n = history_len - len(w)
            first_row = w.iloc[[0]].copy()
            pad_block = pd.concat([first_row] * pad_n, ignore_index=True)
            w = pd.concat([pad_block, w.reset_index(drop=True)], ignore_index=True)

        w = w.ffill().bfill()
        w = w.fillna(global_means)

        arr = w.to_numpy(dtype=float)
        if np.isnan(arr).any():
            skipped_ids.append(str(pid))
            continue

        templates.append(arr)
        kept_ids.append(str(pid))

    if not templates:
        raise RuntimeError(
            "No valid template patients found after filtering. "
            "Check that train_split_ids_json is correct and that patients have enough history."
        )

    templates_arr = np.stack(templates, axis=0)  # (N, L, C)
    means_arr = templates_arr.mean(axis=1)       # (N, C)

    meta = {
        "n_train_ids_provided": int(len(train_ids)),
        "n_rows_train_long": int(len(df)),
        "n_templates": int(len(templates_arr)),
        "n_skipped_short_or_nan": int(len(skipped_ids)),
        "example_skipped_ids": skipped_ids[:10],
        "example_kept_ids": kept_ids[:10],
    }
    return templates_arr, means_arr, meta


def _validate_outputs(history: np.ndarray, windows: np.ndarray, history_len: int, pred_len: int) -> None:
    if history.ndim != 3:
        raise ValueError(f"bootstrap_history.npy must be 3D, got shape {history.shape}")
    if windows.ndim != 3:
        raise ValueError(f"bootstrap_windows_for_model.npy must be 3D, got shape {windows.shape}")

    n, l, c = history.shape
    if l != history_len:
        raise ValueError(f"history_len mismatch: expected {history_len}, got {l}")
    if windows.shape != (n, history_len + pred_len, c):
        raise ValueError(f"windows shape mismatch: expected {(n, history_len + pred_len, c)}, got {windows.shape}")

    if np.isnan(history).any() or np.isnan(windows).any():
        raise ValueError("Found NaNs in bootstrap outputs.")

    future = windows[:, history_len:, :]
    if not np.allclose(future, 0.0):
        raise ValueError("Future part of bootstrap_windows_for_model.npy must be all zeros.")

    if np.allclose(history, 0.0):
        raise ValueError("All-zero histories detected (unexpected).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap CTGAN static patients into history windows for longitudinal models.")
    ap.add_argument("--ctgan_static_parquet", required=True, type=str)
    ap.add_argument("--real_long_parquet", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)

    ap.add_argument("--id_col", required=True, type=str)
    ap.add_argument("--time_col", required=True, type=str)
    ap.add_argument("--target_cols", required=True, type=str, help="Comma-separated target cols")

    ap.add_argument("--history_len", required=True, type=int)
    ap.add_argument("--pred_len", required=True, type=int)

    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--train_split_ids_json", required=True, type=str)

    ap.add_argument("--baseline_map_json", default=None, type=str)
    ap.add_argument("--ctgan_id_col", default=None, type=str, help="Optional CTGAN id column for logging only")
    ap.add_argument("--non_strict_baseline", action="store_true", help="If set, allow missing baseline values (uses template mean).")
    args = ap.parse_args()

    set_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))

    target_cols = _parse_target_cols(args.target_cols)
    history_len = int(args.history_len)
    pred_len = int(args.pred_len)
    id_col = str(args.id_col)
    time_col = str(args.time_col)

    ctgan_static = load_parquet(args.ctgan_static_parquet)
    real_long = load_parquet(args.real_long_parquet)

    df_checks(real_long, id_col, time_col)

    missing = [c for c in target_cols if c not in real_long.columns]
    if missing:
        raise ValueError(
            f"Real longitudinal parquet is missing target columns: {missing}. "
            f"Available cols (first 50): {list(real_long.columns)[:50]}"
        )

    train_ids = _load_train_ids(args.train_split_ids_json)
    train_ids = _normalize_train_ids_to_match_real_ids(train_ids, real_long[id_col])

    templates, template_means, template_meta = _build_template_bank(
        real_long=real_long,
        train_ids=train_ids,
        id_col=id_col,
        time_col=time_col,
        target_cols=target_cols,
        history_len=history_len,
    )

    baseline_map = _load_baseline_map(args.baseline_map_json)

    n_ctgan = len(ctgan_static)
    c = len(target_cols)
    bootstrap_history = np.zeros((n_ctgan, history_len, c), dtype=float)
    chosen_template_idx = np.zeros((n_ctgan,), dtype=int)

    strict = not bool(args.non_strict_baseline)

    for i in range(n_ctgan):
        row = ctgan_static.iloc[i]

        t_idx = int(rng.integers(0, templates.shape[0]))
        chosen_template_idx[i] = t_idx

        T = templates[t_idx]        # (L, C)
        mu = template_means[t_idx]  # (C,)

        b = _static_row_to_baseline_vector(row, target_cols, baseline_map, strict=strict)

        if (not strict) and np.isnan(b).any():
            mask = np.isnan(b)
            b[mask] = mu[mask]

        H = T - mu + b
        bootstrap_history[i] = H

    bootstrap_windows = np.zeros((n_ctgan, history_len + pred_len, c), dtype=float)
    bootstrap_windows[:, :history_len, :] = bootstrap_history

    _validate_outputs(bootstrap_history, bootstrap_windows, history_len, pred_len)

    np.save(out_dir / "bootstrap_history.npy", bootstrap_history)
    np.save(out_dir / "bootstrap_windows_for_model.npy", bootstrap_windows)
    np.save(out_dir / "chosen_template_idx.npy", chosen_template_idx)

    meta = {
        "seed": int(args.seed),
        "history_len": history_len,
        "pred_len": pred_len,
        "target_cols": target_cols,
        "ctgan_static_parquet": str(args.ctgan_static_parquet),
        "real_long_parquet": str(args.real_long_parquet),
        "train_split_ids_json": str(args.train_split_ids_json),
        "baseline_map_json": str(args.baseline_map_json) if args.baseline_map_json else None,
        "baseline_mapping_used": baseline_map if baseline_map is not None else "DIRECT_IF_PRESENT",
        "template_bootstrap_method": "random_train_template + mean_shift_to_ctgan_baseline",
        "n_ctgan_patients": int(n_ctgan),
        "template_meta": template_meta,
        "chosen_template_idx_path": "chosen_template_idx.npy",
        "notes": "future part of bootstrap_windows_for_model.npy is all zeros (masked) for the longitudinal model to generate.",
    }
    save_json(meta, str(out_dir / "bootstrap_meta.json"))

    print(f"[bootstrap] Wrote: {out_dir / 'bootstrap_history.npy'}")
    print(f"[bootstrap] Wrote: {out_dir / 'bootstrap_windows_for_model.npy'}")
    print(f"[bootstrap] Wrote: {out_dir / 'bootstrap_meta.json'}")


if __name__ == "__main__":
    main()
