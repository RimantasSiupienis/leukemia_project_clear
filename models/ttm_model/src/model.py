from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _save_yaml(obj: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def _basic_df_checks(df: pd.DataFrame, id_col: str, time_col: str, target_cols: List[str]) -> None:
    missing = [c for c in [id_col, time_col] + target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in long parquet: {missing}")

    if df.empty:
        raise ValueError("Loaded dataframe is empty.")

    # enforces types
    df[id_col] = df[id_col].astype(str)
    df[time_col] = pd.to_numeric(df[time_col], errors="raise").astype(int)

    # sorts
    df.sort_values([id_col, time_col], inplace=True)


def _patient_level_split(
    patient_ids: List[str], seed: int, train_ratio: float, val_ratio: float, test_ratio: float
) -> Dict[str, List[str]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    ids = np.array(patient_ids, dtype=object)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"Bad split sizes: n={n}, train={n_train}, val={n_val}, test={n_test}")

    train_ids = ids[:n_train].tolist()
    val_ids = ids[n_train : n_train + n_val].tolist()
    test_ids = ids[n_train + n_val :].tolist()

    return {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}


def _impute_per_patient(df: pd.DataFrame, id_col: str, time_col: str, target_cols: List[str]) -> pd.DataFrame:
    """
    Anti-nan imputation since TTM cant handle NaNs:
      per patient: sort by time, forward fill then backward fill
      remaining nans filled with global column median
      if a column median is nan, fall back to 0.0
      if any NaNs remain, fill with 0.0 and warn but tbh if we reach that point were probably screwed anyway
    """
    d = df[[id_col, time_col] + target_cols].copy()
    d.sort_values([id_col, time_col], inplace=True)

    # global medians
    med: Dict[str, float] = {}
    for c in target_cols:
        s = pd.to_numeric(d[c], errors="coerce")
        m = float(s.median()) if not s.empty else float("nan")
        if not np.isfinite(m):  # catches NaN/inf
            m = 0.0
        med[c] = m

    out_parts = []
    for _, g in d.groupby(id_col, sort=False):
        gg = g.copy()
        for c in target_cols:
            gg[c] = pd.to_numeric(gg[c], errors="coerce")
            gg[c] = gg[c].ffill().bfill()
            gg[c] = gg[c].fillna(med[c])
        out_parts.append(gg)

    out = pd.concat(out_parts, axis=0, ignore_index=True)
    out.sort_values([id_col, time_col], inplace=True)

    # last safety check
    nan_counts = out[target_cols].isna().sum()
    if int(nan_counts.sum()) > 0:
        # If this triggers, something is very wrong with the data.
        bad = {k: int(v) for k, v in nan_counts.to_dict().items() if int(v) > 0}
        print(f"[ttm][warn] NaNs remained after imputation, filling with 0.0: {bad}")
        out[target_cols] = out[target_cols].fillna(0.0)

    return out


def _make_history_future(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    target_cols: List[str],
    patient_ids: List[str],
    history_len: int,
    pred_len: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      hist_df: rows where t in [0, history_len-1] for chosen patients
      fut_df: rows where t in [history_len, history_len+pred_len-1] for chosen patients
    Assumes fixed-grid data, but works as long as those rows exist.
    """
    d = df[df[id_col].isin(patient_ids)].copy()

    hist = d[(d[time_col] >= 0) & (d[time_col] < history_len)].copy()
    fut = d[(d[time_col] >= history_len) & (d[time_col] < history_len + pred_len)].copy()

    # requires full coverage per patient
    need_hist = history_len
    need_fut = pred_len
    ok_ids = []
    for pid, g in d.groupby(id_col):
        n_hist = int(((g[time_col] >= 0) & (g[time_col] < history_len)).sum())
        n_fut = int(((g[time_col] >= history_len) & (g[time_col] < history_len + pred_len)).sum())
        if n_hist == need_hist and n_fut == need_fut:
            ok_ids.append(pid)

    if not ok_ids:
        raise RuntimeError(
            "No patients have full (history_len + pred_len) coverage. "
            "Check your dataset grid / task lengths."
            "At that point just die"
        )

    hist = hist[hist[id_col].isin(ok_ids)].copy()
    fut = fut[fut[id_col].isin(ok_ids)].copy()

    # keep only needed cols
    hist = hist[[id_col, time_col] + target_cols].sort_values([id_col, time_col]).reset_index(drop=True)
    fut = fut[[id_col, time_col] + target_cols].sort_values([id_col, time_col]).reset_index(drop=True)
    return hist, fut


def _to_panel_multiindex(df: pd.DataFrame, id_col: str, time_col: str, target_cols: List[str]) -> pd.DataFrame:
    """
    sktime panel format pd-multiindex: index = (instance, time)
    """
    y = df.set_index([id_col, time_col])[target_cols].sort_index()
    return y


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("config_yaml", type=str)
    args = ap.parse_args()

    cfg_path = Path(args.config_yaml).expanduser().resolve()
    cfg = _load_yaml(cfg_path)

    seed = int(cfg.get("seed", 42))
    _set_seed(seed)

    long_path = Path(cfg["data"]["long_path"]).expanduser()
    id_col = str(cfg["data"]["id_col"])
    time_col = str(cfg["data"]["time_col"])
    target_cols = [str(x) for x in cfg["data"]["target_cols"]]

    history_len = int(cfg["task"]["history_len"])
    pred_len = int(cfg["task"]["pred_len"])

    out_dir = Path(cfg["output"]["dir"]).expanduser()
    _ensure_dir(out_dir)

    # saves resolved config for reproducibility
    _save_yaml(cfg, out_dir / "config_resolved.yaml")

    df = pd.read_parquet(long_path)
    _basic_df_checks(df, id_col=id_col, time_col=time_col, target_cols=target_cols)

    # imputes 
    df = _impute_per_patient(df, id_col=id_col, time_col=time_col, target_cols=target_cols)

    patient_ids = sorted(df[id_col].unique().tolist())
    split_cfg = cfg["split"]
    splits = _patient_level_split(
        patient_ids=patient_ids,
        seed=seed,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
    )
    _save_json(splits, out_dir / "split_ids.json")

    # forecast only on test patients 
    # one trajectory each
    test_ids = splits["test_ids"]
    hist_df, real_future_df = _make_history_future(
        df=df,
        id_col=id_col,
        time_col=time_col,
        target_cols=target_cols,
        patient_ids=test_ids,
        history_len=history_len,
        pred_len=pred_len,
    )

    # TTM via sktime
    try:
        from sktime.forecasting.ttm import TinyTimeMixerForecaster
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: sktime (and its TTM extra deps).\n"
            "Install something like:\n"
            "  pip install sktime[all_extras]\n"
            "and ensure transformers/torch are available.\n"
            f"Original import error: {e}"
        )

    ttm_cfg = cfg.get("ttm", {})
    model_path = ttm_cfg.get("model_path", "ibm/TTM")
    revision = ttm_cfg.get("revision", "main")
    fit_strategy = ttm_cfg.get("fit_strategy", "zero-shot")
    training_args = dict(ttm_cfg.get("training_args", {}) or {})

    # HF Trainer needs output_dir if any training happens
    if fit_strategy in ("minimal", "full"):
        training_args.setdefault("output_dir", str(out_dir / "_ttm_finetune"))
        _ensure_dir(Path(training_args["output_dir"]))

    fh = list(range(1, pred_len + 1))

    # panel history (panda multiindex)
    y_hist = _to_panel_multiindex(hist_df, id_col=id_col, time_col=time_col, target_cols=target_cols)

    forecaster = TinyTimeMixerForecaster(
        model_path=model_path,
        revision=revision,
        fit_strategy=fit_strategy,
        training_args=training_args,
        broadcasting=True, # fit/predict per series internally
    )

    # For zero-shot, fit is fast no training 
    # For fine-tune it uses provided training_args
    print("[ttm] starting fit...")
    print(
        "[ttm] training_args:",
        {k: training_args.get(k) for k in ["num_train_epochs", "per_device_train_batch_size"]}
    )

    forecaster.fit(y_hist, fh=fh)
    print("[ttm] fit done, starting predict...")
    y_pred = forecaster.predict()
    print("[ttm] predict done, writing outputs...")

    # Converts predictions to long format and forces timestamp alignment to real_future_df
    pred_panel = y_pred.copy().reset_index()

    # sktime returns MultiIndex [patient_id, fh] where fh is typically 1 to pred_len
    pred_panel.columns = [id_col, "fh"] + target_cols
    pred_panel[id_col] = pred_panel[id_col].astype(str)
    pred_panel["fh"] = pred_panel["fh"].astype(int)

    # Builds per patient list of real future times 
    # guarantees matching keys
    real_future_df[id_col] = real_future_df[id_col].astype(str)
    real_times = (
        real_future_df[[id_col, time_col]]
        .sort_values([id_col, time_col])
        .groupby(id_col)[time_col]
        .apply(list)
        .to_dict()
    )

    # ensuring each patient has exactly pred_len future points
    bad = {pid: len(ts) for pid, ts in real_times.items() if len(ts) != pred_len}
    if bad:
        raise RuntimeError(
            f"Some patients do not have exactly pred_len={pred_len} future points: "
            f"{list(bad.items())[:10]}"
        )

    # Mapping predictions to real future timestamps by position, not fh numeric value
    # This handles cases where fh is absolute instead of 1..pred_len.
    pred_panel = pred_panel.sort_values([id_col, "fh"]).reset_index(drop=True)

    # 0..pred_len-1 within each patient
    pred_panel["_step"] = pred_panel.groupby(id_col).cumcount()

    # Ensuring each patient has exactly pred_len predictions
    bad_pred = pred_panel.groupby(id_col)["_step"].max()
    bad_pred = bad_pred[bad_pred != (pred_len - 1)]
    if len(bad_pred) > 0:
        example = bad_pred.head(10).to_dict()
        raise RuntimeError(
            f"Predictions do not have exactly pred_len={pred_len} rows per patient. Example max steps: {example}"
        )

    pred_panel[time_col] = pred_panel.apply(
        lambda r: real_times[r[id_col]][int(r["_step"])],
        axis=1,
    )

    pred_panel = pred_panel.drop(columns=["fh", "_step"])

    pred_panel = pred_panel.drop(columns=["fh"], errors="ignore")
    pred_panel = (
        pred_panel[[id_col, time_col] + target_cols]
        .sort_values([id_col, time_col])
        .reset_index(drop=True)
    )
    # Saves outputs for your evaluation pipeline
    real_future_path = out_dir / "real_future_long.parquet"
    synth_future_path = out_dir / "synth_future_long.parquet"
    meta_path = out_dir / "run_meta.json"

    # enforces identical ordering for evaluation
    real_future_df[id_col] = real_future_df[id_col].astype(str)
    pred_panel[id_col] = pred_panel[id_col].astype(str)

    real_future_df = real_future_df.sort_values([id_col, time_col]).reset_index(drop=True)
    pred_panel = pred_panel.sort_values([id_col, time_col]).reset_index(drop=True)

    real_future_df.to_parquet(real_future_path, index=False)
    pred_panel.to_parquet(synth_future_path, index=False)


    meta = {
        "model": "ttm",
        "seed": seed,
        "long_path": str(long_path),
        "id_col": id_col,
        "time_col": time_col,
        "target_cols": target_cols,
        "history_len": history_len,
        "pred_len": pred_len,
        "n_test_patients": int(len(pred_panel[id_col].unique())),
        "ttm": {
            "model_path": model_path,
            "revision": revision,
            "fit_strategy": fit_strategy,
            "training_args": training_args,
        },
        "outputs": {
            "real_future_long": str(real_future_path),
            "synth_future_long": str(synth_future_path),
        },
    }
    _save_json(meta, meta_path)

    print(f"[ttm] wrote: {real_future_path}")
    print(f"[ttm] wrote: {synth_future_path}")
    print(f"[ttm] wrote: {out_dir / 'split_ids.json'}")
    print(f"[ttm] wrote: {meta_path}")


if __name__ == "__main__":
    main()
