from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import yaml
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.chronos import ChronosForecaster


# Repo utils
def repo_root() -> Path:
    return Path(__file__).resolve().parents[3] # models/chronos_model/src/model.py to repo root


def import_shared_utils():
    root = repo_root()
    sys.path.insert(0, str(root))
    from models.s4m_model.src.utils import (  # type: ignore
        set_seed,
        ensure_dir,
        load_parquet,
        save_json,
        df_checks,
    )
    return set_seed, ensure_dir, load_parquet, save_json, df_checks


set_seed, ensure_dir, load_parquet, save_json, df_checks = import_shared_utils()

# Config utils
def load_cfg(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p: str) -> Path:
    root = repo_root()
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp)

def fill_missing_1d(s: pd.Series) -> pd.Series:
    """
    ChronosForecaster rejects NaNs.
    Keep it simple: ffill to bfill to leftover NaNs to 0.0.
    """
    if s.isna().any():
        s = s.ffill().bfill().fillna(0.0)
    return s


# Data splitting 
def split_patients(
    ids: List[Any],
    method: str,
    test_fraction: float,
    seed: int,
    hash_salt: str,
) -> Tuple[List[str], List[str]]:
    ids = [str(x) for x in ids]
    method = str(method).strip().lower()

    if not (0.0 < float(test_fraction) < 1.0):
        raise ValueError("split.test_fraction has to be in between 0 and 1")

    if method == "patient_random":
        rng = np.random.default_rng(int(seed))
        ids_shuf = ids.copy()
        rng.shuffle(ids_shuf)
        n_test = int(round(len(ids_shuf) * float(test_fraction)))
        test_ids = ids_shuf[:n_test]
        train_ids = ids_shuf[n_test:]
        return train_ids, test_ids

    if method == "patient_hash":
        salt = str(hash_salt) if hash_salt is not None else ""

        def score(pid: str) -> float:
            h = hashlib.sha256(f"{salt}|{pid}".encode("utf-8")).hexdigest()
            return int(h[:8], 16) / float(16**8)

        scored = [(score(pid), pid) for pid in ids]
        scored.sort(key=lambda t: t[0])
        n_test = int(round(len(scored) * float(test_fraction)))
        test_ids = [pid for _, pid in scored[:n_test]]
        train_ids = [pid for _, pid in scored[n_test:]]
        return train_ids, test_ids

    raise ValueError("split.method can only be patient_random or patient_hash.")

# Core logic: converting various sktime outputs to 1d float numpy array
def as_float_1d(x: Any) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("Expected univariate output per target column.")
        arr = x.iloc[:, 0].to_numpy()
    elif isinstance(x, pd.Series):
        arr = x.to_numpy()
    else:
        arr = np.asarray(x)
    return arr.astype(float, copy=False).reshape(-1)


def main(cfg_path: str | None = None) -> None:
    here = Path(__file__).resolve().parent
    if cfg_path is None:
        cfg_path = str(here.parent / "configs" / "chronos_base.yaml")

    cfg_file = resolve_path(cfg_path)
    cfg = load_cfg(cfg_file)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    data_cfg = cfg["data"]
    task_cfg = cfg.get("task", {})
    split_cfg = cfg.get("split", {})
    model_cfg = cfg.get("model", {})
    out_cfg = cfg.get("output", {})

    id_col = str(data_cfg["id_col"])
    time_col = str(data_cfg["time_col"])
    target_cols = data_cfg.get("target_cols", ["target"])
    if isinstance(target_cols, str):
        target_cols = [target_cols]
    target_cols = [str(c) for c in target_cols]

    long_path = resolve_path(str(data_cfg["long_path"]))

    context_len = int(task_cfg.get("context_len", 96))
    horizon = int(task_cfg.get("horizon", 24))
    min_history = int(task_cfg.get("min_history", context_len + horizon))

    split_method = str(split_cfg.get("method", "patient_hash"))
    split_seed = int(split_cfg.get("seed", seed))
    test_fraction = float(split_cfg.get("test_fraction", 0.2))
    hash_salt = str(split_cfg.get("hash_salt", "chronos"))

    out_root = resolve_path(str(out_cfg.get("dir", "models/chronos_model/results")))
    run_name = str(out_cfg.get("run_name", "chronos_run"))
    save_preds_long = bool(out_cfg.get("save_preds_long", False))

    run_dir = out_root / run_name
    ensure_dir(run_dir)


    # leaving a trace so the run dir looks alive while the loop is still running
    with open(run_dir / "_run_started.txt", "w", encoding="utf-8") as f:
        f.write("chronos run started\n")

    # config snapshot early
    with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    t0 = time.time()

    df = load_parquet(str(long_path))
    df_checks(df, id_col=id_col, time_col=time_col)

    # checking for targets
    for col in target_cols:
        if col not in df.columns:
            raise ValueError(f"Missing target column: {col}")


    df = df.copy()
    df[id_col] = df[id_col].astype(str)
    df = df.sort_values([id_col, time_col], kind="mergesort").reset_index(drop=True)

    # splitting patients
    all_ids = df[id_col].unique().tolist()
    train_ids, test_ids = split_patients(
        ids=all_ids,
        method=split_method,
        test_fraction=test_fraction,
        seed=split_seed,
        hash_salt=hash_salt,
    )
    test_set = set(test_ids)

    # chronos forecaster configuration and initialization
    mparams = dict((model_cfg.get("params", {}) or {}))
    model_path = str(mparams.get("model_path", "amazon/chronos-t5-small"))
    use_source_package = bool(mparams.get("use_source_package", False))
    ignore_deps = bool(mparams.get("ignore_deps", False))

    chronos_config = dict(mparams)
    for k in ["model_path", "use_source_package", "ignore_deps"]:
        chronos_config.pop(k, None)
    if len(chronos_config) == 0:
        chronos_config = None

    fh = ForecastingHorizon(list(range(1, horizon + 1)), is_relative=True)

    forecaster = ChronosForecaster(
        model_path=model_path,
        config=chronos_config,
        seed=seed,
        use_source_package=use_source_package,
        ignore_deps=ignore_deps,
    )

    real_rows: List[pd.DataFrame] = []
    synth_rows: List[pd.DataFrame] = []
    preds_long_rows: List[Dict[str, Any]] = []

    used_test_ids: List[str] = []
    skipped_test_ids: List[str] = []

    # last-window-per-patient eval
    # doing this to avoid leakage from future data
    for pid, pdf in df.groupby(id_col, sort=False):
        if pid not in test_set:
            continue

        pdf = pdf.sort_values(time_col, kind="mergesort")
        if len(pdf) < max(min_history, context_len + horizon):
            skipped_test_ids.append(pid)
            continue

        # evaluation window = last (context + horizon)
        window = pdf.iloc[-(context_len + horizon) :]
        hist = window.iloc[:context_len]
        fut = window.iloc[context_len:] # real future horizon (H rows)

        used_test_ids.append(pid)

        # needed artifact: real_future_long.parquet
        real_part = fut[[id_col, time_col] + target_cols].copy()
        real_rows.append(real_part)

        # forecasts each target independently since Chronos is univariate forecaster
        y_index = pd.RangeIndex(start=0, stop=len(hist))
        pred_mat = np.zeros((horizon, len(target_cols)), dtype=float)

        for j, col in enumerate(target_cols):
            y_hist = pd.Series(hist[col].to_numpy(dtype=float), index=y_index)

            # missing values = bad for Chronos
            y_hist = fill_missing_1d(y_hist)

            forecaster.fit(y_hist, fh=fh)
            y_hat = forecaster.predict(fh=fh)

            pred = as_float_1d(y_hat)
            if pred.shape[0] != horizon:
                raise RuntimeError(
                    f"Chronos returned {pred.shape[0]} steps, expected horizon={horizon}."
                )
            pred_mat[:, j] = pred

            if save_preds_long:
                for h in range(horizon):
                    preds_long_rows.append(
                        {
                            id_col: pid,
                            "target": col,
                            "horizon_step": int(h + 1),
                            "pred": float(pred[h]),
                        }
                    )

        # required synth_future_long.parquet
        # also timestamps have to be alligned
        synth_part = fut[[id_col, time_col]].copy()
        for j, col in enumerate(target_cols):
            synth_part[col] = pred_mat[:, j]
        synth_rows.append(synth_part)

    if len(real_rows) == 0:
        raise RuntimeError(
            "No valid test patients produced outputs. Check task.context_len/task.horizon/task.min_history vs the per-patient lengths.")

    real_future_long = pd.concat(real_rows, axis=0, ignore_index=True)
    synth_future_long = pd.concat(synth_rows, axis=0, ignore_index=True)

    # final sorting for deterministic outputs
    real_future_long = real_future_long.sort_values([id_col, time_col], kind="mergesort")
    synth_future_long = synth_future_long.sort_values([id_col, time_col], kind="mergesort")

    # writing artifacts
    real_future_long.to_parquet(run_dir / "real_future_long.parquet", index=False)
    synth_future_long.to_parquet(run_dir / "synth_future_long.parquet", index=False)

    # required pipeline artifacts
    split_ids = {
        "train_ids": train_ids,
        "test_ids": test_ids,
        "used_test_ids": used_test_ids,
        "skipped_test_ids": skipped_test_ids,
    }
    save_json(split_ids, str(run_dir / "split_ids.json"))

    with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    run_meta = {
        "seed": seed,
        "split_seed": split_seed,
        "split_method": split_method,
        "test_fraction": test_fraction,
        "hash_salt": hash_salt,
        "context_len": context_len,
        "horizon": horizon,
        "min_history": min_history,
        "framework": "sktime",
        "model": "ChronosForecaster",
        "model_path": model_path,
        "chronos_config": chronos_config or {},
        "use_source_package": use_source_package,
        "ignore_deps": ignore_deps,
        "n_patients_total": int(len(all_ids)),
        "n_train_patients": int(len(train_ids)),
        "n_test_patients": int(len(test_ids)),
        "n_used_test_patients": int(len(used_test_ids)),
        "n_skipped_test_patients": int(len(skipped_test_ids)),
        "rows_real_future": int(len(real_future_long)),
        "rows_synth_future": int(len(synth_future_long)),
        "elapsed_sec": float(time.time() - t0),
    }
    save_json(run_meta, str(run_dir / "run_meta.json"))

    if save_preds_long:
        preds_long_df = pd.DataFrame(preds_long_rows)
        preds_long_df.to_parquet(run_dir / "preds_long.parquet", index=False)


if __name__ == "__main__":
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg_arg)
