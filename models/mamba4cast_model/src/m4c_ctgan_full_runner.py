import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import yaml

THIS_FILE = Path(__file__).resolve()
# parents[0]=src, [1]=mamba4cast_model, [2]=models, [3]=leukemia_project
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from models.s4m_model.src.utils import (  # type: ignore
    set_seed,
    ensure_dir,
    load_parquet,
    save_json,
    df_checks,
)

# Helper functions
def _resolve_path(repo_root: Path, p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (repo_root / pp).resolve()


def _load_yaml(cfg_path: Path) -> Dict[str, Any]:
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _write_resolved_cfg(cfg: Dict[str, Any], out_path: Path) -> None:
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _patient_split(ids: List[str], seed: int, test_frac: float) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(seed)
    ids = list(map(str, ids))
    rng.shuffle(ids)
    n_test = int(round(len(ids) * test_frac))
    n_test = max(1, n_test)
    test_ids = ids[:n_test]
    train_ids = ids[n_test:]
    return train_ids, test_ids


def _build_real_future_long(
    df_long: pd.DataFrame,
    id_col: str,
    time_col: str,
    target_cols: List[str],
    test_ids: List[str],
    context_len: int,
    pred_len: int,
) -> pd.DataFrame:
    df = df_long[df_long[id_col].astype(str).isin(set(map(str, test_ids)))].copy()
    df[id_col] = df[id_col].astype(str)
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    out_rows = []
    for pid, g in df.groupby(id_col, sort=False):
        g = g.sort_values(time_col)
        if len(g) < context_len + pred_len:
            continue
        fut = g.iloc[context_len : context_len + pred_len]
        out_rows.append(fut[[id_col, time_col] + target_cols])

    if not out_rows:
        return pd.DataFrame(columns=[id_col, time_col] + target_cols)
    out = pd.concat(out_rows, axis=0, ignore_index=True)
    return out


def _make_time_feats_from_t(t_arr: np.ndarray) -> np.ndarray:
    """
    M4C expects time features with indices:
    YEAR, MONTH, DAY, DOW, DOY, HOUR, MINUTE  (len=7)
    For Physionet (t=hours) I'm creating the encoding:
      day = t//24, hour = t%24, dow = day%7, doy = day
      year/month/minute = 0
    """
    t_arr = t_arr.astype(int)
    day = (t_arr // 24).astype(int)
    hour = (t_arr % 24).astype(int)
    dow = (day % 7).astype(int)
    doy = day.astype(int)

    year = np.zeros_like(day)
    month = np.zeros_like(day)
    minute = np.zeros_like(day)

    feats = np.stack([year, month, day, dow, doy, hour, minute], axis=-1).astype(np.float32)
    return feats


def _load_official_m4c(repo_dir: Path, ckpt_path: Path, device: str):
    # Making repo importable
    sys.path.insert(0, str(repo_dir))
    sys.path.insert(0, str(repo_dir / "src_torch"))

    try:
        from src_torch.training import models as m4c_models  # type: ignore
    except Exception as e:
        raise ImportError("Could not import src_torch.training.models from the official repo") from e

    import torch  # local import to avoid torch requirement in dry-run paths

    # checkpoint will contain: {"model_state_dict":..., "config":...}
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # Matching what the repo uses since it has multiple model classes
    # Most usage instantiates via config 
    # Handling both patterns
    if isinstance(ckpt, dict) and "config" in ckpt:
        cfg = ckpt["config"]
        model_cfg = cfg.get("model", cfg)
        model_name = model_cfg.get("name", None) or model_cfg.get("model_name", None)

        # Trying known constructors from actual models.py
        model = None
        candidates = []
        for attr in dir(m4c_models):
            if attr.lower().startswith("mamba"):
                candidates.append(attr)

        # Wrote so that it prefers explicit model_name, else falls back to SSMModelMulti if present
        if model_name and hasattr(m4c_models, model_name):
            cls = getattr(m4c_models, model_name)
            model = cls(**{k: v for k, v in model_cfg.items() if k not in {"name", "model_name"}})
        else:
            # fall back to SSMModelMulti if present else first mamba-like class
            if hasattr(m4c_models, "SSMModelMulti"):
                model = m4c_models.SSMModelMulti(**{k: v for k, v in model_cfg.items() if k not in {"name", "model_name"}})
            elif candidates:
                cls = getattr(m4c_models, candidates[0])
                model = cls(**{k: v for k, v in model_cfg.items() if k not in {"name", "model_name"}})
            else:
                raise RuntimeError("Could not find a suitable model class in official models.py")

        state = ckpt.get("model_state_dict", ckpt.get("state_dict", None))
        if state is None:
            raise RuntimeError("Checkpoint missing model_state_dict/state_dict")
        model.load_state_dict(state, strict=True)
    else:
        raise RuntimeError("Unsupported checkpoint format (expected dict with 'config' and 'model_state_dict').")

    model.to(device)
    model.eval()
    return model


def _predict_univariate_m4c(
    model,
    history_vals: np.ndarray, # [N, L]
    context_t: np.ndarray, # [N, L]
    target_t: np.ndarray, # [N, pred_len]
    pred_len: int,
    device: str,
) -> np.ndarray:
    import torch

    x: Dict[str, Any] = {}
    x["history"] = torch.tensor(history_vals, dtype=torch.float32, device=device)
    x["ts"] = torch.tensor(_make_time_feats_from_t(context_t), dtype=torch.float32, device=device)
    x["target_dates"] = torch.tensor(_make_time_feats_from_t(target_t), dtype=torch.float32, device=device)
    # task: SINGLE_POINT (0)
    x["task"] = torch.zeros((history_vals.shape[0], pred_len), dtype=torch.int64, device=device)

    with torch.no_grad():
        out = model(x, prediction_length=pred_len) if "prediction_length" in model.forward.__code__.co_varnames else model(x)
        # M4Cs forward returns {'result': ..., 'scale': ...}
        y = out["result"]
        scale = out.get("scale", None)

    # y expected shape = [N, pred_len] or [N, pred_len, 1]
    if y.ndim == 3:
        y = y[..., 0]
    y = y.detach().cpu().numpy()

    if scale is not None:
        # scale usually shape [N,1,1] or [N,1] into broadcast
        sc = scale.detach().cpu().numpy()
        y = y * sc.reshape(sc.shape[0], -1)[:, :1]

    return y.astype(np.float32)


# main function
def main(cfg_path_arg: str, ctgan_static_parquet_arg: str) -> None:
    cfg_path = Path(cfg_path_arg)
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")

    cfg = _load_yaml(cfg_path)

    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    data_cfg = cfg["data"]
    id_col = str(data_cfg["id_col"])
    time_col = str(data_cfg["time_col"])
    target_cols = list(data_cfg["target_cols"])
    context_len = int(data_cfg.get("context_len", data_cfg.get("context_length", 24)))
    pred_len = int(data_cfg.get("pred_len", data_cfg.get("pred_length", 24)))
    test_frac = float(data_cfg.get("test_frac", 0.2))

    debug_cfg = cfg.get("debug", {}) or {}
    skip_model = bool(debug_cfg.get("skip_model", False))
    max_ctgan = debug_cfg.get("max_ctgan_patients", None)
    max_ctgan = int(max_ctgan) if max_ctgan is not None else None
    max_test = debug_cfg.get("max_test_patients", None)
    max_test = int(max_test) if max_test is not None else None

    real_long_parquet = _resolve_path(REPO_ROOT, str(data_cfg["parquet_path"]))
    ctgan_static_parquet = _resolve_path(REPO_ROOT, str(ctgan_static_parquet_arg))

    if not real_long_parquet.exists():
        raise FileNotFoundError(f"real parquet not found: {real_long_parquet}")
    if not ctgan_static_parquet.exists():
        raise FileNotFoundError(f"ctgan static parquet not found: {ctgan_static_parquet}")

    out_cfg = cfg["output"]
    out_dir = _resolve_path(REPO_ROOT, str(out_cfg["dir"]))
    base_run_name = str(out_cfg.get("run_name", "m4c_zeroshot"))
    run_name = base_run_name + "_ctgan_full"
    run_dir = out_dir / run_name
    ensure_dir(str(run_dir))

    # loading real df + checks
    real_df = load_parquet(str(real_long_parquet))
    df_checks(real_df, id_col, time_col)
    missing_targets = [c for c in target_cols if c not in real_df.columns]
    if missing_targets:
        raise ValueError(f"Missing target cols in real parquet: {missing_targets}")

    # splitting train/test ids
    all_ids = sorted(real_df[id_col].astype(str).unique().tolist())
    train_ids, test_ids = _patient_split(all_ids, seed=seed, test_frac=test_frac)
    if max_test is not None:
        test_ids = test_ids[:max_test]

    split_ids = {"train_ids": train_ids, "test_ids": test_ids}
    save_json(split_ids, str(run_dir / "split_ids.json"))

    # real future from test patients
    real_future = _build_real_future_long(
        df_long=real_df,
        id_col=id_col,
        time_col=time_col,
        target_cols=target_cols,
        test_ids=test_ids,
        context_len=context_len,
        pred_len=pred_len,
    )

    # bootstrapping CTGAN to history windows 
    # using TRAIN ids only
    bootstrap_dir = run_dir / "bootstrap"
    ensure_dir(str(bootstrap_dir))

    # writing a temp train_split json for bthe ootstrap script
    train_split_path = bootstrap_dir / "train_split_ids.json"
    save_json({"train_ids": train_ids}, str(train_split_path))

    bootstrap_script = (REPO_ROOT / "scripts" / "bootstrap_ctgan_to_history.py").resolve()
    if not bootstrap_script.exists():
        raise FileNotFoundError(f"bootstrap script not found: {bootstrap_script}")

    cmd = [
        sys.executable,
        str(bootstrap_script),
        "--ctgan_static_parquet", str(ctgan_static_parquet),
        "--real_long_parquet", str(real_long_parquet),
        "--out_dir", str(bootstrap_dir),
        "--id_col", id_col,
        "--time_col", time_col,
        "--target_cols", ",".join(target_cols),
        "--non_strict_baseline",
        "--history_len", str(context_len),
        "--pred_len", str(pred_len),
        "--seed", str(seed),
        "--train_split_ids_json", str(train_split_path),
    ]
    subprocess.check_call(cmd)

    hist_path = bootstrap_dir / "bootstrap_history.npy"
    meta_path = bootstrap_dir / "bootstrap_meta.json"
    if not hist_path.exists():
        raise FileNotFoundError(f"bootstrap_history.npy not found at: {hist_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"bootstrap_meta.json not found at: {meta_path}")

    hist = np.load(str(hist_path)) # expected: [N, context_len, C_targets]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    ctgan_df = pd.read_parquet(str(ctgan_static_parquet))
    if id_col not in ctgan_df.columns:
        raise ValueError(f"CTGAN parquet missing id_col '{id_col}' (expected same id col).")
    ctgan_ids = ctgan_df[id_col].astype(str).tolist()

    n = min(len(ctgan_ids), hist.shape[0])
    if max_ctgan is not None:
        n = min(n, max_ctgan)
    ctgan_ids = ctgan_ids[:n]
    hist = hist[:n]

    # Building context + target t arrays for M4C
    # Context is t = 0..context_len-1 for all CTGAN patients 
    # bootstrap templates are aligned
    # Target is next pred_len steps: t= context_len..context_len+pred_len-1
    context_t = np.tile(np.arange(context_len, dtype=int)[None, :], (n, 1))
    target_t = np.tile((np.arange(pred_len, dtype=int) + context_len)[None, :], (n, 1))

    # Forecasting CTGAN futures
    inference_failed = False
    if skip_model:
        print("[m4c][dry-run] debug.skip_model=true -> skipping official model import/inference.")
        # write NaNs but with correct shape/columns
        preds_all = np.full((n, pred_len, len(target_cols)), np.nan, dtype=np.float32)
    else:
        try:
            model_cfg = cfg["model"]
            repo_dir = _resolve_path(REPO_ROOT, str(model_cfg["repo_dir"]))
            ckpt_cfg = str(model_cfg["checkpoint_path"]).strip()

            # Resolving checkpoint path:
            # if absolute: use as-is
            # if relative: resolve relative to repo_dir 
            # if fallback: repo_dir/models/filename
            ckpt_path = Path(ckpt_cfg)
            if not ckpt_path.is_absolute():
                ckpt_path = (repo_dir / ckpt_cfg).resolve()

            if not ckpt_path.exists():
                ckpt_path2 = (repo_dir / "models" / Path(ckpt_cfg).name).resolve()
                if ckpt_path2.exists():
                    ckpt_path = ckpt_path2

            device = str(cfg.get("device", "cpu"))

            if not repo_dir.exists():
                raise FileNotFoundError(f"repo_dir not found: {repo_dir}")
            if not ckpt_path.exists():
                raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

            model = _load_official_m4c(repo_dir=repo_dir, ckpt_path=ckpt_path, device=device)

            # M4C is univariate hence run per target column
            preds_list = []
            for j, col in enumerate(target_cols):
                h_j = hist[:, :, j]  # [N, L]
                yhat = _predict_univariate_m4c(
                    model=model,
                    history_vals=h_j,
                    context_t=context_t,
                    target_t=target_t,
                    pred_len=pred_len,
                    device=device,
                )  # [N, pred_len]
                preds_list.append(yhat[:, :, None])

            preds_all = np.concatenate(preds_list, axis=2)  # [N, pred_len, C]
        except Exception as e:
            inference_failed = True
            print(f"[m4c][dry-run] Official model import/inference failed: {e} -> skipping official model inference.")
            preds_all = np.full((n, pred_len, len(target_cols)), np.nan, dtype=np.float32)

    # building synth_future_long.parquet with CTGAN ids
    rows = []
    for i, pid in enumerate(ctgan_ids):
        for k in range(pred_len):
            row = {id_col: pid, time_col: int(context_len + k)}
            for j, col in enumerate(target_cols):
                row[col] = float(preds_all[i, k, j]) if not np.isnan(preds_all[i, k, j]) else np.nan
            rows.append(row)
    synth_future = pd.DataFrame(rows, columns=[id_col, time_col] + target_cols)
    synth_future.to_parquet(run_dir / "synth_future_long.parquet", index=False)

    # writing run_meta + resolved cfg
    # Save real_future_long.parquet with NaNs if model skipped/failed
    if skip_model or inference_failed:
        for col in target_cols:
            real_future[col] = np.nan
    real_future.to_parquet(run_dir / "real_future_long.parquet", index=False)
    run_meta = {
        "model": "mamba4cast",
        "mode": "ctgan_full",
        "seed": seed,
        "context_len": context_len,
        "pred_len": pred_len,
        "n_train_ids": int(len(train_ids)),
        "n_test_ids": int(len(test_ids)),
        "n_ctgan": int(n),
        "paths": {
            "real_long_parquet": str(real_long_parquet),
            "ctgan_static_parquet": str(ctgan_static_parquet),
            "run_dir": str(run_dir),
            "bootstrap_dir": str(bootstrap_dir),
        },
        "bootstrap_meta": meta,
    }
    save_json(run_meta, str(run_dir / "run_meta.json"))

    # resolved cfg snapshot with final run name
    cfg_res = dict(cfg)
    cfg_res.setdefault("output", {})
    cfg_res["output"] = dict(cfg_res["output"])
    cfg_res["output"]["run_name"] = run_name
    _write_resolved_cfg(cfg_res, run_dir / "config_resolved.yaml")

    print(f"M4C CTGAN run_dir: {run_dir}")
    print("M4C wrote: config_resolved.yaml, split_ids.json, run_meta.json")
    print(f"M4C wrote: real_future_long.parquet ({len(real_future)} rows)")
    print(f"M4C wrote: synth_future_long.parquet ({len(synth_future)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str)
    ap.add_argument("ctgan_static_parquet", type=str)
    args = ap.parse_args()
    main(args.config, args.ctgan_static_parquet)
