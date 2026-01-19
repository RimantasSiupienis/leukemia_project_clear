"""
Runner for the Mamba4Cast repo 
zero-shot, one-pass horizon prediction

- data loading
- patient split (no leakage)
- standardised outputs + artifacts
- logging + reproducibility
- minimal device/paths fixes

Outputs under: output.dir/output.run_name/
  config_resolved.yaml
  split_ids.json
  run_meta.json
  real_future_long.parquet
  synth_future_long.parquet

Parquet schema:
  [id_col, time_col] + target_cols
"""

from __future__ import annotations

import sys
from pathlib import Path
import os


# Ensure project root is on PYTHONPATH so imports work
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from models.s4m_model.src.utils import (
    set_seed,
    ensure_dir,
    load_parquet,
    save_json,
    df_checks,
)


# Path helper functions
def _repo_root() -> Path:
    # models/mamba4cast_model/src/m4c_runner.py -> repo root
    return Path(__file__).resolve().parents[3]


def resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (_repo_root() / pp)


# Debug helpers, limit lists, will be useless later on
def _maybe_limit_list(xs: List[Any], n: Optional[int]) -> List[Any]:
    if n is None:
        return xs
    n = int(n)
    if n <= 0:
        return xs
    return xs[: min(len(xs), n)]


# Official repo import + model loading
def _import_official(repo_dir: str) -> None:
    repo_path = resolve_path(repo_dir).expanduser().resolve()
    src_path = repo_path / "src_torch"
    if not src_path.exists():
        raise FileNotFoundError(f"Expected '{src_path}', check model.repo_dir in YAML")

    # allow both repo root and src_torch to be on sys.path
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _instantiate_model_from_checkpoint(ModelCls: Any, state: Any) -> Any:
    """
    - if checkpoint includes hyperparams dict -> try ModelCls(**hyperparams)
    - else try default constructor
    """
    if isinstance(state, dict):
        for k in ("hyper_parameters", "hparams", "hyperparams"):
            hp = state.get(k, None)
            if isinstance(hp, dict):
                try:
                    return ModelCls(**hp)
                except Exception:
                    break
    return ModelCls()


def _env_flag(name: str, default: str = "0") -> bool:
    v = str(os.environ.get(name, default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def load_m4c_model(repo_dir: str, checkpoint_path: str, device: torch.device):
    """
    Loads model class from official repo and weights.
    Only glue: import paths and checkpoint path resolution.
    """
    _import_official(repo_dir)

    try:
        from src_torch.training import models as m4c_models  # type: ignore
    except Exception as e:
        raise ImportError("Could not import src_torch.training.models from the official repo") from e

    ModelCls = getattr(m4c_models, "SSMModelMulti", None) or getattr(m4c_models, "SSMModel", None)
    if ModelCls is None:
        raise ImportError("Neither SSMModelMulti nor SSMModel found in src_torch.training.models")

    repo_path = resolve_path(repo_dir).expanduser().resolve()
    ckpt = Path(checkpoint_path)
    if not ckpt.is_absolute():
        ckpt = repo_path / ckpt
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model = None
    errs: List[str] = []

    # Trying lightning style load if available
    try:
        load_fn = getattr(ModelCls, "load_from_checkpoint", None)
        if callable(load_fn):
            model = load_fn(str(ckpt), map_location=device)
    except Exception as e:
        errs.append(f"load_from_checkpoint failed: {e!r}")

    # try torch.load + load_state_dict if lightning failed
    if model is None:
        try:
            state = torch.load(str(ckpt), map_location=device)
            model = _instantiate_model_from_checkpoint(ModelCls, state)

            if isinstance(state, dict) and "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
                model.load_state_dict(state["model_state_dict"], strict=False)
            elif isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                model.load_state_dict(state["state_dict"], strict=False)
            elif isinstance(state, dict):
                model.load_state_dict(state, strict=False)
            else:
                raise RuntimeError("Checkpoint is not a dict/state_dict.")
        except Exception as e:
            errs.append(f"torch.load + load_state_dict failed: {e!r}")
            model = None

    if model is None:
        raise RuntimeError("Failed to load Mamba4Cast model:\n" + "\n".join(errs))

    model.to(device)
    model.eval()
    return model


def _m4c_forward(model: Any, x: Dict[str, torch.Tensor], pred_len: int) -> Dict[str, Any]:
    """
    Official forward = forward(self, x, prediction_length=None, ...)
    Two call styles are used in the repo, we try both:
    """
    tried: List[str] = []
    try:
        return model(x, prediction_length=pred_len)
    except Exception as e:
        tried.append(f"model(x, prediction_length=pred_len) -> {type(e).__name__}: {e}")

    try:
        return model(x, pred_len)
    except Exception as e:
        tried.append(f"model(x, pred_len) -> {type(e).__name__}: {e}")

    raise RuntimeError("Could not call official Mamba4Cast forward.\nTried:\n- " + "\n- ".join(tried))


def _m4c_unscale(output: Dict[str, Any], scaler: str) -> torch.Tensor:
    """
    Model returns {result: <scaled preds>, scale: <scale params>}
    Applies the same style of descaling as helper code.
    - custom_robust: y = result * scale[1] + scale[0]
    - min_max: y = result * (scale[0] - scale[1]) + scale[1]
    - identity: y = result
    """
    scaler = (scaler or "min_max").lower().strip()

    result = output["result"]
    scale = output.get("scale", None)

    if scaler == "identity" or scale is None:
        return result

    if scaler == "custom_robust":
        return (result * scale[1].squeeze(-1)) + scale[0].squeeze(-1)

    if scaler == "min_max":
        return (result * (scale[0].squeeze(-1) - scale[1].squeeze(-1))) + scale[1].squeeze(-1)

    return result



# Splitting helpers, no leakage
def _hash_score(pid: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}|{pid}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / float(16**8)


def split_patients(
    ids: List[Any],
    *,
    method: str,
    test_fraction: float,
    seed: int,
    hash_salt: str,
) -> Tuple[List[str], List[str]]:
    ids = [str(x) for x in ids]
    method = str(method).strip().lower()

    if not (0.0 < float(test_fraction) < 1.0):
        raise ValueError("split.test_fraction must be in (0, 1)")

    if method == "patient_random":
        rng = np.random.default_rng(int(seed))
        ids_shuf = ids.copy()
        rng.shuffle(ids_shuf)
        n_test = int(round(len(ids_shuf) * float(test_fraction)))
        n_test = max(1, min(len(ids_shuf) - 1, n_test)) if len(ids_shuf) >= 2 else len(ids_shuf)
        test_ids = ids_shuf[:n_test]
        train_ids = ids_shuf[n_test:]
        return train_ids, test_ids

    if method == "patient_hash":
        salt = str(hash_salt) if hash_salt is not None else ""
        scored = [(_hash_score(pid, salt), pid) for pid in ids]
        scored.sort(key=lambda t: t[0])
        n_test = int(round(len(scored) * float(test_fraction)))
        n_test = max(1, min(len(scored) - 1, n_test)) if len(scored) >= 2 else len(scored)
        test_ids = [pid for _, pid in scored[:n_test]]
        train_ids = [pid for _, pid in scored[n_test:]]
        return train_ids, test_ids

    raise ValueError("split.method must be patient_hash or patient_random")


# Time feature builder
#
def build_time_features_from_t(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t).astype(np.int64)
    day = (t // 24).astype(np.int64)
    hour = (t % 24).astype(np.int64)

    year = np.zeros_like(t, dtype=np.int64)
    month = np.zeros_like(t, dtype=np.int64)
    dow = (day % 7).astype(np.int64)
    doy = day.astype(np.int64)
    minute = np.zeros_like(t, dtype=np.int64)

    feats = np.stack([year, month, day, dow, doy, hour, minute], axis=1)
    return feats

# Builds test context + future arrays
def build_test_context_and_future(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    target_col: str,
    test_ids: List[str],
    context_length: int,
    prediction_length: int,
) -> Tuple[np.ndarray, List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
    min_len = int(context_length + prediction_length)

    history_vals: List[np.ndarray] = []
    kept_ids: List[str] = []
    ctx_times: List[np.ndarray] = []
    fut_times: List[np.ndarray] = []
    real_future_vals: List[np.ndarray] = []
    skipped_short: List[str] = []

    test_set = set(map(str, test_ids))
    df_sub = df[df[id_col].astype(str).isin(test_set)].copy()

    for sid, g in df_sub.groupby(id_col, sort=False):
        sid = str(sid)
        g = g.sort_values(time_col, kind="mergesort")
        T = len(g)
        if T < min_len:
            skipped_short.append(sid)
            continue

        g_ctx = g.iloc[-(context_length + prediction_length) : -prediction_length]
        g_fut = g.iloc[-prediction_length:]

        t_ctx = g_ctx[time_col].to_numpy(np.int64)
        t_fut = g_fut[time_col].to_numpy(np.int64)

        y_ctx = g_ctx[target_col].to_numpy(np.float32)
        y_fut = g_fut[target_col].to_numpy(np.float32)

        if len(y_ctx) != context_length or len(y_fut) != prediction_length:
            skipped_short.append(sid)
            continue

        history_vals.append(y_ctx)
        kept_ids.append(sid)
        ctx_times.append(t_ctx)
        fut_times.append(t_fut)
        real_future_vals.append(y_fut)

    if not kept_ids:
        raise RuntimeError("No test patients produced outputs. Check context_length/prediction_length.")

    stats: Dict[str, Any] = {
        "n_test_requested": int(len(test_ids)),
        "n_test_used": int(len(kept_ids)),
        "n_test_skipped_short": int(len(skipped_short)),
        "skipped_short_ids": skipped_short,
        "min_required_length": int(min_len),
    }

    return (
        np.stack(history_vals, axis=0).astype(np.float32),
        kept_ids,
        ctx_times,
        fut_times,
        real_future_vals,
        stats,
    )

# Builds long DataFrame from arrays, output formatting
def make_long_df(
    *,
    ids: List[str],
    times_list: List[np.ndarray],
    values: np.ndarray,  # [B,H,C]
    id_col: str,
    time_col: str,
    target_cols: List[str],
) -> pd.DataFrame:
    B, H, C = values.shape
    if len(ids) != B or len(times_list) != B or C != len(target_cols):
        raise ValueError("Shape mismatch while building long DataFrame.")

    rows: List[Dict[str, Any]] = []
    for i in range(B):
        sid = ids[i]
        t = times_list[i]
        if len(t) != H:
            raise ValueError("Future times length does not match horizon.")
        for h in range(H):
            r: Dict[str, Any] = {id_col: sid, time_col: int(t[h])}
            for c, col in enumerate(target_cols):
                r[col] = float(values[i, h, c])
            rows.append(r)
    return pd.DataFrame(rows)


# Main runner
def main(cfg_path: Optional[str] = None) -> None:
    if cfg_path is None:
        p1 = Path(__file__).resolve().parents[1] / "configs" / "m4c_base.yaml"
        p2 = Path(__file__).resolve().parents[1] / "config" / "m4c_base.yaml"
        cfg_path = str(p1 if p1.exists() else p2)

    cfg_file = resolve_path(str(cfg_path))
    if not cfg_file.is_file():
        raise FileNotFoundError(f"config not found: {cfg_file}")

    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    t0 = time.time()

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device_str = str(cfg.get("device", "auto")).lower()
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    split_cfg = cfg.get("split", {}) or {}
    out_cfg = cfg["output"]

    parquet_path = resolve_path(str(data_cfg["parquet_path"]))
    id_col = str(data_cfg["id_col"])
    time_col = str(data_cfg["time_col"])
    target_cols_all = list(data_cfg["target_cols"])
    context_length = int(data_cfg["context_length"])
    pred_len = int(data_cfg["prediction_length"])

    m4c_scaler = str(model_cfg.get("scaler", "min_max"))

    debug_cfg = cfg.get("debug", {}) or {}
    debug_enabled = bool(debug_cfg.get("enabled", False))

    if _env_flag("M4C_DEBUG", "0"):
        debug_enabled = True
        debug_cfg = dict(debug_cfg)
        debug_cfg["enabled"] = True
        if "M4C_MAX_PATIENTS_TOTAL" in os.environ:
            debug_cfg["max_patients_total"] = int(os.environ["M4C_MAX_PATIENTS_TOTAL"])
        if "M4C_MAX_TEST_PATIENTS" in os.environ:
            debug_cfg["max_test_patients"] = int(os.environ["M4C_MAX_TEST_PATIENTS"])
        if "M4C_MAX_TARGETS" in os.environ:
            debug_cfg["max_targets"] = int(os.environ["M4C_MAX_TARGETS"])

    max_patients_total = debug_cfg.get("max_patients_total", None) if debug_enabled else None
    max_test_patients = debug_cfg.get("max_test_patients", None) if debug_enabled else None
    max_targets = debug_cfg.get("max_targets", None) if debug_enabled else None

    if max_targets is not None:
        max_targets = int(max_targets)
        if max_targets > 0:
            target_cols_all = target_cols_all[:max_targets]

#####################################################
    # Making config_resolved reflect target_cols used
    # Because target_cols_all may be truncated by debug.max_targets
    cfg = dict(cfg)
    cfg_data = dict(cfg.get("data", {}) or {})
    cfg_data["target_cols"] = list(target_cols_all)
    cfg["data"] = cfg_data
#####################################################
    split_method = str(split_cfg.get("method", "patient_hash"))
    split_seed = int(split_cfg.get("seed", seed))
    test_fraction = float(split_cfg.get("test_fraction", split_cfg.get("test_frac", 0.2)))
    hash_salt = str(split_cfg.get("hash_salt", "mamba4cast"))

    out_root = resolve_path(str(out_cfg.get("dir", out_cfg.get("out_dir", "results/mamba4cast"))))
    run_name = str(out_cfg.get("run_name", "m4c_run")).strip()
    run_dir = out_root / run_name
    ensure_dir(str(run_dir))

    df = load_parquet(str(parquet_path))
    df = df_checks(df, id_col=id_col, time_col=time_col)
    df = df.copy()
    df[id_col] = df[id_col].astype(str)
    df = df.sort_values([id_col, time_col], kind="mergesort").reset_index(drop=True)

    missing = [c for c in [id_col, time_col] + target_cols_all if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in parquet: {missing}")

    all_ids = df[id_col].dropna().astype(str).unique().tolist()
    if max_patients_total is not None:
        all_ids = _maybe_limit_list(all_ids, int(max_patients_total))

    train_ids, test_ids = split_patients(
        all_ids,
        method=split_method,
        test_fraction=test_fraction,
        seed=split_seed,
        hash_salt=hash_salt,
    )

    if max_test_patients is not None:
        test_ids = _maybe_limit_list(test_ids, int(max_test_patients))

    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    save_json({"train_ids": train_ids, "test_ids": test_ids}, str(run_dir / "split_ids.json"))

    repo_dir = str(model_cfg["repo_dir"])
    ckpt_path = str(model_cfg["checkpoint_path"])

    skip_model = bool(debug_cfg.get("skip_model", False)) if debug_enabled else False
    model = None

    if not skip_model:
        skip_model = _env_flag("M4C_SKIP_MODEL", "0") or bool(debug_cfg.get("skip_model", False))
        model = None
        if skip_model:
            print("[m4c][dry-run] skipping official model import/inference (M4C_SKIP_MODEL=1 or debug.skip_model=true).")
        else:
            model = load_m4c_model(repo_dir=repo_dir, checkpoint_path=ckpt_path, device=device)
    else:
        print("[m4c][dry-run] debug.skip_model=true -> skipping official model import/inference.")

    stats_all: Dict[str, Any] = {"per_target": {}}
    per_target_preds: List[np.ndarray] = []
    per_target_reals: List[np.ndarray] = []
    common_times: Optional[List[np.ndarray]] = None
    common_ids: Optional[List[str]] = None
    base_stats: Optional[Dict[str, Any]] = None

    for tgt in target_cols_all:
        history_vals, kept_ids, ctx_times, fut_times, real_fut_vals, stats = build_test_context_and_future(
            df=df,
            id_col=id_col,
            time_col=time_col,
            target_col=tgt,
            test_ids=test_ids,
            context_length=context_length,
            prediction_length=pred_len,
        )

        if common_times is None:
            common_times = fut_times
            common_ids = kept_ids
            base_stats = stats
        else:
            if kept_ids != common_ids:
                raise ValueError("Kept test IDs differ across targets.")
            for i in range(len(fut_times)):
                if not np.array_equal(fut_times[i], common_times[i]):
                    raise ValueError("Future time grids differ across targets; cannot standardize horizon.")

        ts_hist = np.stack([build_time_features_from_t(t) for t in ctx_times], axis=0)
        ts_fut = np.stack([build_time_features_from_t(t) for t in fut_times], axis=0)

        x: Dict[str, torch.Tensor] = {
            "ts": torch.from_numpy(ts_hist).to(device=device, dtype=torch.long),
            "history": torch.from_numpy(history_vals).to(device=device, dtype=torch.float32),
            "target_dates": torch.from_numpy(ts_fut).to(device=device, dtype=torch.long),
            "task": torch.zeros((history_vals.shape[0], pred_len), dtype=torch.int64, device=device),
        }

        if skip_model:
            last = x["history"][:, -1].detach().cpu().numpy().astype(np.float32)
            preds_np = np.repeat(last[:, None], pred_len, axis=1).astype(np.float32)
        else:
            assert model is not None
            with torch.no_grad():
                out = _m4c_forward(model, x, pred_len)
                preds = _m4c_unscale(out, m4c_scaler)
            preds = preds.detach().cpu()
            if preds.ndim == 3 and preds.shape[-1] == 1:
                preds = preds[..., 0]
            if preds.ndim != 2:
                raise RuntimeError(f"Unexpected prediction shape from M4C for target={tgt}: {tuple(preds.shape)}")
            preds_np = preds.numpy().astype(np.float32)

        real_np = np.stack(real_fut_vals, axis=0).astype(np.float32)

        per_target_preds.append(preds_np)
        per_target_reals.append(real_np)
        stats_all["per_target"][tgt] = stats

    assert common_times is not None and common_ids is not None and base_stats is not None

    synth_future_all = np.stack(per_target_preds, axis=-1).astype(np.float32)
    real_future_all = np.stack(per_target_reals, axis=-1).astype(np.float32)

    kept_ids_final = common_ids
    future_times_kept = common_times
    stats_all.update(base_stats)

    real_long = make_long_df(
        ids=kept_ids_final,
        times_list=future_times_kept,
        values=real_future_all,
        id_col=id_col,
        time_col=time_col,
        target_cols=target_cols_all,
    )
    synth_long = make_long_df(
        ids=kept_ids_final,
        times_list=future_times_kept,
        values=synth_future_all,
        id_col=id_col,
        time_col=time_col,
        target_cols=target_cols_all,
    )

    real_long = real_long.sort_values([id_col, time_col], kind="mergesort").reset_index(drop=True)
    synth_long = synth_long.sort_values([id_col, time_col], kind="mergesort").reset_index(drop=True)

    real_path = run_dir / "real_future_long.parquet"
    synth_path = run_dir / "synth_future_long.parquet"
    real_long.to_parquet(real_path, index=False)
    synth_long.to_parquet(synth_path, index=False)

    run_meta = {
        "seed": seed,
        "device": str(device),
        "context_length": int(context_length),
        "prediction_length": int(pred_len),
        "target_cols": list(target_cols_all),
        "parquet_path": str(parquet_path.as_posix()),
        "repo_dir": str(resolve_path(repo_dir).as_posix()),
        "checkpoint_path": str(ckpt_path),
        "m4c_scaler": m4c_scaler,
        "split_method": split_method,
        "split_seed": int(split_seed),
        "test_fraction": float(test_fraction),
        "hash_salt": hash_salt,
        "n_patients_total": int(len(all_ids)),
        "n_patients_train": int(len(train_ids)),
        "n_patients_test": int(len(test_ids)),
        "n_patients_test_used": int(len(kept_ids_final)),
        "rows_real_future": int(len(real_long)),
        "rows_synth_future": int(len(synth_long)),
        "stats": stats_all,
        "elapsed_sec": float(time.time() - t0),
    }
    save_json(run_meta, str(run_dir / "run_meta.json"))

    print(f"M4C run_dir: {run_dir}")
    print("M4C wrote: config_resolved.yaml, split_ids.json, run_meta.json")
    print(f"M4C wrote: real_future_long.parquet ({len(real_long)} rows)")
    print(f"M4C wrote: synth_future_long.parquet ({len(synth_long)} rows)")


if __name__ == "__main__":
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg_arg)
