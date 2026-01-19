#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

THIS_FILE = Path(__file__).resolve()
# repo root is parents[3]
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(REPO_ROOT))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_yaml(p: Path) -> Dict[str, Any]:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _save_yaml(obj: Dict[str, Any], p: Path) -> None:
    p.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def _save_json(obj: Dict[str, Any], p: Path) -> None:
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _parse_target_cols(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, str):
        return [c.strip() for c in x.split(",") if c.strip()]
    raise ValueError("data.target_cols must be list[str] or comma-separated str")


def _infer_ctgan_id_col(cols: List[str]) -> str:
    for c in ["patient_id", "ctgan_id", "id", "ID"]:
        if c in cols:
            return c
    # first column as fallback
    return cols[0]


def _impute_per_patient(df: pd.DataFrame, id_col: str, time_col: str, target_cols: List[str]) -> pd.DataFrame:
    """
    NaN imputation for template bank creation
    Per patient ffill/bfill, then global median, then 0.0.
    """
    d = df[[id_col, time_col] + target_cols].copy()
    d[id_col] = d[id_col].astype(str)
    d[time_col] = pd.to_numeric(d[time_col], errors="raise").astype(int)
    d.sort_values([id_col, time_col], inplace=True)

    med: Dict[str, float] = {}
    for c in target_cols:
        s = pd.to_numeric(d[c], errors="coerce")
        m = float(s.median()) if not s.empty else float("nan")
        if not np.isfinite(m):
            m = 0.0
        med[c] = m

    out_parts = []
    for _, g in d.groupby(id_col, sort=False):
        gg = g.copy()
        for c in target_cols:
            gg[c] = pd.to_numeric(gg[c], errors="coerce")
            gg[c] = gg[c].ffill().bfill().fillna(med[c])
        out_parts.append(gg)

    out = pd.concat(out_parts, axis=0, ignore_index=True)
    out.sort_values([id_col, time_col], inplace=True)

    nan_counts = out[target_cols].isna().sum()
    if int(nan_counts.sum()) > 0:
        out[target_cols] = out[target_cols].fillna(0.0)

    return out


def _run_bootstrap_script(
    *,
    ctgan_static_parquet: Path,
    real_long_parquet: Path,
    out_dir: Path,
    train_ids_json: Path,
    id_col: str,
    time_col: str,
    target_cols: List[str],
    history_len: int,
    pred_len: int,
    seed: int,
    baseline_map_json: Optional[Path],
    non_strict_baseline: bool,
) -> None:
    script = REPO_ROOT / "scripts" / "bootstrap_ctgan_to_history.py"
    if not script.exists():
        raise FileNotFoundError(f"bootstrap script not found: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--ctgan_static_parquet",
        str(ctgan_static_parquet),
        "--real_long_parquet",
        str(real_long_parquet),
        "--out_dir",
        str(out_dir),
        "--id_col",
        id_col,
        "--time_col",
        time_col,
        "--target_cols",
        ",".join(target_cols),
        "--history_len",
        str(history_len),
        "--pred_len",
        str(pred_len),
        "--seed",
        str(seed),
        "--train_split_ids_json",
        str(train_ids_json),
    ]
    if baseline_map_json is not None:
        cmd += ["--baseline_map_json", str(baseline_map_json)]
    if non_strict_baseline:
        cmd += ["--non_strict_baseline"]

    # running bootstrap script
    import subprocess

    p = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if p.returncode != 0:
        raise RuntimeError(f"Bootstrap failed (exit={p.returncode}): {' '.join(cmd)}")


def _history_npy_to_panel_long(
    history: np.ndarray, synth_ids: List[str], history_len: int, target_cols: List[str], id_col: str, time_col: str
) -> pd.DataFrame:
    """
    history shape: [N, history_len, C]
    returns long df with t=0...history_len-1
    """
    n, h, c = history.shape
    if h != history_len:
        raise ValueError(f"history_len mismatch: npy has {h}, expected {history_len}")
    if c != len(target_cols):
        raise ValueError(f"channel mismatch: npy has {c}, expected {len(target_cols)}")

    if len(synth_ids) != n:
        m = min(len(synth_ids), n)
        synth_ids = synth_ids[:m]
        history = history[:m]
        n = m

    rows = []
    for i in range(n):
        pid = str(synth_ids[i])
        block = {
            id_col: [pid] * history_len,
            time_col: list(range(history_len)),
        }
        for j, col in enumerate(target_cols):
            x = history[i, :, j].astype(float)
            # safety
            if np.isnan(x).any():
                s = pd.Series(x).ffill().bfill().fillna(0.0).to_numpy(dtype=float)
                x = s
            block[col] = x.tolist()
        rows.append(pd.DataFrame(block))

    out = pd.concat(rows, ignore_index=True)
    out[id_col] = out[id_col].astype(str)
    out[time_col] = out[time_col].astype(int)
    out = out.sort_values([id_col, time_col]).reset_index(drop=True)
    return out


def _to_panel_multiindex(df: pd.DataFrame, id_col: str, time_col: str, target_cols: List[str]) -> pd.DataFrame:
    return df.set_index([id_col, time_col])[target_cols].sort_index()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CTGAN static to bootstrap history to TTM forecast (reusinng pipeline)"
    )
    ap.add_argument("ttm_config", type=str, help="models/ttm_model/configs/ttm_base.yaml")
    ap.add_argument("ttm_run_dir", type=str, help="Existing trained TTM run dir (for split_ids.json)")
    ap.add_argument("ctgan_static_parquet", type=str, help="results/ctgan/.../ctgan_static.parquet")

    ap.add_argument("--out_dir", type=str, default="", help="Override output directory")
    ap.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--baseline_map_json", type=str, default="", help="Optional mapping JSON for CTGAN baseline cols")
    ap.add_argument("--non_strict_baseline", action="store_true", help="Allow missing baselines (fallback to template mean)")
    ap.add_argument("--seed", type=int, default=-1, help="Override seed (default from config)")

    args = ap.parse_args()

    cfg_path = Path(args.ttm_config).expanduser().resolve()
    cfg = _load_yaml(cfg_path)

    ttm_run_dir = Path(args.ttm_run_dir).expanduser().resolve()
    if not ttm_run_dir.exists():
        raise FileNotFoundError(f"TTM run_dir not found: {ttm_run_dir}")

    split_path = ttm_run_dir / "split_ids.json"
    if not split_path.exists():
        raise FileNotFoundError(f"TTM split_ids.json not found: {split_path}")

    splits = _load_json(split_path)
    if "train_ids" not in splits:
        raise KeyError(f"split_ids.json missing 'train_ids': {split_path}")

    seed = int(cfg.get("seed", 42)) if args.seed < 0 else int(args.seed)

    data_cfg = cfg.get("data", {}) or {}
    task_cfg = cfg.get("task", {}) or {}
    out_cfg = cfg.get("output", {}) or {}
    split_cfg = cfg.get("split", {}) or {}
    ttm_cfg = cfg.get("ttm", {}) or {}

    long_path = Path(str(data_cfg["long_path"])).expanduser().resolve()
    id_col = str(data_cfg["id_col"])
    time_col = str(data_cfg["time_col"])
    target_cols = _parse_target_cols(data_cfg["target_cols"])

    history_len = int(task_cfg["history_len"])
    pred_len = int(task_cfg["pred_len"])

    # Deriving dataset name from the existing ttm output dir(debugging/tracking))
    dataset_name = Path(str(out_cfg.get("dir", "unknown"))).expanduser().name or "unknown"
    ttm_run_name = ttm_run_dir.name

    if args.out_dir.strip():
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = REPO_ROOT / "results" / "ttm" / "full_ctgan_trajectories" / dataset_name / f"{ttm_run_name}_ctgan"
        out_dir = out_dir.resolve()

    _ensure_dir(out_dir)

    # Persisting resolved config/meta
    cfg_resolved = dict(cfg)
    cfg_resolved.setdefault("ctgan_reuse", {})
    cfg_resolved["ctgan_reuse"]["ttm_run_dir"] = str(ttm_run_dir)
    cfg_resolved["ctgan_reuse"]["ctgan_static_parquet"] = str(Path(args.ctgan_static_parquet).expanduser().resolve())
    cfg_resolved["ctgan_reuse"]["out_dir"] = str(out_dir)
    cfg_resolved["ctgan_reuse"]["device"] = args.device
    cfg_resolved["ctgan_reuse"]["seed"] = seed
    _save_yaml(cfg_resolved, out_dir / "ttm_config_resolved.yaml")

    # Loading CTGAN static ids
    ctgan_static_path = Path(args.ctgan_static_parquet).expanduser().resolve()
    ctgan_df = pd.read_parquet(ctgan_static_path)
    ctgan_id_col = _infer_ctgan_id_col(list(ctgan_df.columns))
    synth_ids = ctgan_df[ctgan_id_col].astype(str).tolist()

    # Preparing train only ids for templates 
    # train_ids.json format expected by bootstrap script
    train_ids_for_bootstrap = {"train_ids": [str(x) for x in splits["train_ids"]]}
    train_ids_json = out_dir / "train_ids_for_bootstrap.json"
    _save_json(train_ids_for_bootstrap, train_ids_json)

    # Imputes real_long first so template windows contain no nans
    real_long = pd.read_parquet(long_path)
    real_long = _impute_per_patient(real_long, id_col=id_col, time_col=time_col, target_cols=target_cols)
    imputed_long_path = out_dir / "real_long_imputed.parquet"
    real_long.to_parquet(imputed_long_path, index=False)

    # Bootstraps to npy windows
    bootstrap_dir = out_dir / "bootstrap"
    _ensure_dir(bootstrap_dir)

    baseline_map_json = Path(args.baseline_map_json).expanduser().resolve() if args.baseline_map_json.strip() else None

    _run_bootstrap_script(
        ctgan_static_parquet=ctgan_static_path,
        real_long_parquet=imputed_long_path,
        out_dir=bootstrap_dir,
        train_ids_json=train_ids_json,
        id_col=id_col,
        time_col=time_col,
        target_cols=target_cols,
        history_len=history_len,
        pred_len=pred_len,
        seed=seed,
        baseline_map_json=baseline_map_json,
        non_strict_baseline=bool(args.non_strict_baseline),
    )

    hist_path = bootstrap_dir / "bootstrap_history.npy"
    if not hist_path.exists():
        raise FileNotFoundError(f"bootstrap_history.npy not found: {hist_path}")

    history = np.load(hist_path) # [N, history_len, C]

    # Building panel history for sktime
    hist_long = _history_npy_to_panel_long(
        history=history,
        synth_ids=synth_ids,
        history_len=history_len,
        target_cols=target_cols,
        id_col=id_col,
        time_col=time_col,
    )
    y_hist = _to_panel_multiindex(hist_long, id_col=id_col, time_col=time_col, target_cols=target_cols)

    # TTM via sktime
    try:
        from sktime.forecasting.ttm import TinyTimeMixerForecaster
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: sktime (and its TTM extra deps).\n"
            "Try: pip install sktime[all_extras]\n"
            f"Original import error: {e}"
        )

    model_path = ttm_cfg.get("model_path", "ibm/TTM")
    revision = ttm_cfg.get("revision", "main")
    fit_strategy = ttm_cfg.get("fit_strategy", "zero-shot")
    training_args = dict(ttm_cfg.get("training_args", {}) or {})

    if fit_strategy in ("minimal", "full"):
        training_args.setdefault("output_dir", str(out_dir / "_ttm_finetune"))
        _ensure_dir(Path(training_args["output_dir"]))

    fh = list(range(1, pred_len + 1))

    forecaster = TinyTimeMixerForecaster(
        model_path=model_path,
        revision=revision,
        fit_strategy=fit_strategy,
        training_args=training_args,
        broadcasting=True,
    )

    print("[ttm-ctgan] starting fit...")
    forecaster.fit(y_hist, fh=fh)
    print("[ttm-ctgan] fit done, starting predict...")
    y_pred = forecaster.predict()
    print("[ttm-ctgan] predict done, writing outputs...")

    # Normalise prediction to long format with absolute t = history_len -- history_len+pred_len-1
    pred_panel = y_pred.reset_index()
    pred_panel.columns = [id_col, "fh"] + target_cols
    pred_panel[time_col] = pred_panel["fh"].astype(int) - 1 + history_len
    pred_panel = pred_panel.drop(columns=["fh"])
    pred_panel = pred_panel[[id_col, time_col] + target_cols].sort_values([id_col, time_col]).reset_index(drop=True)

    # Writes outputs
    synth_out = out_dir / "ctgan_synth_future_long.parquet"
    pred_panel.to_parquet(synth_out, index=False)

    meta = {
        "mode": "ctgan_static_ttm_reuse",
        "seed": seed,
        "device": args.device,
        "inputs": {
            "ttm_config": str(cfg_path),
            "ttm_run_dir": str(ttm_run_dir),
            "ctgan_static_parquet": str(ctgan_static_path),
            "real_long_parquet": str(long_path),
            "real_long_imputed_parquet": str(imputed_long_path),
            "split_ids_json": str(split_path),
        },
        "bootstrap": {
            "dir": str(bootstrap_dir),
            "history_npy": str(hist_path),
            "train_ids_for_bootstrap_json": str(train_ids_json),
        },
        "ttm": {
            "model_path": model_path,
            "revision": revision,
            "fit_strategy": fit_strategy,
            "training_args": training_args,
        },
        "task": {
            "history_len": history_len,
            "pred_len": pred_len,
            "target_cols": target_cols,
        },
        "outputs": {
            "ctgan_synth_future_long": str(synth_out),
        },
        "n_synth": int(pred_panel[id_col].nunique()),
    }
    _save_json(meta, out_dir / "ctgan_run_meta.json")

    print(f"[ttm-ctgan] wrote: {synth_out}")
    print(f"[ttm-ctgan] wrote: {out_dir / 'ctgan_run_meta.json'}")
    print(f"[ttm-ctgan] wrote: {out_dir / 'ttm_config_resolved.yaml'}")


if __name__ == "__main__":
    main()
