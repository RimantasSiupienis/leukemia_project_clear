import os
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()

# parents[0]=src, [1]=chronos_model, [2]=models, [3]=leukemia_project
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import subprocess
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from models.s4m_model.src.utils import (  # type: ignore
    ensure_dir,
    set_seed,
    save_json,
)

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.chronos import ChronosForecaster


def _load_cfg(p: str) -> Dict[str, Any]:
    # loads yaml config file
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_target_cols(obj: Any) -> List[str]:
    # normalizes target columns format
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, list):
        return [str(x) for x in obj]
    raise ValueError("data.target_cols must be a string or list of strings")


def _bootstrap_history(
    *,
    ctgan_static_parquet: str,
    real_long_parquet: str,
    out_dir: Path,
    train_split_ids_json: str,
    id_col: str,
    time_col: str,
    target_cols: List[str],
    history_len: int,
    pred_len: int,
    seed: int,
) -> None:
    # creates initial history windows for synthetic patients
    out_dir.mkdir(parents=True, exist_ok=True)

    script = REPO_ROOT / "scripts" / "bootstrap_ctgan_to_history.py"
    if not script.exists():
        raise FileNotFoundError(f"bootstrap_ctgan_to_history.py not found at {script}")

    cmd = [
        sys.executable,
        str(script),
        "--ctgan_static_parquet",
        ctgan_static_parquet,
        "--real_long_parquet",
        real_long_parquet,
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
        train_split_ids_json,
        "--non_strict_baseline",
    ]

    subprocess.run(cmd, check=True)


def _default_full_traj_out_dir(cfg: Dict[str, Any], chronos_run_dir: Path) -> Path:
    # builds results/chronos/full_ctgan_trajectories/dataset/run_name
    out_cfg = cfg.get("output", {}) or {}
    dataset_dir = "unknown_dataset"

    cfg_out_dir = out_cfg.get("dir")
    if isinstance(cfg_out_dir, str) and cfg_out_dir.strip():
        dataset_dir = Path(cfg_out_dir).name

    run_name = chronos_run_dir.name
    base = REPO_ROOT / "results" / "chronos" / "full_ctgan_trajectories" / dataset_dir / run_name
    return base


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CTGAN static baseline + Chronos synthetic future generation"
    )
    ap.add_argument("--chronos_config", required=True, type=str)
    ap.add_argument("--chronos_run_dir", required=True, type=str)
    ap.add_argument("--ctgan_static_parquet", required=True, type=str)
    ap.add_argument("--device", default="cpu", type=str)
    ap.add_argument(
        "--out_dir",
        default=None,
        type=str,
        help="Output folder for full CTGAN trajectories. Default is results/chronos/full_ctgan_trajectories/<dataset>/<run_name>.",
    )

    args = ap.parse_args()

    # reduces huggingface log spam during repeated predict calls (debugging)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cfg = _load_cfg(args.chronos_config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    data_cfg = cfg["data"]
    task_cfg = cfg.get("task", {})
    model_cfg = cfg.get("model", {})
    model_params = (model_cfg.get("params", {}) or {}).copy()

    long_path = str(data_cfg["long_path"])
    id_col = str(data_cfg["id_col"])
    time_col = str(data_cfg["time_col"])
    target_cols = _parse_target_cols(data_cfg["target_cols"])

    context_len = int(task_cfg["context_len"])
    horizon = int(task_cfg["horizon"])

    chronos_run_dir = Path(args.chronos_run_dir)
    ensure_dir(str(chronos_run_dir))

    split_ids_json = chronos_run_dir / "split_ids.json"
    if not split_ids_json.exists():
        raise FileNotFoundError(f"split_ids.json missing in {chronos_run_dir}")

    if args.out_dir is None:
        out_dir = _default_full_traj_out_dir(cfg, chronos_run_dir)
    else:
        out_dir = Path(args.out_dir)

    ensure_dir(str(out_dir))

    pipeline_dir = out_dir / "ctgan_pipeline"
    ensure_dir(str(pipeline_dir))

    bootstrap_dir = pipeline_dir / "bootstrap"
    ensure_dir(str(bootstrap_dir))

    _bootstrap_history(
        ctgan_static_parquet=args.ctgan_static_parquet,
        real_long_parquet=long_path,
        out_dir=bootstrap_dir,
        train_split_ids_json=str(split_ids_json),
        id_col=id_col,
        time_col=time_col,
        target_cols=target_cols,
        history_len=context_len,
        pred_len=horizon,
        seed=seed,
    )

    hist_path = bootstrap_dir / "bootstrap_history.npy"
    if not hist_path.exists():
        raise FileNotFoundError("bootstrap_history.npy not created")

    history = np.load(hist_path)
    if history.ndim != 3:
        raise RuntimeError("bootstrap_history.npy must be 3D")

    n, L, C = history.shape
    if L != context_len or C != len(target_cols):
        raise RuntimeError("bootstrap_history.npy shape mismatch")

    model_path = str(model_params.get("model_path", "amazon/chronos-t5-small"))
    use_source_package = bool(model_params.get("use_source_package", False))
    ignore_deps = bool(model_params.get("ignore_deps", False))

    chronos_cfg = {
        k: v
        for k, v in model_params.items()
        if k not in {"model_path", "use_source_package", "ignore_deps"}
    }

    if args.device.lower() == "cuda":
        chronos_cfg["device_map"] = "cuda"
    else:
        chronos_cfg["device_map"] = "cpu"

    forecaster = ChronosForecaster(
        model_path=model_path,
        config=chronos_cfg if chronos_cfg else None,
        seed=seed,
        use_source_package=use_source_package,
        ignore_deps=ignore_deps,
    )

    fh = ForecastingHorizon(list(range(1, horizon + 1)), is_relative=True)

    # warmup run so any debug initialization happens once
    try:
        y0 = pd.Series(np.zeros(context_len, dtype=float))
        forecaster.fit(y0, fh=fh)
        _ = forecaster.predict(fh=fh)
    except Exception:
        pass

    ctgan_df = pd.read_parquet(args.ctgan_static_parquet)
    if id_col not in ctgan_df.columns:
        raise ValueError("id column missing in CTGAN parquet")

    synth_ids = ctgan_df[id_col].astype(str).tolist()

    if len(synth_ids) != n:
        m = min(len(synth_ids), n)
        synth_ids = synth_ids[:m]
        history = history[:m]
        n = m

    rows = []
    for i in range(n):
        pid = synth_ids[i]
        hist = history[i]

        t_vals = list(range(1, horizon + 1))
        block = {id_col: [pid] * horizon, time_col: t_vals}

        for j, col in enumerate(target_cols):
            y = pd.Series(hist[:, j].astype(float))
            if y.isna().any():
                y = y.ffill().bfill().fillna(0.0)

            forecaster.fit(y, fh=fh)
            yhat = forecaster.predict(fh=fh)
            block[col] = np.asarray(yhat).reshape(-1).astype(float)

        rows.append(pd.DataFrame(block))

    if not rows:
        raise RuntimeError("no synthetic predictions generated")

    synth_future = pd.concat(rows, ignore_index=True)

    out_parquet = pipeline_dir / "ctgan_synth_future_long.parquet"
    synth_future.to_parquet(out_parquet, index=False)
    print(f"[chronos] wrote {out_parquet}")

    meta = {
        "mode": "ctgan_static_chronos_full",
        "seed": seed,
        "chronos_run_dir": str(chronos_run_dir.resolve()),
        "inputs": {
            "ctgan_static_parquet": str(Path(args.ctgan_static_parquet).resolve()),
            "real_long_parquet": str(Path(long_path).resolve()),
            "split_ids_json": str(split_ids_json.resolve()),
        },
        "bootstrap": {
            "dir": str(bootstrap_dir.resolve()),
            "history_npy": str(hist_path.resolve()),
        },
        "chronos": {
            "model_path": model_path,
            "device": args.device,
            "config": chronos_cfg,
        },
        "task": {
            "context_len": context_len,
            "horizon": horizon,
            "targets": target_cols,
        },
        "outputs": {
            "ctgan_synth_future_long": str(out_parquet.resolve()),
        },
        "n_synth": int(n),
    }

    save_json(meta, str(pipeline_dir / "ctgan_run_meta.json"))
    print("[chronos] wrote ctgan_run_meta.json")

    with open(pipeline_dir / "chronos_config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("[chronos] wrote chronos_config_resolved.yaml")


if __name__ == "__main__":
    main()
