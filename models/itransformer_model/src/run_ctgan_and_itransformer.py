import os
import sys
from pathlib import Path
import argparse
import subprocess
import numpy as np
import pandas as pd
import yaml
import torch
from iTransformer import iTransformer  # type: ignore

THIS_FILE = Path(__file__).resolve()
# .parents[0] = scripts/, [1] = /src, [2] = models/, [3] = repo root
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(REPO_ROOT))

def _load_cfg(path: str):
    """Loading a YAML configuration file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _parse_target_cols(obj):
    """Normalising target_cols to a list of strings."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, list):
        return [str(x) for x in obj]
    raise ValueError("data.target_cols must be a string or list of strings")

def _default_out_dir(cfg: dict, run_dir: Path) -> Path:
    """
    Build default output directory:
    results/itransformer/<dataset>_ctgan_itransformer/
    Derives <dataset> from config output dir or CTGAN path, and appends suffix.
    """
    out_cfg = cfg.get("output", {}) or {}
    base_dir = out_cfg.get("dir") or "results/itransformer"
    base_path = Path(base_dir)
    # Derive dataset name: try from output.dir or fall back to CTGAN static path
    dataset_name = base_path.name if base_path.name not in ("itransformer", "") else "unknown_dataset"
    # If output.dir is a generic path without dataset, attempt to get name from long_path or static path
    try:
        # e.g., data.long_path might contain dataset name
        long_path = str(cfg["data"].get("long_path", ""))
        if long_path:
            # e.g., "data/processed/physionet2012_long.parquet" -> "physionet2012"
            stem = Path(long_path).stem
            if stem.endswith("_long"):
                dataset_name = stem[: -len("_long")]
    except Exception:
        pass
    # If static CTGAN path is available, use its dataset folder name
    try:
        static_cfg = cfg.get("static", {}) or {}
        ctgan_path = static_cfg.get("ctgan_path") or static_cfg.get("fake_path") or ""
        if ctgan_path:
            dataset_dir = Path(ctgan_path).parent.name
            if dataset_dir:
                dataset_name = str(dataset_dir)
    except Exception:
        pass
    run_suffix = "ctgan_itransformer"
    # Use run_dir name if it seems to contain dataset (to avoid double dataset prefix)
    run_name = run_dir.name
    if dataset_name and not run_name.startswith(dataset_name):
        # Compose new run name with dataset
        out_run_name = f"{dataset_name}_{run_suffix}"
    else:
        out_run_name = f"{run_name}_{run_suffix}" if run_name else run_suffix
    return REPO_ROOT / "results" / "itransformer" / out_run_name

def _get_pred_tensor(model_out, horizon: int):
    # Works with iTransformer variants that return either:
    # - Tensor [B, H, C]
    # - dict [horizon: Tensor]
    if isinstance(model_out, dict):
        if horizon not in model_out:
            raise KeyError(f"iTransformer returned horizons {list(model_out.keys())}, expected horizon={horizon}")
        return model_out[horizon]
    if torch.is_tensor(model_out):
        return model_out
    raise TypeError(f"Unexpected model output type: {type(model_out)}")


def _bootstrap_history(ctgan_static_parquet: str,
                       real_long_parquet: str,
                       out_dir: Path,
                       train_split_ids_json: str,
                       id_col: str,
                       time_col: str,
                       target_cols: list,
                       history_len: int,
                       pred_len: int,
                       seed: int):
    """Run the bootstrap script to generate initial history windows for synthetic patients."""
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = REPO_ROOT / "scripts" / "bootstrap_ctgan_to_history.py"
    if not script_path.exists():
        raise FileNotFoundError(f"bootstrap_ctgan_to_history.py not found at {script_path}")
    cmd = [
        sys.executable, str(script_path),
        "--ctgan_static_parquet", ctgan_static_parquet,
        "--real_long_parquet", real_long_parquet,
        "--out_dir", str(out_dir),
        "--id_col", id_col,
        "--time_col", time_col,
        "--target_cols", ",".join(target_cols),
        "--history_len", str(history_len),
        "--pred_len", str(pred_len),
        "--seed", str(seed),
        "--train_split_ids_json", train_split_ids_json,
        "--non_strict_baseline"
    ]
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser(description="CTGAN static baseline + iTransformer synthetic future generation")
    ap.add_argument("--itransformer_config", required=True, type=str, help="Path to iTransformer YAML config")
    ap.add_argument("--itransformer_run_dir", required=True, type=str, help="Path to a completed iTransformer run directory (with model.pt and split_ids.json)")
    ap.add_argument("--ctgan_static_parquet", type=str,
                    default=None, help="Path to CTGAN synthetic static output Parquet (default uses config.static.ctgan_path)")
    ap.add_argument("--device", default="cpu", type=str, help="Computation device, 'cpu' or 'cuda'")
    ap.add_argument("--out_dir", default=None, type=str,
                    help="Output folder for synthetic trajectories. Default is results/itransformer/<dataset>_ctgan_itransformer/")
    args = ap.parse_args()

    # Loading config and set seed for reproducibility
    cfg = _load_cfg(args.itransformer_config)
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # Ensuring reproducibility (may slow down)
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    data_cfg = cfg["data"]
    id_col = str(data_cfg["id_col"])
    time_col = str(data_cfg["time_col"])
    target_cols = _parse_target_cols(data_cfg.get("target_cols", []))
    if not target_cols:
        raise ValueError("No target_cols specified in config")

    task_cfg = cfg.get("task", {}) or {}
    context_len = int(task_cfg.get("context_len", 0))
    horizon = int(task_cfg.get("horizon", 0))
    if context_len <= 0 or horizon <= 0:
        raise ValueError("context_len and horizon must be positive in config.task")

    # Determines CTGAN static parquet path
    ctgan_static_path = args.ctgan_static_parquet
    if ctgan_static_path is None:
        static_cfg = cfg.get("static", {}) or {}
        ctgan_static_path = static_cfg.get("ctgan_path") or static_cfg.get("fake_path")
        if not ctgan_static_path:
            raise ValueError("CTGAN static Parquet path not provided and not found in config.static")
        ctgan_static_path = str(Path(ctgan_static_path))
    # Determines real longitudinal data path
    long_path = str(cfg["data"]["long_path"])

    # Ensures needed files exist
    itransformer_run_dir = Path(args.itransformer_run_dir)
    if not itransformer_run_dir.exists():
        raise FileNotFoundError(f"iTransformer run directory not found: {itransformer_run_dir}")
    split_ids_json = itransformer_run_dir / "split_ids.json"
    if not split_ids_json.exists():
        raise FileNotFoundError(f"split_ids.json missing in {itransformer_run_dir}")
    ckpt_path = itransformer_run_dir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"model.pt (iTransformer checkpoint) not found in {itransformer_run_dir}")

    # Determines output directory
    if args.out_dir is None:
        out_dir = _default_out_dir(cfg, itransformer_run_dir)
    else:
        out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    bootstrap_dir = out_dir / "bootstrap"
    os.makedirs(bootstrap_dir, exist_ok=True)

    # 1. Bootstraping synth patient history from CTGAN static output
    _bootstrap_history(ctgan_static_parquet=ctgan_static_path,
                       real_long_parquet=long_path,
                       out_dir=bootstrap_dir,
                       train_split_ids_json=str(split_ids_json),
                       id_col=id_col,
                       time_col=time_col,
                       target_cols=target_cols,
                       history_len=context_len,
                       pred_len=horizon,
                       seed=seed)

    # Loading the bootstrapped history windows
    hist_path = bootstrap_dir / "bootstrap_history.npy"
    if not hist_path.exists():
        raise FileNotFoundError("bootstrap_history.npy not found (bootstrap step failed)")
    history = np.load(hist_path)
    if history.ndim != 3:
        raise RuntimeError("bootstrap_history.npy must be a 3D array")
    n, L, C = history.shape
    if L != context_len or C != len(target_cols):
        raise RuntimeError(f"Bootstrap history shape mismatch: expected ({context_len}, {len(target_cols)}) got ({L}, {C})")

    # 2. Loading trained iTransformer model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_params = ckpt.get("model_params", {})
    # Instantiate model with saved parameters
    model = iTransformer(**model_params)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if args.device.lower() == "cuda" and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Preparing input data for model
    # Includes static covariates if used
    static_cols = []
    static_cfg = cfg.get("static", {}) or {}
    if static_cfg.get("enabled", False):
        static_cols = static_cfg.get("use_cols", []) or []

    
    ctgan_df = pd.read_parquet(ctgan_static_path)
    MAX_SYNTH = 200  # laptop smoke test (forget)
    ctgan_df = ctgan_df.head(MAX_SYNTH).copy()

    if id_col not in ctgan_df.columns:
        raise ValueError(f"ID column '{id_col}' missing in CTGAN static Parquet")
    # Aligning synth ids with history array
    synth_ids = ctgan_df[id_col].astype(str).tolist()
    if len(synth_ids) != n:
        m = min(len(synth_ids), n)
        synth_ids = synth_ids[:m]
        history = history[:m]
        n = m
    # Building model input array [N, L, input_features]
    history = history.astype(np.float32)
    if static_cols:
        for col in static_cols:
            if col not in ctgan_df.columns:
                raise ValueError(f"Static column '{col}' not found in CTGAN output")
        static_vals = ctgan_df.loc[: n-1, static_cols].to_numpy(dtype=np.float32)
        # Repeat static vals across the context length dimension
        static_expanded = np.repeat(static_vals[:, np.newaxis, :], L, axis=1)  # shape (n, L, len(static_cols))
        X = np.concatenate([history, static_expanded], axis=2).astype(np.float32)
    else:
        X = history # shape [n, L, C_target_only]
    # Converting to torch tensor and run model prediction
    # Doing inference in batches to avoid freezing
    BATCH = 64  #set 16 or 32 if laptop lags
    pred_chunks = []
    model.eval()
    with torch.no_grad():
        for i in range(0, X.shape[0], BATCH):
            xb = torch.from_numpy(X[i:i+BATCH]).to(device)
            out = model(xb)
            yb = _get_pred_tensor(out, horizon)  # Tensor [B, H, C]
            pred_chunks.append(yb.detach().cpu().numpy())

    preds_all = np.concatenate(pred_chunks, axis=0) # [N, H, C_in]
    target_idx = list(range(len(target_cols)))
    y_pred = preds_all[:, :, target_idx] # [N, H, C_tgt]


    # 3. Constructing output dataframe of synth future trajectories
    N, H, C_tgt = y_pred.shape
    # Preparing patient IDs and time indices for each forecasted step
    id_values = np.repeat([str(pid) for pid in synth_ids[:N]], H)
    time_values = np.tile(np.arange(1, H + 1), N)
    data = {id_col: id_values, time_col: time_values}
    # Add each target column's predictions
    y_flat = y_pred.reshape(N * H, C_tgt)
    for j, col in enumerate(target_cols):
        data[col] = y_flat[:, j].astype(float)
    synth_future_df = pd.DataFrame(data)


    # Saving synth trajectories to parquet
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = traj_dir / "synth_future_long.parquet"

    synth_future_df.to_parquet(out_parquet, index=False)
    print(f"[itransformer] wrote {out_parquet}")

    # 4. Saving run metadata and resolved config
    meta = {
        "mode": "ctgan_static_itransformer_full",
        "seed": seed,
        "itransformer_run_dir": str(itransformer_run_dir.resolve()),
        "inputs": {
            "ctgan_static_parquet": str(Path(ctgan_static_path).resolve()),
            "real_long_parquet": str(Path(long_path).resolve()),
            "split_ids_json": str(split_ids_json.resolve()),
        },
        "bootstrap": {
            "dir": str(bootstrap_dir.resolve()),
            "history_npy": str(hist_path.resolve()),
        },
        "itransformer": {
            "model_impl": "lucidrains/iTransformer",
            "device": str(device),
            "params": None,
        },
        "task": {
            "context_len": context_len,
            "horizon": horizon,
            "targets": target_cols,
        },
        "outputs": {
            "ctgan_itransformer_synth_future_long": str(out_parquet.resolve()),
        },
        "n_synth": int(N),
    }
    # Includes model params 
    # converts tuple to list for JSON compatibility
    mp = model_params.copy()
    if "pred_length" in mp and isinstance(mp["pred_length"], tuple):
        mp["pred_length"] = list(mp["pred_length"])
    meta["itransformer"]["params"] = mp
    # Writes metadata JSON
    meta_path = traj_dir / "run_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    print("[itransformer] wrote run_meta.json")
    # Writes resolved config YAML snapshot
    config_res_path = traj_dir / "itransformer_config_resolved.yaml"
    with open(config_res_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("[itransformer] wrote itransformer_config_resolved.yaml")

if __name__ == "__main__":
    main()
