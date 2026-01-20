from __future__ import annotations

import hashlib
import sys
import time
import platform
import numpy as np
import pandas as pd
import torch
import yaml

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from iTransformer import iTransformer  # type: ignore


# Repo root resolution and shared utils import
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def impute_long_df(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    value_cols: List[str],
    train_ids: List[str],
) -> pd.DataFrame:
    """
    Imputation for forecasting baselines:
    - for each patient: forward fill then backward fill (within-patient only)
    - remaining NaNs: fill with TRAIN split column means (no test leakage)
    - remaining NaNs (column fully missing in train): fill 0.0
    """
    df = df.copy()
    df = df.sort_values([id_col, time_col])

    # per-patient ffill + bfill
    df[value_cols] = df.groupby(id_col, sort=False)[value_cols].ffill()
    df[value_cols] = df.groupby(id_col, sort=False)[value_cols].bfill()

    # train split means for remaining NaNs
    train_mask = df[id_col].astype(str).isin(set(map(str, train_ids)))
    col_means = df.loc[train_mask, value_cols].mean(numeric_only=True)
    df[value_cols] = df[value_cols].fillna(col_means)

    # if still NaN (entire column missing in train)
    df[value_cols] = df[value_cols].fillna(0.0)

    return df


def _import_shared_utils():
    root = _repo_root()
    sys.path.insert(0, str(root))
    from models.s4m_model.src.utils import (
        set_seed,
        ensure_dir,
        load_parquet,
        save_json,
        df_checks,
    )
    return set_seed, ensure_dir, load_parquet, save_json, df_checks


set_seed, ensure_dir, load_parquet, save_json, df_checks = _import_shared_utils()


# Configs
def load_cfg(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p: str) -> Path:
    root = _repo_root()
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp)


# Splitting
@dataclass
class SplitCfg:
    method: str
    seed: int
    train: float
    val: float
    test: float


def patient_split(ids: List[Any], split: SplitCfg) -> Dict[str, List[Any]]:
    ids = list(ids)

    if split.method == "patient_hash":
        # stable across machines (no RNG), based only on patient id
        def bucket(x: Any) -> float:
            h = hashlib.sha256(str(x).encode("utf-8")).hexdigest()
            return int(h[:8], 16) / float(16**8)

        train_ids, val_ids, test_ids = [], [], []
        for pid in ids:
            r = bucket(pid)
            if r < split.train:
                train_ids.append(pid)
            elif r < split.train + split.val:
                val_ids.append(pid)
            else:
                test_ids.append(pid)
        return {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}

    if split.method != "patient_random":
        raise ValueError(f"Not real split.method: {split.method}")

    rng = np.random.default_rng(split.seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * split.train))
    n_val = int(round(n * split.val))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]

    return {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}


# Static covariate handling
def load_static_table(cfg: dict) -> Optional[pd.DataFrame]:
    if not cfg.get("enabled", False):
        return None

    source = cfg.get("source", "ctgan")
    if source not in {"ctgan", "real"}:
        raise ValueError(f"static.source must be 'ctgan' or 'real', got: {source}")

    path_key = "ctgan_path" if source == "ctgan" else "real_path"
    static_path = resolve_path(cfg[path_key])
    static_df = load_parquet(str(static_path))

    sid = cfg.get("id_col", "patient_id")
    if sid not in static_df.columns:
        raise ValueError(f"Baseline missing id_col='{sid}' in {static_path}")

    # enforce 1 row per patient
    if static_df.duplicated(subset=[sid]).any():
        raise ValueError(f"Baseline has duplicate ids in '{sid}': {static_path}")

    use_cols = cfg.get("use_cols", None)
    if use_cols is not None:
        use_cols = [c for c in use_cols if c in static_df.columns and c != sid]
        static_df = static_df[[sid] + use_cols].copy()

    return static_df


def attach_static_to_long(
    long_df: pd.DataFrame, static_df: pd.DataFrame, id_col: str, static_id_col: str
) -> Tuple[pd.DataFrame, List[str]]:
    merged = long_df.merge(
        static_df,
        left_on=id_col,
        right_on=static_id_col,
        how="left",
        validate="many_to_one",
        suffixes=("", "_static"),
    )
    if static_id_col != id_col and static_id_col in merged.columns:
        merged = merged.drop(columns=[static_id_col])

    static_cols = [c for c in static_df.columns if c != static_id_col]
    if static_cols:
        miss_all = merged[static_cols].isna().all(axis=1)
        if miss_all.any():
            raise ValueError(
                "Missing static features for some patients. "
                "Static table must cover all ids used in this run."
            )
    return merged, static_cols


# Windowing
def _sorted_patient_df(df: pd.DataFrame, id_col: str, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out[id_col] = out[id_col].astype(str)
    out = out.sort_values([id_col, time_col])
    return out


def build_windows_for_patients(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    input_cols: List[str],
    target_cols: List[str],
    context_len: int,
    horizon: int,
    step_stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X [N, context_len, C_in]
      Y [N, horizon, C_tgt]
    """
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []

    df = _sorted_patient_df(df, id_col, time_col)

    for _, pdf in df.groupby(id_col, sort=False):
        vals_in = pdf[input_cols].to_numpy(dtype=np.float32)
        vals_tgt = pdf[target_cols].to_numpy(dtype=np.float32)

        T = len(pdf)
        total = context_len + horizon
        if T < total:
            continue

        for start in range(0, T - total + 1, step_stride):
            x = vals_in[start : start + context_len]
            y = vals_tgt[start + context_len : start + total]
            X_list.append(x)
            Y_list.append(y)

    if not X_list:
        return (
            np.zeros((0, context_len, len(input_cols)), dtype=np.float32),
            np.zeros((0, horizon, len(target_cols)), dtype=np.float32),
        )

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        assert x.ndim == 3 and y.ndim == 3
        assert x.shape[0] == y.shape[0]
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"x": self.x[idx], "y": self.y[idx]}


# Training and eval
def configure_torch_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _get_pred_tensor(model_out: Any, horizon: int) -> torch.Tensor:
    """
    Supports iTransformer variants that return either:
      - Tensor [B, H, C] (most common)
      - dict {horizon: Tensor}
    """
    if isinstance(model_out, dict):
        if horizon not in model_out:
            raise KeyError(
                f"iTransformer returned horizons {list(model_out.keys())}, expected horizon={horizon}."
            )
        return model_out[horizon]
    if torch.is_tensor(model_out):
        return model_out
    raise TypeError(f"Unexpected model output type: {type(model_out)}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    horizon: int,
    target_idx: List[int],
    grad_clip_norm: float,
) -> float:
    model.train()
    mse = nn.MSELoss()
    total = 0.0
    n_batches = 0

    for batch in loader:
        x = batch["x"].to(device)  # [B, L, C_in]
        y_true = batch["y"].to(device)  # [B, H, C_tgt]

        model_out = model(x)
        y_pred_all = _get_pred_tensor(model_out, horizon)  # [B, H, C_in] or [B, H, C]
        y_pred = y_pred_all[:, :, target_idx]  # [B, H, C_tgt]

        loss = mse(y_pred, y_true)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        total += float(loss.detach().cpu())
        n_batches += 1

    return total / max(1, n_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    horizon: int,
    target_idx: List[int],
) -> float:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x = batch["x"].to(device)
        y_true = batch["y"].to(device)

        model_out = model(x)
        y_pred_all = _get_pred_tensor(model_out, horizon)
        y_pred = y_pred_all[:, :, target_idx]

        total_loss += float(mse(y_pred, y_true).detach().cpu())
        total_count += int(y_true.numel())

    return float("nan") if total_count == 0 else total_loss / total_count


# Inference exports
def _infer_future_times(last_time: Any, horizon: int) -> List[Any]:
    """
    Creates future time index for predicted rows.
      - numeric: last_time + 1..H
      - datetime: last_time + 1..H days
      - fallback: 1..H
    """
    if pd.api.types.is_datetime64_any_dtype(pd.Series([last_time])):
        base = pd.Timestamp(last_time)
        return [base + pd.Timedelta(days=i) for i in range(1, horizon + 1)]
    try:
        x = float(last_time)
        return [x + i for i in range(1, horizon + 1)]
    except Exception:
        return [i for i in range(1, horizon + 1)]


@torch.no_grad()
def export_forecast_horizon_artifacts(
    model: nn.Module,
    df_test: pd.DataFrame,
    id_col: str,
    time_col: str,
    input_cols: List[str],
    target_cols: List[str],
    context_len: int,
    horizon: int,
    device: torch.device,
    target_idx: List[int],
    min_history: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates 3 artifacts:
      preds_long_df: [id_col, target, horizon_step, pred]
      real_future_df: [id_col, time_col] + target_cols
      synth_future_df: same schema as real_future_df
    """
    model.eval()
    df_test = _sorted_patient_df(df_test, id_col, time_col)

    preds_long: List[Dict[str, Any]] = []
    real_rows: List[Dict[str, Any]] = []
    synth_rows: List[Dict[str, Any]] = []

    for pid, pdf in df_test.groupby(id_col, sort=False):
        pdf = pdf.sort_values(time_col)

        if len(pdf) < max(min_history, context_len + horizon):
            continue

        # define eval window as last (context + horizon)
        window = pdf.iloc[-(context_len + horizon) :]
        hist = window.iloc[:context_len]
        fut = window.iloc[context_len:]

        x_in = hist[input_cols].to_numpy(dtype=np.float32)  # [L, C_in]
        x = torch.from_numpy(x_in).unsqueeze(0).to(device)  # [1, L, C_in]

        model_out = model(x)
        y_pred_all = _get_pred_tensor(model_out, horizon)  # [1, H, C_in] or [1, H, C]
        y_pred = y_pred_all[:, :, target_idx].squeeze(0)  # [H, C_tgt]

        # preds_long.parquet
        for h in range(horizon):
            for j, tgt in enumerate(target_cols):
                preds_long.append(
                    {
                        id_col: str(pid),
                        "target": tgt,
                        "horizon_step": int(h + 1),
                        "pred": float(y_pred[h, j]),
                    }
                )

        # real_future_long.parquet (use actual timestamps)
        for i in range(horizon):
            row: Dict[str, Any] = {id_col: str(pid), time_col: fut.iloc[i][time_col]}
            for tgt in target_cols:
                row[tgt] = float(fut.iloc[i][tgt])
            real_rows.append(row)

        # synth_future_long.parquet (use same timestamps)
        fut_times = fut[time_col].tolist()
        if len(fut_times) != horizon:
            fut_times = _infer_future_times(hist.iloc[-1][time_col], horizon)

        for i in range(horizon):
            row = {id_col: str(pid), time_col: fut_times[i]}
            for j, tgt in enumerate(target_cols):
                row[tgt] = float(y_pred[i, j])
            synth_rows.append(row)

    preds_long_df = pd.DataFrame(preds_long)
    real_future_df = pd.DataFrame(real_rows)
    synth_future_df = pd.DataFrame(synth_rows)
    return preds_long_df, real_future_df, synth_future_df


# Main runner
def main(cfg_path: str | None = None) -> None:
    if cfg_path is None:
        cfg_path = str(Path(__file__).resolve().parents[1] / "configs" / "itransformer_base.yaml")

    cfg_file = resolve_path(cfg_path)
    cfg = load_cfg(cfg_file)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    configure_torch_determinism(seed)

    out_cfg = cfg.get("output", {})
    out_root = resolve_path(out_cfg.get("dir", "results/itransformer"))
    run_name = out_cfg.get("run_name", "itransformer_run")

    run_dir = out_root / run_name
    ensure_dir(str(run_dir))

    t0 = time.time()

    # save resolved config snapshot
    with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # data configuration
    data_cfg = cfg["data"]
    long_path = resolve_path(data_cfg["long_path"])
    id_col = str(data_cfg["id_col"])
    time_col = str(data_cfg["time_col"])
    target_cols = list(data_cfg["target_cols"])
    static_cfg = cfg.get("static", {"enabled": False})

    df = load_parquet(str(long_path))
    df_checks(df, id_col=id_col, time_col=time_col)

    missing_targets = [c for c in target_cols if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns in data: {missing_targets}")

    # attach static covariates (constant channels)
    static_df = load_static_table(static_cfg)
    if static_df is not None:
        static_id_col = static_cfg.get("id_col", id_col)
        df, static_cols = attach_static_to_long(df, static_df, id_col=id_col, static_id_col=static_id_col)
        static_cols = [c for c in static_cols if c not in set(target_cols)]
    else:
        static_cols = []

    input_cols = list(target_cols) + list(static_cols)

    # patient-level split
    split_cfg = cfg.get("split", {})
    split = SplitCfg(
        method=str(split_cfg.get("method", "patient_hash")),
        seed=int(split_cfg.get("seed", seed)),
        train=float(split_cfg.get("train", 0.7)),
        val=float(split_cfg.get("val", 0.1)),
        test=float(split_cfg.get("test", 0.2)),
    )

    ids = df[id_col].astype(str).unique().tolist()
    split_ids = patient_split(ids, split)
    save_json(split_ids, str(run_dir / "split_ids.json"))

    # impute targets (and static covariates) for stability
    value_cols = list(target_cols) + list(static_cols)
    df = impute_long_df(
        df=df,
        id_col=id_col,
        time_col=time_col,
        value_cols=value_cols,
        train_ids=split_ids["train_ids"],
    )

    # task config
    task_cfg = cfg.get("task", {})
    context_len = int(task_cfg.get("context_len", 96))
    horizon = int(task_cfg.get("horizon", 24))
    step_stride = int(task_cfg.get("step_stride", 1))
    min_history = int(task_cfg.get("min_history", context_len))

    df_train = df[df[id_col].astype(str).isin(set(map(str, split_ids["train_ids"])))].copy()
    df_val = df[df[id_col].astype(str).isin(set(map(str, split_ids["val_ids"])))].copy()
    df_test = df[df[id_col].astype(str).isin(set(map(str, split_ids["test_ids"])))].copy()

    X_train, Y_train = build_windows_for_patients(
        df_train, id_col, time_col, input_cols, target_cols, context_len, horizon, step_stride
    )
    X_val, Y_val = build_windows_for_patients(
        df_val, id_col, time_col, input_cols, target_cols, context_len, horizon, step_stride
    )

    if X_train.shape[0] == 0:
        raise RuntimeError("No training windows could be built.")

    optim_cfg = cfg.get("optim", {})
    batch_size = int(optim_cfg.get("batch_size", 32))
    num_workers = int(optim_cfg.get("num_workers", 0))

    train_loader = DataLoader(
        WindowDataset(X_train, Y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = (
        DataLoader(
            WindowDataset(X_val, Y_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if X_val.shape[0] > 0
        else None
    )

    # model params
    mcfg = cfg.get("model", {})
    mparams = dict(mcfg.get("params", {}) or {})
    num_variates = len(input_cols)

    mparams["num_variates"] = num_variates
    mparams["lookback_len"] = context_len
    mparams["pred_length"] = (horizon,)

    model = iTransformer(**mparams)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(optim_cfg.get("lr", 1e-3)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )

    epochs = int(optim_cfg.get("epochs", 10))
    grad_clip_norm = float(optim_cfg.get("grad_clip_norm", 0.0))

    # target columns are first in input_cols by construction
    target_idx = list(range(len(target_cols)))

    best_val_mse = float("inf")
    best_state: Optional[Dict[str, Any]] = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, horizon, target_idx, grad_clip_norm
        )

        if val_loader is not None:
            val_mse = evaluate(model, val_loader, device, horizon, target_idx)
        else:
            val_mse = float("nan")

        print(f"iTransformer [Epoch {epoch:03d}] train_loss={train_loss:.6f} val_mse={val_mse:.6f}")

        improved = (best_state is None) or (val_loader is None) or (np.isfinite(val_mse) and val_mse < best_val_mse)
        if improved:
            if val_loader is not None and np.isfinite(val_mse):
                best_val_mse = val_mse
            best_state = {
                "model_state_dict": model.state_dict(),
                "model_params": mparams,
                "seed": seed,
                "input_cols": input_cols,
                "target_cols": target_cols,
                "static_cols": static_cols,
                "context_len": context_len,
                "horizon": horizon,
            }

    ckpt_path = run_dir / "model.pt"
    if best_state is None:
        raise RuntimeError("Training didn't produce a checkpoint.")
    torch.save(best_state, ckpt_path)

    # export evaluation artifacts (on current model weights)
    preds_long_df, real_future_df, synth_future_df = export_forecast_horizon_artifacts(
        model=model,
        df_test=df_test,
        id_col=id_col,
        time_col=time_col,
        input_cols=input_cols,
        target_cols=target_cols,
        context_len=context_len,
        horizon=horizon,
        device=device,
        target_idx=target_idx,
        min_history=min_history,
    )

    preds_path = run_dir / "preds_long.parquet"
    preds_long_df.to_parquet(preds_path, index=False)

    real_future_path = run_dir / "real_future_long.parquet"
    synth_future_path = run_dir / "synth_future_long.parquet"
    real_future_df.to_parquet(real_future_path, index=False)
    synth_future_df.to_parquet(synth_future_path, index=False)

    t1 = time.time()

    run_meta: Dict[str, Any] = {
        "seed": seed,
        "framework": "PyTorch",
        "model_impl": "lucidrains/iTransformer",
        "device": str(device),
        "run_dir": str(run_dir),
        "started_unix": float(t0),
        "ended_unix": float(t1),
        "duration_sec": float(t1 - t0),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_patients_total": int(df[id_col].nunique()),
        "n_patients_train": int(len(split_ids["train_ids"])),
        "n_patients_val": int(len(split_ids["val_ids"])),
        "n_patients_test": int(len(split_ids["test_ids"])),
        "n_train_windows": int(X_train.shape[0]),
        "n_val_windows": int(X_val.shape[0]),
        "context_len": int(context_len),
        "horizon": int(horizon),
        "step_stride": int(step_stride),
        "targets": target_cols,
        "static_enabled": bool(static_df is not None),
        "static_source": static_cfg.get("source", None) if static_df is not None else None,
        "n_static_cols": int(len(static_cols)),
        "pred_rows": int(len(preds_long_df)),
        "real_future_rows": int(len(real_future_df)),
        "synth_future_rows": int(len(synth_future_df)),
        "best_val_mse": float(best_val_mse),
        "checkpoint": str(ckpt_path),
        "preds_path": str(preds_path),
        "real_future_path": str(real_future_path),
        "synth_future_path": str(synth_future_path),
        "split_ids_path": str(run_dir / "split_ids.json"),
    }
    save_json(run_meta, str(run_dir / "run_meta.json"))

    print(f"iTransformer saved checkpoint: {ckpt_path}")
    print(f"iTransformer saved preds: {preds_path}")
    print(f"iTransformer saved real future: {real_future_path}")
    print(f"iTransformer saved synth future: {synth_future_path}")
    print(f"iTransformer saved meta: {run_dir / 'run_meta.json'}")
    print(f"iTransformer saved split: {run_dir / 'split_ids.json'}")


if __name__ == "__main__":
    cfg_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg_arg)
