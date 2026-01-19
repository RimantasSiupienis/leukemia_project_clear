import argparse
import time
import numpy as np
import pandas as pd
import yaml

from pathlib import Path
from models.s4m_model.src.utils import set_seed, ensure_dir, save_json

from ardm_adapter_dataset import prep_dataset
from official_api import train_and_predict_official, get_official_git_commit



def write_config_resolved(cfg: dict, out_dir: Path) -> None:
    p = out_dir / "config_resolved.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def reshape_preds(preds: np.ndarray, n_series: int, pred_len: int, n_targets: int) -> np.ndarray:
    """
    Returns shape: [n_series, pred_len, n_targets]
    Layouts function can handle:
      - [n_series, pred_len, n_targets]
      - [n_series, n_targets, pred_len]
      - [n_series, pred_len * n_targets]
      - [n_series, n_targets * pred_len]
    """
    if preds.ndim == 3:
        if preds.shape == (n_series, pred_len, n_targets):
            return preds
        if preds.shape == (n_series, n_targets, pred_len):
            return np.transpose(preds, (0, 2, 1))
        raise ValueError(f"Unexpected 3D predictions shape: {preds.shape}")

    if preds.ndim == 2:
        if preds.shape[0] != n_series:
            raise ValueError(f"Predictions first dim {preds.shape[0]} != n_series {n_series}")
        flat = preds
        if flat.shape[1] == pred_len * n_targets:
            return flat.reshape(n_series, pred_len, n_targets)
        raise ValueError(f"Unexpected 2D predictions shape: {preds.shape}")

    raise ValueError(f"Unexpected predictions ndim={preds.ndim}, shape={preds.shape}")


def build_synth_future_long_from_real_index(
    real_future_long: pd.DataFrame,
    id_col: str,
    time_col: str,
    target_cols: list,
    preds_3d: np.ndarray,
) -> pd.DataFrame:
    """
    Sorts synth_future_long to match real_future_long ordering:
      - groups by patient in the same order as they appear in real_future_long
      - for each patient: replace target columns with predictions in time order
    real_future_long has to contain pred_len rows for every test patient each.
    """
    df = real_future_long.copy()
    df[id_col] = df[id_col].astype(str)

    patient_order = df[id_col].drop_duplicates().tolist()
    pred_len = preds_3d.shape[1]
    n_targets = preds_3d.shape[2]
    if n_targets != len(target_cols):
        raise ValueError("Predictions targets dim doesn't match target_cols")

    rows = []
    for i, pid in enumerate(patient_order):
        g = df[df[id_col] == str(pid)].sort_values(time_col)
        if len(g) != pred_len:
            raise ValueError(f"Patient {pid} has {len(g)} future rows, expected {pred_len}")
        out_g = g[[id_col, time_col] + target_cols].copy()
        out_g[target_cols] = preds_3d[i]
        rows.append(out_g)

    out = pd.concat(rows, axis=0, ignore_index=True)
    return out


def main(cfg_path: str) -> None:
    """
    Main ARDM training + prediction runner
    """

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["seed"]))

    repo = Path(cfg["official"]["repo_dir"])
    if not repo.exists():
        raise RuntimeError("ARDM repo not cloned. Expected at cfg.official.repo_dir")

    out_root = Path(cfg["output"]["dir"])
    run_name = str(cfg["output"]["run_name"])
    out_dir = out_root / run_name
    ensure_dir(out_dir)

    write_config_resolved(cfg, out_dir)

    t0 = time.time()

    train_p, val_p, test_p, manifest = prep_dataset(cfg)

    save_json({"split_ids": manifest["split_ids"]}, out_dir / "split_ids.json")

    pred_npy = train_and_predict_official(
        cfg,
        train_path=train_p,
        val_path=val_p,
        test_path=test_p,
        out_dir=out_dir / "_official_out",
    )


    preds = np.load(pred_npy)

    real_future_long = pd.read_parquet(manifest["paths"]["real_future_long"])

    id_col = cfg["data"]["id_col"]
    time_col = cfg["data"]["time_col"]
    target_cols = list(cfg["data"]["target_cols"])
    pred_len = int(cfg["data"]["pred_len"])

    real_future_long[id_col] = real_future_long[id_col].astype(str)
    test_patient_order = real_future_long[id_col].drop_duplicates().tolist()
    n_series = len(test_patient_order)
    n_targets = len(target_cols)

    preds_3d = reshape_preds(preds, n_series=n_series, pred_len=pred_len, n_targets=n_targets)

    synth_future_long = build_synth_future_long_from_real_index(
        real_future_long=real_future_long,
        id_col=id_col,
        time_col=time_col,
        target_cols=target_cols,
        preds_3d=preds_3d,
    )

    real_future_long.to_parquet(out_dir / "real_future_long.parquet", index=False)
    synth_future_long.to_parquet(out_dir / "synth_future_long.parquet", index=False)

    run_meta = {
        "model": "ARDM",
        "start_time_unix": t0,
        "end_time_unix": time.time(),
        "seed": int(cfg["seed"]),
        "official_repo_dir": str(repo),
        "official_git_commit": get_official_git_commit(cfg),
        "cache_key": manifest.get("cache_key"),
        "n_test_patients": int(n_series),
        "pred_len": int(pred_len),
        "targets": target_cols,
        "artifacts": {
            "config_resolved": "config_resolved.yaml",
            "split_ids": "split_ids.json",
            "run_meta": "run_meta.json",
            "real_future_long": "real_future_long.parquet",
            "synth_future_long": "synth_future_long.parquet",
        },
    }
    save_json(run_meta, out_dir / "run_meta.json")

    print(f"ARDM wrote: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
