#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from likelihood_fitness import evaluate_likelihood_fitness
from ml_efficacy import evaluate_ml_efficacy


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def numeric_metrics(real: pd.Series, syn: pd.Series) -> Dict[str, float]:
    r = real.dropna().astype(float)
    s = syn.dropna().astype(float)

    out = {
        "real_mean": float(r.mean()) if len(r) else np.nan,
        "syn_mean": float(s.mean()) if len(s) else np.nan,
        "real_std": float(r.std()) if len(r) else np.nan,
        "syn_std": float(s.std()) if len(s) else np.nan,
    }

    if len(r) >= 10 and len(s) >= 10:
        ks = ks_2samp(r, s)
        out["ks_stat"] = float(ks.statistic)
        out["ks_pvalue"] = float(ks.pvalue)
    else:
        out["ks_stat"] = np.nan
        out["ks_pvalue"] = np.nan

    return out


def base_eval(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"per_column": {}, "scorecard": {}}

    for col in real_df.columns:
        if col not in syn_df.columns:
            continue

        r = real_df[col]
        s = syn_df[col]

        col_res = {
            "missing_rate_real": float(r.isna().mean()),
            "missing_rate_syn": float(s.isna().mean()),
        }

        if pd.api.types.is_numeric_dtype(r):
            col_res["type"] = "numeric"
            col_res.update(numeric_metrics(r, s))
        else:
            col_res["type"] = "categorical"

        metrics["per_column"][col] = col_res

    ks_vals = [
        v["ks_stat"]
        for v in metrics["per_column"].values()
        if v.get("type") == "numeric" and not np.isnan(v.get("ks_stat", np.nan))
    ]

    miss_diffs = [
        abs(v["missing_rate_real"] - v["missing_rate_syn"])
        for v in metrics["per_column"].values()
    ]

    metrics["scorecard"] = {
        "mean_ks": float(np.mean(ks_vals)) if ks_vals else np.nan,
        "mean_missingness_diff": float(np.mean(miss_diffs)),
        "n_columns": len(metrics["per_column"]),
    }

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--target_col", required=True)
    ap.add_argument("--id_col", default="patient_id")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()

    with open(run_dir / "ctgan_config.json", "r") as f:
        cfg = json.load(f)

    real_df = load_table(Path(cfg["real_static_path"]))
    syn_df = load_table(Path(cfg["synthetic_static_path"]))

    discrete_cols: List[str] = cfg.get("discrete_columns", [])

    # drop ID col
    real_df = real_df.drop(columns=[args.id_col], errors="ignore")
    syn_df = syn_df.drop(columns=[args.id_col], errors="ignore")

    results: Dict[str, Any] = {}

    # Base stats
    results["base"] = base_eval(real_df, syn_df)

    # Likelihood fitness
    try:
        results["likelihood_fitness"] = evaluate_likelihood_fitness(
            real_df=real_df,
            syn_df=syn_df,
            discrete_cols=discrete_cols,
        )
    except Exception as e:
        results["likelihood_fitness_error"] = str(e)

    # ML efficacy
    try:
        results["ml_efficacy"] = evaluate_ml_efficacy(
            real_df=real_df,
            syn_df=syn_df,
            target_col=args.target_col,
            discrete_cols=discrete_cols,
        )
    except Exception as e:
        results["ml_efficacy_error"] = str(e)

    out_path = run_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[CTGAN eval] saved → {out_path}")


if __name__ == "__main__":
    main()
