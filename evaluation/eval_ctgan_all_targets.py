#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from likelihood_fitness import evaluate_likelihood_fitness
from ml_efficacy import evaluate_ml_efficacy  # :contentReference[oaicite:0]{index=0}


def align_common(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = [c for c in real_df.columns if c in syn_df.columns]
    return real_df[common].copy(), syn_df[common].copy()


def simple_impute_like_ctgan(df: pd.DataFrame, discrete_cols: list[str]) -> pd.DataFrame:
    out = df.copy()

    # drop all-null cols (otherwise median stays NaN forever)
    all_null = [c for c in out.columns if out[c].isna().all()]
    if all_null:
        out = out.drop(columns=all_null)

    for c in out.columns:
        if not out[c].isna().any():
            continue

        if c in discrete_cols or pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_bool_dtype(out[c]):
            mode_val = out[c].mode(dropna=True)
            fill_val = mode_val.iloc[0] if len(mode_val) else "MISSING"
            out[c] = out[c].fillna(fill_val)
        else:
            med = out[c].median()
            if pd.isna(med):
                med = 0.0
            out[c] = out[c].fillna(med)

    return out


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def pick_targets(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    *,
    exclude_missing_flags: bool = True,
    exclude_cols: List[str] | None = None,
) -> List[str]:
    common = [c for c in real_df.columns if c in syn_df.columns]
    if exclude_cols:
        common = [c for c in common if c not in set(exclude_cols)]
    if exclude_missing_flags:
        common = [c for c in common if not c.endswith("__is_missing")]
    return common


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--id_col", default="patient_id")
    ap.add_argument("--include_missing_flags", action="store_true")
    ap.add_argument("--max_targets", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg_path = run_dir / "ctgan_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    real_path = Path(cfg["real_static_path"])
    syn_path = Path(cfg["synthetic_static_path"])
    real_df = load_table(real_path)
    syn_df = load_table(syn_path)

    discrete_cols: List[str] = cfg.get("discrete_columns", [])

    # drop ID column
    real_df = real_df.drop(columns=[args.id_col], errors="ignore")
    syn_df = syn_df.drop(columns=[args.id_col], errors="ignore")

    results: Dict[str, Any] = {"meta": {}, "likelihood_fitness": None, "ml_efficacy": {}}

    results["meta"] = {
        "run_dir": str(run_dir),
        "real_path": str(real_path),
        "syn_path": str(syn_path),
        "id_col_dropped": args.id_col,
        "n_real": int(len(real_df)),
        "n_syn": int(len(syn_df)),
        "n_discrete_cols": int(len(discrete_cols)),
    }

    # Likelihood fitness (once) — align columns first to avoid "col not in index"
    real_lf, syn_lf = align_common(real_df, syn_df)
    try:
        results["likelihood_fitness"] = evaluate_likelihood_fitness(
            real_df=real_lf,
            syn_df=syn_lf,
            discrete_cols=discrete_cols,
        )
    except Exception as e:
        results["likelihood_fitness_error"] = str(e)

    # ML efficacy (per target column) — impute + align first (RF can't take NaNs)
    real_ml = simple_impute_like_ctgan(real_df, discrete_cols)
    syn_ml = simple_impute_like_ctgan(syn_df, discrete_cols)
    real_ml, syn_ml = align_common(real_ml, syn_ml)

    targets = pick_targets(
        real_ml,
        syn_ml,
        exclude_missing_flags=(not args.include_missing_flags),
    )

    if args.max_targets and args.max_targets > 0:
        targets = targets[: args.max_targets]

    ratios = []
    for target_col in targets:
        try:
            out = evaluate_ml_efficacy(
                real_df=real_ml,
                syn_df=syn_ml,
                target_col=target_col,
                discrete_cols=discrete_cols,
            )
            results["ml_efficacy"][target_col] = out
            if "efficacy_ratio" in out:
                ratios.append(out["efficacy_ratio"])
        except Exception as e:
            results["ml_efficacy"][target_col] = {"error": str(e)}

    results["ml_efficacy_summary"] = {
        "n_targets": int(len(targets)),
        "n_success": int(sum(1 for v in results["ml_efficacy"].values() if "error" not in v)),
        "mean_efficacy_ratio": float(np.mean(ratios)) if ratios else None,
        "median_efficacy_ratio": float(np.median(ratios)) if ratios else None,
    }

    out_path = run_dir / "metrics_all_targets.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[CTGAN eval] saved → {out_path}")


if __name__ == "__main__":
    main()
