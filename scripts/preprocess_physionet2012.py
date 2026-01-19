#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# Static vars are stored as normal rows at time 00:00
STATIC_PARAMS = ["Age", "Gender", "Height", "ICUType", "Weight"]

# Targets chosen from the real PhysioNet 2012 labels (see sample file)
# BP in this dataset is usually NISysABP / NIDiasABP / NIMAP (non-invasive)
DEFAULT_TARGETS = [
    "HR",
    "NISysABP",
    "NIDiasABP",
    "NIMAP",
    "RespRate",
    "Temp",
    "SpO2",
]


def parse_time_to_hour(t: str) -> int:
    # Time format: "HH:MM"
    hh, mm = t.split(":")
    return int((int(hh) * 60 + int(mm)) // 60)


def read_patient_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if not {"Time", "Parameter", "Value"}.issubset(df.columns):
        raise ValueError(f"Bad format in {path.name}: {df.columns.tolist()}")

    df["Time"] = df["Time"].astype(str)
    df["Parameter"] = df["Parameter"].astype(str)

    # values are mostly numeric; non-numeric becomes NaN
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    # a few fields use -1 for missing (e.g. Height, Weight in your sample)
    df.loc[df["Value"] == -1, "Value"] = np.nan

    df["t"] = df["Time"].apply(parse_time_to_hour).astype(int)
    return df[["t", "Parameter", "Value"]]


def extract_static(long_df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in STATIC_PARAMS:
        sub = long_df.loc[long_df["Parameter"] == p, "Value"].dropna()
        if len(sub) > 0:
            out[p] = float(sub.iloc[-1])
    return out


def pivot_targets(long_df: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
    d = long_df[long_df["Parameter"].isin(targets)].copy()
    if d.empty:
        return pd.DataFrame({"t": np.arange(0, 48, dtype=int)})

    # if there are multiple readings in the same hour, keep the last one
    d = d.sort_values(["t"])
    wide = (
        d.pivot_table(index="t", columns="Parameter", values="Value", aggfunc="last")
        .reset_index()
    )

    # make sure all target cols exist
    for c in targets:
        if c not in wide.columns:
            wide[c] = np.nan

    wide = wide[["t"] + targets]
    return wide


def build_patient_frame(patient_id: str, file_path: Path, targets: List[str]) -> pd.DataFrame:
    long_df = read_patient_file(file_path)
    static = extract_static(long_df)
    wide = pivot_targets(long_df, targets)

    # fixed grid 0..47 hours so downstream windowing is easy
    full = pd.DataFrame({"t": np.arange(0, 48, dtype=int)})
    wide = full.merge(wide, on="t", how="left")

    # attach static as constant columns
    for k in STATIC_PARAMS:
        wide[k] = static.get(k, np.nan)

    wide.insert(0, "patient_id", str(patient_id))
    return wide


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--targets", type=str, default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    set_a = raw_root / "set-a"
    set_b = raw_root / "set-b"
    if not set_a.is_dir() or not set_b.is_dir():
        raise FileNotFoundError(f"Expected set-a/ and set-b/ under {raw_root}")

    out_path = Path(args.out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if not targets:
        raise ValueError("No targets provided")

    files_a = sorted(set_a.glob("*.txt"))
    files_b = sorted(set_b.glob("*.txt"))

    if args.limit and args.limit > 0:
        files_a = files_a[: max(0, args.limit // 2)]
        files_b = files_b[: max(0, args.limit - len(files_a))]

    frames: List[pd.DataFrame] = []

    # prefix ids so set-a and set-b never collide
    for fp in files_a:
        frames.append(build_patient_frame(f"A_{fp.stem}", fp, targets))

    for fp in files_b:
        frames.append(build_patient_frame(f"B_{fp.stem}", fp, targets))

    df = pd.concat(frames, axis=0, ignore_index=True)
    df["patient_id"] = df["patient_id"].astype(str)
    df["t"] = df["t"].astype(int)
    df = df.sort_values(["patient_id", "t"]).reset_index(drop=True)

    df.to_parquet(out_path, index=False)

    print("wrote:", out_path)
    print("shape:", df.shape)
    print("targets:", targets)
    print("static:", STATIC_PARAMS)


if __name__ == "__main__":
    main()
