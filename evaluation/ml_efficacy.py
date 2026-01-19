from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def _fit_preprocessors(
    X_train: pd.DataFrame,
    discrete_cols: List[str],
) -> Tuple[Optional[OrdinalEncoder], Optional[StandardScaler], List[str], List[str]]:
    disc = [c for c in discrete_cols if c in X_train.columns]
    cont = [c for c in X_train.columns if c not in disc]

    enc = None
    if disc:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(X_train[disc].astype("object"))

    scaler = None
    if cont:
        scaler = StandardScaler()
        scaler.fit(X_train[cont].astype(float))

    return enc, scaler, disc, cont


def _transform(
    X: pd.DataFrame,
    enc: Optional[OrdinalEncoder],
    scaler: Optional[StandardScaler],
    disc: List[str],
    cont: List[str],
) -> np.ndarray:
    parts = []
    if disc:
        if enc is None:
            raise RuntimeError("Encoder missing.")
        parts.append(enc.transform(X[disc].astype("object")))
    if cont:
        if scaler is None:
            raise RuntimeError("Scaler missing.")
        parts.append(scaler.transform(X[cont].astype(float)))
    if not parts:
        raise ValueError("No features after preprocessing.")
    return np.concatenate(parts, axis=1)


def evaluate_ml_efficacy(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    target_col: str,
    discrete_cols: List[str],
    seed: int = 42,
    test_size: float = 0.2,
) -> Dict[str, float]:
    """
    Train a predictor on:
      (a) real train -> evaluate on real test
      (b) synthetic -> evaluate on real test
    Return both and the ratio.

    If target is non-numeric or has small cardinality, treat as classification.
    Otherwise treat as regression.
    """
    if target_col not in real_df.columns:
        raise ValueError(f"target_col='{target_col}' not in real_df")
    if target_col not in syn_df.columns:
        raise ValueError(f"target_col='{target_col}' not in syn_df")

    # align columns intersection (defensive)
    common_cols = [c for c in real_df.columns if c in syn_df.columns]
    real_df = real_df[common_cols].copy()
    syn_df = syn_df[common_cols].copy()

    X_real = real_df.drop(columns=[target_col])
    y_real = real_df[target_col]
    X_syn = syn_df.drop(columns=[target_col])
    y_syn = syn_df[target_col]

    # patient-level splitting should happen outside; here it’s row-level
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_real, y_real, test_size=test_size, random_state=seed
    )

    # preprocessors fit ONLY on real train
    enc, scaler, disc, cont = _fit_preprocessors(Xr_train, discrete_cols)

    Xr_train_t = _transform(Xr_train, enc, scaler, disc, cont)
    Xr_test_t = _transform(Xr_test, enc, scaler, disc, cont)
    Xs_t = _transform(X_syn, enc, scaler, disc, cont)

    # decide task type
    is_classification = (
        (not pd.api.types.is_numeric_dtype(y_real))
        or (y_real.nunique(dropna=True) <= 20)
    )

    if is_classification:
        # encode labels
        # map using real-train label set; unknown synth labels become -1
        labels = pd.Index(yr_train.dropna().unique())
        label_to_int = {lab: i for i, lab in enumerate(labels)}
        yr_train_i = yr_train.map(label_to_int).fillna(-1).astype(int).values
        yr_test_i = yr_test.map(label_to_int).fillna(-1).astype(int).values
        y_syn_i = y_syn.map(label_to_int).fillna(-1).astype(int).values

        clf_real = RandomForestClassifier(random_state=seed, n_estimators=200)
        clf_syn = RandomForestClassifier(random_state=seed, n_estimators=200)

        clf_real.fit(Xr_train_t, yr_train_i)
        pred_real = clf_real.predict(Xr_test_t)

        # synthetic-trained
        # filter out unknown labels in synthetic to avoid training garbage
        keep = y_syn_i != -1
        clf_syn.fit(Xs_t[keep], y_syn_i[keep])
        pred_syn = clf_syn.predict(Xr_test_t)

        f1_real = f1_score(yr_test_i, pred_real, average="macro")
        f1_syn = f1_score(yr_test_i, pred_syn, average="macro")
        ratio = (f1_syn / f1_real) if f1_real != 0 else 0.0

        return {
            "task": "classification",
            "f1_real": float(f1_real),
            "f1_syn": float(f1_syn),
            "efficacy_ratio": float(ratio),
        }

    # regression
    yr_train_v = yr_train.astype(float).values
    yr_test_v = yr_test.astype(float).values
    y_syn_v = y_syn.astype(float).values

    reg_real = RandomForestRegressor(random_state=seed, n_estimators=300)
    reg_syn = RandomForestRegressor(random_state=seed, n_estimators=300)

    reg_real.fit(Xr_train_t, yr_train_v)
    pred_real = reg_real.predict(Xr_test_t)

    reg_syn.fit(Xs_t, y_syn_v)
    pred_syn = reg_syn.predict(Xr_test_t)

    r2_real = r2_score(yr_test_v, pred_real)
    r2_syn = r2_score(yr_test_v, pred_syn)
    ratio = (r2_syn / r2_real) if r2_real != 0 else 0.0

    return {
        "task": "regression",
        "r2_real": float(r2_real),
        "r2_syn": float(r2_syn),
        "efficacy_ratio": float(ratio),
    }
