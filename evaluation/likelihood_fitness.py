from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


@dataclass
class LikelihoodFitnessResult:
    ll_real_test_under_S: float
    ll_syn_under_S: float
    ll_real_test_under_Sprime: float
    ll_syn_under_Sprime: float


class LikelihoodFitness:
    """
    Likelihood-based fitness via two Gaussian Mixture oracles:

      - Oracle S: fitted on encoded/scaled REAL TRAIN data.
      - Oracle S': fitted on encoded/scaled SYN data using the SAME real-train encoders/scalers.

    Returns average log-likelihoods per row.
    """

    def __init__(self, n_components: int = 5, seed: int = 42):
        self.n_components = n_components
        self.seed = seed

        self._enc: Optional[OrdinalEncoder] = None
        self._scaler: Optional[StandardScaler] = None

        self.oracle_S: Optional[GaussianMixture] = None
        self.oracle_Sprime: Optional[GaussianMixture] = None

        self._discrete_cols: List[str] = []
        self._continuous_cols: List[str] = []

    def _split_cols(self, df: pd.DataFrame, discrete_cols: List[str]) -> Tuple[List[str], List[str]]:
        disc = [c for c in discrete_cols if c in df.columns]
        cont = [c for c in df.columns if c not in disc]
        return disc, cont

    def fit_real_encoders(self, X_real_train: pd.DataFrame, discrete_cols: List[str]) -> None:
        disc, cont = self._split_cols(X_real_train, discrete_cols)
        self._discrete_cols = disc
        self._continuous_cols = cont

        # OrdinalEncoder handles unknown categories from test/synth safely
        if disc:
            self._enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            self._enc.fit(X_real_train[disc].astype("object"))
        else:
            self._enc = None

        if cont:
            self._scaler = StandardScaler()
            self._scaler.fit(X_real_train[cont].astype(float))
        else:
            self._scaler = None

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        parts = []

        if self._discrete_cols:
            if self._enc is None:
                raise RuntimeError("Encoder not fit.")
            d = self._enc.transform(X[self._discrete_cols].astype("object"))
            parts.append(d)

        if self._continuous_cols:
            if self._scaler is None:
                raise RuntimeError("Scaler not fit.")
            c = self._scaler.transform(X[self._continuous_cols].astype(float))
            parts.append(c)

        if not parts:
            raise ValueError("No columns to evaluate.")
        return np.concatenate(parts, axis=1)

    def fit_oracles(
        self,
        X_real: pd.DataFrame,
        X_syn: pd.DataFrame,
        discrete_cols: List[str],
        test_size: float = 0.2,
    ) -> LikelihoodFitnessResult:
        Xr_train, Xr_test = train_test_split(X_real, test_size=test_size, random_state=self.seed)

        self.fit_real_encoders(Xr_train, discrete_cols)

        Xr_train_t = self.transform(Xr_train)
        Xr_test_t = self.transform(Xr_test)
        Xs_t = self.transform(X_syn)

        # Oracle S on real train
        self.oracle_S = GaussianMixture(
            n_components=self.n_components,
            random_state=self.seed,
        ).fit(Xr_train_t)

        # Oracle S' on synthetic (but in same feature space)
        self.oracle_Sprime = GaussianMixture(
            n_components=self.n_components,
            random_state=self.seed,
        ).fit(Xs_t)

        ll_real_test_under_S = float(np.mean(self.oracle_S.score_samples(Xr_test_t)))
        ll_syn_under_S = float(np.mean(self.oracle_S.score_samples(Xs_t)))

        ll_real_test_under_Sprime = float(np.mean(self.oracle_Sprime.score_samples(Xr_test_t)))
        ll_syn_under_Sprime = float(np.mean(self.oracle_Sprime.score_samples(Xs_t)))

        return LikelihoodFitnessResult(
            ll_real_test_under_S=ll_real_test_under_S,
            ll_syn_under_S=ll_syn_under_S,
            ll_real_test_under_Sprime=ll_real_test_under_Sprime,
            ll_syn_under_Sprime=ll_syn_under_Sprime,
        )


def evaluate_likelihood_fitness(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    discrete_cols: List[str],
    n_components: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    lf = LikelihoodFitness(n_components=n_components, seed=seed)
    res = lf.fit_oracles(real_df, syn_df, discrete_cols=discrete_cols)

    # Provide simple, interpretable ratios (higher ~ better match)
    # ratio_S compares how likely synthetic is under oracle trained on real.
    ratio_S = np.exp(res.ll_syn_under_S - res.ll_real_test_under_S)

    # ratio_Sprime compares how likely real test is under oracle trained on synthetic.
    ratio_Sprime = np.exp(res.ll_real_test_under_Sprime - res.ll_syn_under_Sprime)

    return {
        "ll_real_test_under_S": res.ll_real_test_under_S,
        "ll_syn_under_S": res.ll_syn_under_S,
        "ll_real_test_under_Sprime": res.ll_real_test_under_Sprime,
        "ll_syn_under_Sprime": res.ll_syn_under_Sprime,
        "likelihood_ratio_S": float(ratio_S),
        "likelihood_ratio_Sprime": float(ratio_Sprime),
    }
