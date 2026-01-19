"""
Module:
    - validates inputs
    - flattens multivariate data to 1D sequences
    - clones estimator to avoid data leakage
    - trains on synthetic data 
    - evaluates on real data via:
        - classification accuracy (default)
        - regression mean squared error (default for float targets)
        - a custom metric function provided by the user

Assumptions:
    - X_synth and X_real are numpy arrays with shape:
        - (N, T) for univariate time series
        - (N, T, D) for multivariate time series
    - y_synth and y_real are 1D numpy arrays with length N
    - an estimator compatible with sklearn's fit/predict interface is provided

Functions:
    train_on_synth_test_on_real(
        X_synth, y_synth, X_real, y_real,
        model,
        task = None
        metric = None
        featurizer = None
    ) -> float

"""


from __future__ import annotations
from typing import Tuple
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, mean_squared_error

TaskType = ["classification", "regression"]

def validate_inputs(
        X_synth: np.ndarray,
        y_synth: np.ndarray,
        X_real: np.ndarray,
        y_real: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validates shapes, dtyps and consistency of input data."""
    for name, arr in (("X_synth", X_synth),("X_real", X_real)):
        assert isinstance(arr, np.ndarray) # arr should be a numpy array
        assert arr.ndim in (2, 3) # arr should be 2D or 3D
        assert np.isfinite(arr).all() # arr should not contain nans or infs

        assert X_synth.shape[0] == y_synth.shape[0] # consistent number of samples
        assert X_real.shape[0] == y_real.shape[0] # consistent number of samples
        assert X_synth.shape[1] == X_real.shape[1] # consistent time steps
        if X_synth.ndim == 3 or X_real.ndim == 3:
            assert X_synth.ndim == X_real.ndim # both must be univariate or multivariate
            assert X_synth.shape[2] == X_real.shape[2] # consistent feature dimensions D

        return X_synth, y_synth, X_real, y_real
    
def reduce_to_1d(data: np.ndarray) -> np.ndarray:
    """
    Default: flatten (N, T, D) or (N, T) to (N, T * D)
    This treats each trajectory as a singe feature vector.
    """
    if data.ndim == 2:
        return data.astype(float)
    
    N, T, D = data.shape
    return data.reshape(N, T * D).astype(float)


def decide_task(y: np.ndarray) -> None: #FINISH LATER
    return None
