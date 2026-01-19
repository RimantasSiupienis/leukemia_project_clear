"""
TD preservation using lagged autocorrelation
"""
from __future__ import annotations
import numpy as np
from typing import Tuple

def validate_pair(real: np.ndarray, synth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert real.ndim in (2, 3)
    assert synth.ndim in (2, 3)
    assert real.shape[1] == synth.shape[1]
    if real.ndim == 3:
        assert real.shape == synth.shape
    return real, synth

def lagged_autocorrelation(x: np.ndarray, lag: int) -> float:
    """
    lag-k autocorrelation of a 1D array
    """
    x1 = x[:-lag]
    x2 = x[lag:]
    if x1.std() == 0 or x2.std() == 0:
        return 0.0
    return float(np.corrcoef(x1, x2)[0, 1])

def mean_lagged_autocorrelation(data:np.ndarray, lag: int) -> np.ndarray:
    """
    mean lag-k autocorrelation over all features and samples
    """
    if data.ndim == 2:
        data = data[..., None]
    
    N, T, D = data.shape
    results = []

    for d in range(D):
        vals = []
        for n in range(N):
            vals.append(lagged_autocorrelation(data[n, :, d], lag))
        results.append(np.mean(vals))
    
    return np.array(results)

def temporal_dependency(real: np.ndarray, synth: np.ndarray, lag: int = 1) -> dict:
    """
    Compute mean absolute error between lagged autocorrelations of real and synthetic data.
    """
    real, synth = validate_pair(real, synth)

    real_ac = mean_lagged_autocorrelation(real, lag)
    synth_ac = mean_lagged_autocorrelation(synth, lag)

    error = np.mean(np.abs(real_ac - synth_ac))

    return {
        float(error),
        real_ac,
        synth_ac
    }

"""
Low error means synth trajectories preserve temporal dependencies of real ones."""