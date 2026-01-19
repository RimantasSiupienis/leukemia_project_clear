"""
Privacy metric: k-Nearest Neighbors Distance
Assumptions:
    - real and synthetic datasets are numpy arrays of shape:
        -- (N, T) for univariate time series
        -- (N, T, D) for multivariate time series
    - for each synthetic trajectory, the distance to its k-th nearest neighbour among 
        real trajectories is computed after flattening

Functions:
    nearest_neighbors_distance(real, synthetic, k=1) -> dict
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from typing import Tuple
from __future__ import annotations

def validate_inputs(real: np.ndarray, synth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert isinstance(real, np.ndarray), "Should be a numpy array" 
    assert isinstance(synth, np.ndarray), "Should be a numpy array"
    assert real.ndim in {2, 3}, "Input data must be 2D or 3D numpy arrays."
    assert synth.ndim in {2, 3}, "Input data must be 2D or 3D numpy arrays."
    assert real.shape[1] == synth.shape[1], "Both datasets must have the same trajectory length."
    if real.ndim == 3 or synth.ndim == 3:
        assert real.ndim == synth.ndim, "Both datasets must have the same number of dimensions."
        assert real.shape[2] == synth.shape[2], "Both datasets must have the same feature dimension."
    assert real.size > 0 and synth.size > 0, "Input datasets must not be empty."
    return real, synth 

def flatten_trajectories(data: np.ndarray) -> np.ndarray:
    if data.ndim == 3:
        N, T, D = data.shape
        return data.reshape(N, T * D)
    return data

def nearest_neighbors_distance(real: np.ndarray, synth: np.ndarray, k: int = 1) -> dict:
    """
    Input:
        real: np.ndarray of shape (N_r, T) or (N_r, T, D)
        synth: np.ndarray of shape (N_s, T) or (N_s, T, D)
        k: int, the k-th nearest neighbor to consider

    Output:
        dict{
            mean_distance: float,
            std_distance: float,
            min_distance: float,
            max_distance: float
        }
    """

    assert isinstance(k, int) and k > 0

    real, synth = validate_inputs(real, synth)
    X_real = flatten_trajectories(real)
    X_synth = flatten_trajectories(synth)

    n_real = X_real.shape[0]
    assert k < n_real

    nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_real)
    distances, _ = nn.kneighbors(X_synth)

    per_sample = distances.mean(axis = 1)

    return {
        float(per_sample.mean()),
        float(per_sample.std()),
        float(per_sample.min()),
        float(per_sample.max())
    }