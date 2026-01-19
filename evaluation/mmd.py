# Maximum Mean Dicrepancy (MMD) implementation for evaluating generative models.
from __future__ import annotations
import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Tuple


"""
Module:
    -validates inputs
    -flattens time series data into feature vectors of shape (N, T, D)
      N = number of trajectories, T = trajectory length, D = feature dimension
    - uses RBF kernel to compute MMD between two datasets
    - Implements an unbiased estimator for MMD^2

Assumptions:
    - Input data is in the form of numpy arrays
    - (N, T) for univariate time series
    - (N, T, D) for multivariate time series
    - All trajectories have the same length T

Functions:
    - compute_mmd(real_data, generated_data) -> float
"""

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


def flatten_data(data: np.ndarray) -> np.ndarray:
    """
    Flattens time series data into feature vectors.
    Univariate: (N, T) -> (N, T)
    Multivariate: (N, T, D) -> (N, T*D)
    """
    if data.ndim == 2:
        return data.astype(float)
    
    n, t, d = data.shape
    return data.reshape(n, t * d).astype(float)


def median_heuristic(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes RBF kernel bandwidth(gamma) using squared distances
    """
    combined =  np.vstack([x, y])
    max_samples = 1000
    if combined.shape[0] > max_samples:
        idx = np.random.choice(combined.shape[0], max_samples, replace=False)
        combined = combined[idx]
    
    dist_sq = pairwise_distances(combined, combined, metric="euclidian", squared = "True")

    # Excludding zero distances (the diagonal line)
    i_upper = np.triu_indices_from(dist_sq, k = 1)
    values = dist_sq[i_upper]
    values = values[values > 0]

    if values.size == 0:
        return 1.0
    
    median_sq = np.median(values)

    if median_sq <= 0:
        return 1.0
    
    # RBF kernel bandwidth parameter
    gamma = 1 / (2 * median_sq)

    return gamma


def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    """
    Computes RBF kernel matrix between x and y
    """
    dist_sq = pairwise_distances(x, y, metric="euclidean", sqared=True)
    kernel_matrix = np.exp(-gamma * dist_sq)

    return kernel_matrix


def compute_mmd(real: np.ndarray, synth: np.ndarray) -> float:
    """
    Returns unbiased MMD^2 between real and synthetic datasets
    """
    real, synth = validate_inputs(real, synth)
    x = flatten_data(real)
    y = flatten_data(synth)

    n = x.shape[0]
    m = y.shape[0]

    gamma = median_heuristic(x, y)

    k_xx = rbf_kernel(x, x, gamma)
    k_yy = rbf_kernel(y, y, gamma)
    k_xy = rbf_kernel(x, y, gamma)

    sum_xx = (np.sum(k_xx) -np.trace(k_xx)) / (n * (n - 1))
    sum_yy = (np.sum(k_yy) -np.trace(k_yy)) / (m * (m - 1))
    sum_xy = np.mean(k_xy)

    mmd2 = sum_xx + sum_yy - 2 * sum_xy
    
    return float(mmd2)



