"""
Module:
    - reduces multivariate data to 1D sequences
    - implements DTW distance between two datasets
    - computes average nearest neighbor DTW distance (i.e. for each real sample, find
    the closest synthetic sample and average these minimal distances)

Assumptions:  
    - real and synthetic data are numpy arrays with shape:
        - (N, T) for univariate time series
        - (N, T, D) for multivariate time series
    - for multivariate data, DTW is computed a 1D reduction per time step

Functions:
    - average_nearest_dtw(real, synth) -> float
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


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


def reduce_to_1d(data: np.ndarray) -> np.ndarray:
    """
    Converts series to 1D per trajectory by averaging across feature dimensions.
    Univariate: (N, T) -> (N, T)
    Multivariate: (N, T, D) -> (N, T) by mean across D(features)    
    """

    if data.ndim == 2:
        return data.astype(float)
    
    return float(data.mean(axis=-1))

def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Compute dinamic time warping distance between two 1D sequences.
    seq1, seq2: 1D numpy arrays of shape (T,)
    """
    assert seq1.ndim == 1 and seq2.ndim == 1
    n = len(seq1)
    m = len(seq2)
    dp = np.full((n + 1, m + 1), np.inf) # Initializing distance matrix with infinity
    dp[0, 0] = 0.0 # Starting point

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(seq1[i-1] - seq2[j-2])
            dp[i, j] = cost + min(dp[i-1, j], dp[i-1, j-1], dp[i, j-1])
        
    return float(dp[n,m])

def average_nearest_dtw(real: np.ndarray, synth: np.ndarray) -> float:
    """
    Computes average nearest neighbour DTW distance from real data to synthetic data.

    Inputs:
        - real: numpy array of shape (N_real, T) or (N_real, T, D)
        - synth: numpy array of shape (N_synth, T) or (N_synth, T, D)

    Returns:
        -average of real trajectories of the minimal DTW distance to any synthetic trajectory
    """

    real, synth = validate_inputs(real, synth)
    real_1d = reduce_to_1d(real)
    synth_1d = reduce_to_1d(synth)

    n_r = real_1d.shape[0]
    n_s = synth_1d.shape[0]
    assert n_r > 0
    
    min_distances = []

    for i in range(n_r):
        r = real_1d[i]
        min_dist = float('inf')

        for j in range(n_s):
            d = dtw_distance(r, synth_1d[j])
            if d < min_dist:
                min_dist = d

        min_distances.append(min_dist)

    return float(np.mean(min_distances))