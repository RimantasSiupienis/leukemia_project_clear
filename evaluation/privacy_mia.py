from __future__ import annotations
from typing import Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

def validate_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert isinstance(a, np.ndarray) #a should be a numpy array
    assert isinstance(b, np.ndarray) #b should be a numpy array
    assert a.ndim in (2,3)
    assert b.ndim in (2,3)
    assert a.shape[1] == b.shape[1]
    if a.ndim == 3 or b.ndim == 3:
        assert a.ndim == b.ndim
        assert a.shape[2] == b.shape[2]
    return a, b

def flatten_trajectories(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        return data.astype(float)

    N, T, D = data.shape
    return data.reshape(N, T * D).astype(float)

    return data

def mia_distance_auc(
    train_real: np.ndarray,
    test_real: np.ndarray,
    synth: np.ndarray,
    k:int = 1
    ) -> dict:
    """
    Computes distance-based membership inference ROC area under curve.
    Input:
        train_real: np.ndarray of shape (N_tr, T) or (N_tr, T, D)
        test_real: np.ndarray of shape (N_te, T) or (N_te, T, D)
        synth: np.ndarray of shape (N_s, T) or (N_s, T, D)
        k: int, the k-th nearest neighbor to consider
    
    Output:
        dict{
            auc: float,
            mean_train_distance: float,
            mean_test_distance: float
        }
    """

    assert k > 0

    train_real, synth = validate_pair(train_real, synth)
    test_real, synth = validate_pair(test_real, synth)

    X_train = flatten_trajectories(train_real)
    X_test = flatten_trajectories(test_real)
    X_synth = flatten_trajectories(synth)

    assert X_synth.shape[0] >= k
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(X_synth)

    d_train, _ = nn.kneighbors(X_train)
    d_test, _ = nn.kneighbors(X_test)
    # Avg distance to k-nearest neighbors in synth data
    d_train_mean = d_train.mean(axis=1)
    d_test_mean = d_test.mean(axis=1)

    # Labelss: train_real = 1, test_real = 0
    y_true = np.concatenate([np.ones_like(d_train_mean), np.zeros_like(d_test_mean)])
    # Scores are negative distances (smaller distance means higher chance of being in training set)
    scores = -np.concatenate([d_train_mean, d_test_mean])
    
    auc = roc_auc_score(y_true, scores)
    return {
        float(auc),
        float(d_train_mean.mean()),
        float(d_test_mean.mean())
    }





