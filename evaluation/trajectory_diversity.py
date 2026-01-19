from __future__ import annotations
import numpy as np
from typing import Tuple
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

def validate_inputs(data: np.ndarray) -> np.ndarray:
    assert isinstance(data, np.ndarray)
    assert data.ndim in (2, 3)
    assert data.size > 0
    return data

def to_tslearn_format(data: np.ndarray) -> np.ndarray:
    """
    Converts data:
    (N, T) to (N, T, 1)
    (N, T, D) to (N, T, D)
    """
    if data.ndim == 2:
        return data[..., None]
    return data.astype(float)

def cluster_entropy(labels: np.ndarray) -> float:
    """Shannon entropy of cluster labels. Bigger means more diverse"""
    _, counts = np.unique(labels, return_counts = True)
    probs = counts/counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(entropy)

def trajectory_diversity(
        synth: np.ndarray,
        n_clusters: int = 10,
        max_iter: int = 20,
        random_state: int = 0
) -> dict:
    """
    Returns entropy of clusters formed by synthetic trajectories and 
    number of culsters used.
    """
    synth = validate_inputs(synth)
    X = to_tslearn_format(synth)
    X = TimeSeriesScalerMeanVariance().fit_transform(X)

    kmeans = TimeSeriesKMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        random_state=random_state,
        metric="dtw"
    )

    labels = kmeans.fit_predict(X)

    entropy = cluster_entropy(labels)
    used = len(np.unique(labels))

    return { float(entropy), int(used)}

"""
Low entropy indicates low diversity among trajectories(Mode collapse).
High entropy indicates high diversity among trajectories(Good).
"""