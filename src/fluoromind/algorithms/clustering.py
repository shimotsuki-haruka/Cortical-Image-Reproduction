"""Clustering analysis module for fluoromind.

This module provides utilities for performing clustering analysis on fluorescence data,
including methods for finding optimal cluster numbers and performing k-means clustering.
"""

import numpy as np
from typing import Tuple, List, Optional, Literal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings


def find_optimal_clusters(
    X: np.ndarray,
    k_range: tuple[int, int] = (2, 10),
    method: Literal["elbow", "silhouette"] = "elbow",
    random_state: Optional[int] = None,
) -> Tuple[int, List[float]]:
    """Find optimal number of clusters using either elbow method or silhouette analysis.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    k_range : tuple[int, int], default=(2, 10)
        Range of k values to test (min_k, max_k)
    method : {"elbow", "silhouette"}
        Method to use for determining optimal number of clusters
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    Tuple[int, List[float]]
        - Optimal number of clusters
        - List of scores for each k value tested

    Raises
    ------
    ValueError
        If method is not one of "elbow" or "silhouette"
    """
    scores = []
    k_values = range(k_range[0], k_range[1] + 1)

    # Vectorized scoring
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(X)

        if method == "elbow":
            scores.append(kmeans.inertia_)
        elif method == "silhouette":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores.append(silhouette_score(X, labels))
        else:
            raise ValueError('Method must be either "elbow" or "silhouette"')

    if method == "elbow":
        # Find elbow point using the kneedle algorithm
        scores_arr = np.array(scores)
        diffs = np.diff(scores_arr)
        optimal_k = k_values[np.argmax(np.abs(np.diff(diffs))) + 1]
    else:
        # For silhouette, higher score is better
        optimal_k = k_values[np.argmax(scores)]

    return optimal_k, scores


def perform_clustering(
    X: np.ndarray, n_clusters: int, random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform k-means clustering on input data.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix to cluster
    n_clusters : int
        Number of clusters
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Cluster labels and cluster centers
    """
    k_means = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = k_means.fit_predict(X)
    centers = k_means.cluster_centers_

    return labels, centers
