"""
Co-activation Patterns (CAPs) analysis module.

This module implements CAPs analysis to identify recurring spatial patterns
in time series data using k-means clustering. It provides functionality for:
- Pattern identification through k-means clustering
- Automatic optimal cluster detection
- Result persistence through save/load operations
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.metrics import silhouette_score
from .clustering import find_optimal_clusters, perform_clustering
from ..io import BaseResult, SaveableMixin
from typing import Optional


@dataclass
class CAPsResult(BaseResult):
    """
    Container for CAPs analysis results.

    Attributes
    ----------
    patterns : np.ndarray
        Identified co-activation patterns, shape (n_patterns, n_features)
    labels : np.ndarray
        Cluster assignments for each timepoint, shape (n_timepoints,)
    explained_var : float
        Proportion of variance explained by the clustering
    centroids : np.ndarray
        Cluster centroids, shape (n_patterns, n_features)
    silhouette_score : float, optional
        Clustering quality metric (-1 to 1, higher is better)
    """

    patterns: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    explained_var: Optional[float] = None
    centroids: Optional[np.ndarray] = None
    silhouette_score: Optional[float] = None
    explained_variance_ratio: Optional[float] = None


@dataclass
class PreparedData:
    """
    Container for preprocessed CAPs data.

    Attributes
    ----------
    X_norm : np.ndarray
        Normalized input data
    active_frames : np.ndarray
        Indices of frames exceeding threshold
    X_active : np.ndarray
        Normalized data for active frames only
    """

    X_norm: np.ndarray
    active_frames: np.ndarray
    X_active: np.ndarray


class CAPs(SaveableMixin):
    """
    Co-activation Patterns (CAPs) analysis.

    Identifies recurring spatial patterns in time series data using k-means clustering
    and threshold-based frame selection.

    Parameters
    ----------
    n_patterns : int
        Number of co-activation patterns to identify
    threshold : float, optional
        Activation threshold in standard deviations, by default 1.0
    random_state : int, optional
        Random seed for reproducibility
    """

    def __init__(self, n_patterns: Optional[int] = None, threshold: float = 1.0, random_state: Optional[int] = None):
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.n_patterns = n_patterns
        self.threshold = threshold
        self.random_state = random_state
        self.prepared_data_: Optional[PreparedData] = None
        self.result_ = None

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {"n_patterns": self.n_patterns, "threshold": self.threshold, "random_state": self.random_state}

    def set_params(self, **params) -> "CAPs":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : CAPs
            Estimator instance.
        """
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"Invalid parameter {key} for estimator {self.__class__.__name__}")
            setattr(self, key, value)
        return self

    def _prepare_data(self, X: np.ndarray) -> None:
        """
        Normalize data and identify active frames.

        Parameters
        ----------
        X : np.ndarray
            Input time series data
        """
        X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
        active_frames = np.any(X_norm > self.threshold, axis=1).nonzero()[0]
        X_active = X_norm[active_frames]

        self.prepared_data_ = PreparedData(X_norm=X_norm, active_frames=active_frames, X_active=X_active)

    def fit(self, X: np.ndarray) -> CAPs:
        """
        Perform CAPs analysis on input data.

        Parameters
        ----------
        X : np.ndarray
            Input time series data of shape (n_timepoints, n_features)

        Returns
        -------
        CAPs
            The fitted CAPs instance

        Raises
        ------
        ValueError
            If input data is empty or contains invalid values
        """
        if not isinstance(X, np.ndarray) or X.size == 0:
            raise ValueError("Input data must be a non-empty numpy array")

        self._prepare_data(X)

        if self.n_patterns is None:
            self.n_patterns, _ = find_optimal_clusters(
                self.prepared_data_.X_active, k_range=(2, 20), method="elbow", random_state=self.random_state
            )

        labels_active, centroids = perform_clustering(self.prepared_data_.X_active, self.n_patterns, self.random_state)

        # Create full labels array more efficiently
        labels = np.zeros(len(X), dtype=int)
        labels[self.prepared_data_.active_frames] = labels_active

        # Vectorized explained variance calculation
        total_var = np.sum((self.prepared_data_.X_active - self.prepared_data_.X_active.mean(axis=0)) ** 2)
        within_var = sum(
            np.sum((self.prepared_data_.X_active[labels_active == k] - centroids[k]) ** 2)
            for k in range(self.n_patterns)
        )
        explained_var = 1 - (within_var / total_var)

        # Vectorized pattern creation
        patterns = np.array(
            [
                (
                    self.prepared_data_.X_active[labels_active == k].mean(axis=0)
                    if np.any(labels_active == k)
                    else np.zeros_like(centroids[0])
                )
                for k in range(self.n_patterns)
            ]
        )

        # Calculate silhouette score for quality assessment
        sil_score = (
            silhouette_score(self.prepared_data_.X_active, labels_active) if len(np.unique(labels_active)) > 1 else 0
        )

        self.result_ = CAPsResult(
            patterns=patterns,
            labels=labels,
            explained_var=explained_var,
            explained_variance_ratio=explained_var / self.n_patterns,
            centroids=centroids,
            silhouette_score=sil_score,
        )

        return self
