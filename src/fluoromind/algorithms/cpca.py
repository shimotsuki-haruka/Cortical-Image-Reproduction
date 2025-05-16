"""
Complex Principal Component Analysis (CPCA) for Neuroimaging Data Analysis

This module implements Complex Principal Component Analysis, a variant of PCA
that can handle complex-valued data, particularly useful for neuroimaging analysis.
"""

from typing import Optional
import numpy as np
from scipy.signal import hilbert
from scipy.stats import zscore
from scipy.sparse.linalg import svds
from dataclasses import dataclass
from ..io import BaseResult, SaveableMixin


@dataclass
class CPCAResult(BaseResult):
    """Container for CPCA analysis results.

    Parameters
    ----------
    components : ndarray
        Principal components (eigenvectors)
    scores : ndarray
        Projection of the data onto the principal components
    singular_values : ndarray
        The singular values corresponding to each component
    explained_variance : ndarray
        The amount of variance explained by each component
    explained_variance_ratio : ndarray
        The percentage of variance explained by each component
    n_components : int
        Number of components used in the analysis
    force_complex : bool
        Whether complex conversion was forced
    standardize : bool
        Whether data was standardized
    """

    components: np.ndarray
    scores: np.ndarray
    singular_values: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    n_components: int
    force_complex: bool
    standardize: bool


class CPCA(SaveableMixin):
    """Complex Principal Component Analysis (CPCA).

    A dimensionality reduction technique that extends traditional PCA to complex-valued data.
    It can be used to analyze phase-amplitude relationships in time series data.

    Parameters
    ----------
    n_components : int, optional
        Number of components to keep. If None, keeps all components.
    force_complex : bool, default=True
        If True, converts real-valued input to complex using Hilbert transform.
    standardize : bool, default=True
        If True, standardizes the input data using z-score normalization.
    solver_args : dict, default=None
        Additional arguments to pass to scipy.sparse.linalg.svds.

    Attributes
    ----------
    components_ : ndarray
        Principal components (eigenvectors)
    singular_values_ : ndarray
        The singular values corresponding to each component
    explained_variance_ : ndarray
        The amount of variance explained by each component
    explained_variance_ratio_ : ndarray
        The percentage of variance explained by each component
    n_components_ : int
        The estimated number of components
    n_features_in_ : int
        Number of features seen during fit
    feature_names_in_ : ndarray
        Names of features seen during fit
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        force_complex: bool = True,
        standardize: bool = True,
        solver_args: Optional[dict] = None,
    ):
        """Initialize CPCA."""
        self.n_components = n_components
        self.force_complex = force_complex
        self.standardize = standardize
        self.solver_args = solver_args or {}

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
        return {
            "n_components": self.n_components,
            "force_complex": self.force_complex,
            "standardize": self.standardize,
            "solver_args": self.solver_args,
        }

    def set_params(self, **params) -> "CPCA":
        """Set the parameters of this estimator.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: np.ndarray) -> "CPCA":
        """Fit CPCA model to complex-valued data.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix of shape (n_samples, n_features)

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if not np.iscomplexobj(X) and self.force_complex:
            X = hilbert(X, axis=0)

        X_scaled = zscore(X) if self.standardize else X
        if self.n_components is None:
            self.n_components_ = n_features
        else:
            self.n_components_ = self.n_components

        U, S, V = svds(X_scaled, k=self.n_components_, **self.solver_args)

        # Store results in reverse order
        self.scores_ = U[:, ::-1]
        self.components_ = V[::-1]
        self.singular_values_ = S[::-1]
        self.explained_variance_ = (self.singular_values_**2) / (n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(np.var(X_scaled, ddof=1, axis=0))

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data into the complex principal component space.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Transformed data matrix of shape (n_samples, n_components)

        Raises
        ------
        ValueError
            If CPCA model has not been fitted yet
        """
        if not hasattr(self, "components_"):
            raise ValueError("CPCA model not fitted yet.")

        if not np.iscomplexobj(X) and self.force_complex:
            X = hilbert(X, axis=0)

        X_scaled = zscore(X) if self.standardize else X
        return X_scaled @ self.components_.conjugate().T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Transformed data matrix of shape (n_samples, n_components)
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Transform data back to original space.

        Parameters
        ----------
        X_transformed : np.ndarray
            Data matrix in component space of shape (n_samples, n_components)

        Returns
        -------
        np.ndarray
            Data matrix in original space of shape (n_samples, n_features)

        Raises
        ------
        ValueError
            If CPCA model has not been fitted yet
        """
        if not hasattr(self, "components_"):
            raise ValueError("CPCA model not fitted yet.")
        return X_transformed @ self.components_

    def spatiotemporal_patterns(self, i: int, start_idx: int = 0, end_idx: Optional[int] = None) -> np.ndarray:
        """Get the spatial patterns of the i-th component.

        Parameters
        ----------
        i : int
            Index of the component
        start_idx : int, default=0
            Start index for temporal pattern
        end_idx : int, optional
            End index for temporal pattern

        Returns
        -------
        np.ndarray
            Spatiotemporal patterns for the i-th component

        Raises
        ------
        ValueError
            If model is not fitted or if component index is invalid
        """
        if not hasattr(self, "components_"):
            raise ValueError("CPCA model not fitted yet.")

        if end_idx is None:
            end_idx = self.scores_.shape[0]

        if i >= self.n_components_:
            raise ValueError(f"Component index {i} is out of bounds for the number of components {self.n_components_}.")

        scores_i = self.scores_[start_idx:end_idx, i].reshape(-1, 1)
        components_i = self.components_[i].reshape(1, -1)
        return self.singular_values_[i] * (scores_i @ components_i)
