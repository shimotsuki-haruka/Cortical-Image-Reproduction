"""Correlation analysis module for fluoromind.

This module provides classes and utilities for computing various types of correlations
between fluorescence data, with support for parallel processing and statistical significance.
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple, Literal, Dict, Any
from functools import lru_cache
from .parallel import ParallelMixin


class BaseCorrelation(ParallelMixin):
    """Base class for correlation analyses.

    This class provides core functionality for computing various types of correlations
    between fluorescence data, with support for parallel processing and optional
    statistical significance calculations.

    Supported correlation methods:
    - Pearson correlation
    - Spearman correlation
    - Kendall correlation

    The class uses caching and parallel processing optimizations when appropriate.
    """

    def __init__(self, **kwargs):
        """Initialize correlation parameters."""
        super().__init__(**kwargs)
        self._cache: Dict[str, Any] = {}

    @staticmethod
    @lru_cache(maxsize=128)
    def _get_triu_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Cache and return upper triangle indices."""
        return np.triu_indices(n, k=1)

    def _parallel_correlations(
        self,
        data: np.ndarray,
        indices: Tuple[np.ndarray, np.ndarray],
        method: str,
        compute_significance: bool,
    ) -> list:
        """Compute correlations in parallel."""
        tasks = [(data[:, i], data[:, j], method, compute_significance) for i, j in zip(*indices, strict=True)]

        return self._parallel_map(self._compute_single_correlation, tasks)

    @staticmethod
    def _compute_single_correlation(
        x: np.ndarray,
        y: np.ndarray,
        method: str,
        compute_significance: bool,
    ) -> Tuple[float, Optional[float]]:
        """Compute correlation between two vectors."""
        if method == "pearson":
            result = stats.pearsonr(x, y)
        elif method == "spearman":
            result = stats.spearmanr(x, y)
        else:  # kendall
            result = stats.kendalltau(x, y)

        return result if compute_significance else (result[0], None)

    def _compute_correlation(
        self,
        data: np.ndarray,
        seed_indices: Optional[np.ndarray] = None,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        compute_significance: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute correlation matrix using specified method.

        Parameters
        ----------
        data : np.ndarray
            Input data matrix of shape (n_samples, n_features)
        seed_indices : np.ndarray, optional
            Indices of seed points for seed-based correlation
        method : {"pearson", "spearman", "kendall"}
            Correlation method to use
        compute_significance : bool, default=False
            Whether to compute statistical significance (p-values)

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            - Correlation matrix of shape (n_features, n_features)
            - P-values matrix of same shape if compute_significance=True, else None

        Raises
        ------
        ValueError
            If method is not one of "pearson", "spearman", or "kendall"
        """
        if method not in ["pearson", "spearman", "kendall"]:
            raise ValueError(f"Unknown correlation method: {method}")

        if seed_indices is not None:
            data = data[:, seed_indices]

        n_features = data.shape[1]

        # Use optimized implementations for full matrix when available
        if not compute_significance:
            if method == "pearson":
                corr_matrix = np.corrcoef(data.T)
                return corr_matrix, None
            elif method == "spearman":
                corr_matrix = stats.spearmanr(data)[0]
                return corr_matrix, None

        # Initialize matrices
        corr_matrix = np.eye(n_features, dtype=np.float64)
        p_matrix = np.zeros_like(corr_matrix) if compute_significance else None

        # Get cached indices
        triu_indices = self._get_triu_indices(n_features)

        # Use the mixin's helper method
        use_parallel = self.should_use_parallel(n_features)

        if use_parallel:
            results = self._parallel_correlations(data, triu_indices, method, compute_significance)

            # Unpack results
            for idx, (corr, p_val) in enumerate(results):
                i, j = triu_indices[0][idx], triu_indices[1][idx]
                corr_matrix[i, j] = corr
                if compute_significance:
                    p_matrix[i, j] = p_val
        else:
            # Sequential processing
            for i, j in zip(*triu_indices, strict=True):
                corr, p_val = self._compute_single_correlation(data[:, i], data[:, j], method, compute_significance)
                corr_matrix[i, j] = corr
                if compute_significance:
                    p_matrix[i, j] = p_val

        # Mirror upper triangle efficiently
        corr_matrix = corr_matrix + corr_matrix.T - np.diag(np.diag(corr_matrix))
        if compute_significance:
            p_matrix = p_matrix + p_matrix.T

        return corr_matrix, p_matrix
