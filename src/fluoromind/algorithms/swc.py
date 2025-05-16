"""Module for Sliding Window Correlation (SWC) analysis."""

import numpy as np
from typing import Optional, Tuple, Union, List, Literal
from dataclasses import dataclass
from .clustering import find_optimal_clusters, perform_clustering
from .correlation import BaseCorrelation
from ..io import BaseResult, SaveableMixin
import weakref


@dataclass
class SWCResult(BaseResult):
    """Container for SWC analysis results.

    Attributes
    ----------
    correlations : ndarray
        Correlation matrices for each window. Shape depends on analysis mode:
        - Full matrix: (n_windows, n_features, n_features)
        - Seed-based: (n_windows, n_seeds, n_features)
    window_centers : ndarray
        Time points corresponding to window centers
    significance : ndarray, optional
        Statistical significance (p-values) for correlations
    clusters : ndarray or None
        Cluster labels for each window
    cluster_centers : ndarray or None
        Cluster centroids for each window
    seed_indices : ndarray or None
        Indices of seed points used in analysis
    """

    correlations: np.ndarray
    window_centers: np.ndarray
    significance: Optional[np.ndarray] = None
    n_clusters: Optional[int] = None
    clusters: Optional[np.ndarray] = None
    cluster_centers: Optional[np.ndarray] = None
    seed_indices: Optional[np.ndarray] = None


class SWC(BaseCorrelation, SaveableMixin):
    """Sliding Window Correlation analysis for time series data.

    Parameters
    ----------
    window_size : int
        Size of the sliding window in samples
    stride : int, default=1
        Number of samples to move the window in each step
    method : str, default='pearson'
        Correlation method to use. Options:
        - 'pearson': Pearson correlation coefficient
        - 'spearman': Spearman rank correlation
    compute_significance : bool, default=False
        Whether to compute statistical significance of correlations
    seed_indices : array-like or None, default=None
        Indices of seed points for seed-based correlation analysis.
        If None, computes full correlation matrix.
    parallel : bool, optional
        Whether to use parallel computing, by default True
    n_jobs : int, optional
        Number of parallel jobs. If None, uses all available CPU cores
    parallel_backend : {"processes", "threads"}, optional
        Parallelization method to use, by default "processes"
        - "processes": Process-based parallelization (better for CPU-bound tasks)
        - "threads": Thread-based parallelization (better for I/O-bound tasks)
    chunk_size : int, optional
        Size of chunks for parallel processing. If None, auto-determined
    parallel_threshold : int, optional
        Minimum matrix size to trigger parallel computing, by default 1000
    """

    def __init__(
        self,
        window_size: int,
        stride: int = 1,
        method: str = "pearson",
        compute_significance: bool = False,
        seed_indices: Optional[Union[List[int], np.ndarray]] = None,
        parallel: bool = True,
        n_jobs: Optional[int] = None,
        parallel_backend: Literal["processes", "threads"] = "processes",
        chunk_size: Optional[int] = None,
        parallel_threshold: int = 1000,
        **kwargs,
    ):
        """Initialize SWC analysis.

        Parameters
        ----------
        window_size : int
            Size of the sliding window in samples
        stride : int, optional
            Number of samples to move the window in each step, by default 1
        method : {"pearson", "spearman"}, optional
            Correlation method to use, by default "pearson"
            - "pearson": Pearson correlation coefficient
            - "spearman": Spearman rank correlation
        compute_significance : bool, optional
            Whether to compute statistical significance of correlations, by default False
        seed_indices : array-like, optional
            Indices of seed points for seed-based correlation analysis.
            If None, computes full correlation matrix.
        parallel : bool, optional
            Whether to use parallel computing, by default True
        n_jobs : int, optional
            Number of parallel jobs. If None, uses all available CPU cores
        parallel_backend : {"processes", "threads"}, optional
            Parallelization method to use, by default "processes"
            - "processes": Process-based parallelization (better for CPU-bound tasks)
            - "threads": Thread-based parallelization (better for I/O-bound tasks)
        chunk_size : int, optional
            Size of chunks for parallel processing. If None, auto-determined
        parallel_threshold : int, optional
            Minimum matrix size to trigger parallel computing, by default 1000
        **kwargs : dict
            Additional keyword arguments

        Examples
        --------
        >>> # Basic usage
        >>> swc = SWC(window_size=50, stride=2)
        >>>
        >>> # With significance testing and parallel processing
        >>> swc = SWC(
        ...     window_size=50,
        ...     compute_significance=True,
        ...     n_jobs=4,
        ...     parallel_backend="processes"
        ... )
        >>>
        >>> # Seed-based analysis
        >>> swc = SWC(
        ...     window_size=50,
        ...     seed_indices=[0, 1, 2],  # Use first three nodes as seeds
        ...     parallel=True,
        ...     chunk_size=100
        ... )
        """
        self.window_size = window_size
        self.stride = stride
        self.method = method
        self.compute_significance = compute_significance
        self.seed_indices = np.array(seed_indices) if seed_indices is not None else None

        super().__init__(
            parallel=parallel,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            chunk_size=chunk_size,
            parallel_threshold=parallel_threshold,
            **kwargs,
        )

        # Attributes set during fitting
        self.n_windows_ = None
        self.window_centers_ = None
        self.result_: Optional[SWCResult] = None

        self._validate_params()

        self._cluster_cache = {}
        self._cache_ref = weakref.ref(self._cluster_cache)  # Weak reference for garbage collection

    def _validate_params(self) -> None:
        """Validate initialization parameters."""
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if self.stride < 1:
            raise ValueError("stride must be at least 1")
        if self.method not in ["pearson", "spearman"]:
            raise ValueError("method must be 'pearson' or 'spearman'")

    def _compute_window_correlation(self, window_data: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute correlation for a single window."""
        return self._compute_correlation(
            data=window_data,
            seed_indices=self.seed_indices,
            method=self.method,
            compute_significance=self.compute_significance,
        )

    def fit(self, X: np.ndarray) -> "SWC":
        """Perform SWC analysis on input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input time series data

        Returns
        -------
        SWCResult
            Container with correlation matrices and metadata
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        if n_samples < self.window_size:
            raise ValueError("Input data length must be >= window_size")

        if self.seed_indices is not None and np.max(self.seed_indices) >= n_features:
            raise ValueError("Seed indices must be less than number of features")

        # Calculate number of windows and their centers
        self.n_windows_ = (n_samples - self.window_size) // self.stride + 1
        self.window_centers_ = np.arange(
            self.window_size // 2, self.window_size // 2 + self.n_windows_ * self.stride, self.stride
        )

        # Initialize output arrays
        if self.seed_indices is not None:
            corr_shape = (self.n_windows_, len(self.seed_indices), n_features)
        else:
            corr_shape = (self.n_windows_, n_features, n_features)

        correlations = np.zeros(corr_shape)
        significance = np.zeros_like(correlations) if self.compute_significance else None

        # Prepare tasks for parallel processing
        tasks = []
        for i in range(self.n_windows_):
            start = i * self.stride
            end = start + self.window_size
            window_data = X[start:end]
            tasks.append((window_data,))

        # Determine if parallel processing should be used
        use_parallel = self.should_use_parallel(self.n_windows_)

        if use_parallel:
            # Compute correlations in parallel
            results = self._parallel_map(self._compute_window_correlation, tasks)
            for i, (corr, pvals) in enumerate(results):
                correlations[i] = corr
                if self.compute_significance:
                    significance[i] = pvals
        else:
            # Sequential processing
            for i, task in enumerate(tasks):
                corr, pvals = self._compute_window_correlation(*task)
                correlations[i] = corr
                if self.compute_significance:
                    significance[i] = pvals

        self.result_ = SWCResult(
            correlations=correlations,
            window_centers=self.window_centers_,
            significance=significance,
            seed_indices=self.seed_indices,
        )

        return self

    def get_temporal_correlation(self, roi1: int, roi2: Optional[int] = None) -> np.ndarray:
        """Extract temporal correlation between ROIs.

        Parameters
        ----------
        roi1 : int
            Index of first ROI or seed index
        roi2 : int, optional
            Index of second ROI. Not needed if using seed-based analysis.

        Returns
        -------
        ndarray
            Time series of correlation values
        """
        if self.result_ is None:
            raise ValueError("No results available. Run fit first.")

        if self.result_.seed_indices is not None:
            if roi1 >= len(self.result_.seed_indices):
                raise ValueError("roi1 must be a valid seed index")

        return self.result_.correlations[:, roi1, roi2]

    def cluster(self, n_clusters: Optional[int] = None, random_state: Optional[int] = None) -> "SWC":
        """Perform k-means clustering on correlation matrices to identify brain states."""
        if self.result_ is None:
            raise ValueError("No results available. Run fit first.")

        if n_clusters is None:
            n_clusters, _ = self.find_optimal_clusters(random_state=random_state)

        clusters, cluster_centers = self._cluster_from_result(self.result_.correlations, n_clusters, random_state)

        # Clear the cache after final clustering
        self._cluster_cache.clear()

        # Update result with clustering information
        self.result_.n_clusters = n_clusters
        self.result_.clusters = clusters
        self.result_.cluster_centers = cluster_centers

        return self

    def _cluster_from_result(
        self, correlations: np.ndarray, n_clusters: int, random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform clustering on correlation matrices."""
        # Check cache first
        cache_key = (n_clusters, random_state)
        if cache_key in self._cluster_cache:
            return self._cluster_cache[cache_key]

        n_samples = correlations.shape[0]
        X = correlations.reshape(n_samples, -1)

        # Use shared clustering utility
        clusters, centers = perform_clustering(X, n_clusters, random_state)
        cluster_centers = centers.reshape(n_clusters, *correlations.shape[1:])

        # Cache the result
        self._cluster_cache[cache_key] = (clusters, cluster_centers)

        return clusters, cluster_centers

    def find_optimal_clusters(
        self, k_range: tuple[int, int] = (2, 20), method: str = "elbow", random_state: Optional[int] = None
    ) -> Tuple[int, List[float]]:
        """Find optimal number of clusters."""
        if self.result_ is None:
            raise ValueError("No results available. Run fit first.")

        n_windows = self.result_.correlations.shape[0]
        X_cluster = self.result_.correlations.reshape(n_windows, -1)

        return find_optimal_clusters(X_cluster, k_range=k_range, method=method, random_state=random_state)

    def switch_graph(self) -> np.ndarray:
        """Compute state transition matrix from stored result.

        Returns
        -------
        np.ndarray
            Square matrix of shape (n_clusters, n_clusters) containing
            counts of transitions between states. Element [i,j] represents
            the number of transitions from state i to state j.

        Raises
        ------
        ValueError
            If no results are available or clustering hasn't been performed
        """
        if self.result_ is None:
            raise ValueError("No results available. Run fit first.")
        if self.result_.clusters is None:
            raise ValueError("No clustering results available. Run cluster_states first.")

        return self._switch_graph_from_result(self.result_.clusters, self.result_.n_clusters)

    def _switch_graph_from_result(self, n_clusters, clusters) -> np.ndarray:
        """Compute state transition matrix from cluster labels."""
        switch_matrix = np.zeros((n_clusters, n_clusters), dtype=int)

        # Count transitions between consecutive time points
        for i in range(len(clusters) - 1):
            current_state = clusters[i]
            next_state = clusters[i + 1]
            switch_matrix[current_state, next_state] += 1

        return switch_matrix

    def clear_cluster_cache(self):
        """Manually clear the clustering cache."""
        self._cluster_cache.clear()
