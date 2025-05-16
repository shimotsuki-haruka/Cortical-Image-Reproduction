import numpy as np
from typing import Optional, Tuple, List, Any
from ..algorithms import SWC, SWCResult
from ..algorithms.clustering import find_optimal_clusters


class GroupSWC(SWC):
    """Group-level Sliding Window Correlation (SWC).

    A variant of SWC that handles group-level data and provides sequence-like access
    to individual subject SWC results while maintaining group-level information.
    """

    def __init__(self, window_size: int = 50, stride: int = 1, seed_indices: Optional[List[int]] = None):
        super().__init__(window_size=window_size, stride=stride, seed_indices=seed_indices)
        self.n_samples_: Optional[List[int]] = None
        self.start_indices_: Optional[np.ndarray[Any, np.dtype[Any]]] = None
        self.subject_results_: Optional[List[SWCResult]] = None
        self.group_correlations_: Optional[np.ndarray] = None

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
        return super().get_params(deep=deep)

    def set_params(self, **params) -> "GroupSWC":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : GroupSWC
            Estimator instance.
        """
        super().set_params(**params)
        return self

    def fit(self, data: List[np.ndarray]) -> "GroupSWC":
        """Fit the SWC model to group data.

        Parameters
        ----------
        data : List[np.ndarray]
            List of time series data for each subject

        Returns
        -------
        GroupSWC
            The fitted GroupSWC instance
        """
        self.n_samples_ = [sub_data.shape[0] for sub_data in data]

        # Calculate number of windows for each subject
        n_windows = []
        for sub_data in data:
            n_samples = sub_data.shape[0]
            n_win = (n_samples - self.window_size) // self.stride + 1
            n_windows.append(n_win)

        self.start_indices_ = np.insert(np.cumsum(n_windows), 0, 0)

        # Compute SWC for each subject
        self.subject_results_ = []
        for sub_data in data:
            result = super().fit(sub_data)
            self.subject_results_.append(result.result_)

        # Compute average correlations across subjects
        if self.seed_indices is not None:
            # For seed-based analysis
            avg_shape = (max(n_windows), len(self.seed_indices), data[0].shape[1])
        else:
            # For full correlation matrix
            avg_shape = (max(n_windows), data[0].shape[1], data[0].shape[1])

        # Initialize arrays for averaging
        sum_correlations = np.zeros(avg_shape)
        count_windows = np.zeros(avg_shape[0])

        # Sum up correlations across subjects
        for i, result in enumerate(self.subject_results_):
            n_win = n_windows[i]
            sum_correlations[:n_win] += result.correlations
            count_windows[:n_win] += 1

        # Compute average, handling potential division by zero
        mask = count_windows > 0
        self.group_correlations = np.zeros_like(sum_correlations)
        for t in range(avg_shape[0]):
            if mask[t]:
                self.group_correlations[t] = sum_correlations[t] / count_windows[t]

        return self

    def __len__(self) -> int:
        if self.n_samples_ is None:
            raise ValueError("GroupSWC model not fitted yet.")
        return len(self.n_samples_)

    def __getitem__(self, index: int) -> SWCResult:
        if self.subject_results_ is None:
            raise ValueError("GroupSWC model not fitted yet.")
        return self.subject_results_[index]

    def _all_correlations(self) -> np.ndarray:
        if self.subject_results_ is None:
            raise ValueError("No results available. Run fit first.")
        all_correlations = []
        for result in self.subject_results_:
            all_correlations.append(result.correlations)
        return np.concatenate(all_correlations, axis=0)

    def find_optimal_clusters(
        self, k_range: tuple[int, int] = (2, 20), method: str = "elbow", random_state: Optional[int] = None
    ) -> Tuple[int, List[float]]:
        """Find optimal number of clusters."""
        if self.subject_results_ is None:
            raise ValueError("No results available. Run fit first.")

        all_correlations = self._all_correlations()

        n_windows = all_correlations.shape[0]
        X_cluster = all_correlations.reshape(n_windows, -1)

        return find_optimal_clusters(X_cluster, k_range=k_range, method=method, random_state=random_state)

    def cluster(self, n_clusters: Optional[int] = None, random_state: Optional[int] = None) -> "GroupSWC":
        """Perform clustering on both group and individual subject results."""
        # Cluster group-level results
        all_correlations = self._all_correlations()

        if n_clusters is None:
            n_clusters, _ = self.find_optimal_clusters(random_state=random_state)
        # Perform clustering
        clusters, centroids = self._cluster_from_result(all_correlations, n_clusters, random_state)

        # Split clusters back to individual subjects
        start_idx = 0
        for i, result in enumerate(self.subject_results_):
            n_win = len(result.correlations)
            subject_clusters = clusters[start_idx : start_idx + n_win]
            self.subject_results_[i] = SWCResult(
                correlations=result.correlations,
                window_centers=result.window_centers,
                significance=result.significance,
                seed_indices=result.seed_indices,
                clusters=subject_clusters,
                centroids=centroids.reshape(-1, *all_correlations.shape[1:]),
            )
            start_idx += n_win

        # Store group-level results
        self.n_clusters = n_clusters
        self.clusters = clusters
        self.cluster_centers = centroids.reshape(-1, *all_correlations.shape[1:])

        return self

    def switch_graph(self, subject_idx: Optional[int] = None) -> np.ndarray:
        """Compute state transition matrix.

        Parameters
        ----------
        subject_idx : int, optional
            If provided, returns the switch graph for a specific subject.
            If None, returns the group-level switch graph.

        Returns
        -------
        np.ndarray
            State transition matrix
        """
        if subject_idx is not None:
            if self.subject_results_ is None:
                raise ValueError("GroupSWC model not fitted yet.")
            return self._switch_graph_from_result(
                self.subject_results_[subject_idx].clusters, self.subject_results_[subject_idx].n_clusters
            )
        else:
            return self._switch_graph_from_result(self.clusters, self.n_clusters)
