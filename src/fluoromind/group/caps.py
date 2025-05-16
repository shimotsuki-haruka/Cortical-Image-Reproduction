import numpy as np
from typing import Optional, List, Any
from ..algorithms import CAPs, CAPsResult


class GroupCAPs(CAPs):
    """Group-level Co-Activation Patterns (CAPs).

    A variant of CAPs that handles group-level data by concatenating subjects along the time dimension.
    Inherits from CAPs and adds functionality to extract subject-specific results.
    """

    def __init__(self, n_patterns: Optional[int] = None, threshold: float = 1.0, random_state: Optional[int] = None):
        super().__init__(n_patterns=n_patterns, threshold=threshold, random_state=random_state)
        self.result_ = None
        self.n_samples_: Optional[List[int]] = None
        self.start_indices_: Optional[np.ndarray[Any, np.dtype[Any]]] = None
        self.subject_explained_variance_: Optional[List[np.ndarray]] = None

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

    def set_params(self, **params) -> "GroupCAPs":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : GroupCAPs
            Estimator instance.
        """
        super().set_params(**params)
        return self

    def fit(self, data: List[np.ndarray]) -> "GroupCAPs":
        """Fit the GroupCAPs model to group data.

        Parameters
        ----------
        data : List[np.ndarray]
            List of subject data arrays, each of shape (n_timepoints, n_features)

        Returns
        -------
        self : GroupCAPs
            The fitted GroupCAPs instance
        """
        self.n_samples_ = [sub_data.shape[0] for sub_data in data]
        self.start_indices_ = np.insert(np.cumsum(self.n_samples_), 0, 0)

        data_2d = np.concatenate(data, axis=0)
        result = super().fit(data_2d)

        # Calculate subject-specific explained variance
        self.subject_explained_variance_ = []

        for i in range(len(self.n_samples_)):
            start_idx = self.start_indices_[i]
            end_idx = self.start_indices_[i + 1]
            sub_data = data_2d[start_idx:end_idx]
            sub_assignments = self.result_.assignments[start_idx:end_idx]
            sub_variance = self._compute_subject_variance(sub_data, sub_assignments)
            self.subject_explained_variance_.append(sub_variance)

        return result

    def _compute_subject_variance(self, data: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """Compute explained variance for a single subject."""
        total_var = np.var(data, axis=0).sum()
        pattern_var = np.zeros(self.result_.n_patterns)

        for i in range(self.result_.n_patterns):
            mask = assignments == i
            if mask.any():
                pattern_var[i] = np.var(data[mask], axis=0).sum() / total_var

        return pattern_var

    def __getitem__(self, index: int) -> CAPsResult:
        if self.result_ is None:
            raise ValueError("GroupCAPs model not fitted yet.")

        start_idx = self.start_indices_[index]
        end_idx = self.start_indices_[index + 1]
        return CAPsResult(
            patterns=self.result_.patterns,
            assignments=self.result_.assignments[start_idx:end_idx],
            explained_variance=self.subject_explained_variance_[index],
            n_patterns=self.result_.n_patterns,
        )
