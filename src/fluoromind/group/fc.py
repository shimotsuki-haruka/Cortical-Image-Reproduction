import numpy as np
from typing import Optional, List, Any

from ..core.stats import fisher_z, inverse_fisher_z
from ..algorithms import FC

class GroupFC:
    """Group-level Functional Connectivity (FC).

    A variant of FC that handles group-level data and provides sequence-like access
    to individual subject FC results while maintaining group-level information.

    Parameters
    ----------
    **kwargs
        Additional arguments for FC
    """

    def __init__(self, seed_indices: Optional[List[int]] = None, **kwargs):
        self.result_ = None
        self.n_samples: Optional[List[int]] = None
        self.average_fc: Optional[np.ndarray] = None
        self.kwargs = kwargs
        self.seed_indices = seed_indices

    def fit(self, data: List[np.ndarray], significance: bool = False) -> "GroupFC":
        """Fit the FC model to group data.

        Parameters
        ----------
        data : List[np.ndarray]
            List of time series data for each subject
        significance : bool, optional
            Whether to compute significance values, by default False

        Returns
        -------
        GroupFC
            The fitted GroupFC instance
        """

        # Compute FC for each subject using separate FC instances
        results = []
        for sub_data in data:
            fc_instance = FC(seed_indices=self.seed_indices, **self.kwargs)
            fc_instance.fit(sub_data, significance=significance)
            results.append(fc_instance.result_)
        self.result_ = results

        # Compute average FC using Fisher z-transform
        fc_matrices = np.array([result.fc_matrix for result in results])

        if fc_matrices.size == 0:
            self.average_fc = None
        else:
            if fc_matrices.ndim == 2:
                fc_matrices = fc_matrices[:, np.newaxis, :]
            self.average_fc = inverse_fisher_z(np.mean(fisher_z(fc_matrices), axis=0))

        return self

    def __len__(self) -> int:
        if self.n_samples is None:
            raise ValueError("GroupFC model not fitted yet.")
        return len(self.n_samples)

    def __getitem__(self, index: int):
        if self.result_ is None:
            raise ValueError("GroupFC model not fitted yet.")
        return self.result_[index]

    @property
    def group_fc(self) -> np.ndarray:
        """Get the group-average FC matrix.

        Returns
        -------
        np.ndarray
            The group-average FC matrix
        """
        if self.average_fc is None:
            raise ValueError("GroupFC model not fitted yet.")
        return self.average_fc
