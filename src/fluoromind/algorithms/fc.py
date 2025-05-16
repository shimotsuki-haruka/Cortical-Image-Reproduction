import numpy as np
from typing import Optional, Union, Literal, List
from dataclasses import dataclass
from .correlation import BaseCorrelation
from ..io import BaseResult, SaveableMixin


@dataclass
class FCResult(BaseResult):
    """Container for functional connectivity analysis results.

    Parameters
    ----------
    fc_matrix : Optional[np.ndarray]
        FC matrix (nodes, nodes)
    p_values : Optional[np.ndarray]
        P-values matrix (nodes, nodes), only if significance=True
    thresholded : Optional[np.ndarray]
        Thresholded FC matrix (nodes, nodes)
    """

    fc_matrix: Optional[np.ndarray] = None
    p_values: Optional[np.ndarray] = None
    thresholded: Optional[np.ndarray] = None


class FC(BaseCorrelation, SaveableMixin):
    """Functional Connectivity (FC) analysis class for individual-level analysis.

    This class implements methods to compute and analyze functional connectivity
    matrices from time series data for single subjects.

    Methods
    -------
    calculate_connectivity(data, method='pearson', significance=False)
        Calculate and store FC matrix from time series data
    apply_threshold(threshold=None, percentile=None, keep_diagonal=False)
        Apply threshold to stored FC matrix
    clear()
        Reset all stored results

    Examples
    --------
    >>> # Basic usage
    >>> fc = FC()
    >>> fc.fit(time_series_data)
    >>> fc_matrix = fc.result_.fc_matrix
    >>>
    >>> # With significance testing
    >>> fc.fit(time_series_data, significance=True)
    >>> p_values = fc.result_.p_values
    >>>
    >>> # With thresholding
    >>> fc.apply_threshold(percentile=95)
    >>> thresholded = fc.result_.thresholded
    >>>
    >>> # Access results
    >>> result = fc.result_
    >>> print(result.fc_matrix)     # Original FC matrix
    >>> print(result.p_values)      # P-values if computed
    >>> print(result.thresholded)   # Thresholded matrix if applied
    """

    def __init__(
        self,
        seed_indices: Optional[Union[List[int], np.ndarray]] = None,
        parallel: bool = True,
        n_jobs: Optional[int] = None,
        parallel_backend: Literal["processes", "threads"] = "processes",
        chunk_size: Optional[int] = None,
        parallel_threshold: int = 1000,
        **kwargs,
    ):
        """Initialize FC class.

        Parameters
        ----------
        parallel : bool, optional
            Whether to use parallel computing, by default True
        n_jobs : int, optional
            Number of parallel jobs. If None, uses all available CPU cores
        parallel_backend : {"processes", "threads"}, optional
            Parallelization method to use, by default "processes"
        chunk_size : int, optional
            Size of chunks for parallel processing. If None, auto-determined
        parallel_threshold : int, optional
            Minimum matrix size to trigger parallel computing, by default 1000
        """
        self.result_ = None
        self.seed_indices = seed_indices
        super(BaseCorrelation, self).__init__(
            parallel=parallel,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            chunk_size=chunk_size,
            parallel_threshold=parallel_threshold,
            **kwargs,
        )

    def fit(
        self,
        data: np.ndarray,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        significance: bool = False,
    ) -> "FC":
        """Calculate and store functional connectivity matrix from time series data.

        Parameters
        ----------
        data : np.ndarray
            Time series data with shape (time_points, nodes)
        method : {"pearson", "spearman", "kendall"}
            Correlation method to use
        significance : bool, default=False
            Whether to compute significance (p-values)

        Returns
        -------
        FC
            The fitted FC instance

        Raises
        ------
        ValueError
            If input data is not 2-dimensional
        """
        if data.ndim != 2:
            raise ValueError("Input data must be 2-dimensional (time_points, nodes)")

        fc_matrix, p_matrix = self._compute_correlation(
            data=data,
            seed_indices=self.seed_indices,
            method=method,
            compute_significance=significance,
        )

        self.result_ = FCResult(fc_matrix=fc_matrix, p_values=p_matrix)

        return self

    def apply_threshold(
        self,
        threshold: Optional[float] = None,
        percentile: Optional[float] = None,
        keep_diagonal: bool = False,
    ) -> "FC":
        """Apply threshold to the stored functional connectivity matrix.

        Parameters
        ----------
        threshold : float, optional
            Absolute threshold value
        percentile : float, optional
            Percentile threshold (0-100)
        keep_diagonal : bool, optional
            Whether to keep diagonal elements, by default False

        Raises
        ------
        ValueError
            If FC matrix hasn't been computed yet
            If neither threshold nor percentile is provided
            If both threshold and percentile are provided
        """
        if self.result_ is None:
            raise ValueError("No FC matrix available. Run compute first.")

        if threshold is None and percentile is None:
            raise ValueError("Either threshold or percentile must be provided")
        if threshold is not None and percentile is not None:
            raise ValueError("Only one of threshold or percentile should be provided")

        # Create copy to avoid modifying original
        thresholded = self.result_.fc_matrix.copy()

        # Calculate threshold value if percentile is given
        if percentile is not None:
            if not 0 <= percentile <= 100:
                raise ValueError("Percentile must be between 0 and 100")
            threshold = np.percentile(np.abs(thresholded), percentile)

        # Apply threshold
        thresholded[np.abs(thresholded) < threshold] = 0

        # Set diagonal to zero if requested
        if not keep_diagonal:
            np.fill_diagonal(thresholded, 0)

        self.result_.thresholded = thresholded

        return self
