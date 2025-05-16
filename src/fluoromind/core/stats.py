"""Statistical utilities for fluoromind.

This module provides statistical transformation and correction methods commonly used
in fluorescence data analysis.
"""

import numpy as np
from scipy import stats
from typing import Optional


def fisher_z(matrix: np.ndarray, threshold: Optional[float] = 0.99) -> np.ndarray:
    """Apply Fisher z-transformation to correlation matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Correlation matrix

    Returns
    -------
    np.ndarray
        Z-transformed matrix
    """
    if threshold is not None:
        matrix = np.clip(matrix, -threshold, threshold)
    return np.arctanh(matrix)


def inverse_fisher_z(matrix: np.ndarray) -> np.ndarray:
    """Apply inverse Fisher z-transformation.

    Parameters
    ----------
    matrix : np.ndarray
        Z-transformed matrix

    Returns
    -------
    np.ndarray
        Correlation matrix
    """
    return np.tanh(matrix)


def correct_pvalues(p_vals: np.ndarray, alpha: float, method: str = "fdr") -> np.ndarray:
    """Apply multiple comparison correction to p-values.

    Parameters
    ----------
    p_vals : np.ndarray
        Uncorrected p-values
    alpha : float
        Significance level (typically 0.05)
    method : str
        Correction method:
        - "fdr": False Discovery Rate correction (Benjamini-Hochberg)
        - "bonferroni": Bonferroni correction (more conservative)

    Returns
    -------
    np.ndarray
        Corrected p-values

    Raises
    ------
    ValueError
        If an unknown correction method is specified
    """
    if method == "fdr":
        # Use scipy.stats.false_discovery_control for FDR correction
        # The function returns adjusted p values
        p_corrected = stats.false_discovery_control(p_vals, method="bh")
        return p_corrected
    elif method == "bonferroni":
        p_corrected = np.minimum(p_vals * len(p_vals), 1.0)
    else:
        raise ValueError(f"Unknown correction method: {method}")
    return p_corrected


def symmetrize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Make matrix symmetric by copying upper triangle to lower triangle.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix

    Returns
    -------
    np.ndarray
        Symmetric matrix
    """
    return matrix + matrix.T - np.diag(np.diag(matrix))
