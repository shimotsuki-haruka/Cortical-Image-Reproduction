"""Algorithms module for fluoromind.

This package provides various algorithms for analyzing fluorescence microscopy data,
including correlation analysis, clustering, and statistical utilities.
"""

from .fc import FC, FCResult
from .cpca import CPCA, CPCAResult
from .caps import CAPs, CAPsResult
from .swc import SWC, SWCResult
from .clustering import find_optimal_clusters, perform_clustering
from ..core.stats import (
    fisher_z,
    inverse_fisher_z,
    correct_pvalues,
    symmetrize_matrix,
)

__all__ = [
    "FC",
    "FCResult",
    "CPCA",
    "CPCAResult",
    "CAPs",
    "CAPsResult",
    "SWC",
    "SWCResult",
    "find_optimal_clusters",
    "perform_clustering",
    "fisher_z",
    "inverse_fisher_z",
    "correct_pvalues",
    "symmetrize_matrix",
]
