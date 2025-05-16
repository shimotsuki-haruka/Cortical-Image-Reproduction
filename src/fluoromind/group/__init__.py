"""Group-level analysis module for functional connectivity.

This module provides functions and classes for managing and analyzing group-level data,
including statistical comparisons between groups.
"""

from .fc import GroupFC
from .pca import GroupPCA
from .cpca import GroupCPCA
from .caps import GroupCAPs
from .swc import GroupSWC

__all__ = ["GroupFC", "GroupPCA", "GroupCPCA", "GroupCAPs", "GroupSWC"]
