"""FluoroMind: A Python package for fluorescence microscopy data analysis.

FluoroMind provides a comprehensive suite of tools for analyzing fluorescence microscopy
data, with the following key features:

Features
--------
- Image I/O operations for common microscopy formats
- Correlation analysis tools:
    - Pearson correlation
    - Spearman correlation
    - Kendall correlation
- Advanced clustering algorithms:
    - K-means with automatic cluster detection
    - Spatial clustering
- Statistical utilities:
    - Fisher z-transformation
    - Multiple comparison corrections (FDR, Bonferroni)
- Core preprocessing capabilities
- Parallel processing support
- HTTP server capabilities (optional)

Server Usage
-----------
The package includes an optional HTTP server that can be started using:
    python -m fluoromind.server start

For server options and documentation:
    python -m fluoromind.server --help

Server features require additional dependencies:
    pip install fluoromind[server]

For documentation and examples, visit:
https://fluoromind.github.io/FluoroMind
"""

from __future__ import annotations

# Version information
__version__: str = "0.1.0"
__author__: str = "Shi Liang"
__license__: str = "MIT"

# Import core functionality
from .core.analysis import *  # noqa: F403
from .core.preprocessing import *  # noqa: F403
from .group import *  # noqa: F403

# Define public API
__all__ = (
    # Add the main classes and functions that should be publicly available
    "__version__",
    "__author__",
    "__license__",
)
