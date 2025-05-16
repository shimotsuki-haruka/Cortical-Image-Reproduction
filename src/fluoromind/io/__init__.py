"""IO operations for fluoromind.

This package provides utilities for reading, writing, and managing fluorescence data.

Module Structure:
    - data: Core data handling classes (FluoroData, FluoroDataManager)
    - readers: Image and data reading utilities
    - storage: Data persistence and serialization strategies
    - base: Abstract base classes and interfaces

Example:
    >>> from fluoromind.io import FluoroData
    >>> data = FluoroData(array_data)
    >>> data.save("output.h5")
"""

# Core data handling
from .handlers import MaskedData

# Storage and serialization
from .storage import BaseResult, SaveableMixin

# Image reading utilities
from .readers import (
    read_stack_image,
    read_single_image,
    read_image_array,
    read_stack_array,
)

__all__ = [
    "MaskedData",
    "BaseResult",
    "SaveableMixin",
    "read_stack_image",
    "read_single_image",
    "read_image_array",
    "read_stack_array",
]
