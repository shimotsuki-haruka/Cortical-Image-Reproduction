import numpy as np
from typing import Optional, Tuple
from numpy.typing import NDArray


class MaskedData(np.ndarray):
    """
    A handler class for masked array operations that behaves like a numpy array.

    This class extends numpy.ndarray to provide transparent masking operations
    while maintaining array-like behavior. When performing calculations,
    it automatically applies the mask and returns results in the correct shape.

    Parameters
    ----------
    data : NDArray
        The input data array of shape (n_timepoints, height, width)
    mask : NDArray
        Binary mask of shape (height, width)

    Examples
    --------
    >>> # Create from data and mask
    >>> masked = MaskedData(data, mask)
    >>>
    >>> # Use as numpy array (automatically applies mask)
    >>> mean_signal = masked.mean(axis=0)  # Mean across time
    >>> std_signal = np.std(masked, axis=1)  # Std across voxels
    >>>
    >>> # Perform calculations
    >>> normalized = (masked - masked.mean()) / masked.std()
    >>> correlation = np.corrcoef(masked[0], masked[1])
    >>>
    >>> # Get full 3D data back
    >>> data_3d = masked.restore()
    """

    def __new__(cls, data: NDArray, mask: NDArray):
        """Create new instance with numpy array inheritance."""
        # Ensure data is 3D
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        # Get mask indices
        mask_indices = np.where(mask.flatten())[0]

        # Create masked 2D array (time points × valid voxels)
        masked_data = data.reshape(data.shape[0], -1)[:, mask_indices]

        # Create numpy array instance
        obj = masked_data.view(cls)

        # Store additional attributes
        obj.mask = mask
        obj.mask_indices = mask_indices
        obj.original_shape = data.shape

        return obj

    def __array_finalize__(self, obj):
        """Finalize new array instance."""
        if obj is None:
            return
        self.mask = getattr(obj, "mask", None)
        self.mask_indices = getattr(obj, "mask_indices", None)
        self.original_shape = getattr(obj, "original_shape", None)

    def restore(self, data: Optional[NDArray] = None) -> NDArray:
        """
        Restore masked data to full 3D array.

        Parameters
        ----------
        data : NDArray, optional
            Data to restore. If None, uses internal data.
            Must have compatible shape with mask.

        Returns
        -------
        NDArray
            Restored 3D array of shape (n_timepoints, height, width)
        """
        if data is None:
            data = self.view(np.ndarray)  # Get raw array data

        if data.ndim == 1:
            # Single time point
            data = data.reshape(1, -1)

        # Create output array
        restored = np.zeros((data.shape[0], np.prod(self.original_shape[1:])))

        # Vectorized restoration
        restored[:, self.mask_indices] = data

        # Reshape back to original 3D shape
        return restored.reshape(data.shape[0], *self.original_shape[1:])

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle numpy ufuncs by applying them to the masked data.

        This method ensures that numpy operations work seamlessly with
        the masked data while maintaining the masking information.

        Parameters
        ----------
        ufunc : numpy.ufunc
            The ufunc object that was called
        method : str
            The method of the ufunc that was called
        *inputs : tuple
            The input arguments to the ufunc
        **kwargs : dict
            The keyword arguments to the ufunc

        Returns
        -------
        MaskedData or ndarray
            The result of applying the ufunc to the masked data
        """
        # Convert MaskedData inputs to regular arrays
        inputs = tuple(i.view(np.ndarray) if isinstance(i, MaskedData) else i for i in inputs)

        # Apply the ufunc
        result = getattr(ufunc, method)(*inputs, **kwargs)

        # If the result is a tuple, apply the mask to each element
        if isinstance(result, tuple):
            return tuple(self.__class__(r, self.mask) if isinstance(r, np.ndarray) else r for r in result)

        # If the result is an array, apply the mask
        if isinstance(result, np.ndarray):
            return self.__class__(result, self.mask)

        return result
        """Handle numpy universal functions."""
        # Convert MaskedData inputs to numpy arrays
        inputs = tuple(x.view(np.ndarray) if isinstance(x, MaskedData) else x for x in inputs)

        # Apply the ufunc
        result = getattr(ufunc, method)(*inputs, **kwargs)

        # If the result should be a MaskedData, convert it
        if method == "__call__" and isinstance(result, np.ndarray):
            # Create new MaskedData with same metadata
            new_result = result.view(MaskedData)
            new_result.mask = self.mask
            new_result.mask_indices = self.mask_indices
            new_result.original_shape = self.original_shape
            return new_result

        return result

    def __array_function__(self, func, types, args, kwargs):
        """Handle numpy functions."""
        # Convert MaskedData inputs to numpy arrays
        args = tuple(x.view(np.ndarray) if isinstance(x, MaskedData) else x for x in args)

        # Apply the function
        result = func(*args, **kwargs)

        # If the result should be a MaskedData, convert it
        if isinstance(result, np.ndarray):
            # Create new MaskedData with same metadata
            new_result = result.view(MaskedData)
            new_result.mask = self.mask
            new_result.mask_indices = self.mask_indices
            new_result.original_shape = self.original_shape
            return new_result

        return result

    def apply_mask(self, data: NDArray) -> NDArray:
        """
        Apply mask to new data.

        Parameters
        ----------
        data : NDArray
            Data to mask, must be compatible with original shape

        Returns
        -------
        NDArray
            Masked data in 2D format (time points × valid voxels)
        """
        if data.shape[-2:] != self.original_shape[-2:]:
            raise ValueError("Data shape incompatible with mask")

        masked = data.reshape(data.shape[0], -1)[:, self.mask_indices]

        return masked

    def update(self, data: NDArray):
        """
        Update internal data while keeping the same mask.

        Parameters
        ----------
        data : NDArray
            New data to use, must be compatible with original shape
        """
        masked = self.apply_mask(data)
        self[:] = masked

    @property
    def shape_3d(self) -> Tuple[int, ...]:
        """Get the shape of the original 3D data."""
        return self.original_shape

    def __repr__(self) -> str:
        """String representation."""
        return f"MaskedData(shape={self.shape}, original_shape={self.original_shape}, dtype={self.dtype})"
