"""Image reading utilities for fluorescence data.

This module provides functions for reading single images, image stacks,
and arrays of images using OpenCV. It handles common image formats and
provides proper error handling.

Functions:
    read_stack_image: Read a multi-page image stack from a file
    read_single_image: Read a single image from a file
    read_image_array: Read multiple single images from a list of files
    read_stack_array: Read multiple image stacks from a list of files
"""

import cv2
from typing import List, Union
import numpy as np
from pathlib import Path


def read_stack_image(filename: Union[str, Path]) -> List[np.ndarray]:
    """Read a multi-page image stack from a file.

    Args:
        filename: Path to the image stack file

    Returns:
        List of numpy arrays representing each image in the stack

    Raises:
        ValueError: If the file cannot be read
    """
    ext = Path(filename).suffix
    try:
        if ext == ".npy":
            img = np.load(str(filename))
            return img
        else:
            success, img = cv2.imreadmulti(str(filename), cv2.IMREAD_UNCHANGED)
            if not success:
                raise ValueError(f"Failed to read stack from {filename}")
            return img
    except Exception as err:
        raise ValueError(f"Unsupported file type: {ext} or broken file {filename}") from err


def read_single_image(filename: Union[str, Path]) -> np.ndarray:
    """Read a single image from a file.

    Args:
        filename: Path to the image file

    Returns:
        Numpy array representing the image

    Raises:
        ValueError: If the file cannot be read
    """
    ext = Path(filename).suffix
    try:
        if ext == ".npy":
            img = np.load(str(filename))
            return img
        else:
            img = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
            return img
    except Exception as err:
        raise ValueError(f"Unsupported file type: {ext} or broken file {filename}") from err


def read_image_array(filenames: List[Union[str, Path]]) -> List[np.ndarray]:
    """Read multiple single images from a list of files.

    Args:
        filenames: List of paths to image files

    Returns:
        List of numpy arrays representing each image
    """
    return [read_single_image(filename) for filename in filenames]


def read_stack_array(filenames: List[Union[str, Path]]) -> List[List[np.ndarray]]:
    """Read multiple image stacks from a list of files.

    Args:
        filenames: List of paths to image stack files

    Returns:
        List of image stacks, where each stack is a list of numpy arrays
    """
    return [read_stack_image(filename) for filename in filenames]
