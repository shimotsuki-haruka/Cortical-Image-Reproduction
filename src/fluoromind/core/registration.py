"""Registration module for aligning imaging data.

This module provides functions and classes for registering 2D time series data
to anatomical reference spaces, supporting both rigid and affine transformations.
"""

import numpy as np
import cv2


def get_transform(src_points, dst_points):
    """Get affine transformation matrix from corresponding points.

    Args:
        src_points (NDArray): Source points (3x2 array of float32)
        dst_points (NDArray): Destination points (3x2 array of float32)

    Returns:
        NDArray: 2x3 affine transformation matrix mapping src_points to dst_points

    Raises:
        ValueError: If less than 3 point pairs are provided
    """
    if len(src_points) < 3 or len(dst_points) < 3:
        raise ValueError("At least 3 point pairs are required for affine transform")
    return cv2.getAffineTransform(src_points[:3], dst_points[:3])


def apply_transform(
    image,
    translation,
    rotation,
    scale,
    shear,
    shape,
    interpolation=cv2.INTER_LINEAR,
    inverse=False,
    border_mode=cv2.BORDER_CONSTANT,
    border_value=0,
):
    """Apply affine transformation to an image.

    Args:
        image (NDArray): Input image
        translation (tuple): (x, y) coordinates of the transformation center
        rotation (float): Counter-clockwise rotation angle in degrees
        scale (Union[float, tuple]): Scale factor(s). If float, uniform scaling.
            If tuple (sx, sy), separate scaling in x and y directions
        shear (float): Shear angle in degrees
        shape (tuple): Output image size as (width, height)
        interpolation (int, optional): OpenCV interpolation method. Defaults to INTER_LINEAR
        inverse (bool, optional): Whether to apply inverse transform. Defaults to False
        border_mode (int, optional): OpenCV border mode. Defaults to BORDER_CONSTANT
        border_value (Union[int, tuple], optional): Value to fill borders. Defaults to 0

    Returns:
        NDArray: Transformed image of specified shape
    """
    if inverse:
        interpolation |= cv2.WARP_INVERSE_MAP
    center = (shape[1] // 2, shape[0] // 2)
    M = get_transform_matrix(translation, rotation, scale, shear, center)
    resized = cv2.resize(image, shape, interpolation=interpolation)
    return cv2.warpAffine(resized, M, shape, flags=interpolation, borderMode=border_mode, borderValue=border_value)


def get_transform_matrix(translation, rotation, scale, shear, center=(0, 0)):
    """Create an affine transformation matrix from transformation parameters.

    Args:
        translation (tuple): (tx, ty) translation shifts
        rotation (float): Counter-clockwise rotation angle in degrees
        scale (tuple): (sx, sy) scale factors for x and y directions
        shear (float): Shear angle in degrees
        center (tuple, optional): (cx, cy) coordinates of the rotation center point. Defaults to (0, 0).

    Returns:
        NDArray: 2x3 affine transformation matrix
    """
    # Convert angles to radians
    rotation_rad = np.deg2rad(rotation)
    shear_rad = np.deg2rad(shear)

    # Get scale factors
    sx, sy = scale if isinstance(scale, tuple) else (scale, scale)

    # Create rotation + scale + shear matrix
    a = sx * np.cos(rotation_rad)
    b = sx * np.sin(rotation_rad)
    c = -sy * np.sin(rotation_rad - shear_rad)
    d = sy * np.cos(rotation_rad - shear_rad)

    cx, cy = center
    tx_shift, ty_shift = translation

    # Calculate translation to keep the center invariant
    tx = cx - (a * cx + b * cy) + tx_shift
    ty = cy - (c * cx + d * cy) + ty_shift

    # Combine into 2x3 affine matrix
    M = np.array([[a, b, tx], [c, d, ty]], dtype=np.float32)

    return M


def get_matching_points(image, target, n_points=None):
    """Find matching points between an image and a target using SIFT feature detection.

    Args:
        image (NDArray): Input image to find features in
        target (NDArray): Target image to find matching features with
        n_points (Optional[int]): Number of matching point pairs to return.
            If None, returns all good matches found

    Returns:
        Tuple[NDArray, NDArray]: Two arrays of corresponding points (src_pts, dst_pts),
            each with shape (n_points, 2) containing (x, y) coordinates

    Raises:
        ValueError: If not enough features are found in either image
        ValueError: If not enough good matches are found between images
    """
    # Convert inputs to 8-bit grayscale
    img1 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    img2 = cv2.normalize(target, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # Initialize feature detector and matcher
    try:
        # OpenCV 4.x version
        sift = cv2.SIFT_create()
    except AttributeError:
        # OpenCV 3.x version
        sift = cv2.xfeatures2d.SIFT_create()

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        raise ValueError("Not enough features found in images")

    if n_points is not None and (len(kp1) < n_points or len(kp2) < n_points):
        raise ValueError(
            f"Not enough features found in images (required: {n_points}, found: {min(len(kp1), len(kp2))})"
        )

    # Match descriptors
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test to find good matches
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:  # 确保我们有两个匹配点进行比较
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    # Sort matches by distance
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # Take top n_points matches
    if n_points is not None:
        good_matches = good_matches[:n_points]

    if len(good_matches) == 0:
        raise ValueError("No good matches found between images")

    if n_points is not None and len(good_matches) < n_points:
        raise ValueError(
            f"Not enough good matches found between images (required: {n_points}, found: {len(good_matches)})"
        )

    # Extract matching point coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    return src_pts, dst_pts
