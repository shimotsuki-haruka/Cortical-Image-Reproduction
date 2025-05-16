from sklearn.decomposition import PCA
from ..algorithms import FC, CPCA, CAPs, SWC
from ..group import GroupPCA, GroupCPCA, GroupCAPs, GroupSWC, GroupFC
from typing import List
import numpy as np


def pca(X, n_components=None, **kwargs) -> "PCA":
    """
    Perform Principal Component Analysis on the input data.

    This function is a wrapper around scikit-learn's PCA implementation that provides
    a simplified interface for dimensionality reduction.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples
        and n_features is the number of features.
    n_components : int or None, default=None
        Number of components to keep. If None, all components are kept.
        If int, selects the n_components with highest explained variance.
    **kwargs : dict
        Additional arguments to be passed to sklearn.decomposition.PCA.
        See scikit-learn documentation for full list of parameters.

    Returns
    -------
    PCA
        Fitted PCA model that can be used for transform, inverse_transform, etc.
        The model contains attributes like:
        - components_ : Principal axes in feature space
        - explained_variance_ratio_ : Percentage of variance explained by each component
        - mean_ : Per-feature empirical mean

    See Also
    --------
    sklearn.decomposition.PCA : scikit-learn PCA implementation
    cpca : Complex PCA implementation for complex-valued data
    """

    return PCA(n_components=n_components, **kwargs).fit(X)


def cpca(X, n_components=None, **kwargs) -> "CPCA":
    """
    Perform Complex Principal Component Analysis on the input data.

    CPCA extends traditional PCA to handle complex-valued data by properly accounting
    for both magnitude and phase information in the decomposition process.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples
        and n_features is the number of features. Can be real or complex-valued.
    n_components : int or None, default=None
        Number of components to keep. If None, all components are kept.
        If int, selects the n_components with highest explained variance.
    **kwargs : dict
        Additional arguments to be passed to CPCA constructor, including:
        - force_complex : bool, default=False
            If True, forces complex decomposition even for real-valued input
        - whiten : bool, default=False
            When True, the components_ vectors are scaled to have unit variance
        - random_state : int or RandomState instance
            Controls randomization in the algorithm

    Returns
    -------
    CPCA
        Fitted CPCA model that can be used for transform, inverse_transform, etc.
        The model contains attributes similar to standard PCA, but adapted for
        complex-valued data.

    See Also
    --------
    fluoromind.algorithms.CPCA : CPCA implementation
    pca : Standard PCA for real-valued data
    """

    return CPCA(n_components=n_components, **kwargs).fit(X)


def swc(X, window_size: int, **kwargs) -> "SWC":
    """
    Perform Sliding Window Correlation analysis on the input data.

    SWC computes dynamic functional connectivity by calculating correlation matrices
    within sliding time windows. This allows detection of time-varying connectivity
    patterns and network states.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input time series data, where n_samples is the number of timepoints
        and n_features is the number of regions/signals.
    window_size : int
        Size of the sliding window in samples. Should be large enough to get
        reliable correlation estimates but small enough to capture dynamics.
    **kwargs : dict
        Additional arguments to be passed to SWC constructor, including:
        - stride : int, default=1
            Number of samples to slide the window forward in each step
        - method : {'pearson', 'spearman'}, default='pearson'
            Correlation method to use
        - compute_significance : bool, default=False
            Whether to compute statistical significance of correlations
        - cluster : bool, default=False
            Whether to perform clustering of window-wise correlation matrices

    Returns
    -------
    SWC
        Fitted SWC model that can be used for transform, inverse_transform, etc.
        The model contains attributes similar to standard PCA, but adapted for
        complex-valued data.

    See Also
    --------
    fluoromind.algorithms.SWC : SWC implementation
    fc : Static functional connectivity analysis
    """

    return SWC(window_size=window_size, **kwargs).fit_transform(X)


def caps(X, **kwargs) -> "CAPs":
    """
    Perform Co-activation Patterns (CAPs) analysis on the input data.

    CAPs analysis identifies recurring spatial patterns in the data by clustering
    timepoints based on their activation profiles. This method is particularly
    useful for detecting transient brain states in fMRI data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input time series data, where n_samples is the number of timepoints
        and n_features is the number of regions/signals.
    **kwargs : dict
        Additional arguments to be passed to CAPs constructor, including:
        - n_patterns : int, default=None
            Number of patterns to identify. If None, determined automatically.
        - threshold : float or None, default=None
            Activation threshold for frame selection
        - standardize : bool, default=True
            Whether to z-score the data before analysis
        - random_state : int or RandomState instance
            Controls randomization in clustering

    Returns
    -------
    CAPs
        Fitted CAPs model that can be used for transform, inverse_transform, etc.
        The model contains attributes similar to standard PCA, but adapted for
        complex-valued data.

    See Also
    --------
    fluoromind.algorithms.CAPs : CAPs implementation
    """

    return CAPs(**kwargs).fit(X)


def fc(X, **kwargs) -> "FC":
    """
    Perform static Functional Connectivity analysis on the input data.

    Computes correlation-based functional connectivity between all pairs of
    signals/regions in the input data. This provides a static representation
    of the coupling between different parts of the system.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input time series data, where n_samples is the number of timepoints
        and n_features is the number of regions/signals.
    **kwargs : dict
        Additional arguments to be passed to FC constructor, including:
        - method : {'pearson', 'spearman'}, default='pearson'
            Correlation method to use
        - compute_significance : bool, default=False
            Whether to compute statistical significance
        - threshold : float or None, default=None
            Threshold for correlation values
        - fisher_transform : bool, default=False
            Whether to apply Fisher z-transform to correlations

    Returns
    -------
    FC
        Fitted FC model that can be used for transform, inverse_transform, etc.
        The model contains attributes similar to standard PCA, but adapted for
        complex-valued data.

    See Also
    --------
    fluoromind.algorithms.FC : FC implementation
    swc : Dynamic functional connectivity analysis
    """

    return FC(**kwargs).fit(X)


def group_pca(data: List[np.ndarray], n_components=None, **kwargs) -> "GroupPCA":
    """
    Perform Group-level Principal Component Analysis on multiple subjects' data.

    This function extends PCA to handle group-level data by concatenating subjects
    along the time dimension while maintaining individual subject information.

    Parameters
    ----------
    data : List[array-like]
        List of time series data for each subject, where each element is of shape
        (n_samples_i, n_features) and n_samples_i can vary across subjects.
    n_components : int or None, default=None
        Number of components to keep. If None, all components are kept.
    **kwargs : dict
        Additional arguments to be passed to GroupPCA constructor.

    Returns
    -------
    GroupPCA
        Fitted GroupPCA model that provides both group-level results and
        access to individual subject results.

    See Also
    --------
    pca : Standard PCA for single subject
    group_cpca : Group-level Complex PCA
    """
    return GroupPCA(n_components=n_components, **kwargs).fit(data)


def group_cpca(data: List[np.ndarray], n_components=None, **kwargs) -> "GroupCPCA":
    """
    Perform Group-level Complex Principal Component Analysis on multiple subjects' data.

    This function extends CPCA to handle group-level data by concatenating subjects
    along the time dimension while maintaining individual subject information.

    Parameters
    ----------
    data : List[array-like]
        List of time series data for each subject, where each element is of shape
        (n_samples_i, n_features) and n_samples_i can vary across subjects.
    n_components : int or None, default=None
        Number of components to keep. If None, all components are kept.
    **kwargs : dict
        Additional arguments to be passed to GroupCPCA constructor, including:
        - force_complex : bool, default=False
            If True, forces complex decomposition even for real-valued input
        - whiten : bool, default=False
            When True, the components_ vectors are scaled to have unit variance

    Returns
    -------
    GroupCPCA
        Fitted GroupCPCA model that provides both group-level results and
        access to individual subject results.

    See Also
    --------
    cpca : Complex PCA for single subject
    group_pca : Group-level standard PCA
    """
    return GroupCPCA(n_components=n_components, **kwargs).fit(data)


def group_swc(data: List[np.ndarray], window_size: int, **kwargs) -> "GroupSWC":
    """
    Perform Group-level Sliding Window Correlation analysis on multiple subjects' data.

    This function extends SWC to handle group-level data by computing correlations
    for each subject separately and then combining them into group-level results.

    Parameters
    ----------
    data : List[array-like]
        List of time series data for each subject, where each element is of shape
        (n_samples_i, n_features) and n_samples_i can vary across subjects.
    **kwargs : dict
        Additional arguments to be passed to GroupSWC constructor, including:
        - window_size : int
            Size of the sliding window in samples. Should be large enough to get
            reliable correlation estimates but small enough to capture dynamics.
        - stride : int, default=1
            Number of samples to slide the window forward in each step
        - method : {'pearson', 'spearman'}, default='pearson'
            Correlation method to use
        - compute_significance : bool, default=False
            Whether to compute statistical significance of correlations
        - seed_indices : array-like or None, default=None
            Indices for seed-based correlation analysis

    Returns
    -------
    GroupSWC
        Fitted GroupSWC model that provides both group-level results and
        access to individual subject results.

    See Also
    --------
    swc : SWC for single subject
    group_fc : Group-level static functional connectivity
    """
    return GroupSWC(window_size=window_size, **kwargs).fit(data)


def group_caps(data: List[np.ndarray], n_patterns=None, **kwargs) -> "GroupCAPs":
    """
    Perform Group-level Co-Activation Patterns analysis on multiple subjects' data.

    This function extends CAPs to handle group-level data by concatenating subjects
    along the time dimension while maintaining individual subject information.

    Parameters
    ----------
    data : List[array-like]
        List of time series data for each subject, where each element is of shape
        (n_samples_i, n_features) and n_samples_i can vary across subjects.
    **kwargs : dict
        Additional arguments to be passed to GroupCAPs constructor, including:
        - n_patterns : int, default=None
            Number of patterns to identify. If None, determined automatically.
        - threshold : float or None, default=None
            Activation threshold for frame selection
        - standardize : bool, default=True
            Whether to z-score the data before analysis

    Returns
    -------
    GroupCAPs
        Fitted GroupCAPs model that provides both group-level results and
        access to individual subject results.

    See Also
    --------
    caps : CAPs for single subject
    group_pca : Group-level PCA
    """
    return GroupCAPs(**kwargs).fit(data, n_patterns=n_patterns)


def group_fc(data: List[np.ndarray], **kwargs) -> "GroupFC":
    """
    Perform Group-level Functional Connectivity analysis on multiple subjects' data.

    This function extends FC to handle group-level data by computing correlations
    for each subject separately and then combining them using Fisher z-transform.

    Parameters
    ----------
    data : List[array-like]
        List of time series data for each subject, where each element is of shape
        (n_samples_i, n_features) and n_samples_i can vary across subjects.
    **kwargs : dict
        Additional arguments to be passed to GroupFC constructor, including:
        - method : {'pearson', 'spearman'}, default='pearson'
            Correlation method to use
        - compute_significance : bool, default=False
            Whether to compute statistical significance
        - threshold : float or None, default=None
            Threshold for correlation values
        - fisher_transform : bool, default=True
            Whether to apply Fisher z-transform when averaging

    Returns
    -------
    GroupFC
        Fitted GroupFC model that provides both group-level results and
        access to individual subject results.

    See Also
    --------
    fc : FC for single subject
    group_swc : Group-level dynamic functional connectivity
    """
    return GroupFC(**kwargs).fit(data)
