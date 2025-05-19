"""Preprocessing module for functional imaging data.

This module provides functions for preprocessing time series data,
including filtering, denoising, and signal correction.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from scipy import signal
from functools import lru_cache


@lru_cache
def _get_sos_coefficients(low: float, high: float, fs: float, cheby_order: int, cheby_ripple: float) -> NDArray:
    """
    Calculate second-order sections (SOS) coefficients for Chebyshev Type I bandpass filter.

    Parameters
    ----------
    low : float
        Lower cutoff frequency in Hz
    high : float
        Upper cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
    cheby_order : int
        Order of the Chebyshev filter
    cheby_ripple : float
        Maximum ripple allowed in the passband (dB)

    Returns
    -------
    NDArray
        Second-order sections representation of the IIR filter
    """
    nyq = 0.5 * fs
    low_normalized = low / nyq
    high_normalized = high / nyq
    return signal.cheby1(cheby_order, cheby_ripple, [low_normalized, high_normalized], btype="bandpass", output="sos")


def gsr(X: NDArray) -> NDArray:
    """
    Perform Global Signal Regression (GSR) on input data.

    Removes global signal by fitting a linear regression model to the mean signal
    and subtracting the predicted values from the original data.

    Parameters
    ----------
    X : NDArray
        Input data array of shape (n_samples, n_features)

    Returns
    -------
    NDArray
        Data array with global signal removed, same shape as input
    """
    indices = np.arange(X.shape[0]).reshape(-1, 1)
    vmean = X.mean(axis=1, keepdims=True)
    model = LinearRegression()
    model.fit(indices, vmean)
    y_pred = model.predict(indices)
    return X - y_pred


def bandpass(
    data: NDArray, low: float, high: float, fs: float, cheby_order: int = 4, cheby_ripple: float = 0.1
) -> NDArray:
    """
    Apply a Chebyshev Type I bandpass filter to the input data.

    Parameters
    ----------
    data : NDArray
        Input signal to be filtered
    low : float
        Lower cutoff frequency in Hz
    high : float
        Upper cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
    cheby_order : int, optional
        Order of the Chebyshev filter (default=4)
    cheby_ripple : float, optional
        Maximum ripple allowed in the passband in dB (default=0.1)

    Returns
    -------
    NDArray
        Filtered signal
    """
    sos = _get_sos_coefficients(low, high, fs, cheby_order, cheby_ripple)
    return signal.sosfiltfilt(sos, data, axis=1)


def debleaching(X: NDArray) -> NDArray:
    """
    Remove photobleaching effects from fluorescence data.

    Fits a linear regression model to predict the mean signal and
    subtracts the predicted values to correct for photobleaching decay.

    Parameters
    ----------
    X : NDArray
        Input fluorescence data array of shape (n_samples, n_features)

    Returns
    -------
    NDArray
        Corrected fluorescence data with photobleaching effects removed
    """
    # 时间序列索引
    time_indices = np.arange(X.shape[0]).reshape(-1, 1)
    
    # 计算每个时间点的平均信号
    vmean = X.mean(axis=1, keepdims=True)
    
    # 使用线性回归模型拟合时间与平均信号的关系
    model = LinearRegression()
    model.fit(time_indices, vmean)

    # 预测每个时间点的信号衰减趋势
    y_pred = model.predict(time_indices)

    # 计算衰减因子
    decay_factor = y_pred / y_pred[0]

    # 将每个数据点除以对应的衰减因子进行校正
    return X / decay_factor
