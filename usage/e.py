#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import welch
from skimage.filters import threshold_otsu
from fluoromind.group.cpca import GroupCPCA

def compute_psd(X, fs=30.0, nperseg=256):
    """计算主成分的功率谱密度 (PSD)"""
    # Adjust nperseg if input is too short
    if len(X) < nperseg:
        nperseg = len(X)
    f, Pxx = welch(X, fs=fs, nperseg=nperseg, return_onesided=False)
    return f, Pxx

def plot_frequency_band_power(X_cpca_scores, components, fs=30.0, height=128, width=128, n_bands=6, component_idx=0, valid_pixel_mask=None):
    """绘制指定频带的空间功率图"""
    freq_bands = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 5.0), (5.0, 14.0)]

    f, Pxx = compute_psd(X_cpca_scores[component_idx], fs=fs)

    fig, axes = plt.subplots(1, n_bands, figsize=(18, 4))

    for i, (f_min, f_max) in enumerate(freq_bands):
        band_idx = np.where((f >= f_min) & (f <= f_max))[0]
        band_power = np.sum(Pxx[band_idx])  # 频带功率（标量）

        # 将频带功率乘以对应主成分的空间分布得到空间图
        spatial_map = band_power * components[component_idx]  # shape = (valid_pixels, )

        # 归一化（单位能量归一化）
        norm = np.linalg.norm(spatial_map)
        if norm > 1e-10:  # Only normalize if norm is significant
            spatial_map = spatial_map / norm
        else:
            spatial_map = np.zeros_like(spatial_map)

        # 创建 full image 并填入有效像素
        full_map = np.zeros(height * width)
        full_map[valid_pixel_mask] = np.abs(spatial_map)  # Take absolute value
        full_map = full_map.reshape(height, width)

        im = axes[i].imshow(full_map, cmap='YlOrRd')
        axes[i].set_title(f"{f_min}-{f_max} Hz")
        axes[i].axis('off')

    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.015, pad=0.03)
    plt.subplots_adjust(wspace=0.05, hspace=0.1)  # Adjust spacing manually
    plt.show()

def perform_cpca(X_concat, n_components=10):
    """对拼接的数据进行主成分分析(PCA)"""
    cpca_model = GroupCPCA(n_components=n_components)
    cpca_model.fit([X_concat])
    X_pca = cpca_model.result_.scores_
    components = cpca_model.result_.components_
    return X_pca, components

def main():
    files = [
        'F:/college/科研/斑马鱼（张志鹏&史良）/fluorodata/Adultvglut21pre1.npy',
    ]
    X_list = [np.load(f) for f in files]

    height, width, _ = X_list[0].shape

    X_selected = [X[:, :, :1800].reshape(-1, 1800).T for X in X_list]
    X_concat = np.concatenate(X_selected, axis=0)

    thresh = threshold_otsu(X_concat)
    valid_pixel_mask = (X_concat > thresh).any(axis=0)
    X_concat = X_concat[:, valid_pixel_mask]
    print(f"[筛选后] 有效像素数: {np.sum(valid_pixel_mask)}")

    X_cpca, components = perform_cpca(X_concat, n_components=10)

    plot_frequency_band_power(X_cpca, components, fs=30.0, height=height, width=width,
                              component_idx=0, valid_pixel_mask=valid_pixel_mask)

if __name__ == "__main__":
    main()
