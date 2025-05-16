#Average power spectral density

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from fluoromind.group.cpca import GroupCPCA
from scipy.signal import welch
from scipy.signal import spectrogram, get_window
import matplotlib.ticker as ticker
from skimage.filters import threshold_otsu

def normalize_power_unit(power):
    # Ensure input is at least 1D array
    power = np.atleast_1d(power)
    # For 1D input (single component), normalize across frequencies
    if power.ndim == 1:
        total = np.sum(power)
        return power / total if total != 0 else power
    # For 2D input (multiple components), normalize each component separately
    totals = np.sum(power, axis=1, keepdims=True)
    return np.where(totals != 0, power / totals, power)

def plot_psd_from_data(X_concat, fs, nperseg):
    """
    对拼接后的数据计算每个像素的PSD,并绘制平均功率谱密度图
    """
    n_pixels = X_concat.shape[1]
    all_psd = []

    for pixel_idx in range(n_pixels):
        pixel_signal = X_concat[:, pixel_idx]
        f, psd = welch(pixel_signal, fs=fs, nperseg=nperseg)
        all_psd.append(psd)

    all_psd = np.array(all_psd)
    norm_psd_unit = normalize_power_unit(all_psd)

    # 绘图
    plt.figure(figsize=(14, 4))
    mean_psd = np.mean(norm_psd_unit, axis=0)
    sem_psd = np.std(norm_psd_unit, axis=0) / np.sqrt(norm_psd_unit.shape[0])
    sem_psd_expanded = sem_psd * 15

    plt.plot(f, mean_psd, color='blue', linewidth=2, label='Mean PSD')
    plt.fill_between(f, mean_psd - sem_psd_expanded, mean_psd + sem_psd_expanded,
                     color='orange', alpha=0.3, label='±SEM')

    plt.title('Power Spectral Density (0.1-10Hz)', fontsize=14)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Power', fontsize=12)
    plt.xlim(0.1, 10)
    plt.ylim(0.0, 0.25)
    plt.xscale('log')
    plt.yscale('linear')

    # 美化
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(False)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=0.2, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=0.1, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=10, color='black', linestyle='--', linewidth=1)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def main():
    # 加载并拼接数据
    files = [
        'F:/college/科研/斑马鱼（张志鹏&史良）/fluorodata/Adultvglut21pre1.npy',
    ]
    X_list = [np.load(f) for f in files]

    # 择前1800帧数据并 reshape 为 2D：每一行为时间点，每一列为像素点
    X_selected = [X[:, :, :1800].reshape(-1, 1800).T for X in X_list]  # (time_points, height*width)

    # 拼接所有小鼠数据
    X_concat = np.concatenate(X_selected, axis=0)  # 拼接后的数据形状为 (8 * 1800, height*width)

    # 基于大津法计算阈值
    global_thresh = threshold_otsu(X_concat)
    # 创建掩码（像素维度上）：找出至少在某个时间点上激活过的像素
    signal_mask = X_concat > global_thresh
    valid_pixel_mask = signal_mask.any(axis=0)   # shape: (pixels,)
    print("有效像素数量:", np.sum(valid_pixel_mask))
    print(global_thresh)  # 输出阈值

    # 提取有效像素对应的数据
    X_filtered = X_concat[:, valid_pixel_mask]     # shape: (time_points, n_valid_pixels)
    print("过滤后的数据形状:", X_filtered.shape)

    # 计算并绘制功率谱图
    plot_psd_from_data(X_concat, fs=30.0, nperseg=128)

if __name__ == '__main__':
    main()

