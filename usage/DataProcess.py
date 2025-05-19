import numpy as np
import matplotlib.pyplot as plt
from fluoromind import bandpass, debleaching, gsr
from skimage.filters import threshold_otsu
from sklearn.linear_model import LinearRegression
from DataView import *

def MeanSignals(*signals, labels=None, title='Mean Signal Over Time'):
    plt.figure(figsize=(10, 4))

    for i, signal in enumerate(signals):
        mean_signal = np.mean(signal, axis=1)
        label = labels[i] if labels and i < len(labels) else f'Signal {i+1}'
        plt.plot(mean_signal, label=label, alpha=0.7)

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Mean Signal')
    plt.legend()
    plt.grid(True)
    plt.show()

def PixelSignals(signal_pixel_pairs, labels=None, title=None):
    plt.figure(figsize=(12, 5))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (X, pix_idx) in enumerate(signal_pixel_pairs):
        if pix_idx >= X.shape[1]:
            print(f"Warning: pixel index {pix_idx} out of range for signal {i}")
            continue
        
        signal = X[:, pix_idx]
        f20 = float(np.percentile(signal, 20))
        label = labels[i] if labels and i < len(labels) else f'Signal{i+1} - Pixel {pix_idx}'
        color = color_cycle[i % len(color_cycle)]
        plt.plot(signal, label=label, color=color, alpha=0.8)
        plt.axhline(f20, linestyle='--', color=color, alpha=0.5)

    plt.xlabel('Time')
    plt.ylabel('Signal Intensity')
    plt.title(title or 'Custom Pixel Signal Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def NormalIntensity(X):
    f0 = np.percentile(X, 20, axis=0)
    f80 = np.percentile(X, 80, axis=0)

    pixel_index = 100
    pixel_signal = X[:, pixel_index]

    print("\n像素", pixel_index, "的 20 分位数:", f0[pixel_index])
    print("像素", pixel_index, "的 80 分位数:", f80[pixel_index])
    print("最大值:", pixel_signal.max(), "时间点:", pixel_signal.argmax())
    print("最小值:", pixel_signal.min(), "时间点:", pixel_signal.argmin(),"\n")

    X_normalized = (X - f0) / f0  
    pixel_signal = X_normalized[:, pixel_index]

    print("归一化后像素", pixel_index, "信号:", X_normalized[:, pixel_index])
    print("归一化最大值:", pixel_signal.max(), "时间点:", pixel_signal.argmax())
    print("归一化最小值:", pixel_signal.min(), "时间点:", pixel_signal.argmin(),"\n")

    return X_normalized

def datapre(X):
    X_concat = np.concatenate(X, axis=2)
    print(X_concat.shape)
    thresh = threshold_otsu(X_concat)
    valid_pixel_mask = (X_concat > thresh).any(axis=2)
    print(f"[筛选后] 有效像素数: {np.sum(valid_pixel_mask)}")
    print(valid_pixel_mask.shape)

    X_transposed = np.transpose(X_concat, (2, 0, 1)) 
    #X_reshaped = X_transposed.reshape(X_transposed.shape[0], -1)
    X_reshaped =  X_transposed[:, valid_pixel_mask]
    print("合并后形状:", X_reshaped.shape)
    X_corrected = debleaching(X_reshaped)
    

    #X_gsr = gsr(X_corrected)  #[gsr(X) for X in X_corrected]
    #X_filtered =  bandpass(X_gsr, low=0.1, high=14.5, fs=fs)  #[bandpass(X, low=0.1, high=14.5, fs=fs) for X in X_gsr]

    X_normalized =  NormalIntensity(X_corrected)
    X_final = np.full((X_normalized.shape[0], height, width), np.inf)
    X_final[:, valid_pixel_mask] = X_normalized
    
    print(X_final.shape)
    visualize(X_final, 100, -1)
    PixelSignals([(X_corrected,100), (X_reshaped,100)], ['Corrected 100', "Reshaped 100"])
    
    return X_final

if __name__ == "__main__":

    file_paths = [
        'E:/GithubData/Cortical-Image-Reproduction/fluorodata/Adultvglut21pre1.npy',
        'E:/GithubData/Cortical-Image-Reproduction/fluorodata/Adultvglut21pre2.npy',
    ]

    jointed = []
    try:
        jointed, height, width = load_data(file_paths)
        print(f"\n成功加载 {len(jointed)} 个文件")
    except Exception as e:
        print(f"加载数据时出错: {e}")
    
    datapre(jointed)

# 没有把mask的部分剔除掉导致存在0,计算出现错误