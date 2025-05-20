import numpy as np
import matplotlib.pyplot as plt
from fluoromind import bandpass, debleaching, gsr
from skimage.filters import threshold_otsu
from sklearn.linear_model import LinearRegression
from DataView import *

file_paths = [
        'E:/GithubData/Cortical-Image-Reproduction/fluorodata/Adultvglut21pre1.npy',
        'E:/GithubData/Cortical-Image-Reproduction/fluorodata/Adultvglut21pre2.npy',
    ]

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

    print("\npixel", pixel_index, "'s 20 percentile:", f0[pixel_index])
    print("pixel", pixel_index, "'s 80 percentile:", f80[pixel_index])
    print("Max:", pixel_signal.max(), "Time:", pixel_signal.argmax())
    print("Min:", pixel_signal.min(), "Time:", pixel_signal.argmin(),"\n")

    X_normalized = (X - f0) / f0  
    pixel_signal = X_normalized[:, pixel_index]

    print("Normalized pixel", pixel_index, "Signal:", X_normalized[:, pixel_index])
    print("Normalized max:", pixel_signal.max(), "Time:", pixel_signal.argmax())
    print("Normalized min:", pixel_signal.min(), "Time:", pixel_signal.argmin(),"\n")

    return X_normalized

def datapre(X):
    X_concat = np.concatenate(X, axis=2)
    print(X_concat.shape)
    thresh = threshold_otsu(X_concat)
    valid_pixel_mask = (X_concat > thresh).any(axis=2)
    print(f"Valid pixel count: {np.sum(valid_pixel_mask)}")
    print(valid_pixel_mask.shape)

    X_transposed = np.transpose(X_concat, (2, 0, 1)) 
    #X_reshaped = X_transposed.reshape(X_transposed.shape[0], -1)
    X_reshaped =  X_transposed[:, valid_pixel_mask]
    print("Merged shape:", X_reshaped.shape)
    X_corrected = debleaching(X_reshaped)
    

    X_normalized =  NormalIntensity(X_corrected)
    X_filtered =  bandpass(X_normalized, low=0.1, high=14.5, fs=30)
    X_final = np.full((X_filtered.shape[0], height, width), np.inf)
    X_final[:, valid_pixel_mask] = X_filtered
    
    print(X_final.shape)
    visualize(X_final, 100, -1)
    PixelSignals([(X_filtered, 100)], ['Filtered 100'])
    
    return np.transpose(X_final, (1, 2, 0)) 

if __name__ == "__main__":

    jointed = []
    try:
        jointed, height, width = load_data(file_paths)
        print(f"\nSuccessfully loaded {len(jointed)} file(s)")
    except Exception as e:
        print(f"Error: {e}")
    
    datapre(jointed)

# 没有把mask的部分剔除掉导致存在0,计算出现错误