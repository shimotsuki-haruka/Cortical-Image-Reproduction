import numpy as np
from fluoromind import bandpass, debleaching, gsr
from skimage.filters import threshold_otsu
from sklearn.linear_model import LinearRegression
from DataView import *

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
    X_corrected = debleaching(X_reshaped)
    print("合并后形状:", X_reshaped.shape)

    #X_gsr = gsr(X_corrected)  #[gsr(X) for X in X_corrected]
    #X_filtered =  bandpass(X_gsr, low=0.1, high=14.5, fs=fs)  #[bandpass(X, low=0.1, high=14.5, fs=fs) for X in X_gsr]

    X_final = np.zeros((X_corrected.shape[0], height, width))
    X_final[:, valid_pixel_mask] = X_corrected
    print(X_final.shape)
    visualize(X_final, 100, -1)

    avg_original = X_reshaped.mean(axis=1)
    avg_corrected = X_corrected.mean(axis=1)
    plt.figure(figsize=(10, 4))
    plt.plot(avg_original, label='Original Avg', alpha=0.7)
    plt.plot(avg_corrected, label='Corrected Avg', alpha=0.7)
    plt.title('Mean Signal Over Time (All Valid Pixels)')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.grid(True)
    plt.show()

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