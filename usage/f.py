import numpy as np
from fluoromind.group.cpca import GroupCPCA
from fluoromind.group.pca import GroupPCA
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.signal import savgol_filter
from scipy.signal import correlate, welch
import matplotlib.ticker as ticker

def perform_time_frequency_analysis(X_concat, chunk_size=1800, fs=30.0, n_components=10, method='cpca'):
    """
    计算主成分
    :param X_concat: 数据，形状为 (time_points, height * width)
    :param chunk_size: 数据分块的大小（仅对 cpca 有效）
    :param fs: 采样频率
    :param n_components: 提取的主成分数量
    :param method: 使用的分析方法 ('cpca' 或 'pca')
    :return: 所有主成分和解释的方差比例
    """
    if method == 'pca':
        group_pca = GroupPCA(n_components=n_components)
        group_pca.fit([X_concat])  # 整体拟合
        scores = group_pca.pca_.transform(X_concat)  # shape: (time_points, n_components)
        components = scores.T  # shape: (n_components, time_points)
        explained_variance_ratio = group_pca.pca_.explained_variance_ratio_
        return components, explained_variance_ratio

    elif method == 'cpca':
        n_chunks = X_concat.shape[0] // chunk_size + (1 if X_concat.shape[0] % chunk_size != 0 else 0)
        all_components = []
        all_explained_variance_ratios = []

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, X_concat.shape[0])
            X_chunk = X_concat[start_idx:end_idx, :]
            print(f"Chunk {chunk_idx} shape = {X_chunk.shape}, rank = {np.linalg.matrix_rank(X_chunk)}")

            group_cpca = GroupCPCA(n_components=n_components)
            group_cpca.fit([X_chunk])
            components = group_cpca.components_
            explained_variance_ratio = group_cpca.explained_variance_ratio_

            all_components.append(components)
            all_explained_variance_ratios.append(explained_variance_ratio)

        all_components = np.concatenate(all_components, axis=0)
        all_explained_variance_ratios = np.concatenate(all_explained_variance_ratios, axis=0)
        return all_components, all_explained_variance_ratios

    else:
        raise ValueError("Method should be 'pca' or 'cpca'")


def plot_waveforms_and_autocorrelation(waveforms, max_lag=30):
    """绘制波形和自相关函数"""
    n_components = len(waveforms)
    fig, axes = plt.subplots(n_components, 2, figsize=(18, 10))
    fig.suptitle("Waveform and autocorrelation of the cpca signal", fontsize=16)

    for i, waveform in enumerate(waveforms):
        waveform = waveform[:900]  # 截取前900个数据点

        # 归一化或标准化信号
        waveform = (waveform - np.mean(waveform)) / np.std(waveform)

        # 计算自相关
        autocorr = correlate(waveform, waveform, mode='full', method='auto')
        autocorr = autocorr[len(autocorr)//2:]  # 只保留正时间延迟部分
        autocorr = savgol_filter(autocorr, window_length=17, polyorder=3)  # 对自相关进行平滑
        autocorr = autocorr / np.max(np.abs(autocorr))  # 归一化到[-1,1]
        autocorr = autocorr[:max_lag * 30]  # 限制自相关函数绘制的时间滞后范围
        time_lags = np.arange(len(autocorr)) / 30  # 时间单位为秒

        # 绘制时域信号（第一列）
        axes[i, 0].plot(np.arange(900) / 30, waveform)
        if i == 0:
            axes[i, 0].set_title("ϕi(t)")
        if i == len(waveforms) - 1:
            axes[i, 0].set_xlabel("Time (s)")
        axes[i, 0].set_ylabel(f'ϕ{i}')
        axes[i, 0].set_xlim([0, 30])
        axes[i, 0].xaxis.set_major_locator(ticker.MultipleLocator(5))  # 每隔5秒设置一个刻度
        waveform_max = np.max(waveform)
        waveform_min = np.min(waveform)
        axes[i, 0].set_ylim([waveform_min, waveform_max])
        axes[i, 0].grid(True)

        # 绘制自相关函数（第二列）
        axes[i, 1].plot(time_lags, autocorr)
        if i == 0:
            axes[i, 1].set_title("Autocorrelation")
        if i == len(waveforms) - 1:
            axes[i, 1].set_xlabel("Time lag (s)")
        axes[i, 1].set_ylabel(f'ϕ{i}')
        axes[i, 1].set_xlim([0, 30])
        axes[i, 1].xaxis.set_major_locator(ticker.MultipleLocator(5))  # 每隔5秒设置一个刻度
        axes[i, 1].set_ylim([-0.5, 1])
        axes[i, 1].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def rank_periodic_components(components, fs=30.0, max_lag_sec=10):
    """
    根据自相关周期性强度对主成分排序
    :param components: shape = (n_components, time_points)
    :param fs: 采样频率
    :param max_lag_sec: 自相关函数最大时间滞后（秒）
    :return: 排序后的主成分索引列表 和 周期性评分列表
    """
    max_lag = int(max_lag_sec * fs)
    periodic_scores = []

    for waveform in components:
        waveform = waveform[:900]
        waveform = (waveform - np.mean(waveform)) / np.std(waveform)
        autocorr = correlate(waveform, waveform, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = savgol_filter(autocorr, window_length=17, polyorder=3)
        autocorr = autocorr[:max_lag]
        autocorr = autocorr / np.max(np.abs(autocorr))
        score = np.mean(np.abs(autocorr[1:]))  # 排除0滞后的主峰
        periodic_scores.append(score)

    periodic_scores = np.array(periodic_scores)
    ranked_indices = np.argsort(periodic_scores)[::-1]

    ranked_scores = periodic_scores[ranked_indices]
    return ranked_indices, ranked_scores


def main(mode='cpca'):
    """
    mode: 分析模式，可选：
        - 'raw'   ：直接对原始数据进行时频分析（像素级）
        - 'pca'   ：对原始数据进行 PCA 后分析主成分的时频特征
        - 'cpca'  ：对原始数据进行 CPCA 后分析主成分的时频特征
    """
    fs = 30.0
    chunk_size = 1800
    n_components = 10

    if mode == 'raw':
        # 单文件：原始荧光堆栈数据
        file = 'E:/demo_fm/gssidnaim-main/demodata/raw_data/tif_images_stack.npy'
        X = np.load(file)  # shape: (height, width, time_points)
        X_selected = X.reshape(X.shape[2], -1).T  # (time_points, height*width)

        # 选择若干像素进行可视化
        selected_pixels = [0, 1, 2, 3]
        waveforms = [X_selected[:, pixel_idx] for pixel_idx in selected_pixels]
        plot_waveforms_and_autocorrelation(waveforms)

    else:
        # 多个处理文件
        files = [
            'F:/college/科研/斑马鱼（张志鹏&史良）/fluorodata/Adultvglut21pre1.npy',
        ]
        X_list = [np.load(f) for f in files]

        # 预处理并拼接
        X_selected = [X[:, :, :chunk_size].reshape(-1, chunk_size).T for X in X_list]
        X_concat = np.concatenate(X_selected, axis=0)  # shape: (time_points, pixels)

        #print(np.isnan(X_concat).sum())
        #print(np.isinf(X_concat).sum())
        #print(X_concat.shape)

        # 分析方法选择
        if mode == 'cpca':
            components, _ = perform_time_frequency_analysis(X_concat, fs=fs, chunk_size=chunk_size, method='cpca')
        elif mode == 'pca':
            components, _ = perform_time_frequency_analysis(X_concat, fs=fs, chunk_size=chunk_size, method='pca')
        else:
            raise ValueError("mode should be 'raw', 'pca', or 'cpca'")

        # 可视化主成分
        ranked_indices, scores = rank_periodic_components(components, fs=30)
        selected_components = ranked_indices[:4]  # 选择前4个周期性最强的主成分
        # selected_components = [0, 1, 3, 5]
        waveforms = [components[i] for i in selected_components]
        plot_waveforms_and_autocorrelation(waveforms)

if __name__ == "__main__":
    main()
