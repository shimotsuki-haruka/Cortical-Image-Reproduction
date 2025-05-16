import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from scipy.signal import spectrogram, get_window
from skimage.filters import threshold_otsu
import matplotlib.ticker as ticker

def load_data(files):
    X_list = [np.load(f) for f in files]
    height, width, _ = X_list[0].shape
    return X_list, height, width


def auto_choose_nperseg(signal_len, fs, min_cycles=3, max_windows=100):
    min_freq = 0.3
    min_window = int(fs * min_cycles / min_freq)
    max_window = signal_len // max_windows
    nperseg = min(max(min_window, 16), max_window, signal_len)
    noverlap = nperseg // 2
    return nperseg, noverlap


def plot_pixel_spectrogram(signal, fs=30, nperseg=64, noverlap=32, title=None):
    window = get_window('hann', nperseg)
    f, t, Sxx = spectrogram(signal, fs=fs, window=window,
                            nperseg=nperseg, noverlap=noverlap, scaling='density')
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    vmin = np.percentile(Sxx_db, 10)
    vmax = np.percentile(Sxx_db, 90)

    fig, ax = plt.subplots(figsize=(10, 5))
    c = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='inferno', vmin=vmin, vmax=vmax)

    ax.set_ylim(0, 15)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title(title if title else 'Spectrogram of Selected Pixel', fontsize=14)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    plt.colorbar(c, ax=ax, label='Power (dB)')
    plt.tight_layout()
    plt.show()

def on_click(event, X_concat, height, width, fs):
    if event.inaxes:
        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
        idx = y * width + x
        print(f"[点击像素] 坐标: ({x}, {y}) → 索引: {idx}")
        pixel_signal = X_concat[:, idx]
        nperseg, noverlap = auto_choose_nperseg(len(pixel_signal), fs)
        title = f"Spectrogram of Pixel ({x}, {y}) — Index {idx}"
        plot_pixel_spectrogram(pixel_signal, fs, nperseg, noverlap, title)


def main():
    fs = 30  # 采样率
    files = [
        'F:/college/科研/斑马鱼（张志鹏&史良）/fluorodata/Adultvglut21pre1.npy',
    ]
    X_list, height, width = load_data(files)
    X_selected = [X[:, :, :1800].reshape(-1, 1800).T for X in X_list]
    X_concat = np.concatenate(X_selected, axis=0)  # shape = (time, pixels)

    # 筛选有效像素
    thresh = threshold_otsu(X_concat)
    valid_pixel_mask = (X_concat > thresh).any(axis=0)
    # X_concat = X_concat[:, valid_pixel_mask]  # shape = (time, valid_pixels)
    print(f"[筛选后] 有效像素数: {np.sum(valid_pixel_mask)}")

    # 建立 full_map 以恢复 pixel index → (x, y) 映射
    full_map = np.arange(height * width).reshape(height, width)
    valid_idx_map = np.full(height * width, -1)
    valid_idx_map[valid_pixel_mask] = np.arange(np.sum(valid_pixel_mask))

    # 显示第一帧图像（可点击）
    fig, ax = plt.subplots(figsize=(6, 6))
    first_frame = X_list[0][:, :, 0]
    img = ax.imshow(first_frame, cmap='gray')
    cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
    ax.set_title("Click on the pixel to view the time-frequency graph")
    fig.canvas.mpl_connect('button_press_event',
        lambda event: on_click(event, X_concat, height, width, fs)
    )
    plt.show()


if __name__ == '__main__':
    main()

'''
def plot_pixel_spectrogram(pixel_signal, fs=30, nperseg=64, noverlap=32, save_path=None):
    """
    绘制单个像素在指定时间段的高质量时频图

    参数:
        pixel_signal: 1D array,像素时间序列
        fs: 采样频率 (Hz)
        nperseg: 每段长度
        noverlap: 重叠长度
        save_path: 如果提供路径，则保存图像；否则直接展示
    """
    # 计算 spectrogram
    window = get_window('hann', nperseg)
    f, t, Sxx = spectrogram(pixel_signal, fs=fs, window=window,
                            nperseg=nperseg, noverlap=noverlap, scaling='density')

    # 转换为 dBFS（不归一化，保留真实幅值）
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    # 自动设置 vmin/vmax（避免亮/暗爆）
    vmin = np.percentile(Sxx_db, 10)
    vmax = np.percentile(Sxx_db, 90)

    print(f"[dB Range] min = {np.min(Sxx_db):.2f}, max = {np.max(Sxx_db):.2f}, vmin = {vmin:.2f}, vmax = {vmax:.2f}")

    # 开始绘图
    fig, ax = plt.subplots(figsize=(10, 5))
    c = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='inferno', vmin=vmin, vmax=vmax)

    # 坐标轴设置
    ax.set_ylim(0, 10)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title('Spectrogram of a Single Pixel', fontsize=13, weight='bold')

    # 坐标刻度美化
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(20))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.tick_params(labelsize=11)

    # Colorbar 设置
    cbar = plt.colorbar(c, ax=ax, pad=0.03)
    cbar.set_label('Power (dB)', fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    # 右侧和上方辅助坐标轴（可选）
    ax_secondary_x = ax.secondary_xaxis('top')
    ax_secondary_x.set_xticks(ax.get_xticks())
    ax_secondary_x.set_xticklabels([f"{int(v)}" for v in ax.get_xticks()])
    ax_secondary_y = ax.secondary_yaxis('right')
    ax_secondary_y.set_yticks(ax.get_yticks())
    ax_secondary_y.set_yticklabels([f"{v:.0f}" for v in ax.get_yticks()])

    # 导出或展示
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    else:
        plt.show()

def find_most_active_pixel(X_concat):
    stds = np.std(X_concat, axis=0)
    idx = np.argmax(stds)
    return idx

def plot_pixel_spectrogram_by_index_and_time(X_concat, pixel_idx, fs, time_range=None, nperseg=128, noverlap=64):
    """
    选择某个像素在某个时间段的信号，绘制其时频图。
    :param X_concat: shape = (time_points, pixels)
    :param pixel_idx: 要分析的像素索引
    :param fs: 采样率
    :param time_range: 元组，如 (10, 20)，单位为秒；若为 None则取全部时间段
    """
    signal = X_concat[:, pixel_idx]

    # 截取指定时间段
    if time_range is not None:
        start_idx = int(time_range[0] * fs)
        end_idx = int(time_range[1] * fs)
        signal = signal[start_idx:end_idx]

    plot_pixel_spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

def auto_choose_nperseg(signal_len, fs, min_cycles=3, max_windows=100):
    """
    根据信号长度和采样率自动选择合理的 nperseg 和 noverlap。

    参数:
        signal_len: 信号总采样点数
        min_cycles: 至少包含的最小振荡周期数（默认 3 个）
        max_windows: 最大时频窗数量（默认不超过 100 个）

    返回:
        (nperseg, noverlap)
    """
    # 最小窗口长度：至少包含若干个周期（例如 0.3Hz → 10s → 300点）
    min_freq = 0.3  # 可根据实际修改
    min_window = int(fs * min_cycles / min_freq)

    # 最大窗口长度：不能太大，最多分成多少个窗口
    max_window = signal_len // max_windows

    # 理想窗口长度
    nperseg = min(max(min_window, 16), max_window, signal_len)

    # 重叠设为 50%
    noverlap = nperseg // 2

    print(f"[Auto config] signal_len={signal_len}, fs={fs} → nperseg={nperseg}, noverlap={noverlap}")
    return nperseg, noverlap


def main():
    # 加载并拼接数据
    files = [
        'E:/demo_fm/gssidnaim-main/demodata/adult_vglut2/Adultvglut21pre1.npy',
        'E:/demo_fm/gssidnaim-main/demodata/adult_vglut2/Adultvglut21pre2.npy',
        'E:/demo_fm/gssidnaim-main/demodata/adult_vglut2/Adultvglut21pre3.npy',
        'E:/demo_fm/gssidnaim-main/demodata/adult_vglut2/Adultvglut22pre1.npy',
        'E:/demo_fm/gssidnaim-main/demodata/adult_vglut2/Adultvglut22pre2.npy',
        'E:/demo_fm/gssidnaim-main/demodata/adult_vglut2/Adultvglut23pre1.npy',
        'E:/demo_fm/gssidnaim-main/demodata/adult_vglut2/Adultvglut23pre2.npy',
        'E:/demo_fm/gssidnaim-main/demodata/adult_vglut2/Adultvglut24pre1.npy'
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
    # plot_psd_from_data(X_concat, fs=30.0, nperseg=128)

    # 找到最活跃像素
    pixel_idx = find_most_active_pixel(X_filtered)
    pixel_signal = X_filtered[:, pixel_idx]  # 正确提取像素信号
    nperseg, noverlap = auto_choose_nperseg(len(X_filtered), fs=30)

    plot_pixel_spectrogram_by_index_and_time(
        X_concat,
        pixel_idx=pixel_idx,
        fs=30,
        time_range=(0, 480),
        nperseg=nperseg,
        noverlap=noverlap
    )
'''