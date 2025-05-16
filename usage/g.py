import numpy as np
import matplotlib.pyplot as plt
from fluoromind.group.cpca import GroupCPCA
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import os
from skimage.filters import threshold_otsu
from scipy.signal import get_window
from collections import defaultdict
from numpy.lib.stride_tricks import sliding_window_view
import gc
from scipy.optimize import curve_fit

def normalize_power_unit(power):
    power = np.atleast_1d(power)
    if power.ndim == 1:
        total = np.sum(power)
        return power / total if total > 0 else power
    totals = np.sum(power, axis=1, keepdims=True)
    return np.where(totals > 0, power / totals, power)

def compute_psd_using_stft(signal, fs, nperseg=1024, noverlap=768, verbose=False):
    signal = np.asarray(signal).flatten()
    step = nperseg - noverlap
    if (len(signal) - noverlap) // step <= 0:
        return np.array([]), np.array([])

    frames = sliding_window_view(signal, window_shape=nperseg)[::step].copy()
    if frames.shape[0] == 0:
        return np.array([]), np.array([])

    window = get_window("hann", nperseg, fftbins=True)
    frames *= window

    fft_result = np.fft.fft(frames, axis=1)
    psd_matrix = (np.abs(fft_result) ** 2) / fs  # 不再除以窗能量

    psd = np.mean(psd_matrix, axis=0)
    freqs = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1/fs))
    psd = np.fft.fftshift(psd)

    # 归一化为 Unit-energy PSD
    psd = normalize_power_unit(psd)

    if verbose:
        print(f"[debug] PSD max: {np.max(psd):.4e}, mean: {np.mean(psd):.4e}")

    return freqs, psd

def run_cpca_on_whole(concatenated, fs, n_components, selected_indices):
    rep_f0, rep_zeta = defaultdict(list), defaultdict(list)
    psd_all = {idx: [] for idx in selected_indices}
    freqs = None

    # 提取更多主成分，以便选择解释方差最高的
    group_cpca = GroupCPCA(n_components=30)
    group_cpca.fit([concatenated])
    components = group_cpca.components_
    variances = group_cpca.explained_variance_ratio_

    component_info = []

    for idx, (var, signal) in enumerate(zip(variances, components)):
        freqs, psd = compute_psd_using_stft(signal, fs, verbose=False)
        if psd.size == 0:
            continue
        peakness = np.max(psd) / (np.mean(psd) + 1e-10)
        component_info.append((var, peakness, idx, signal, psd))

    # 按解释方差从高到低排序
    top_components = sorted(component_info, key=lambda x: -x[0])[:n_components]

    for i_true, (var_ratio, pk, original_idx, signal, psd) in enumerate(top_components):
        f0, zeta = compute_natural_frequency_and_damping(freqs, psd, verbose=False)
        rep_f0[i_true].append(f0)
        rep_zeta[i_true].append(zeta)

        if i_true in selected_indices:
            psd_all[i_true].append(psd)

        print(f"Top-{i_true} φ variance = {var_ratio:.4f}, original φ{original_idx}, peakness = {pk:.3f}, f₀ = {f0:.3f} Hz, ζ = {zeta:.3f}")

    return rep_f0, rep_zeta, psd_all, freqs


def lorentzian(f, f0, gamma, A):
    return A / (1 + ((f - f0) / gamma) ** 2)

def compute_natural_frequency_and_damping(freqs, psd, verbose=False):

    # 只保留正频率
    mid = len(freqs) // 2
    f = freqs[mid:]
    p = psd[mid:]

    if len(f) == 0 or np.max(p) == 0:
        return np.nan, np.nan

    # 限制拟合频段
    freq_mask = (f > 0.2) & (f < 0.5)
    f_fit = f[freq_mask]
    p_fit = p[freq_mask]

    if len(f_fit) == 0 or np.max(p_fit) == 0:
        return np.nan, np.nan

    # 初值设置
    f0_init = f_fit[np.argmax(p_fit)]
    gamma_init = 0.1 * f0_init
    A_init = np.max(p_fit)

    try:
        popt, _ = curve_fit(
            lorentzian, f_fit, p_fit,
            p0=[f0_init, gamma_init, A_init],
            bounds=([0.05, 1e-3, 0], [f_fit[-1], f_fit[-1]*2, 10*A_init]),
            maxfev=8000
        )
        f0, gamma, _ = popt
        zeta = gamma / f0 if f0 > 0 else np.nan

        # 加入有效性判断：太小的 zeta 一般拟合不准
        if zeta < 0.005 or zeta > 5:
            if verbose:
                print(f"⚠️ zeta={zeta:.4f} out of range, discard")
            return f0, np.nan

        if verbose:
            print(f"[Lorentz] f0 = {f0:.3f} Hz, ζ = {zeta:.4f}, γ = {gamma:.4f}")
        return f0, zeta

    except Exception as e:
        if verbose:
            print(f"⚠️ Lorentzian 拟合失败: {e}")
        return np.nan, np.nan


def plot_right_f0_zeta(rep_f0, rep_zeta, save_path):
    n_components = len(rep_f0)
    x = np.arange(n_components)
    num_trials = len(next(iter(rep_f0.values())))

    # Filter out NaN values before calculations
    all_f0 = [
        [val for val in rep_f0[i] if not np.isnan(val)]
        for i in range(n_components)
    ]
    all_zeta = [
        [val for val in rep_zeta[i] if not np.isnan(val)]
        for i in range(n_components)
    ]

    # Only plot components with valid data
    valid_components = [i for i in range(n_components)
                       if len(all_f0[i]) > 0 and len(all_zeta[i]) > 0]
    if not valid_components:
        print("Warning: No valid components found to plot")
        return

    mean_f0 = np.array([np.mean(all_f0[i]) if len(all_f0[i]) > 0 else np.nan
                       for i in range(n_components)])
    se_f0 = np.array([np.std(all_f0[i])/np.sqrt(len(all_f0[i])) if len(all_f0[i]) > 0 else 0
                       for i in range(n_components)])
    mean_zeta = np.array([np.mean(all_zeta[i]) if len(all_zeta[i]) > 0 else np.nan
                         for i in range(n_components)])
    se_zeta = np.array([np.std(all_zeta[i])/np.sqrt(len(all_zeta[i])) if len(all_zeta[i]) > 0 else 0
                         for i in range(n_components)])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    for i in range(n_components):
        ax1.scatter([x[i]] * len(rep_f0[i]), rep_f0[i], color='black', s=12)
        ax2.scatter([x[i]] * len(rep_zeta[i]), rep_zeta[i], color='black', s=12)
    ax1.errorbar(x, mean_f0, yerr=se_f0, fmt='o', color='orange')
    ax2.errorbar(x, mean_zeta, yerr=se_zeta, fmt='o', color='orange')
    ax1.plot(x, mean_f0, '--', color='orange')
    ax2.plot(x, mean_zeta, '--', color='orange')
    ax1.set_ylabel("Natural frequency (Hz)")
    ax2.set_ylabel("Damping ratio ($\\xi$)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"$\\varphi_{{{i}}}$" for i in x])
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_ylim(0, 6.0)
    ax2.set_ylim(0, 0.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600)
    plt.close()

def plot_left_psd(psd_all, freqs, selected_indices, save_path):
    fig, axes = plt.subplots(len(selected_indices), 1, figsize=(6, 5), sharex=True)

    for i, idx in enumerate(selected_indices):
        arr = np.array(psd_all[idx])
        if arr.size == 0:
            continue

        mean_psd = np.mean(arr, axis=0)
        std_psd = np.std(arr, axis=0)

        axes[i].fill_between(freqs, mean_psd - std_psd, mean_psd + std_psd,
                             color='sandybrown', alpha=0.5)

        # 蓝色线：选择某一次实验（第一个或中位数或最大）
        raw_psd = arr[np.argmax(arr.max(axis=1))]  # 选主峰最大那次
        axes[i].plot(freqs, raw_psd, color='steelblue', linewidth=1.2)
        # 蓝色线：直接画平均 PSD（不是某次实验）
        # axes[i].plot(freqs, mean_psd, color='steelblue', linewidth=1.5)

        axes[i].set_ylabel(f"$\\varphi_{{{idx}}}$", rotation=0, labelpad=20)
        axes[i].set_xlim([0.1, 10])
        axes[i].set_ylim(0, 0.1)

        axes[i].set_xscale("log")
        axes[i].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

    axes[-1].set_xlabel("Frequency (Hz)")
    axes[0].set_title("Unit-energy PSDs of selected $\\varphi$")
    fig.text(0.03, 0.5, "Power", va="center", rotation="vertical", fontsize=12)

    plt.tight_layout(rect=[0.05, 0.03, 1, 0.97])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600)
    plt.close()

def generate_mouse_groups(n_mice: int, group_size: int) -> list:
    return [list(range(i, i + group_size)) for i in range(n_mice - group_size + 1)]

def main():
    fs = 30.0
    n_components = 10
    group_size = 1
    selected_indices = [0, 1, 3, 5]
    chunk_size = 5400
    save_dir = "F:/college/科研/斑马鱼（张志鹏&史良）/fluorodata/"

    files = [
        'F:/college/科研/斑马鱼（张志鹏&史良）/fluorodata/Adultvglut21pre1.npy',
    ]

    X_list = [np.load(f) for f in files]
    X_selected = [X[:, :, :1800].reshape(-1, 1800).T for X in X_list]
    X_concat = np.concatenate(X_selected, axis=0)
    thresh = threshold_otsu(X_concat)
    valid_pixel_mask = (X_concat > thresh).any(axis=0)
    print(f"[筛选后] 有效像素数: {np.sum(valid_pixel_mask)}")

    group_indices_list = generate_mouse_groups(len(X_list), group_size)
    rep_f0, rep_zeta = defaultdict(list), defaultdict(list)
    psd_all = {idx: [] for idx in selected_indices}

    for indices in tqdm(group_indices_list, desc="Running trials"):
        sampled = [X_list[i][:, :, :1800].reshape(-1, 1800).T[:, valid_pixel_mask] for i in indices]
        concatenated = np.concatenate(sampled, axis=0)
        rep_f0_chunk, rep_zeta_chunk, psd_all_chunk, freqs = run_cpca_on_whole(
            concatenated, fs, n_components, selected_indices
        )

        for k in rep_f0_chunk:
            rep_f0[k].extend(rep_f0_chunk[k])
            rep_zeta[k].extend(rep_zeta_chunk[k])
        for k in psd_all_chunk:
            psd_all[k].extend(psd_all_chunk[k])

        # 清理不再需要的变量释放内存
        del sampled, concatenated, rep_f0_chunk, rep_zeta_chunk, psd_all_chunk
        gc.collect()

    #print(f"rep_zeta的内容: {rep_zeta}")

    plot_right_f0_zeta(rep_f0, rep_zeta, os.path.join(save_dir, "right_f0_zeta.png"))
    plot_left_psd(psd_all, freqs, selected_indices, os.path.join(save_dir, "left_PSDs.png"))

if __name__ == "__main__":
    main()
