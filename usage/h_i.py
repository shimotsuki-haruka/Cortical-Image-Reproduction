import numpy as np
import matplotlib.pyplot as plt
from fluoromind.group.cpca import GroupCPCA
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import os
from skimage.filters import threshold_otsu
from scipy.signal import get_window
from numpy.linalg import eigvals
from collections import defaultdict
from numpy.lib.stride_tricks import sliding_window_view
import gc

def run_cpca_for_h_i(concatenated, chunk_size, n_components):
    ranked_components = []

    for start in range(0, concatenated.shape[0], chunk_size):
        end = min(start + chunk_size, concatenated.shape[0])
        chunk = concatenated[start:end]

        cpca = GroupCPCA(n_components=10)
        cpca.fit([chunk])

        for comp, var_ratio in zip(cpca.components_, cpca.explained_variance_ratio_):
            energy = np.sum(np.abs(comp) ** 2)
            ranked_components.append((energy, comp, var_ratio))  # 合并打包

    # 根据能量排序并选出前 n_components 个
    ranked_components.sort(key=lambda x: -x[2])   # 按能量降序排0,2是解释方差
    top_components = [item[1] for item in ranked_components[:n_components]]
    top_var_ratios = [item[2] for item in ranked_components[:n_components]]

    # 相位方差计算
    phase_var = [np.var(np.angle(c)) for c in top_components]

    return top_var_ratios, phase_var

def plot_variance_explained(subject_explained_ratio_all, save_path):
    data = np.array(subject_explained_ratio_all) * 100  # 转换为百分比
    mean = np.mean(data, axis=0)
    sem = np.std(data, axis=0) / np.sqrt(data.shape[0])
    x = np.arange(data.shape[1])

    plt.figure(figsize=(6, 4))
    plt.bar(x, mean, yerr=sem, color='skyblue', capsize=3)
    for i in range(data.shape[1]):
        plt.scatter([i] * len(data[:, i]), data[:, i], color='black', s=10)

    plt.xlabel("ϕ index")
    plt.ylabel("Variance explained (%)")
    plt.xticks(x, [f"$\\varphi_{{{i}}}$" for i in x])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_phase_variance(phase_var_all, save_path):
    data = np.array(phase_var_all)
    mean = np.mean(data, axis=0)
    sem = np.std(data, axis=0) / np.sqrt(data.shape[0])
    x = np.arange(data.shape[1])

    plt.figure(figsize=(6, 4))
    plt.bar(x, mean, yerr=sem, color='skyblue', capsize=3)
    for i in range(data.shape[1]):
        plt.scatter([i] * len(data[:, i]), data[:, i], color='black', s=10)

    plt.xlabel("ϕ index")
    plt.ylabel("Phase variance (rad²)")
    plt.xticks(x, [f"$\\varphi_{{{i}}}$" for i in x])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def generate_mouse_groups(n_mice: int, group_size: int) -> list:
    return [list(range(i, i + group_size)) for i in range(n_mice - group_size + 1)]

def main():
    fs = 30.0
    n_components = 10
    group_size = 1
    chunk_size = 5400
    save_dir = "F:/college/科研/斑马鱼（张志鹏&史良）/fluorodata/"

    files = [
         'F:/college/科研/斑马鱼（张志鹏&史良）/fluorodata/Adultvglut21pre1.npy',
    ]

    X_list = [np.load(f).astype(np.float32) for f in files]
    X_selected = [X[:, :, :1800].reshape(-1, 1800).T for X in X_list]
    X_concat = np.concatenate(X_selected, axis=0)
    thresh = threshold_otsu(X_concat)
    valid_pixel_mask = (X_concat > thresh).any(axis=0)
    print(f"[筛选后] 有效像素数: {np.sum(valid_pixel_mask)}")

    group_indices_list = generate_mouse_groups(len(X_list), group_size)
    subject_explained_ratio_all = []
    phase_var_all = []

    for indices in tqdm(group_indices_list, desc="Running h & i图数据"):
        sampled = [X_list[i][:, :, :1800].reshape(-1, 1800).T[:, valid_pixel_mask] for i in indices]
        concatenated = np.concatenate(sampled, axis=0)

        var_ratio, phase_var = run_cpca_for_h_i(concatenated, chunk_size, n_components)
        subject_explained_ratio_all.append(var_ratio)
        phase_var_all.append(phase_var)

        del sampled, concatenated, var_ratio, phase_var
        gc.collect()

    plot_variance_explained(subject_explained_ratio_all, os.path.join(save_dir, "h.png"))
    plot_phase_variance(phase_var_all, os.path.join(save_dir, "i.png"))

if __name__ == "__main__":
    main()
