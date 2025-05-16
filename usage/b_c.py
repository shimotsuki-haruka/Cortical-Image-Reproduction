#CPCA analysis

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from fluoromind.group.cpca import GroupCPCA
from skimage.filters import threshold_otsu
plt.rcParams['text.usetex'] = False

def load_data(files):
    X_list = [np.load(f) for f in files]
    height, width, _ = X_list[0].shape
    return X_list, height, width

def run_cpca_analysis(X_masked_concat, n_components=10):
    chunk_size = 1800
    n_chunks = X_masked_concat.shape[0] // chunk_size + (1 if X_masked_concat.shape[0] % chunk_size != 0 else 0)

    all_components = []
    all_explained_variance_ratios = []

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, X_masked_concat.shape[0])
        X_chunk = X_masked_concat[start_idx:end_idx, :]

        group_cpca = GroupCPCA(n_components=n_components)
        group_cpca.fit([X_chunk])

        components = group_cpca.components_
        explained_variance_ratio = group_cpca.explained_variance_ratio_

        all_components.append(components)
        all_explained_variance_ratios.append(explained_variance_ratio)

    all_components = np.concatenate(all_components, axis=0)
    all_explained_variance_ratios = np.concatenate(all_explained_variance_ratios, axis=0)

    return all_components, all_explained_variance_ratios

def plot_wave_patterns(components, height, width, selected_components):
    fig, axes = plt.subplots(len(selected_components), 4, figsize=(12, 8))
    row_labels = [fr"r'$\Phi_{i}$'" for i in selected_components]
    col_labels = ["t = 0", r"$t = T/4$", r"$t = T/2$", r"$t = 3T/4$"]

    for i, comp_idx in enumerate(selected_components):
        comp_reshaped = np.resize(components[comp_idx], (height, width))
        for j in range(4):
            phase_shift = np.exp(1j * (j * np.pi / 2))
            temp_img = np.real(comp_reshaped * phase_shift)
            temp_img[temp_img == 0] = np.nan
            cmap = cm.coolwarm
            cmap.set_bad(color='none')
            im = axes[i, j].imshow(temp_img, cmap=cmap, vmin=-0.01, vmax=0.01)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(col_labels[j], fontsize=12)
        axes[i, 0].set_ylabel(row_labels[i], fontsize=14, rotation=0, labelpad=40,
                               verticalalignment='center', horizontalalignment='center', color='black')

    plt.suptitle("Spatiotemporal Patterns", fontsize=14)
    plt.colorbar(im, ax=axes[:, -1], fraction=0.03, pad=0.03)
    plt.show()

def plot_amplitude_phase(components, height, width, selected_components):
    fig, axes = plt.subplots(2, len(selected_components), figsize=(12, 6))
    row_labels = [r"$\rho_i$", r"$\theta_i$"]
    col_labels = [fr"$\Phi_{{{i}}}$" for i in selected_components]

    for i, comp_idx in enumerate(selected_components):
        comp_reshaped = np.resize(components[comp_idx], (height, width))

        amp_img = np.abs(comp_reshaped)
        amp_img[amp_img == 0] = np.nan
        vmin1, vmax1 = np.percentile(amp_img, [2, 98])
        cmap1 = plt.cm.YlOrRd
        cmap1.set_bad(color='none')
        im1 = axes[0, i].imshow(amp_img, cmap=cmap1, vmin=vmin1, vmax=vmax1)
        axes[0, i].axis('off')

        phase_img = np.angle(comp_reshaped)
        phase_img[phase_img == 0] = np.nan
        vmin2, vmax2 = np.percentile(phase_img, [2, 98])
        cmap2 = plt.cm.twilight
        cmap2.set_bad(color='none')
        im2 = axes[1, i].imshow(phase_img, cmap=cmap2, vmin=vmin2, vmax=vmax2)
        axes[1, i].axis('off')

        axes[0, i].set_title(col_labels[i], fontsize=14)

    for i, label in enumerate(row_labels):
        fig.text(0.08, 0.75 - i * 0.5, label, fontsize=14, verticalalignment='center', color='black')

    plt.suptitle("Spatial Distribution of Waves", fontsize=14)
    plt.colorbar(im1, ax=axes[0, -1], fraction=0.03, pad=0.03)
    plt.colorbar(im2, ax=axes[1, -1], fraction=0.03, pad=0.03)
    plt.show()

def main():
    print("Running CPCA analysis...")

    files = [
        'F:/college/科研/斑马鱼（张志鹏&史良）/fluorodata/Adultvglut21pre1.npy',
    ]
    X_list, height, width = load_data(files)

    X_selected = [X[:, :, :1800].reshape(-1, 1800).T for X in X_list]  # shape: (time, pixels)
    X_concat = np.concatenate(X_selected, axis=0)

    thresh = threshold_otsu(X_concat)
    valid_pixel_mask = (X_concat > thresh).any(axis=0)
    print(f"[筛选后] 有效像素数: {np.sum(valid_pixel_mask)}")

    X_masked = [X[:, :, :1800].reshape(-1, 1800).T[:, valid_pixel_mask] for X in X_list]
    X_masked_concat = np.concatenate(X_masked, axis=0)

    components, var_ratio = run_cpca_analysis(X_masked_concat, n_components=10)

    selected = [0, 1, 3, 5]
    plot_wave_patterns(components, height, width, selected)
    plot_amplitude_phase(components, height, width, selected)

    print("CPCA analysis completed successfully.")

if __name__ == '__main__':
    main()