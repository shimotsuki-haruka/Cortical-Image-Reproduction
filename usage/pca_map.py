import numpy as np
import matplotlib.pyplot as plt
from fluoromind.group.pca import GroupPCA
from skimage.filters import threshold_otsu

# 加载数据函数
def load_data(files):
    """
    加载神经影像数据并将其转换为合适的形状。
    """
    X_list = [np.load(f) for f in files]
    height, width, _ = X_list[0].shape
    return X_list, height, width

# PCA分析函数
def perform_pca(X_concat, n_components=10):
    """
    对拼接后的数据进行GroupPCA处理。
    """
    chunk_size = 1800  # 每个块的大小
    n_chunks = X_concat.shape[0] // chunk_size + (1 if X_concat.shape[0] % chunk_size != 0 else 0)

    all_components = []
    all_explained_variance_ratios = []

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, X_concat.shape[0])

        X_chunk = X_concat[start_idx:end_idx, :]

        # 执行 GroupPCA
        group_pca = GroupPCA(n_components=n_components)
        group_pca.fit([X_chunk])  # 这里输入的是数据块

        # 获取PCA结果
        components = group_pca.pca_.components_  # 提取主成分
        explained_variance_ratio = group_pca.pca_.explained_variance_ratio_  # 解释方差占比

        # 将每个块的结果添加到总列表中
        all_components.append(components)
        all_explained_variance_ratios.append(explained_variance_ratio)

    # 将所有块的结果合并
    all_components = np.concatenate(all_components, axis=0)
    all_explained_variance_ratios = np.concatenate(all_explained_variance_ratios, axis=0)

    return all_components, all_explained_variance_ratios

# 绘制结果图函数
def plot_results(all_components, all_explained_variance_ratios, mask, height, width, valid_pixel_mask):
    selected_components = [0, 1, 3, 5]

    fig, axes = plt.subplots(1, len(selected_components), figsize=(15, 4))

    for i, comp_idx in enumerate(selected_components):
        comp_vector = all_components[comp_idx]  # shape: (9305,)

        # 构造 shape 为 (height * width) 的空图像，并填入有效像素位置
        full_map = np.full(height * width, np.nan)
        full_map[valid_pixel_mask] = comp_vector
        full_map = full_map.reshape(height, width)

        im = axes[i].imshow(full_map, cmap='RdBu_r',
                            vmin=-np.nanmax(np.abs(full_map)),
                            vmax=np.nanmax(np.abs(full_map)))

        axes[i].contour(mask, levels=[0.5], colors='black', linewidths=0.8)
        axes[i].axis('off')
        axes[i].set_title(f"PCA{comp_idx+1}\n{all_explained_variance_ratios[comp_idx]:.4f}")

    plt.suptitle("GroupPCA Spatial Patterns", fontsize=14)
    plt.colorbar(im, ax=axes, orientation='vertical', fraction=0.01, pad=0.01)
    plt.show()


# 主函数
def main():
    # 加载数据
    files = [
        'F:/college/科研/斑马鱼（张志鹏&史良）/fluorodata/Adultvglut21pre1.npy',
    ]
    X_list, height, width = load_data(files)

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

    # 使用有效像素做 GroupPCA 分析
    all_components, all_explained_variance_ratios = perform_pca(X_filtered, n_components=10)

    # 加载mask (用于叠加轮廓)
    mask = np.load('E:/demo_fm/gssidnaim-main/demodata/mask.npy')

    # 绘制结果图
    plot_results(all_components, all_explained_variance_ratios, mask, height, width, valid_pixel_mask)

if __name__ == '__main__':
    main()
