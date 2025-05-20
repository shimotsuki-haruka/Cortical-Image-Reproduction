import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap

# 数据文件路径
file_paths = {'VGLUT2': 'E:/GithubData/Cortical-Image-Reproduction/fluorodata/Adultvglut21pre1.npy',
    #'SOM': 'D:/Fluoro/data/adult_som/Adultsom2pre1.npy',
    #'VIP': 'D:/Fluoro/data/adult_vip/Adultvip1pre1.npy',
    #'PV': 'D:/Fluoro/data/adult_pv/Adultpv2pre5.npy'
}

# 创建results目录（如果不存在）
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# 1. 加载数据样本并获取尺寸
def load_sample_data():
    sample_data = np.load(list(file_paths.values())[0])
    data_height, data_width = sample_data.shape[0], sample_data.shape[1]
    print(f"数据尺寸: {data_height} x {data_width}")
    return sample_data, data_height, data_width

# 2. 加载脑区图
def load_brain_map():
    brain_map = plt.imread('E:/GithubData/Cortical-Image-Reproduction/BrainRegion.png')
    brain_height, brain_width = brain_map.shape[0], brain_map.shape[1]
    print(f"脑区图尺寸: {brain_height} x {brain_width}")
    return brain_map, brain_height, brain_width

# 3. 选择种子点并映射到数据坐标
def select_seed_point(brain_map, data_width, data_height, brain_width, brain_height):
    plt.figure(figsize=(10, 8))
    plt.imshow(brain_map)
    plt.title('Click to select seed point')
    plt.axis('off')
    
    # 获取用户点击的种子点坐标
    brain_seed_x, brain_seed_y = plt.ginput(1)[0]
    brain_seed_x, brain_seed_y = int(brain_seed_x), int(brain_seed_y)
    print(f"在脑区图上选择的点: ({brain_seed_x}, {brain_seed_y})")
    
    # 映射到数据坐标系
    seed_x = int(brain_seed_x * data_width / brain_width)
    seed_y = int(brain_seed_y * data_height / brain_height)
    seed_x = min(max(0, seed_x), data_width - 1)
    seed_y = min(max(0, seed_y), data_height - 1)
    
    plt.close()
    return seed_x, seed_y

# 4. 加载神经元数据并检查其形状
def load_neuron_data(file_paths):
    neuron_data = {}
    for neuron_type, file_path in file_paths.items():
        data = np.load(file_path)
        if len(data.shape) == 3:  # (height, width, time)
            print(f"{neuron_type} 数据形状: {data.shape}")
            neuron_data[neuron_type] = {'raw_data': data}
        else:
            print(f"警告: {neuron_type} 数据形状不是3D")
    return neuron_data

# 5. 全局信号回归
def global_signal_regression(data):
    """
    对3D数据进行全局信号回归
    
    参数:
    data - 3D数组，形状为(height, width, time)
    
    返回:
    gsr_data - 全局信号回归后的数据
    """
    h, w, t = data.shape
    gsr_data = data.copy()
    
    # 计算每个时间点的全局信号（所有有效像素的平均值）
    global_signal = np.zeros(t)
    valid_pixels = 0
    
    for i in range(h):
        for j in range(w):
            if np.std(data[i, j, :]) > 0.5:  # 只考虑有效信号
                global_signal += data[i, j, :]
                valid_pixels += 1
    
    if valid_pixels > 0:
        global_signal /= valid_pixels
    
    # 对每个像素进行全局信号回归
    for i in range(h):
        for j in range(w):
            if np.std(data[i, j, :]) > 0.5:
                slope, intercept, _, _, _ = stats.linregress(global_signal, data[i, j, :])
                gsr_data[i, j, :] = data[i, j, :] - (slope * global_signal + intercept)
    
    return gsr_data

# 6. 创建功能连接图
def create_fc_map(data, seed_coordinates, use_gsr=False):
    h, w, t = data.shape
    seed_y, seed_x = seed_coordinates
    
    # 如果需要，进行全局信号回归
    if use_gsr:
        processed_data = global_signal_regression(data)
    else:
        processed_data = data
    
    fc_map = np.full((h, w), np.nan)  # 初始化为NaN
    
    # 获取种子点的时间序列
    seed_ts = processed_data[seed_y, seed_x, :]
    
    # 计算每个点与种子点的相关性
    for i in range(h):
        for j in range(w):
            if np.std(processed_data[i, j, :]) > 0.5:  # 只处理有效信号
                fc_map[i, j] = np.corrcoef(seed_ts, processed_data[i, j, :])[0, 1]
    
    return fc_map

# 7. 创建每种类型的空间功能连接图（原始和GSR）
def generate_fc_maps(neuron_data, seed_coordinates):
    fc_maps = {}
    for neuron_type, data_dict in neuron_data.items():
        data = data_dict['raw_data']
        
        if len(data.shape) == 3:
            fc_map_raw = create_fc_map(data, seed_coordinates, use_gsr=False)
            fc_map_gsr = create_fc_map(data, seed_coordinates, use_gsr=True)
            fc_maps[neuron_type] = {
                'raw': fc_map_raw,
                'gsr': fc_map_gsr,
                'seed_point': seed_coordinates
            }
        else:
            print(f"警告: {neuron_type} 数据不是3D空间+时间格式，无法创建空间功能连接图")
    
    return fc_maps

# 8. 可视化功能连接图
def plot_fc_maps(fc_maps):
    plt.figure(figsize=(16, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    # 设置色谱
    yellow_red_cmap = LinearSegmentedColormap.from_list(
        'yellow_red', [(1, 1, 0.7), (1, 0.8, 0.4), (1, 0.6, 0.2), (0.8, 0.2, 0)]
    )
    blue_red_cmap = LinearSegmentedColormap.from_list(
        'blue_red', [(-1, 0, 0.8), (0, 0.7, 1), (1, 1, 1), (1, 0.7, 0), (0.8, 0, 0)]
    )

    # 显示原始FC图和GSR后的FC图
    for i, neuron_type in enumerate(fc_maps.keys()):
        if neuron_type in fc_maps:
            fc_map_data = fc_maps[neuron_type]
            seed_y, seed_x = fc_map_data['seed_point']
            
            # 原始FC图
            plt.subplot(2, len(fc_maps), i + 1)
            im_raw = plt.imshow(fc_map_data['raw'], cmap=yellow_red_cmap, vmin=0, vmax=1)
            plt.plot(seed_x, seed_y, 'k.', markersize=5)  # 标记种子点
            plt.title(f"{neuron_type} FC Map (Raw)\nSeed: ({seed_x}, {seed_y})")
            plt.colorbar(im_raw, fraction=0.02, pad=0.1)
            plt.axis('off')
            
            # GSR后的FC图
            plt.subplot(2, len(fc_maps), i + 1 + len(fc_maps))
            im_gsr = plt.imshow(fc_map_data['gsr'], cmap=blue_red_cmap, vmin=-1, vmax=1)
            plt.plot(seed_x, seed_y, 'k.', markersize=5)  # 标记种子点
            plt.title(f"{neuron_type} FC Map (GSR)\nSeed: ({seed_x}, {seed_y})")
            plt.colorbar(im_gsr, fraction=0.02, pad=0.1)
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'custom_seed_fc_maps_with_gsr.png'), dpi=300, bbox_inches='tight')
    plt.show()

# 主程序
def main():
    sample_data, data_height, data_width = load_sample_data()
    brain_map, brain_height, brain_width    = load_brain_map()
    seed_x, seed_y = select_seed_point(brain_map, data_width, data_height, brain_width, brain_height)
    neuron_data = load_neuron_data(file_paths)
    fc_maps = generate_fc_maps(neuron_data, (seed_y, seed_x))
    plot_fc_maps(fc_maps)

if __name__ == "__main__":
    main()