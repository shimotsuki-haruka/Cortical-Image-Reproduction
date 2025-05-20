import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
from DataView import *
from DataProcess import *

file_paths = [
        'E:/GithubData/Cortical-Image-Reproduction/fluorodata/Adultvglut21pre1.npy',

    ]

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

def select_seed_point(brain_map, data_width, data_height, brain_width, brain_height):
    plt.figure(figsize=(10, 8))
    plt.imshow(brain_map)
    plt.title('Click to select seed point')
    plt.axis('off')
    
    brain_seed_x, brain_seed_y = plt.ginput(1, -1)[0]
    brain_seed_x, brain_seed_y = int(brain_seed_x), int(brain_seed_y)
    print(f"Select point: ({brain_seed_x}, {brain_seed_y})")
    
    seed_x = int(brain_seed_x * data_width / brain_width)
    seed_y = int(brain_seed_y * data_height / brain_height)
    seed_x = min(max(0, seed_x), data_width - 1)
    seed_y = min(max(0, seed_y), data_height - 1)
    
    plt.close()
    return seed_x, seed_y

def load_neuron_data(file_paths):
    neuron_data = {}
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        neuron_type = filename.split('.')[0].rstrip('0123456789')
        data = np.load(file_path)
            
        if len(data.shape) == 3: 
            print(f"{neuron_type} shape: {data.shape}")
            neuron_data[neuron_type] = {'raw_data': data}
        else:
            print(f"Error: {neuron_type} shape is not 3D, is {data.shape}")
    
    return neuron_data

def global_signal_regression(data):
    h, w, t = data.shape
    gsr_data = data.copy()
    
    global_signal = np.zeros(t)
    valid_pixels = 0
    
    for i in range(h):
        for j in range(w):
            if np.std(data[i, j, :]) > 0.5:
                global_signal += data[i, j, :]
                valid_pixels += 1
    
    if valid_pixels > 0:
        global_signal /= valid_pixels
    
    for i in range(h):
        for j in range(w):
            if np.std(data[i, j, :]) > 0.5:
                slope, intercept, _, _, _ = stats.linregress(global_signal, data[i, j, :])
                gsr_data[i, j, :] = data[i, j, :] - (slope * global_signal + intercept)
    
    return gsr_data

def create_fc_map(data, seed_coordinates, use_gsr=False):
    h, w, t = data.shape
    seed_y, seed_x = seed_coordinates
    
    if use_gsr:
        processed_data = global_signal_regression(data)
    else:
        processed_data = data
    
    fc_map = np.full((h, w), np.nan) 
    seed_ts = processed_data[seed_y, seed_x, :]
    
    for i in range(h):
        for j in range(w):
            if np.std(processed_data[i, j, :]) > 0.5:
                fc_map[i, j] = np.corrcoef(seed_ts, processed_data[i, j, :])[0, 1]
    
    return fc_map

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
            print(f"Error: {neuron_type} data shape does not match")
    
    return fc_maps

def plot_fc_maps(fc_maps):
    num_maps = len(fc_maps)
    fig, axes = plt.subplots(nrows=2, ncols=num_maps, figsize=(6 * num_maps, 10))
    if num_maps == 1:
        axes = np.array([[axes[0]], [axes[1]]]) 

    yellow_red_cmap = LinearSegmentedColormap.from_list(
        'yellow_red', [(1, 1, 0.7), (1, 0.8, 0.4), (1, 0.6, 0.2), (0.8, 0.2, 0)]
    )
    blue_red_cmap = LinearSegmentedColormap.from_list(
        'blue_red', [(-1, 0, 0.8), (0, 0.7, 1), (1, 1, 1), (1, 0.7, 0), (0.8, 0, 0)]
    )

    for idx, (neuron_type, fc_map_data) in enumerate(fc_maps.items()):
        seed_y, seed_x = fc_map_data['seed_point']

        ax_raw = axes[0, idx]
        im_raw = ax_raw.imshow(fc_map_data['raw'], cmap=yellow_red_cmap, vmin=0.2, vmax=0.8)
        ax_raw.plot(seed_x, seed_y, 'k.', markersize=5)
        ax_raw.set_title(f"{neuron_type} FC (Raw)\nSeed: ({seed_x}, {seed_y})", fontsize=14)
        ax_raw.axis('off')
        fig.colorbar(im_raw, ax=ax_raw, fraction=0.046, pad=0.04)

        ax_gsr = axes[1, idx]
        im_gsr = ax_gsr.imshow(fc_map_data['gsr'], cmap=blue_red_cmap, vmin=-0.5, vmax=0.5)
        ax_gsr.plot(seed_x, seed_y, 'k.', markersize=5)
        ax_gsr.set_title(f"{neuron_type} FC (GSR)", fontsize=14)
        ax_gsr.axis('off')
        fig.colorbar(im_gsr, ax=ax_gsr, fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path = os.path.join(results_dir, 'fc_raw&gsr_click.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    brain_map, brain_height, brain_width = load_brainmap()
    jointed, data_height, data_width = load_data(file_paths)

    seed_x, seed_y = select_seed_point(brain_map, data_width, data_height, brain_width, brain_height)
    neuron_data = load_neuron_data(file_paths)
    print(neuron_data.shape)
    processd = datapre(neuron_data)
    fc_maps = generate_fc_maps(processd, (seed_y, seed_x))
    plot_fc_maps(fc_maps)