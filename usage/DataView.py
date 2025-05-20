import numpy as np
import matplotlib.pyplot as plt

file_paths = [
        'E:/GithubData/Cortical-Image-Reproduction/fluorodata/Adultvglut21pre1.npy',
        'E:/GithubData/Cortical-Image-Reproduction/fluorodata/Adultvglut21pre2.npy',
    ]

def load_data(files):#合并
    X_list = [np.load(f) for f in files]
    if X_list:
        height, width = X_list[0].shape[:2]
        print(f"Files: {height} x {width}")
        return X_list, height, width
    else:
        return [], 0, 0
    
def load_brainmap():
    brain_map = plt.imread('E:/GithubData/Cortical-Image-Reproduction/BrainRegion.png')
    brain_height, brain_width = brain_map.shape[0], brain_map.shape[1]
    print(f"Brain: {brain_height} x {brain_width}")
    return brain_map, brain_height, brain_width


def inspect_npy_file(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        
        print(f"\nFile: {file_path}")
        print(f"Data type: {type(data)}")
        if isinstance(data, np.ndarray):
            print(f"Shape: {data.shape}")
            #print(f"Dimension: {data.ndim}")
            print(f"Element type: {data.dtype}")
            #print(f"Size: {data.size}")
            
            if data.dtype.names is not None:
                print("Structured array field:")
                for name in data.dtype.names:
                    print(f"- {name}: {data.dtype.fields[name]}")
            
            '''print("\nExample of data content:")
            if data.size > 0:
                if data.ndim == 0:
                    print(data.item())
                elif data.size <= 10:
                    print(data)
                else:
                    if data.ndim == 1:
                        print(data[:10], "...")
                    else:
                        print(data[40:60, 20:40], "...")
            else:
                print("(Empty array)")'''
        else:
            print(f"Object content: {data}")
            
    except FileNotFoundError:
        print(f"Error: file '{file_path}' is not found")
    except ValueError as e:
        print(f"Error: the file format is invalid - {e}")
    except Exception as e:
        print(f"Unknown error: {e}")

def inspect_multiple_files(file_paths):
    for file_path in file_paths:
        inspect_npy_file(file_path)

def visualize(X, frame_index, array_index):
    if array_index == -1:
        frame = X[frame_index, :, : ]
    else:
        frame = X[array_index][:, :, frame_index]
    plt.imshow(frame, cmap='gray')
    plt.title(f"Frame {frame_index}")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    
    inspect_multiple_files(file_paths)
    jointed = []
    try:
        jointed, height, width = load_data(file_paths)
        print(f"\nSuccessfully loaded {len(jointed)} file(s)")
    except Exception as e:
        print(f"Error: {e}")

    visualize(jointed, 100, 1)

