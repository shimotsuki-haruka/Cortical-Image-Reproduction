import numpy as np
import matplotlib.pyplot as plt

def load_data(files):#合并
    X_list = [np.load(f) for f in files]
    if X_list:
        height, width = X_list[0].shape[:2]
        return X_list, height, width
    else:
        return [], 0, 0

def inspect_npy_file(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        
        print(f"\n文件: {file_path}")
        print(f"数据类型: {type(data)}")
        if isinstance(data, np.ndarray):
            print(f"数组形状: {data.shape}")
            print(f"数组维度: {data.ndim}")
            print(f"数组元素类型: {data.dtype}")
            print(f"数组大小: {data.size}")
            
            if data.dtype.names is not None:
                print("结构化数组字段:")
                for name in data.dtype.names:
                    print(f"- {name}: {data.dtype.fields[name]}")
            
            print("\n数据内容示例:")
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
                print("(空数组)")
        else:
            print(f"对象内容: {data}")
            
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到")
    except ValueError as e:
        print(f"错误: 文件格式无效 - {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

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
    file_paths = [
        'E:/GithubData/Cortical-Image-Reproduction/fluorodata/Adultvglut21pre1.npy',
        'E:/GithubData/Cortical-Image-Reproduction/fluorodata/Adultvglut21pre2.npy',
    ]
    
    inspect_multiple_files(file_paths)
    jointed = []
    try:

        jointed, height, width = load_data(file_paths)
        print(f"\n成功加载 {len(jointed)} 个文件")
    except Exception as e:
        print(f"加载数据时出错: {e}")

    visualize(jointed, 100, 1)

