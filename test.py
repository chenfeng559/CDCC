from load_data_np import split_sample
import numpy as np 

try:
    # 假设这是您的输入数据
    x = np.random.rand(69680, 6)
    window_size = [24, 1]
    step = 1
    result = split_sample(x, window_size, step)
    print(f"Result shape: {result.shape}")
except ValueError as e:
    print(f"Error: {e}")