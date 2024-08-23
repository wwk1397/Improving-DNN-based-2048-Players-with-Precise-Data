import numpy as np

def print_shape(array_a: np.ndarray):
    print("The shape of array_a is:", array_a.shape)

# 测试代码
array_a = np.random.rand(5, 3)  # 生成一个形状为(5, 3)的随机数组
print_shape(array_a)
print(array_a.shape != 2)
