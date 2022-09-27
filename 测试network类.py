import network as nt
import numpy as np


# 定义网络的结构
network = nt.Network([2, 2, 1])

# 自定义初始化的参数
weight = [np.array([[0.1, 0.8], [0.4, 0.6]]), np.array([[0.3, 0.9]])]
biases = [np.array([[0], [0]]), np.array([[0]])]
network.personal_initializer(weights=weight, biases=biases)

# 网络输入层的数据，和weights中第一个元素的列数维度相同
a = np.array([[0.35], [0.9]])
# 网络输出层的值
y = 0.5

print(network.backprop(a, y, bias=True))

# # 调用随机梯度下降算法，
# network.sgd([(a, y)], 100, 1, 1)
# print(network.cost([(a, y)]))

# # -----------不含偏置训练100次----------
# network.weights = weight
# network.biases = biases
# for i in range(100):
#     network.update_mini_batch([(a, y)], 1, bias=False)
#     print(network.cost([(a, y)]))

# # -----------等价于前面的随机梯度下降----------
# network.weights = weight
# network.biases = biases
#
# for i in range(100):
#     network.update_mini_batch([(a, y)], 1, bias=True)
#     print(network.cost([(a, y)]))  # 结果是2.31*10-13
