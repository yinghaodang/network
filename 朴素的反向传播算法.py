import numpy as np
from network import sigmoid, sigmoid_prime

x = np.array([[0.35], [0.9]])
y = np.array([[0.5]])

w0 = np.array([[0.1, 0.8], [0.4, 0.6]])
w1 = np.array([[0.3, 0.9]])

for j in range(100):
    a0 = x

    z1 = np.dot(w0, a0)
    a1 = sigmoid(z1)

    z2 = np.dot(w1, a1)
    a2 = sigmoid(z2)

    l2_error = a2 - y

    Error = (y - a2) ** 2
    print(Error)

    l2_delta = (a2 - y) * sigmoid_prime(z2)  # this will backpack

    l1_error = w1.T * l2_delta  # 反向传播
    l1_delta = l1_error * sigmoid_prime(z1)

    w1 -= a1.T * l2_delta  # 修改权值
    w0 -= a0.T * l1_delta
