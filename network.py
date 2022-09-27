"""
network.py
~~~~~~~~~~

来自 https://github.com/mnielsen/neural-networks-and-deep-learning/tree/master/src
基于python创建的神经网络类。
"""

import random
import numpy as np


class Network(object):

    def __init__(self, sizes):
        # 神经网络初始化方法，sizes为网络的结构元组，例如：[2,3,1]代表一层输入层，一层隐含层，一层输出层的神经网络
        # 创建一个神经网络之初就会随机初始化参数，参数是基于均值为0，方差为1的正态分布数据
        # biases : bias[l]是n[l]x1的列向量
        # weights : weight[l]是n[l]xn[l-1]的矩阵，这样设置的目的方便记忆————正向传播不转置，反向传播需转置(n[x]代表x层的节点数)
        # num_layers : 数学推导中的L+1,最后一层神经网络编号是L
        self.num_layers = len(sizes) 
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def personal_initializer(self, weights, biases):
        # 自定义的初始权重，偏置的方法
        # 参数中行向量需写成[[...]]，列向量需写成[[.],[.]]，实数需写成[[.]]
        if len(self.weights) != len(weights):
            raise "dim is wrong"
        for i, j in zip(self.weights, weights):
            if i.shape != j.shape:
                raise "dim is wrong"
        self.weights = weights
        self.biases = biases

    def feedforward(self, a, bias=True):
        # 前项传播算法，供训练完毕后快速计算出输出层的数据，而不关心每一层的中间量a[l]
        if bias:
            for b, w in zip(self.biases, self.weights):
                a = sigmoid(np.dot(w, a) + b)
        else:
            for w in self.weights:
                a = sigmoid(np.np.dot(w, a))
        return a

    def sgd(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        # 批梯度下降法、随机梯度下降法、小批量梯度下降法
        # training_data是[(x1,y1),(x2,y2)]形式的数据
        # epochs : 迭代的次数
        # mini_batch_size : 每次小批量的样本量
        # eta : 学习率
        # 注意:mini_batch_size设置成1就是随机梯度下降法，
        #                    设置成len(training_data)就是批梯度下降法
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta, bias=True):
        # 更新梯度
        # mini_batch : mini_batches中的一个样本
        #
        if bias:
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            for x, y in mini_batch:
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.weights = [w - (eta / len(mini_batch)) * nw
                            for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (eta / len(mini_batch)) * nb
                           for b, nb in zip(self.biases, nabla_b)]
        else:
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            for x, y in mini_batch:
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.weights = [w - (eta / len(mini_batch)) * nw
                            for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y, bias=True):
        # 反向传播算法
        # x : 初始值
        # y : 输出值
        # 返回各个层级网络中参数对应于损失函数的导数
        if bias:
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            # feedforward
            activation = x
            activations = [x]  # list to store all the activations, layer by layer
            zs = []  # list to store all the z vectors, layer by layer
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation) + b
                zs.append(z)
                activation = sigmoid(z)
                activations.append(activation)
            # backward pass
            delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())
            # Note that the variable l in the loop below is used a little
            # differently to the notation in Chapter 2 of the book.  Here,
            # l = 1 means the last layer of neurons, l = 2 is the
            # second-last layer, and so on.  It's a renumbering of the
            # scheme in the book, used here to take advantage of the fact
            # that Python can use negative indices in lists.
            for l in range(2, self.num_layers):
                z = zs[-l]
                sp = sigmoid_prime(z)
                delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
            return nabla_b, nabla_w
        else:
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            # feedforward
            activation = x
            activations = [x]
            zs = []
            for w in self.weights:
                z = np.dot(w, activation)
                zs.append(z)
                activation = sigmoid(z)
                activations.append(activation)
            # backward pass
            delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())
            for l in range(2, self.num_layers):
                z = zs[-l]
                sp = sigmoid_prime(z)
                delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
                nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
            return nabla_w

    def evaluate(self, test_data):

        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    @staticmethod
    def cost_derivative(output_activations, y):
        # 返回损失函数对最后一层输出a[L]求导的值，即为delta[L] / sigmoid(z[L])
        return output_activations - y

    def cost(self, test_data):
        s = 0
        for x, y in test_data:
            s += (self.feedforward(x) - y) ** 2

        return s / len(test_data)


# Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    return (abs(z) + z) / 2


def relu_prime(z):
    return (abs(z) + z) / (2 * z)
