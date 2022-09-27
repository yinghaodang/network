import network as nt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv(r'D:/jupyter_notebook/abalone.data', header=None)
    data.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight",
                    "Shell weight", "Rings"]
    data["Sex"] = data["Sex"].apply(lambda x: 1 if x == "M" else 0)

    data = data.sample(frac=1)  # 随机采样，即打散样本

    x = data.iloc[:, :-1]  # x是输入特征，y是输出值
    y = data.iloc[:, -1]

    x = np.array(x)  # 转化为numpy的结构
    y = np.array(y)

    # 划分为训练集，验证集和测试集
    train_data_x = x[:3341, :]
    verify_data_x = x[3341:3759, :]
    test_data_x = x[3759:4177, :]

    train_data_y = y[:3341]
    verify_data_y = y[3341:3759]
    test_data_y = y[3759:4177]

    train_data = [(x.reshape((-1, 1)), y) for x, y in zip(train_data_x, train_data_y)]
    verify_data = [(x.reshape((-1, 1)), y) for x, y in zip(verify_data_x, verify_data_y)]
    test_data = [(x.reshape((-1, 1)), y) for x, y in zip(test_data_x, test_data_y)]

    # 构建一个8*8*1的神经网络
    network = nt.Network([8, 8, 8, 8, 8, 1])

    print(network.cost(train_data))

    network.sgd(train_data, 100, 1, 1)

    print(network.cost(train_data))
    print(network.cost(test_data))
