import numpy as np
from data_set import *

from fnn import FNN
from optimizer import SGD, RMSProp
from fully_connected_layer import FullyConnectedLayer
from normalization_layer import NormalizationLayer
import torch

from cnn import CNN
from conv2d_layer import Conv2DLayer
from pooling_layer import PoolingLayer
from flatten_layer import FlattenLayer
from dense_layer import DenseLayer

# 归一化图像处理
def normalize(data):
    """
    归一化图像处理
    """
    # 将数据转换为float64类型
    data = data.astype(np.float64)
    # 将数据归一化到0~1之间
    data = data / 255.0
    return data



# 利用torch构建前馈神经网络
def torch_fnn(train_data, train_label, test_data, test_label):
    """
    利用torch构建前馈神经网络
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # 构建前馈神经网络
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 500, dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(500, 400, dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(400, 10, dtype=torch.float64),
        torch.nn.Softmax(dim=1)
    )
    
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 将数据转换为tensor
    train_data = torch.from_numpy(train_data).to(device)
    train_label = torch.from_numpy(train_label).to(device)
    test_data = torch.from_numpy(test_data).to(device)
    test_label = torch.from_numpy(test_label).to(device)

    # 将数据分批次
    batch_size = 64
    train_data_batch = torch.split(train_data, batch_size)
    train_label_batch = torch.split(train_label, batch_size)

    # 训练模型
    for i in range(50):
        for x, y in zip(train_data_batch, train_label_batch):
            y_pred = model(x)
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        #计算准确率
        with torch.no_grad():
            # 计算训练集准确率
            y_pred = model(train_data)
            train_acc = (y_pred.argmax(1) == torch.argmax(train_label, dim=1)).type(torch.float).mean().item()
            # 计算测试集准确率
            y_pred = model(test_data)
            test_acc = (y_pred.argmax(1) == torch.argmax(test_label, dim=1)).type(torch.float).mean().item()
            print(f"epoch: {i}, loss: {l.item():.4f}, train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}")
    
if __name__ == '__main__':
    # 获取训练集和标签
    train_images, train_labels = get_train_data()
    # 获取测试集和标签
    test_images, test_labels = get_test_data()

    # 打印训练集和标签的数量
    print('train_images.shape: ', train_images.shape)
    print('train_labels.shape: ', train_labels.shape)
    # 打印测试集和标签的数量
    print('test_images.shape: ', test_images.shape)
    print('test_labels.shape: ', test_labels.shape)
    train_images = train_images.reshape(-1, 28*28)
    train_images = normalize(train_images)
    test_images = test_images.reshape(-1, 28*28)
    test_images = normalize(test_images)
    torch_fnn(train_images, train_labels, test_images, test_labels)

    # 实例化业务处理类
    # fnn = FNN()
    # # 增加全连接层，输入层节点(28*28), 隐藏层2, 输出层10
    # fnn.add_layer(DenseLayer(28*28, 1000, activator="relu"))
    # # fnn.add_layer(NormalizationLayer())
    # fnn.add_layer(DenseLayer(1000, 800, activator="relu"))
    # fnn.add_layer(NormalizationLayer())
    # # fnn.add_layer(DenseLayer(300, 500, activator="relu"))
    # fnn.add_layer(DenseLayer(800, 10, activator="softmax"))
    # # fnn.set_optimizer(RMSProp(learning_rate=0.001, decay_rate=0.9, epsilon=1e-8))
    # fnn.set_optimizer(SGD(learning_rate=0.001))
    # # 打印网络结构信息
    # print(fnn)
    # train_images = train_images.reshape(-1, 28*28)
    # # 归一化 b
    # train_images = normalize(train_images)
    # fnn.fit(train_images, train_labels, 500, batch_size=256)

    # 实例化业务处理类
    # cnn = CNN([
    #     Conv2DLayer((1, 28, 28), (6, 1, 5, 5), 3, 3, activator="relu"),
    #     PoolingLayer((6, 24, 24), (6, 12, 12), 2, 2, activator="max"),
    #     Conv2DLayer((6, 12, 12), (16, 5, 5), 3, 3, activator="relu"),
    #     PoolingLayer((16, 8, 8), (16, 4, 4), 2, 2, activator="max"),
    #     FlattenLayer(),
    #     DenseLayer(16*4*4, 10, activator="softmax")
    # ])

    # cnn.train(train_images, train_labels, 64, 50, "cross_entropy", "sgd")