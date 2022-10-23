import numpy as np
from data_set import *

from fnn import FNN
from optimizer import SGD
from fully_connected_layer import FullyConnectedLayer

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
    
    # 实例化业务处理类
    fnn = FNN()
    # 增加全连接层，输入层节点(28*28), 隐藏层2, 输出层10
    fnn.add_layer(FullyConnectedLayer(28*28, 400, activator="relu"))
    fnn.add_layer(FullyConnectedLayer(400, 300, activator="relu"))
    fnn.add_layer(FullyConnectedLayer(300, 500, activator="relu"))
    fnn.add_layer(FullyConnectedLayer(500, 10, activator="sigmoid"))
    # fnn.add_layer(OutputLayer(activator="softmax"))
    fnn.set_optimizer(SGD(learning_rate=0.01))
    # 打印网络结构信息
    print(fnn)
    train_images = train_images.reshape(-1, 28*28)
    # 归一化 b
    train_images = normalize(train_images)
    fnn.fit(train_images, train_labels, 500, batch_size=64)

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