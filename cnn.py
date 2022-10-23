import numpy as np
from loss import get_loss
from optimizer import get_optimizer
from dense_layer import DenseLayer
from conv2d_layer import Conv2DLayer

# 卷积神经网络
class CNN(object):
    def __init__(self, layers) -> None:
        # 网络层
        self.layers = layers
        # 损失函数
        self.loss = None
        # 激活函数
        self.activator = None
        # 优化器
        self.optimizer = None
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, delta):
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
        return delta
    
    def update(self):
        for layer in self.layers:
            layer.update(self.optimizer)
    
    def train(self, x, y, batch_size, epoch, loss, optimizer, verbose=1):
        # 初始化损失函数
        self.loss = get_loss(loss)
        # 初始化优化器
        self.optimizer = get_optimizer(optimizer)
        # 初始化参数
        # self.init_params()
        # 训练
        for i in range(epoch):
            # 计算损失
            y_hat = self.forward(x)
            loss = self.loss.forward(y_hat, y)
            # 计算梯度
            delta = self.loss.backward(y_hat, y)
            # 反向传播
            self.backward(delta)
            # 更新参数
            self.update()
            # 打印损失
            if verbose == 1:
                print("epoch: %d, loss: %f" % (i, loss))
    
    def predict(self, x):
        return self.activator.forward(self.forward(x))
    
    def init_params(self):
        for layer in self.layers:
            if isinstance(layer, Conv2DLayer):
                layer.init_params()
            elif isinstance(layer, DenseLayer):
                layer.init_params()
    
    def __str__(self) -> str:
        return "cnn: " + str(self.layers)