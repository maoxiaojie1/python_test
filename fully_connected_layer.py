import numpy as np
from layer import Layer

# 全连接层
class FullyConnectedLayer(Layer):
    def __init__(self, m:int, n:int, activator="linear") -> None:
        super().__init__(activator=activator)
        self.weights = np.random.normal(0, 0.01, (m, n))
        self.bias = np.zeros((1, n))
        # 梯度
        self.weights_grad = np.zeros((m, n))
        self.bias_grad = np.zeros((1, n))
    
    def forward(self, x):
        self.input = x
        self.output = np.dot(x, self.weights) + self.bias
        return self.activator.forward(self.output)
    
    def backward(self, delta):
        delta = self.activator.backward(delta)
        self.weights_grad = np.dot(self.input.T, delta)
        self.bias_grad = np.sum(delta, axis=0, keepdims=True)
        # 计算当前层的误差
        return np.dot(delta, self.weights.T)

    def update(self, optimizer):
        optimizer.update([self.weights, self.bias], [self.weights_grad, self.bias_grad])

    def __str__(self):
        return "fully connected layer: " + str(self.weights.shape) + " "  + str(self.activator)