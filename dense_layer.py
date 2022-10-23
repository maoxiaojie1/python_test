import numpy as np
from layer import Layer

class DenseLayer(Layer):
    def __init__(self, input_shape, output_shape, activator="relu") -> None:
        super().__init__(input_shape, output_shape, activator)
        # 权重
        self.weights = np.random.randn(self.input_shape, self.output_shape)
        # 偏置
        self.bias = np.random.randn(self.output_shape)
    
    def forward(self, x):
        self.input = x
        # 计算输出
        output = np.dot(x, self.weights) + self.bias
        # 激活
        self.output = self.activator.forward(output)
        return self.output
    
    def backward(self, delta):
        # 计算当前层的误差
        return self.get_delta(delta)

    def update(self, optimizer):
        # 更新权重
        self.weights = optimizer.update(self.weights, self.get_weights_delta())
        # 更新偏置
        self.bias = optimizer.update(self.bias, self.get_bias_delta())

    def get_weights_delta(self):
        # 计算当前层的权重误差
        return np.dot(self.input.T, self.get_delta(self.output))

    def get_bias_delta(self):
        # 计算当前层的偏置误差
        return np.sum(self.get_delta(self.output), axis=0)

    def get_delta(self, delta):
        # 计算当前层的误差
        return self.activator.backward(delta) * delta

    def __str__(self) -> str:
        return "dense layer: " + str(self.input_shape) + " " + str(self.output_shape) + " " + str(self.activator)