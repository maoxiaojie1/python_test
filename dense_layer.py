import numpy as np
from layer import Layer

class DenseLayer(Layer):
    def __init__(self, input_shape, output_shape, activator="relu") -> None:
        super().__init__(input_shape, output_shape, activator)
        # 权重
        self.weights = np.random.normal(0, 0.01, (input_shape, output_shape))
        # 偏置
        self.bias = np.zeros((1, output_shape))
        # delta
        self.delta = None

        self.weights_state = None # 用于优化器
        self.bias_state = None # 用于优化器
    
    def forward(self, x):
        self.input = x
        # 计算输出
        output = np.dot(x, self.weights) + self.bias
        # 激活
        return self.activator.forward(output)
    
    def backward(self, delta):
        # 计算当前层的误差
        self.delta = self.activator.backward(delta)
        return np.dot(delta, self.weights.T)

    def update(self, optimizer):
        # 更新权重
        self.weights, self.weights_state = optimizer.update(self.weights, self.get_weights_delta(), self.weights_state)
        # 更新偏置
        self.bias, self.bias_state = optimizer.update(self.bias, self.get_bias_delta(), self.bias_state)

    def get_weights_delta(self):
        # 计算当前层的权重误差
        return np.dot(self.input.T, self.delta)

    def get_bias_delta(self):
        # 计算当前层的偏置误差
        return np.sum(self.delta, axis=0, keepdims=True)

    def __str__(self) -> str:
        return "dense layer: " + str(self.input_shape) + " " + str(self.output_shape) + " " + str(self.activator)