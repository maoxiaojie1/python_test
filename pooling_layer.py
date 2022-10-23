import numpy as np
from layer import Layer

# 池化层
class PoolingLayer(Layer):
    def __init__(self, input_shape, pool_shape, stride, padding, activator="max") -> None:
        super().__init__(input_shape=input_shape, activator=activator)
        # 池化核的形状
        self.pool_shape = pool_shape
        # 步长
        self.stride = stride
        # 填充
        self.padding = padding
    
    def forward(self, x):
        self.input = x
        # 计算输出形状
        output_shape = self.get_output_shape()
        # 初始化输出
        output = np.zeros(output_shape)
        # 计算池化
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                # 获取当前池化核的输入
                input = self.get_input(x, i, j)
                # 计算池化
                output[i, j] = self.activator.forward(input)
        self.output = output
        return self.output
    
    def backward(self, delta):
        # 计算当前层的误差
        return self.get_delta(delta)

    def update(self, optimizer):
        pass

    def get_output_shape(self):
        return (self.input_shape[0] - self.pool_shape[0] + 2 * self.padding) // self.stride + 1, \
            (self.input_shape[1] - self.pool_shape[1] + 2 * self.padding) // self.stride + 1
    
    def get_input(self, x, i, j):
        return x[i * self.stride : i * self.stride + self.pool_shape[0], j * self.stride : j * self.stride + self.pool_shape[1]]

    def get_delta(self, delta):
        # 初始化当前层的误差
        delta = np.zeros(self.input_shape)
        # 计算当前层的误差
        for i in range(self.pool_shape[0]):
            for j in range(self.pool_shape[1]):
                # 获取当前池化核的输入
                input = self.get_input(self.input, i, j)
                # 计算当前层的误差
                delta[i * self.stride : i * self.stride + self.pool_shape[0], j * self.stride : j * self.stride + self.pool_shape[1]] += self.activator.backward(input) * delta
        return delta

    def __str__(self) -> str:
        return "pooling layer: " + str(self.input_shape) + " " + str(self.pool_shape) + " " + str(self.stride) + " " + str(self.padding) + " " + str(self.activator)
