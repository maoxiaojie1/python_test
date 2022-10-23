import numpy as np
from layer import Layer

# 卷积层
class Conv2DLayer(Layer):
    def __init__(self, input_shape, filter_shape, stride, padding, activator="linear") -> None:
        super().__init__(input_shape=input_shape, activator=activator)
        # 卷积核的形状
        self.filter_shape = filter_shape
        # 步长
        self.stride = stride
        # 填充
        self.padding = padding
        # 初始化卷积核
        self.filters = np.random.normal(0, 0.01, filter_shape)
        # 初始化偏置
        self.bias = np.zeros((filter_shape[0], 1))
        # 梯度
        self.filters_grad = np.zeros(filter_shape)
        self.bias_grad = np.zeros((filter_shape[0], 1))
    
    def forward(self, x):
        self.input = x
        # 计算输出形状
        output_shape = self.get_output_shape()
        # 初始化输出
        output = np.zeros(output_shape)
        # 计算卷积
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                # 获取当前卷积核的输入
                input = self.get_input(x, i, j)
                # 计算卷积
                output[i, j] = np.sum(input * self.filters) + self.bias
        # 激活函数
        self.output = self.activator.forward(output)
        return self.output
    
    def backward(self, delta):
        # 计算激活函数的梯度
        delta = self.activator.backward(delta)
        # 计算偏置的梯度
        self.bias_grad = np.sum(delta, axis=(1, 2), keepdims=True)
        # 计算卷积核的梯度
        for i in range(self.filter_shape[0]):
            for j in range(self.filter_shape[1]):
                # 获取当前卷积核的输入
                input = self.get_input(self.input, i, j)
                # 计算卷积核的梯度
                self.filters_grad[i, j] = np.sum(input * delta)
        # 计算当前层的误差
        return self.get_delta(delta)

    def update(self, optimizer):
        optimizer.update([self.filters, self.bias], [self.filters_grad, self.bias_grad])

    def get_output_shape(self):
        return (self.input_shape[0] - self.filter_shape[0] + 2 * self.padding) // self.stride + 1, \
            (self.input_shape[1] - self.filter_shape[1] + 2 * self.padding) // self.stride + 1
    
    def get_input(self, x, i, j):
        return x[i * self.stride : i * self.stride + self.filter_shape[0], j * self.stride : j * self.stride + self.filter_shape[1]]

    def get_delta(self, delta):
        # 初始化当前层的误差
        delta = np.zeros(self.input_shape)
        # 计算当前层的误差
        for i in range(self.filter_shape[0]):
            for j in range(self.filter_shape[1]):
                # 获取当前卷积核的输入
                input = self.get_input(self.input, i, j)
                # 计算当前层的误差
                delta[i * self.stride : i * self.stride + self.filter_shape[0], j * self.stride : j * self.stride + self.filter_shape[1]] += self.filters[i, j] * delta
        return delta

    def __str__(self) -> str:
        return "convolution layer: " + str(self.input_shape) + " " + str(self.filter_shape) + " " + str(self.stride) + " " + str(self.padding) + " " + str(self.activator)
