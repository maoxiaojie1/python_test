import numpy as np

from layer import Layer

# flatten层
class FlattenLayer(Layer):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        self.input = x
        # 计算输出形状
        output_shape = self.get_output_shape()
        # 计算输出
        self.output = x.reshape(output_shape)
        return self.output
    
    def backward(self, delta):
        # 计算当前层的误差
        return self.get_delta(delta)

    def get_output_shape(self):
        return self.input_shape[0] * self.input_shape[1]
    
    def get_delta(self, delta):
        # 初始化当前层的误差
        delta = np.zeros(self.input_shape)
        # 计算当前层的误差
        delta = delta.reshape(self.input_shape)
        return delta

    def __str__(self) -> str:
        return "flatten layer: " + str(self.input_shape)