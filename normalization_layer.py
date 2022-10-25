import numpy as np
from layer import Layer

# 归一化层
class NormalizationLayer(Layer):
    def __init__(self):
        super().__init__()
        # 归一化参数
        self.__mean = None
        self.__std = None

    def forward(self, x):
        self.input = x
        # 计算均值和方差
        self.__mean = np.mean(x, axis=0)
        self.__std = np.std(x, axis=0) + 1e-8
        # 归一化
        self.output = (x - self.__mean) / self.__std
        return self.output

    def backward(self, delta):
        # 计算当前层的误差
        return delta / self.__std

    def __str__(self) -> str:
        return "normalization layer: -"