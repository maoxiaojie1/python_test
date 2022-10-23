from module import Module
from activation import get_activation

# layer抽象类
class Layer(Module):
    def __init__(self, input_shape=None, output_shape=None, activator='linear') -> None:
        super().__init__()
        # 输入数据的形状
        self.__input_shape = input_shape
        # 输出数据的形状
        self.__output_shape = output_shape
        # 当前层的输入
        self.input = None
        # 当前层的输出
        self.output = None
        # 激活函数
        self.__activator = get_activation(activator)

    @property
    def activator(self):
        return self.__activator

    @property
    def input_shape(self):
        return self.__input_shape
    
    @property
    def output_shape(self):
        return self.__output_shape