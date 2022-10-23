import numpy as np

# model抽象类
class Model(object):
    def __init__(self) -> None:
        # 损失函数
        self.loss = None
        # 层列表
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def compile(self, loss):
        self.loss = loss
    
    def forward(self, x):
        pass
    
    def backward(self, y_hat, y):
        pass
    
    def update(self, optimizer):
        pass
    
    def fit(self, x, y, optimizer, epochs=1, batch_size=32):
        pass
    
    def evaluate(self, x, y):
        pass
    
    def predict(self, x):
        pass

