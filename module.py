# 模块类
class Module(object):
    def __init__(self) -> None:
        pass

    def forward(self, x):
        raise NotImplementedError("forward method is not implemented")
    
    def backward(self, delta):
        raise NotImplementedError("backward method is not implemented")
    
    def update(self, optimizer):
        pass

    def __str__(self) -> str:
        raise NotImplementedError("__str__ method is not implemented")