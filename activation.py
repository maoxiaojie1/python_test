import numpy as np

# 激活函数sigmoid
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        # out = 1 / (1 + np.exp(-x))
        # self.out = out
        # return out
        # 解决溢出问题
        # 把大于0和小于0的元素分别处理
        # 原来的sigmoid函数是 1/(1+np.exp(-Z))
        # 当Z是比较小的负数时会出现上溢，此时可以通过计算exp(Z) / (1+exp(Z)) 来解决
        
        mask = (x > 0)
        positive_out = np.zeros_like(x, dtype='float64')
        negative_out = np.zeros_like(x, dtype='float64')
        
        # 大于0的情况
        positive_out = 1 / (1 + np.exp(-x, positive_out, where=mask))
        # 清除对小于等于0元素的影响
        positive_out[~mask] = 0
        
        # 小于等于0的情况
        expZ = np.exp(x,negative_out,where=~mask)
        negative_out = expZ / (1+expZ)
        # 清除对大于0元素的影响
        negative_out[mask] = 0
        self.out = positive_out + negative_out
        return self.out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

# 激活函数softmax
class Softmax:
    """
    softmax激活函数
    """
    def __init__(self):
        self.y = None
    
    def forward(self, x):
        """
        前向传播
        """
        # 获取批量大小
        batch_size = x.shape[0]
        # 获取输入数据的最大值
        c = np.max(x)
        # 计算指数
        exp_x = np.exp(x - c)
        # 计算softmax
        y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.y = y
        return y
    
    def backward(self, t):
        """
        反向传播
        """
        # 获取批量大小
        batch_size = t.shape[0]
        # 计算梯度
        dx = (self.y - t) / batch_size
        return dx
    
    def __str__(self):
        return "softmax"

# 激活函数relu
class Relu:
    """
    relu激活函数
    """
    def __init__(self):
        self.mask = None

    def forward(self, x):
        """
        前向传播
        """
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, delta):
        """
        反向传播
        """
        # 计算梯度
        delta[self.mask] = 0
        grad = delta
        return grad

    def __str__(self):
        return "relu"

# 默认的激活函数为线性函数
class Linear:
    """
    线性激活函数
    """
    def forward(self, x):
        """
        前向传播
        """
        return x

    def backward(self, delta):
        """
        反向传播
        """
        return delta

    def __str__(self):
        return "linear"

def get_activation(activation):
    """
    根据激活函数名称获取激活函数
    """
    if activation == 'relu':
        return Relu()
    elif activation == 'softmax':
        return Softmax()
    elif activation == 'sigmoid':
        return Sigmoid()
    else:
        return Linear()