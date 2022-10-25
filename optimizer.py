import numpy as np

def get_optimizer(optimizer):
    if optimizer == "sgd":
        return SGD()
    elif optimizer == "momentum":
        return Momentum()
    elif optimizer == "rmsprop":
        return RMSProp()
    elif optimizer == "adam":
        return Adam()
    else:
        raise Exception("optimizer not found")

# SGD 优化器
class SGD(object):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, param, grad, state=None):
        return param - self.learning_rate * grad, state

    # 反向传播
    def backward(self, layers, delta):
        for layer in layers[::-1]:
          delta = layer.backward(delta)
          layer.update(self)


    # 交叉熵损失函数
    def loss(self, y, y_hat):
        return np.sum(-y * np.log(y_hat + 1e-9) - (1-y)*np.log(1 - y_hat + 1e-9))

    # 计算损失函数
    def calc_loss(self, y, y_hat):
        return self.loss(y, y_hat)
    
    # 交叉熵损失函数的梯度
    def loss_grad(self, y, y_hat):
        return (y_hat - y) #/ y.shape[0]
        # return (y_hat - y) / (y_hat * (1 - y_hat) + 1e-9)

class Momentum(object):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = None
    
    def update(self, param, grad, state=None):
        if state is None:
            state = np.zeros_like(param)
        state = self.momentum * state - self.learning_rate * grad
        return param + state, state

class RMSProp(object):
    def __init__(self, learning_rate=0.01, decay_rate=0.99, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
    
    def update(self, param, grad, state=None):
        if state is None:
            state = np.zeros_like(param)
        state = self.decay_rate * state + (1 - self.decay_rate) * np.square(grad)
        return param - self.learning_rate * grad / (np.sqrt(state) + self.epsilon), state
    
    def backward(self, layers, delta):
        for layer in layers[::-1]:
            delta = layer.backward(delta)
            layer.update(self)

class Adam(object):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)