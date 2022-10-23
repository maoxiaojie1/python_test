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
    def __init__(self, learning_rate=0.01) -> None:
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.learning_rate * grads[i]

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
    def __init__(self, learning_rate=0.01, momentum=0.9) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))
        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.learning_rate * grads[i]
            params[i] += self.v[i]

class RMSProp(object):
    def __init__(self, learning_rate=0.01, decay_rate=0.99, epsilon=1e-8) -> None:
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.s = None
    
    def update(self, params, grads):
        if self.s is None:
            self.s = []
            for param in params:
                self.s.append(np.zeros_like(param))
        for i in range(len(params)):
            self.s[i] = self.decay_rate * self.s[i] + (1 - self.decay_rate) * np.square(grads[i])
            params[i] -= self.learning_rate * grads[i] / (np.sqrt(self.s[i]) + self.epsilon)

class Adam(object):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params, grads):
        if self.m is None:
            self.m = []
            self.v = []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        self.t += 1
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * np.square(grads[i])
            m_hat = self.m[i] / (1 - np.power(self.beta1, self.t))
            v_hat = self.v[i] / (1 - np.power(self.beta2, self.t))
            params[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)