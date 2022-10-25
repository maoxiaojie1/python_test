import numpy as np

def get_loss(loss):
    if loss == "mse":
        return MSE()
    elif loss == "cross_entropy":
        return CrossEntropy()
    else:
        raise Exception("loss not found")

class MSE(object):   
    def forward(self, x, y):
        return np.mean(np.square(x - y))

    def backward(self, x, y):
        return 2 * (x - y) / x.shape[0]

class CrossEntropy(object):
    def forward(self, y_hat, y):
        # 多分类交叉熵
        return -np.sum(np.multiply(y, np.log(y_hat)))/y.shape[0]
    
    def backward(self, y_hat, y):
        return -np.divide(y, y_hat) / y.shape[0]