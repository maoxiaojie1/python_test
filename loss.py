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
        return -np.mean(np.multiply(y, np.log(y_hat)) + np.multiply(1 - y, np.log(1 - y_hat)))
    
    def backward(self, y_hat, y):
        return (y_hat - y) / (y_hat * (1 - y_hat))