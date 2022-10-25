import numpy as np
from optimizer import SGD
from loss import MSE, CrossEntropy

# 前馈神经网络
class FNN(object):
    def __init__(self, optimizer=SGD()) -> None:
        self.layers = []
        # 优化器
        self.optimizer = optimizer

    # 设置优化器
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    # 增加全连接层
    def add_layer(self, layer):
        self.layers.append(layer)
    
    # 前馈输出
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    # 训练函数
    def fit(self, images, labels, num_iters, batch_size=256):
        # L = CrossEntropy()
        L = MSE()
        # 生成shuffle后的batch
        batches = self.generate_batches(images, labels, batch_size)
        for i in range(num_iters):
            for batch in batches:
                images_batch, labels_batch = batch
                # 前馈
                y_hat = self.forward(images_batch)
                # 计算损失
                loss = L.forward(y_hat, labels_batch)
                # 计算梯度
                delta = L.backward(y_hat, labels_batch)
                # 反向传播
                self.optimizer.backward(self.layers, delta)
            # 预测
            y_hat = self.predict(images)
            # 准确率
            accuracy = self.calc_accuracy(labels, y_hat)
            print("iter: %d, loss: %f, accuracy: %f" % (i, loss, accuracy))

    # 准确率
    def calc_accuracy(self, y, y_hat):
        y_hat = np.argmax(y_hat, axis=1)
        y = np.argmax(y, axis=1)
        return np.sum(y == y_hat) / y.shape[0]

    # 生成batch
    def generate_batches(self, images, labels, batch_size):
        batches = []
        num = images.shape[0]
        for i in range(0, num, batch_size):
            batches.append((images[i:i+batch_size], labels[i:i+batch_size]))
        return batches

    # 评分函数
    def score(self, images, labels):
        # 前馈
        output = self.forward(images)
        # 计算准确率
        accuracy = np.mean(np.argmax(labels, axis=1) == np.argmax(output, axis=1))
        return accuracy
    
    # 预测函数
    def predict(self, x):
        return self.forward(x)
    
    #打印网络结构信息
    def __str__(self) -> str:
        info = 'FNN:\n'
        for layer in self.layers:
            info += str(layer) + '\n'
        
        return info
