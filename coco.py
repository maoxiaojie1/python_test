import numpy as np
import struct
import os

# 文件路径
data_path = r'data'
file_names = [
  'train-images-idx3-ubyte',
  'train-labels-idx1-ubyte',
  't10k-images-idx3-ubyte',
  't10k-labels-idx1-ubyte'
]

def decode_idx3_ubyte(file):
    """
    解析数据文件
    """
    # 读取二进制数据
    with open(file, 'rb') as fp:
        bin_data = fp.read()
        fp.close()
    
    # 解析文件中的头信息
    # 从文件头部依次读取四个32位，分别为：
    # magic，numImgs, numRows, numCols
    # 偏置
    offset = 0
    # 读取格式: 大端
    fmt_header = '>iiii'
    magic, numImgs, numRows, numCols = struct.unpack_from(fmt_header, bin_data, offset)
    # print(magic,numImgs,numRows,numCols)
    
    # 解析图片数据
    # 偏置掉头文件信息
    offset = struct.calcsize(fmt_header)
    # 读取格式
    fmt_image = '>'+str(numImgs*numRows*numCols)+'B'
    data = struct.unpack_from(fmt_image, bin_data, offset)
    # 转换成numpy数组
    data = np.array(data).reshape((numImgs, 1, numRows, numCols))
    return data

# one-hot编码
def one_hot(labels, num_classes):
    """
    将标签转换为one-hot编码
    """
    # 创建一个与标签数量相同的全0数组
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    # 将标签对应的索引位置的值置为1
    one_hot_labels[np.arange(labels.shape[0]), labels] = 1
    return one_hot_labels

def decode_idx1_ubyte(file):
    """
    解析标签文件
    """
    # 读取二进制数据
    with open(file, 'rb') as fp:
        bin_data = fp.read()
        fp.close()
    
    # 解析文件中的头信息
    # 从文件头部依次读取两个个32位，分别为：
    # magic，numImgs
    # 偏置
    offset = 0
    # 读取格式: 大端
    fmt_header = '>ii'
    magic, numImgs = struct.unpack_from(fmt_header, bin_data, offset)
    # print(magic,numImgs)
    
    # 解析图片数据
    # 偏置掉头文件信息
    offset = struct.calcsize(fmt_header)
    # 读取格式
    fmt_image = '>'+str(numImgs)+'B'
    data = struct.unpack_from(fmt_image, bin_data, offset)
    # 转换成numpy数组
    data = np.array(data)
    one_hot_data = one_hot(data, 10)
    return one_hot_data

# 获取训练集和标签
def get_train_data():
    """
    获取训练集和标签
    """
    # 获取训练集
    train_images = decode_idx3_ubyte(os.path.join(data_path, file_names[0]))
    # 获取标签
    train_labels = decode_idx1_ubyte(os.path.join(data_path, file_names[1]))
    return train_images, train_labels

# 获取测试集和标签
def get_test_data():
    """
    获取测试集和标签
    """
    # 获取测试集
    test_images = decode_idx3_ubyte(os.path.join(data_path, file_names[2]))
    # 获取标签
    test_labels = decode_idx1_ubyte(os.path.join(data_path, file_names[3]))
    return test_images, test_labels

# 激活函数sigmoid
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

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
        # 指数归一化
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        # 计算softmax
        self.y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.y
    def backward(self, delta):
        """
        反向传播
        """
        # 计算梯度
        dx = self.y * (1 - self.y) * delta
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
        # 保存正向传播的输入
        self.mask = (x <= 0)
        # 计算relu
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


# 全连接层
class FullyConnectedLayer(object):
    def __init__(self, m:int, n:int, activator="linear") -> None:
        self.weights = np.random.normal(0.0, 1.0, (m, n))
        self.bias = np.random.normal(0.0, 1.0, (1, n))
        self.activator = get_activation(activator)
        # 梯度
        self.weights_grad = np.zeros((m, n))
        self.bias_grad = np.zeros((1, n))
        # 当前层的输入
        self.input = None
        # 当前层的输出
        self.output = None
    
    def forward(self, x):
        self.input = x
        self.output = np.dot(x, self.weights) + self.bias
        return self.activator.forward(self.output)
    
# SGD 优化器
class SGD(object):
    def __init__(self, learning_rate=0.01) -> None:
        self.learning_rate = learning_rate
    
    def update(self, layer):
        layer.weights -= self.learning_rate * layer.weights_grad
        layer.bias -= self.learning_rate * layer.bias_grad

    # 反向传播
    def backward(self, layers, delta):
        for layer in layers[::-1]:
          # 计算梯度
          delta = layer.activator.backward(delta)
          layer.weights_grad = np.dot(layer.input.T, delta)
          layer.bias_grad = np.sum(delta, axis=0, keepdims=True)
          # 计算当前层的误差
          delta = np.dot(delta, layer.weights.T)
          self.update(layer)


    # 交叉熵损失函数
    def loss(self, y, y_hat):
        return -np.sum(y * np.log(y_hat + 1e-7)) / y.shape[0]

    # 计算损失函数
    def calc_loss(self, y, y_hat):
        return self.loss(y, y_hat)
    
    # 交叉熵损失函数的梯度
    def loss_grad(self, y, y_hat):
        return -y / (y_hat + 1e-9) + (1 - y) / (1 - y_hat + 1e-9)


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
    def fit(self, images, labels, num_iters, reset_weights=False):
        # 重置权重
        if reset_weights:
            for layer in self.layers:
                layer.weights = np.random.normal(0.0, 1.0, layer.weights.shape)
                layer.bias = np.random.normal(0.0, 1.0, layer.bias.shape)
        # 训练
        for i in range(num_iters):
            # 前馈
            output = self.forward(images)
            # 计算误差
            loss = self.optimizer.loss_grad(labels, output)
            # 计算准确率
            accuracy = np.mean(np.argmax(labels, axis=1) == np.argmax(output, axis=1))
            # 打印误差和准确率
            print('iter: %d, loss: %f, accuracy: %f' % (i, np.mean(loss), accuracy))
            # 反向传播
            self.optimizer.backward(self.layers, loss)

      
    # 评分函数
    def score(self, images, labels):
        # 前馈
        output = self.forward(images)
        # 计算准确率
        accuracy = np.mean(np.argmax(labels, axis=1) == np.argmax(output, axis=1))
        return accuracy
    
    # 预测函数
    def predict(self, x):
        # 前馈
        output = self.forward(x)
        # 返回预测结果
        return np.argmax(output, axis=1)
    
    # 打印网络结构信息
    def __str__(self) -> str:
        info = 'FNN:\n'
        for layer in self.layers:
            info += str(layer.weights.shape) + ''
        
        return info
    
    # 网络类型
    def __repr__(self) -> str:
        return 'FNN'
    

        
if __name__ == '__main__':
    # 获取训练集和标签
    train_images, train_labels = get_train_data()
    # 获取测试集和标签
    test_images, test_labels = get_test_data()

    # 打印训练集和标签的数量
    print('train_images.shape: ', train_images.shape)
    print('train_labels.shape: ', train_labels.shape)
    # 打印测试集和标签的数量
    print('test_images.shape: ', test_images.shape)
    print('test_labels.shape: ', test_labels.shape)
    
    # 实例化业务处理类
    fnn = FNN()
    # 增加全连接层，输入层节点(28*28), 隐藏层2, 输出层10
    fnn.add_layer(FullyConnectedLayer(28*28, 350, activator="relu"))
    fnn.add_layer(FullyConnectedLayer(350, 10, activator="sigmoid"))
    # 打印网络结构信息
    print(fnn)
    train_images = train_images.reshape(-1, 28*28)
    fnn.fit(train_images, train_labels, 20)