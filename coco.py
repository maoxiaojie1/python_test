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

# 归一化图像处理
def normalize(data):
    """
    归一化图像处理
    """
    # 将数据转换为float64类型
    data = data.astype(np.float64)
    # 将数据归一化到0~1之间
    data = data / 255.0
    return data

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


# 全连接层
class FullyConnectedLayer(object):
    def __init__(self, m:int, n:int, activator="linear") -> None:
        self.weights = np.random.normal(0, 0.01, (m, n))
        self.bias = np.zeros((1, n))
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
    
    def backward(self, delta):
        delta = self.activator.backward(delta)
        self.weights_grad = np.dot(self.input.T, delta)
        self.bias_grad = np.sum(delta, axis=0, keepdims=True)
        # 计算当前层的误差
        return np.dot(delta, self.weights.T)

    def update(self, optimizer):
        optimizer.update([self.weights, self.bias], [self.weights_grad, self.bias_grad])

    def __str__(self):
        return "fully connected layer: " + str(self.weights.shape) + " "  + str(self.activator)

# 输出层
class OutputLayer(object):
    def __init__(self, activator="softmax") -> None:
        self.activator = get_activation(activator)
    
    def forward(self, x):
        return self.activator.forward(x)
    
    def backward(self, delta):
        return self.activator.backward(delta)

    def update(self, optimizer):
        pass

    def __str__(self) -> str:
        return "output: " + str(self.activator)

# 卷积层
class ConvolutionLayer(object):
    def __init__(self, input_shape, filter_shape, stride, padding, activator="linear") -> None:
        # 输入数据的形状
        self.input_shape = input_shape
        # 卷积核的形状
        self.filter_shape = filter_shape
        # 步长
        self.stride = stride
        # 填充
        self.padding = padding
        # 激活函数
        self.activator = get_activation(activator)
        # 初始化卷积核
        self.filters = np.random.normal(0, 0.01, filter_shape)
        # 初始化偏置
        self.bias = np.zeros((filter_shape[0], 1))
        # 梯度
        self.filters_grad = np.zeros(filter_shape)
        self.bias_grad = np.zeros((filter_shape[0], 1))
        # 当前层的输入
        self.input = None
        # 当前层的输出
        self.output = None
    
    def forward(self, x):
        self.input = x
        # 计算输出形状
        output_shape = self.get_output_shape()
        # 初始化输出
        output = np.zeros(output_shape)
        # 计算卷积
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                # 获取当前卷积核的输入
                input = self.get_input(x, i, j)
                # 计算卷积
                output[i, j] = np.sum(input * self.filters) + self.bias
        # 激活函数
        self.output = self.activator.forward(output)
        return self.output
    
    def backward(self, delta):
        # 计算激活函数的梯度
        delta = self.activator.backward(delta)
        # 计算偏置的梯度
        self.bias_grad = np.sum(delta, axis=(1, 2), keepdims=True)
        # 计算卷积核的梯度
        for i in range(self.filter_shape[0]):
            for j in range(self.filter_shape[1]):
                # 获取当前卷积核的输入
                input = self.get_input(self.input, i, j)
                # 计算卷积核的梯度
                self.filters_grad[i, j] = np.sum(input * delta)
        # 计算当前层的误差
        return self.get_delta(delta)

    def update(self, optimizer):
        optimizer.update([self.filters, self.bias], [self.filters_grad, self.bias_grad])

    def get_output_shape(self):
        return (self.input_shape[0] - self.filter_shape[0] + 2 * self.padding) // self.stride + 1, \
            (self.input_shape[1] - self.filter_shape[1] + 2 * self.padding) // self.stride + 1
    
    def get_input(self, x, i, j):
        return x[i * self.stride : i * self.stride + self.filter_shape[0], j * self.stride : j * self.stride + self.filter_shape[1]]

    def get_delta(self, delta):
        # 初始化当前层的误差
        delta = np.zeros(self.input_shape)
        # 计算当前层的误差
        for i in range(self.filter_shape[0]):
            for j in range(self.filter_shape[1]):
                # 获取当前卷积核的输入
                input = self.get_input(self.input, i, j)
                # 计算当前层的误差
                delta[i * self.stride : i * self.stride + self.filter_shape[0], j * self.stride : j * self.stride + self.filter_shape[1]] += self.filters[i, j] * delta
        return delta

    def __str__(self) -> str:
        return "convolution layer: " + str(self.input_shape) + " " + str(self.filter_shape) + " " + str(self.stride) + " " + str(self.padding) + " " + str(self.activator)

# 池化层
class PoolingLayer(object):
    def __init__(self, input_shape, pool_shape, stride, padding, activator="max") -> None:
        # 输入数据的形状
        self.input_shape = input_shape
        # 池化核的形状
        self.pool_shape = pool_shape
        # 步长
        self.stride = stride
        # 填充
        self.padding = padding
        # 激活函数
        self.activator = get_activation(activator)
        # 当前层的输入
        self.input = None
        # 当前层的输出
        self.output = None
    
    def forward(self, x):
        self.input = x
        # 计算输出形状
        output_shape = self.get_output_shape()
        # 初始化输出
        output = np.zeros(output_shape)
        # 计算池化
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                # 获取当前池化核的输入
                input = self.get_input(x, i, j)
                # 计算池化
                output[i, j] = self.activator.forward(input)
        self.output = output
        return self.output
    
    def backward(self, delta):
        # 计算当前层的误差
        return self.get_delta(delta)

    def update(self, optimizer):
        pass

    def get_output_shape(self):
        return (self.input_shape[0] - self.pool_shape[0] + 2 * self.padding) // self.stride + 1, \
            (self.input_shape[1] - self.pool_shape[1] + 2 * self.padding) // self.stride + 1
    
    def get_input(self, x, i, j):
        return x[i * self.stride : i * self.stride + self.pool_shape[0], j * self.stride : j * self.stride + self.pool_shape[1]]

    def get_delta(self, delta):
        # 初始化当前层的误差
        delta = np.zeros(self.input_shape)
        # 计算当前层的误差
        for i in range(self.pool_shape[0]):
            for j in range(self.pool_shape[1]):
                # 获取当前池化核的输入
                input = self.get_input(self.input, i, j)
                # 计算当前层的误差
                delta[i * self.stride : i * self.stride + self.pool_shape[0], j * self.stride : j * self.stride + self.pool_shape[1]] += self.activator.backward(input) * delta
        return delta

    def __str__(self) -> str:
        return "pooling layer: " + str(self.input_shape) + " " + str(self.pool_shape) + " " + str(self.stride) + " " + str(self.padding) + " " + str(self.activator)

# flatten层
class FlattenLayer(object):
    def __init__(self) -> None:
        # 输入数据的形状
        self.input_shape = None
        # 当前层的输入
        self.input = None
        # 当前层的输出
        self.output = None
    
    def forward(self, x):
        self.input = x
        # 计算输出形状
        output_shape = self.get_output_shape()
        # 初始化输出
        output = np.zeros(output_shape)
        # 计算输出
        output = x.reshape(output_shape)
        self.output = output
        return self.output
    
    def backward(self, delta):
        # 计算当前层的误差
        return self.get_delta(delta)

    def update(self, optimizer):
        pass

    def get_output_shape(self):
        return self.input_shape[0] * self.input_shape[1]
    
    def get_delta(self, delta):
        # 初始化当前层的误差
        delta = np.zeros(self.input_shape)
        # 计算当前层的误差
        delta = delta.reshape(self.input_shape)
        return delta

    def __str__(self) -> str:
        return "flatten layer: " + str(self.input_shape)

class DenseLayer(object):
    def __init__(self, input_shape, output_shape, activator="relu") -> None:
        # 输入数据的形状
        self.input_shape = input_shape
        # 输出数据的形状
        self.output_shape = output_shape
        # 激活函数
        self.activator = get_activation(activator)
        # 权重
        self.weights = np.random.randn(self.input_shape, self.output_shape)
        # 偏置
        self.bias = np.random.randn(self.output_shape)
        # 当前层的输入
        self.input = None
        # 当前层的输出
        self.output = None
    
    def forward(self, x):
        self.input = x
        # 计算输出
        output = np.dot(x, self.weights) + self.bias
        # 激活
        output = self.activator.forward(output)
        self.output = output
        return self.output
    
    def backward(self, delta):
        # 计算当前层的误差
        return self.get_delta(delta)

    def update(self, optimizer):
        # 更新权重
        self.weights = optimizer.update(self.weights, self.get_weights_delta())
        # 更新偏置
        self.bias = optimizer.update(self.bias, self.get_bias_delta())

    def get_weights_delta(self):
        # 计算当前层的权重误差
        return np.dot(self.input.T, self.get_delta(self.output))

    def get_bias_delta(self):
        # 计算当前层的偏置误差
        return np.sum(self.get_delta(self.output), axis=0)

    def get_delta(self, delta):
        # 计算当前层的误差
        return self.activator.backward(delta) * delta

    def __str__(self) -> str:
        return "dense layer: " + str(self.input_shape) + " " + str(self.output_shape) + " " + str(self.activator)

# 卷积神经网络
class CNN(object):
    def __init__(self, layers) -> None:
        # 网络层
        self.layers = layers
        # 损失函数
        self.loss = None
        # 激活函数
        self.activator = None
        # 优化器
        self.optimizer = None
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, delta):
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
        return delta
    
    def update(self):
        for layer in self.layers:
            layer.update(self.optimizer)
    
    def train(self, x, y, batch_size, epoch, loss, optimizer, verbose=1):
        # 初始化损失函数
        self.loss = get_loss(loss)
        # 初始化优化器
        self.optimizer = get_optimizer(optimizer)
        # 初始化参数
        # self.init_params()
        # 训练
        for i in range(epoch):
            # 计算损失
            y_hat = self.forward(x)
            loss = self.loss.forward(y_hat, y)
            # 计算梯度
            delta = self.loss.backward(y_hat, y)
            # 反向传播
            self.backward(delta)
            # 更新参数
            self.update()
            # 打印损失
            if verbose == 1:
                print("epoch: %d, loss: %f" % (i, loss))
    
    def predict(self, x):
        return self.activator.forward(self.forward(x))
    
    def init_params(self):
        for layer in self.layers:
            if isinstance(layer, ConvolutionLayer):
                layer.init_params()
            elif isinstance(layer, DenseLayer):
                layer.init_params()
    
    def __str__(self) -> str:
        return "cnn: " + str(self.layers)

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
        # 生成shuffle后的batch
        batches = self.generate_batches(images, labels, batch_size)
        for i in range(num_iters):
            for batch in batches:
                images_batch, labels_batch = batch
                # 前馈
                y_hat = self.forward(images_batch)
                # 计算损失
                loss = self.optimizer.calc_loss(labels_batch, y_hat)
                # 计算梯度
                delta = self.optimizer.loss_grad(labels_batch, y_hat)
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
    # fnn = FNN()
    # # 增加全连接层，输入层节点(28*28), 隐藏层2, 输出层10
    # fnn.add_layer(FullyConnectedLayer(28*28, 400, activator="relu"))
    # fnn.add_layer(FullyConnectedLayer(400, 300, activator="relu"))
    # fnn.add_layer(FullyConnectedLayer(300, 500, activator="relu"))
    # fnn.add_layer(FullyConnectedLayer(500, 10, activator="sigmoid"))
    # # fnn.add_layer(OutputLayer(activator="softmax"))
    # fnn.set_optimizer(SGD(learning_rate=0.01))
    # # 打印网络结构信息
    # print(fnn)
    # train_images = train_images.reshape(-1, 28*28)
    # # 归一化 b
    # train_images = normalize(train_images)
    # fnn.fit(train_images, train_labels, 500, batch_size=64)

    # 实例化业务处理类
    cnn = CNN([
        ConvolutionLayer((1, 28, 28), (6, 1, 5, 5), 3, 3, activator="relu"),
        PoolingLayer((6, 24, 24), (6, 12, 12), 2, 2, activator="max"),
        ConvolutionLayer((6, 12, 12), (16, 5, 5), 3, 3, activator="relu"),
        PoolingLayer((16, 8, 8), (16, 4, 4), 2, 2, activator="max"),
        FlattenLayer(),
        DenseLayer(16*4*4, 10, activator="softmax")
    ])

    cnn.train(train_images, train_labels, 64, 50, "cross_entropy", "sgd")