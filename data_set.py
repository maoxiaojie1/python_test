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