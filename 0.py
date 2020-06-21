# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:57:05 2019

@author: xiaohao
"""


"""
将MNIST数据集由二进制文件转为图片形式，保存于指定文件夹下
"""
import os
import struct
import numpy as np
import matplotlib.pyplot as plt

# 读MNIST数据集的图片数据
def mnist_load_img(img_path):
    with open(img_path, "rb") as fp:
        # >是以大端模式读取，i是整型模式，读取前四位的标志位，
        # unpack()函数：是将4个字节联合后再解析成一个数，(读取后指针自动后移)
        msb = struct.unpack('>i', fp.read(4))[0]
        # 标志位为2051，后存图像数据；标志位为2049，后存图像标签
        if msb == 2051:
            # 读取样本个数60000，存入cnt
            cnt = struct.unpack('>i', fp.read(4))[0]
            # rows：行数28；cols：列数28
            rows = struct.unpack('>i', fp.read(4))[0]
            cols = struct.unpack('>i', fp.read(4))[0]
            imgs = np.empty((cnt, rows, cols), dtype="int")
            for i in range(0, cnt):
                for j in range(0, rows):
                    for k in range(0, cols):
                        # 16进制转10进制
                        pxl = int(hex(fp.read(1)[0]), 16)
                        imgs[i][j][k] = pxl
            return imgs
        else:
            return np.empty(1)

# 读MNIST数据集的图片标签
def mnist_load_label(label_path):
    with open(label_path, "rb") as fp:
        msb = struct.unpack('>i', fp.read(4))[0];
        if msb == 2049:
            cnt = struct.unpack('>i', fp.read(4))[0];
            labels = np.empty(cnt, dtype="int");
            for i in range(0, cnt):
                label = int(hex(fp.read(1)[0]), 16);
                labels[i] = label;
            return labels;
        else:
            return np.empty(1);

# 分割训练、测试集的图片数据与图片标签
def mnist_load_data(train_img_path, train_label_path, test_img_path, test_label_path):
    x_train = mnist_load_img(train_img_path);
    y_train = mnist_load_label(train_label_path);
    x_test = mnist_load_img(test_img_path);
    y_test = mnist_load_label(test_label_path);
    return (x_train, y_train), (x_test, y_test);

# 输出打印图片
def mnist_plot_img(img):
    (rows, cols) = img.shape;
    plt.figure();
    plt.gray();
    plt.imshow(img);
    plt.show();

# 按指定位置保存图片
def mnist_save_img(img, path, name):
    if not os.path.exists(path):
        os.mkdir(path)
    (rows, cols) = img.shape
    fig = plt.figure()
    plt.gray()
    plt.imshow(img)
    # 在既定路径里保存图片
    fig.savefig(path + name)



# [start]
x_train = mnist_load_img("train-images.idx3-ubyte")
y_train = mnist_load_label("train-labels.idx1-ubyte")

# 将打印出的MNIST数据集中所有的图片存入一个data文件夹下
"""for i in range(0, 10):
    path = "D:\daima\data\data_a"
    name = str(i) + ".png"
    mnist_save_img(x_train[i], path, name)
"""
# 按图片标签的不同，打印MNIST数据集的图片存入不同文件夹下
for i in range (31,32):#任意标签的数据
    path = "D:\daima\data\data_c" + str(y_train[i]) +"/"
    name = str(i)+".png"
    mnist_save_img(x_train[i], path, name)
    print(y_train[i])



#mnist_plot_img(x_train[0, :, :])
"""
x_test = mnist_load_img("t10k-images.idx3-ubyte")
y_test = mnist_load_label("t10k-labels.idx1-ubyte")
"""
