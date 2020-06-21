# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 06:32:35 2020
优化模型
@author: xiaohao
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#占位
x = tf.placeholder(tf.float32, [None, 784])#784=28*28
y_ = tf.placeholder(tf.float32, [None, 10])#10分类


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)#产生截断正态分布随机数
    return tf.Variable(initial)

#偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#卷积
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#池化
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#第一层，一个卷积接一个max pooling
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


#dropout
#用一个placeholder来代表一个神经元的输出在drop中保持不变的概率
#在训练过程中启用dropout，在测试过程中关闭dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#训练与评估模型
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))#交叉熵，即损失函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))#判断是否相等
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))#平均相等个数

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)#从集合中选取50个训练数据
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    saver.save(sess, 'WModel/model.ckpt')

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
















































































