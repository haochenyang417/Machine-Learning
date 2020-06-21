# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 22:51:16 2020
softmax模型
@author: xiaohao
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data #input_data下载用于训练和测试的MNIST数据集的源码
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)



# 构建Softmax 回归模型
#使用TensorFlow程序的流程是先创建一个图，然后在session中启动它。

sess = tf.InteractiveSession()
#占位
x=tf.placeholder("float",shape=[None,784]) #784是一张展平的MNIST图片的维度(28*28)
y_=tf.placeholder("float",shape=[None,10])#十分类

# 初始化变量
W=tf.Variable(tf.zeros([784,10]))#卷积核
b=tf.Variable(tf.zeros([10]))#偏置

#变量需要通过seesion初始化后，才能在session中使用
init=tf.global_variables_initializer()
sess.run(init)

# 把向量化后的图片x和卷积核W相乘，加上偏置b，然后计算每个分类的softmax概率值。
y=tf.nn.softmax(tf.matmul(x,W)+b)

#损失函数是目标类别和预测类别之间的交叉熵。
cross_entropy=-tf.reduce_sum(y_*tf.log(y))

#训练模型
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#返回的train_step操作对象，在运行时会使用梯度下降来更新参数。
#整个模型的训练可以通过反复地运行train_step来完成。
for i in range(1000):
    batch=mnist.train.next_batch(100)

#我们都会加载50个训练样本，然后执行一次train_step，
#并通过feed_dict将x 和 y_张量占位符用训练训练数据替代。

    train_step.run(feed_dict={x:batch[0],y_:batch[1]})
    
#返回一个布尔数组。为了计算分类的准确率，
#将布尔值转换为浮点数来代表对、错，然后取平均值。
#评估模型
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))


print(accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
