""" ops.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages


def xavier_initializer(uniform=True, seed=None, dtype=tf.float32):
    return tf.contrib.layers.xavier_initializer(
        uniform=uniform, seed=seed, dtype=dtype)  #返回一个用于初始化权重的程序Xavier，用来保持每一层的梯度大小都差不多相同
                                                  #uniform或normal分布来随机初始化;seed用来生成随机数；dtype只支持浮点数
                                                  #返回值为初始化权重矩阵

#卷积层 
#输入参数：输入 滤波器个数 滤波器尺寸 步长 是否补零 
def conv2d(incoming, num_filters, filter_size, stride=1, pad='SAME',   
           activation=tf.identity,
           weight_init=xavier_initializer(),
           bias_init=tf.constant_initializer(0.0),
           reuse=False, name="conv2d"):

    x = incoming

    input_shape = incoming.get_shape().as_list()
    filter_shape = [filter_size, filter_size, input_shape[-1], num_filters]  #滤波器尺寸
    strides = [1, stride, stride, 1] #步长
    bias_shape = [num_filters]       #b值的size

    with tf.variable_scope(name, reuse=reuse) as scope:
        weight = tf.get_variable(name + "_weight", filter_shape,initializer=weight_init)
        bias = tf.get_variable(name + "_bias", bias_shape,
                               initializer=bias_init)

        conved = tf.nn.conv2d(x, weight, strides, pad) #以一定步长、是否补零做卷积
        conved = tf.nn.bias_add(conved, bias)  #加上偏置项
        output = activation(conved)     #激活项？

    return output


#最大池化层
def maxpool2d(incoming, pool_size, stride=2, pad='SAME', name="maxpool2d"):
    x = incoming
    input_shape = incoming.get_shape().as_list()
    filter_shape = [1, pool_size, pool_size, 1]
    strides = [1, stride, stride, 1]

    with tf.name_scope(name) as scope:
        pooled = tf.nn.max_pool(x, filter_shape, strides, pad)

    return pooled


#带索引的最大值池化
def maxpool2d_with_argmax(incoming, pool_size=2, stride=2,
                          name='maxpool_with_argmax'):
    x = incoming
    filter_shape = [1, pool_size, pool_size, 1]
    strides = [1, stride, stride, 1]

    with tf.name_scope(name):
        _, mask = tf.nn.max_pool_with_argmax(   #返回一个二维元组(output,argmax)，返回最大值及其相应索引
            x, ksize=filter_shape, strides=strides, padding='SAME')
        mask = tf.stop_gradient(mask)

        pooled = tf.nn.max_pool(         #最大值池化
            x, ksize=filter_shape, strides=strides, padding='SAME')

    return pooled, mask

 
#最近邻上采样层
def upsample(incoming, size, name='upsample'):
    x = incoming
    with tf.name_scope(name) as scope:
        resized = tf.image.resize_nearest_neighbor(x, size=size)
    return resized



#池化索引
# https://github.com/Pepslee/tensorflow-contrib/blob/master/unpooling.py
def maxunpool2d(incoming, mask, stride=2, name='unpool'):
    x = incoming

    input_shape = incoming.get_shape().as_list()
    strides = [1, stride, stride, 1]
    output_shape = (input_shape[0],
                    input_shape[1] * strides[1],
                    input_shape[2] * strides[2],
                    input_shape[3])                  #输出图变成原图的两倍

    flat_output_shape = [output_shape[0], np.prod(output_shape[1:])]  #prod全部元素相乘
    with tf.name_scope(name):
        flat_input_size = tf.size(x)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=mask.dtype),
                                 shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(mask) * batch_range   #返回一个用1填充的跟输入mask形状和类型一致的数组
        b = tf.reshape(b, [flat_input_size, 1]) 
        mask_ = tf.reshape(mask, [flat_input_size, 1])
        mask_ = tf.concat([b, mask_], 1)  #数字1代表在第二维对b与mask进行连接

        x_ = tf.reshape(x, [flat_input_size])
        ret = tf.scatter_nd(mask_, x_, shape=flat_output_shape) #根据mask将x散布到新的初始值为0形状为shape的张量中，输出为shape大小
        ret = tf.reshape(ret, output_shape)
        return ret



#批归一化
# https://github.com/tflearn/tflearn/blob/master/tflearn/layers/normalization.py
def batch_norm(incoming, phase_train,
               epsilon=1e-4, alpha=0.1, decay=0.9,
               beta_init=tf.constant_initializer(0.0),
               gamma_init=tf.random_normal_initializer(mean=1.0, stddev=0.002),
               reuse=False, name='batch_norm'):

    x = incoming

    input_shape = incoming.get_shape().as_list()

    depth = input_shape[-1]
    with tf.variable_scope(name, reuse=reuse) as scope:
        beta = tf.get_variable(name + '_beta', shape=depth,
                               initializer=beta_init, trainable=True)
        gamma = tf.get_variable(name + '_gamma', shape=depth,
                                initializer=gamma_init, trainable=True)

        axes = list(range(len(input_shape) - 1))
        batch_mean, batch_variance = tf.nn.moments(incoming, axes) # channel
        moving_mean = tf.get_variable(
            name + '_moving_mean', shape=depth,
            initializer=tf.zeros_initializer(),
            trainable=False)
        moving_variance = tf.get_variable(
            name + '_moving_variance', shape=depth,
            initializer=tf.constant_initializer(1.0),
            trainable=False)

        def update():
            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, batch_mean, decay, zero_debias=False)
            update_moving_variance = moving_averages.assign_moving_average(
                moving_variance, batch_variance,  decay, zero_debias=False)

            with tf.control_dependencies(
                [update_moving_mean, update_moving_variance]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

        mean, variance = tf.cond(phase_train,
                                 update,
                                 lambda: (moving_mean, moving_variance))#用phase_train决定是采用update还是lambda:...

        output = tf.nn.batch_normalization(
            x, mean, variance, beta, gamma, epsilon)

    return output

#relu激活函数
def relu(incoming, summary=False, name='relu'):
    x = incoming
    with tf.name_scope(name) as scope:
        output = tf.nn.relu(x)
    return output
