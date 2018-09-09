""" train.py
"""

import os
import sys
import time
import logging
import importlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from datetime import datetime

from utils import make_dirs
from inputs import read_and_decode

flags = tf.app.flags
FLAGS = flags.FLAGS

#tf定义了tf.app.flags，用于支持接受命令行传递参数，相当于接受argv。
# Basic arguments
flags.DEFINE_string('arch', 'segnet', 'Network architecure')#参数名称、默认值、参数描述
flags.DEFINE_string('outdir', 'output/skydata', 'Output directory')

# Dataset arguments
flags.DEFINE_string('dataset', 'skydata', 'Dataset name') 
flags.DEFINE_string('tfrecord',
    '/tmp/data/skydata/skydata-train.tfrecord', 'TFRecord path')

# Model arguments
flags.DEFINE_integer('channel', 3, 'Channel of an input image') #输入图像的通道
flags.DEFINE_integer('num_class', 3, 'Number of class to classify') #需要分类的类别数
flags.DEFINE_integer('height', 224, 'Input height') #输入图像的高
flags.DEFINE_integer('width', 224, 'Input width')   #输入图像的宽

# Training arguments
flags.DEFINE_integer('batch_size', 5, 'Batch size')      #每一批输入5张图片
flags.DEFINE_integer('iteration', 8000, 'Number of training iterations') #迭代次数
flags.DEFINE_integer('num_threads', 8, 'Number of threads to read batches')  #批量读取线程数
flags.DEFINE_integer('min_after_dequeue', 10, 'min_after_dequeue') #队列做dequeue（取数据）操作后，queue runner线程要保证队列
                                                                   #中至少剩下min_after_dequeue个数据，设置过少，达不到混合效果
flags.DEFINE_integer('seed', 1234, 'Random seed')    #种子随机数
flags.DEFINE_integer('snapshot', 2000, 'Snapshot')   #snapshot？？
flags.DEFINE_integer('print_step', 1, 'Number of step to print training log')   #每次训练都输出结果
flags.DEFINE_string('optimizer', 'sgd', 'optimizer')    #随机梯度下降法
flags.DEFINE_float('learning_rate', 0.6, 'learning rate')  #学习率
flags.DEFINE_float('momentum', 0.9, 'momentum')   #冲量
flags.DEFINE_boolean('cb', False, 'Class Balancing') #class balancing??


np.random.seed(FLAGS.seed) #随机数生成器，使用相同随机数种子，每次生成的随机数相同
tf.set_random_seed(FLAGS.seed)#图形种子？？


def save_model(sess, saver, step, outdir, message):
    print('Saving...')
    saver.save(sess, outdir + '/model', global_step=step) #在outdir + '/model'路径下创建模型文件，在迭代setp次数后，保存模型，生成checkpoint等文件
    logging.info(message)
    print(message)


def train(model_dir, summary_dir):
    logging.info('Training {}'.format(FLAGS.arch))   #保存 training segnet 
    logging.info('FLAGS: {}'.format(FLAGS.__flags))  #保存所有flags参数
    print(FLAGS.__flags)

    graph = tf.Graph()
    with graph.as_default():
        dataset = importlib.import_module(FLAGS.dataset)  #载入skydata模块作为数据集dataset

        fn_queue = tf.train.string_input_producer([FLAGS.tfrecord]) #将skydata-train.tfrecord作为序列
        images, labels = read_and_decode(      #read_and_decode函数，返回打乱的图像以及标注
            fn_queue=fn_queue,
            target_height=FLAGS.height,
            target_width=FLAGS.width,
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_threads,
            min_after_dequeue=FLAGS.min_after_dequeue,
            shuffle=True)

        phase_train = tf.placeholder(tf.bool, name='phase_train') 

        model = importlib.import_module(FLAGS.arch)  #载入模块segnet
        logits = model.inference(images, phase_train,name='logits') #得出卷积后的图像logits
        acc = model.acc(logits, labels,name='accuracy')  #调用segnet中的acc函数，返回准确率

        if FLAGS.cb:
            loss = model.loss(logits, labels, cb=dataset.label_info['cb']) #损失函数权重
        else:
            loss = model.loss(logits, labels)

        summary = model.setup_summary(loss, acc)

        train_op = model.train_op(loss, FLAGS.optimizer,
            lr=FLAGS.learning_rate, momentum=FLAGS.momentum,name='train_op')   #降低loss算法

        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))           

        writer = tf.summary.FileWriter(summary_dir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        start_time = time.time()

        step = 0
        logging.info('Start training...')  #日志文件 确认一切按预期进行
        try:
            while not coord.should_stop():  #should_stop()如果线程应该停止则返回True
                feed_dict = {phase_train: True}
                _, loss_value, acc_value, summary_str = sess.run(
                    [train_op, loss, acc, summary], feed_dict=feed_dict)

                duration = time.time() - start_time
                message = \
                    'arch: {} '.format(FLAGS.arch) + \
                    'step: {} '.format(step + 1) + \
                    'loss: {:.3f} '.format(loss_value) + \
                    'acc: {:.3f} '.format(acc_value) + \
                    'duration: {:.3f}sec '.format(duration) + \
                    'time_per_step: {:.3f}sec'.format(duration / (step + 1))

                writer.add_summary(summary_str, step) 

                if not step % FLAGS.print_step:
                    print(message)
                    logging.info(message)

                if not step % FLAGS.snapshot and not step == 0:
                    message = 'Done training for {} steps.'.format(step)
                    save_model(sess, saver, step, model_dir, message)

                if step == FLAGS.iteration: break

                step += 1

        except KeyboardInterrupt:
            coord.request_stop()  #请求该线程停止

        finally:
            coord.request_stop()

        coord.join(threads)


def main(_):
    outdir = os.path.join(FLAGS.outdir, FLAGS.arch)
    trained_model = os.path.join(outdir, 'trained_model')
    summary_dir = os.path.join(outdir, 'summary')

    make_dirs(trained_model)

    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename='{}/train.log'.format(outdir),
        filemode='w', level=logging.INFO)

    train(trained_model, summary_dir)


if __name__ == '__main__':
    tf.app.run()
