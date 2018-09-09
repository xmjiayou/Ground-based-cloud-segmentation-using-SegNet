""" tfrecord.py
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
from utils import make_dirs


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) #tf.train.Feature指定每个记录各特征的名称和数据类型


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def convert_to_tfrecord(pairs, outdir, name):
    make_dirs(outdir)

    writer = tf.python_io.TFRecordWriter(os.path.join(outdir, name))
    print('Writing', name)
    for image_path, label_path in pairs:
        image = np.array(Image.open(image_path))
        label = np.array(Image.open(label_path))
        height = image.shape[0]
        width = image.shape[1]

        image_raw = image.tostring()  #返回一个使用标准“raw”编码器生成的包含像素数据的字符串
        label_raw = label.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={  
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(image_raw),
            'label_raw': _bytes_feature(label_raw)}))  #tf.train.Example作为基本单位来进行数据读取
        writer.write(example.SerializeToString())
    writer.close()
