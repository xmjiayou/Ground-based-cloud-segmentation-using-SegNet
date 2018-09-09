""" skydata.py
    将数据集转换为tfrecord格式
"""

import os
import argparse #用于命令项选项与参数解析的模块
import numpy as np
import tensorflow as tf
from PIL import Image

from utils import make_dirs #如果不存在路径，则创建路径
from tfrecord import *

class_names = np.array(['sky', 'cloud', 'unlabelled'])

#camp为3×3的矩阵
cmap = np.array([
    [0, 0, 0],
    [0, 0, 200],
    [200,200, 200]])

#cb为1×3的矩阵
cb = np.array([
    0.2595,
    0.1826,
   4.5640])

label_info = {
    'name': class_names,
    'num_class': len(class_names), 
    'id': np.arange(len(class_names)),
    'cmap': cmap,
    'cb': cb
}


def parse(line, root):
    line = line.rstrip() #删除string字符串末尾的指定字符，返回生成的新字符串
    line = line.replace('/SegNet/skydata', root)#将root替换'/SegNet/skydata'
    return line.split(',') #以逗号分割line，返回分割后的字符串列表

 #对于文本的每一行，执行上述操作，分割文本
def load_path(txt_path, root):
    with open(txt_path) as f:
         img_gt_pairs= [parse(line, root) for line in f.readlines()]
    return img_gt_pairs


#分割出图片及标注的地址
def load_splited_path(txt_path, root):  
    images = []
    labels = []
    with open(txt_path) as f:
        for line in f.readlines():
            image_path,label_path = parse(line, root)
            images.append(image_path)
            labels.append(label_path)
    return images, labels


def main():
    parser = argparse.ArgumentParser() #创建对象
    parser.add_argument('--outdir', type=str, default='/tmp/data/skydata', 
        help='Path to save tfrecord')  #调用add_argument（）方法添加参数
    parser.add_argument('--indir', type=str, default='data/skydata',
        help='Dataset path.')
    parser.add_argument('--target', type=str, default='train',
        help='train, test, val')
    args = parser.parse_args() #使用parse_args()解析添加的参数

    txt_path = os.path.join(args.indir, '{}.txt'.format(args.target)) #os.path.join()： 将多个路径组合后返回
    pairs= load_path(txt_path, args.indir)

    fname = 'skydata-{}.tfrecord'.format(args.target)
    convert_to_tfrecord(pairs, args.outdir, fname)


if __name__ == '__main__':
    main()
