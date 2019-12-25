# coding=utf-8
import numpy as np
import paddle
from tb_paddle import SummaryWriter
import matplotlib
matplotlib.use('TkAgg')

writer = SummaryWriter('log')

BATCH_SIZE = 768
reader_shuffle = paddle.reader.shuffle(
                     paddle.dataset.mnist.train(), buf_size=5120)

train_reader = paddle.batch(reader_shuffle, batch_size=BATCH_SIZE)

mat = np.zeros([BATCH_SIZE, 784])
for step_id, data in enumerate(train_reader()):
    # type(data) : <class 'list'>
    # len(data)  : BATCH_SIZE
    for i in range(len(data)):
        # type(data[i][0]) : <class 'numpy.ndarray'>
        # data[i][0].shape : (784,)
        mat[i] = data[i][0]

video_data = mat.reshape((8, 96, 1, 28, 28))
writer.add_video('mnist_video_fps4', video_data)
writer.add_video('mnist_video_fps1', video_data, fps=1)

writer.close()
