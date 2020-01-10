# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import paddle
import paddle.fluid as fluid
from tb_paddle import SummaryWriter

BATCH_SIZE = 1024
PASS_NUM = 2
USE_GPU = False


def loss_net(hidden, label):
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return prediction, avg_loss, acc, hidden


def build_network(img, label):
    with fluid.name_scope("conv_pool_1"):
        conv_pool_1 = fluid.nets.simple_img_conv_pool(
            input=img,
            filter_size=5,
            num_filters=20,
            pool_size=2,
            pool_stride=2,
            act="relu")

    with fluid.name_scope("batch_norm"):
        batch_norm_1 = fluid.layers.batch_norm(conv_pool_1)

    with fluid.name_scope("conv_pool_2"):
        conv_pool_2 = fluid.nets.simple_img_conv_pool(
            input=batch_norm_1,
            filter_size=5,
            num_filters=50,
            pool_size=2,
            pool_stride=2,
            act="relu")

    return loss_net(conv_pool_2, label)


def train(use_gpu, save_path=None):
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    prediction, avg_loss, acc, weights = build_network(img, label)
    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
            batch_size=BATCH_SIZE)

    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
    train_writer = SummaryWriter(logdir="tb_log/train")

    global_step = -1
    for epoch in range(PASS_NUM):
        for step_id, data in enumerate(train_reader()):
            global_step += 1
            print("Epoch {}, Step {}".format(epoch, step_id))
            metrics = exe.run(
                feed=feeder.feed(data),
                fetch_list=[avg_loss, acc, weights])
             
            # add_scalar 
            train_writer.add_scalar('train/loss', metrics[0], global_step)
            train_writer.add_scalar('train/acc_top1', metrics[1][0], global_step)
            # add_histogram
            train_writer.add_histogram('train/weights', metrics[2][0], global_step)
            # embedding data
            if step_id == 0:
                embedding_data = data 
 
        # add_embedding
        mat = np.zeros([BATCH_SIZE, 784])
        metadata = np.zeros(BATCH_SIZE)
        for i in range(len(embedding_data)):       
            mat[i] = embedding_data[i][0]
            metadata[i] = embedding_data[i][1]

        label_img = mat.reshape(BATCH_SIZE, 1, 28, 28)
        train_writer.add_embedding(mat, metadata, label_img, epoch)
        # add_image
        for i in range(BATCH_SIZE):
            image_data = mat[i].reshape(1, 28, 28)
            train_writer.add_image('image/epoch_{}'.format(epoch), image_data, i, dataformats='CHW')
        # add_video
        video_data = mat.reshape((32, 32, 1, 28, 28))
        train_writer.add_video('video/epoch_{}'.format(epoch), video_data, fps=2)

        if save_path is not None:
            fluid.io.save_inference_model(
                dirname=os.path.join(save_path, str(epoch)),
                feeded_var_names=["img"], 
                target_vars=[prediction],
                executor=exe)

    train_writer.close()


def infer(use_gpu, load_path):
    if load_path is None:
        raise ValueError("load_path can't be None.")
     
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
     
    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, 
            fetch_targets] = fluid.io.load_inference_model(load_path, exe)
     
        # add_paddle_graph
        infer_writer = SummaryWriter(logdir="tb_log/infer_program")
        infer_writer.add_paddle_graph(inference_program)
        infer_writer.close()


if __name__ == '__main__':
    save_path = "recognize_digits_cnn.inference.model"
    train(USE_GPU, save_path)
    infer(USE_GPU, os.path.join(save_path, str(PASS_NUM - 1)))

