# coding=utf-8
from __future__ import print_function

import os
import argparse
from PIL import Image
import numpy as np
import paddle
import paddle.fluid as fluid
import matplotlib
matplotlib.use('TkAgg')

from tb_paddle import SummaryWriter

data_writer = SummaryWriter(logdir="log/data")

def parse_args():
    parser = argparse.ArgumentParser("mnist")
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help="If set, run the task with continuous evaluation logs.")
    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=False,
        help="Whether to use GPU or not.")
    parser.add_argument(
        '--num_epochs', type=int, default=1, help="number of epochs.")
    args = parser.parse_args()
    return args


def loss_net(hidden, label):
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return prediction, avg_loss, acc, hidden


def convolutional_neural_network(img, label):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    return loss_net(conv_pool_2, label)


def train(nn_type,
          use_cuda,
          save_dirname=None,
          model_filename=None,
          params_filename=None):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    startup_program = fluid.default_startup_program()
    main_program = fluid.default_main_program()

    if args.enable_ce:
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE)
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)
        startup_program.random_seed = 90
        main_program.random_seed = 90
    else:
        train_reader = paddle.batch(
            paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
            batch_size=BATCH_SIZE)
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)

    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if nn_type == 'softmax_regression':
        net_conf = softmax_regression
    elif nn_type == 'multilayer_perceptron':
        net_conf = multilayer_perceptron
    else:
        net_conf = convolutional_neural_network

    prediction, avg_loss, acc, weights = net_conf(img, label)

    test_program = main_program.clone(for_test=True)
    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
    exe.run(startup_program)
    epochs = [epoch_id for epoch_id in range(PASS_NUM)]

    lists = []
    step = 0
    for epoch_id in epochs:
        data_id0 = []
        for step_id, data in enumerate(train_reader()):
            if step_id == 0:
                data_id0 = data 
            
            metrics = exe.run(main_program,
                              feed=feeder.feed(data),
                              fetch_list=[avg_loss, acc, weights])
            
            # 用 add_scalar 打点标量数据
            data_writer.add_scalar("train/avg_loss", metrics[0], step)
            data_writer.add_scalar("train/acc", metrics[1][0], step)
            
            # 用add_histogram 打点向量数据
            data_writer.add_histogram("train/weights", metrics[2][0], step)
        
            step += 1
            print("Step:", step)
       
        # 用 add_embedding 增加PROJECTOR 数据
        mat = np.zeros([BATCH_SIZE, 784])
        metadata = np.zeros(BATCH_SIZE)

        for i in range(len(data_id0)):       
            mat[i] = data_id0[i][0]
            metadata[i] = data_id0[i][1]

        label_img = mat.reshape(BATCH_SIZE, 1, 28, 28)
        data_writer.add_embedding(mat=mat, metadata=metadata, label_img=label_img, global_step=epoch_id)

        # 用 add_image 打点图片数据
        for i in range(BATCH_SIZE):
            image_data = mat[i].reshape(1, 28, 28)
            data_writer.add_image("mnist/picture", image_data, i, dataformats='CHW')

        # 用 add_video 打点视频数据
        video_data = mat.reshape((32, 32, 1, 28, 28))
        data_writer.add_video('mnist_video_fps_2', vid_tensor=video_data, fps=2)

        if save_dirname is not None:
            fluid.io.save_inference_model(
                save_dirname, ["img"], [prediction],
                exe,
                model_filename=model_filename,
                params_filename=params_filename)


def infer(use_cuda, save_dirname=None, model_filename=None, params_filename=None):
    if save_dirname is None:
        return
     
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
     
    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(
             save_dirname, exe, model_filename, params_filename)
     
        # 通过 add_paddle_graph 添加fluid.program，从而画出计算图
        infer_writer = SummaryWriter(logdir="log/infer_program")
        infer_writer.add_paddle_graph(fluid_program=inference_program, verbose=True)
        infer_writer.close()
     
     
def main(use_cuda, nn_type):
    model_filename = None
    params_filename = None
    save_dirname = "recognize_digits_" + nn_type + ".inference.model"

    train(nn_type=nn_type, use_cuda=use_cuda, save_dirname=save_dirname,
          model_filename=model_filename, params_filename=params_filename)
    
    infer(use_cuda=use_cuda, save_dirname=save_dirname,
        model_filename=model_filename, params_filename=params_filename)


if __name__ == '__main__':
    args = parse_args()
    BATCH_SIZE = 1024
    PASS_NUM = args.num_epochs
    use_cuda = args.use_gpu
    predict = 'convolutional_neural_network'
    main(use_cuda=use_cuda, nn_type=predict)
    data_writer.close()
