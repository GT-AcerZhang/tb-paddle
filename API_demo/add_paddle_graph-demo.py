# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math, os
import numpy as np
import paddle
import paddle.dataset.conll05 as conll05
import paddle.fluid as fluid
import six
import time
import argparse
from tb_paddle import SummaryWriter

with_gpu = os.getenv('WITH_GPU', '0') != '0'
word_dict, verb_dict, label_dict = conll05.get_dict()
word_dict_len = len(word_dict)
label_dict_len = len(label_dict)
pred_dict_len = len(verb_dict)

mark_dict_len = 2
word_dim = 32
mark_dim = 5
hidden_dim = 512
depth = 8
mix_hidden_lr = 1e-3
IS_SPARSE = True
PASS_NUM = 10
BATCH_SIZE = 10
embedding_name = 'emb'

def parse_args():
    parser = argparse.ArgumentParser("label_semantic_roles")
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help="If set, run the task with continuous evaluation log.")
    parser.add_argument(
        '--use_gpu', type=int, default=0, help="Whether to use GPU or not.")
    parser.add_argument(
        '--num_epochs', type=int, default=1, help="number of epochs.")
    args = parser.parse_args()
    return args


def load_parameter(file_name, h, w):
    with open(file_name, 'rb') as f:
        f.read(16)  # skip header.
        return np.fromfile(f, dtype=np.float32).reshape(h, w)


def db_lstm(word, predicate, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2, mark,
            **ignored):
    with fluid.name_scope("db_lstm"):
        # 8 features
        with fluid.name_scope("predicate_embedding"):
            predicate_embedding = fluid.layers.embedding(
                input=predicate,
                size=[pred_dict_len, word_dim],
                dtype='float32',
                is_sparse=IS_SPARSE,
                param_attr='vemb')

        with fluid.name_scope("mark_embedding"):
            mark_embedding = fluid.layers.embedding(
                input=mark,
                size=[mark_dict_len, mark_dim],
                dtype='float32',
                is_sparse=IS_SPARSE)

        word_input = [word, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2]
        with fluid.name_scope("emb_layers"):
            emb_layers = [fluid.layers.embedding(
                            size=[word_dict_len, word_dim],
                            input=x,
                            param_attr=fluid.ParamAttr(name=embedding_name, trainable=False)
                            ) for x in word_input]
           
        emb_layers.append(predicate_embedding)
        emb_layers.append(mark_embedding)

        with fluid.name_scope("hidden_0"):
            hidden_0_layers = [fluid.layers.fc(
                                input=emb, size=hidden_dim, act='tanh') for emb in emb_layers]
            hidden_0 = fluid.layers.sums(input=hidden_0_layers)
         
        lstm_0 = fluid.layers.dynamic_lstm(
            input=hidden_0,
            size=hidden_dim,
            candidate_activation='relu',
            gate_activation='sigmoid',
            cell_activation='sigmoid')
         
        # stack L-LSTM and R-LSTM with direct edges
        input_tmp = [hidden_0, lstm_0]
         
        for i in range(1, depth):
            layer_name = "Layer_" + str(i)
            with fluid.name_scope(layer_name):
                mix_hidden = fluid.layers.sums(input=[
                    fluid.layers.fc(input=input_tmp[0], size=hidden_dim, act='tanh'),
                    fluid.layers.fc(input=input_tmp[1], size=hidden_dim, act='tanh')])
            
                lstm = fluid.layers.dynamic_lstm(
                    input=mix_hidden,
                    size=hidden_dim,
                    candidate_activation='relu',
                    gate_activation='sigmoid',
                    cell_activation='sigmoid',
                    is_reverse=((i % 2) == 1))
             
                input_tmp = [mix_hidden, lstm]

        with fluid.name_scope("sums"):             
            feature_out = fluid.layers.sums(input=[
                fluid.layers.fc(input=input_tmp[0], size=label_dict_len, act='tanh'),
                fluid.layers.fc(input=input_tmp[1], size=label_dict_len, act='tanh')])
         
        return feature_out


def train(use_cuda, save_dirname=None, is_local=True):
    # define data layers
    word = fluid.layers.data(name='word_data', shape=[1], dtype='int64', lod_level=1)
    predicate = fluid.layers.data(name='verb_data', shape=[1], dtype='int64', lod_level=1)
    ctx_n2 = fluid.layers.data(name='ctx_n2_data', shape=[1], dtype='int64', lod_level=1)
    ctx_n1 = fluid.layers.data(name='ctx_n1_data', shape=[1], dtype='int64', lod_level=1)
    ctx_0 = fluid.layers.data(name='ctx_0_data', shape=[1], dtype='int64', lod_level=1)
    ctx_p1 = fluid.layers.data(name='ctx_p1_data', shape=[1], dtype='int64', lod_level=1)
    ctx_p2 = fluid.layers.data(name='ctx_p2_data', shape=[1], dtype='int64', lod_level=1)
    mark = fluid.layers.data(name='mark_data', shape=[1], dtype='int64', lod_level=1)

    if args.enable_ce:
        fluid.default_startup_program().random_seed = 90
        fluid.default_main_program().random_seed = 90

    # define network topology
    feature_out = db_lstm(**locals())
    target = fluid.layers.data(name='target', shape=[1], dtype='int64', lod_level=1)
    crf_cost = fluid.layers.linear_chain_crf(
        input=feature_out,
        label=target,
        param_attr=fluid.ParamAttr(name='crfw', learning_rate=mix_hidden_lr))

    avg_cost = fluid.layers.mean(crf_cost)

    sgd_optimizer = fluid.optimizer.SGD(
        learning_rate=fluid.layers.exponential_decay(
            learning_rate=0.01,
            decay_steps=100000,
            decay_rate=0.5,
            staircase=True))

    sgd_optimizer.minimize(avg_cost)

    crf_decode = fluid.layers.crf_decoding(
        input=feature_out, param_attr=fluid.ParamAttr(name='crfw'))

    if args.enable_ce:
        train_data = paddle.batch(
            paddle.dataset.conll05.test(), batch_size=BATCH_SIZE)
    else:
        train_data = paddle.batch(
            paddle.reader.shuffle(paddle.dataset.conll05.test(), buf_size=8192),
            batch_size=BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    feeder = fluid.DataFeeder(
        feed_list=[word, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2, predicate, mark, target],
        place=place)
    exe = fluid.Executor(place)

    def train_loop(main_program):
        exe.run(fluid.default_startup_program())
        embedding_param = fluid.global_scope().find_var(
            embedding_name).get_tensor()
        embedding_param.set(
            load_parameter(conll05.get_embedding(), word_dict_len, word_dim),
            place)

        # ?????????????????????????????????
        train_writer = SummaryWriter("log/train")
        train_writer.add_paddle_graph(fluid_program=main_program, verbose=True)
        train_writer.close()
        
        start_time = time.time()
        batch_id = 0
        for pass_id in six.moves.xrange(PASS_NUM):
            for data in train_data():
                cost = exe.run(main_program, feed=feeder.feed(data), fetch_list=[avg_cost])
                cost = cost[0]
                
                if batch_id % 10 == 0:
                    # Set the threshold low to speed up the CI test
                    if float(cost) < 60.0:
                        if save_dirname is not None:
                            # TODO(liuyiqun): Change the target to crf_decode
                            fluid.io.save_inference_model(save_dirname, [
                                'word_data', 'verb_data', 'ctx_n2_data',
                                'ctx_n1_data', 'ctx_0_data', 'ctx_p1_data',
                                'ctx_p2_data', 'mark_data'
                            ], [feature_out], exe)
                        return

                batch_id = batch_id + 1

    train_loop(fluid.default_main_program())


def infer(use_cuda, save_dirname=None):
    if save_dirname is None:
        return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)
        
        # ?????????????????????????????????
        infer_writer = SummaryWriter("./log")
        infer_writer.add_paddle_graph(inference_program)
        infer_writer.close()
        exit()
        
        lod = [[3, 4, 2]]
        base_shape = [1]
        # The range of random integers is [low, high]
        word = fluid.create_random_int_lodtensor(
            lod, base_shape, place, low=0, high=word_dict_len - 1)
        pred = fluid.create_random_int_lodtensor(
            lod, base_shape, place, low=0, high=pred_dict_len - 1)
        ctx_n2 = fluid.create_random_int_lodtensor(
            lod, base_shape, place, low=0, high=word_dict_len - 1)
        ctx_n1 = fluid.create_random_int_lodtensor(
            lod, base_shape, place, low=0, high=word_dict_len - 1)
        ctx_0 = fluid.create_random_int_lodtensor(
            lod, base_shape, place, low=0, high=word_dict_len - 1)
        ctx_p1 = fluid.create_random_int_lodtensor(
            lod, base_shape, place, low=0, high=word_dict_len - 1)
        ctx_p2 = fluid.create_random_int_lodtensor(
            lod, base_shape, place, low=0, high=word_dict_len - 1)
        mark = fluid.create_random_int_lodtensor(
            lod, base_shape, place, low=0, high=mark_dict_len - 1)

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        assert feed_target_names[0] == 'word_data'
        assert feed_target_names[1] == 'verb_data'
        assert feed_target_names[2] == 'ctx_n2_data'
        assert feed_target_names[3] == 'ctx_n1_data'
        assert feed_target_names[4] == 'ctx_0_data'
        assert feed_target_names[5] == 'ctx_p1_data'
        assert feed_target_names[6] == 'ctx_p2_data'
        assert feed_target_names[7] == 'mark_data'

        results = exe.run(
            inference_program,
            feed={
                feed_target_names[0]: word,
                feed_target_names[1]: pred,
                feed_target_names[2]: ctx_n2,
                feed_target_names[3]: ctx_n1,
                feed_target_names[4]: ctx_0,
                feed_target_names[5]: ctx_p1,
                feed_target_names[6]: ctx_p2,
                feed_target_names[7]: mark
            },
            fetch_list=fetch_targets,
            return_numpy=False)


def main(use_cuda, is_local=True):
    save_dirname = "label_semantic_roles.inference.model" # Directory for saving the trained model
    train(use_cuda, save_dirname, is_local)
    infer(use_cuda, save_dirname)

if __name__ == '__main__':
    args = parse_args()
    use_cuda = args.use_gpu
    PASS_NUM = args.num_epochs
    main(use_cuda)
