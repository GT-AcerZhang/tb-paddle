# coding=utf-8
import os
from tb_paddle import SummaryWriter
from tb_paddle import hparams_api as hp

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
HP_LAYERS = hp.HParam('layers', hp.IntInterval(30, 50))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.3))


def run(run_dir, hparams, session_num):
    tb_writer_run = SummaryWriter(run_dir)
    tb_writer_run.add_hparams(hparams, trial_id=str(session_num))

    # 定义该组超参对应的实验结果，比如为 acc, loss
    accuracy = hparams[HP_NUM_UNITS] + \
        hparams[HP_DROPOUT] * 100 + \
        hparams[HP_LAYERS] + \
        len(hparams[HP_OPTIMIZER])

    loss = accuracy / 100

    # 必须使用 add_scalar 打点实验结果
    tb_writer_run.add_scalar('accuracy', accuracy, 1)
    tb_writer_run.add_scalar('loss', loss, 1)
    tb_writer_run.close()


session_num = 0
save_dir_name = 'log'
for num_units in HP_NUM_UNITS.domain.values:
    for optimizer in HP_OPTIMIZER.domain.values:
        for layers in (HP_LAYERS.domain.min_value, HP_LAYERS.domain.max_value):
            dropout_rate = HP_DROPOUT.domain.sample_uniform()
            run_name = "run_{}".format(session_num)
            hparams = {
                HP_NUM_UNITS: num_units,
                HP_OPTIMIZER: optimizer,
                HP_LAYERS: layers,
                HP_DROPOUT: dropout_rate
                }
                       
            run(os.path.join(save_dir_name, run_name), hparams, session_num)
            session_num += 1

