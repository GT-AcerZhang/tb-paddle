# coding=utf-8
import os
from tb_paddle import SummaryWriter
from tb_paddle import hparams_api as hp

HP_NUM_UNITS = hp.HParam(
    name='num_units', 
    domain=hp.Discrete([16, 32]),
    display_name="NUM_UNITS"
    )

HP_OPTIMIZER = hp.HParam(
    name='optimizer', 
    domain=hp.Discrete(['adam', 'sgd']),
    display_name="OPTIMIZER"
    )

HP_LAYERS = hp.HParam(
    name='layers',
    domain=hp.IntInterval(30, 50),
    display_name="NET_LAYERS"
    )

HP_DROPOUT = hp.HParam(
    name='dropout', 
    domain=hp.RealInterval(0.1, 0.3),
    display_name='DROPOUT')

hparams=[HP_NUM_UNITS, HP_OPTIMIZER, HP_LAYERS, HP_DROPOUT]
# 只有 tag 为 'accuracy' 的 scalar 才是该组超参实验的结果 
metrics=[hp.Metric('accuracy', display_name='ACCURACY')]
tb_writer = SummaryWriter('log')
tb_writer.add_hparams_config(hparams, metrics)

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
