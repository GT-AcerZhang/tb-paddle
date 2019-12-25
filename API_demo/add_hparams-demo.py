# coding=utf-8
import os
from tb_paddle import SummaryWriter
from tb_paddle import hparams_api as hp

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

def run(run_dir, hparams, session_num):
    tb_writer_run = SummaryWriter(run_dir)
    tb_writer_run.add_hparams(hparams, trial_id=str(session_num))
    accuracy = hparams[HP_NUM_UNITS] + hparams[HP_DROPOUT] * 100 + len(hparams[HP_OPTIMIZER])
    tb_writer_run.add_scalar('accuracy', accuracy, 1)
    tb_writer_run.close()

session_num = 0
save_dir_name = 'logs'
for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for optimizer in HP_OPTIMIZER.domain.values:
            run_name = "run_{}".format(session_num)
            hparams = {HP_NUM_UNITS: num_units,
                       HP_DROPOUT: dropout_rate,
                       HP_OPTIMIZER: optimizer,}
                       
            run(os.path.join(save_dir_name, run_name), hparams, session_num)
            session_num += 1

