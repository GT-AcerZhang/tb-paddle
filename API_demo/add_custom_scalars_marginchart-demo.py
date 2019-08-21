# coding=utf-8
from numpy.random import randn
from tb_paddle import SummaryWriter
import math

with SummaryWriter('log') as writer:
    for n_iter in range(100):
        writer.add_scalar('HK/00700', 360 + math.sqrt(360)*randn(), n_iter)
        writer.add_scalar('HK/000568', 9.78  + math.sqrt(9.78)*randn(), n_iter)
        writer.add_scalar('HK/00001', 80 + math.sqrt(80)*randn(), n_iter)        

    writer.add_custom_scalars_marginchart(
       ['HK/00700', 'HK/00001', 'HK/000568'], category='China', title='HK')
