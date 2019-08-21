# coding=utf-8
from numpy.random import randn
from tb_paddle import SummaryWriter

with SummaryWriter('log') as writer:
    for n_iter in range(100):
        writer.add_scalar('SZ/000858', 123 + 12*randn(), n_iter)
        writer.add_scalar('SZ/000568', 81 + 8*randn(), n_iter)

    writer.add_custom_scalars_multilinechart(
        ['SZ/000858', 'SZ/000568'], category='China', title='SZ')
