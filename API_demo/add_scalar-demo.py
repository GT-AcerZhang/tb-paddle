# coding=utf-8
from tb_paddle import SummaryWriter

writer = SummaryWriter(logdir='./log')

for i in range(100):
    writer.add_scalar('y = 2x', i * 2, i)

writer.close()
