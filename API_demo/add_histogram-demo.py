# coding=utf-8
from tb_paddle import SummaryWriter
import numpy as np

writer = SummaryWriter('./log')

for step in range(1, 101):
    interval_start = 1 + 2 * step / 100.0
    interval_end = 6 - 2 * step / 100.0
    data = np.random.uniform(interval_start, interval_end, size=(10000))
    writer.add_histogram('Distribution Centers', data, step)

writer.close()
