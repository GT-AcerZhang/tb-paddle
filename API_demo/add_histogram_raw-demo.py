# coding=utf-8
import numpy as np
import random
from tb_paddle import SummaryWriter

dummy_data = []

for idx, value in enumerate(range(30)):
    for i in range(idx):
        dummy_data += [idx + random.random()]

values = np.array(dummy_data).astype(float).reshape(-1)
counts, limits = np.histogram(values)
sum_sq = values.dot(values)

with SummaryWriter('./log') as summary_writer:
    summary_writer.add_histogram_raw(
            tag='hist_dummy_data',
            min=values.min(),
            max=values.max(),
            num=len(values),
            sum=values.sum(),
            sum_squares=sum_sq,
            bucket_limits=limits[1:],
            bucket_counts=counts,
            global_step=0)
