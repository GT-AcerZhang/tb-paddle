# coding=utf-8
import numpy as np
from tb_paddle import SummaryWriter

writer = SummaryWriter('./log')

sample_rate = 44100
data = np.zeros(sample_rate * 5)

for step in range(10):
    for i in range(data.shape[0]):
        data[i] = np.cos(np.pi * np.random.randn())
    
    writer.add_audio('sound_cos', data, step)

writer.close()
