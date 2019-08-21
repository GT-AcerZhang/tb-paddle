# coding=utf-8
from tb_paddle import SummaryWriter
import numpy as np

batch_num = 16
Channels = 3 
Height = 224 
Width = 180 
img_batch = np.zeros((batch_num, Channels, Height, Width))

H_W = Height * Width
for i in range(batch_num):
    img_batch[i, 0] = np.arange(0, H_W).reshape(Height, Width) / H_W / batch_num * i 
    img_batch[i, 1] = (1 - np.arange(0, H_W).reshape(Height, Width) / H_W) / batch_num * i 

writer = SummaryWriter('./log')
writer.add_images('image_batch', img_batch, 0, dataformats='NCWH')
writer.close()
