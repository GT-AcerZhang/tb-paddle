# coding=utf-8
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from tb_paddle import SummaryWriter

fig = plt.figure()
c1 = plt.Circle((0.2, 0.5), 0.2, color='r')
c2 = plt.Circle((0.8, 0.5), 0.2, color='r')
ax = plt.gca()
ax.add_patch(c1)
ax.add_patch(c2)
plt.axis('scaled')

writer = SummaryWriter('log')
writer.add_figure('matplotlib', fig)
writer.close()
