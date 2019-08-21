# coding=utf-8
import numpy as np
from tb_paddle import SummaryWriter

writer = SummaryWriter('log')

r = 5 
for i in range(100):
    writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                    'xcosx':i*np.cos(i/r),
                                    'tanx': np.tan(i/r)}, i)
writer.close()
