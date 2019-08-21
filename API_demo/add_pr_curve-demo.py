# coding=utf-8
import numpy as np
from tb_paddle import SummaryWriter

writer = SummaryWriter('./log')

# 生成一个数组，包含 100 个 0/1  
labels_ = np.random.randint(2, size=100)

for step_ in range(10):
    predictions_ = np.random.rand(100)
    
    for num_thresholds_ in range(7, 197, 20):
        tag_ = 'pr_curve-' + str(num_thresholds_)
        writer.add_pr_curve(tag=tag_, 
                            labels=labels_, 
                            predictions=predictions_, 
                            global_step=step_, 
                            num_thresholds=num_thresholds_)

writer.close()
