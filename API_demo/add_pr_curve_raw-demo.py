# coding=utf-8
import numpy as np
from tb_paddle import SummaryWriter

writer = SummaryWriter('log')

true_positive_counts = [75, 64, 21, 5, 0]
false_positive_counts = [150, 105, 18, 0, 0]
true_negative_counts = [0, 45, 132, 150, 150]
false_negative_counts = [0, 11, 54, 70, 75] 
precision = [0.3333333, 0.3786982, 0.5384616, 1.0, 0.0]
recall = [1.0, 0.8533334, 0.28, 0.0666667, 0.0]

step = 0 
for threshold in range(7, 207, 20):
    writer.add_pr_curve_raw('prcurve with raw data',
               true_positive_counts, false_positive_counts, 
               true_negative_counts, false_negative_counts, 
               precision, recall, global_step=step,
               num_thresholds=threshold)
    step += 1

writer.close()
