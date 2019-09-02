# PR CURVES

TensorBoard 的 **PR CURVES** 栏目显示 [Precision-Recall curve](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/README.md) 。

## 机器学习性能评估指标

### 混淆矩阵

* True Positive (真正, TP)：将正类预测为正类数. 
* True Negative (真负, TN)：将负类预测为负类数. 
* False Positive(假正, FP)：将负类预测为正类数 → 误报
* False Negative(假负, FN)：将正类预测为负类数 → 漏报

举个例子，假设我们手上有60个正样本，40个负样本，现要找出所有的正样本。
某系统总共找出50个，其中只有40个是真正的正样本，计算上述各指标。

TP = 40
TN = 30
FP = 10
FN = 20

### 准确率（Accuracy）

准确率的计算公式：

`acc = (TP + TN)/(TP + TN + FP + FN)`

### 错误率 (Error Rate)

错误率的计算公式：

`error_rate = 1 - acc = (FP + FN)/(TP + TN + FP + FN)`

### 精度

精度的计算公式：

`precision = TP/(TP + FP)`

表示被分为正类的样本中实际为正类的比例。

### 召回率

召回率的计算公式：

`recall = TP/(TP + FN)`

## pr-curve 简介

pr-curve 全称为 precison-recall curve，根据预测的概率值及其对应的准确答案来计算 precision-recall，并将结果保存并以折线图形式展示。

class SummaryWriter 中用于打点pr-curve数据的成员函数包括：

* <a href="#1"> add_pr_curve </a>
* <a href="#2"> add_pr_curve_raw </a>

<a name="1"></a>
## Class SummaryWriter 的成员函数 add_pr_curve

函数定义：

```python
def add_pr_curve(self, tag, labels, predictions, global_step=None, 
                 num_thresholds=127, weights=None, walltime=None):
    """根据预测的概率值，以及其对应的标准答案计算 precision-recall 的结果。
    
    :param tag: Data identifier.
    :type tag: string
    :param labels: 标准答案，每一个元素都为 0/1（或者 True/False)。
    :type labels: numpy.array
    :param predictions: 预测结果，The probability that an element be classified as true，
                        Value should be set in [0, 1].
    :type predictions: numpy.array
    :param global_step: Global step value to record.
    :type global_step: int
    :param num_thresholds: Number of thresholds used to draw the curve.
    :type num_thresholds: int
    :param walltime: 实际时间。
    :type walltime: float
    """
```

Demo-1 add_pr_curve-demo.py

```python
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
```

执行以下指令，启动服务器：

```
python add_pr_curve-demo.py
tensorboard --logdir ./log/ --host 0.0.0.0 --port 6066
```

打开浏览器地址 [http://0.0.0.0:6066/](http://0.0.0.0:6066/)，则可在 tensorboard 的**PR CURVES**栏目中查看图表：

<p align="center">
<img src="../screenshots/pr_curve.png" width=600><br/>
图1. Precision-Recall 曲线 <br/>
</p>

其中`Precison`为横坐标，`Recall`为纵坐标。

<a name="2"></a>
## class SummaryWriter 的成员函数 add_pr_curve_raw

```
def add_pr_curve_raw(self, tag, true_positive_counts,
                         false_positive_counts,
                         true_negative_counts,
                         false_negative_counts,
                         precision,
                         recall,
                         global_step=None,
                         num_thresholds=127,
                         weights=None,
                         walltime=None):
    """Adds precision recall curve with raw data.

    :param tag: Data identifier.
    :type tag: string
    :param true_positive_counts: true positive counts.
    :type true_positive_counts: numpy.array
    :param false_positive_counts: false positive counts.
    :type false_positive_counts: numpy.array
    :param true_negative_counts: true negative counts.
    :type true_negative_counts: numpy.array
    :param false_negative_counts: false negative counts.
    :type false_negative_counts: numpy.array
    :param precision: precision
    :type precision: numpy.array
    :param recall: recall
    :type recall: numpy.array
    :param global_step: Global step value to record
    :type global_step: int
    :param num_thresholds: Number of thresholds used to draw the curve.
    :type num_thresholds: int
    :param walltime: Optional override default walltime (time.time()) of event
    :type walltime: float
    """
```

Demo-2 add_pr_curve_raw-demo.py

```python
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
```

运行程序`add_pr_curve_raw-demo.py`，则可在 tensorboard 的**PR CURVES**栏目中查看图表：

<p align="center">
<img src="../screenshots/add_pr_curve_raw.png" width=600><br/>
图2. add_pr_curve_raw - 显示 precision-recall 曲线 <br/>
</p>
