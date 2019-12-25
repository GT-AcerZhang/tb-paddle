# HPARAMS

TensorBoard 的 **HPARAMS** 栏目是超参调优（Hyperparameter Optimization）的得力工具。

## 1. Domain 超参的取值范围

Tensorboard 将超参的取值范围定义成以下三个类：

* [Discrete](./HPARAMS/Discrete.md)  ：离散值
* [IntInterval](./HPARAMS/IntInterval.md) ：整数间隔
* [RealInterval](./HPARAMS/RealInterval.md) ：实数间隔

## HParam

Hparam 是 HyperParameter 的缩写，即为超参。


[HParam](../tb_paddle/hparams_summary.py) 的初始化函数为：

```python
class HParam(object):
    def __init__(self, name, domain=None, display_name=None, description=None):
        """Create a hyperparameter object.

        :param name: A string for this hyperparameter, which should be unique within an experiment.
        :type name: str
        :param domain: Describing the values that this hyperparameter can take on.
        :type domain: <class `Domain`> object, optional
        :param display_name: 在前端的展示名称
        :type display_name: str, optional
        :param description: 仅被其成员函数 __repr__() 调用
        :type description: str, optional
        """
```

简而言之，每一个 HParam 对象都有独一无二的名称(name)，而且有相应的取值范围(Domain)。

## Metric

Metric，译为度量，它对应于一个实数，是一组超参的实验结果。
当 Metric 对象被传入 SummaryWriter 的成员函数 add_hparams_config 时，才会有作用。

demo-1 Metric_util.py

```python
from tb_paddle import hparams_api as hp
metrics=[hp.Metric('accuracy', display_name='Accuracy')]
tb_writer = SummaryWriter('/PATH/TO/logdir')
# hparams : a list of HParam object
tb_writer.add_hparams_config(hparams, metrics) 

# ...执行代码
tb_writer_run.add_scalar('accuracy', accuracy, 1)
tb_writer_run.add_scalar('loss', loss, 1)
```

示例程序 `Metric_util.py` 中，只有 accuracy 是该组超参实验的结果，而 loss 不是该组超参实验的结果。



