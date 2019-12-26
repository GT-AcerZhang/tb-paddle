# HPARAMS

TensorBoard 的 **HPARAMS** 栏目是超参调优（Hyperparameter Optimization）的得力工具。

## 1. Domain 超参的取值范围

Tensorboard 将超参的取值范围定义成以下三个类：

* [Discrete](./HPARAMS/Discrete.md)  ：离散值
* [IntInterval](./HPARAMS/IntInterval.md) ：整数间隔
* [RealInterval](./HPARAMS/RealInterval.md) ：实数间隔

## 2. HParam

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

## 3. Metric

Metric，译为度量，它对应于一个实数(scalar)，是一组超参的实验结果。
只有当 Metric 对象被传入 SummaryWriter 的成员函数 add_hparams_config 时，才会发挥作用。

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

## 4. 数据记录 API 

class SummaryWriter 中用于打点记录 HParam 和 Metric 的 API 包括：

* add_hparams
* add_hparams_config


这两个 API 的定义与实现均在
文件[../tb_paddle/summary_writer.py](../tb_paddle/summary_writer.py) 中。

### class SummaryWriter 的成员函数 add_hparams

add_hparams 用于打点记录 HParam 对象，add_hparams 的函数定义为：

```python
def add_hparams(self, hparams, trial_id=None, start_time_secs=None):
    """Write hyperparameter values for a single trial.
    
    :type hparams: dict
    :param hparams: hparams 为字典，字典的 key 为 HParam 对象或者 HParam 对象的 name,
        字典的 value 为该组实验中超参的实际取值，
        数据类型仅限于 `bool`, `int`, `float`, or `string`。
    
    :param trial_id: 实验 id
    :type trial_id: str, optional
    :param start_time_secs: 实验开始时间
    """
```

demo-2 add_hparams-demo.py

```python
# coding=utf-8
import os
from tb_paddle import SummaryWriter
from tb_paddle import hparams_api as hp

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
HP_LAYERS = hp.HParam('layers', hp.IntInterval(30, 50))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.3))


def run(run_dir, hparams, session_num):
    tb_writer_run = SummaryWriter(run_dir)
    tb_writer_run.add_hparams(hparams, trial_id=str(session_num))

    # 定义该组超参对应的实验结果，比如为 acc, loss
    accuracy = hparams[HP_NUM_UNITS] + \
        hparams[HP_DROPOUT] * 100 + \
        hparams[HP_LAYERS] + \
        len(hparams[HP_OPTIMIZER])

    loss = accuracy / 100

    # 必须使用 add_scalar 打点实验结果
    tb_writer_run.add_scalar('accuracy', accuracy, 1)
    tb_writer_run.add_scalar('loss', loss, 1)
    tb_writer_run.close()


session_num = 0
save_dir_name = 'logs'
for num_units in HP_NUM_UNITS.domain.values:
    for optimizer in HP_OPTIMIZER.domain.values:
        for layers in (HP_LAYERS.domain.min_value, HP_LAYERS.domain.max_value):
            dropout_rate = HP_DROPOUT.domain.sample_uniform()
            run_name = "run_{}".format(session_num)
            hparams = {
                HP_NUM_UNITS: num_units,
                HP_OPTIMIZER: optimizer,
                HP_LAYERS: layers,
                HP_DROPOUT: dropout_rate
                }
                       
            run(os.path.join(save_dir_name, run_name), hparams, session_num)
            session_num += 1
```


### class SummaryWriter 的成员函数 add_hparams_config

add_hparams_config 函数的输入参数为 HParam object 和 Metric object，
add_hparams_config 的作用有两个：

* HParam 和 Metric 的 display_name 将被解析为网页界面的显示名称；
* Metric 指定的 scalar 为该组超参对应的实验结果。

demo-2 add_hparams_config-demo.py

```python
# coding=utf-8
import os
from tb_paddle import SummaryWriter
from tb_paddle import hparams_api as hp

HP_NUM_UNITS = hp.HParam(
    name='num_units',
    domain=hp.Discrete([16, 32]),
    display_name="NUM_UNITS"
    )

HP_OPTIMIZER = hp.HParam(
    name='optimizer',
    domain=hp.Discrete(['adam', 'sgd']),
    display_name="OPTIMIZER"
    )

HP_LAYERS = hp.HParam(
    name='layers',
    domain=hp.IntInterval(30, 50),
    display_name="NET_LAYERS"
    )

HP_DROPOUT = hp.HParam(
    name='dropout',
    domain=hp.RealInterval(0.1, 0.3),
    display_name='DROPOUT')

hparams=[HP_NUM_UNITS, HP_OPTIMIZER, HP_LAYERS, HP_DROPOUT]
# 只有 tag 为 'accuracy' 的 scalar 才是该组超参实验的结果 
metrics=[hp.Metric('accuracy', display_name='ACCURACY')]
tb_writer = SummaryWriter('logs')
tb_writer.add_hparams_config(hparams, metrics)

def run(run_dir, hparams, session_num):
    tb_writer_run = SummaryWriter(run_dir)
    tb_writer_run.add_hparams(hparams, trial_id=str(session_num))

    # 定义该组超参对应的实验结果，比如为 acc, loss
    accuracy = hparams[HP_NUM_UNITS] + \
        hparams[HP_DROPOUT] * 100 + \
        hparams[HP_LAYERS] + \
        len(hparams[HP_OPTIMIZER])

    loss = accuracy / 100

    # 必须使用 add_scalar 打点实验结果
    tb_writer_run.add_scalar('accuracy', accuracy, 1)
    tb_writer_run.add_scalar('loss', loss, 1)
    tb_writer_run.close()


session_num = 0
save_dir_name = 'logs'
for num_units in HP_NUM_UNITS.domain.values:
    for optimizer in HP_OPTIMIZER.domain.values:
        for layers in (HP_LAYERS.domain.min_value, HP_LAYERS.domain.max_value):
            dropout_rate = HP_DROPOUT.domain.sample_uniform()
            run_name = "run_{}".format(session_num)
            hparams = {
                HP_NUM_UNITS: num_units,
                HP_OPTIMIZER: optimizer,
                HP_LAYERS: layers,
                HP_DROPOUT: dropout_rate
                }

            run(os.path.join(save_dir_name, run_name), hparams, session_num)
            session_num += 1
```


