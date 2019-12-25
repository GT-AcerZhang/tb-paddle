# HPARAMS

TensorBoard 的 **HPARAMS** 栏目是超参调优（Hyperparameter Optimization）的得力工具。

## 1. Domain 超参的取值范围

Tensorboard 将超参的取值范围定义成以下三个类：

* <a href="#1.1"> Discrete </a> (离散值)
* <a href="#1.2"> IntInterval </a> (整数间隔)
* <a href="#1.3"> RealInterval </a> (实数间隔)

<a name="1.1"></a>
### 1.1 Discrete

`class Discrete` 的初始化函数为：

```python
class Discrete(Domain):
    """离散值"""

    def __init__(self, values, dtype=None):
        """
        :param values: 可选的值列表，所有元素的数据类型都必须相同
        :param dtype: 只能为 `int`, `float`, `bool`, or `str` 中的一种，
            dtype 可为 None，此时会自动推断数据类型。
        """
```

demo-1 创建 Discrete 对象 

```python
>>> from tb_paddle import hparams_api as hp
>>> hp.Discrete([16, 32, 48])
Discrete([16, 32, 48])
>>> hp.Discrete([0.1, 0.2, 0.3])
Discrete([0.1, 0.2, 0.3])
>>> hp.Discrete([0.1, 2])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/linshuliang/software/anaconda3/envs/paddle_py36/lib/python3.6/site-packages/tb_paddle/hparams_summary.py", line 103, in __init__
    % (value, self._dtype.__name__)
TypeError: dtype mismatch: not isinstance(2, float)
>>> hp.Discrete(['True', 'False'])
Discrete(['False', 'True'])
>>> optimizer = hp.Discrete(['Adam', 'SGD', 'Momentum', 'RMSPropOptimizer'])
>>> optimizer
Discrete(['Adam', 'Momentum', 'RMSPropOptimizer', 'SGD'])
```

<a name="1.2"></a>
### 1.2 IntInterval

`class IntInterval` 的初始化函数为：

```python
class IntInterval(Domain):
    """整数间隔"""

    def __init__(self, min_value=None, max_value=None):
    	"""
        :param min_value: The lower bound (inclusive) of the interval.
        :type min_value: int
        :param max_value: The upper bound (inclusive) of the interval.
        :type max_value: int
        """
```

<a name="1.3"></a>
### 1.3 RealInterval

`class RealInterval` 的初始化函数为：

```python
class RealInterval(Domain):
    """实数间隔"""

    def __init__(self, min_value=None, max_value=None):
        """
        :param min_value: The lower bound (inclusive) of the interval.
        :type min_value: float
        :param max_value: The upper bound (inclusive) of the interval.
        :type max_value: float
        """
```

