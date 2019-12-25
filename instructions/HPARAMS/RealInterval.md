# RealInterval

## 创建 RealInterval 对象

[class RealInterval](../../tb_paddle/hparams_summary.py) 的初始化函数为：

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

demo-1 创建 RealInterval 对象 

```python
>>> from tb_paddle import hparams_api as hp
>>> hp.RealInterval(min_value=1.0, max_value=10.0)
RealInterval(1.0, 10.0)
>>> hp.RealInterval(11.0, 20.0)
RealInterval(11.0, 20.0)
>>> hp.RealInterval(21, 30.0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/linshuliang/software/anaconda3/envs/paddle_py36/lib/python3.6/site-packages/tb_paddle/hparams_summary.py", line 203, in __init__
    raise TypeError("min_value must be a float: %r" % (min_value,))
TypeError: min_value must be a float: 21
```

## RealInterval 对象的属性和成员函数

demo-2 RealInterval 对象的成员函数和属性

```python
# coding=utf-8
from tb_paddle import hparams_api as hp

num_units = hp.RealInterval(1.0, 10.0)

# 调用 __str__() 成员函数
print(num_units)

# 调用 __repr__() 成员函数
print(repr(num_units))

# RealInterval 对象有属性值 min_value，表示取值区间的下限
print(num_units.min_value)

# RealInterval 对象有属性值 max_value，表示取值区间的上限
print(num_units.max_value)

# 成员函数 sample_uniform 随机返回取值区间的一个实数值
print(num_units.sample_uniform())
```
