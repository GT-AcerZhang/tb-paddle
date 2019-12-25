# IntInterval

## 创建 IntInterval 对象

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

demo-1 创建 IntInterval 对象 

```python
>>> from tb_paddle import hparams_api as hp
>>> hp.IntInterval(min_value=1, max_value=10)
IntInterval(1, 10)
>>> hp.IntInterval(11, 20)
IntInterval(11, 20)
>>> hp.IntInterval(21, 30.0)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/linshuliang/software/anaconda3/envs/paddle_py36/lib/python3.6/site-packages/tb_paddle/hparams_summary.py", line 153, in __init__
    raise TypeError("max_value must be an int: %r" % (max_value,))
TypeError: max_value must be an int: 30.0
```

## IntInterval 对象的属性和成员函数

demo-2 IntInterval 对象的成员函数和属性

```python
# coding=utf-8
from tb_paddle import hparams_api as hp

num_units = hp.IntInterval(1, 10)

# 调用 __str__() 成员函数
print(num_units)

# 调用 __repr__() 成员函数
print(repr(num_units))

# IntInterval 对象有属性值 min_value，表示取值区间的下限
print(num_units.min_value)

# IntInterval 对象有属性值 max_value，表示取值区间的上限
print(num_units.max_value)

# 成员函数 sample_uniform 随机返回取值区间的一个整数值
print(num_units.sample_uniform())
```